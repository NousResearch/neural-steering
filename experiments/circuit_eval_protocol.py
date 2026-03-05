"""Generalized circuit evaluation protocol.

Extends Arora et al.'s faithfulness/completeness framework to behavioral tasks.

Metrics:
  1. NECESSITY (generalized faithfulness):
     - N_H: ablate circuit on target set, measure R(x) drop
     - N_B: ablate circuit on control set, should be ~0
     - ΔQ: coherence preservation (perplexity proxy)

  2. SUFFICIENCY via MEDIATION (generalized completeness):
     - S+: transplant circuit activations from target→control context, R should increase
     - S-: transplant circuit activations from control→target context, R should decrease

  3. RANDOM CONTROLS: matched random circuits (same layer dist, same count)

R(x) = P(target_token) for factual tasks, P("I") for behavioral tasks.
Reduces to Arora's protocol when control set is empty and mediation context is null.
"""

import argparse
import sys
import json
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron_steer.core import NeuronSteerer, NeuronIdx, Circuit, steer_neurons
from experiments.prompts import (
    CAPITALS_DISCOVERY,
    CAPITALS_TEST,
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST,
    BENIGN_PROMPTS,
    SVA_PROMPTS,
)


# ============================================================
# Activation collection and transplant
# ============================================================

def collect_circuit_activations(
    steerer: NeuronSteerer,
    prompt: str,
    circuit: Circuit,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> Dict[int, torch.Tensor]:
    """Collect activation values for circuit neurons from a prompt.

    Returns dict mapping layer_idx -> tensor of activation values
    for the circuit neurons in that layer, at the last token position.
    """
    # Group circuit neurons by layer
    by_layer: Dict[int, List[int]] = {}
    for nidx in circuit.neurons:
        by_layer.setdefault(nidx.layer, []).append(nidx.neuron)
    for layer in by_layer:
        by_layer[layer] = sorted(set(by_layer[layer]))

    layer_acts = {}
    hooks = []

    for layer_idx, neuron_indices in by_layer.items():
        idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)

        def make_hook(l_idx, idx_t):
            def hook_fn(module, args):
                x = args[0]
                # Collect activations at last position for circuit neurons
                layer_acts[l_idx] = x[0, -1, idx_t.to(x.device)].detach().clone()
            return hook_fn

        h = steerer.model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_hook(layer_idx, idx_tensor)
        )
        hooks.append(h)

    try:
        if use_chat_template:
            formatted = steerer._format_prompt(prompt, seed_response)
        else:
            formatted = prompt + seed_response
        input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)
        with torch.no_grad():
            steerer.model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return layer_acts


def collect_mean_circuit_activations(
    steerer: NeuronSteerer,
    prompts: List[str],
    circuit: Circuit,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> Dict[int, torch.Tensor]:
    """Collect mean activation values across multiple prompts."""
    all_acts = [collect_circuit_activations(steerer, p, circuit, seed_response, use_chat_template) for p in prompts]

    layers = all_acts[0].keys()
    mean_acts = {}
    for l in layers:
        mean_acts[l] = torch.stack([a[l] for a in all_acts]).mean(0)
    return mean_acts


@contextmanager
def mean_ablate_neurons(
    model,
    circuit: Circuit,
    mean_activations: Dict[int, torch.Tensor],
):
    """Context manager that replaces circuit neurons with dataset mean.

    This matches Arora et al.'s evaluation: removes prompt-specific signal
    while preserving baseline activation level. Less destructive than zero ablation.
    """
    by_layer: Dict[int, List[int]] = {}
    for nidx in circuit.neurons:
        by_layer.setdefault(nidx.layer, []).append(nidx.neuron)
    for layer in by_layer:
        by_layer[layer] = sorted(set(by_layer[layer]))

    hooks = []

    for layer_idx, neuron_indices in by_layer.items():
        if layer_idx not in mean_activations:
            continue
        idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)
        mean_vals = mean_activations[layer_idx][idx_tensor]  # select circuit neurons from mean

        def make_hook(idx_t, mv):
            def pre_hook(module, args):
                x = args[0].clone()
                device_idx = idx_t.to(x.device)
                x[:, :, device_idx] = mv.to(x.device)  # replace at ALL positions
                return (x,)
            return pre_hook

        hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_hook(idx_tensor, mean_vals)
        )
        hooks.append(hook)

    try:
        yield model
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def transplant_neurons(
    model,
    circuit: Circuit,
    source_activations: Dict[int, torch.Tensor],
):
    """Context manager that replaces circuit neuron activations with source values.

    During forward pass, at the last token position, circuit neurons are
    replaced with the collected source activations. Other positions/neurons
    are untouched.
    """
    by_layer: Dict[int, List[int]] = {}
    for nidx in circuit.neurons:
        by_layer.setdefault(nidx.layer, []).append(nidx.neuron)
    for layer in by_layer:
        by_layer[layer] = sorted(set(by_layer[layer]))

    hooks = []

    for layer_idx, neuron_indices in by_layer.items():
        if layer_idx not in source_activations:
            continue
        idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)
        src_vals = source_activations[layer_idx]

        def make_hook(idx_t, src):
            def pre_hook(module, args):
                x = args[0].clone()
                device_idx = idx_t.to(x.device)
                x[:, -1, device_idx] = src.to(x.device)
                return (x,)
            return pre_hook

        hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_hook(idx_tensor, src_vals)
        )
        hooks.append(hook)

    try:
        yield model
    finally:
        for h in hooks:
            h.remove()


# ============================================================
# Metrics
# ============================================================

def measure_R(
    steerer: NeuronSteerer,
    prompt: str,
    target_token: str,
    circuit: Optional[Circuit] = None,
    multiplier: float = 1.0,
    source_activations: Optional[Dict[int, torch.Tensor]] = None,
    mean_activations: Optional[Dict[int, torch.Tensor]] = None,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> float:
    """Measure R(x) = P(target_token) with optional steering or transplant.

    Modes:
      - No circuit: baseline P(target)
      - Circuit + multiplier: zero ablation (multiplier=0) or amplification
      - Circuit + mean_activations: mean ablation (Arora-style)
      - Circuit + source_activations: mediation (transplant)
    """
    if use_chat_template:
        formatted = steerer._format_prompt(prompt, seed_response)
    else:
        formatted = prompt + seed_response
    input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)
    target_id = steerer.tokenizer.encode(target_token, add_special_tokens=False)[-1]

    if source_activations is not None and circuit is not None:
        ctx = transplant_neurons(steerer.model, circuit, source_activations)
    elif mean_activations is not None and circuit is not None:
        ctx = mean_ablate_neurons(steerer.model, circuit, mean_activations)
    elif circuit is not None:
        ctx = steer_neurons(steerer.model, circuit.neurons, multiplier)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        with torch.no_grad():
            outputs = steerer.model(input_ids)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            return probs[target_id].item()


def measure_coherence(
    steerer: NeuronSteerer,
    prompt: str,
    circuit: Circuit,
    multiplier: float = 0.0,
    max_new_tokens: int = 30,
    use_chat_template: bool = True,
) -> Tuple[str, bool]:
    """Generate under ablation and check coherence.

    Returns (generated_text, is_coherent).
    Coherence heuristic: has at least 3 real words, no excessive repetition.
    """
    out = steerer.steer_and_generate(
        prompt, circuit, multiplier=multiplier,
        max_new_tokens=max_new_tokens, all_positions=True,
        use_chat_template=use_chat_template,
    )

    # Simple coherence check
    words = out.split()
    if len(words) < 3:
        return out, False

    # Check for excessive repetition (same word >50% of output)
    word_counts = defaultdict(int)
    for w in words:
        word_counts[w.lower()] += 1
    max_repeat = max(word_counts.values())
    if max_repeat > len(words) * 0.5 and len(words) > 5:
        return out, False

    return out, True


# ============================================================
# Random control generation
# ============================================================

def make_random_circuit(
    reference_circuit: Circuit,
    n_layers: int,
    d_mlp: int,
    blacklist: set,
) -> Circuit:
    """Generate a random circuit matching the reference's layer distribution."""
    # Count neurons per layer in reference
    layer_counts = defaultdict(int)
    for nidx in reference_circuit.neurons:
        layer_counts[nidx.layer] += 1

    # Sample random neurons with same per-layer counts
    neurons = {}
    for layer, count in layer_counts.items():
        available = [n for n in range(d_mlp) if (layer, n) not in blacklist]
        # Don't sample neurons that are in the reference circuit
        ref_neurons_in_layer = {nidx.neuron for nidx in reference_circuit.neurons if nidx.layer == layer}
        available = [n for n in available if n not in ref_neurons_in_layer]

        sampled = random.sample(available, min(count, len(available)))
        for n in sampled:
            nidx = NeuronIdx(layer=layer, position=-1, neuron=n)
            neurons[nidx] = random.gauss(0, 1)  # Random weight

    return Circuit(
        neurons=neurons,
        prompt="[random control]",
        target_token="[random]",
        total_logit_diff=0.0,
    )


# ============================================================
# Main evaluation
# ============================================================

def evaluate_circuit(
    steerer: NeuronSteerer,
    circuit: Circuit,
    label: str,
    target_token: str,
    target_prompts: List[str],
    control_prompts: List[str],
    mediation_source_prompts: Optional[List[str]] = None,
    mediation_target_prompts: Optional[List[str]] = None,
    n_random: int = 5,
    seed_response: str = "",
    mean_activations: Optional[Dict[int, torch.Tensor]] = None,
    ablation_mode: str = "both",
    use_chat_template: bool = True,
) -> dict:
    """Run full evaluation protocol on a circuit.

    Args:
        circuit: The circuit to evaluate
        label: Name for display
        target_token: Token for R(x) measurement (e.g., " I" or " Austin")
        target_prompts: Prompts where behavior should be present (harmful / France)
        control_prompts: Prompts where behavior should NOT be present (benign / Germany)
        mediation_source_prompts: Source prompts for transplant (defaults to target_prompts)
        mediation_target_prompts: Target prompts for transplant (defaults to control_prompts)
        n_random: Number of random control circuits
        seed_response: Seed text appended before target (for factual tasks)
        mean_activations: Dataset mean activations for mean ablation (Arora-style)
        ablation_mode: "zero", "mean", or "both"
    """
    if mediation_source_prompts is None:
        mediation_source_prompts = target_prompts
    if mediation_target_prompts is None:
        mediation_target_prompts = control_prompts

    results = {"label": label, "n_neurons": len(circuit.neurons)}
    uct = use_chat_template  # shorthand for threading through all calls

    # --- Phase 1: Baseline R(x) ---
    R_target_base = np.mean([measure_R(steerer, p, target_token, seed_response=seed_response, use_chat_template=uct) for p in target_prompts])
    R_control_base = np.mean([measure_R(steerer, p, target_token, seed_response=seed_response, use_chat_template=uct) for p in control_prompts])
    results["R_target_baseline"] = R_target_base
    results["R_control_baseline"] = R_control_base

    # --- Phase 2: Necessity (ablation) ---
    ablation_modes = []
    if ablation_mode in ("zero", "both"):
        ablation_modes.append("zero")
    if ablation_mode in ("mean", "both") and mean_activations is not None:
        ablation_modes.append("mean")

    for abl_mode in ablation_modes:
        suffix = f"_{abl_mode}" if len(ablation_modes) > 1 else ""

        if abl_mode == "zero":
            R_target_abl = np.mean([
                measure_R(steerer, p, target_token, circuit=circuit, multiplier=0.0, seed_response=seed_response, use_chat_template=uct)
                for p in target_prompts
            ])
            R_control_abl = np.mean([
                measure_R(steerer, p, target_token, circuit=circuit, multiplier=0.0, seed_response=seed_response, use_chat_template=uct)
                for p in control_prompts
            ])
        else:  # mean ablation
            R_target_abl = np.mean([
                measure_R(steerer, p, target_token, circuit=circuit, mean_activations=mean_activations, seed_response=seed_response, use_chat_template=uct)
                for p in target_prompts
            ])
            R_control_abl = np.mean([
                measure_R(steerer, p, target_token, circuit=circuit, mean_activations=mean_activations, seed_response=seed_response, use_chat_template=uct)
                for p in control_prompts
            ])

        N_H = R_target_base - R_target_abl
        N_B = R_control_base - R_control_abl

        results[f"R_target_ablated{suffix}"] = R_target_abl
        results[f"R_control_ablated{suffix}"] = R_control_abl
        results[f"N_H{suffix}"] = N_H
        results[f"N_B{suffix}"] = N_B

    # Use the first ablation mode for backward-compat keys
    primary = ablation_modes[0] if ablation_modes else "zero"
    p_suffix = f"_{primary}" if len(ablation_modes) > 1 else ""
    results.setdefault("N_H", results.get(f"N_H{p_suffix}", 0))
    results.setdefault("N_B", results.get(f"N_B{p_suffix}", 0))

    # Coherence check — zero ablation (stricter test)
    coherent_count = 0
    sample_outputs = []
    for p in target_prompts[:3]:
        out, is_coherent = measure_coherence(steerer, p, circuit, multiplier=0.0, use_chat_template=uct)
        if is_coherent:
            coherent_count += 1
        sample_outputs.append((p, out, is_coherent))
    results["coherence_rate"] = coherent_count / min(3, len(target_prompts))
    results["sample_outputs"] = sample_outputs

    # --- Phase 3: Sufficiency via mediation ---
    # Collect mean activations from source prompts
    source_acts = collect_mean_circuit_activations(
        steerer, mediation_source_prompts[:5], circuit, seed_response, use_chat_template=uct
    )

    # Transplant into control prompts (S+: should increase R)
    R_mediated_plus = np.mean([
        measure_R(steerer, p, target_token,
                  circuit=circuit, source_activations=source_acts, seed_response=seed_response, use_chat_template=uct)
        for p in mediation_target_prompts[:5]
    ])
    S_plus = R_mediated_plus - R_control_base

    # Collect mean activations from control prompts
    control_acts = collect_mean_circuit_activations(
        steerer, control_prompts[:5], circuit, seed_response, use_chat_template=uct
    )

    # Transplant into target prompts (S-: should decrease R)
    R_mediated_minus = np.mean([
        measure_R(steerer, p, target_token,
                  circuit=circuit, source_activations=control_acts, seed_response=seed_response, use_chat_template=uct)
        for p in target_prompts[:5]
    ])
    S_minus = R_target_base - R_mediated_minus

    results["R_mediated_plus"] = R_mediated_plus
    results["R_mediated_minus"] = R_mediated_minus
    results["S_plus"] = S_plus  # Positive = transplant induces behavior
    results["S_minus"] = S_minus  # Positive = transplant suppresses behavior

    # --- Phase 4: Random controls ---
    n_layers = steerer.model.config.num_hidden_layers
    d_mlp = steerer.model.config.intermediate_size
    random_N_H = []
    random_S_plus = []
    random_S_minus = []

    for i in range(n_random):
        rc = make_random_circuit(circuit, n_layers, d_mlp, steerer.blacklist)

        # Random necessity
        r_target_abl = np.mean([
            measure_R(steerer, p, target_token, circuit=rc, multiplier=0.0, seed_response=seed_response, use_chat_template=uct)
            for p in target_prompts[:3]
        ])
        random_N_H.append(R_target_base - r_target_abl)

        # Random mediation (S+)
        try:
            rc_source_acts = collect_mean_circuit_activations(
                steerer, mediation_source_prompts[:3], rc, seed_response, use_chat_template=uct
            )
            r_med = np.mean([
                measure_R(steerer, p, target_token,
                          circuit=rc, source_activations=rc_source_acts, seed_response=seed_response, use_chat_template=uct)
                for p in mediation_target_prompts[:3]
            ])
            random_S_plus.append(r_med - R_control_base)
        except Exception:
            random_S_plus.append(0.0)

        # Random mediation (S-)
        try:
            rc_ctrl_acts = collect_mean_circuit_activations(
                steerer, control_prompts[:3], rc, seed_response, use_chat_template=uct
            )
            r_med_minus = np.mean([
                measure_R(steerer, p, target_token,
                          circuit=rc, source_activations=rc_ctrl_acts, seed_response=seed_response, use_chat_template=uct)
                for p in target_prompts[:3]
            ])
            random_S_minus.append(R_target_base - r_med_minus)
        except Exception:
            random_S_minus.append(0.0)

    results["n_random"] = n_random
    results["random_N_H_mean"] = np.mean(random_N_H)
    results["random_N_H_std"] = np.std(random_N_H)
    results["random_S_plus_mean"] = np.mean(random_S_plus)
    results["random_S_plus_std"] = np.std(random_S_plus)
    results["random_S_minus_mean"] = np.mean(random_S_minus)
    results["random_S_minus_std"] = np.std(random_S_minus)

    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    label = results["label"]
    print(f"\n{'='*65}")
    print(f"  {label} ({results['n_neurons']} neurons)")
    print(f"{'='*65}")

    print(f"\n  Baseline:")
    print(f"    R(target) = {results['R_target_baseline']:.4f}")
    print(f"    R(control) = {results['R_control_baseline']:.4f}")

    print(f"\n  Necessity (ablation):")
    for mode in ["zero", "mean"]:
        suffix = f"_{mode}"
        if f"N_H{suffix}" in results:
            print(f"    [{mode:4s}] R(target) ablated = {results[f'R_target_ablated{suffix}']:.4f}  "
                  f"N_H = {results[f'N_H{suffix}']:.4f}  N_B = {results[f'N_B{suffix}']:.4f}")
    # Fallback for single-mode runs
    if "N_H_zero" not in results and "N_H_mean" not in results:
        print(f"    R(target) ablated = {results.get('R_target_ablated', 0):.4f}")
        print(f"    N_H = {results['N_H']:.4f}  N_B = {results['N_B']:.4f}")
    print(f"    Coherence (zero abl) = {results['coherence_rate']:.0%}")

    # Grade using mean ablation if available, else zero
    nh_key = "N_H_mean" if "N_H_mean" in results else "N_H"
    nh = results[nh_key]
    if nh > 0.1 and results['coherence_rate'] > 0.5:
        necessity_grade = "GOOD (specific + coherent)"
    elif nh > 0.1:
        necessity_grade = "LOBOTOMY (drops R but breaks model)"
    else:
        necessity_grade = "WEAK (doesn't affect behavior)"
    print(f"    Assessment: {necessity_grade}")

    print(f"\n  Sufficiency (mediation):")
    print(f"    S+ transplant (source→control): R = {results['R_mediated_plus']:.4f} "
          f"(Δ = {results['S_plus']:+.4f})")
    print(f"    S- transplant (control→target): R = {results['R_mediated_minus']:.4f} "
          f"(Δ = {-results['S_minus']:+.4f})")

    print(f"\n  Random controls ({results.get('n_random', '?')} circuits):")
    print(f"    Random N_H  = {results['random_N_H_mean']:.4f} ± {results['random_N_H_std']:.4f}")
    print(f"    Random S+   = {results['random_S_plus_mean']:.4f} ± {results['random_S_plus_std']:.4f}")
    print(f"    Random S-   = {results['random_S_minus_mean']:.4f} ± {results['random_S_minus_std']:.4f}")

    # Effect sizes vs random
    def effect_size(real, rand_mean, rand_std):
        if rand_std < 1e-6:
            return float('inf') if abs(real - rand_mean) > 1e-6 else 0.0
        return (real - rand_mean) / rand_std

    es_nh = effect_size(results['N_H'], results['random_N_H_mean'], results['random_N_H_std'])
    es_sp = effect_size(results['S_plus'], results['random_S_plus_mean'], results['random_S_plus_std'])
    es_sm = effect_size(results['S_minus'], results['random_S_minus_mean'], results['random_S_minus_std'])
    print(f"    Effect size N_H: {es_nh:.1f}σ")
    print(f"    Effect size S+:  {es_sp:.1f}σ")
    print(f"    Effect size S-:  {es_sm:.1f}σ")

    # Sample outputs
    print(f"\n  Sample ablation outputs:")
    for prompt, out, coherent in results.get("sample_outputs", []):
        tag = "✓" if coherent else "✗"
        out_short = out[:100].replace("\n", " ")
        print(f"    [{tag}] Q: {prompt[:50]}")
        print(f"        A: {out_short}")


def main():
    p = argparse.ArgumentParser(description="Generalized circuit evaluation protocol")
    p.add_argument("--model", default="llama",
                   choices=["llama", "llama8b"])
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--n_random", type=int, default=5)
    p.add_argument("--task", default="both",
                   choices=["factual", "behavioral", "sva", "both", "all"])
    p.add_argument("--skip_relp", action="store_true")
    args = p.parse_args()

    model_name = {
        "llama": "meta-llama/Llama-3.2-1B-Instruct",
        "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    }[args.model]

    print(f"Loading {model_name}...")
    steerer = NeuronSteerer(model_name)

    # Compute dataset mean activations for mean ablation (Arora-style)
    print("Computing dataset mean activations...")
    mean_acts = steerer.compute_mean_activations()

    all_results = []

    # ============================================================
    # FACTUAL TASK: Capitals
    # ============================================================
    if args.task in ("factual", "both", "all"):
        print(f"\n{'#'*65}")
        print(f"  FACTUAL TASK: Capitals")
        print(f"{'#'*65}")

        # Use France as target, other capitals as control
        # seed_response positions the target token at the generation point
        seed_response_f = "Answer: "
        target_prompts_f = ["What is the capital of France?"]
        control_prompts_f = [
            "What is the capital of Germany?",
            "What is the capital of Japan?",
            "What is the capital of Italy?",
        ]
        target_token_f = " Paris"

        # --- Contrastive factual ---
        print("\nDiscovering contrastive-factual circuit...")
        mid = len(CAPITALS_DISCOVERY) // 2
        cf_circuit = steerer.discover_contrastive(
            positive_prompts=[p for p, _ in CAPITALS_DISCOVERY[:mid]],
            negative_prompts=[p for p, _ in CAPITALS_DISCOVERY[mid:]],
            top_k=args.top_k,
        )

        res_cf = evaluate_circuit(
            steerer, cf_circuit, "Contrastive-Factual",
            target_token=target_token_f,
            target_prompts=target_prompts_f,
            control_prompts=control_prompts_f,
            n_random=args.n_random,
            seed_response=seed_response_f,
            mean_activations=mean_acts,
            ablation_mode="both",
        )
        all_results.append(res_cf)
        print_results(res_cf)

        # --- RelP factual ---
        if not args.skip_relp:
            print("\nDiscovering RelP-factual circuit...")
            relp_circuit_f = steerer.discover_circuit_multi(
                prompts=[p for p, _ in CAPITALS_DISCOVERY],
                target_tokens=[t for _, t in CAPITALS_DISCOVERY],
                counterfactual_tokens=[None] * len(CAPITALS_DISCOVERY),
                selection_method="topk",
                top_k=args.top_k,
            )

            res_rf = evaluate_circuit(
                steerer, relp_circuit_f, "RelP-Factual",
                target_token=target_token_f,
                target_prompts=target_prompts_f,
                control_prompts=control_prompts_f,
                n_random=args.n_random,
                seed_response=seed_response_f,
                mean_activations=mean_acts,
                ablation_mode="both",
            )
            all_results.append(res_rf)
            print_results(res_rf)

    # ============================================================
    # BEHAVIORAL TASK: Refusal
    # ============================================================
    if args.task in ("behavioral", "both", "all"):
        print(f"\n{'#'*65}")
        print(f"  BEHAVIORAL TASK: Refusal")
        print(f"{'#'*65}")

        target_prompts_b = REFUSAL_TEST
        control_prompts_b = BENIGN_PROMPTS[:5]
        target_token_b = "I"

        # --- Contrastive behavioral ---
        print("\nDiscovering contrastive-behavioral circuit...")
        cb_circuit = steerer.discover_contrastive(
            positive_prompts=REFUSAL_DISCOVERY_POSITIVE,
            negative_prompts=REFUSAL_DISCOVERY_NEGATIVE,
            top_k=args.top_k,
        )

        res_cb = evaluate_circuit(
            steerer, cb_circuit, "Contrastive-Behavioral",
            target_token=target_token_b,
            target_prompts=target_prompts_b,
            control_prompts=control_prompts_b,
            mediation_source_prompts=REFUSAL_DISCOVERY_POSITIVE[:5],
            mediation_target_prompts=control_prompts_b,
            n_random=args.n_random,
            mean_activations=mean_acts,
            ablation_mode="both",
        )
        all_results.append(res_cb)
        print_results(res_cb)

        # --- RelP behavioral (target "I") ---
        if not args.skip_relp:
            print("\nDiscovering RelP-behavioral circuit (target 'I')...")
            relp_circuit_b = steerer.discover_circuit_multi(
                prompts=REFUSAL_DISCOVERY_POSITIVE,
                target_tokens=["I"] * len(REFUSAL_DISCOVERY_POSITIVE),
                selection_method="topk",
                top_k=args.top_k,
            )

            res_rb = evaluate_circuit(
                steerer, relp_circuit_b, "RelP-Behavioral (I)",
                target_token=target_token_b,
                target_prompts=target_prompts_b,
                control_prompts=control_prompts_b,
                mediation_source_prompts=REFUSAL_DISCOVERY_POSITIVE[:5],
                mediation_target_prompts=control_prompts_b,
                n_random=args.n_random,
                mean_activations=mean_acts,
                ablation_mode="both",
            )
            all_results.append(res_rb)
            print_results(res_rb)

    # ============================================================
    # FACTUAL TASK: SVA (Subject-Verb Agreement)
    # ============================================================
    if args.task in ("sva", "all"):
        print(f"\n{'#'*65}")
        print(f"  FACTUAL TASK: Subject-Verb Agreement")
        print(f"{'#'*65}")

        # SVA is a bare completion task (no chat template).
        # R(x) = P(" is") — we track the singular verb form.
        # Target prompts = singular subjects (where " is" is correct, high R).
        # Control prompts = plural subjects (where " are" is correct, low R).
        sva_singular = [p for p in SVA_PROMPTS if p[1].strip() == "is"]  # " is" is correct
        sva_plural = [p for p in SVA_PROMPTS if p[1].strip() == "are"]   # " are" is correct

        target_prompts_sva = [p[0] for p in sva_singular]
        control_prompts_sva = [p[0] for p in sva_plural]
        target_token_sva = " is"

        # --- RelP SVA ---
        if not args.skip_relp:
            print("\nDiscovering RelP-SVA circuit...")
            relp_circuit_sva = steerer.discover_circuit_multi(
                prompts=[p[0] for p in SVA_PROMPTS],
                target_tokens=[p[1] for p in SVA_PROMPTS],
                counterfactual_tokens=[p[2] for p in SVA_PROMPTS],
                selection_method="topk",
                top_k=args.top_k,
                use_chat_template=False,
            )

            res_rs = evaluate_circuit(
                steerer, relp_circuit_sva, "RelP-SVA",
                target_token=target_token_sva,
                target_prompts=target_prompts_sva,
                control_prompts=control_prompts_sva,
                n_random=args.n_random,
                seed_response="",
                mean_activations=mean_acts,
                ablation_mode="both",
                use_chat_template=False,
            )
            all_results.append(res_rs)
            print_results(res_rs)

        # --- Contrastive SVA ---
        # Positive = singular (where " is" fires), Negative = plural (where " are" fires)
        print("\nDiscovering contrastive-SVA circuit...")
        cs_circuit = steerer.discover_contrastive(
            positive_prompts=[p[0] for p in sva_singular],
            negative_prompts=[p[0] for p in sva_plural],
            top_k=args.top_k,
        )

        res_cs = evaluate_circuit(
            steerer, cs_circuit, "Contrastive-SVA",
            target_token=target_token_sva,
            target_prompts=target_prompts_sva,
            control_prompts=control_prompts_sva,
            n_random=args.n_random,
            seed_response="",
            mean_activations=mean_acts,
            ablation_mode="both",
            use_chat_template=False,
        )
        all_results.append(res_cs)
        print_results(res_cs)

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n\n{'#'*65}")
    print(f"  SUMMARY TABLE")
    print(f"{'#'*65}")

    header = f"  {'Circuit':<28} {'N_H(0)':>7} {'N_H(μ)':>7} {'N_B(0)':>7} {'N_B(μ)':>7} {'Coh':>4} {'S+':>7} {'S-':>7}"
    print(f"\n{header}")
    print(f"  {'-'*80}")

    for r in all_results:
        nh_zero = r.get('N_H_zero', r.get('N_H', 0))
        nh_mean = r.get('N_H_mean', 0)
        nb_zero = r.get('N_B_zero', r.get('N_B', 0))
        nb_mean = r.get('N_B_mean', 0)
        line = (f"  {r['label']:<28} "
                f"{nh_zero:>7.3f} {nh_mean:>7.3f} {nb_zero:>7.3f} {nb_mean:>7.3f} "
                f"{r['coherence_rate']:>4.0%} "
                f"{r['S_plus']:>7.3f} {r['S_minus']:>7.3f}")
        print(line)

    print(f"\n  N_H(0) = necessity under zero ablation")
    print(f"  N_H(μ) = necessity under mean ablation (Arora-style)")
    print(f"  N_B = specificity leak (should be ~0)")
    print(f"  Coh = coherence rate (zero ablation)")
    print(f"  S+/S- = mediation sufficiency")

    # Save results
    out_path = Path(__file__).parent / "eval_results.json"
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "sample_outputs"}
        sr = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in sr.items()}
        serializable.append(sr)
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
