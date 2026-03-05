"""Sufficiency test: can activating circuit neurons induce behavior?

Complements surgical ablation (which tests necessity).
Two approaches:
  1. Transplantation: collect activations from refusal prompts, inject into benign
  2. Amplification: multiply neuron activations on benign prompts

Key question: Is the refusal circuit sufficient to induce refusal on benign inputs?

Protocol:
  0. Baseline P("I") on benign prompts (should be low)
  1. Collect mean neuron activations from source prompts (refusal for behavioral)
  2. Full-circuit transplant into control prompts (sufficiency ceiling)
  3. Single-neuron transplant (which neurons are individually sufficient?)
  4. Progressive transplant (how many neurons needed to induce behavior?)
  5. Amplification sweep (2x, 3x, 5x on top neurons, benign prompts)
  6. Random neuron controls
"""

import argparse
import json
import random as rng_mod
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron_steer.core import NeuronSteerer, NeuronIdx, Circuit, steer_neurons

from experiments.surgical_ablation import (
    measure_R,
    measure_R_batch,
    make_single_neuron_ctx,
    load_bottleneck_candidates,
    unique_bottleneck_neurons,
    load_circuit,
    resolve_topology_dir,
    TASK_CONFIGS,
)
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    BENIGN_PROMPTS,
    CAPITALS_DISCOVERY,
    SYCOPHANCY_DISCOVERY_POSITIVE,
    FC_REFUSAL_DISCOVERY_POSITIVE,
    FC_BENIGN,
    FC_BELIEF_DISCOVERY,
    FC_BELIEF_NO_DISCOVERY,
    FC_BELIEF_NO_TEST,
)


# ============================================================
# Activation collection (single-neuron level)
# ============================================================

def collect_single_neuron_activation(
    model, layer: int, neuron_idx: int, input_ids: torch.Tensor,
) -> float:
    """Collect a single neuron's activation at last position (pre-down_proj)."""
    result = {}

    def pre_hook(module, args):
        x = args[0]
        result['val'] = x[0, -1, neuron_idx].item()

    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(input_ids)
    h.remove()
    return result['val']


def collect_mean_neuron_activation(
    steerer: NeuronSteerer,
    layer: int,
    neuron_idx: int,
    prompts: List[str],
    seed_response: str = "",
    use_chat_template: bool = True,
) -> float:
    """Collect mean activation of a neuron across prompts at last position."""
    vals = []
    for prompt in prompts:
        if use_chat_template:
            formatted = steerer._format_prompt(prompt, seed_response)
        else:
            formatted = prompt + seed_response
        input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)
        val = collect_single_neuron_activation(steerer.model, layer, neuron_idx, input_ids)
        vals.append(val)
    return float(np.mean(vals))


def collect_full_circuit_activations(
    steerer: NeuronSteerer,
    prompts: List[str],
    circuit: Circuit,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> Dict[int, torch.Tensor]:
    """Collect mean circuit neuron activations across prompts.

    Returns dict: layer_idx -> 1D tensor of activation values for circuit neurons in that layer.
    """
    by_layer: Dict[int, List[int]] = {}
    for nidx in circuit.neurons:
        by_layer.setdefault(nidx.layer, []).append(nidx.neuron)
    for layer in by_layer:
        by_layer[layer] = sorted(set(by_layer[layer]))

    all_prompt_acts = []

    for prompt in prompts:
        if use_chat_template:
            formatted = steerer._format_prompt(prompt, seed_response)
        else:
            formatted = prompt + seed_response
        input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)

        layer_acts = {}
        hooks = []

        for layer_idx, neuron_indices in by_layer.items():
            idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)

            def make_hook(l_idx, idx_t):
                def hook_fn(module, args):
                    x = args[0]
                    layer_acts[l_idx] = x[0, -1, idx_t.to(x.device)].detach().clone()
                return hook_fn

            h = steerer.model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                make_hook(layer_idx, idx_tensor)
            )
            hooks.append(h)

        with torch.no_grad():
            steerer.model(input_ids)

        for h in hooks:
            h.remove()

        all_prompt_acts.append(layer_acts)

    # Average across prompts
    mean_acts = {}
    for l_idx in by_layer:
        mean_acts[l_idx] = torch.stack([a[l_idx] for a in all_prompt_acts]).mean(0)
    return mean_acts


# ============================================================
# Transplant context managers
# ============================================================

def make_transplant_single_ctx(model, layer: int, neuron_idx: int, target_val: float):
    """Factory: replace one neuron at last position with target value."""
    def factory():
        @contextmanager
        def ctx():
            def pre_hook(module, args):
                x = args[0].clone()
                x[:, -1, neuron_idx] = target_val
                return (x,)
            h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre_hook)
            try:
                yield model
            finally:
                h.remove()
        return ctx()
    return factory


def make_transplant_multi_ctx(model, transplants: List[Tuple[int, int, float]]):
    """Factory: replace multiple neurons at last position.

    transplants: list of (layer, neuron_idx, target_val) tuples
    """
    def factory():
        @contextmanager
        def ctx():
            hooks = []
            by_layer: Dict[int, List[Tuple[int, float]]] = {}
            for layer, neuron_idx, target_val in transplants:
                by_layer.setdefault(layer, []).append((neuron_idx, target_val))

            for layer_idx, neuron_vals in by_layer.items():
                def make_hook(nv_list):
                    def pre_hook(module, args):
                        x = args[0].clone()
                        for nidx, val in nv_list:
                            x[:, -1, nidx] = val
                        return (x,)
                    return pre_hook
                h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                    make_hook(neuron_vals)
                )
                hooks.append(h)
            try:
                yield model
            finally:
                for h in hooks:
                    h.remove()
        return ctx()
    return factory


def make_full_transplant_ctx(model, circuit: Circuit, source_acts: Dict[int, torch.Tensor]):
    """Factory: transplant all circuit neuron activations at last position."""
    by_layer: Dict[int, List[int]] = {}
    for nidx in circuit.neurons:
        by_layer.setdefault(nidx.layer, []).append(nidx.neuron)
    for layer in by_layer:
        by_layer[layer] = sorted(set(by_layer[layer]))

    def factory():
        @contextmanager
        def ctx():
            hooks = []
            for layer_idx, neuron_indices in by_layer.items():
                if layer_idx not in source_acts:
                    continue
                idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)
                src_vals = source_acts[layer_idx]

                def make_hook(idx_t, src):
                    def pre_hook(module, args):
                        x = args[0].clone()
                        device_idx = idx_t.to(x.device)
                        x[:, -1, device_idx] = src.to(x.device)
                        return (x,)
                    return pre_hook

                h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                    make_hook(idx_tensor, src_vals)
                )
                hooks.append(h)
            try:
                yield model
            finally:
                for h in hooks:
                    h.remove()
        return ctx()
    return factory


def make_amplify_single_ctx(model, layer: int, neuron_idx: int, multiplier: float):
    """Factory: amplify one neuron at all positions."""
    def factory():
        neurons = {NeuronIdx(layer=layer, position=-1, neuron=neuron_idx): 0.0}
        return steer_neurons(model, neurons, multiplier=multiplier, all_positions=True)
    return factory


# ============================================================
# Main experiment
# ============================================================

def run_sufficiency_test(
    steerer: NeuronSteerer,
    task: str,
    topology_base: str,
    max_neurons: int = 24,
    n_random: int = 20,
    output_dir: str = None,
):
    """Run sufficiency test for a task."""
    config = TASK_CONFIGS[task]
    topo_dir = resolve_topology_dir(topology_base, task, config)

    print(f"\n{'='*70}")
    print(f"SUFFICIENCY TEST: {task.upper()}")
    print(f"{'='*70}")

    analysis_path = topo_dir / "analysis.json"
    circuit_path = topo_dir / "circuit.json"

    if not analysis_path.exists():
        print(f"  ERROR: {analysis_path} not found, skipping")
        return None

    candidates = load_bottleneck_candidates(str(analysis_path))
    all_bn = unique_bottleneck_neurons(candidates)
    bottleneck_neurons = all_bn[:max_neurons]
    circuit = load_circuit(str(circuit_path))

    print(f"  Circuit size: {len(circuit.neurons)} neurons")
    print(f"  Bottleneck neurons: {len(bottleneck_neurons)}")

    uct = config["use_chat_template"]
    seed = config["seed_response"]
    M = "logit_margin"

    # Task-specific setup
    if task == "behavioral":
        target_token = config["target_token"]
        # Source: refusal prompts (where behavior IS active)
        source_prompts = REFUSAL_DISCOVERY_POSITIVE
        # Target: benign prompts (where behavior is NOT active)
        control_prompts = BENIGN_PROMPTS
    elif task == "sycophancy":
        target_token = config["target_token"]
        source_prompts = SYCOPHANCY_DISCOVERY_POSITIVE
        control_prompts = BENIGN_PROMPTS
    elif task == "fc_refusal":
        target_token = config["target_token"]  # "No"
        source_prompts = FC_REFUSAL_DISCOVERY_POSITIVE  # harmful forced-choice (model says No)
        control_prompts = FC_BENIGN  # benign forced-choice (model says Yes)
    elif task == "fc_belief":
        target_token = config["target_token"]  # "Yes"
        source_prompts = FC_BELIEF_DISCOVERY  # opinion questions (model says Yes)
        control_prompts = FC_BELIEF_NO_TEST  # "No" questions as control for sufficiency
    elif task == "factual":
        target_token = None  # per-prompt
        # Source: capitals discovery set (correct answers)
        source_prompts = [p for p, _ in CAPITALS_DISCOVERY]
        # Target: test set where some may have low P(correct)
        control_prompts = None  # handled per-prompt below
    else:
        print(f"  ERROR: unsupported task {task}")
        return None

    results = {
        "task": task,
        "n_bottleneck_neurons": len(bottleneck_neurons),
    }

    # Helper for measuring on control prompts
    def _measure_control(make_ctx=None):
        """Measure on control prompts. Returns (mean_dict, per_prompt)."""
        if task in ("behavioral", "sycophancy", "fc_refusal", "fc_belief"):
            return measure_R_batch(steerer, control_prompts, target_token,
                                   make_ctx=make_ctx, seed_response=seed, use_chat_template=uct)
        elif task == "factual":
            # Use test prompts (some have low baseline, interesting targets)
            rr = []
            for prompt, target in config["test_prompts"]:
                rr.append(measure_R(steerer, prompt, target, make_ctx=make_ctx,
                                    seed_response=seed, use_chat_template=uct))
            mean_d = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
            return mean_d, rr

    # -------------------------------------------------------
    # Phase 0: Baseline on control prompts
    # -------------------------------------------------------
    print("\n--- Phase 0: Baseline on control prompts ---")

    ctrl_baseline, ctrl_per = _measure_control()

    if task in ("behavioral", "sycophancy", "fc_refusal", "fc_belief"):
        print(f"  Control prompts — P(\"{target_token}\"):")
        for p, r in zip(control_prompts, ctrl_per):
            print(f"    {p[:45]:45s} → P={r['prob']:.4f}  margin={r['logit_margin']:+.2f}")
    elif task == "factual":
        print(f"  Factual test prompts — P(correct):")
        for (p, t), r in zip(config["test_prompts"], ctrl_per):
            print(f"    {p[:45]:45s} → P={r['prob']:.4f}  margin={r['logit_margin']:+.2f}")

    print(f"\n  Control baseline: P={ctrl_baseline['prob']:.4f}  margin={ctrl_baseline['logit_margin']:+.2f}")
    results["control_baseline"] = ctrl_baseline

    # -------------------------------------------------------
    # Phase 1: Collect mean neuron activations from source prompts
    # -------------------------------------------------------
    print(f"\n--- Phase 1: Collecting source activations ({len(source_prompts)} prompts) ---")

    # Collect per-neuron mean activations from source prompts
    neuron_source_acts = {}
    for i, (layer, neuron) in enumerate(bottleneck_neurons):
        mean_act = collect_mean_neuron_activation(
            steerer, layer, neuron, source_prompts,
            seed_response=seed, use_chat_template=uct,
        )
        neuron_source_acts[(layer, neuron)] = mean_act
        if i < 10 or (i + 1) % 10 == 0:
            print(f"  [{i+1:2d}] L{layer:02d}/N{neuron:5d}: mean_act={mean_act:+.4f}")

    # Also collect on control prompts for comparison
    neuron_ctrl_acts = {}
    for layer, neuron in bottleneck_neurons:
        if task in ("behavioral", "sycophancy", "fc_refusal", "fc_belief"):
            ctrl_act = collect_mean_neuron_activation(
                steerer, layer, neuron, control_prompts[:5],
                seed_response=seed, use_chat_template=uct,
            )
        else:
            ctrl_act = collect_mean_neuron_activation(
                steerer, layer, neuron, [p for p, _ in config["test_prompts"]],
                seed_response=seed, use_chat_template=uct,
            )
        neuron_ctrl_acts[(layer, neuron)] = ctrl_act

    # Report activation deltas
    print(f"\n  Activation deltas (source - control):")
    act_deltas = {}
    for layer, neuron in bottleneck_neurons:
        delta = neuron_source_acts[(layer, neuron)] - neuron_ctrl_acts[(layer, neuron)]
        act_deltas[(layer, neuron)] = delta

    sorted_deltas = sorted(act_deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    for (l, n), d in sorted_deltas[:10]:
        src = neuron_source_acts[(l, n)]
        ctrl = neuron_ctrl_acts[(l, n)]
        print(f"    L{l:02d}/N{n:5d}: src={src:+.4f} ctrl={ctrl:+.4f} Δ={d:+.4f}")

    results["neuron_activations"] = [
        {"layer": l, "neuron": n,
         "source_mean": neuron_source_acts[(l, n)],
         "control_mean": neuron_ctrl_acts[(l, n)],
         "delta": act_deltas[(l, n)]}
        for l, n in bottleneck_neurons
    ]

    # -------------------------------------------------------
    # Phase 2: Full-circuit transplant (sufficiency ceiling)
    # -------------------------------------------------------
    print("\n--- Phase 2: Full-circuit transplant ---")

    full_source_acts = collect_full_circuit_activations(
        steerer, source_prompts[:5], circuit,
        seed_response=seed, use_chat_template=uct,
    )
    full_tx_ctx = make_full_transplant_ctx(steerer.model, circuit, full_source_acts)
    full_tx_mean, _ = _measure_control(make_ctx=full_tx_ctx)

    dS_full = full_tx_mean[M] - ctrl_baseline[M]
    print(f"  Full transplant: P={full_tx_mean['prob']:.4f}  margin={full_tx_mean['logit_margin']:+.2f}")
    print(f"  dSufficiency (full): {dS_full:+.4f}")
    results["full_transplant"] = full_tx_mean
    results["dSufficiency_full"] = dS_full

    # -------------------------------------------------------
    # Phase 3: Single-neuron transplant
    # -------------------------------------------------------
    print(f"\n--- Phase 3: Single-neuron transplant ({len(bottleneck_neurons)} neurons) ---")
    print(f"  {'':6s} {'Neuron':14s} {'P':>8s} {'margin':>8s} {'dS':>8s}")

    single_tx_results = []
    for i, (layer, neuron) in enumerate(bottleneck_neurons):
        src_val = neuron_source_acts[(layer, neuron)]
        tx_ctx = make_transplant_single_ctx(steerer.model, layer, neuron, src_val)
        tx_mean, _ = _measure_control(make_ctx=tx_ctx)
        dS = tx_mean[M] - ctrl_baseline[M]

        single_tx_results.append({
            "layer": layer, "neuron": neuron,
            "source_activation": src_val,
            **{f"tx_{k}": v for k, v in tx_mean.items()},
            "dSufficiency": dS,
        })
        print(f"  [{i+1:2d}] L{layer:02d}/N{neuron:5d}  "
              f"P={tx_mean['prob']:.4f}  margin={tx_mean['logit_margin']:+.2f}  dS={dS:+.4f}")

    results["single_neuron_transplant"] = single_tx_results

    # -------------------------------------------------------
    # Phase 4: Progressive transplant (by single-neuron dS rank)
    # -------------------------------------------------------
    print(f"\n--- Phase 4: Progressive transplant (cumulative by dS rank) ---")

    sorted_by_suff = sorted(single_tx_results, key=lambda x: x["dSufficiency"], reverse=True)

    progressive_results = []
    print(f"  {'k':>3s} {'margin':>8s} {'dS':>8s} {'%full':>7s} {'added':20s}")
    for k in range(1, len(sorted_by_suff) + 1):
        transplants = [
            (r["layer"], r["neuron"], r["source_activation"])
            for r in sorted_by_suff[:k]
        ]
        tx_ctx = make_transplant_multi_ctx(steerer.model, transplants)
        tx_mean, _ = _measure_control(make_ctx=tx_ctx)
        cum_dS = tx_mean[M] - ctrl_baseline[M]
        frac = cum_dS / dS_full if abs(dS_full) > 1e-6 else 0

        r = sorted_by_suff[k - 1]
        progressive_results.append({
            "k": k,
            "added_neuron": {"layer": r["layer"], "neuron": r["neuron"]},
            "cumulative_dSufficiency": cum_dS,
            "fraction_of_full": frac,
            "margin": tx_mean[M],
        })
        print(f"  {k:3d}  margin={tx_mean['logit_margin']:+.2f}  dS={cum_dS:+.4f}  "
              f"({frac:5.1%})  +L{r['layer']:02d}/N{r['neuron']:5d}")

    results["progressive_transplant"] = progressive_results

    # -------------------------------------------------------
    # Phase 5: Amplification sweep (behavioral only)
    # -------------------------------------------------------
    if task in ("behavioral", "sycophancy", "fc_refusal", "fc_belief"):
        print(f"\n--- Phase 5: Amplification sweep (top 5 neurons, control prompts) ---")
        multipliers = [2.0, 3.0, 5.0, 10.0]

        # Sort by single-neuron dS
        top_neurons = [(r["layer"], r["neuron"]) for r in sorted_by_suff[:5]]

        amp_results = []
        for l, n in top_neurons:
            neuron_amps = {"layer": l, "neuron": n, "multipliers": {}}
            for mult in multipliers:
                amp_ctx = make_amplify_single_ctx(steerer.model, l, n, mult)
                amp_mean, _ = _measure_control(make_ctx=amp_ctx)
                dA = amp_mean[M] - ctrl_baseline[M]
                neuron_amps["multipliers"][str(mult)] = {
                    **amp_mean, "dAmplify": dA,
                }
            amp_results.append(neuron_amps)

            # Print
            amp_strs = [f"{mult}x→dA={neuron_amps['multipliers'][str(mult)]['dAmplify']:+.3f}"
                        for mult in multipliers]
            print(f"  L{l:02d}/N{n:5d}: {', '.join(amp_strs)}")

        results["amplification"] = amp_results

    # -------------------------------------------------------
    # Phase 6: Random neuron controls
    # -------------------------------------------------------
    print(f"\n--- Phase 6: Random neuron transplant controls (n={n_random}) ---")

    d_mlp = steerer.model.config.intermediate_size
    bottleneck_layers = sorted(set(l for l, n in bottleneck_neurons))
    rng_mod.seed(42)
    random_dS = []

    for i in range(n_random):
        rand_layer = rng_mod.choice(bottleneck_layers)
        rand_neuron = rng_mod.randint(0, d_mlp - 1)
        # Collect source activation for this random neuron
        rand_src = collect_mean_neuron_activation(
            steerer, rand_layer, rand_neuron, source_prompts[:3],
            seed_response=seed, use_chat_template=uct,
        )
        tx_ctx = make_transplant_single_ctx(steerer.model, rand_layer, rand_neuron, rand_src)
        tx_mean, _ = _measure_control(make_ctx=tx_ctx)
        random_dS.append(tx_mean[M] - ctrl_baseline[M])

    rand_mean = float(np.mean(random_dS))
    rand_std = float(np.std(random_dS))
    print(f"  Random single-neuron transplant dS: mean={rand_mean:.4f} std={rand_std:.4f}")

    results["random_controls"] = {
        "n": n_random,
        "mean": rand_mean,
        "std": rand_std,
        "values": random_dS,
    }

    # Effect sizes
    print("\n--- Effect sizes (σ above random) ---")
    for sr in sorted_by_suff[:10]:
        if rand_std > 1e-10:
            sigma = (sr["dSufficiency"] - rand_mean) / rand_std
        else:
            sigma = 0.0
        sr["sigma_above_random"] = sigma
        print(f"  L{sr['layer']:02d}/N{sr['neuron']:5d}: "
              f"dS={sr['dSufficiency']:+.4f} ({sigma:+.1f}σ)")

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY — {task.upper()} SUFFICIENCY TEST")
    print(f"{'='*70}")
    print(f"  Control baseline margin: {ctrl_baseline[M]:+.2f}")
    print(f"  Full-circuit transplant dS: {dS_full:+.4f}")
    print(f"  Random single-neuron dS: {rand_mean:.4f} ± {rand_std:.4f}")

    if sorted_by_suff:
        top = sorted_by_suff[0]
        print(f"\n  Most sufficient neuron: L{top['layer']:02d}/N{top['neuron']:5d} "
              f"dS={top['dSufficiency']:+.4f}")

    if progressive_results:
        for threshold in [0.25, 0.50]:
            for pr in progressive_results:
                if abs(dS_full) > 1e-6 and pr["fraction_of_full"] >= threshold:
                    print(f"  {threshold:.0%} of full sufficiency at: k={pr['k']} neurons")
                    break

    # Verdict
    print()
    if abs(dS_full) > 0.5:
        print(f"  >> SUFFICIENT: full circuit transplant induces dS={dS_full:+.2f}")
        if sorted_by_suff and sorted_by_suff[0]["dSufficiency"] > 0.1:
            top = sorted_by_suff[0]
            print(f"  >> Single-neuron sufficiency: L{top['layer']:02d}/N{top['neuron']:5d} dS={top['dSufficiency']:+.4f}")
        else:
            print(f"  >> No single neuron is individually sufficient")
    elif abs(dS_full) > 0.1:
        print(f"  >> WEAK sufficiency: full circuit dS={dS_full:+.2f}")
    else:
        print(f"  >> NOT SUFFICIENT: full circuit transplant barely changes behavior (dS={dS_full:+.4f})")

    # Save
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"sufficiency_{task}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path / f'sufficiency_{task}.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sufficiency test: can activating neurons induce behavior?")
    parser.add_argument("--model", default="llama8b", choices=["llama8b"])
    parser.add_argument("--task", default="behavioral", choices=["behavioral", "factual", "sycophancy", "fc_refusal", "fc_belief", "all"])
    parser.add_argument("--topology_dir", type=str, required=True)
    parser.add_argument("--max_neurons", type=int, default=24)
    parser.add_argument("--n_random", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    model_name = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}[args.model]
    print(f"Loading model: {model_name}")
    steerer = NeuronSteerer(model_name)
    print(f"Model loaded on {steerer.device}")

    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/sufficiency_{args.model}_{timestamp}"

    tasks = ["behavioral", "factual", "sycophancy", "fc_refusal", "fc_belief"] if args.task == "all" else [args.task]
    all_results = {}

    for task in tasks:
        result = run_sufficiency_test(
            steerer, task, args.topology_dir,
            max_neurons=args.max_neurons,
            n_random=args.n_random,
            output_dir=args.output_dir,
        )
        if result:
            all_results[task] = result

    out_path = Path(args.output_dir)
    with open(out_path / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
