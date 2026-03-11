"""Matched comparison: Arora et al. evaluation protocol vs ours on the same circuits.

Codex's suggestion: run both evaluation protocols on identical discovered circuits
to isolate whether different conclusions come from different circuits or different
evaluation metrics.

Arora et al. protocol:
  - Mean ablation (set to dataset-mean activations)
  - Complement ablation (ablate everything OUTSIDE the circuit) for faithfulness
  - Direct ablation (ablate the circuit) for completeness
  - Ratio normalization: (metric_circuit - metric_null) / (metric_full - metric_null)
  - Metric: logit_diff (target - counterfactual)

Our protocol:
  - Zero ablation (set to 0)
  - Direct ablation only (ablate circuit neurons) for necessity
  - Transplant for sufficiency
  - Absolute delta + sigma (vs random neuron controls)
  - Metric: logit_margin (target - max_other)

This script runs the 2x2:
  {mean ablation, zero ablation} x {complement ablation, direct ablation}
plus both normalization schemes on the same measurements.

Usage:
    python experiments/arora_comparison.py --model llama8b --topology_base experiments/topology_llama8b_*/
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron_steer.core import NeuronSteerer, NeuronIdx, Circuit, steer_neurons
from experiments.surgical_ablation import (
    measure_R,
    measure_R_batch,
    load_circuit,
    resolve_topology_dir,
    TASK_CONFIGS,
)
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_TEST,
    BENIGN_PROMPTS,
    CAPITALS_DISCOVERY,
    CAPITALS_TEST,
)


# ============================================================
# Mean activation collection
# ============================================================

def collect_mean_activations(
    steerer: NeuronSteerer,
    prompts: List[str],
    seed_response: str = "",
    use_chat_template: bool = True,
) -> Dict[Tuple[int, int], float]:
    """Collect mean neuron activations across prompts at all positions.

    Returns dict of (layer, neuron) -> mean activation value.
    Used for mean ablation baseline (Arora's approach).
    """
    # Accumulate per-(layer, neuron) across prompts
    # We need activations at ALL positions for complement ablation
    accum = defaultdict(list)
    model = steerer.model

    for prompt in prompts:
        if use_chat_template:
            formatted = steerer._format_prompt(prompt, seed_response)
        else:
            formatted = prompt + seed_response
        input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)

        # Hook every layer's down_proj input
        layer_acts = {}
        hooks = []
        for i, layer in enumerate(model.model.layers):
            def make_hook(layer_idx):
                def hook_fn(module, args):
                    # args[0] shape: [1, seq_len, intermediate_size]
                    # Mean across sequence positions for each neuron
                    layer_acts[layer_idx] = args[0][0].detach().mean(dim=0)  # [intermediate_size]
                return hook_fn
            h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        for layer_idx, mean_act in layer_acts.items():
            for n in range(mean_act.shape[0]):
                accum[(layer_idx, n)].append(mean_act[n].item())

    # Average across prompts
    return {k: float(np.mean(v)) for k, v in accum.items()}


def collect_mean_activations_lastpos(
    steerer: NeuronSteerer,
    prompts: List[str],
    seed_response: str = "",
    use_chat_template: bool = True,
) -> Dict[Tuple[int, int], float]:
    """Collect mean neuron activations at LAST position only.

    More efficient variant — since we measure R(x) at last position,
    mean ablation values at last position may be more appropriate.
    """
    accum = defaultdict(list)
    model = steerer.model

    for prompt in prompts:
        if use_chat_template:
            formatted = steerer._format_prompt(prompt, seed_response)
        else:
            formatted = prompt + seed_response
        input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)

        layer_acts = {}
        hooks = []
        for i, layer in enumerate(model.model.layers):
            def make_hook(layer_idx):
                def hook_fn(module, args):
                    layer_acts[layer_idx] = args[0][0, -1].detach()  # [intermediate_size]
                return hook_fn
            h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        for layer_idx, act in layer_acts.items():
            for n in range(act.shape[0]):
                accum[(layer_idx, n)].append(act[n].item())

    return {k: float(np.mean(v)) for k, v in accum.items()}


# ============================================================
# Complement ablation context managers
# ============================================================

def make_complement_ctx_zero(model, circuit: Circuit):
    """Zero-ablate everything OUTSIDE the circuit (Arora's complement ablation, zero variant)."""
    # Get the set of (layer, neuron) pairs IN the circuit
    circuit_neurons: Set[Tuple[int, int]] = set()
    for nidx in circuit.neurons:
        circuit_neurons.add((nidx.layer, nidx.neuron))

    # Group circuit neurons by layer for efficient hook
    by_layer: Dict[int, Set[int]] = defaultdict(set)
    for l, n in circuit_neurons:
        by_layer[l].add(n)

    # For each layer, zero ALL neurons EXCEPT those in the circuit
    hooks = []
    n_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size

    for layer_idx in range(n_layers):
        keep_neurons = by_layer.get(layer_idx, set())
        if not keep_neurons:
            # No circuit neurons in this layer — zero everything
            def make_hook_all():
                def pre_hook(module, args):
                    x = args[0].clone()
                    x[:, :, :] = 0.0
                    return (x,)
                return pre_hook
            h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(make_hook_all())
        else:
            # Zero everything except kept neurons
            keep_set = sorted(keep_neurons)
            keep_tensor = torch.tensor(keep_set, dtype=torch.long)

            def make_hook_partial(kt):
                def pre_hook(module, args):
                    x = args[0].clone()
                    # Save kept values
                    device_kt = kt.to(x.device)
                    saved = x[:, :, device_kt].clone()
                    # Zero everything
                    x[:, :, :] = 0.0
                    # Restore kept
                    x[:, :, device_kt] = saved
                    return (x,)
                return pre_hook
            h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(make_hook_partial(keep_tensor))
        hooks.append(h)

    return hooks


def make_complement_ctx_mean(model, circuit: Circuit, mean_acts: Dict[Tuple[int, int], float]):
    """Mean-ablate everything OUTSIDE the circuit (Arora's complement ablation, mean variant)."""
    circuit_neurons: Set[Tuple[int, int]] = set()
    for nidx in circuit.neurons:
        circuit_neurons.add((nidx.layer, nidx.neuron))

    by_layer: Dict[int, Set[int]] = defaultdict(set)
    for l, n in circuit_neurons:
        by_layer[l].add(n)

    hooks = []
    n_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size

    for layer_idx in range(n_layers):
        keep_neurons = by_layer.get(layer_idx, set())

        # Build mean activation tensor for this layer
        mean_tensor = torch.zeros(intermediate_size)
        for n in range(intermediate_size):
            if (layer_idx, n) in mean_acts:
                mean_tensor[n] = mean_acts[(layer_idx, n)]

        if not keep_neurons:
            def make_hook_all(mt):
                def pre_hook(module, args):
                    x = args[0].clone()
                    x[:, :, :] = mt.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
                    return (x,)
                return pre_hook
            h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                make_hook_all(mean_tensor)
            )
        else:
            keep_set = sorted(keep_neurons)
            keep_tensor = torch.tensor(keep_set, dtype=torch.long)

            def make_hook_partial(kt, mt):
                def pre_hook(module, args):
                    x = args[0].clone()
                    device_kt = kt.to(x.device)
                    saved = x[:, :, device_kt].clone()
                    x[:, :, :] = mt.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
                    x[:, :, device_kt] = saved
                    return (x,)
                return pre_hook
            h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                make_hook_partial(keep_tensor, mean_tensor)
            )
        hooks.append(h)

    return hooks


def make_direct_ctx_mean(model, circuit: Circuit, mean_acts: Dict[Tuple[int, int], float]):
    """Mean-ablate only the circuit neurons (our direct ablation, mean variant)."""
    by_layer: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for nidx in circuit.neurons:
        mean_val = mean_acts.get((nidx.layer, nidx.neuron), 0.0)
        by_layer[nidx.layer].append((nidx.neuron, mean_val))

    hooks = []
    for layer_idx, neuron_vals in by_layer.items():
        indices = torch.tensor([n for n, _ in neuron_vals], dtype=torch.long)
        values = torch.tensor([v for _, v in neuron_vals], dtype=torch.float)

        def make_hook(idx_t, val_t):
            def pre_hook(module, args):
                x = args[0].clone()
                device_idx = idx_t.to(x.device)
                device_val = val_t.to(x.device, dtype=x.dtype)
                # Replace circuit neurons with their mean values
                x[:, :, device_idx] = device_val.unsqueeze(0).unsqueeze(0)
                return (x,)
            return pre_hook

        h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_hook(indices, values)
        )
        hooks.append(h)

    return hooks


# ============================================================
# Full ablation (null baseline for Arora normalization)
# ============================================================

def make_full_ablation_zero(model):
    """Zero-ablate ALL neurons in ALL layers. Null baseline."""
    hooks = []
    for layer_idx in range(len(model.model.layers)):
        def make_hook():
            def pre_hook(module, args):
                x = args[0].clone()
                x[:, :, :] = 0.0
                return (x,)
            return pre_hook
        h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(make_hook())
        hooks.append(h)
    return hooks


def make_full_ablation_mean(model, mean_acts: Dict[Tuple[int, int], float]):
    """Mean-ablate ALL neurons in ALL layers. Null baseline."""
    intermediate_size = model.config.intermediate_size
    hooks = []
    for layer_idx in range(len(model.model.layers)):
        mean_tensor = torch.zeros(intermediate_size)
        for n in range(intermediate_size):
            if (layer_idx, n) in mean_acts:
                mean_tensor[n] = mean_acts[(layer_idx, n)]

        def make_hook(mt):
            def pre_hook(module, args):
                x = args[0].clone()
                x[:, :, :] = mt.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
                return (x,)
            return pre_hook
        h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(make_hook(mean_tensor))
        hooks.append(h)
    return hooks


# ============================================================
# Measurement with hook lists (non-context-manager pattern)
# ============================================================

def measure_with_hooks(
    steerer: NeuronSteerer,
    prompt: str,
    target_token: str,
    hooks: list,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> dict:
    """Measure R(x) with pre-installed hooks, then remove them."""
    try:
        return measure_R(steerer, prompt, target_token,
                         seed_response=seed_response,
                         use_chat_template=use_chat_template)
    finally:
        for h in hooks:
            h.remove()


def measure_batch_with_hook_factory(
    steerer: NeuronSteerer,
    prompts: List[str],
    target_token: str,
    hook_factory: Callable,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> Tuple[dict, List[dict]]:
    """Measure mean R(x) across prompts, creating fresh hooks for each.

    hook_factory: callable that returns a list of hooks (installed on model).
    Must create fresh hooks each call since they're removed after each prompt.
    """
    results = []
    for p in prompts:
        hooks = hook_factory()
        try:
            r = measure_R(steerer, p, target_token,
                          seed_response=seed_response,
                          use_chat_template=use_chat_template)
        finally:
            for h in hooks:
                h.remove()
        results.append(r)

    mean_dict = {
        "prob": float(np.mean([r["prob"] for r in results])),
        "logit": float(np.mean([r["logit"] for r in results])),
        "logit_margin": float(np.mean([r["logit_margin"] for r in results])),
    }
    return mean_dict, results


# ============================================================
# Arora normalization
# ============================================================

def arora_faithfulness(metric_circuit, metric_null, metric_full):
    """Arora et al. faithfulness: how well circuit preserves behavior.

    Faithfulness(C) = (m(C) - m(null)) / (m(full) - m(null))
    where C = complement ablated (only circuit active),
    null = everything ablated, full = no ablation.
    """
    denom = metric_full - metric_null
    if abs(denom) < 1e-8:
        return 0.0
    return (metric_circuit - metric_null) / denom


def arora_completeness(metric_complement, metric_null, metric_full):
    """Arora et al. completeness: how much ablating circuit degrades behavior.

    Completeness(C) = (m(C_bar) - m(null)) / (m(full) - m(null))
    where C_bar = circuit ablated (only complement active).
    Ideal = 0 (ablating circuit completely kills behavior).
    """
    denom = metric_full - metric_null
    if abs(denom) < 1e-8:
        return 0.0
    return (metric_complement - metric_null) / denom


# ============================================================
# Main comparison
# ============================================================

def run_comparison(
    steerer: NeuronSteerer,
    task: str,
    topology_base: str,
    n_random: int = 20,
    output_dir: str = None,
):
    """Run matched Arora vs ours comparison on a single task."""
    config = TASK_CONFIGS[task]
    topo_dir = resolve_topology_dir(topology_base, task, config)

    circuit_path = topo_dir / "circuit.json"
    if not circuit_path.exists():
        print(f"  ERROR: {circuit_path} not found, skipping {task}")
        return None

    circuit = load_circuit(str(circuit_path))
    circuit_size = len(set((n.layer, n.neuron) for n in circuit.neurons))

    target_token = config["target_token"]
    seed_response = config["seed_response"]
    uct = config["use_chat_template"]

    # For factual, use first test prompt's target
    if task == "factual":
        test_prompts = [p for p, t in CAPITALS_TEST]
        test_targets = [t for p, t in CAPITALS_TEST]
        target_token = test_targets[0]  # for display
        discovery_prompts = [p for p, t in CAPITALS_DISCOVERY]
    else:
        test_prompts = config["test_prompts"]
        test_targets = None
        discovery_prompts = REFUSAL_DISCOVERY_POSITIVE if task == "behavioral" else test_prompts

    print(f"\n{'='*70}")
    print(f"  ARORA vs OURS COMPARISON: {task.upper()}")
    print(f"  Circuit: {circuit_size} unique (layer, neuron) pairs")
    print(f"{'='*70}")

    # Step 1: Collect mean activations for mean ablation (Arora's approach)
    print(f"\n  [1/6] Collecting mean activations for mean ablation baseline...")
    t0 = time.time()
    mean_acts = collect_mean_activations_lastpos(
        steerer, discovery_prompts,
        seed_response=seed_response, use_chat_template=uct,
    )
    print(f"         Done in {time.time()-t0:.1f}s ({len(mean_acts)} neurons)")

    # Helper for factual per-prompt targets
    def measure_task(prompts, targets, hook_factory=None):
        """Measure R across prompts, handling per-prompt targets for factual."""
        if targets is not None:
            # Per-prompt targets (factual)
            results = []
            for p, t in zip(prompts, targets):
                if hook_factory:
                    hooks = hook_factory()
                    try:
                        r = measure_R(steerer, p, t, seed_response=seed_response, use_chat_template=uct)
                    finally:
                        for h in hooks:
                            h.remove()
                else:
                    r = measure_R(steerer, p, t, seed_response=seed_response, use_chat_template=uct)
                results.append(r)
        else:
            if hook_factory:
                _, results = measure_batch_with_hook_factory(
                    steerer, prompts, target_token, hook_factory,
                    seed_response=seed_response, use_chat_template=uct,
                )
            else:
                _, results = measure_R_batch(
                    steerer, prompts, target_token,
                    seed_response=seed_response, use_chat_template=uct,
                )
        mean_dict = {
            "prob": float(np.mean([r["prob"] for r in results])),
            "logit": float(np.mean([r["logit"] for r in results])),
            "logit_margin": float(np.mean([r["logit_margin"] for r in results])),
        }
        return mean_dict

    # Step 2: Baselines
    print(f"\n  [2/6] Measuring baselines...")
    m_full = measure_task(test_prompts, test_targets)
    print(f"    Full model:  prob={m_full['prob']:.4f}  margin={m_full['logit_margin']:.3f}")

    # Null baselines (everything ablated)
    m_null_zero = measure_task(
        test_prompts, test_targets,
        lambda: make_full_ablation_zero(steerer.model),
    )
    m_null_mean = measure_task(
        test_prompts, test_targets,
        lambda: make_full_ablation_mean(steerer.model, mean_acts),
    )
    print(f"    Null (zero): prob={m_null_zero['prob']:.4f}  margin={m_null_zero['logit_margin']:.3f}")
    print(f"    Null (mean): prob={m_null_mean['prob']:.4f}  margin={m_null_mean['logit_margin']:.3f}")

    # Step 3: The 2x2 — {zero, mean} x {direct, complement}
    print(f"\n  [3/6] Running 2x2 ablation grid...")

    def measure_task_with_ctx(prompts, targets, make_ctx):
        """Measure with a context manager factory (for steer_neurons)."""
        results = []
        if targets is not None:
            for p, t in zip(prompts, targets):
                r = measure_R(steerer, p, t, make_ctx=make_ctx,
                              seed_response=seed_response, use_chat_template=uct)
                results.append(r)
        else:
            for p in prompts:
                r = measure_R(steerer, p, target_token, make_ctx=make_ctx,
                              seed_response=seed_response, use_chat_template=uct)
                results.append(r)
        return {
            "prob": float(np.mean([r["prob"] for r in results])),
            "logit": float(np.mean([r["logit"] for r in results])),
            "logit_margin": float(np.mean([r["logit_margin"] for r in results])),
        }

    # 3a: Direct ablation, zero (OURS)
    m_direct_zero = measure_task_with_ctx(
        test_prompts, test_targets,
        lambda: steer_neurons(steerer.model, circuit.neurons, multiplier=0.0, all_positions=True),
    )
    print(f"    Direct+Zero:       prob={m_direct_zero['prob']:.4f}  margin={m_direct_zero['logit_margin']:.3f}")

    # 3b: Direct ablation, mean (Arora's direct/completeness)
    m_direct_mean = measure_task(
        test_prompts, test_targets,
        lambda: make_direct_ctx_mean(steerer.model, circuit, mean_acts),
    )
    print(f"    Direct+Mean:       prob={m_direct_mean['prob']:.4f}  margin={m_direct_mean['logit_margin']:.3f}")

    # 3c: Complement ablation, zero (faithfulness, zero variant)
    m_complement_zero = measure_task(
        test_prompts, test_targets,
        lambda: make_complement_ctx_zero(steerer.model, circuit),
    )
    print(f"    Complement+Zero:   prob={m_complement_zero['prob']:.4f}  margin={m_complement_zero['logit_margin']:.3f}")

    # 3d: Complement ablation, mean (ARORA'S STANDARD)
    m_complement_mean = measure_task(
        test_prompts, test_targets,
        lambda: make_complement_ctx_mean(steerer.model, circuit, mean_acts),
    )
    print(f"    Complement+Mean:   prob={m_complement_mean['prob']:.4f}  margin={m_complement_mean['logit_margin']:.3f}")

    # Step 4: Compute Arora's metrics
    print(f"\n  [4/6] Computing Arora metrics (faithfulness/completeness)...")

    # Arora faithfulness: complement ablation preserves behavior
    # Using their standard: mean ablation, logit_margin
    faith_mean = arora_faithfulness(
        m_complement_mean["logit_margin"],
        m_null_mean["logit_margin"],
        m_full["logit_margin"],
    )
    # Also compute with zero ablation for comparison
    faith_zero = arora_faithfulness(
        m_complement_zero["logit_margin"],
        m_null_zero["logit_margin"],
        m_full["logit_margin"],
    )

    # Arora completeness: direct ablation degrades behavior
    comp_mean = arora_completeness(
        m_direct_mean["logit_margin"],
        m_null_mean["logit_margin"],
        m_full["logit_margin"],
    )
    comp_zero = arora_completeness(
        m_direct_zero["logit_margin"],
        m_null_zero["logit_margin"],
        m_full["logit_margin"],
    )

    print(f"    Faithfulness (mean ablation): {faith_mean:.4f}")
    print(f"    Faithfulness (zero ablation): {faith_zero:.4f}")
    print(f"    Completeness (mean ablation): {comp_mean:.4f}  (ideal=0)")
    print(f"    Completeness (zero ablation): {comp_zero:.4f}  (ideal=0)")

    # Step 5: Compute our metrics
    print(f"\n  [5/6] Computing our metrics (N_H, delta, sigma)...")

    # N_H = baseline - ablated (direct, zero)
    nh_zero = m_full["logit_margin"] - m_direct_zero["logit_margin"]
    nh_mean = m_full["logit_margin"] - m_direct_mean["logit_margin"]

    # Random controls for sigma
    rng = np.random.default_rng(42)
    n_layers = len(steerer.model.model.layers)
    intermediate = steerer.model.config.intermediate_size
    random_deltas = []

    for i in range(n_random):
        rl = int(rng.integers(0, n_layers))
        rn = int(rng.integers(0, intermediate))
        # Build a 1-neuron "circuit" for direct zero ablation
        r_abl = measure_task_with_ctx(
            test_prompts, test_targets,
            lambda l=rl, n=rn: steer_neurons(
                steerer.model,
                {NeuronIdx(layer=l, position=-1, neuron=n): 0.0},
                multiplier=0.0, all_positions=True,
            ),
        )
        random_deltas.append(m_full["logit_margin"] - r_abl["logit_margin"])

    sigma_null = float(np.std(random_deltas)) if random_deltas else 1.0
    sigma_mean = float(np.mean(random_deltas))

    nh_sigma = nh_zero / sigma_null if sigma_null > 0 else 0.0

    print(f"    N_H (zero ablation):  {nh_zero:.4f}  ({nh_sigma:.1f}σ)")
    print(f"    N_H (mean ablation):  {nh_mean:.4f}")
    print(f"    Random control: mean={sigma_mean:.6f}, std={sigma_null:.6f}")

    # Step 6: Summary comparison
    print(f"\n  [6/6] SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Metric':40s} {'Value':>10s}")
    print(f"  {'-'*60}")
    print(f"  Full model logit_margin:               {m_full['logit_margin']:>10.4f}")
    print(f"  Null (zero) logit_margin:              {m_null_zero['logit_margin']:>10.4f}")
    print(f"  Null (mean) logit_margin:              {m_null_mean['logit_margin']:>10.4f}")
    print(f"  ")
    print(f"  Direct+Zero logit_margin:              {m_direct_zero['logit_margin']:>10.4f}")
    print(f"  Direct+Mean logit_margin:              {m_direct_mean['logit_margin']:>10.4f}")
    print(f"  Complement+Zero logit_margin:          {m_complement_zero['logit_margin']:>10.4f}")
    print(f"  Complement+Mean logit_margin:          {m_complement_mean['logit_margin']:>10.4f}")
    print(f"  ")
    print(f"  --- Arora Protocol ---")
    print(f"  Faithfulness (mean):                   {faith_mean:>10.4f}")
    print(f"  Faithfulness (zero):                   {faith_zero:>10.4f}")
    print(f"  Completeness (mean, ideal=0):          {comp_mean:>10.4f}")
    print(f"  Completeness (zero, ideal=0):          {comp_zero:>10.4f}")
    print(f"  ")
    print(f"  --- Our Protocol ---")
    print(f"  N_H (zero, absolute):                  {nh_zero:>10.4f}")
    print(f"  N_H (zero, sigma):                     {nh_sigma:>10.1f}σ")
    print(f"  N_H (mean, absolute):                  {nh_mean:>10.4f}")
    print(f"  {'='*60}")

    results = {
        "task": task,
        "circuit_size": circuit_size,
        "full_model": m_full,
        "null_zero": m_null_zero,
        "null_mean": m_null_mean,
        "direct_zero": m_direct_zero,
        "direct_mean": m_direct_mean,
        "complement_zero": m_complement_zero,
        "complement_mean": m_complement_mean,
        "arora_faithfulness_mean": faith_mean,
        "arora_faithfulness_zero": faith_zero,
        "arora_completeness_mean": comp_mean,
        "arora_completeness_zero": comp_zero,
        "our_nh_zero": nh_zero,
        "our_nh_mean": nh_mean,
        "our_nh_sigma": nh_sigma,
        "random_controls": {
            "n": n_random,
            "mean_delta": sigma_mean,
            "std_delta": sigma_null,
        },
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"arora_comparison_{task}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {out_path}/arora_comparison_{task}.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="Matched Arora vs Ours comparison")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--topology_base", required=True, help="Base directory with topology results")
    parser.add_argument("--task", default="all", help="Task to evaluate (behavioral, factual, or all)")
    parser.add_argument("--n_random", type=int, default=20, help="Random neuron controls")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    steerer = NeuronSteerer(args.model)

    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = f"experiments/arora_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    tasks = ["behavioral", "factual"] if args.task == "all" else [args.task]
    all_results = {}

    for task in tasks:
        result = run_comparison(
            steerer, task, args.topology_base,
            n_random=args.n_random,
            output_dir=args.output_dir,
        )
        if result:
            all_results[task] = result

    # Cross-task summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-TASK SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Task':15s} {'Faith(mean)':>12s} {'Faith(zero)':>12s} {'Comp(mean)':>12s} {'Comp(zero)':>12s} {'N_H(zero)':>12s} {'N_H(σ)':>10s}")
        for task, r in all_results.items():
            print(f"  {task:15s} {r['arora_faithfulness_mean']:>12.4f} {r['arora_faithfulness_zero']:>12.4f} {r['arora_completeness_mean']:>12.4f} {r['arora_completeness_zero']:>12.4f} {r['our_nh_zero']:>12.4f} {r['our_nh_sigma']:>9.1f}σ")

    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
