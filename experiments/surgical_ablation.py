"""Surgical single-neuron causal ablation.

Tests whether individual bottleneck neurons (from topology analysis) are
causally necessary for task behavior. The true bottleneck test: edge degree
is correlation, ablation is causation.

Protocol:
  1. Baseline R(x) on held-out test prompts
  2. For each bottleneck neuron (layer, neuron):
     a. Zero-ablate that single neuron at ALL positions → measure R(x) drop
     b. Zero-ablate at LAST token position only → measure R(x) drop
  3. Small-set ablation (top-2, top-3, top-5 bottleneck neurons together)
  4. Full circuit ablation (known N_H from k* analysis)
  5. Random single-neuron controls (null distribution)

This answers: does killing one neuron break the behavior?
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron_steer.core import NeuronSteerer, NeuronIdx, Circuit, steer_neurons
from experiments.prompts import (
    CAPITALS_DISCOVERY,
    CAPITALS_TEST,
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_TEST,
    BENIGN_PROMPTS,
    SVA_PROMPTS,
    SYCOPHANCY_DISCOVERY_POSITIVE,
    SYCOPHANCY_TEST,
    FC_REFUSAL_DISCOVERY_POSITIVE,
    FC_REFUSAL_TEST,
    FC_BENIGN,
    FC_BELIEF_DISCOVERY,
    FC_BELIEF_TEST,
    FC_BELIEF_NO_DISCOVERY,
    FC_BELIEF_NO_TEST,
    FC_REFUSAL_MIXED_DISCOVERY,
    FC_REFUSAL_MIXED_TEST,
)


# ============================================================
# Measure R(x)
# ============================================================

def measure_R(
    steerer: NeuronSteerer,
    prompt: str,
    target_token: str,
    make_ctx: Optional[Callable] = None,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> dict:
    """Measure R(x) = P(target_token) under optional steering context.

    Returns dict with 'prob', 'logit', and 'logit_margin' (target - max_other).
    make_ctx: callable that returns a fresh context manager each time.
    """
    if use_chat_template:
        formatted = steerer._format_prompt(prompt, seed_response)
    else:
        formatted = prompt + seed_response
    input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)
    target_id = steerer.tokenizer.encode(target_token, add_special_tokens=False)[-1]

    ctx = make_ctx() if make_ctx is not None else nullcontext()

    with ctx:
        with torch.no_grad():
            outputs = steerer.model(input_ids)
            logits = outputs.logits[0, -1].float()  # float32 for precision
            probs = F.softmax(logits, dim=-1)
            target_logit = logits[target_id].item()
            target_prob = probs[target_id].item()
            # Margin: target logit minus max logit among all OTHER tokens
            masked = logits.clone()
            masked[target_id] = float('-inf')
            max_other = masked.max().item()
            return {
                "prob": target_prob,
                "logit": target_logit,
                "logit_margin": target_logit - max_other,
            }


def measure_R_batch(
    steerer: NeuronSteerer,
    prompts: List[str],
    target_token: str,
    make_ctx: Optional[Callable] = None,
    seed_response: str = "",
    use_chat_template: bool = True,
) -> Tuple[dict, List[dict]]:
    """Measure mean R(x) across prompts. Returns (mean_dict, individual_dicts)."""
    results = []
    for p in prompts:
        r = measure_R(steerer, p, target_token, make_ctx=make_ctx,
                      seed_response=seed_response, use_chat_template=use_chat_template)
        results.append(r)
    mean_dict = {
        "prob": float(np.mean([r["prob"] for r in results])),
        "logit": float(np.mean([r["logit"] for r in results])),
        "logit_margin": float(np.mean([r["logit_margin"] for r in results])),
    }
    return mean_dict, results


# ============================================================
# Context manager factories
# ============================================================

def make_single_neuron_ctx(model, layer: int, neuron: int):
    """Returns a factory for zero-ablating a single neuron at ALL positions."""
    def factory():
        neurons = {NeuronIdx(layer=layer, position=-1, neuron=neuron): 0.0}
        return steer_neurons(model, neurons, multiplier=0.0, all_positions=True)
    return factory


def make_single_neuron_lastpos_ctx(model, layer: int, neuron: int, seq_len: int):
    """Returns a factory for zero-ablating a single neuron at the LAST position."""
    def factory():
        neurons = {NeuronIdx(layer=layer, position=seq_len - 1, neuron=neuron): 0.0}
        return steer_neurons(model, neurons, multiplier=0.0, all_positions=False)
    return factory


def make_multi_neuron_ctx(model, neuron_list: List[Tuple[int, int]]):
    """Returns a factory for zero-ablating multiple neurons at ALL positions."""
    def factory():
        neurons = {}
        for l, n in neuron_list:
            neurons[NeuronIdx(layer=l, position=-1, neuron=n)] = 0.0
        return steer_neurons(model, neurons, multiplier=0.0, all_positions=True)
    return factory


def make_circuit_ctx(model, circuit: Circuit):
    """Returns a factory for zero-ablating an entire circuit."""
    def factory():
        return steer_neurons(model, circuit.neurons, multiplier=0.0, all_positions=True)
    return factory


# ============================================================
# Load topology analysis results
# ============================================================

def resolve_topology_dir(topology_base: str, task: str, config: dict) -> Path:
    """Resolve the topology directory for a task.

    If config["topology_dir"] is set, use it directly.
    Otherwise, auto-detect by scanning topology_base for relp-{task}_kstar* dirs.
    """
    if config.get("topology_dir"):
        return Path(topology_base) / config["topology_dir"]
    # Auto-detect: find relp-{task}_kstar* directories
    base = Path(topology_base)
    matches = sorted(base.glob(f"relp-{task}_kstar*"))
    if matches:
        return matches[-1]  # latest/largest k*
    # Also try relp-{task}_k* (non-star topology comparison dirs)
    matches = sorted(base.glob(f"relp-{task}_k*"))
    if matches:
        # Prefer kstar dirs
        kstar = [m for m in matches if "kstar" in m.name]
        return kstar[-1] if kstar else matches[-1]
    return base / f"relp-{task}_NOTFOUND"


def load_bottleneck_candidates(analysis_path: str) -> List[dict]:
    """Load position-aware bottleneck candidates from topology analysis."""
    with open(analysis_path) as f:
        data = json.load(f)
    return data.get("bottlenecks_position_aware", [])


def unique_bottleneck_neurons(candidates: List[dict]) -> List[Tuple[int, int]]:
    """Collapse position-aware bottlenecks to unique (layer, neuron) pairs.

    Preserves order by first appearance (highest combined degree first).
    """
    seen = set()
    result = []
    for c in candidates:
        key = (c["layer"], c["neuron"])
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def unique_circuit_neurons(circuit: Circuit) -> List[Tuple[int, int]]:
    """Collapse a position-aware circuit to unique (layer, neuron) pairs.

    Sorts by the largest absolute attribution across positions so full-circuit
    intervention runs keep the same "most-attributed first" convention as the
    bottleneck list while covering every unique MLP neuron in the circuit.
    """
    best_attr: Dict[Tuple[int, int], float] = {}
    for nidx, weight in circuit.neurons.items():
        key = (nidx.layer, nidx.neuron)
        if key not in best_attr or abs(weight) > abs(best_attr[key]):
            best_attr[key] = weight
    return [
        key for key, _ in sorted(
            best_attr.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
    ]


def load_circuit(circuit_path: str) -> Circuit:
    """Load a circuit from saved JSON.

    Format: {"neurons": {"layer,position,neuron": weight, ...}, ...}
    """
    with open(circuit_path) as f:
        data = json.load(f)
    neurons = {}
    for key, weight in data["neurons"].items():
        parts = key.split(",")
        layer, position, neuron = int(parts[0]), int(parts[1]), int(parts[2])
        nidx = NeuronIdx(layer=layer, position=position, neuron=neuron)
        neurons[nidx] = weight
    return Circuit(
        neurons=neurons,
        prompt=data.get("prompt", ""),
        target_token=data.get("target_token", ""),
        total_logit_diff=data.get("total_logit_diff", 0.0),
    )


# ============================================================
# Task configurations
# ============================================================

TASK_CONFIGS = {
    "behavioral": {
        "target_token": "I",  # no space — after chat template \n\n, first token is "I" (token 40)
        "seed_response": "",
        "use_chat_template": True,
        "test_prompts": REFUSAL_TEST,
        "control_prompts": BENIGN_PROMPTS[:5],
        "topology_dir": "relp-behavioral_refusal_kstar91",
    },
    "factual": {
        "target_token": None,  # varies per prompt
        "seed_response": "Answer: ",  # matches circuit_eval_protocol.py
        "use_chat_template": True,
        "test_prompts": CAPITALS_TEST,
        "control_prompts": [],
        "topology_dir": "relp-factual_capitals_kstar114",
    },
    "sycophancy": {
        "target_token": "I",  # sycophantic responses start "I agree..." / "I think you're right..."
        "seed_response": "",
        "use_chat_template": True,
        "test_prompts": SYCOPHANCY_TEST,
        "control_prompts": BENIGN_PROMPTS[:5],
        "topology_dir": None,  # set dynamically after k* discovery
    },
    "fc_refusal": {
        "target_token": "No",  # forced-choice refusal: model says "No" to harmful requests
        "seed_response": "",
        "use_chat_template": True,
        "test_prompts": FC_REFUSAL_TEST,
        "control_prompts": FC_BENIGN,
        "topology_dir": None,
    },
    "fc_belief": {
        "target_token": "Yes",  # forced-choice belief: model affirms opinion questions
        "seed_response": "",
        "use_chat_template": True,
        "test_prompts": FC_BELIEF_TEST,
        "control_prompts": FC_BELIEF_NO_TEST,  # "No" prompts as control for sufficiency
        "topology_dir": None,
    },
    "fc_refusal_mixed": {
        "target_token": None,  # mixed: per-prompt targets
        "seed_response": "",
        "use_chat_template": True,
        "test_prompts": [p for p, t in FC_REFUSAL_MIXED_TEST],
        "test_targets": [t for p, t in FC_REFUSAL_MIXED_TEST],
        "control_prompts": [],
        "topology_dir": None,
    },
}


# ============================================================
# Main experiment
# ============================================================

def run_surgical_ablation(
    steerer: NeuronSteerer,
    task: str,
    topology_base: str,
    n_random: int = 20,
    output_dir: str = None,
    filter_universal: set = None,
    candidate_source: str = "bottlenecks",
):
    """Run surgical ablation experiment for a task.

    Args:
        filter_universal: Optional set of (layer, neuron) tuples to exclude
            from bottleneck analysis (e.g., super-weight neurons found in all circuits).
    """
    config = TASK_CONFIGS[task]
    topo_dir = resolve_topology_dir(topology_base, task, config)

    print(f"\n{'='*70}")
    print(f"SURGICAL ABLATION: {task.upper()}")
    print(f"{'='*70}")

    # Load topology data
    analysis_path = topo_dir / "analysis.json"
    circuit_path = topo_dir / "circuit.json"

    if not analysis_path.exists():
        print(f"  ERROR: {analysis_path} not found, skipping")
        return None

    circuit = load_circuit(str(circuit_path))
    candidates = load_bottleneck_candidates(str(analysis_path))
    if candidate_source == "circuit":
        bottleneck_neurons = unique_circuit_neurons(circuit)
    else:
        bottleneck_neurons = unique_bottleneck_neurons(candidates)

    if filter_universal:
        n_before = len(bottleneck_neurons)
        bottleneck_neurons = [(l, n) for l, n in bottleneck_neurons
                              if (l, n) not in filter_universal]
        n_removed = n_before - len(bottleneck_neurons)
        if n_removed:
            print(f"  Filtered {n_removed} universal neurons from bottleneck list")

    print(f"  Circuit size: {len(circuit.neurons)} neurons")
    print(f"  Candidate source: {candidate_source}")
    print(f"  Bottleneck candidates: {len(candidates)} (position-aware)")
    print(f"  Unique candidate neurons: {len(bottleneck_neurons)}")
    print(f"  Top 10: {['L{}/N{}'.format(l,n) for l,n in bottleneck_neurons[:10]]}")

    uct = config["use_chat_template"]
    seed = config["seed_response"]

    results = {
        "task": task,
        "circuit_size": len(circuit.neurons),
        "candidate_source": candidate_source,
        "n_bottleneck_candidates": len(candidates),
        "n_unique_candidate_neurons": len(bottleneck_neurons),
        "bottleneck_neurons": [{"layer": l, "neuron": n} for l, n in bottleneck_neurons],
        "candidate_neurons": [{"layer": l, "neuron": n} for l, n in bottleneck_neurons],
    }

    # -------------------------------------------------------
    # Phase 0: Debug — verify target token encoding
    # -------------------------------------------------------
    if task == "behavioral":
        target_token = config["target_token"]
        tok_ids = steerer.tokenizer.encode(target_token, add_special_tokens=False)
        print(f"\n  DEBUG: target_token={repr(target_token)} → token_ids={tok_ids}")
        print(f"  DEBUG: using token_id={tok_ids[-1]} → "
              f"decoded={repr(steerer.tokenizer.decode([tok_ids[-1]]))}")

        # Quick sanity: what does the model actually predict for one prompt?
        test_p = config["test_prompts"][0]
        formatted = steerer._format_prompt(test_p, seed)
        input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)
        print(f"  DEBUG: prompt={repr(test_p)}")
        print(f"  DEBUG: formatted length={input_ids.shape[1]} tokens")
        print(f"  DEBUG: last 5 tokens={input_ids[0, -5:].tolist()}")
        print(f"  DEBUG: decoded last 5={[steerer.tokenizer.decode([t]) for t in input_ids[0, -5:].tolist()]}")
        with torch.no_grad():
            outputs = steerer.model(input_ids)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            target_id = tok_ids[-1]
            print(f"  DEBUG: P(target={target_id})={probs[target_id].item():.6f}")
            top5 = torch.topk(probs, 5)
            for i in range(5):
                tid = top5.indices[i].item()
                tp = top5.values[i].item()
                print(f"  DEBUG: top-{i+1}: token={tid} "
                      f"({repr(steerer.tokenizer.decode([tid]))}) P={tp:.6f}")

    # Helper: run measure_R for factual (per-prompt targets) or behavioral (single target)
    def _measure_task(make_ctx=None):
        """Run measurement across test prompts. Returns mean dict."""
        if task == "factual":
            rr = []
            for prompt, target in config["test_prompts"]:
                rr.append(measure_R(steerer, prompt, target, make_ctx=make_ctx,
                                    seed_response=seed, use_chat_template=uct))
            mean_d = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
            return mean_d, rr
        else:
            return measure_R_batch(steerer, config["test_prompts"], target_token,
                                   make_ctx=make_ctx, seed_response=seed, use_chat_template=uct)

    # Primary metric key: logit_margin resolves softmax saturation
    M = "logit_margin"

    # -------------------------------------------------------
    # Phase 1: Baseline
    # -------------------------------------------------------
    print("\n--- Phase 1: Baseline R(x) ---")

    if task != "factual":
        target_token = config["target_token"]

    baseline_mean, baseline_per = _measure_task()

    if task == "factual":
        for (prompt, target), r in zip(config["test_prompts"], baseline_per):
            print(f"  {prompt[:40]:40s} → P={r['prob']:.4f}  logit={r['logit']:.2f}  margin={r['logit_margin']:.2f}")
        results["baselines_per_prompt"] = [
            {"prompt": p, "target": t, **r}
            for (p, t), r in zip(config["test_prompts"], baseline_per)
        ]
    else:
        for p, r in zip(config["test_prompts"], baseline_per):
            print(f"  {p[:40]:40s} → P={r['prob']:.4f}  logit={r['logit']:.2f}  margin={r['logit_margin']:.2f}")
        results["baselines_per_prompt"] = [
            {"prompt": p, **r}
            for p, r in zip(config["test_prompts"], baseline_per)
        ]

    print(f"\n  Baseline: P={baseline_mean['prob']:.6f}  logit={baseline_mean['logit']:.2f}  margin={baseline_mean['logit_margin']:.2f}")
    results["baseline"] = baseline_mean

    # -------------------------------------------------------
    # Phase 2: Full circuit ablation (reference)
    # -------------------------------------------------------
    print("\n--- Phase 2: Full circuit ablation (reference) ---")

    circuit_ctx = make_circuit_ctx(steerer.model, circuit)
    full_mean, _ = _measure_task(make_ctx=circuit_ctx)

    dM_full = baseline_mean[M] - full_mean[M]
    print(f"  Full ablated: P={full_mean['prob']:.6f}  logit={full_mean['logit']:.2f}  margin={full_mean['logit_margin']:.2f}")
    print(f"  dMargin (full circuit) = {dM_full:.2f}")
    results["full_ablated"] = full_mean
    results["dMargin_full"] = dM_full

    # -------------------------------------------------------
    # Phase 3: Single-neuron ablation (all positions)
    # -------------------------------------------------------
    print(f"\n--- Phase 3: Single-neuron ablation ({len(bottleneck_neurons)} neurons) ---")
    print(f"  {'':6s} {'Neuron':14s} {'P':>8s} {'logit':>8s} {'margin':>8s} {'dMargin':>8s} {'%full':>7s}")

    single_neuron_results = []
    for i, (layer, neuron) in enumerate(bottleneck_neurons):
        ctx_factory = make_single_neuron_ctx(steerer.model, layer, neuron)
        abl_mean, _ = _measure_task(make_ctx=ctx_factory)
        dM = baseline_mean[M] - abl_mean[M]
        frac = dM / dM_full if abs(dM_full) > 1e-6 else 0

        single_neuron_results.append({
            "layer": layer, "neuron": neuron,
            **{f"abl_{k}": v for k, v in abl_mean.items()},
            "dMargin": dM, "fraction_of_full": frac,
        })
        print(f"  [{i+1:2d}] L{layer:02d}/N{neuron:5d}  "
              f"P={abl_mean['prob']:.4f}  logit={abl_mean['logit']:+.2f}  "
              f"margin={abl_mean['logit_margin']:+.2f}  dM={dM:+.2f}  ({frac:5.1%})")

    results["single_neuron_ablation"] = single_neuron_results

    # -------------------------------------------------------
    # Phase 3b: Single-neuron last-token-only ablation (top 5)
    # -------------------------------------------------------
    print(f"\n--- Phase 3b: Single-neuron LAST-TOKEN ablation (top 5) ---")

    last_token_results = []
    for i, (layer, neuron) in enumerate(bottleneck_neurons[:5]):
        # Per-prompt (different seq lengths)
        per_prompt = []
        if task == "factual":
            for prompt, target in config["test_prompts"]:
                formatted = steerer._format_prompt(prompt, seed)
                seq_len = steerer.tokenizer(formatted, return_tensors="pt").input_ids.shape[1]
                ctx_f = make_single_neuron_lastpos_ctx(steerer.model, layer, neuron, seq_len)
                per_prompt.append(measure_R(steerer, prompt, target, make_ctx=ctx_f,
                                            seed_response=seed, use_chat_template=uct))
        else:
            for p in config["test_prompts"]:
                formatted = steerer._format_prompt(p, seed)
                seq_len = steerer.tokenizer(formatted, return_tensors="pt").input_ids.shape[1]
                ctx_f = make_single_neuron_lastpos_ctx(steerer.model, layer, neuron, seq_len)
                per_prompt.append(measure_R(steerer, p, target_token, make_ctx=ctx_f,
                                            seed_response=seed, use_chat_template=uct))
        abl_mean = {k: float(np.mean([r[k] for r in per_prompt])) for k in per_prompt[0]}
        dM = baseline_mean[M] - abl_mean[M]
        frac = dM / dM_full if abs(dM_full) > 1e-6 else 0
        last_token_results.append({
            "layer": layer, "neuron": neuron,
            **{f"abl_{k}": v for k, v in abl_mean.items()},
            "dMargin": dM, "fraction_of_full": frac,
        })
        print(f"  [{i+1:2d}] L{layer:02d}/N{neuron:5d}  margin={abl_mean['logit_margin']:+.2f}  dM={dM:+.2f}  ({frac:5.1%})")

    results["single_neuron_last_token"] = last_token_results

    # -------------------------------------------------------
    # Phase 4: Small-set ablation
    # -------------------------------------------------------
    print("\n--- Phase 4: Small-set ablation ---")

    set_results = []
    for k in [2, 3, 5, 10]:
        if k > len(bottleneck_neurons):
            continue
        neuron_set = bottleneck_neurons[:k]
        ctx_factory = make_multi_neuron_ctx(steerer.model, neuron_set)
        abl_mean, _ = _measure_task(make_ctx=ctx_factory)
        dM = baseline_mean[M] - abl_mean[M]
        frac = dM / dM_full if abs(dM_full) > 1e-6 else 0
        set_results.append({
            "k": k,
            "neurons": [{"layer": l, "neuron": n} for l, n in neuron_set],
            **{f"abl_{kk}": v for kk, v in abl_mean.items()},
            "dMargin": dM, "fraction_of_full": frac,
        })
        print(f"  Top-{k:2d}: margin={abl_mean['logit_margin']:+.2f}  dM={dM:+.2f}  ({frac:5.1%})")

    results["small_set_ablation"] = set_results

    # -------------------------------------------------------
    # Phase 5: Random single-neuron controls
    # -------------------------------------------------------
    print(f"\n--- Phase 5: Random single-neuron controls (n={n_random}) ---")

    d_mlp = steerer.model.config.intermediate_size
    bottleneck_layers = sorted(set(l for l, n in bottleneck_neurons))
    bottleneck_set = set(bottleneck_neurons)
    random_dM = []
    import random as rng
    rng.seed(42)

    for i in range(n_random):
        rand_layer = rng.choice(bottleneck_layers)
        rand_neuron = rng.randint(0, d_mlp - 1)
        if (rand_layer, rand_neuron) in bottleneck_set:
            rand_neuron = (rand_neuron + 1) % d_mlp
        ctx_factory = make_single_neuron_ctx(steerer.model, rand_layer, rand_neuron)
        abl_mean, _ = _measure_task(make_ctx=ctx_factory)
        random_dM.append(baseline_mean[M] - abl_mean[M])

    rand_mean = float(np.mean(random_dM))
    rand_std = float(np.std(random_dM))
    print(f"  Random single-neuron dMargin: mean={rand_mean:.4f} std={rand_std:.4f}")

    # Compute effect sizes
    print("\n--- Effect sizes (sigma above random) ---")
    for snr in single_neuron_results:
        if rand_std > 1e-10:
            sigma = (snr["dMargin"] - rand_mean) / rand_std
        else:
            sigma = float('inf') if snr["dMargin"] > rand_mean else 0.0
        snr["sigma_above_random"] = sigma
        print(f"  L{snr['layer']:02d}/N{snr['neuron']:5d}: "
              f"dMargin={snr['dMargin']:+.4f}  ({sigma:+.1f}sigma)")

    results["random_controls"] = {
        "n": n_random,
        "dMargin_values": random_dM,
        "mean": rand_mean,
        "std": rand_std,
    }

    # -------------------------------------------------------
    # Phase 6: Specificity — ablate on control prompts
    # -------------------------------------------------------
    if config["control_prompts"] and task != "factual":
        print("\n--- Phase 6: Specificity on benign prompts ---")

        ctrl_base, _ = measure_R_batch(
            steerer, config["control_prompts"], target_token,
            seed_response=seed, use_chat_template=uct,
        )
        print(f"  Benign baseline: margin={ctrl_base['logit_margin']:.2f}")

        ctrl_full, _ = measure_R_batch(
            steerer, config["control_prompts"], target_token,
            make_ctx=circuit_ctx, seed_response=seed, use_chat_template=uct,
        )
        dM_ctrl = ctrl_base[M] - ctrl_full[M]
        print(f"  Full circuit on benign: margin={ctrl_full['logit_margin']:.2f} dM={dM_ctrl:+.2f}")

        specificity = []
        for i, (layer, neuron) in enumerate(bottleneck_neurons[:5]):
            ctx_factory = make_single_neuron_ctx(steerer.model, layer, neuron)
            ctrl_abl, _ = measure_R_batch(
                steerer, config["control_prompts"], target_token,
                make_ctx=ctx_factory, seed_response=seed, use_chat_template=uct,
            )
            dM_b = ctrl_base[M] - ctrl_abl[M]
            specificity.append({
                "layer": layer, "neuron": neuron,
                **{f"ctrl_{kk}": v for kk, v in ctrl_abl.items()},
                "dMargin_benign": dM_b,
            })
            print(f"  L{layer:02d}/N{neuron:5d} on benign: margin={ctrl_abl['logit_margin']:.2f} dM={dM_b:+.2f}")

        results["specificity"] = {
            "control_baseline": ctrl_base,
            "control_full_ablated": ctrl_full,
            "dMargin_full_benign": dM_ctrl,
            "single_neuron": specificity,
        }

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY — {task.upper()} (logit_margin space)")
    print(f"{'='*70}")
    print(f"  Baseline margin       = {baseline_mean[M]:.2f}")
    print(f"  Full circuit dMargin  = {dM_full:.2f} (k={len(circuit.neurons)})")
    print(f"  Random single-neuron  = {rand_mean:.4f} +/- {rand_std:.4f}")
    print()

    sorted_singles = sorted(single_neuron_results, key=lambda x: x["dMargin"], reverse=True)
    print("  Top 5 single-neuron (by dMargin):")
    for i, snr in enumerate(sorted_singles[:5]):
        sigma = snr.get("sigma_above_random", 0)
        print(f"    {i+1}. L{snr['layer']:02d}/N{snr['neuron']:5d}: "
              f"dM={snr['dMargin']:+.4f} ({snr['fraction_of_full']:5.1%} of full, {sigma:+.1f}sigma)")

    print()
    print("  Small-set ablation:")
    for sr in set_results:
        print(f"    Top-{sr['k']:2d}: dM={sr['dMargin']:+.4f} ({sr['fraction_of_full']:5.1%} of full)")

    print()
    if sorted_singles and sorted_singles[0]["fraction_of_full"] > 0.1:
        top = sorted_singles[0]
        print(f"  >> L{top['layer']:02d}/N{top['neuron']:5d} carries "
              f"{top['fraction_of_full']:.0%} of full circuit effect in logit space")
    elif set_results and any(sr["fraction_of_full"] > 0.1 for sr in set_results):
        sr = next(s for s in set_results if s["fraction_of_full"] > 0.1)
        print(f"  >> Top-{sr['k']} carry {sr['fraction_of_full']:.0%} — distributed but concentrated")
    else:
        print(f"  >> Circuit remains fault-tolerant even in logit space")

    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"surgical_{task}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path / f'surgical_{task}.json'}")

    return results


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Surgical single-neuron causal ablation")
    parser.add_argument("--model", default="llama8b", choices=["llama8b"])
    parser.add_argument("--task", default="all", choices=["behavioral", "factual", "sycophancy", "fc_refusal", "fc_belief", "all"])
    parser.add_argument("--topology_dir", type=str, required=True,
                        help="Path to topology results directory (e.g. experiments/topology_llama8b_...)")
    parser.add_argument("--n_random", type=int, default=20,
                        help="Number of random single-neuron controls")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-named)")
    parser.add_argument("--filter_universal", action="store_true",
                        help="Auto-detect and filter universal neurons across circuits")
    parser.add_argument("--candidate_source", default="bottlenecks",
                        choices=["bottlenecks", "circuit"],
                        help="Which unique (layer, neuron) list to test: topology bottlenecks (default) or the full circuit")
    args = parser.parse_args()

    # Model setup
    model_name = {
        "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    }[args.model]

    print(f"Loading model: {model_name}")
    steerer = NeuronSteerer(model_name)
    print(f"Model loaded on {steerer.device}")

    # Output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/surgical_{args.model}_{timestamp}"

    # Run tasks
    tasks = ["behavioral", "factual", "sycophancy", "fc_refusal", "fc_belief"] if args.task == "all" else [args.task]

    # Auto-detect universal neurons if requested
    universal_set = None
    if args.filter_universal and len(tasks) >= 2:
        from neuron_steer.core import Circuit
        circuits = []
        for task in tasks:
            config = TASK_CONFIGS[task]
            topo_dir = resolve_topology_dir(args.topology_dir, task, config)
            circ_path = topo_dir / "circuit.json"
            if circ_path.exists():
                circuits.append(Circuit.load(str(circ_path)))
        if len(circuits) >= 2:
            universal_set = Circuit.find_universal_neurons(*circuits)
            if universal_set:
                print(f"\nFiltering {len(universal_set)} universal neurons:")
                for l, n in sorted(universal_set):
                    print(f"  L{l:02d}/N{n:5d}")

    all_results = {}

    for task in tasks:
        result = run_surgical_ablation(
            steerer, task, args.topology_dir,
            n_random=args.n_random,
            output_dir=args.output_dir,
            filter_universal=universal_set,
            candidate_source=args.candidate_source,
        )
        if result:
            all_results[task] = result

    # Save combined summary
    out_path = Path(args.output_dir)
    with open(out_path / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
