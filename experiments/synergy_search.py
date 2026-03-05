"""Greedy synergy search: pair/triple superadditive effects.

Tests whether small sets of neurons combine synergistically when ablated.
If pairs stay small, redundancy claim is bulletproof.
If pairs crack it, we've found distributed control sets.

Protocol:
  1. Re-measure single-neuron effects (verify consistency)
  2. Exhaustive pair search on all bottleneck neurons
  3. Greedy triple search from top-5 pairs
  4. Progressive ablation (cumulative by single-neuron causal rank)
  5. Random pair controls (null distribution)
  6. Superadditivity analysis (pair_dM / (single_i_dM + single_j_dM))
"""

import argparse
import json
import random as rng
import sys
import time
from itertools import combinations
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
    make_multi_neuron_ctx,
    make_circuit_ctx,
    load_bottleneck_candidates,
    unique_bottleneck_neurons,
    load_circuit,
    resolve_topology_dir,
    TASK_CONFIGS,
)


def run_synergy_search(
    steerer: NeuronSteerer,
    task: str,
    topology_base: str,
    max_neurons: int = 24,
    n_random_pairs: int = 50,
    output_dir: str = None,
):
    """Run greedy synergy search for a task."""
    config = TASK_CONFIGS[task]
    topo_dir = resolve_topology_dir(topology_base, task, config)

    print(f"\n{'='*70}")
    print(f"SYNERGY SEARCH: {task.upper()}")
    print(f"{'='*70}")

    # Load topology data
    analysis_path = topo_dir / "analysis.json"
    circuit_path = topo_dir / "circuit.json"

    if not analysis_path.exists():
        print(f"  ERROR: {analysis_path} not found, skipping")
        return None

    candidates = load_bottleneck_candidates(str(analysis_path))
    all_bottleneck_neurons = unique_bottleneck_neurons(candidates)
    circuit = load_circuit(str(circuit_path))

    # Cap neuron count for tractability
    bottleneck_neurons = all_bottleneck_neurons[:max_neurons]

    print(f"  Circuit size: {len(circuit.neurons)} neurons")
    print(f"  Unique bottleneck neurons: {len(all_bottleneck_neurons)} (using top {len(bottleneck_neurons)})")

    uct = config["use_chat_template"]
    seed = config["seed_response"]
    M = "logit_margin"

    if task == "factual":
        target_token = None
    else:
        target_token = config["target_token"]

    def _measure_task(make_ctx=None):
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

    results = {
        "task": task,
        "n_bottleneck_neurons": len(bottleneck_neurons),
        "max_neurons": max_neurons,
    }

    # -------------------------------------------------------
    # Phase 0: Baseline + full circuit reference
    # -------------------------------------------------------
    print("\n--- Phase 0: Baseline & full circuit reference ---")
    baseline_mean, _ = _measure_task()
    circuit_ctx = make_circuit_ctx(steerer.model, circuit)
    full_mean, _ = _measure_task(make_ctx=circuit_ctx)
    dM_full = baseline_mean[M] - full_mean[M]

    print(f"  Baseline margin: {baseline_mean[M]:.2f}")
    print(f"  Full circuit dMargin: {dM_full:.2f}")
    results["baseline"] = baseline_mean
    results["full_ablated"] = full_mean
    results["dMargin_full"] = dM_full

    # -------------------------------------------------------
    # Phase 1: Single-neuron re-measurement
    # -------------------------------------------------------
    print(f"\n--- Phase 1: Single-neuron effects ({len(bottleneck_neurons)} neurons) ---")
    single_dM = {}
    for i, (layer, neuron) in enumerate(bottleneck_neurons):
        ctx_f = make_single_neuron_ctx(steerer.model, layer, neuron)
        abl_mean, _ = _measure_task(make_ctx=ctx_f)
        dM = baseline_mean[M] - abl_mean[M]
        single_dM[(layer, neuron)] = dM
        frac = dM / dM_full if abs(dM_full) > 1e-6 else 0
        print(f"  [{i+1:2d}] L{layer:02d}/N{neuron:5d}: dM={dM:+.4f} ({frac:5.1%})")

    results["single_neuron_dM"] = [
        {"layer": l, "neuron": n, "dMargin": dM}
        for (l, n), dM in single_dM.items()
    ]

    # -------------------------------------------------------
    # Phase 2: Exhaustive pair search
    # -------------------------------------------------------
    n_pairs = len(bottleneck_neurons) * (len(bottleneck_neurons) - 1) // 2
    print(f"\n--- Phase 2: Exhaustive pair search ({n_pairs} pairs) ---")

    pair_results = []
    for idx, ((l1, n1), (l2, n2)) in enumerate(combinations(bottleneck_neurons, 2)):
        ctx_f = make_multi_neuron_ctx(steerer.model, [(l1, n1), (l2, n2)])
        abl_mean, _ = _measure_task(make_ctx=ctx_f)
        pair_dM = baseline_mean[M] - abl_mean[M]

        sum_singles = single_dM[(l1, n1)] + single_dM[(l2, n2)]
        if abs(sum_singles) > 1e-6:
            synergy = pair_dM / sum_singles
        else:
            synergy = float('inf') if pair_dM > 0.01 else 1.0

        pair_results.append({
            "neurons": [(l1, n1), (l2, n2)],
            "dMargin": pair_dM,
            "sum_singles": sum_singles,
            "synergy_ratio": synergy,
            "fraction_of_full": pair_dM / dM_full if abs(dM_full) > 1e-6 else 0,
        })

        if (idx + 1) % 50 == 0 or idx + 1 == n_pairs:
            print(f"  Completed {idx+1}/{n_pairs} pairs...")

    pair_results.sort(key=lambda x: x["dMargin"], reverse=True)

    print(f"\n  Top 10 pairs by dMargin:")
    print(f"  {'Pair':30s} {'dM':>8s} {'sum1':>8s} {'syn':>6s} {'%full':>7s}")
    for pr in pair_results[:10]:
        (l1, n1), (l2, n2) = pr["neurons"]
        print(f"  L{l1:02d}/N{n1:5d} + L{l2:02d}/N{n2:5d}  "
              f"dM={pr['dMargin']:+.4f}  sum={pr['sum_singles']:+.4f}  "
              f"{pr['synergy_ratio']:5.2f}  ({pr['fraction_of_full']:5.1%})")

    print(f"\n  Top 5 most synergistic (ratio > 1):")
    syn_sorted = sorted([p for p in pair_results if p["dMargin"] > 0.01],
                        key=lambda x: x["synergy_ratio"], reverse=True)
    for pr in syn_sorted[:5]:
        (l1, n1), (l2, n2) = pr["neurons"]
        print(f"  L{l1:02d}/N{n1:5d} + L{l2:02d}/N{n2:5d}  "
              f"dM={pr['dMargin']:+.4f}  syn={pr['synergy_ratio']:.2f}")

    results["pair_search"] = [
        {**pr, "neurons": [{"layer": l, "neuron": n} for l, n in pr["neurons"]]}
        for pr in pair_results
    ]

    # -------------------------------------------------------
    # Phase 3: Greedy triple search from top-5 pairs
    # -------------------------------------------------------
    print(f"\n--- Phase 3: Greedy triple search (from top-5 pairs) ---")

    triple_results = []
    for pair_idx, top_pair in enumerate(pair_results[:5]):
        (l1, n1), (l2, n2) = top_pair["neurons"]
        pair_dM = top_pair["dMargin"]
        best_triple = None
        best_triple_dM = -float('inf')

        for l3, n3 in bottleneck_neurons:
            if (l3, n3) == (l1, n1) or (l3, n3) == (l2, n2):
                continue
            ctx_f = make_multi_neuron_ctx(steerer.model, [(l1, n1), (l2, n2), (l3, n3)])
            abl_mean, _ = _measure_task(make_ctx=ctx_f)
            triple_dM = baseline_mean[M] - abl_mean[M]

            if triple_dM > best_triple_dM:
                best_triple_dM = triple_dM
                best_triple = (l3, n3)

        if best_triple:
            l3, n3 = best_triple
            sum_singles = single_dM[(l1, n1)] + single_dM[(l2, n2)] + single_dM[(l3, n3)]
            synergy = best_triple_dM / sum_singles if abs(sum_singles) > 1e-6 else float('inf')
            frac = best_triple_dM / dM_full if abs(dM_full) > 1e-6 else 0

            triple_results.append({
                "pair": [(l1, n1), (l2, n2)],
                "third": (l3, n3),
                "triple_dMargin": best_triple_dM,
                "pair_dMargin": pair_dM,
                "sum_singles": sum_singles,
                "synergy_ratio": synergy,
                "fraction_of_full": frac,
            })
            print(f"  Pair #{pair_idx+1}: L{l1:02d}/N{n1:5d}+L{l2:02d}/N{n2:5d} "
                  f"(dM={pair_dM:+.4f}) + best L{l3:02d}/N{n3:5d} "
                  f"→ triple dM={best_triple_dM:+.4f} (syn={synergy:.2f}, {frac:.1%})")

    results["triple_search"] = [
        {**tr,
         "pair": [{"layer": l, "neuron": n} for l, n in tr["pair"]],
         "third": {"layer": tr["third"][0], "neuron": tr["third"][1]}}
        for tr in triple_results
    ]

    # -------------------------------------------------------
    # Phase 4: Progressive ablation (by causal rank)
    # -------------------------------------------------------
    print(f"\n--- Phase 4: Progressive ablation (cumulative, by causal rank) ---")

    sorted_neurons = sorted(single_dM.items(), key=lambda x: x[1], reverse=True)

    progressive_results = []
    print(f"  {'k':>3s} {'dMargin':>8s} {'%full':>7s} {'marginal':>9s} {'added':20s}")
    for k in range(1, len(sorted_neurons) + 1):
        neuron_set = [ln for ln, _ in sorted_neurons[:k]]
        ctx_f = make_multi_neuron_ctx(steerer.model, neuron_set)
        abl_mean, _ = _measure_task(make_ctx=ctx_f)
        cum_dM = baseline_mean[M] - abl_mean[M]
        frac = cum_dM / dM_full if abs(dM_full) > 1e-6 else 0

        prev_dM = progressive_results[-1]["cumulative_dMargin"] if progressive_results else 0
        marginal = cum_dM - prev_dM
        l, n = sorted_neurons[k - 1][0]

        progressive_results.append({
            "k": k,
            "added_neuron": {"layer": l, "neuron": n},
            "cumulative_dMargin": cum_dM,
            "fraction_of_full": frac,
            "marginal_dMargin": marginal,
        })
        print(f"  {k:3d}  dM={cum_dM:+.4f}  ({frac:5.1%})  Δ={marginal:+.4f}  +L{l:02d}/N{n:5d}")

    results["progressive_ablation"] = progressive_results

    # -------------------------------------------------------
    # Phase 5: Random pair controls
    # -------------------------------------------------------
    print(f"\n--- Phase 5: Random pair controls (n={n_random_pairs}) ---")

    d_mlp = steerer.model.config.intermediate_size
    bottleneck_layers = sorted(set(l for l, n in bottleneck_neurons))
    rng.seed(42)
    random_pair_dMs = []

    for i in range(n_random_pairs):
        rl1 = rng.choice(bottleneck_layers)
        rl2 = rng.choice(bottleneck_layers)
        rn1 = rng.randint(0, d_mlp - 1)
        rn2 = rng.randint(0, d_mlp - 1)
        ctx_f = make_multi_neuron_ctx(steerer.model, [(rl1, rn1), (rl2, rn2)])
        abl_mean, _ = _measure_task(make_ctx=ctx_f)
        random_pair_dMs.append(baseline_mean[M] - abl_mean[M])

    rand_mean = float(np.mean(random_pair_dMs))
    rand_std = float(np.std(random_pair_dMs))
    print(f"  Random pair dMargin: mean={rand_mean:.4f} std={rand_std:.4f}")

    results["random_pair_controls"] = {
        "n": n_random_pairs,
        "mean": rand_mean,
        "std": rand_std,
        "values": random_pair_dMs,
    }

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY — {task.upper()} SYNERGY SEARCH")
    print(f"{'='*70}")
    print(f"  Full circuit dMargin: {dM_full:.2f}")

    if sorted_neurons:
        best_single = sorted_neurons[0]
        (bl, bn), bv = best_single
        print(f"  Best single: L{bl:02d}/N{bn:5d} dM={bv:+.4f} ({bv/dM_full:.1%})")

    if pair_results:
        bp = pair_results[0]
        (l1, n1), (l2, n2) = bp["neurons"]
        print(f"  Best pair: L{l1:02d}/N{n1:5d}+L{l2:02d}/N{n2:5d} "
              f"dM={bp['dMargin']:+.4f} ({bp['fraction_of_full']:.1%}), "
              f"synergy={bp['synergy_ratio']:.2f}")

    if triple_results:
        bt = max(triple_results, key=lambda x: x["triple_dMargin"])
        print(f"  Best triple: dM={bt['triple_dMargin']:+.4f} ({bt['fraction_of_full']:.1%}), "
              f"synergy={bt['synergy_ratio']:.2f}")

    # Compute k for 25% and 50% thresholds
    if progressive_results:
        for threshold in [0.25, 0.50]:
            for pr in progressive_results:
                if pr["fraction_of_full"] >= threshold:
                    print(f"  {threshold:.0%} of full circuit at: k={pr['k']} neurons")
                    break
            else:
                last = progressive_results[-1]
                print(f"  {threshold:.0%} not reached: max k={last['k']} → {last['fraction_of_full']:.1%}")

    # Synergy distribution
    if pair_results:
        # Filter to pairs where both singles have measurable effects
        meaningful = [p for p in pair_results
                      if abs(p["sum_singles"]) > 0.01 and p["dMargin"] > 0.01]
        if meaningful:
            synergistic = sum(1 for p in meaningful if p["synergy_ratio"] > 1.2)
            additive = sum(1 for p in meaningful if 0.8 <= p["synergy_ratio"] <= 1.2)
            subadditive = sum(1 for p in meaningful if p["synergy_ratio"] < 0.8)
            print(f"\n  Synergy distribution ({len(meaningful)} meaningful pairs):")
            print(f"    Synergistic (>1.2x): {synergistic}")
            print(f"    Additive (0.8-1.2x): {additive}")
            print(f"    Subadditive (<0.8x):  {subadditive}")

        # Effect sizes for pairs vs random
        if rand_std > 1e-10:
            for pr in pair_results[:5]:
                (l1, n1), (l2, n2) = pr["neurons"]
                sigma = (pr["dMargin"] - rand_mean) / rand_std
                print(f"    L{l1:02d}/N{n1:5d}+L{l2:02d}/N{n2:5d}: {sigma:.1f}σ above random pairs")

    # Verdict
    print()
    if pair_results and pair_results[0]["fraction_of_full"] > 0.15:
        bp = pair_results[0]
        (l1, n1), (l2, n2) = bp["neurons"]
        if bp["synergy_ratio"] > 1.5:
            print(f"  >> SYNERGISTIC: pair L{l1:02d}/N{n1:5d}+L{l2:02d}/N{n2:5d} "
                  f"captures {bp['fraction_of_full']:.0%} with {bp['synergy_ratio']:.1f}x synergy")
        else:
            print(f"  >> CONCENTRATED pair found at {bp['fraction_of_full']:.0%} but additive, not synergistic")
    elif progressive_results:
        last = progressive_results[-1]
        if last["fraction_of_full"] < 0.25:
            print(f"  >> FAULT-TOLERANT: even all {last['k']} bottleneck neurons "
                  f"capture only {last['fraction_of_full']:.0%}")
        else:
            # Find k for 25%
            for pr in progressive_results:
                if pr["fraction_of_full"] >= 0.25:
                    print(f"  >> DISTRIBUTED: need {pr['k']} neurons for 25% effect")
                    break
    else:
        print(f"  >> No clear pattern")

    # Save
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"synergy_{task}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path / f'synergy_{task}.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Greedy synergy search")
    parser.add_argument("--model", default="llama8b", choices=["llama8b"])
    parser.add_argument("--task", default="all", choices=["behavioral", "factual", "sycophancy", "fc_refusal", "fc_belief", "all"])
    parser.add_argument("--topology_dir", type=str, required=True)
    parser.add_argument("--max_neurons", type=int, default=24,
                        help="Max bottleneck neurons to test (for tractability)")
    parser.add_argument("--n_random_pairs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    model_name = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}[args.model]
    print(f"Loading model: {model_name}")
    steerer = NeuronSteerer(model_name)
    print(f"Model loaded on {steerer.device}")

    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/synergy_{args.model}_{timestamp}"

    tasks = ["behavioral", "factual", "sycophancy", "fc_refusal", "fc_belief"] if args.task == "all" else [args.task]
    all_results = {}

    for task in tasks:
        result = run_synergy_search(
            steerer, task, args.topology_dir,
            max_neurons=args.max_neurons,
            n_random_pairs=args.n_random_pairs,
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
