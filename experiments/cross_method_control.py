"""Cross-Method Control: Contrastive discovery on factual tasks.

The Paleas et al. paper claims:
  - Factual circuits (RelP) → broadly distributed across layers
  - Behavioral circuits (contrastive) → concentrated in late layers

This experiment tests whether late-layer concentration is a property of
BEHAVIORAL tasks or an artifact of the CONTRASTIVE METHOD.

Design:
  1. RelP on capitals (standard — should be broadly distributed)
  2. Contrastive on capitals (the critical test)
  3. Contrastive on refusal (baseline — should be late-concentrated)
  4. Compare layer distributions across all three

If contrastive-on-capitals also concentrates late → the paper's
layer-localization claim is a methodological confound.
"""

import argparse
import sys
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron_steer.core import NeuronSteerer, NeuronIdx
from experiments.prompts import (
    CAPITALS_DISCOVERY,
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    BENIGN_PROMPTS,
)


def layer_distribution(circuit, n_layers):
    """Compute per-layer neuron counts and stats."""
    layer_counts = defaultdict(int)
    layer_weights = defaultdict(float)
    for nidx, attr in circuit.neurons.items():
        layer_counts[nidx.layer] += 1
        layer_weights[nidx.layer] += abs(attr)

    layers = sorted(layer_counts.keys())
    counts = [layer_counts[l] for l in layers]

    # Weighted mean layer
    total_weight = sum(layer_weights.values())
    if total_weight > 0:
        mean_layer = sum(l * w for l, w in layer_weights.items()) / total_weight
    else:
        mean_layer = sum(l * c for l, c in zip(layers, counts)) / max(sum(counts), 1)

    # Fraction in final 20% of layers
    cutoff = int(n_layers * 0.8)
    late_count = sum(c for l, c in layer_counts.items() if l >= cutoff)
    late_frac = late_count / max(sum(counts), 1)

    return {
        "layer_counts": dict(layer_counts),
        "mean_layer": mean_layer,
        "late_fraction": late_frac,
        "total_neurons": sum(counts),
        "n_layers_active": len(layers),
    }


def jaccard(circuit_a, circuit_b):
    """Jaccard similarity between two circuits (layer, neuron) basis."""
    set_a = {(n.layer, n.neuron) for n in circuit_a.neurons}
    set_b = {(n.layer, n.neuron) for n in circuit_b.neurons}
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / max(len(union), 1), len(inter)


def print_layer_histogram(dist, label, n_layers):
    """ASCII histogram of layer distribution."""
    counts = dist["layer_counts"]
    max_count = max(counts.values()) if counts else 1
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  mean_layer={dist['mean_layer']:.1f}  late_frac={dist['late_fraction']:.2f}  "
          f"total={dist['total_neurons']}")
    print(f"{'='*60}")
    for l in range(n_layers):
        c = counts.get(l, 0)
        bar = "█" * int(40 * c / max_count) if c > 0 else ""
        if c > 0:
            print(f"  L{l:02d} │{bar} {c}")


def main():
    p = argparse.ArgumentParser(description="Cross-method control experiment")
    p.add_argument("--model", default="llama",
                   choices=["llama", "llama8b"],
                   help="Model to use (llama=1B, llama8b=8B)")
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--skip_relp", action="store_true",
                   help="Skip RelP discovery (faster)")
    p.add_argument("--skip_steer", action="store_true",
                   help="Skip steering validation")
    p.add_argument("--n_capitals", type=int, default=None,
                   help="Limit number of capital prompts (for speed)")
    args = p.parse_args()

    model_name = {
        "llama": "meta-llama/Llama-3.2-1B-Instruct",
        "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    }[args.model]

    print(f"Loading {model_name}...")
    steerer = NeuronSteerer(model_name)
    n_layers = steerer.model.config.num_hidden_layers

    capitals = CAPITALS_DISCOVERY[:args.n_capitals] if args.n_capitals else CAPITALS_DISCOVERY

    # ================================================================
    # 1. Contrastive on CAPITALS (the critical test)
    # ================================================================
    # Split capitals into two groups for contrastive comparison.
    # Group A countries vs Group B countries — the contrastive diff
    # captures country-group-specific factual neurons.
    mid = len(capitals) // 2
    group_a = [p for p, _ in capitals[:mid]]
    group_b = [p for p, _ in capitals[mid:]]

    print(f"\n--- Contrastive on CAPITALS ---")
    print(f"  Group A ({len(group_a)}): {group_a[0]}, ...")
    print(f"  Group B ({len(group_b)}): {group_b[0]}, ...")

    contrastive_factual = steerer.discover_contrastive(
        positive_prompts=group_a,
        negative_prompts=group_b,
        top_k=args.top_k,
    )
    dist_cf = layer_distribution(contrastive_factual, n_layers)
    print_layer_histogram(dist_cf, "Contrastive — Factual (Capitals)", n_layers)

    # ================================================================
    # 2. Contrastive on REFUSAL (baseline)
    # ================================================================
    print(f"\n--- Contrastive on REFUSAL ---")
    contrastive_refusal = steerer.discover_contrastive(
        positive_prompts=REFUSAL_DISCOVERY_POSITIVE,
        negative_prompts=REFUSAL_DISCOVERY_NEGATIVE,
        top_k=args.top_k,
    )
    dist_cr = layer_distribution(contrastive_refusal, n_layers)
    print_layer_histogram(dist_cr, "Contrastive — Behavioral (Refusal)", n_layers)

    # ================================================================
    # 3. RelP on CAPITALS (standard approach)
    # ================================================================
    if not args.skip_relp:
        print(f"\n--- RelP on CAPITALS ---")
        prompts = [p for p, _ in capitals]
        targets = [t for _, t in capitals]

        relp_factual = steerer.discover_circuit_multi(
            prompts=prompts,
            target_tokens=targets,
            counterfactual_tokens=[None] * len(prompts),
            selection_method="topk",
            top_k=args.top_k,
        )
        dist_rf = layer_distribution(relp_factual, n_layers)
        print_layer_histogram(dist_rf, "RelP — Factual (Capitals)", n_layers)
    else:
        relp_factual = None
        dist_rf = None

    # ================================================================
    # 4. Comparison
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  LAYER DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'Mean Layer':>10} {'Late Frac':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Contrastive-Factual':<30} {dist_cf['mean_layer']:>10.1f} {dist_cf['late_fraction']:>10.2f}")
    print(f"  {'Contrastive-Refusal':<30} {dist_cr['mean_layer']:>10.1f} {dist_cr['late_fraction']:>10.2f}")
    if dist_rf:
        print(f"  {'RelP-Factual':<30} {dist_rf['mean_layer']:>10.1f} {dist_rf['late_fraction']:>10.2f}")

    # Overlap analysis
    print(f"\n  CIRCUIT OVERLAP (Jaccard)")
    print(f"  {'-'*50}")
    j_cf_cr, n_cf_cr = jaccard(contrastive_factual, contrastive_refusal)
    print(f"  Contrastive-Factual ↔ Contrastive-Refusal: J={j_cf_cr:.3f} ({n_cf_cr} shared)")
    if relp_factual:
        j_cf_rf, n_cf_rf = jaccard(contrastive_factual, relp_factual)
        j_cr_rf, n_cr_rf = jaccard(contrastive_refusal, relp_factual)
        print(f"  Contrastive-Factual ↔ RelP-Factual:       J={j_cf_rf:.3f} ({n_cf_rf} shared)")
        print(f"  Contrastive-Refusal ↔ RelP-Factual:       J={j_cr_rf:.3f} ({n_cr_rf} shared)")

    # ================================================================
    # 5. Steering validation (optional)
    # ================================================================
    if not args.skip_steer:
        print(f"\n{'='*60}")
        print(f"  STEERING VALIDATION")
        print(f"{'='*60}")

        test_prompts = [
            "What is the capital of France?",
            "What is the capital of Germany?",
        ]

        for label, circuit in [
            ("Contrastive-Factual", contrastive_factual),
            ("Contrastive-Refusal", contrastive_refusal),
        ]:
            print(f"\n  [{label}] ablation (α=0.0):")
            for prompt in test_prompts:
                out = steerer.steer_and_generate(
                    prompt, circuit, multiplier=0.0,
                    max_new_tokens=30, all_positions=True,
                )
                # Truncate for display
                out_short = out[:120].replace("\n", " ")
                print(f"    Q: {prompt}")
                print(f"    A: {out_short}")

        # Benign preservation
        print(f"\n  Benign preservation (α=0.0):")
        for label, circuit in [
            ("Contrastive-Factual", contrastive_factual),
            ("Contrastive-Refusal", contrastive_refusal),
        ]:
            print(f"\n  [{label}]:")
            for prompt in BENIGN_PROMPTS[:3]:
                out = steerer.steer_and_generate(
                    prompt, circuit, multiplier=0.0,
                    max_new_tokens=30, all_positions=True,
                )
                out_short = out[:120].replace("\n", " ")
                print(f"    Q: {prompt}")
                print(f"    A: {out_short}")


if __name__ == "__main__":
    main()
