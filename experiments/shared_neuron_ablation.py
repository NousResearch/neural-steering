#!/usr/bin/env python3
"""Ablate shared neurons across fc_refusal and fc_belief to test format-specificity.

Hypothesis: L21/N13111 and L15/N14179 appear in both fc_refusal and fc_belief
circuits but NOT in open-ended refusal. They may be "forced-choice format" neurons
rather than task-specific neurons.

Test: ablate each neuron individually and measure:
  - P("No") on fc_refusal prompts (harmful yes/no)
  - P("Yes") on fc_belief prompts (opinion yes/no)
  - P("Yes") on fc_benign prompts (benign yes/no) — specificity control
  - P("I") on open-ended refusal prompts — cross-format control

If the neuron is format-specific, it should affect fc_refusal AND fc_belief
but NOT open-ended refusal.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from neuron_steer.core import NeuronSteerer, NeuronIdx, steer_neurons
from prompts import (
    FC_REFUSAL_TEST, FC_BELIEF_TEST, FC_BENIGN,
    FC_BELIEF_NO_TEST, REFUSAL_TEST, BENIGN_PROMPTS,
)


def measure_p_target(steerer, prompts, target_token, use_chat=True, ablate_neurons=None):
    """Measure mean P(target_token) across prompts, optionally with neuron ablation."""
    tok_id = steerer.tokenizer.encode(target_token, add_special_tokens=False)[0]
    probs = []
    for p in prompts:
        if use_chat:
            inputs = steerer.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                return_tensors="pt", add_generation_prompt=True,
            )
            if not isinstance(inputs, torch.Tensor):
                inputs = inputs["input_ids"]
        else:
            inputs = steerer.tokenizer(p, return_tensors="pt")["input_ids"]
        inputs = inputs.to(steerer.model.device)

        if ablate_neurons:
            # Use steer_neurons context manager (handles intermediate MLP activations)
            neuron_dict = {}
            for layer, neuron in ablate_neurons:
                neuron_dict[NeuronIdx(layer=layer, position=-1, neuron=neuron)] = 0.0
            with steer_neurons(steerer.model, neuron_dict, multiplier=0.0, all_positions=True):
                with torch.no_grad():
                    logits = steerer.model(inputs).logits[0, -1]
        else:
            with torch.no_grad():
                logits = steerer.model(inputs).logits[0, -1]

        p_target = torch.softmax(logits, dim=-1)[tok_id].item()
        probs.append(p_target)
    return np.mean(probs), probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_random", type=int, default=20, help="Random neuron controls")
    args = parser.parse_args()

    steerer = NeuronSteerer(args.model)

    # Neurons to test
    shared_neurons = [
        (21, 13111, "shared: fc_refusal+fc_belief"),
        (15, 14179, "shared: fc_refusal+fc_belief"),
    ]
    refusal_specific = [
        (24, 14331, "fc_refusal only (dominant)"),
        (25, 891, "fc_refusal only"),
        (30, 14210, "fc_refusal only"),
    ]
    belief_specific = [
        (21, 2382, "fc_belief (high necessity)"),
        (19, 9196, "fc_belief (high necessity)"),
        (26, 11419, "fc_belief (high necessity)"),
    ]

    all_neurons = shared_neurons + refusal_specific + belief_specific

    # Prompt sets to evaluate
    eval_sets = [
        ("fc_refusal", FC_REFUSAL_TEST, "No", True),
        ("fc_belief", FC_BELIEF_TEST, "Yes", True),
        ("fc_benign", FC_BENIGN, "Yes", True),
        ("fc_belief_no", FC_BELIEF_NO_TEST, "No", True),
        ("open_refusal", REFUSAL_TEST, "I", True),
    ]

    # Baselines
    print("=" * 70)
    print("  BASELINES (no ablation)")
    print("=" * 70)
    baselines = {}
    for name, prompts, target, use_chat in eval_sets:
        mean_p, _ = measure_p_target(steerer, prompts, target, use_chat)
        baselines[name] = mean_p
        print(f"  {name:20s}  P({target})={mean_p:.4f}")

    # Ablate each neuron
    print("\n" + "=" * 70)
    print("  SINGLE-NEURON ABLATION")
    print("=" * 70)

    results = {}
    for layer, neuron, label in all_neurons:
        print(f"\n--- L{layer:02d}/N{neuron:5d} ({label}) ---")
        neuron_results = {}
        for name, prompts, target, use_chat in eval_sets:
            abl_p, _ = measure_p_target(
                steerer, prompts, target, use_chat,
                ablate_neurons=[(layer, neuron)],
            )
            delta = abl_p - baselines[name]
            neuron_results[name] = {"ablated": float(abl_p), "delta": float(delta)}
            print(f"  {name:20s}  P({target})={abl_p:.4f}  Δ={delta:+.4f}")
        results[f"L{layer:02d}/N{neuron}"] = {"label": label, "results": neuron_results}

    # Also test ablating BOTH shared neurons together
    print(f"\n--- BOTH shared (L21/N13111 + L15/N14179) ---")
    both_results = {}
    for name, prompts, target, use_chat in eval_sets:
        abl_p, _ = measure_p_target(
            steerer, prompts, target, use_chat,
            ablate_neurons=[(21, 13111), (15, 14179)],
        )
        delta = abl_p - baselines[name]
        both_results[name] = {"ablated": float(abl_p), "delta": float(delta)}
        print(f"  {name:20s}  P({target})={abl_p:.4f}  Δ={delta:+.4f}")
    results["BOTH_SHARED"] = {"label": "L21/N13111 + L15/N14179", "results": both_results}

    # Random controls
    print("\n" + "=" * 70)
    print(f"  RANDOM CONTROLS (n={args.n_random})")
    print("=" * 70)
    rng = np.random.default_rng(42)
    n_layers = len(steerer.model.model.layers)
    hidden = steerer.model.config.intermediate_size
    random_deltas = {name: [] for name, _, _, _ in eval_sets}

    for i in range(args.n_random):
        rl = int(rng.integers(0, n_layers))
        rn = int(rng.integers(0, hidden))
        for name, prompts, target, use_chat in eval_sets:
            abl_p, _ = measure_p_target(
                steerer, prompts, target, use_chat,
                ablate_neurons=[(rl, rn)],
            )
            random_deltas[name].append(abl_p - baselines[name])

    print(f"  {'Eval set':20s}  {'mean Δ':>10s}  {'std Δ':>10s}")
    random_stats = {}
    for name in random_deltas:
        deltas = random_deltas[name]
        m, s = float(np.mean(deltas)), float(np.std(deltas))
        random_stats[name] = {"mean": m, "std": s}
        print(f"  {name:20s}  {m:+.6f}  {s:.6f}")

    # Sigma scores
    print("\n" + "=" * 70)
    print("  EFFECT SIZES (sigma)")
    print("=" * 70)
    for neuron_key, data in results.items():
        print(f"\n{neuron_key} ({data['label']})")
        for name, nr in data["results"].items():
            s = random_stats[name]["std"]
            sigma = nr["delta"] / s if s > 0 else 0
            print(f"  {name:20s}  Δ={nr['delta']:+.4f}  ({sigma:+.1f}σ)")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"shared_neuron_ablation_{ts}")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "baselines": baselines,
            "neurons": results,
            "random_stats": random_stats,
        }, f, indent=2, default=float)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
