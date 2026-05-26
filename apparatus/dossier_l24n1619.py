"""L24/N1619 dossier.

The substrate-writer / tokenwise-suppressor dissociation identified in
Apparatus 2a is the single most surprising neuron-level finding so far.
This script collects evidence about what L24/N1619 actually does.

Three measurements:
1. Per-prompt activation across the standard refusal prompt sets, alongside
   L24/N2598 (canonical writer), L26/N11984 (consistent suppressor),
   L20/N9928 (writer-only).
2. Generation rollouts with L24/N1619 ablated (m=0) on harmful prompts —
   does refusal still happen? In what surface form?
3. Generation rollouts with L24/N1619 amplified (m=2) on benign prompts —
   if it writes substrate, amplifying should push toward refusal-shaped output.

Output: JSON file with per-prompt activations + rollouts for inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuron_steer.core import NeuronSteerer, NeuronIdx, steer_neurons
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST,
    BENIGN_PROMPTS,
)
from experiments.sufficiency_test import collect_single_neuron_activation


# Neurons of interest from Apparatus 1 + 2a findings
COMPARISON_NEURONS = [
    ("L24/N1619",  24, 1619, "substrate-writer / token-suppressor (target of investigation)"),
    ("L24/N2598",  24, 2598, "canonical token-writer + reader (tok_suff=+1.79, tok_nec=+9σ)"),
    ("L22/N3319",  22, 3319, "writer-only by tokenwise"),
    ("L20/N9928",  20, 9928, "writer-only by tokenwise"),
    ("L26/N11984", 26, 11984, "suppressor-consistent (both apparatuses, downstream of L24)"),
    ("L18/N8429",  18, 8429, "reader-only, substantial probe-suff"),
    ("L15/N9635",  15, 9635, "diagonal substrate+token writer (max probe-suff)"),
]


PROMPT_SETS = {
    "refusal_discovery_pos": REFUSAL_DISCOVERY_POSITIVE,
    "refusal_discovery_neg": REFUSAL_DISCOVERY_NEGATIVE,
    "refusal_test":          REFUSAL_TEST,
    "benign":                BENIGN_PROMPTS,
}


def format_input(steerer: NeuronSteerer, prompt: str) -> torch.Tensor:
    formatted = steerer._format_prompt(prompt, "")
    return steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)


def collect_activations(steerer: NeuronSteerer) -> dict:
    """Per-(neuron, prompt) last-token activation."""
    out = {}
    for label, layer, neuron, _ in COMPARISON_NEURONS:
        per_set = {}
        for set_name, prompts in PROMPT_SETS.items():
            vals = []
            for p in prompts:
                input_ids = format_input(steerer, p)
                v = collect_single_neuron_activation(steerer.model, layer, neuron, input_ids)
                vals.append({"prompt": p, "activation": v})
            arr = np.array([x["activation"] for x in vals])
            per_set[set_name] = {
                "n": len(vals),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "per_prompt": vals,
            }
        out[label] = {"layer": layer, "neuron": neuron, "by_set": per_set}
    return out


def generate(steerer: NeuronSteerer, prompt: str, max_new_tokens: int = 60,
             feature: dict | None = None, multiplier: float = 1.0,
             seed: int = 0) -> str:
    """Greedy generation with optional feature steering (feature is a
    {NeuronIdx -> float} dict consumed by steer_neurons)."""
    formatted = steerer._format_prompt(prompt, "")
    input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)

    torch.manual_seed(seed)
    if feature is not None and multiplier != 1.0:
        with steer_neurons(steerer.model, feature, multiplier):
            with torch.no_grad():
                out = steerer.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=1.0,
                    pad_token_id=steerer.tokenizer.eos_token_id,
                )
    else:
        with torch.no_grad():
            out = steerer.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=steerer.tokenizer.eos_token_id,
            )
    return steerer.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def single_neuron_set(layer: int, neuron: int) -> dict:
    """Return a {NeuronIdx -> float} mapping suitable for steer_neurons."""
    return {NeuronIdx(layer=layer, position=-1, neuron=neuron): 1.0}


def collect_rollouts(steerer: NeuronSteerer) -> dict:
    """Generation under different steering conditions for the target neuron."""
    target_layer, target_neuron = 24, 1619
    feature = single_neuron_set(target_layer, target_neuron)

    results = {}
    print("\n--- Baseline rollouts (no steering) ---")
    results["baseline"] = {}
    for set_name, prompts in PROMPT_SETS.items():
        print(f"\n  [{set_name}]")
        per = []
        for p in prompts[:5]:  # first 5 of each set, to keep this tractable
            text = generate(steerer, p, max_new_tokens=60)
            per.append({"prompt": p, "completion": text})
            print(f"    > {p[:60]:60s}\n      {text[:120]}")
        results["baseline"][set_name] = per

    print(f"\n--- Ablation rollouts (L{target_layer}/N{target_neuron} m=0) ---")
    results["ablate_target"] = {}
    for set_name in ("refusal_discovery_pos", "refusal_test"):
        prompts = PROMPT_SETS[set_name][:5]
        print(f"\n  [{set_name}]")
        per = []
        for p in prompts:
            text = generate(steerer, p, max_new_tokens=60, feature=feature, multiplier=0.0)
            per.append({"prompt": p, "completion": text})
            print(f"    > {p[:60]:60s}\n      {text[:120]}")
        results["ablate_target"][set_name] = per

    print(f"\n--- Amplification rollouts (L{target_layer}/N{target_neuron} m=2 and m=3) ---")
    results["amplify_target"] = {}
    for set_name in ("benign", "refusal_discovery_neg"):
        prompts = PROMPT_SETS[set_name][:5]
        for mult in (2.0, 3.0):
            key = f"{set_name}_m{mult}"
            print(f"\n  [{key}]")
            per = []
            for p in prompts:
                text = generate(steerer, p, max_new_tokens=60, feature=feature, multiplier=mult)
                per.append({"prompt": p, "completion": text})
                print(f"    > {p[:60]:60s}\n      {text[:120]}")
            results["amplify_target"][key] = per

    print(f"\n--- Comparison: amplify L24/N2598 (canonical writer) on benign ---")
    results["amplify_canonical_writer"] = {}
    for mult in (2.0, 3.0):
        cir = single_neuron_set(24, 2598)
        key = f"benign_m{mult}"
        print(f"\n  [{key}]")
        per = []
        for p in PROMPT_SETS["benign"][:5]:
            text = generate(steerer, p, max_new_tokens=60, feature=cir, multiplier=mult)
            per.append({"prompt": p, "completion": text})
            print(f"    > {p[:60]:60s}\n      {text[:120]}")
        results["amplify_canonical_writer"][key] = per

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--skip-rollouts", action="store_true",
                    help="Only compute activation dossier; skip generation rollouts.")
    args = ap.parse_args()

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"apparatus/output/dossier_L24N1619_{stamp}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    steerer = NeuronSteerer(args.model)

    print("=" * 70)
    print("Activation dossier")
    print("=" * 70)
    activations = collect_activations(steerer)

    # Compact summary printed inline
    print("\nPer-neuron mean(activation) by set:")
    print(f"  {'Neuron':14s} | {'disc_pos':>10s} {'disc_neg':>10s} {'test':>10s} {'benign':>10s}")
    for label, info in activations.items():
        b = info["by_set"]
        print(f"  {label:14s} | {b['refusal_discovery_pos']['mean']:+10.4f} "
              f"{b['refusal_discovery_neg']['mean']:+10.4f} "
              f"{b['refusal_test']['mean']:+10.4f} "
              f"{b['benign']['mean']:+10.4f}")

    dossier = {
        "model": args.model,
        "neurons_of_interest": COMPARISON_NEURONS,
        "activations": activations,
    }

    if not args.skip_rollouts:
        print("\n" + "=" * 70)
        print("Rollouts")
        print("=" * 70)
        dossier["rollouts"] = collect_rollouts(steerer)

    out_path = args.output_dir / "dossier.json"
    out_path.write_text(json.dumps(dossier, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
