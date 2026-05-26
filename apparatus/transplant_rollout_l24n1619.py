"""Transplant-rollout test for L24/N1619.

The first L24/N1619 dossier used multiplier steering on benign prompts. That
does not test the prior sufficiency result well because L24/N1619 has near-zero
activation on benign prompts, so multiplying it mostly injects nothing.

This script instead uses the sufficiency-test intervention directly: collect a
source activation from harmful prompts, transplant that scalar into benign
prompts, then inspect next-token logits and greedy rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuron_steer.core import NeuronSteerer
from experiments.prompts import (
    BENIGN_PROMPTS,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_DISCOVERY_POSITIVE,
)
from experiments.sufficiency_test import (
    collect_mean_neuron_activation,
    make_transplant_single_ctx,
)


NEURONS = [
    ("L24/N1619", 24, 1619, "substrate-writer / token-suppressor"),
    ("L24/N2598", 24, 2598, "canonical token-writer"),
    ("L22/N3319", 22, 3319, "token writer-only"),
    ("L20/N9928", 20, 9928, "token writer-only"),
    ("L26/N11984", 26, 11984, "consistent token suppressor"),
]

PROMPT_SETS = {
    "benign": BENIGN_PROMPTS[:5],
    "refusal_discovery_neg": REFUSAL_DISCOVERY_NEGATIVE[:5],
}

WATCH_TOKENS = ["I", "Sorry", "Unfortunately", "No", "Sure", "Here", "The", "It"]


def format_input(steerer: NeuronSteerer, prompt: str) -> dict[str, torch.Tensor]:
    formatted = steerer._format_prompt(prompt, "")
    return steerer.tokenizer(formatted, return_tensors="pt").to(steerer.device)


def token_id(steerer: NeuronSteerer, token: str) -> int:
    return steerer.tokenizer.encode(token, add_special_tokens=False)[-1]


def top_next_tokens(
    steerer: NeuronSteerer,
    prompt: str,
    make_ctx: Callable | None = None,
    k: int = 12,
) -> dict:
    inputs = format_input(steerer, prompt)
    ctx = make_ctx() if make_ctx is not None else nullcontext()
    with ctx:
        with torch.no_grad():
            logits = steerer.model(**inputs).logits[0, -1].float()
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k=k)
    top_items = []
    for prob, tid in zip(top.values.tolist(), top.indices.tolist()):
        text = steerer.tokenizer.decode([tid])
        top_items.append({
            "token": text,
            "token_id": int(tid),
            "prob": float(prob),
            "logit": float(logits[tid].item()),
        })

    watch = {}
    for tok in WATCH_TOKENS:
        tid = token_id(steerer, tok)
        watch[tok] = {
            "token_id": int(tid),
            "prob": float(probs[tid].item()),
            "logit": float(logits[tid].item()),
        }

    i_id = token_id(steerer, "I")
    masked = logits.clone()
    masked[i_id] = float("-inf")
    return {
        "top": top_items,
        "watch_tokens": watch,
        "p_I": float(probs[i_id].item()),
        "logit_I": float(logits[i_id].item()),
        "margin_I_vs_max_other": float((logits[i_id] - masked.max()).item()),
    }


def generate_text(
    steerer: NeuronSteerer,
    prompt: str,
    make_ctx: Callable | None = None,
    max_new_tokens: int = 80,
) -> str:
    inputs = format_input(steerer, prompt)
    ctx = make_ctx() if make_ctx is not None else nullcontext()
    with ctx:
        with torch.no_grad():
            out = steerer.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=steerer.tokenizer.eos_token_id,
            )
    return steerer.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def run_for_neuron(
    steerer: NeuronSteerer,
    label: str,
    layer: int,
    neuron: int,
    note: str,
    prompts_by_set: dict[str, list[str]],
) -> dict:
    source_mean = collect_mean_neuron_activation(
        steerer,
        layer,
        neuron,
        REFUSAL_DISCOVERY_POSITIVE,
    )
    make_ctx = make_transplant_single_ctx(steerer.model, layer, neuron, source_mean)

    print(f"\n{'=' * 72}")
    print(f"{label} ({note}) target_val={source_mean:+.4f}")
    print(f"{'=' * 72}")

    by_set = {}
    for set_name, prompts in prompts_by_set.items():
        rows = []
        print(f"\n[{set_name}]")
        for prompt in prompts:
            base_next = top_next_tokens(steerer, prompt)
            tx_next = top_next_tokens(steerer, prompt, make_ctx=make_ctx)
            base_text = generate_text(steerer, prompt)
            tx_text = generate_text(steerer, prompt, make_ctx=make_ctx)
            row = {
                "prompt": prompt,
                "baseline_next": base_next,
                "transplant_next": tx_next,
                "delta_p_I": tx_next["p_I"] - base_next["p_I"],
                "delta_margin_I": (
                    tx_next["margin_I_vs_max_other"]
                    - base_next["margin_I_vs_max_other"]
                ),
                "baseline_completion": base_text,
                "transplant_completion": tx_text,
            }
            rows.append(row)
            base_top = ", ".join(
                f"{x['token']!r}:{x['prob']:.3f}" for x in base_next["top"][:5]
            )
            tx_top = ", ".join(
                f"{x['token']!r}:{x['prob']:.3f}" for x in tx_next["top"][:5]
            )
            print(f"  > {prompt[:64]}")
            print(
                f"    dP(I)={row['delta_p_I']:+.4f} "
                f"dMargin(I)={row['delta_margin_I']:+.3f}"
            )
            print(f"    base top: {base_top}")
            print(f"    tx   top: {tx_top}")
            print(f"    tx text: {tx_text[:120]}")
        by_set[set_name] = rows

    return {
        "label": label,
        "layer": layer,
        "neuron": neuron,
        "note": note,
        "source_prompts": "REFUSAL_DISCOVERY_POSITIVE",
        "target_activation": float(source_mean),
        "by_set": by_set,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--neurons", default="L24/N1619,L24/N2598,L22/N3319,L20/N9928,L26/N11984")
    args = ap.parse_args()

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"apparatus/output/transplant_rollout_L24N1619_{stamp}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = {x.strip() for x in args.neurons.split(",") if x.strip()}
    neurons = [n for n in NEURONS if n[0] in selected]
    if not neurons:
        raise ValueError(f"No known neurons selected from: {sorted(selected)}")

    steerer = NeuronSteerer(args.model)
    result = {
        "model": args.model,
        "prompt_sets": {k: v for k, v in PROMPT_SETS.items()},
        "source_prompt_set": "REFUSAL_DISCOVERY_POSITIVE",
        "watch_tokens": WATCH_TOKENS,
        "neurons": {},
    }

    for label, layer, neuron, note in neurons:
        result["neurons"][label] = run_for_neuron(
            steerer,
            label,
            layer,
            neuron,
            note,
            PROMPT_SETS,
        )

    out_path = args.output_dir / "transplant_rollout.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"TRANSPLANT_ROLLOUT_OUTPUT {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
