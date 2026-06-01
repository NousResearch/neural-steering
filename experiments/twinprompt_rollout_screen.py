"""Rollout screen for twinprompt benign/mal pairs.

This is the gate for comply-probe methodology:
  - If off-diagonal cells are populated, fit a harm-balanced comply probe.
  - If off-diagonal cells are sparse, use CNA threshold-to-flip as the causal
    ambivalence measure instead of forcing a weak probe.

Logs behavior labels plus first-token distribution diagnostics for each prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuron_steer.core import NeuronSteerer
from experiments.rollout_screen import classify

MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
DEFAULT_DATASET = Path("/home/jake/claude/greenhouse/refusal-invariance/datasets/twinprompt.json")

REFUSAL_WORDS = ["I", "Sorry", "Unfortunately"]


def resolve_token_ids(tokenizer, words: list[str]) -> dict[str, int]:
    out = {}
    for w in words:
        ids = tokenizer(" " + w, add_special_tokens=False).input_ids
        if not ids:
            ids = tokenizer(w, add_special_tokens=False).input_ids
        out[w] = ids[0]
    return out


def first_token_diag(steerer: NeuronSteerer, prompt: str, top_n: int, refusal_ids: dict[str, int]) -> dict:
    formatted = steerer._format_prompt(prompt)
    enc = steerer.tokenizer(formatted, return_tensors="pt")
    input_ids = enc.input_ids.to(steerer.device)
    attention_mask = enc.attention_mask.to(steerer.device) if "attention_mask" in enc else None
    with torch.no_grad():
        out = steerer.model(input_ids, attention_mask=attention_mask)
    logits = out.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    top_lp, top_idx = log_probs.topk(top_n)
    top = [
        {"token": steerer.tokenizer.decode([int(t)]), "id": int(t), "logprob": float(v)}
        for t, v in zip(top_idx.tolist(), top_lp.tolist())
    ]

    refusal_tids = list(refusal_ids.values())
    refusal_lp = log_probs[refusal_tids]
    best_i = int(refusal_lp.argmax().item())
    best_tid = refusal_tids[best_i]
    best_word = list(refusal_ids.keys())[best_i]
    best_lp = float(refusal_lp[best_i].item())
    best_rank = int((log_probs > refusal_lp[best_i]).sum().item() + 1)
    refusal_mass = float(torch.logsumexp(refusal_lp, dim=0).item())

    refusal_set = set(refusal_tids)
    best_non_refusal = None
    for item in top:
        if item["id"] not in refusal_set:
            best_non_refusal = item
            break

    return {
        "top_n": top,
        "best_refusal_word": best_word,
        "best_refusal_id": best_tid,
        "best_refusal_logprob": best_lp,
        "best_refusal_rank": best_rank,
        "refusal_mass": refusal_mass,
        "best_non_refusal_top": best_non_refusal,
    }


def generate(steerer: NeuronSteerer, prompt: str, max_new: int) -> str:
    formatted = steerer._format_prompt(prompt)
    enc = steerer.tokenizer(formatted, return_tensors="pt")
    input_ids = enc.input_ids.to(steerer.device)
    attention_mask = enc.attention_mask.to(steerer.device) if "attention_mask" in enc else None
    with torch.no_grad():
        out = steerer.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=steerer.tokenizer.eos_token_id,
        )
    return steerer.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def summarize(records: list[dict]) -> dict:
    counts = Counter((r["side"], r["label"]) for r in records)
    by_category = defaultdict(Counter)
    for r in records:
        by_category[r["category"]][(r["side"], r["label"])] += 1

    off_diag = {
        "mal_comply_or_hedge": [
            r["id"] for r in records
            if r["side"] == "mal" and r["label"] in {"COMPLY", "COMPLY+HEDGE"}
        ],
        "benign_refuse": [
            r["id"] for r in records
            if r["side"] == "benign" and r["label"].startswith("REFUSE")
        ],
    }
    pair_outcomes = {}
    grouped = defaultdict(dict)
    for r in records:
        grouped[r["pair_id"]][r["side"]] = r
    for pid, pair in grouped.items():
        if "benign" in pair and "mal" in pair:
            pair_outcomes[pid] = {
                "category": pair["benign"]["category"],
                "benign_label": pair["benign"]["label"],
                "mal_label": pair["mal"]["label"],
                "behavior_flips": pair["benign"]["label"] != pair["mal"]["label"],
            }

    return {
        "counts": {f"{side}:{label}": n for (side, label), n in sorted(counts.items())},
        "by_category": {
            cat: {f"{side}:{label}": n for (side, label), n in sorted(c.items())}
            for cat, c in sorted(by_category.items())
        },
        "off_diagonal": off_diag,
        "pair_outcomes": pair_outcomes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=list(MODELS))
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="Debug limit on number of pairs (0=all)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(f"experiments/output/twinprompt_rollout_screen_{args.model}_{ts}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.load(open(args.dataset))
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Loading {MODELS[args.model]}...")
    steerer = NeuronSteerer(MODELS[args.model], auto_blacklist=False)
    refusal_ids = resolve_token_ids(steerer.tokenizer, REFUSAL_WORDS)
    print("Refusal token ids:", {w: (tid, steerer.tokenizer.decode([tid]))
                                  for w, tid in refusal_ids.items()})
    print(f"Screening {len(pairs)} twinprompt pairs ({2 * len(pairs)} prompts)")

    records = []
    for i, pair in enumerate(pairs):
        for side in ("benign", "mal"):
            prompt = pair[side]
            diag = first_token_diag(steerer, prompt, args.top_n, refusal_ids)
            gen = generate(steerer, prompt, args.max_new)
            label = classify(gen)
            rec = {
                "id": f"{i:03d}_{side}",
                "pair_id": i,
                "side": side,
                "category": pair.get("category", "unknown"),
                "prompt": prompt,
                "label": label,
                "first_token": diag,
                "generation": gen,
            }
            records.append(rec)
            print(f"[{i+1:03d}/{len(pairs)} {side:6s}] {label:12s} "
                  f"ref_rank={diag['best_refusal_rank']:6d} "
                  f"top={diag['top_n'][0]['token']!r}  {prompt[:70]}")

    summary = summarize(records)
    (args.out_dir / "rollouts.json").write_text(json.dumps(records, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY")
    print(json.dumps(summary["counts"], indent=2))
    print("off_diagonal:", json.dumps(summary["off_diagonal"], indent=2))
    print(f"\nWrote rollouts.json + summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
