"""Apparatus 7 Phase 1b: true user-prefix decision-time oscilloscope.

`decision_time.py` reads every position in one formatted chat prompt. That is a
valid per-position trace, but it is NOT equivalent to asking "if the user prompt
ended here, what would the assistant answer?" for chat models: adding the
assistant generation header is a future suffix, and states inside the user
message do not see it under the causal mask.

This script runs the correct prefix sweep. For each user-token prefix, it
formats that prefix as a complete chat prompt with `add_generation_prompt=True`,
then reads the final assistant-position state. It is more forward passes, but
the held-out refusal prompts are short, and this is the right object.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuron_steer.core import NeuronSteerer
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST,
)
from apparatus.probe_role_table import fit_mean_diff_probe, format_prompt
from apparatus.decision_time import (
    MODELS,
    PIVOTS,
    REFUSAL_WORDS,
    COMPLIANCE_WORDS,
    resolve_token_ids,
    collect_all_traces,
    plot_prompt,
)


def prefix_strings(tokenizer, prompt: str) -> tuple[list[str], list[str]]:
    """Return token labels and decoded prefix strings for the raw user prompt."""
    raw_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    labels = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in raw_ids]
    prefixes = [
        tokenizer.decode(raw_ids[: i + 1], clean_up_tokenization_spaces=False)
        for i in range(len(raw_ids))
    ]
    return labels, prefixes


def collect_prefix_sweep(
    steerer: NeuronSteerer,
    prompt: str,
    probes: dict,
    refusal_ids: dict[str, int],
    compliance_ids: dict[str, int],
    top_n: int,
) -> tuple[list[str], dict, dict, np.ndarray]:
    """Trace final assistant-position signals for each user-prefix."""
    token_labels, prefixes = prefix_strings(steerer.tokenizer, prompt)
    n = len(prefixes)

    traces = {
        "seq_len": n,
        "margin": np.zeros(n),
        "refusal_mass": np.zeros(n),
        "compliance_mass": np.zeros(n),
        "probe_traces": {f"L{layer}": np.zeros(n) for layer in probes},
        "audit_per_pos": [],
    }
    pivot_acts = {name: np.zeros(n) for name in PIVOTS}
    rms = np.zeros(n)

    for i, prefix in enumerate(prefixes):
        input_ids = format_prompt(steerer, prefix)
        rec = collect_all_traces(
            steerer, input_ids, probes, refusal_ids, compliance_ids, PIVOTS, top_n=top_n
        )
        j = rec["seq_len"] - 1
        traces["margin"][i] = rec["margin"][j]
        traces["refusal_mass"][i] = rec["refusal_mass"][j]
        traces["compliance_mass"][i] = rec["compliance_mass"][j]
        traces["audit_per_pos"].append(rec["audit_per_pos"][j])
        for layer in probes:
            key = f"L{layer}"
            traces["probe_traces"][key][i] = rec["probe_traces"][key][j]
        for name in PIVOTS:
            pivot_acts[name][i] = rec["pivot_acts"][name][j]
        rms[i] = rec["rms_pre_final_norm"][j]

    return token_labels, traces, pivot_acts, rms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=list(MODELS))
    ap.add_argument("--probe-layers", default="24,32")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    probe_layers = [int(x) for x in args.probe_layers.split(",") if x.strip()]
    model_name = MODELS[args.model]
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(f"apparatus/output/decision_time_prefix_{args.model}_{ts}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name}...")
    steerer = NeuronSteerer(model_name, auto_blacklist=False)

    probes = {}
    probe_validation = {}
    for layer in probe_layers:
        probe, summary = fit_mean_diff_probe(
            steerer, layer,
            positive_prompts=REFUSAL_DISCOVERY_POSITIVE,
            negative_prompts=REFUSAL_DISCOVERY_NEGATIVE,
        )
        probes[layer] = probe
        probe_validation[f"L{layer}"] = summary
        print(f"  Probe L{layer}: AUC={summary.get('auc'):.4f}, "
              f"margin={summary.get('margin'):+.4f} "
              f"(pos {summary.get('pos_mean'):+.3f} / neg {summary.get('neg_mean'):+.3f})")

    refusal_ids = resolve_token_ids(steerer.tokenizer, REFUSAL_WORDS)
    compliance_ids = resolve_token_ids(steerer.tokenizer, COMPLIANCE_WORDS)
    print("Refusal token ids:", {w: (tid, steerer.tokenizer.decode([tid]))
                                  for w, tid in refusal_ids.items()})
    print("Compliance token ids:", {w: (tid, steerer.tokenizer.decode([tid]))
                                     for w, tid in compliance_ids.items()})

    prompts = REFUSAL_TEST[:args.n_prompts]
    all_records = {}
    print(f"\nTracing {len(prompts)} held-out prompts with true user-prefix sweep:")
    for i, prompt in enumerate(prompts):
        token_labels, traces, pivot_acts, rms = collect_prefix_sweep(
            steerer, prompt, probes, refusal_ids, compliance_ids, args.top_n
        )
        safe = "".join(c if c.isalnum() else "_" for c in prompt)[:40]
        png_path = args.out_dir / f"prompt{i:02d}_{safe}.png"
        plot_prompt(prompt, token_labels, traces, pivot_acts, rms, probe_layers, png_path)
        final_audit = traces["audit_per_pos"][-1]
        print(f"  [{i+1}/{len(prompts)}] {prompt!r} raw_tokens={traces['seq_len']} "
              f"final_margin={traces['margin'][-1]:+.3f} "
              f"best_comply={final_audit['best_compliance_word']!r}"
              f"@rank{final_audit['best_compliance_rank']} -> {png_path.name}")

        all_records[f"prompt{i:02d}"] = {
            "prompt": prompt,
            "tokens": token_labels,
            "seq_len": traces["seq_len"],
            "margin": traces["margin"].tolist(),
            "refusal_mass": traces["refusal_mass"].tolist(),
            "compliance_mass": traces["compliance_mass"].tolist(),
            "probe_traces": {k: v.tolist() for k, v in traces["probe_traces"].items()},
            "pivot_activations": {k: v.tolist() for k, v in pivot_acts.items()},
            "rms_pre_final_norm": rms.tolist(),
            "audit_per_pos": traces["audit_per_pos"],
        }

    run_summary = {
        "model": model_name,
        "mode": "true_user_prefix_sweep",
        "probe_layers": probe_layers,
        "probe_validation": probe_validation,
        "refusal_token_ids": refusal_ids,
        "compliance_token_ids": compliance_ids,
        "pivots": {k: list(v) for k, v in PIVOTS.items()},
        "n_prompts": len(prompts),
        "prompts": prompts,
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    (args.out_dir / "traces.json").write_text(json.dumps(all_records, indent=2))
    print(f"\nWrote {len(prompts)} prefix-sweep figures + traces.json + run_summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
