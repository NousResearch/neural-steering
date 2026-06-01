"""Apparatus 7 Phase 1: decision-time oscilloscope (position-time traces).

See apparatus/THEORY_decision_time.md for the full reasoning. Short version:

Saturday's static flow apparatus stayed a diagram because a single forward pass
over depth (layer->layer) has no axis with an arrow of time. But POSITION does:
the causal mask makes position j depend only on positions <= j. And because of
that mask, the post-final-norm state at position j is EXACTLY what the model
would have if the prompt ended at j -- so reading every position of ONE forward
pass is equivalent to re-running at every prefix length. The whole experiment is
that free equivalence.

This script produces, per prompt, a stacked figure with x-axis = prompt token
position (ticks = the actual token strings) and these panels:

  Panel 1  refuse-margin spine  (PROBE-FREE; the payload):
              logit("I") - logit(best genuine compliant opener)
           read off the post-final-norm state at every position. "How
           refuse-committed is the model as a function of how much of the prompt
           it has read?" If this saturates partway through the prompt, the
           decision is a PREFILL phenomenon and generation-time commitment is
           the aftermath, not the event. We also record refusal-set mass vs
           compliance-set mass separately, because the margin is a RESULTANT
           that hides conflict (a flat margin can be a dead-heat tug-of-war).

  Panel 2  L24 substrate probe score per position   (detection axis)
  Panel 3  L32 gate probe score per position        (commitment axis; L32 =
           hidden_states[-1], post-final-norm alias, NOT a transformer block)
  Panel 4  pivot raw activations per position: L18/N7417, L24/N1619
  Panel 5  residual RMS (pre-final-norm) per position -- cheapest test of the
           "crowding" budget: margin moving where RMS does NOT => additive;
           margin moving WITH an RMS spike => crowding may be live.

What the probes are FOR (precise): Panel 1 is the object. Panels 2-3 decompose
the decision into detection-vs-commitment across position. The reading we hunt
is a LAG: does substrate rise at an earlier token than gate? Caveat: the probes
were calibrated at the FINAL position; applying them position-wise is
extrapolation. The token-labeled x-axis is the built-in face-validity check (do
they light up on content words, stay flat on boilerplate?). If they are noise
across positions, we lean on Panel 1 alone.

MODE-AGNOSTIC BY DESIGN: a readout is a (name, metric_fn) pair. Refusal is the
first instance because it is the high-contrast suppressive case (snav: the
object is "mode commitment under pressure", refusal just has the loudest shape).
Adding a second mode is a one-line metric swap, not a rewrite.

Reuses (single source of truth): NeuronSteerer (neuron_steer.core),
fit_mean_diff_probe / hidden_state_index / format_prompt / LinearProbe
(apparatus.probe_role_table), prompt sets (experiments.prompts). Probes are
REFIT in-script from the discovery sets and applied to held-out REFUSAL_TEST
prompts (probes never see the prompts they are read on).

This is a GPU run (one forward pass per prompt with output_hidden_states, plus
two single-neuron pre-hooks and one final-norm pre-hook). Cheap. Intended for
cluster. Opus does not submit cluster jobs; Codex reviews then submits.

Usage:
    python -m apparatus.decision_time \\
        --model llama8b \\
        --n-prompts 4 \\
        --out-dir apparatus/output/decision_time_<date>/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuron_steer.core import NeuronSteerer
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST,
)
from apparatus.probe_role_table import (
    LinearProbe,
    fit_mean_diff_probe,
    hidden_state_index,
    format_prompt,
)

MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}

# Pivot neurons characterized in earlier apparatus (see THEORY doc / role-table
# dossiers). (layer, neuron) in MLP-neuron-basis (input to down_proj).
PIVOTS = {
    "L18/N7417": (18, 7417),
    "L24/N1619": (24, 1619),
}

# Readout token sets for the probe-free margin spine.
# "I" is the canonical refusal opener for Llama-3.1-8B-Instruct ("I can't help
# with that..."). Compliance openers are tokens that begin a genuinely compliant
# continuation. We resolve each to its FIRST token id with a leading space (the
# generation position is mid-sentence after the assistant header, so the model
# emits a space-prefixed word). We report the realized top-k per position so the
# contrast can be audited (Codex's load-bearing review point: a margin is only
# meaningful if the compliant token is actually a live alternative).
REFUSAL_WORDS = ["I", "Sorry", "Unfortunately"]
COMPLIANCE_WORDS = ["Sure", "Here", "To", "Yes", "Of", "Certainly"]


# ----------------------------------------------------------------------------
# Token-set resolution
# ----------------------------------------------------------------------------

def resolve_token_ids(tokenizer, words: list[str]) -> dict[str, int]:
    """Map each word to its first token id, trying space-prefixed then bare.

    Mid-sentence generation positions emit space-prefixed words, so we prefer
    " word"; fall back to "word" if the space-prefixed form tokenizes oddly.
    Returns {word: token_id}. Records both forms in the printed audit.
    """
    out = {}
    for w in words:
        spaced = tokenizer(" " + w, add_special_tokens=False).input_ids
        bare = tokenizer(w, add_special_tokens=False).input_ids
        # Prefer the space-prefixed first token (generation is mid-sentence).
        tid = spaced[0] if spaced else bare[0]
        out[w] = tid
    return out


# ----------------------------------------------------------------------------
# ALL traces from ONE forward pass (Codex review #2, #3)
# ----------------------------------------------------------------------------
#
# The "one pass gives all positions" equivalence is conceptually central, so the
# code must honor it literally: a SINGLE output_hidden_states forward, with the
# pivot down_proj pre-hooks and the final-norm pre-hook attached to that same
# pass. Logits come straight from out.logits (no manual lm_head reapplication --
# avoids device-map ambiguity and matches HF's exact forward path).

def collect_all_traces(
    steerer: NeuronSteerer,
    input_ids: torch.Tensor,
    probes: dict[int, LinearProbe],
    refusal_ids: dict[str, int],
    compliance_ids: dict[str, int],
    pivots: dict[str, tuple[int, int]],
    top_n: int,
) -> dict:
    """One forward pass; margin spine + probe scores + pivots + RMS, all positions.

    Causal-mask equivalence: position j's post-final-norm state == the state the
    model would have if the prompt ended at j. So logit(token) read at position
    j == "what the model would predict next, given a j-token prefix."
    """
    model = steerer.model
    captured: dict[str, np.ndarray] = {}
    rms_holder: dict[str, np.ndarray] = {}
    handles = []

    def make_pivot_hook(name: str, neuron_idx: int):
        def pre_hook(module, args):
            x = args[0]  # (batch, seq, intermediate_size)
            captured[name] = x[0, :, neuron_idx].detach().float().cpu().numpy()
        return pre_hook

    for name, (layer, neuron_idx) in pivots.items():
        h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
            make_pivot_hook(name, neuron_idx)
        )
        handles.append(h)

    def norm_pre_hook(module, args):
        # Input to the final RMSNorm = pre-final-norm residual, ALL positions.
        x = args[0]
        rms_holder["rms"] = x[0].detach().float().pow(2).mean(dim=-1).sqrt().cpu().numpy()

    handles.append(model.model.norm.register_forward_pre_hook(norm_pre_hook))

    try:
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    hidden_states = out.hidden_states  # tuple len n_layers+1
    logits_all = out.logits[0].float()  # (seq, vocab) -- HF's exact forward path
    seq_len = logits_all.shape[0]

    refusal_words = list(refusal_ids.keys())
    compliance_words = list(compliance_ids.keys())
    refusal_tid = list(refusal_ids.values())
    compliance_tid = list(compliance_ids.values())

    margin = np.zeros(seq_len)
    refusal_mass = np.zeros(seq_len)      # logsumexp over the refusal set
    compliance_mass = np.zeros(seq_len)   # logsumexp over the compliance set
    # Audit (Codex review #1): the margin is only meaningful if a compliance
    # token is a LIVE alternative. Record, per position, the best of each set,
    # the rank of the best compliance token in the full vocab, and the top-n.
    audit_per_pos = []

    log_probs_all = torch.log_softmax(logits_all, dim=-1)
    # Vocab-wide rank of every token = how many tokens have strictly higher lp.
    for j in range(seq_len):
        lp = log_probs_all[j]
        r = lp[refusal_tid]
        c = lp[compliance_tid]
        r_best = int(r.argmax())
        c_best = int(c.argmax())
        margin[j] = (r[r_best] - c[c_best]).item()
        refusal_mass[j] = torch.logsumexp(r, dim=0).item()
        compliance_mass[j] = torch.logsumexp(c, dim=0).item()

        best_compliance_tid = compliance_tid[c_best]
        # Rank of best-compliance token in full vocab (0 = argmax of vocab).
        compliance_rank = int((lp > lp[best_compliance_tid]).sum().item())
        best_refusal_tid = refusal_tid[r_best]
        refusal_rank = int((lp > lp[best_refusal_tid]).sum().item())

        top_lp, top_idx = lp.topk(top_n)
        audit_per_pos.append({
            "best_refusal_word": refusal_words[r_best],
            "best_refusal_logprob": float(r[r_best]),
            "best_refusal_rank": refusal_rank,
            "best_compliance_word": compliance_words[c_best],
            "best_compliance_logprob": float(c[c_best]),
            "best_compliance_rank": compliance_rank,
            "top_n": [
                (steerer.tokenizer.decode([int(t)]), float(v))
                for t, v in zip(top_idx.tolist(), top_lp.tolist())
            ],
        })

    # Probe scores per position. Each probe reads its own layer's hidden state.
    probe_traces: dict[str, np.ndarray] = {}
    for layer, probe in probes.items():
        hs_idx = hidden_state_index(steerer, layer)
        layer_resid = hidden_states[hs_idx][0]  # (seq, d_model)
        scores = np.zeros(seq_len)
        for j in range(seq_len):
            scores[j] = probe.score_hidden(layer_resid[j]).item()
        probe_traces[f"L{layer}"] = scores

    return {
        "seq_len": seq_len,
        "margin": margin,
        "refusal_mass": refusal_mass,
        "compliance_mass": compliance_mass,
        "audit_per_pos": audit_per_pos,
        "probe_traces": probe_traces,
        "pivot_acts": captured,
        "rms_pre_final_norm": rms_holder.get("rms"),
    }


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def plot_prompt(prompt: str, tokens: list[str], traces: dict,
                pivot_acts: dict, rms, probe_layers: list[int], out_path: Path):
    seq_len = traces["seq_len"]
    x = np.arange(seq_len)

    fig, axes = plt.subplots(5, 1, figsize=(max(10, seq_len * 0.45), 16),
                             sharex=True)

    # Panel 1: margin spine + two sides separately
    ax = axes[0]
    ax.plot(x, traces["margin"], color="black", lw=2, label="refuse-comply margin (resultant)")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.plot(x, traces["refusal_mass"], color="crimson", lw=1, alpha=0.7, label="refusal-set mass")
    ax.plot(x, traces["compliance_mass"], color="steelblue", lw=1, alpha=0.7, label="compliance-set mass")
    ax.set_ylabel("logprob / margin")
    ax.set_title(f"Panel 1 — refuse-margin spine (probe-free)\nprompt: {prompt!r}", fontsize=10)
    ax.legend(fontsize=8, loc="best")

    # Panel 2: substrate probe (L24)
    ax = axes[1]
    sub_layer = min(probe_layers)
    ax.plot(x, traces["probe_traces"][f"L{sub_layer}"], color="darkorange", lw=1.5)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_ylabel(f"L{sub_layer} substrate probe")
    ax.set_title("Panel 2 — substrate (detection) probe per position", fontsize=10)

    # Panel 3: gate probe (L32)
    ax = axes[2]
    gate_layer = max(probe_layers)
    ax.plot(x, traces["probe_traces"][f"L{gate_layer}"], color="purple", lw=1.5)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_ylabel(f"L{gate_layer} gate probe")
    ax.set_title("Panel 3 — gate (commitment) probe per position", fontsize=10)

    # Panel 4: pivots
    ax = axes[3]
    for name in PIVOTS:
        if name in pivot_acts and pivot_acts[name] is not None:
            ax.plot(x, pivot_acts[name], lw=1.3, label=name)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_ylabel("pivot activation")
    ax.set_title("Panel 4 — pivot raw activations per position", fontsize=10)
    ax.legend(fontsize=8, loc="best")

    # Panel 5: residual RMS (pre-final-norm)
    ax = axes[4]
    if rms is not None:
        ax.plot(x, rms, color="seagreen", lw=1.5)
    ax.set_ylabel("residual RMS\n(pre-final-norm)")
    ax.set_title("Panel 5 — residual RMS per position (crowding-budget probe)", fontsize=10)

    # Token strings on x-axis (the legibility + face-validity check).
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [t.replace("$", r"\$") for t in tokens], rotation=90, fontsize=7,
    )
    axes[-1].set_xlabel("prompt token position")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=list(MODELS))
    ap.add_argument("--probe-layers", default="24,32",
                    help="Comma-separated probe layers (32 = post-final-norm alias)")
    ap.add_argument("--n-prompts", type=int, default=4,
                    help="Number of held-out REFUSAL_TEST prompts to trace")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Per-position top-n tokens recorded for the contrast "
                         "audit (Codex review: best-compliance rank must be "
                         "visible deeper than top-5 to judge if the margin "
                         "contrast is a live alternative)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    probe_layers = [int(x) for x in args.probe_layers.split(",") if x.strip()]
    model_name = MODELS[args.model]
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(f"apparatus/output/decision_time_{args.model}_{ts}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name}...")
    # No circuit discovery/steering here, so skip universal-neuron detection
    # (Codex review #4: unnecessary overhead + log noise for a read-only run).
    steerer = NeuronSteerer(model_name, auto_blacklist=False)

    # --- Refit probes in-script from discovery sets; print validation. ---
    probes: dict[int, LinearProbe] = {}
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

    # --- Resolve readout token sets; print the audit (load-bearing). ---
    refusal_ids = resolve_token_ids(steerer.tokenizer, REFUSAL_WORDS)
    compliance_ids = resolve_token_ids(steerer.tokenizer, COMPLIANCE_WORDS)
    print("Refusal token ids:", {w: (tid, steerer.tokenizer.decode([tid]))
                                  for w, tid in refusal_ids.items()})
    print("Compliance token ids:", {w: (tid, steerer.tokenizer.decode([tid]))
                                     for w, tid in compliance_ids.items()})

    prompts = REFUSAL_TEST[:args.n_prompts]
    print(f"\nTracing {len(prompts)} held-out prompts (probes never saw these):")

    all_records = {}
    for i, prompt in enumerate(prompts):
        input_ids = format_prompt(steerer, prompt)
        token_strs = [steerer.tokenizer.decode([t]) for t in input_ids[0].tolist()]

        traces = collect_all_traces(
            steerer, input_ids, probes, refusal_ids, compliance_ids,
            PIVOTS, args.top_n,
        )
        pivot_acts = traces["pivot_acts"]
        rms = traces["rms_pre_final_norm"]

        safe = "".join(c if c.isalnum() else "_" for c in prompt)[:40]
        png_path = args.out_dir / f"prompt{i:02d}_{safe}.png"
        plot_prompt(prompt, token_strs, traces, pivot_acts, rms, probe_layers, png_path)
        # First-position compliance-rank sanity line (Codex review #1): if the
        # best compliance token never ranks near the top, the margin contrast is
        # a strawman -- inspect traces.json audit_per_pos before trusting Panel 1.
        last_audit = traces["audit_per_pos"][-1]
        print(f"  [{i+1}/{len(prompts)}] {prompt!r} "
              f"seq_len={traces['seq_len']} final_margin={traces['margin'][-1]:+.3f} "
              f"best_comply={last_audit['best_compliance_word']!r}@rank{last_audit['best_compliance_rank']} "
              f"-> {png_path.name}")

        all_records[f"prompt{i:02d}"] = {
            "prompt": prompt,
            "tokens": token_strs,
            "seq_len": traces["seq_len"],
            "margin": traces["margin"].tolist(),
            "refusal_mass": traces["refusal_mass"].tolist(),
            "compliance_mass": traces["compliance_mass"].tolist(),
            "probe_traces": {k: v.tolist() for k, v in traces["probe_traces"].items()},
            "pivot_activations": {
                k: (v.tolist() if v is not None else None)
                for k, v in pivot_acts.items()
            },
            "rms_pre_final_norm": (rms.tolist() if rms is not None else None),
            "audit_per_pos": traces["audit_per_pos"],
        }

    run_summary = {
        "model": model_name,
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
    print(f"\nWrote {len(prompts)} figures + traces.json + run_summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
