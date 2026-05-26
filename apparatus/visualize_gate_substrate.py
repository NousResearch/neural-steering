"""Gate/substrate visualization: tokenwise sufficiency vs probe sufficiency.

Restricted to upstream-of-probe neurons (downstream-of-probe neurons return
structural zeros on the probe readout, so plotting them obscures structure).

A diagonal cluster (high tokenwise = high probe) would mean writer role is
substrate-invariant — same neurons write substrate and write tokens.

An off-diagonal pattern would mean writer role is readout-relative — tokenwise
writers are gate neurons that read substrate from elsewhere and produce the
output token, distinct from substrate-writer neurons.

Run after probe_role_table.py and compare_probe_roles.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_one(ax, joined: list[dict], probe_layer: int):
    upst = [r for r in joined if r["upstream_of_probe"]]

    x = np.array([r["token_sufficiency_dS"] for r in upst])
    y = np.array([r["probe_sufficiency_dProbe"] for r in upst])

    ax.axhline(0, color="#888", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#888", linewidth=0.7, zorder=0)

    xlim = (-3.0, 2.5)
    ylim = (-0.10, 0.16)
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Quadrant shading
    q_colors = {
        # (label, color, x-range, y-range)
        ("substrate+token writer", "#cff4d8", (0, xlim[1]), (0, ylim[1])),
        ("token-only writer\n(gate, not substrate)",
         "#fff3c4", (0, xlim[1]), (ylim[0], 0)),
        ("substrate-only writer\n(upstream of gate)",
         "#d6e6fb", (xlim[0], 0), (0, ylim[1])),
        ("both-suppressor", "#fbcdcd", (xlim[0], 0), (ylim[0], 0)),
    }
    for label, color, (x0, x1), (y0, y1) in q_colors:
        ax.fill_betweenx([y0, y1], x0, x1, color=color, alpha=0.30, zorder=0)

    # Quadrant labels
    ax.text(1.7, 0.145, "substrate-writes\n+ token-writes",
            ha="center", va="center", fontsize=8, color="#2a6b3a", weight="bold")
    ax.text(1.7, -0.085, "token writer\n(NOT substrate)",
            ha="center", va="center", fontsize=8, color="#9a7a00", weight="bold")
    ax.text(-2.0, 0.145, "SUBSTRATE writer\n(not token)",
            ha="center", va="center", fontsize=8, color="#2a4a82", weight="bold")
    ax.text(-2.0, -0.085, "both-suppressor",
            ha="center", va="center", fontsize=8, color="#a83232", weight="bold")

    ax.scatter(x, y, s=60, c="#1f77b4", edgecolor="black",
               linewidth=0.5, alpha=0.78, zorder=3)

    # Annotate the standout neurons by tokenwise or probe extremity
    for r in upst:
        layer, neuron = r["layer"], r["neuron"]
        tx = r["token_sufficiency_dS"]
        py = r["probe_sufficiency_dProbe"]
        if tx is None or py is None:
            continue
        if abs(tx) > 1.2 or abs(py) > 0.07:
            ax.annotate(f"L{layer}/N{neuron}",
                        (tx, py),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=7.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", ec="#ccc", alpha=0.85))

    ax.set_xlabel("Tokenwise sufficiency (dS on P(\"I\"))", fontsize=10)
    ax.set_ylabel(f"Probe sufficiency at L{probe_layer} (dProbe)", fontsize=10)
    ax.set_title(f"Upstream of L{probe_layer}: n={len(upst)}", fontsize=11)
    ax.grid(alpha=0.18)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--layers", default="18,24,28")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]
    cmps = {}
    for L in layers:
        cmps[L] = json.load(open(args.probe_dir / f"compare_token_probe_layer{L}.json"))

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 7),
                             sharex=True, sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, L in zip(axes, layers):
        plot_one(ax, cmps[L]["joined_rows"], L)

    fig.suptitle(
        "Gate vs Substrate: tokenwise refusal sufficiency vs probe-readout sufficiency\n"
        "(Llama-3.1-8B, upstream-of-probe neurons only)",
        fontsize=12, y=1.00)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
