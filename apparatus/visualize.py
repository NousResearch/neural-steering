"""Visualize the role table in role-space.

Two scatters side-by-side:
- Position-preserving view (same neuron repeated across positions, n=47 for refusal)
- Collapsed (layer, neuron) view (n=24 unique intervention-tested neurons for refusal)

The collapsed view is the honest unit for intervention-axis claims, since
necessity/sufficiency are position-collapsed in the source data.

X = necessity_sigma; Y = sufficiency_dS; color = attribution sign; size = max edge
degree.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collapse(rows: list[dict]) -> list[dict]:
    """Same logic as analyze.collapse_to_layer_neuron, kept duplicated to avoid
    cross-module dependency. If this drifts, sync from analyze.py."""
    by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["layer"], r["neuron"])].append(r)
    out = []
    for (layer, neuron), group in by_key.items():
        attrs = [g["attribution"] for g in group]
        dom = max(attrs, key=abs)
        merged = {
            "layer": layer,
            "neuron": neuron,
            "n_positions": len(group),
            "attribution": dom,
            "edge_in_count": sum(g.get("edge_in_count", 0) for g in group),
            "edge_out_count": sum(g.get("edge_out_count", 0) for g in group),
            "necessity_sigma": group[0].get("necessity_sigma"),
            "sufficiency_dS": group[0].get("sufficiency_dS"),
            "is_super_weight": group[0].get("is_super_weight", False),
        }
        out.append(merged)
    return out


def plot_role_space(ax, rows: list[dict], subtitle: str) -> None:
    rows_iv = [r for r in rows if r.get("necessity_sigma") is not None
               and r.get("sufficiency_dS") is not None]
    nec = np.array([r["necessity_sigma"] for r in rows_iv])
    suff = np.array([r["sufficiency_dS"] for r in rows_iv])
    attr = np.array([r["attribution"] for r in rows_iv])
    edge_max = np.array([max(r.get("edge_in_count", 0), r.get("edge_out_count", 0))
                          for r in rows_iv])
    is_super = np.array([r.get("is_super_weight", False) for r in rows_iv])

    ax.axhline(0, color="#888", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#888", linewidth=0.7, zorder=0)

    xlim = (-30, 55)
    ylim = (-3.0, 2.5)
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    quad_colors = {
        "reader/writer (+nec, +suff)":  ("#cff4d8", (0, xlim[1]), (0, ylim[1])),
        "writer-only (-nec, +suff)":    ("#fff3c4", (xlim[0], 0), (0, ylim[1])),
        "suppressors (-nec, -suff)":    ("#fbcdcd", (xlim[0], 0), (ylim[0], 0)),
        "anti-writer (+nec, -suff)":    ("#d6e6fb", (0, xlim[1]), (ylim[0], 0)),
    }
    for _, (color, (x0, x1), (y0, y1)) in quad_colors.items():
        ax.fill_betweenx([y0, y1], x0, x1, color=color, alpha=0.30, zorder=0)

    ax.text(40, 2.2, "READERS / WRITERS", ha="center", va="center",
            fontsize=9, color="#2a6b3a", weight="bold")
    ax.text(-20, 2.2, "WRITER-ONLY", ha="center", va="center",
            fontsize=9, color="#9a7a00", weight="bold")
    ax.text(-20, -2.7, "SUPPRESSORS", ha="center", va="center",
            fontsize=9, color="#a83232", weight="bold")
    ax.text(40, -2.7, "ANTI-WRITER", ha="center", va="center",
            fontsize=9, color="#2a4a82", weight="bold")

    pos_attr = attr >= 0
    sizes = 30 + 6 * edge_max

    ax.scatter(nec[pos_attr & ~is_super], suff[pos_attr & ~is_super],
               s=sizes[pos_attr & ~is_super], c="#1f77b4",
               edgecolor="black", linewidth=0.5, alpha=0.78,
               label="attribution > 0", zorder=3)
    ax.scatter(nec[~pos_attr & ~is_super], suff[~pos_attr & ~is_super],
               s=sizes[~pos_attr & ~is_super], c="#d62728",
               edgecolor="black", linewidth=0.5, alpha=0.78,
               label="attribution < 0", zorder=3)
    if is_super.any():
        ax.scatter(nec[is_super], suff[is_super],
                   s=sizes[is_super], marker="^", c="#999",
                   edgecolor="black", linewidth=0.5, alpha=0.78,
                   label="super-weight", zorder=3)

    annotated_keys: set[tuple[int, int]] = set()
    for r in rows_iv:
        layer, neuron = r["layer"], r["neuron"]
        if (layer, neuron) in annotated_keys:
            continue
        nec_v = r["necessity_sigma"]
        suff_v = r["sufficiency_dS"]
        if abs(nec_v) > 12 or abs(suff_v) > 1.3:
            annotated_keys.add((layer, neuron))
            ax.annotate(f"L{layer}/N{neuron}",
                        (nec_v, suff_v),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=7.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", ec="#ccc", alpha=0.85))

    ax.set_xlabel("Necessity σ (ablation effect vs random control)", fontsize=10)
    ax.set_ylabel("Sufficiency dS (transplant effect on benign prompts)", fontsize=10)
    ax.set_title(subtitle, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.18)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title",
                    default="Refusal circuit role-space (Llama-3.1-8B, March-5 data)")
    args = ap.parse_args()

    rows = load_rows(args.inp)
    collapsed = collapse(rows)
    rows_iv = [r for r in rows if r.get("necessity_sigma") is not None]
    coll_iv = [r for r in collapsed if r.get("necessity_sigma") is not None]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17, 8), sharey=True)
    plot_role_space(
        ax_l, rows,
        f"Position-preserved view\n(n={len(rows_iv)} rows; intervention data "
        f"repeated across positions)")
    plot_role_space(
        ax_r, collapsed,
        f"Collapsed (layer, neuron) view\n(n={len(coll_iv)} unique intervention-tested neurons)")

    fig.suptitle(args.title + "\nmarker size = max(edge_in, edge_out)",
                 fontsize=12, y=1.00)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
