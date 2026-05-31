"""Apparatus 6 basin visualization: show the readout-selected circuits as
terminal BASINS (convergent funnels), not neuron knots.

The structural finding (coarse_flow + pivot_and_basins): each readout's circuit
is a near-pure convergent funnel — the L24-substrate field routes ~all nodes to
the L20 substrate hub; the L32-gate field routes ~all nodes to L29/N12010. There
is no community structure (modularity within both nulls); the coarse abstraction
is the basin (where flow terminates), not the cluster.

This draws that. Two panels side by side:
  LEFT:  L24 substrate field, flow converging on the L20 sink
  RIGHT: L32 gate field, flow converging on the L29 sink
Layer on y (rank-spaced, substrate bottom -> gate top). Within-layer x is set by
a SINK-DIRECTED barycenter: nodes are pulled horizontally toward the x of the
sink their flow feeds, so convergence reads as a funnel. Node color = basin
(red=substrate, blue=gate, purple=both, grey=off-path). Sinks drawn large.
The pivot L18/N7417 is ringed in both panels so you can see it switch basins.

A third panel overlays the union, coloring each node by its union basin, to show
the handoff: shared early nodes ('both') feeding two divergent terminals.

Usage:
    python -m apparatus.visualize_basins \\
        --field-dir apparatus/output/probe_edge_fields_<run>/ \\
        --out-dir apparatus/output/basins_<date>/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import networkx as nx

SUPER_WEIGHTS = {(0, 491), (0, 8268), (1, 198), (1, 2427)}
PIVOT = (18, 7417)
SUBSTRATE_SINKS = {(20, 9424), (20, 3972)}
GATE_SINKS = {(29, 12010)}

BASIN_COLOR = {
    "substrate": "#c43030",
    "gate":      "#1f4fb4",
    "both":      "#7a3fa8",
    "off_path":  "#bbbbbb",
}


def load_collapsed_dir(edges_path: Path, drop=SUPER_WEIGHTS) -> nx.DiGraph:
    raw = json.loads(edges_path.read_text())["edges"]
    agg = defaultdict(float)
    for e in raw:
        sl, _, sn = e["source"]; tl, _, tn = e["target"]
        if (sl, sn) in drop or (tl, tn) in drop:
            continue
        agg[((sl, sn), (tl, tn))] += e["weight"]
    G = nx.DiGraph()
    for (src, dst), w in agg.items():
        if src == dst:
            continue
        G.add_edge(src, dst, weight=w, abs_weight=abs(w))
    return G


def terminal_basins(G: nx.DiGraph) -> dict:
    sub = {s for s in SUBSTRATE_SINKS if s in G}
    gate = {s for s in GATE_SINKS if s in G}

    def reaches(node, targets):
        return any(node == t or nx.has_path(G, node, t) for t in targets) if targets else False

    labels = {}
    for n in G.nodes():
        rs, rg = reaches(n, sub), reaches(n, gate)
        labels[n] = ("both" if rs and rg else "substrate" if rs
                     else "gate" if rg else "off_path")
    return labels


def rank_y(nodes) -> dict:
    layers = sorted({l for (l, _) in nodes})
    n = len(layers)
    l2y = {l: (i / max(n - 1, 1)) * (n - 1) for i, l in enumerate(layers)}
    return l2y


def sink_directed_x(G: nx.DiGraph, sinks: set, l2y: dict, n_sweeps=14) -> dict:
    """Within-layer x via barycenter, but anchored so sink nodes sit center-top
    and upstream nodes are pulled toward the sink they feed — produces a funnel.
    """
    by_layer = defaultdict(list)
    for nd in G.nodes():
        by_layer[nd[0]].append(nd)
    order = {l: sorted(m) for l, m in by_layer.items()}
    layers = sorted(by_layer)

    # anchor: sinks at x=0.5
    fixed = {s: 0.5 for s in sinks if s in G}

    def positions():
        pos = {}
        for l in layers:
            m = order[l]
            for j, k in enumerate(m):
                pos[k] = (j + 0.5) / len(m)
        pos.update(fixed)
        return pos

    pos = positions()
    for sweep in range(n_sweeps):
        seq = layers if sweep % 2 == 0 else list(reversed(layers))
        for l in seq:
            def bary(k):
                if k in fixed:
                    return fixed[k]
                # pull toward successors (downstream, toward sink) and predecessors
                nb = list(G.successors(k)) + list(G.predecessors(k))
                vals = [pos[x] for x in nb if x in pos]
                return sum(vals) / len(vals) if vals else pos[k]
            order[l] = sorted(order[l], key=bary)
            for j, k in enumerate(order[l]):
                if k not in fixed:
                    pos[k] = (j + 0.5) / len(order[l])
    pos.update(fixed)
    return pos


def draw_basin(ax, G, sinks, basins, l2y, title, primary_basin):
    pos_x = sink_directed_x(G, sinks, l2y)
    xy = {nd: (pos_x.get(nd, 0.5), l2y[nd[0]]) for nd in G.nodes()}

    # gate band shade
    gate_ys = [l2y[l] for l in l2y if 29 <= l <= 31]
    if gate_ys:
        ax.axhspan(min(gate_ys) - 0.4, max(gate_ys) + 0.4, color="#ffe9b0", alpha=0.4, zorder=0)

    wmax = max((d["abs_weight"] for *_, d in G.edges(data=True)), default=1.0)
    for u, v, d in G.edges(data=True):
        x0, y0 = xy[u]; x1, y1 = xy[v]
        # edge inherits the basin color of its source (where flow is heading)
        col = BASIN_COLOR.get(basins.get(u, "off_path"), "#bbb")
        lw = 0.4 + 3.5 * d["abs_weight"] / wmax
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.08",
                     arrowstyle="-|>", mutation_scale=11, linewidth=lw,
                     color=col, alpha=0.5, zorder=1))

    for nd, (x, y) in xy.items():
        is_sink = nd in sinks
        is_pivot = nd == PIVOT
        ax.scatter([x], [y], s=(260 if is_sink else 50),
                   c=BASIN_COLOR.get(basins.get(nd, "off_path"), "#bbb"),
                   marker=("*" if is_sink else "o"),
                   edgecolor=("#000" if is_pivot else "black"),
                   linewidth=(2.2 if is_pivot else 0.4),
                   zorder=6 if (is_sink or is_pivot) else 4, alpha=0.92)
        if is_sink:
            ax.annotate(f"L{nd[0]}/N{nd[1]}", (x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=7.5, weight="bold")
    # label pivot
    if PIVOT in xy:
        ax.annotate("L18/N7417\n(pivot)", xy[PIVOT], xytext=(8, -2),
                    textcoords="offset points", fontsize=7, color="#000")

    ys = sorted(l2y.items())
    ax.set_yticks([y for _, y in ys]); ax.set_yticklabels([f"L{l}" for l, _ in ys], fontsize=6.5)
    ax.set_ylim(-0.7, (len(l2y) - 1) + 0.7)
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.10, axis="y")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", required=True, type=Path)
    ap.add_argument("--substrate-layer", type=int, default=24)
    ap.add_argument("--gate-layer", type=int, default=32)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    Ga = load_collapsed_dir(args.field_dir / f"L{args.substrate_layer}" / "edges.json")
    Gb = load_collapsed_dir(args.field_dir / f"L{args.gate_layer}" / "edges.json")
    Gu = nx.DiGraph()
    for G in (Ga, Gb):
        for u, v, d in G.edges(data=True):
            if Gu.has_edge(u, v):
                Gu[u][v]["abs_weight"] = max(Gu[u][v]["abs_weight"], d["abs_weight"])
            else:
                Gu.add_edge(u, v, abs_weight=d["abs_weight"], weight=d["weight"])

    ba, bb, bu = terminal_basins(Ga), terminal_basins(Gb), terminal_basins(Gu)
    # shared y across panels (union layers)
    l2y = rank_y(list(Gu.nodes()))

    fig, axes = plt.subplots(1, 3, figsize=(21, 9), sharey=True)
    draw_basin(axes[0], Ga, SUBSTRATE_SINKS, ba, l2y,
               f"L{args.substrate_layer} substrate field -> L20 basin", "substrate")
    draw_basin(axes[1], Gb, GATE_SINKS, bb, l2y,
               f"L{args.gate_layer} gate field -> L29 basin", "gate")
    draw_basin(axes[2], Gu, SUBSTRATE_SINKS | GATE_SINKS, bu, l2y,
               "union: shared early flow (purple) -> two terminals", "both")
    axes[0].set_ylabel("layer, rank-spaced (substrate -> gate)", fontsize=9)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BASIN_COLOR["substrate"],
               markeredgecolor="black", markersize=9, label="substrate basin (-> L20)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BASIN_COLOR["gate"],
               markeredgecolor="black", markersize=9, label="gate basin (-> L29)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BASIN_COLOR["both"],
               markeredgecolor="black", markersize=9, label="both (feeds both terminals)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#444",
               markeredgecolor="black", markersize=14, label="terminal sink"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
               markeredgecolor="#000", markeredgewidth=2, markersize=9, label="L18/N7417 pivot"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, framealpha=0.9)
    fig.suptitle("Readout-selected circuits as terminal basins (not modular communities)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "basins.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"L24 basins: {dict((k, sum(1 for x in ba.values() if x==k)) for k in set(ba.values()))}")
    print(f"L32 basins: {dict((k, sum(1 for x in bb.values() if x==k)) for k in set(bb.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
