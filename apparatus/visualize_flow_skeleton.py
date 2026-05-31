"""Apparatus 6 simplified-structure views: get to the SIMPLE graph the basin
finding implies, by contracting/pruning rather than drawing all 372 raw edges.

Two complementary clean views (the raw funnel is a hairball; if the structure is
really simple, we should draw the simple structure, not the full edge set):

A) DOMINANT-FLOW SKELETON
   For each node keep only its single strongest outgoing edge (|weight|). A
   convergent funnel's skeleton is a forest/tree, so this collapses the hairball
   into a branching structure that terminates at the sinks. Data-driven (no
   hand-defined groups). We report the fraction of total |weight| the skeleton
   retains, so we know whether the tree is faithful or dropping real routes.

B) LAYER-BAND QUOTIENT
   Contract nodes to supernodes = (coarse layer-band x field-basin) and draw
   weighted inter-supernode flow — the "few boxes and arrows" view. The banding
   is HAND-CHOSEN (descriptive contraction, not discovered structure — Louvain
   found no communities), so it's labeled as such. We report compression:
   #supernode-edges vs #raw-edges and the fraction of flow on inter-band edges.

Usage:
    python -m apparatus.visualize_flow_skeleton \\
        --field-dir apparatus/output/probe_edge_fields_<run>/ \\
        --out-dir apparatus/output/skeleton_<date>/
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

# Hand-chosen layer bands for the quotient (descriptive, not discovered).
BANDS = [("early", 0, 13), ("mid", 14, 22), ("gate", 23, 31)]


def band_of(layer: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= layer <= hi:
            return name
    return "other"


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


# ---------- A) dominant-flow skeleton ----------

def dominant_skeleton(G: nx.DiGraph):
    """Keep each node's single strongest outgoing edge. Returns (skeleton DiGraph,
    retained_fraction of total abs-weight)."""
    S = nx.DiGraph()
    total = sum(d["abs_weight"] for *_, d in G.edges(data=True))
    kept = 0.0
    for n in G.nodes():
        outs = [(v, d["weight"], d["abs_weight"]) for _, v, d in G.out_edges(n, data=True)]
        if not outs:
            continue
        v, w, aw = max(outs, key=lambda t: t[2])
        S.add_edge(n, v, weight=w, abs_weight=aw)
        kept += aw
    return S, (kept / total if total else 0.0)


def prune_direct_leaves(S: nx.DiGraph, sinks: set, iterate: bool = False) -> nx.DiGraph:
    """Remove length-1 paths: edges u->v where v is a terminal (skeleton
    out-degree 0, or a named sink) AND u is a source (skeleton in-degree 0).

    These are 'direct contributor' nodes that point straight at the end and have
    nothing feeding them. What remains is the multi-hop RELAY SPINE — nodes that
    are actually on a path (something feeds them, and/or they feed an
    intermediate before the terminal). With iterate=True, repeat until stable so
    only genuine through-paths survive.
    """
    P = S.copy()
    while True:
        sink_set = set(sinks) | {n for n in P.nodes() if P.out_degree(n) == 0}
        to_remove = [(u, v) for u, v in P.edges()
                     if v in sink_set and P.in_degree(u) == 0]
        if not to_remove:
            break
        P.remove_edges_from(to_remove)
        P.remove_nodes_from([n for n in list(P.nodes()) if P.degree(n) == 0])
        if not iterate:
            break
    P.remove_nodes_from([n for n in list(P.nodes()) if P.degree(n) == 0])
    return P


def rank_y(nodes) -> dict:
    layers = sorted({l for (l, _) in nodes})
    n = len(layers)
    return {l: (i / max(n - 1, 1)) * (n - 1) for i, l in enumerate(layers)}


def collapse_shared_successors(S: nx.DiGraph, sinks: set, protect=frozenset()):
    """Quotient the skeleton by SHARED SUCCESSOR: any set of >=2 nodes whose
    single dominant out-edge points to the same target is merged into one bundle
    node. Redirect in-edges to the bundle, iterate until stable.

    On the dominant-flow skeleton each node has <=1 out-edge, so 'shared
    terminus' = same successor. This collapses fan-in funnels into a chain of
    bundle nodes (e.g. all L13/L14 nodes -> L16 become one bundle -> L16).

    Returns (Q DiGraph with node attrs members/n_members/layer_span/summed_abs,
    mapping orig_node -> bundle_id).
    """
    # work on a copy with member-tracking
    members = {n: {n} for n in S.nodes()}
    succ = {}   # node -> (target, abs_weight) of its single dominant edge
    for u, v, d in S.edges(data=True):
        succ[u] = (v, d["abs_weight"])

    # union-find over nodes that share a successor (excluding sinks/protected)
    parent = {n: n for n in S.nodes()}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_target = defaultdict(list)
    for n, (t, _) in succ.items():
        if n in sinks or n in protect:
            continue
        by_target[t].append(n)
    for t, group in by_target.items():
        # don't merge a node into a group if it IS the target (self) ; group are sources
        for n in group[1:]:
            union(group[0], n)

    # build bundle members
    bundles = defaultdict(set)
    for n in S.nodes():
        bundles[find(n)].add(n)

    # bundle id -> nice label + layer span + summed weight of outgoing edges
    Q = nx.DiGraph()
    node2bundle = {n: find(n) for n in S.nodes()}
    for bid, mem in bundles.items():
        layers = sorted({l for (l, _) in mem})
        Q.add_node(bid, members=sorted(mem), n_members=len(mem),
                   layer_span=(min(layers), max(layers)))
    for u, v, d in S.edges(data=True):
        bu, bv = node2bundle[u], node2bundle[v]
        if bu == bv:
            continue
        if Q.has_edge(bu, bv):
            Q[bu][bv]["abs_weight"] += d["abs_weight"]
            Q[bu][bv]["weight"] += d["weight"]
        else:
            Q.add_edge(bu, bv, abs_weight=d["abs_weight"], weight=d["weight"])
    return Q, node2bundle


def draw_quotient_tree(ax, Q, node2bundle, sinks, l2y, title, color):
    """Draw the shared-successor quotient. Node y = mean layer of members; size
    ∝ n_members; sinks starred; bundles labeled with span + member count."""
    sink_bids = {node2bundle[s] for s in sinks if s in node2bundle}
    by = {bid: float(np.mean([l2y[l] for (l, _) in Q.nodes[bid]["members"]]))
          for bid in Q.nodes()}
    # Horizontal layout: spread bundles across the FULL width, ordered by a
    # barycenter pass for locality but NEVER collapsed onto the centerline (that
    # was the overlap problem — a near-linear chain converges to x~0.5 and all
    # edges stack). We compute a barycenter ordering, then assign evenly-spaced
    # x slots within each layer band so width is always used.
    pos = {bid: (0.5, by[bid]) for bid in Q.nodes()}
    for _ in range(12):
        for bid in Q.nodes():
            nb = list(Q.successors(bid)) + list(Q.predecessors(bid))
            if nb:
                pos[bid] = (float(np.mean([pos[x][0] for x in nb])), by[bid])
    # Now spread: bucket by rounded layer band, assign evenly-spaced slots by the
    # barycenter-derived order so connected bundles stay roughly aligned but
    # every node gets a distinct x across [0.1, 0.9].
    band_members = defaultdict(list)
    for bid in Q.nodes():
        band_members[round(by[bid])].append(bid)
    for band, bids in band_members.items():
        ordered = sorted(bids, key=lambda b: (pos[b][0], str(b)))
        m = len(ordered)
        for j, bid in enumerate(ordered):
            x = 0.5 if (m == 1) else 0.1 + 0.8 * j / (m - 1)
            pos[bid] = (x, by[bid])
    # gentle vertical de-collision for same-layer bundles
    import hashlib
    for band, bids in band_members.items():
        if len(bids) <= 1:
            continue
        for bid in bids:
            h = int(hashlib.md5(str(bid).encode()).hexdigest()[:6], 16) / 0xFFFFFF
            pos[bid] = (pos[bid][0], pos[bid][1] + (h - 0.5) * 0.22)

    gate_ys = [l2y[l] for l in l2y if 29 <= l <= 31]
    if gate_ys:
        ax.axhspan(min(gate_ys) - 0.4, max(gate_ys) + 0.4, color="#ffe9b0", alpha=0.4, zorder=0)
    wmax = max((d["abs_weight"] for *_, d in Q.edges(data=True)), default=1.0)
    for u, v, d in Q.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.06",
                     arrowstyle="-|>", mutation_scale=13,
                     linewidth=0.7 + 4.0 * d["abs_weight"] / wmax,
                     color=color, alpha=0.7, zorder=2))
    for bid, (x, y) in pos.items():
        n_mem = Q.nodes[bid]["n_members"]
        lo, hi = Q.nodes[bid]["layer_span"]
        is_sink = bid in sink_bids
        ax.scatter([x], [y], s=(260 if is_sink else 60 + 45 * n_mem), c=color,
                   marker=("*" if is_sink else "o"), edgecolor="black",
                   linewidth=0.5, zorder=5, alpha=0.9)
        lbl = (f"L{lo}/N{Q.nodes[bid]['members'][0][1]}" if n_mem == 1
               else f"bundle L{lo}-{hi}\n({n_mem} nodes)")
        ax.annotate(lbl, (x, y), xytext=(6, 5), textcoords="offset points",
                    fontsize=6.8, weight=("bold" if is_sink or n_mem > 1 else "normal"))
    ys = sorted(l2y.items())
    ax.set_yticks([y for _, y in ys]); ax.set_yticklabels([f"L{l}" for l, _ in ys], fontsize=6.5)
    ax.set_ylim(-0.7, (len(l2y) - 1) + 0.7); ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.10, axis="y")


def tree_x(S: nx.DiGraph, sinks: set) -> dict:
    """x-position by a simple tidy-tree pass: leaves spread evenly, parents at
    mean of children. Sinks anchored center. Works on the skeleton forest."""
    # order nodes by layer; assign leaves (no in-skeleton successors feeding up)
    # Simpler: iterative barycenter toward successors a few sweeps.
    by_layer = defaultdict(list)
    for nd in S.nodes():
        by_layer[nd[0]].append(nd)
    order = {l: sorted(m) for l, m in by_layer.items()}
    layers = sorted(by_layer)
    pos = {}
    for l in layers:
        m = order[l]
        for j, k in enumerate(m):
            pos[k] = (j + 0.5) / len(m)
    for s in sinks:
        if s in pos:
            pos[s] = 0.5
    for sweep in range(20):
        for l in (layers if sweep % 2 == 0 else layers[::-1]):
            def bary(k):
                if k in sinks:
                    return 0.5
                nb = list(S.successors(k)) + list(S.predecessors(k))
                vals = [pos[x] for x in nb if x in pos]
                return sum(vals) / len(vals) if vals else pos[k]
            order[l] = sorted(order[l], key=bary)
            for j, k in enumerate(order[l]):
                pos[k] = (j + 0.5) / len(order[l]) if k not in sinks else 0.5
    return pos


def draw_skeleton(ax, S, sinks, l2y, title, color):
    pos = tree_x(S, sinks)
    # Spread across FULL width per layer band (the chain otherwise collapses onto
    # x~0.5 and edges/labels stack). Order within band by the tree_x barycenter
    # so connected nodes stay roughly aligned, then assign even slots.
    band = defaultdict(list)
    for nd in S.nodes():
        band[nd[0]].append(nd)
    for layer, nds in band.items():
        ordered = sorted(nds, key=lambda n: (pos.get(n, 0.5), n[1]))
        m = len(ordered)
        for j, nd in enumerate(ordered):
            pos[nd] = 0.5 if m == 1 else 0.1 + 0.8 * j / (m - 1)
    xy = {nd: (pos.get(nd, 0.5), l2y[nd[0]]) for nd in S.nodes()}
    gate_ys = [l2y[l] for l in l2y if 29 <= l <= 31]
    if gate_ys:
        ax.axhspan(min(gate_ys) - 0.4, max(gate_ys) + 0.4, color="#ffe9b0", alpha=0.4, zorder=0)
    wmax = max((d["abs_weight"] for *_, d in S.edges(data=True)), default=1.0)
    for u, v, d in S.edges(data=True):
        x0, y0 = xy[u]; x1, y1 = xy[v]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.05",
                     arrowstyle="-|>", mutation_scale=12,
                     linewidth=0.6 + 3.0 * d["abs_weight"] / wmax,
                     color=color, alpha=0.7, zorder=2))
    for nd, (x, y) in xy.items():
        is_sink = nd in sinks; is_pivot = nd == PIVOT
        ax.scatter([x], [y], s=(240 if is_sink else 46), c=color,
                   marker=("*" if is_sink else "o"),
                   edgecolor=("#000" if is_pivot else "black"),
                   linewidth=(2.2 if is_pivot else 0.4),
                   zorder=6 if (is_sink or is_pivot) else 4, alpha=0.9)
        if is_sink or is_pivot:
            ax.annotate(f"L{nd[0]}/N{nd[1]}", (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7, weight="bold")
    ys = sorted(l2y.items())
    ax.set_yticks([y for _, y in ys]); ax.set_yticklabels([f"L{l}" for l, _ in ys], fontsize=6.5)
    ax.set_ylim(-0.7, (len(l2y) - 1) + 0.7); ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.10, axis="y")


# ---------- B) layer-band quotient ----------

def band_quotient(G: nx.DiGraph):
    """Contract to supernodes = layer-band; signed + abs inter-band flow."""
    q = defaultdict(lambda: {"weight": 0.0, "abs": 0.0, "n": 0})
    intra = 0.0; total = 0.0
    for u, v, d in G.edges(data=True):
        bu, bv = band_of(u[0]), band_of(v[0])
        total += d["abs_weight"]
        if bu == bv:
            intra += d["abs_weight"]; continue
        rec = q[(bu, bv)]
        rec["weight"] += d["weight"]; rec["abs"] += d["abs_weight"]; rec["n"] += 1
    return dict(q), (1 - intra / total if total else 0.0)


def draw_band_quotient(ax, q, title):
    band_y = {"early": 0, "mid": 1, "gate": 2}
    pos = {b: (0.5, y) for b, y in band_y.items()}
    wmax = max((v["abs"] for v in q.values()), default=1.0)
    for (bu, bv), v in q.items():
        if bu not in pos or bv not in pos:
            continue
        x0, y0 = pos[bu]; x1, y1 = pos[bv]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1 + 6 * v["abs"] / wmax,
                                    color="#1f4fb4" if v["weight"] >= 0 else "#c43030",
                                    alpha=0.7, connectionstyle="arc3,rad=0.25"))
        ax.text((x0 + x1) / 2 + 0.12, (y0 + y1) / 2,
                f"{v['weight']:+.2f}\n({v['n']} edges)", fontsize=8, va="center")
    for b, (x, y) in pos.items():
        ax.scatter([x], [y], s=2600, c="#ddd", edgecolor="black", zorder=5)
        lo, hi = next((lo, hi) for nm, lo, hi in BANDS if nm == b)
        ax.text(x, y, f"{b}\nL{lo}-{hi}", ha="center", va="center", fontsize=9, weight="bold", zorder=6)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 2.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", required=True, type=Path)
    ap.add_argument("--substrate-layer", type=int, default=24)
    ap.add_argument("--gate-layer", type=int, default=32)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    Ga = load_collapsed_dir(args.field_dir / f"L{args.substrate_layer}" / "edges.json")
    Gb = load_collapsed_dir(args.field_dir / f"L{args.gate_layer}" / "edges.json")
    l2y = rank_y(list(Ga.nodes()) + list(Gb.nodes()))

    # A) skeletons
    Sa, fa = dominant_skeleton(Ga)
    Sb, fb = dominant_skeleton(Gb)
    print(f"Skeleton retains: L24 {fa:.1%} of |weight| ({Sa.number_of_edges()} of "
          f"{Ga.number_of_edges()} edges); L32 {fb:.1%} ({Sb.number_of_edges()} of "
          f"{Gb.number_of_edges()})")

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    draw_skeleton(axes[0], Sa, SUBSTRATE_SINKS, l2y,
                  f"L{args.substrate_layer} substrate dominant-flow skeleton "
                  f"({fa:.0%} of flow)", "#c43030")
    draw_skeleton(axes[1], Sb, GATE_SINKS, l2y,
                  f"L{args.gate_layer} gate dominant-flow skeleton ({fb:.0%} of flow)", "#1f4fb4")
    axes[0].set_ylabel("layer, rank-spaced", fontsize=9)
    fig.suptitle("Dominant-flow skeleton (top-1 outgoing edge per node) — the funnel as a tree",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "flow_skeleton.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_dir}/flow_skeleton.png")

    # A2) relay spine: prune length-1 paths (direct source->terminal contributors)
    Pa = prune_direct_leaves(Sa, SUBSTRATE_SINKS, iterate=True)
    Pb = prune_direct_leaves(Sb, GATE_SINKS, iterate=True)
    print(f"Relay spine (multi-hop paths only): L24 {Pa.number_of_nodes()} nodes / "
          f"{Pa.number_of_edges()} edges (was {Sa.number_of_nodes()}/{Sa.number_of_edges()}); "
          f"L32 {Pb.number_of_nodes()} nodes / {Pb.number_of_edges()} edges "
          f"(was {Sb.number_of_nodes()}/{Sb.number_of_edges()})")
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    draw_skeleton(axes3[0], Pa, SUBSTRATE_SINKS, l2y,
                  f"L{args.substrate_layer} substrate relay spine "
                  f"({Pa.number_of_edges()} edges, length>=2 paths)", "#c43030")
    draw_skeleton(axes3[1], Pb, GATE_SINKS, l2y,
                  f"L{args.gate_layer} gate relay spine "
                  f"({Pb.number_of_edges()} edges, length>=2 paths)", "#1f4fb4")
    axes3[0].set_ylabel("layer, rank-spaced", fontsize=9)
    fig3.suptitle("Relay spine: dominant-flow skeleton with direct (length-1) "
                  "contributors removed — only multi-hop routes",
                  fontsize=12, y=0.99)
    fig3.tight_layout(rect=(0, 0, 1, 0.97))
    fig3.savefig(args.out_dir / "relay_spine.png", dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"wrote {args.out_dir}/relay_spine.png")

    # A3) collapse shared-successor bundles on the relay spine -> minimal structure
    Qa, _ = collapse_shared_successors(Pa, SUBSTRATE_SINKS)
    Qb, _ = collapse_shared_successors(Pb, GATE_SINKS)
    # node2bundle over the relay-spine nodes for sink id lookup
    _, n2b_a = collapse_shared_successors(Pa, SUBSTRATE_SINKS)
    _, n2b_b = collapse_shared_successors(Pb, GATE_SINKS)
    print(f"Shared-successor quotient: L24 {Qa.number_of_nodes()} bundles / "
          f"{Qa.number_of_edges()} edges; L32 {Qb.number_of_nodes()} bundles / "
          f"{Qb.number_of_edges()} edges")
    fig4, axes4 = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    draw_quotient_tree(axes4[0], Qa, n2b_a, SUBSTRATE_SINKS, l2y,
                       f"L{args.substrate_layer} substrate: shared-successor quotient "
                       f"({Qa.number_of_nodes()} bundles)", "#c43030")
    draw_quotient_tree(axes4[1], Qb, n2b_b, GATE_SINKS, l2y,
                       f"L{args.gate_layer} gate: shared-successor quotient "
                       f"({Qb.number_of_nodes()} bundles)", "#1f4fb4")
    axes4[0].set_ylabel("layer, rank-spaced", fontsize=9)
    fig4.suptitle("Shared-successor quotient: nodes pointing to the same target "
                  "merged into one bundle — the minimal route structure",
                  fontsize=12, y=0.99)
    fig4.tight_layout(rect=(0, 0, 1, 0.97))
    fig4.savefig(args.out_dir / "shared_successor_quotient.png", dpi=140, bbox_inches="tight")
    plt.close(fig4)
    print(f"wrote {args.out_dir}/shared_successor_quotient.png")

    # B) band quotient
    qa, ia = band_quotient(Ga)
    qb, ib = band_quotient(Gb)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 7))
    draw_band_quotient(axes2[0], qa,
                       f"L{args.substrate_layer} substrate band-quotient "
                       f"({ia:.0%} flow inter-band)")
    draw_band_quotient(axes2[1], qb,
                       f"L{args.gate_layer} gate band-quotient ({ib:.0%} flow inter-band)")
    fig2.suptitle("Hand-defined layer-band quotient (descriptive contraction; "
                  "blue=+ / red=- signed flow)", fontsize=11, y=0.99)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(args.out_dir / "band_quotient.png", dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {args.out_dir}/band_quotient.png")

    (args.out_dir / "skeleton_summary.json").write_text(json.dumps({
        "skeleton_flow_retained": {"L24": fa, "L32": fb},
        "skeleton_edges": {"L24": Sa.number_of_edges(), "L32": Sb.number_of_edges()},
        "raw_edges": {"L24": Ga.number_of_edges(), "L32": Gb.number_of_edges()},
        "band_quotient": {
            "bands": [list(b) for b in BANDS],
            "L24_inter_band_edges": {f"{u}->{v}": {"signed": round(d["weight"], 4),
                                                   "abs": round(d["abs"], 4), "n": d["n"]}
                                     for (u, v), d in qa.items()},
            "L32_inter_band_edges": {f"{u}->{v}": {"signed": round(d["weight"], 4),
                                                   "abs": round(d["abs"], 4), "n": d["n"]}
                                     for (u, v), d in qb.items()},
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
