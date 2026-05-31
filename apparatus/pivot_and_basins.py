"""Apparatus 6 structure probes (Codex's cheap next-steps 2 & 3):

(2) POSITION-PRESERVING PIVOT DOSSIER for L18/N7417 — test whether the handoff
    is a real route switch or a collapsed-position artifact. Show incoming/outgoing
    by TOKEN POSITION (not collapsed) in the L24-substrate vs L32-gate fields, and
    the downstream target switch (L20 substrate hubs vs L29 gate node).

(3) TERMINAL-BASIN ASSIGNMENT — the better abstraction tool for a hierarchical
    feed-forward DAG than community detection. For each node, follow outgoing flow
    forward and ask where it terminates:
      - substrate basin: reaches the L20 substrate target hubs
      - gate basin:      reaches L29/N12010 (the gate target)
      - both / pre-basin: feeds both downstream basins
      - off-path:        reaches neither sink
    Basins are computed per field (L24 and L32) and compared. If nodes partition
    cleanly into basins, THAT is the coarse abstraction Louvain failed to find.

Plus DIRECTED BETWEENNESS for the pivot vs a layer-preserving null, to test the
'instrument/bottleneck' claim directly rather than via generic clustering.

Usage:
    python -m apparatus.pivot_and_basins \\
        --field-dir apparatus/output/probe_edge_fields_<run>/ \\
        --out-dir apparatus/output/pivot_basins_<date>/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import networkx as nx

SUPER_WEIGHTS = {(0, 491), (0, 8268), (1, 198), (1, 2427)}
PIVOT = (18, 7417)
SUBSTRATE_SINKS = {(20, 9424), (20, 3972)}   # L20 substrate target hubs
GATE_SINKS = {(29, 12010)}                    # L29 gate target


def load_raw(edges_path: Path, drop=SUPER_WEIGHTS):
    """Position-preserving edge list: [(src(l,p,n), dst(l,p,n), w)]."""
    raw = json.loads(edges_path.read_text())["edges"]
    out = []
    for e in raw:
        sl, sp, sn = e["source"]; tl, tp, tn = e["target"]
        if (sl, sn) in drop or (tl, tn) in drop:
            continue
        out.append(((sl, sp, sn), (tl, tp, tn), e["weight"]))
    return out


def collapsed_dir_graph(raw_edges) -> nx.DiGraph:
    """Position-collapsed directed graph keyed (layer,neuron); summed signed w."""
    agg = defaultdict(float)
    for (s, d, w) in raw_edges:
        agg[((s[0], s[2]), (d[0], d[2]))] += w
    G = nx.DiGraph()
    for (src, dst), w in agg.items():
        if src == dst:
            continue
        G.add_edge(src, dst, weight=w, abs_weight=abs(w))
    return G


# ---------- (2) position-preserving pivot dossier ----------

def pivot_dossier(raw_a, raw_b, pivot=PIVOT) -> dict:
    def by_pos(raw):
        inc = defaultdict(list)   # position -> [(src(l,p,n), w)]
        out = defaultdict(list)   # position -> [(dst(l,p,n), w)]
        for (s, d, w) in raw:
            if (d[0], d[2]) == pivot:
                inc[d[1]].append((s, w))
            if (s[0], s[2]) == pivot:
                out[s[1]].append((d, w))
        return inc, out

    inc_a, out_a = by_pos(raw_a)
    inc_b, out_b = by_pos(raw_b)

    def summarize_out(out_by_pos):
        # which downstream LAYER does each position route to, and signed weight
        rows = []
        for pos, lst in sorted(out_by_pos.items()):
            for (d, w) in lst:
                tgt = (d[0], d[2])
                basin = ("substrate" if tgt in SUBSTRATE_SINKS
                         else "gate" if tgt in GATE_SINKS else "other")
                rows.append({"pivot_pos": pos, "target": f"L{d[0]}/N{d[2]}",
                             "target_pos": d[1], "weight": round(w, 4), "basin": basin})
        return rows

    return {
        "pivot": f"L{pivot[0]}/N{pivot[1]}",
        "L24_substrate": {
            "incoming_positions": sorted(inc_a),
            "n_incoming": sum(len(v) for v in inc_a.values()),
            "outgoing": summarize_out(out_a),
            "total_in_weight": round(sum(w for v in inc_a.values() for _, w in v), 4),
        },
        "L32_gate": {
            "incoming_positions": sorted(inc_b),
            "n_incoming": sum(len(v) for v in inc_b.values()),
            "outgoing": summarize_out(out_b),
            "total_in_weight": round(sum(w for v in inc_b.values() for _, w in v), 4),
        },
    }


# ---------- (3) terminal-basin assignment ----------

def terminal_basins(G: nx.DiGraph) -> dict:
    """For each node, can it reach the substrate sinks / gate sinks (following
    directed edges forward)? Assign a basin label."""
    sub = {s for s in SUBSTRATE_SINKS if s in G}
    gate = {s for s in GATE_SINKS if s in G}

    def reaches(node, targets):
        if not targets:
            return False
        return any(t in G and (node == t or nx.has_path(G, node, t)) for t in targets)

    labels = {}
    for node in G.nodes():
        r_sub = reaches(node, sub)
        r_gate = reaches(node, gate)
        if r_sub and r_gate:
            labels[node] = "both"
        elif r_sub:
            labels[node] = "substrate"
        elif r_gate:
            labels[node] = "gate"
        else:
            labels[node] = "off_path"
    return labels


def directed_betweenness_pivot(G: nx.DiGraph, pivot=PIVOT,
                               field_b: dict | None = None,
                               n_null: int = 200, seed: int = 0) -> dict:
    """Directed betweenness centrality of the pivot vs a layer-preserving null.

    Tests the 'instrument/bottleneck' claim: does the pivot sit on an unusually
    large fraction of directed paths, beyond what a layer-preserving random graph
    with the same inter-layer edge counts produces?
    """
    bc = nx.betweenness_centrality(G, weight=None, normalized=True)
    obs = bc.get(pivot, 0.0)
    rank = sorted(bc.values(), reverse=True)
    obs_rank = sum(1 for v in rank if v > obs) + 1

    # layer-preserving null on the directed graph
    rng = np.random.default_rng(seed)
    nodes_by_layer = defaultdict(list)
    for n in G.nodes():
        nodes_by_layer[n[0]].append(n)
    edges = list(G.edges())
    null_bc = []
    for it in range(n_null):
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for (s, d) in edges:
            ns = nodes_by_layer[s[0]][rng.integers(len(nodes_by_layer[s[0]]))]
            nd = nodes_by_layer[d[0]][rng.integers(len(nodes_by_layer[d[0]]))]
            if ns != nd and ns[0] < nd[0]:  # keep feed-forward
                H.add_edge(ns, nd)
        b = nx.betweenness_centrality(H, normalized=True)
        null_bc.append(b.get(pivot, 0.0))
    null_bc = np.array(null_bc, dtype=float)
    mean, std = float(null_bc.mean()), float(null_bc.std())
    z = (obs - mean) / std if std > 1e-9 else float("inf")
    return {
        "pivot": f"L{pivot[0]}/N{pivot[1]}",
        "observed_betweenness": round(obs, 5),
        "rank_among_all_nodes": f"{obs_rank}/{G.number_of_nodes()}",
        "null_mean": round(mean, 5), "null_std": round(std, 5),
        "z_score": round(z, 2),
        "verdict": ("REAL BOTTLENECK (z>3 vs layer-preserving null)" if z > 3
                    else "MILD" if z > 1 else "NOT A BOTTLENECK (within null)"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", required=True, type=Path)
    ap.add_argument("--substrate-layer", type=int, default=24)
    ap.add_argument("--gate-layer", type=int, default=32)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-null", type=int, default=200)
    args = ap.parse_args()

    raw_a = load_raw(args.field_dir / f"L{args.substrate_layer}" / "edges.json")
    raw_b = load_raw(args.field_dir / f"L{args.gate_layer}" / "edges.json")
    Ga = collapsed_dir_graph(raw_a)
    Gb = collapsed_dir_graph(raw_b)
    # union directed graph for basin/betweenness over all routes
    Gu = nx.DiGraph()
    for G in (Ga, Gb):
        for u, v, d in G.edges(data=True):
            if Gu.has_edge(u, v):
                Gu[u][v]["abs_weight"] = max(Gu[u][v]["abs_weight"], d["abs_weight"])
            else:
                Gu.add_edge(u, v, abs_weight=d["abs_weight"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # (2) pivot dossier
    dossier = pivot_dossier(raw_a, raw_b)
    (args.out_dir / "pivot_dossier.json").write_text(json.dumps(dossier, indent=2))
    print("=== PIVOT DOSSIER L18/N7417 (position-preserving) ===")
    for fld in ("L24_substrate", "L32_gate"):
        d = dossier[fld]
        tgts = Counter(r["basin"] for r in d["outgoing"])
        print(f"  {fld}: incoming positions {d['incoming_positions']}, "
              f"in-weight {d['total_in_weight']}, outgoing basins {dict(tgts)}")

    # (3) terminal basins per field
    basins_a = terminal_basins(Ga)
    basins_b = terminal_basins(Gb)
    basins_u = terminal_basins(Gu)
    basin_summary = {
        "substrate_sinks": [f"L{l}/N{n}" for (l, n) in SUBSTRATE_SINKS],
        "gate_sinks": [f"L{l}/N{n}" for (l, n) in GATE_SINKS],
        "L24_field_basin_counts": dict(Counter(basins_a.values())),
        "L32_field_basin_counts": dict(Counter(basins_b.values())),
        "union_basin_counts": dict(Counter(basins_u.values())),
        "pivot_basin": {
            "L24": basins_a.get(PIVOT), "L32": basins_b.get(PIVOT),
            "union": basins_u.get(PIVOT),
        },
        "union_node_basins": {f"L{l}/N{n}": basins_u[(l, n)] for (l, n) in sorted(basins_u)},
    }
    (args.out_dir / "terminal_basins.json").write_text(json.dumps(basin_summary, indent=2))
    print("\n=== TERMINAL BASINS ===")
    print(f"  L24 field: {basin_summary['L24_field_basin_counts']}")
    print(f"  L32 field: {basin_summary['L32_field_basin_counts']}")
    print(f"  union:     {basin_summary['union_basin_counts']}")
    print(f"  pivot basin: {basin_summary['pivot_basin']}")

    # (3b) directed betweenness of pivot vs layer-preserving null
    bet = directed_betweenness_pivot(Gu, n_null=args.n_null)
    (args.out_dir / "pivot_betweenness.json").write_text(json.dumps(bet, indent=2))
    print("\n=== PIVOT DIRECTED BETWEENNESS (vs layer-preserving null) ===")
    print(f"  betweenness={bet['observed_betweenness']} rank {bet['rank_among_all_nodes']}  "
          f"z={bet['z_score']}  -> {bet['verdict']}")

    print(f"\nWrote outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
