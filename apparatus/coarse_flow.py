"""Apparatus 6 coarse-graining: is the differential flow graph a compressible
'instrument', or a pretty knot? (Codex's structural plan, topology before semantics.)

The hypothesis (sharpened with Codex): the readout-selected edge graph admits a
meaningful coarse-graining into STABLE ROUTING UNITS — subgraphs whose internal
edges persist, whose boundary edges switch by readout. This is closer to Freud's
Bahnung (facilitated pathway) than to a neuron-label ontology: the candidate
primitive is the route fragment, not the neuron and not the bag-of-neurons.

This script runs the topological tests that must pass BEFORE any semantic / SAE
mapping is worth attempting:

1. Community structure: Louvain communities on union / shared / L24-only / L32-only.
2. Null comparison: modularity vs degree-preserving (configuration-model) nulls.
   If clustering survives the null, it's structure; if not, it's layout theater.
3. Quotient preservation: contract communities to supernodes; does the quotient
   preserve the big fact (shared backbone -> L18 pivot -> L20 substrate / L29 gate)?
4. Readout differential at quotient level: quotient(L32) - quotient(L24).
5. (Deferred) semantic anchoring to SAE features — NOT done here.

Communities are computed ONCE on the union graph and reused for both fields, so
the per-field quotient flows and their differential are over the same supernode
partition (otherwise the differential isn't comparable).

Outputs:
    <out>/communities.json          partition + per-community member table
    <out>/modularity_null.json      observed modularity vs null distribution
    <out>/quotient_summary.json     supernodes + signed inter-community flow per field + readout specificity
    <out>/quotient_flow.png         coarse flow graph (union, supernodes)
    <out>/quotient_differential.png quotient(L32) - quotient(L24)

Usage:
    python -m apparatus.coarse_flow \\
        --field-dir apparatus/output/probe_edge_fields_<run>/ \\
        --out-dir apparatus/output/coarse_flow_<date>/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import FancyArrowPatch
import numpy as np
import networkx as nx

SUPER_WEIGHTS = {(0, 491), (0, 8268), (1, 198), (1, 2427)}


def load_collapsed(edges_path: Path, drop=SUPER_WEIGHTS) -> dict:
    raw = json.loads(edges_path.read_text())["edges"]
    agg: dict = defaultdict(float)
    for e in raw:
        sl, _, sn = e["source"]; tl, _, tn = e["target"]
        if (sl, sn) in drop or (tl, tn) in drop:
            continue
        agg[((sl, sn), (tl, tn))] += e["weight"]
    return dict(agg)


def union_graph(field_a: dict, field_b: dict) -> nx.Graph:
    """Undirected weighted graph on the union of edges; weight = max abs across
    the two fields (so a route strong in either field is a strong tie for
    community detection)."""
    G = nx.Graph()
    keys = set(field_a) | set(field_b)
    for (src, dst) in keys:
        w = max(abs(field_a.get((src, dst), 0.0)), abs(field_b.get((src, dst), 0.0)))
        if src == dst:
            continue
        if G.has_edge(src, dst):
            G[src][dst]["weight"] = max(G[src][dst]["weight"], w)
        else:
            G.add_edge(src, dst, weight=w)
    return G


def louvain(G: nx.Graph, seed: int = 0):
    parts = nx.community.louvain_communities(G, weight="weight", seed=seed)
    mod = nx.community.modularity(G, parts, weight="weight")
    node2comm = {}
    for ci, comm in enumerate(parts):
        for n in comm:
            node2comm[n] = ci
    return parts, mod, node2comm


def null_modularity(G: nx.Graph, n_null: int = 200, seed: int = 0):
    """Degree-preserving configuration-model null: rewire, recompute best
    modularity. Reports observed vs null mean/std and a z-score."""
    rng = np.random.default_rng(seed)
    degseq = [d for _, d in G.degree()]
    obs_parts, obs_mod, _ = louvain(G, seed=seed)
    null_mods = []
    for i in range(n_null):
        # configuration_model can make multigraph + self-loops; simplify.
        cm = nx.configuration_model(degseq, seed=int(rng.integers(0, 2**31)))
        cm = nx.Graph(cm)
        cm.remove_edges_from(nx.selfloop_edges(cm))
        if cm.number_of_edges() == 0:
            continue
        try:
            parts = nx.community.louvain_communities(cm, seed=i)
            null_mods.append(nx.community.modularity(cm, parts))
        except Exception:
            continue
    null_mods = np.array(null_mods, dtype=float)
    mean, std = float(null_mods.mean()), float(null_mods.std())
    z = (obs_mod - mean) / std if std > 1e-9 else float("inf")
    return {
        "observed_modularity": obs_mod,
        "n_communities": len(obs_parts),
        "null_n": int(len(null_mods)),
        "null_mean": mean,
        "null_std": std,
        "z_score": z,
        "verdict": ("STRUCTURE (survives null)" if z > 3
                    else "WEAK/AMBIGUOUS" if z > 1
                    else "LAYOUT THEATER (within null)"),
    }


def layer_preserving_null(field_a: dict, field_b: dict, n_null: int = 200, seed: int = 0):
    """Stronger null for a feed-forward layered DAG (Codex): preserve the
    layer-pair edge-count structure. We rewire targets WITHIN each (src_layer,
    dst_layer) bucket — i.e. keep how many edges go from layer s to layer t, and
    keep each endpoint's layer, but scramble which specific neurons connect.

    This is the honest baseline: a layered DAG is constrained to be feed-forward
    with a given inter-layer traffic profile, so the question is whether
    community structure exceeds what that constraint alone produces.
    """
    rng = np.random.default_rng(seed)
    # union edge set, with the max-abs weight (same tie definition as union_graph)
    keys = set(field_a) | set(field_b)
    edges = []
    nodes_by_layer = defaultdict(set)
    for (src, dst) in keys:
        if src == dst:
            continue
        w = max(abs(field_a.get((src, dst), 0.0)), abs(field_b.get((src, dst), 0.0)))
        edges.append((src, dst, w))
        nodes_by_layer[src[0]].add(src)
        nodes_by_layer[dst[0]].add(dst)
    nodes_by_layer = {l: sorted(s) for l, s in nodes_by_layer.items()}

    # observed modularity (same Louvain as elsewhere)
    G_obs = union_graph(field_a, field_b)
    _, obs_mod, _ = louvain(G_obs, seed=seed)

    null_mods = []
    for it in range(n_null):
        H = nx.Graph()
        for (src, dst, w) in edges:
            sl, dl = src[0], dst[0]
            # resample endpoints within the same layers (preserves layer-pair count)
            new_src = nodes_by_layer[sl][rng.integers(len(nodes_by_layer[sl]))]
            new_dst = nodes_by_layer[dl][rng.integers(len(nodes_by_layer[dl]))]
            if new_src == new_dst:
                continue
            if H.has_edge(new_src, new_dst):
                H[new_src][new_dst]["weight"] = max(H[new_src][new_dst]["weight"], w)
            else:
                H.add_edge(new_src, new_dst, weight=w)
        if H.number_of_edges() == 0:
            continue
        try:
            parts = nx.community.louvain_communities(H, weight="weight", seed=it)
            null_mods.append(nx.community.modularity(H, parts, weight="weight"))
        except Exception:
            continue
    null_mods = np.array(null_mods, dtype=float)
    mean, std = float(null_mods.mean()), float(null_mods.std())
    z = (obs_mod - mean) / std if std > 1e-9 else float("inf")
    return {
        "null_type": "layer-preserving (inter-layer edge-count preserved)",
        "observed_modularity": obs_mod,
        "null_n": int(len(null_mods)),
        "null_mean": mean,
        "null_std": std,
        "z_score": z,
        "verdict": ("STRUCTURE (survives layer-preserving null)" if z > 3
                    else "WEAK/AMBIGUOUS" if z > 1
                    else "NO STRUCTURE (within layer-preserving null)"),
    }


def signed_quotient(field: dict, node2comm: dict) -> dict:
    """Contract a field to inter-community signed flow.

    Returns {(ci,cj): {"weight": signed_sum, "abs": abs_sum, "n": count}} for
    ci != cj (inter-community edges = the boundary routes that carry the flow).
    """
    q: dict = defaultdict(lambda: {"weight": 0.0, "abs": 0.0, "n": 0})
    for (src, dst), w in field.items():
        ci, cj = node2comm.get(src), node2comm.get(dst)
        if ci is None or cj is None or ci == cj:
            continue
        rec = q[(ci, cj)]
        rec["weight"] += w; rec["abs"] += abs(w); rec["n"] += 1
    return dict(q)


def community_layer_span(comm: set) -> tuple:
    layers = [l for (l, _) in comm]
    return (min(layers), max(layers))


def readout_specificity(ci, cj, qa, qb) -> str:
    in_a = (ci, cj) in qa
    in_b = (ci, cj) in qb
    if in_a and in_b:
        wa, wb = qa[(ci, cj)]["weight"], qb[(ci, cj)]["weight"]
        return "switch" if (wa >= 0) != (wb >= 0) else "shared"
    return "gate" if in_b else "substrate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", required=True, type=Path)
    ap.add_argument("--substrate-layer", type=int, default=24)
    ap.add_argument("--gate-layer", type=int, default=32)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = load_collapsed(args.field_dir / f"L{args.substrate_layer}" / "edges.json")
    b = load_collapsed(args.field_dir / f"L{args.gate_layer}" / "edges.json")
    G = union_graph(a, b)
    print(f"Union graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Test 1: communities on union ---
    parts, mod, node2comm = louvain(G, seed=args.seed)
    print(f"Louvain (union): {len(parts)} communities, modularity={mod:.4f}")

    # --- Test 2: null comparison (degree-preserving AND layer-preserving) ---
    null = null_modularity(G, n_null=args.n_null, seed=args.seed)
    print(f"Null (degree-preserving): observed={null['observed_modularity']:.4f} vs "
          f"null {null['null_mean']:.4f}±{null['null_std']:.4f}  "
          f"z={null['z_score']:.2f}  -> {null['verdict']}")
    null_lp = layer_preserving_null(a, b, n_null=args.n_null, seed=args.seed)
    print(f"Null (layer-preserving):  observed={null_lp['observed_modularity']:.4f} vs "
          f"null {null_lp['null_mean']:.4f}±{null_lp['null_std']:.4f}  "
          f"z={null_lp['z_score']:.2f}  -> {null_lp['verdict']}")

    # --- Test 3+4: quotient per field + differential ---
    qa = signed_quotient(a, node2comm)
    qb = signed_quotient(b, node2comm)

    # community member table
    communities = []
    for ci, comm in enumerate(parts):
        members = sorted(comm)
        lo, hi = community_layer_span(comm)
        # internal edge weight (abs) within this community in the union
        internal = sum(G[u][v]["weight"] for u, v in G.edges()
                       if node2comm[u] == ci and node2comm[v] == ci)
        communities.append({
            "community": ci,
            "n_members": len(members),
            "layer_span": [lo, hi],
            "members": [f"L{l}/N{n}" for (l, n) in members],
            "internal_abs_weight": round(internal, 4),
        })

    # quotient differential: union of community-pairs, delta = wb - wa
    pair_keys = set(qa) | set(qb)
    quotient_edges = []
    for (ci, cj) in pair_keys:
        wa = qa.get((ci, cj), {}).get("weight", 0.0)
        wb = qb.get((ci, cj), {}).get("weight", 0.0)
        quotient_edges.append({
            "src_comm": ci, "dst_comm": cj,
            "w_substrate": round(wa, 4), "w_gate": round(wb, 4),
            "delta": round(wb - wa, 4),
            "specificity": readout_specificity(ci, cj, qa, qb),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "communities.json").write_text(json.dumps({
        "n_communities": len(parts),
        "modularity_union": mod,
        "communities": communities,
    }, indent=2))
    (args.out_dir / "modularity_null.json").write_text(
        json.dumps({"degree_preserving": null, "layer_preserving": null_lp}, indent=2))
    (args.out_dir / "quotient_summary.json").write_text(json.dumps({
        "n_communities": len(parts),
        "n_quotient_edges": len(quotient_edges),
        "specificity_counts": dict(Counter(q["specificity"] for q in quotient_edges)),
        "quotient_edges": sorted(quotient_edges, key=lambda d: -abs(d["delta"])),
    }, indent=2))

    # --- plots: quotient flow (union, mean layer of community as y) ---
    comm_layer = {ci: float(np.mean([l for (l, _) in comm]))
                  for ci, comm in enumerate(parts)}
    comm_size = {ci: len(comm) for ci, comm in enumerate(parts)}
    _plot_quotient(args.out_dir / "quotient_flow.png", comm_layer, comm_size, qb, qa,
                   mode="union",
                   title=f"Quotient flow ({len(parts)} communities, mod={mod:.3f}, {null['verdict']})")
    _plot_quotient(args.out_dir / "quotient_differential.png", comm_layer, comm_size,
                   qb, qa, mode="differential",
                   title=f"Quotient differential: flow(L{args.gate_layer}) - flow(L{args.substrate_layer})")

    print(f"\nWrote outputs to {args.out_dir}")
    print(f"quotient specificity: {dict(Counter(q['specificity'] for q in quotient_edges))}")
    # surface the pivot communities (which community holds L18/N7417?)
    pivot = node2comm.get((18, 7417))
    print(f"L18/N7417 is in community {pivot} "
          f"(layer span {communities[pivot]['layer_span'] if pivot is not None else 'n/a'})" if pivot is not None
          else "L18/N7417 not in graph")
    return 0


def _plot_quotient(out_path, comm_layer, comm_size, qb, qa, mode, title):
    fig, ax = plt.subplots(figsize=(10, 9))
    # positions: y = mean layer, x = spread to reduce overlap (deterministic)
    cis = sorted(comm_layer)
    # order by layer then index for x within same layer band
    xs = {}
    by_band = defaultdict(list)
    for ci in cis:
        by_band[round(comm_layer[ci])].append(ci)
    for band, members in by_band.items():
        m = len(members)
        for j, ci in enumerate(sorted(members)):
            xs[ci] = 0.15 + 0.7 * (j + 0.5) / m
    pos = {ci: (xs[ci], comm_layer[ci]) for ci in cis}

    if mode == "union":
        edges = [(k[0], k[1], v["weight"]) for k, v in {**qa, **qb}.items()]
        cmap = matplotlib.colormaps["RdBu"]
        wmax = max((abs(w) for _, _, w in edges), default=1.0)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-wmax, vcenter=0, vmax=wmax)
        for ci, cj, w in edges:
            color = cmap(norm(w))
            _arrow(ax, pos[ci], pos[cj], color, 0.6 + 3.5 * abs(w) / wmax)
    else:  # differential
        keys = set(qa) | set(qb)
        deltas = {(ci, cj): qb.get((ci, cj), {}).get("weight", 0) - qa.get((ci, cj), {}).get("weight", 0)
                  for (ci, cj) in keys}
        dmax = max((abs(d) for d in deltas.values()), default=1.0)
        cmap = matplotlib.colormaps["RdBu"]
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
        for (ci, cj), d in deltas.items():
            _arrow(ax, pos[ci], pos[cj], cmap(norm(d)), 0.6 + 3.5 * abs(d) / dmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("delta = w(gate) - w(substrate)\nblue: gate-intensifies  red: substrate", fontsize=7.5)

    for ci, (x, y) in pos.items():
        ax.scatter([x], [y], s=80 + 30 * comm_size[ci], c="#444",
                   edgecolor="black", linewidth=0.6, zorder=5, alpha=0.85)
        ax.text(x, y, str(ci), color="white", ha="center", va="center",
                fontsize=8, zorder=6, weight="bold")

    ax.set_ylabel("community mean layer (substrate -> gate)", fontsize=9)
    ax.set_xticks([]); ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.12, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _arrow(ax, p0, p1, color, lw):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle="arc3,rad=0.12", arrowstyle="-|>",
        mutation_scale=14, linewidth=lw, color=color, alpha=0.6, zorder=2))


if __name__ == "__main__":
    raise SystemExit(main())
