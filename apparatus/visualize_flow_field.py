"""Static token-I edge-field viewer (Apparatus 6 Phase A).

This is NOT "the flow field" in the full intended sense. It visualizes the
*existing* token-I RelP edge topology (edges.json) only. It can show how the
original token-readout circuit routes signed contribution through MLP neurons
up to L31. It CANNOT show L24-probe or L32-probe flow (those edge fields don't
exist yet — they require re-running RelP under those probe readouts), and it
has no L32 node because L32 is post-final-norm residual, not an MLP layer.

Per Codex's rigor adjustments, the run produces multiple views rather than one
canonical PNG, because the edge-weight distribution is pathological (the top
edges are L0->L1 infrastructure super-weights ~2-3 orders above the rest).
Drawing them raw makes the plot useless; silently dropping them makes it
misleading. So we draw:

  flow_field_tokenI_full.png          full graph, rank-scaled thickness
  flow_field_tokenI_no_infra.png      super-weight edges/nodes removed
  flow_field_tokenI_top_edges.png     percentile-thresholded (infra-contaminated)
  flow_field_tokenI_top_no_infra.png  PRIMARY: top edges AFTER removing infra
  flow_field_tokenI_summary.json      counts, sign counts, weight quantiles,
                                      cancellation-ratio diagnostic, top layer-
                                      pairs (with/without L31 targets), per-view
                                      filter counts

Layout (layered-DAG / flow-field style): y = LAYER, rank-spaced so every
populated layer gets equal vertical room (the raw distribution is lopsided —
the action is L20-L31, L0-L18 nearly empty — so linear-y wastes the canvas).
Token position is COLLAPSED (single-token circuit). Within a layer, nodes are
ordered left-to-right by role; x is a within-layer slot, NOT a global metric
axis, and is explicitly RESERVED for token position in the multitoken case.
Edges are DIRECTED curved arrows source->target (RelP edges are directed;
almost all run substrate->gate, i.e. upward — the summary reports the
up/down/flat split). Node color = role (shared with analyze.py's assign_role;
we do NOT trust row["role"], which is None for all rows). Edge color = SIGN of
the net signed contribution to the token-I readout (red negative, blue
positive) — "suppressive/excitatory" only as interpretive shorthand; the
quantity is mathematically a signed RelP edge weight, not a biological synapse.

Usage:
    python -m apparatus.visualize_flow_field \\
        --in-edges experiments/.../relp-behavioral_refusal_kstar91/edges.json \\
        --in-roles apparatus/output/role_table_refusal_8b_fullcircuit_20260526.jsonl \\
        --out-dir apparatus/output/flow_field_tokenI_<date>/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Single source of truth for role assignment + collapse. Importing rather than
# duplicating avoids the drift that visualize.py's copied collapse() risks.
from apparatus.analyze import load_rows, collapse_to_layer_neuron, assign_role


# Role -> color/marker. Mirrors the role-space scatter's grammar where it
# overlaps; adds the asymmetric-suppressor subtypes the scatter folds together.
ROLE_STYLE: dict[str, tuple[str, str]] = {
    "infrastructure":         ("#999999", "^"),
    "reader-writer":          ("#2a6b3a", "o"),
    "reader-only":            ("#1f77b4", "o"),
    "writer-only":            ("#9a7a00", "o"),
    "suppressor-consistent":  ("#a83232", "s"),
    "suppressor-transplant":  ("#d62728", "s"),
    "suppressor-ablation":    ("#e8888a", "s"),
    "mixed":                  ("#bbbbbb", "."),
    "unclassified":           ("#dddddd", "."),
}
ROLE_ORDER = list(ROLE_STYLE.keys())

# Within-layer left-to-right ordering of nodes by role. x is NOT a global metric
# axis (layer is the only quantitative axis, on y) — it's a within-layer slot,
# ordered by role so reader/writer/suppressor read left-to-right inside a row.
# This slot is explicitly RESERVED for token position in the multitoken case;
# at single-token every node occupies one slot. Role -> color is the real role
# encoding; the x-ordering is a legibility aid, not a second metric.
ROLE_RANK: dict[str, int] = {
    "infrastructure":        0,
    "reader-only":           1,
    "reader-writer":         2,
    "writer-only":           3,
    "suppressor-consistent": 4,
    "suppressor-transplant": 5,
    "suppressor-ablation":   6,
    "mixed":                 7,
    "unclassified":          8,
}

POS_EDGE_COLOR = "#1f4fb4"   # positive signed contribution to token-I readout
NEG_EDGE_COLOR = "#c43030"   # negative


def load_edges(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    return d["edges"]


def collapse_edges(edges: list[dict]) -> list[dict]:
    """Collapse position-keyed edges to (layer,neuron)->(layer,neuron).

    Single-token circuit, so token position is degenerate. Summing signed
    weight across the parallel position-edges preserves total signed flow
    between each neuron pair. Also tracks abs-weight sum (for thickness) and a
    cancellation_ratio = |sum(w)| / sum(|w|): near 1.0 means the position-edges
    agree in sign (collapse safe); near 0.0 means large positive and negative
    position contributions canceled (collapse is hiding position conflict, and
    the drawn edge's color reflects only a small net sign).
    """
    agg: dict[tuple, dict] = {}
    for e in edges:
        sl, _, sn = e["source"]
        tl, _, tn = e["target"]
        key = (sl, sn, tl, tn)
        rec = agg.setdefault(key, {"src": (sl, sn), "dst": (tl, tn),
                                   "weight": 0.0, "abs_weight": 0.0, "n": 0})
        rec["weight"] += e["weight"]
        rec["abs_weight"] += abs(e["weight"])
        rec["n"] += 1
    for rec in agg.values():
        rec["cancellation_ratio"] = (
            abs(rec["weight"]) / rec["abs_weight"] if rec["abs_weight"] > 0 else 1.0
        )
    return list(agg.values())


def _barycenter_order(by_layer: dict, cedges: list[dict], role_map,
                      n_sweeps: int = 12):
    """Crossing-minimization via the barycenter heuristic (Sugiyama step 2).

    Within each layer, repeatedly move each node to the average x-rank of its
    connected neighbors, then re-rank. Up-and-down sweeps converge nodes that
    talk to each other into vertical alignment, which untangles edge crossings.
    x stays non-metric — this only picks the *ordering* that crosses least.

    Seeded by role rank then neuron id so the result is deterministic. Returns
    {layer: [ordered (layer,neuron) keys]}.
    """
    # Neighbor adjacency (undirected for ordering purposes).
    nbrs: dict[tuple, list[tuple]] = defaultdict(list)
    for e in cedges:
        nbrs[e["src"]].append(e["dst"])
        nbrs[e["dst"]].append(e["src"])

    # Deterministic seed order within each layer.
    order: dict[int, list[tuple]] = {}
    for layer, members in by_layer.items():
        order[layer] = sorted(
            members,
            key=lambda k: (ROLE_RANK.get(role_map.get(k, "unclassified"), 8), k[1]))

    layers = sorted(by_layer)

    def positions():
        # Fractional position in [0,1] of each node within its layer's order.
        pos = {}
        for layer in layers:
            mem = order[layer]
            m = len(mem)
            for j, k in enumerate(mem):
                pos[k] = (j + 0.5) / m
        return pos

    for sweep in range(n_sweeps):
        pos = positions()
        seq = layers if sweep % 2 == 0 else list(reversed(layers))
        for layer in seq:
            mem = order[layer]
            def bary(k):
                ns = nbrs.get(k, [])
                if not ns:
                    return pos[k]
                return sum(pos[n] for n in ns) / len(ns)
            # Stable sort by barycenter; ties keep prior order.
            order[layer] = sorted(mem, key=bary)
            # Refresh positions for this layer so the next layer sees the update.
            m = len(order[layer])
            for j, k in enumerate(order[layer]):
                pos[k] = (j + 0.5) / m
    return order


def build_node_index(cedges: list[dict], role_map: dict[tuple[int, int], str]):
    """Assign each (layer,neuron) node a rank-spaced y and a within-layer x slot.

    y: populated layers are rank-spaced (equal vertical gaps) rather than placed
       at their raw layer index. The raw distribution is lopsided — the action is
       L20-L31 and L0-L18 is nearly empty — so linear-y wastes most of the canvas.
       Rank-spacing gives every populated layer equal room; the gate band still
       reads because the populated late layers (L28-L31) sit at the top.
    x: within-layer ordering is chosen by the barycenter crossing-minimization
       heuristic (Sugiyama), so connected nodes align vertically and edges
       untangle. x is a within-layer slot, NOT a global metric axis. (Reserved
       for token position in multitoken.) Role is encoded by color, not x.

    Returns (nodes, layer_to_y) where layer_to_y maps raw layer -> plotted y.
    """
    # Group circuit nodes by layer.
    by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for e in cedges:
        for (layer, neuron) in (e["src"], e["dst"]):
            if (layer, neuron) not in seen:
                seen.add((layer, neuron))
                by_layer[layer].append((layer, neuron))

    populated = sorted(by_layer)
    n_layers = len(populated)
    # Equal vertical gaps; raw layer order preserved (substrate low -> gate high).
    layer_to_y = {l: (i / max(n_layers - 1, 1)) * (n_layers - 1)
                  for i, l in enumerate(populated)}

    order = _barycenter_order(by_layer, cedges, role_map)

    nodes: dict[tuple[int, int], dict] = {}
    for layer in populated:
        members = order[layer]
        m = len(members)
        for j, key in enumerate(members):
            # Spread across a [0.1, 0.9] band so single-member layers center-ish.
            x = 0.1 + (0.8 * (j + 0.5) / m)
            nodes[key] = {
                "layer": layer,
                "neuron": key[1],
                "role": role_map.get(key, "unclassified"),
                "x": x,
                "y": layer_to_y[layer],
            }
    return nodes, layer_to_y


def edge_thickness_rank(weights_abs: np.ndarray, lo: float = 0.3, hi: float = 4.0) -> np.ndarray:
    """Rank-scale absolute weights to a sane linewidth band.

    Rank (not log) so the pathological 3200-vs-0.01 spread can't blow out the
    plot, and so equal visual steps mean equal rank steps. Empty-safe.
    """
    n = len(weights_abs)
    if n == 0:
        return weights_abs
    order = weights_abs.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    if n > 1:
        ranks /= (n - 1)  # [0, 1]
    return lo + (hi - lo) * ranks


def draw_view(ax, cedges: list[dict], nodes: dict, layer_to_y: dict, title: str,
              gate_band=(29, 31)):
    """Draw one directed flow field. Returns dict of render stats.

    Layered-DAG style: y = rank-spaced layer, x = within-layer role-ordered slot.
    Edges drawn as directed curved arrows source->target (RelP edges are directed;
    almost all run substrate->gate, i.e. upward). Color = sign of net signed
    contribution to the token-I readout; thickness = rank-scaled |weight|.
    """
    from matplotlib.patches import FancyArrowPatch

    if cedges:
        wabs = np.array([e["abs_weight"] for e in cedges], dtype=float)
        lw = edge_thickness_rank(wabs)
    else:
        lw = np.array([])

    # Gate-band shading (in rank-spaced y coords).
    gate_ys = [layer_to_y[l] for l in layer_to_y if gate_band[0] <= l <= gate_band[1]]
    if gate_ys:
        ax.axhspan(min(gate_ys) - 0.5, max(gate_ys) + 0.5,
                   color="#ffe9b0", alpha=0.4, zorder=0)
        ax.text(0.01, max(gate_ys) + 0.35, "gate band (L29-L31)",
                fontsize=8, color="#9a7a00", va="bottom")

    n_pos = n_neg = n_up = n_down = n_flat = 0
    for i, e in enumerate(cedges):
        skey, tkey = e["src"], e["dst"]
        if skey not in nodes or tkey not in nodes:
            continue
        x0, y0 = nodes[skey]["x"], nodes[skey]["y"]
        x1, y1 = nodes[tkey]["x"], nodes[tkey]["y"]
        if e["weight"] >= 0:
            color = POS_EDGE_COLOR; n_pos += 1
        else:
            color = NEG_EDGE_COLOR; n_neg += 1
        if y1 > y0:
            n_up += 1
        elif y1 < y0:
            n_down += 1
        else:
            n_flat += 1
        # Curvature so parallel/antiparallel edges between nearby nodes separate;
        # arrowhead at the target marks direction of contribution flow.
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle="arc3,rad=0.10",
            arrowstyle="-|>", mutation_scale=15,
            linewidth=lw[i], color=color, alpha=0.5, zorder=1,
            shrinkA=4, shrinkB=6,
        )
        ax.add_patch(arrow)

    # Group nodes by role for scatter + legend.
    by_role: dict[str, list] = defaultdict(list)
    for nd in nodes.values():
        by_role[nd["role"]].append(nd)
    for role in ROLE_ORDER:
        nds = by_role.get(role, [])
        if not nds:
            continue
        color, marker = ROLE_STYLE[role]
        xs = [nd["x"] for nd in nds]
        ys = [nd["y"] for nd in nds]
        ax.scatter(xs, ys, c=color, marker=marker, s=54,
                   edgecolor="black", linewidth=0.4, alpha=0.92, zorder=3)

    # y ticks: rank-spaced positions labeled with raw layer index.
    ys_sorted = sorted(layer_to_y.items())  # (layer, y)
    ax.set_yticks([y for _, y in ys_sorted])
    ax.set_yticklabels([f"L{l}" for l, _ in ys_sorted], fontsize=7.5)
    ax.set_ylim(-0.7, (len(layer_to_y) - 1) + 0.7)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("within-layer ordering, barycenter layout; x is non-metric "
                  "(reserved for token position in multitoken)", fontsize=8.5)
    ax.set_ylabel("layer, rank-spaced (substrate -> gate, bottom to top)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.10, axis="y")
    return {"n_pos_edges_drawn": n_pos, "n_neg_edges_drawn": n_neg,
            "n_edges_upward": n_up, "n_edges_downward": n_down,
            "n_edges_same_layer": n_flat}


def role_legend_handles():
    handles = []
    for role in ROLE_ORDER:
        color, marker = ROLE_STYLE[role]
        handles.append(Line2D([0], [0], marker=marker, color="none",
                              markerfacecolor=color, markeredgecolor="black",
                              markersize=8, label=role))
    handles.append(Line2D([0], [0], color=POS_EDGE_COLOR, lw=2.5,
                           label="edge: + contribution to token-I"))
    handles.append(Line2D([0], [0], color=NEG_EDGE_COLOR, lw=2.5,
                           label="edge: - contribution to token-I"))
    return handles


def quantiles(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return {}
    return {
        "n": int(len(vals)),
        "min": float(vals.min()), "max": float(vals.max()),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p99": float(np.percentile(vals, 99)),
    }


def top_layer_pairs(cedges: list[dict], k: int = 12, exclude_l31_targets: bool = False):
    by_count: Counter = Counter()
    by_absw: defaultdict = defaultdict(float)
    for e in cedges:
        if exclude_l31_targets and e["dst"][0] == 31:
            continue
        pair = (e["src"][0], e["dst"][0])
        by_count[pair] += 1
        by_absw[pair] += e["abs_weight"]
    top_c = [{"src_layer": p[0], "dst_layer": p[1], "count": c}
             for p, c in by_count.most_common(k)]
    top_w = sorted(({"src_layer": p[0], "dst_layer": p[1], "abs_weight": w}
                    for p, w in by_absw.items()),
                   key=lambda d: -d["abs_weight"])[:k]
    return top_c, top_w


def render_one(out_path: Path, cedges: list[dict], role_map, title: str):
    nodes, layer_to_y = build_node_index(cedges, role_map)
    fig, ax = plt.subplots(figsize=(12, 10))
    stats = draw_view(ax, cedges, nodes, layer_to_y, title)
    ax.legend(handles=role_legend_handles(), loc="upper left",
              fontsize=7.5, framealpha=0.9, ncol=1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    stats["n_nodes"] = len(nodes)
    stats["n_edges_in_view"] = len(cedges)
    print(f"wrote {out_path}  (edges={len(cedges)} nodes={len(nodes)})")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-edges", required=True, type=Path)
    ap.add_argument("--in-roles", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--top-percentile", type=float, default=90.0,
                    help="for the top-edges view: keep edges above this abs-weight percentile")
    args = ap.parse_args()

    edges = load_edges(args.in_edges)
    rows = load_rows(args.in_roles)
    coll = collapse_to_layer_neuron(rows)
    role_map = {(r["layer"], r["neuron"]): assign_role(r) for r in coll}

    # Collapse position-keyed edges to (layer,neuron) pairs (single-token
    # circuit; position degenerate).
    cedges = collapse_edges(edges)

    # Identify infrastructure (super-weight) nodes to build the no-infra view.
    infra_nodes = {(r["layer"], r["neuron"]) for r in coll if r.get("is_super_weight")}

    def touches_infra(e):
        return e["src"] in infra_nodes or e["dst"] in infra_nodes

    cedges_no_infra = [e for e in cedges if not touches_infra(e)]

    wabs_all = np.array([e["abs_weight"] for e in cedges], dtype=float)
    thresh = float(np.percentile(wabs_all, args.top_percentile))
    cedges_top = [e for e in cedges if e["abs_weight"] >= thresh]

    # PRIMARY view: top edges computed AFTER removing infrastructure, so the
    # legible panel isn't dominated by the L0/L1 super-weight scaffold.
    wabs_no_infra = np.array([e["abs_weight"] for e in cedges_no_infra], dtype=float)
    thresh_ni = float(np.percentile(wabs_no_infra, args.top_percentile))
    cedges_top_no_infra = [e for e in cedges_no_infra if e["abs_weight"] >= thresh_ni]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    stats["full"] = render_one(
        args.out_dir / "flow_field_tokenI_full.png", cedges, role_map,
        "Token-I edge field (FULL, rank-scaled thickness) — Llama-3.1-8B refusal")
    stats["no_infra"] = render_one(
        args.out_dir / "flow_field_tokenI_no_infra.png", cedges_no_infra, role_map,
        "Token-I edge field (super-weight infrastructure removed)")
    stats["top_edges"] = render_one(
        args.out_dir / "flow_field_tokenI_top_edges.png", cedges_top, role_map,
        f"Token-I edge field (|weight| >= p{args.top_percentile:g}, infra-contaminated)")
    stats["top_no_infra"] = render_one(
        args.out_dir / "flow_field_tokenI_top_no_infra.png", cedges_top_no_infra, role_map,
        f"PRIMARY: token-I refusal flow (no infra, |weight| >= p{args.top_percentile:g})")

    # Cancellation-ratio diagnostic: is signed-sum collapse hiding position
    # conflict? Report on the no-infra population (the one we reason from) and
    # specifically on the plotted top-no-infra edges.
    cr_all = np.array([e["cancellation_ratio"] for e in cedges_no_infra], dtype=float)
    cr_top = np.array([e["cancellation_ratio"] for e in cedges_top_no_infra], dtype=float)
    n_low_top = int((cr_top < 0.5).sum())

    top_c, top_w = top_layer_pairs(cedges)
    top_c_ni, top_w_ni = top_layer_pairs(cedges_no_infra)
    # No-L31-target diagnostic: does the substrate->gate story survive when we
    # drop everything ending at the L31 readout hub? If structure persists into
    # L29/L30, it's less likely to be a trivial readout-hub artifact.
    top_c_ni_noL31, top_w_ni_noL31 = top_layer_pairs(
        cedges_no_infra, exclude_l31_targets=True)

    summary = {
        "source_edges_file": str(args.in_edges),
        "source_roles_file": str(args.in_roles),
        "caveat": ("token-I RelP topology only; not probe-conditioned; no L32 "
                   "node (post-final-norm residual, not an MLP layer); position "
                   "collapsed to (layer,neuron) — single-token circuit"),
        "n_edges_raw": len(edges),
        "n_edges_collapsed": len(cedges),
        "n_nodes_total": len({n for e in cedges for n in (e["src"], e["dst"])}),
        "sign_counts_collapsed": {
            "positive": sum(1 for e in cedges if e["weight"] >= 0),
            "negative": sum(1 for e in cedges if e["weight"] < 0),
        },
        "abs_weight_quantiles_collapsed": quantiles(wabs_all),
        "cancellation_ratio_diagnostic": {
            "note": ("|sum(w)|/sum(|w|) per collapsed edge; ~1.0 = position-edges "
                     "agree in sign (collapse safe), ~0.0 = position conflict hidden"),
            "no_infra_quantiles": quantiles(cr_all),
            "top_no_infra_quantiles": quantiles(cr_top),
            "n_top_no_infra_edges_with_ratio_below_0.5": n_low_top,
            "n_top_no_infra_edges_total": len(cedges_top_no_infra),
        },
        "top_layer_pairs_by_count": top_c,
        "top_layer_pairs_by_abs_weight": top_w,
        "top_layer_pairs_no_infra_by_count": top_c_ni,
        "top_layer_pairs_no_infra_by_abs_weight": top_w_ni,
        "readout_hub_diagnostic": {
            "note": ("top layer-pairs after dropping all edges that TARGET L31 "
                     "(the token-I readout hub). If substrate->gate structure "
                     "persists into L29/L30 here, less likely a readout artifact."),
            "no_infra_no_L31_targets_by_count": top_c_ni_noL31,
            "no_infra_no_L31_targets_by_abs_weight": top_w_ni_noL31,
        },
        "role_counts": dict(Counter(role_map.values())),
        "infrastructure_nodes": sorted(f"L{l}/N{n}" for (l, n) in infra_nodes),
        "views": {
            "full": {"n_edges": len(cedges), **stats["full"]},
            "no_infra": {"n_edges": len(cedges_no_infra),
                         "n_edges_filtered_out": len(cedges) - len(cedges_no_infra),
                         **stats["no_infra"]},
            "top_edges": {"percentile": args.top_percentile,
                          "abs_weight_threshold": thresh,
                          "n_edges": len(cedges_top),
                          "n_edges_filtered_out": len(cedges) - len(cedges_top),
                          **stats["top_edges"]},
            "top_no_infra": {"percentile": args.top_percentile,
                             "abs_weight_threshold": thresh_ni,
                             "n_edges": len(cedges_top_no_infra),
                             "primary": True,
                             **stats["top_no_infra"]},
        },
    }
    out_summary = args.out_dir / "flow_field_tokenI_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
