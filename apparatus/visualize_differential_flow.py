"""Apparatus 6 differential flow field: flow(L32 gate) - flow(L24 substrate).

This is the first real flow-*transformation* object (vs the static baseline
field). It takes the two readout-selected edge fields produced by
apparatus/probe_edge_fields.py and renders their difference on the same
layered-DAG layout as the Phase-A viewer.

WHAT THE DIFFERENTIAL MEANS (read carefully):

We have two edge fields, each the neuron->neuron edge topology of the circuit
selected by a different probe readout:
  - L24 substrate probe: "what drives the harm-detector at the substrate band?"
  - L32 gate  probe: "what drives the refusal commitment at the gate/readout?"

discover_edges is readout-agnostic AFTER circuit selection, so this is a
READOUT-SELECTED EDGE-FIELD DIFFERENTIAL. We union the collapsed edge keys
(treating an edge missing from one field as weight 0) and plot:

    delta(edge) = weight_L32 - weight_L24

Reading the delta colormap:
  - strongly POSITIVE  (blue): route intensifies toward the gate, or is
    gate-specific (present at L32, ~0 at L24).
  - strongly NEGATIVE  (red): route falls away toward the gate, or is
    substrate-specific (present at L24, ~0 at L32).
  - near ZERO          (faint): route carries comparable signed flow at both
    readouts (ambient, not part of the transformation).
  - SIGN-FLIP edges (present in both with opposite sign) are the sharpest
    signal — a route that carries signed flow one way at the substrate and the
    opposite way at the gate. These are highlighted.

This is the circuit-wide analogue of the single-neuron L24/N1619 inversion
finding (substrate-positive, gate-negative). NOTE: N1619 itself is a token-I
circuit neuron and need not appear here — the probe-selected circuits are the
neurons that drive the *probe* readouts, a related but distinct population. The
question this answers is whether substrate->gate sign-inversion is a structural
property of the harm-probe circuit, not whether N1619 specifically reappears.

Usage:
    python -m apparatus.visualize_differential_flow \\
        --field-dir apparatus/output/probe_edge_fields_<run>/ \\
        --out-dir apparatus/output/differential_flow_<date>/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np


def load_collapsed_field(edges_path: Path) -> dict:
    """Load an edges.json and collapse position -> (sl,sn,tl,tn) summed signed.

    Returns {(sl,sn,tl,tn): {"weight": signed_sum, "abs_weight": abs_sum,
    "cancellation_ratio": |sum|/sum|}}.
    """
    raw = json.loads(edges_path.read_text())["edges"]
    agg: dict[tuple, dict] = {}
    for e in raw:
        sl, _, sn = e["source"]
        tl, _, tn = e["target"]
        key = (sl, sn, tl, tn)
        rec = agg.setdefault(key, {"weight": 0.0, "abs_weight": 0.0})
        rec["weight"] += e["weight"]
        rec["abs_weight"] += abs(e["weight"])
    for rec in agg.values():
        rec["cancellation_ratio"] = (
            abs(rec["weight"]) / rec["abs_weight"] if rec["abs_weight"] > 0 else 1.0)
    return agg


def load_super_weight_nodes(role_table_path: Path) -> set:
    """Super-weight (layer,neuron) set from the Phase-A role table.

    Same source the static viewer uses for its no_infra view. These L0/L1
    infrastructure neurons carry collapsed edge weights ~3 orders above the rest;
    if left in, they blow out the differential's diverging colormap so every
    interesting edge maps to ~white. Dropped before computing the differential.
    """
    sw = set()
    if not role_table_path.exists():
        return sw
    for line in role_table_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("is_super_weight"):
            sw.add((r["layer"], r["neuron"]))
    return sw


def build_differential(field_a: dict, field_b: dict, drop_nodes: set | None = None) -> list[dict]:
    """delta = weight_B - weight_A over the union of edge keys (missing = 0).

    field_a is the SUBTRAHEND (L24 substrate), field_b the MINUEND (L32 gate),
    so delta>0 = intensifies/gate-specific, delta<0 = falls-away/substrate-specific.
    Edges touching any node in drop_nodes (super-weight infrastructure) are excluded.
    """
    drop_nodes = drop_nodes or set()
    keys = set(field_a) | set(field_b)
    out = []
    for k in keys:
        sl, sn, tl, tn = k
        if (sl, sn) in drop_nodes or (tl, tn) in drop_nodes:
            continue
        wa = field_a.get(k, {}).get("weight", 0.0)
        wb = field_b.get(k, {}).get("weight", 0.0)
        in_a, in_b = k in field_a, k in field_b
        if in_a and in_b:
            kind = "sign_flip" if (wa >= 0) != (wb >= 0) else "shared"
        elif in_b:
            kind = "gate_only"
        else:
            kind = "substrate_only"
        out.append({
            "src": (sl, sn), "dst": (tl, tn),
            "w_substrate": wa, "w_gate": wb,
            "delta": wb - wa, "kind": kind,
        })
    return out


def rank_spaced_layers(diff_edges: list[dict]):
    """Populated layers -> equal-gap y. Returns (layer_to_y, nodes-by-layer)."""
    by_layer: dict[int, list[tuple]] = defaultdict(list)
    seen = set()
    for e in diff_edges:
        for (l, n) in (e["src"], e["dst"]):
            if (l, n) not in seen:
                seen.add((l, n)); by_layer[l].append((l, n))
    populated = sorted(by_layer)
    n = len(populated)
    layer_to_y = {l: (i / max(n - 1, 1)) * (n - 1) for i, l in enumerate(populated)}
    return layer_to_y, by_layer


def barycenter_order(by_layer, diff_edges, n_sweeps: int = 12):
    """Crossing-minimization (same heuristic as the Phase-A viewer)."""
    nbrs = defaultdict(list)
    for e in diff_edges:
        nbrs[e["src"]].append(e["dst"]); nbrs[e["dst"]].append(e["src"])
    order = {l: sorted(m, key=lambda k: k[1]) for l, m in by_layer.items()}
    layers = sorted(by_layer)

    def positions():
        pos = {}
        for l in layers:
            m = order[l]
            for j, k in enumerate(m):
                pos[k] = (j + 0.5) / len(m)
        return pos

    for sweep in range(n_sweeps):
        pos = positions()
        seq = layers if sweep % 2 == 0 else list(reversed(layers))
        for l in seq:
            def bary(k):
                ns = nbrs.get(k, [])
                return sum(pos[x] for x in ns) / len(ns) if ns else pos[k]
            order[l] = sorted(order[l], key=bary)
            for j, k in enumerate(order[l]):
                pos[k] = (j + 0.5) / len(order[l])
    return order


def thickness_rank(vals_abs: np.ndarray, lo=0.4, hi=4.5) -> np.ndarray:
    n = len(vals_abs)
    if n == 0:
        return vals_abs
    order = vals_abs.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    if n > 1:
        ranks /= (n - 1)
    return lo + (hi - lo) * ranks


# Categorical colors for the edge-KIND view (Codex: color by delta-kind, not raw
# signed weight, for the first explanatory figure — keeps the three concepts
# (node membership / signed delta / edge kind) from being visually conflated).
KIND_COLOR = {
    "substrate_only": "#c43030",  # red:   route present at substrate, gone by gate
    "gate_only":      "#1f4fb4",  # blue:  route emerges toward the gate
    "shared":         "#999999",  # grey:  present at both, same sign (ambient)
    "sign_flip":      "#111111",  # black: present at both, OPPOSITE sign
}
KIND_LABEL = {
    "substrate_only": "substrate-only (falls away)",
    "gate_only":      "gate-only (emerges)",
    "shared":         "shared, same sign",
    "sign_flip":      "SIGN-FLIP (inverts)",
}


def _layout(diff_edges: list[dict]):
    layer_to_y, by_layer = rank_spaced_layers(diff_edges)
    order = barycenter_order(by_layer, diff_edges)
    node_xy = {}
    for l in sorted(by_layer):
        m = order[l]
        for j, k in enumerate(m):
            node_xy[k] = (0.1 + 0.8 * (j + 0.5) / len(m), layer_to_y[l])
    return layer_to_y, node_xy


def _draw_gate_band(ax, layer_to_y, gate_band):
    gate_ys = [layer_to_y[l] for l in layer_to_y if gate_band[0] <= l <= gate_band[1]]
    if gate_ys:
        ax.axhspan(min(gate_ys) - 0.5, max(gate_ys) + 0.5,
                   color="#ffe9b0", alpha=0.4, zorder=0)


def _axis_cosmetics(ax, layer_to_y, title):
    ys_sorted = sorted(layer_to_y.items())
    ax.set_yticks([y for _, y in ys_sorted])
    ax.set_yticklabels([f"L{l}" for l, _ in ys_sorted], fontsize=6.5)
    ax.set_ylim(-0.7, (len(layer_to_y) - 1) + 0.7)
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.10, axis="y")


def render_signed(diff_edges, out_path, title, layer_to_y, node_xy,
                  gate_band=(29, 31), clip_percentile=98.0):
    """Single-panel SIGNED-delta view (corrected RdBu: blue=+/gate, red=-/substrate)."""
    fig, ax = plt.subplots(figsize=(11, 10))
    _draw_gate_band(ax, layer_to_y, gate_band)
    ax.text(0.01, max(layer_to_y.values()) + 0.05, "gate band (L29-L31)",
            fontsize=8, color="#9a7a00", va="bottom")

    deltas = np.array([e["delta"] for e in diff_edges], dtype=float)
    lw = thickness_rank(np.abs(deltas))
    vmax = max(float(np.percentile(np.abs(deltas), clip_percentile)), 1e-6) if len(deltas) else 1.0
    cmap = matplotlib.colormaps["RdBu"]  # blue=+ (gate), red=- (substrate). NOT _r.
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for i, e in enumerate(diff_edges):
        if e["src"] not in node_xy or e["dst"] not in node_xy:
            continue
        x0, y0 = node_xy[e["src"]]; x1, y1 = node_xy[e["dst"]]
        is_flip = e["kind"] == "sign_flip"
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle="arc3,rad=0.10",
            arrowstyle="-|>", mutation_scale=14,
            linewidth=lw[i] + (1.2 if is_flip else 0), color=cmap(norm(e["delta"])),
            alpha=0.6, zorder=4 if is_flip else 1))
        if is_flip:
            ax.plot([x0, x1], [y0, y1], color="#111", linewidth=lw[i] + 2.0,
                    alpha=0.3, zorder=3, solid_capstyle="round")

    for k, (x, y) in node_xy.items():
        ax.scatter([x], [y], c="#444", marker="o", s=34,
                   edgecolor="black", linewidth=0.3, alpha=0.85, zorder=5)

    _axis_cosmetics(ax, layer_to_y, title)
    ax.set_ylabel("layer, rank-spaced (substrate -> gate)", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("delta = w(L32 gate) - w(L24 substrate)\n"
                   "BLUE: intensifies toward gate   RED: falls away / substrate",
                   fontsize=7.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def render_panels(diff_edges, out_path, title, layer_to_y, node_xy,
                  gate_band=(29, 31)):
    """4-panel KIND view: substrate-only | gate-only | shared | sign-flip.

    Edges colored categorically by kind (not signed delta), so each panel
    answers one question cleanly. This is the explanatory figure.
    """
    kinds = ["substrate_only", "gate_only", "shared", "sign_flip"]
    by_kind = {k: [e for e in diff_edges if e["kind"] == k] for k in kinds}
    # Shared thickness scale across panels (rank over all |delta|).
    all_abs = np.array([abs(e["delta"]) for e in diff_edges], dtype=float)
    lw_lookup = {}
    if len(all_abs):
        order = all_abs.argsort()
        ranks = np.empty(len(all_abs)); ranks[order] = np.arange(len(all_abs))
        ranks = ranks / max(len(all_abs) - 1, 1)
        for e, r in zip(diff_edges, ranks):
            lw_lookup[id(e)] = 0.5 + 4.0 * r

    fig, axes = plt.subplots(1, 4, figsize=(22, 9), sharey=True)
    for ax, kind in zip(axes, kinds):
        _draw_gate_band(ax, layer_to_y, gate_band)
        edges = by_kind[kind]
        color = KIND_COLOR[kind]
        for e in edges:
            if e["src"] not in node_xy or e["dst"] not in node_xy:
                continue
            x0, y0 = node_xy[e["src"]]; x1, y1 = node_xy[e["dst"]]
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), connectionstyle="arc3,rad=0.10",
                arrowstyle="-|>", mutation_scale=13,
                linewidth=lw_lookup.get(id(e), 1.0), color=color,
                alpha=0.6, zorder=2))
        # faint full node scaffold so panels are spatially comparable
        for k, (x, y) in node_xy.items():
            ax.scatter([x], [y], c="#ccc", marker="o", s=18,
                       edgecolor="none", alpha=0.5, zorder=1)
        _axis_cosmetics(ax, layer_to_y, f"{KIND_LABEL[kind]}  (n={len(edges)})")
    axes[0].set_ylabel("layer, rank-spaced (substrate -> gate)", fontsize=9)
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return Counter(e["kind"] for e in diff_edges)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", required=True, type=Path,
                    help="probe_edge_fields run dir containing L24/ and L32/")
    ap.add_argument("--substrate-layer", type=int, default=24)
    ap.add_argument("--gate-layer", type=int, default=32)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--role-table", type=Path,
                    default=Path("apparatus/output/role_table_refusal_8b_fullcircuit_20260526.jsonl"),
                    help="Phase-A role table, for the super-weight infrastructure node set to drop")
    ap.add_argument("--clip-percentile", type=float, default=98.0,
                    help="Percentile of |delta| for the diverging color scale (robust to outliers)")
    args = ap.parse_args()

    a = load_collapsed_field(args.field_dir / f"L{args.substrate_layer}" / "edges.json")
    b = load_collapsed_field(args.field_dir / f"L{args.gate_layer}" / "edges.json")
    drop_nodes = load_super_weight_nodes(args.role_table)
    print(f"Dropping {len(drop_nodes)} super-weight infrastructure nodes: "
          f"{sorted(f'L{l}/N{n}' for l, n in drop_nodes)}")
    diff_edges = build_differential(a, b, drop_nodes=drop_nodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Shared layout so the signed view and the 4 kind-panels are spatially aligned.
    layer_to_y, node_xy = _layout(diff_edges)
    g, s = args.gate_layer, args.substrate_layer
    render_panels(
        diff_edges,
        args.out_dir / "differential_flow_panels.png",
        f"Differential flow by edge KIND: L{g} gate vs L{s} substrate  [super-weights removed]",
        layer_to_y, node_xy,
    )
    counts = Counter(e["kind"] for e in diff_edges)
    render_signed(
        diff_edges,
        args.out_dir / "differential_flow_signed.png",
        f"Differential flow (signed delta): flow(L{g} gate) - flow(L{s} substrate)",
        layer_to_y, node_xy,
        clip_percentile=args.clip_percentile,
    )

    # Sign-flip edges deserve an explicit listing — the headline signal.
    flips = [e for e in diff_edges if e["kind"] == "sign_flip"]
    flips.sort(key=lambda e: -abs(e["delta"]))
    summary = {
        "field_dir": str(args.field_dir),
        "substrate_layer": args.substrate_layer,
        "gate_layer": args.gate_layer,
        "dropped_super_weight_nodes": sorted(f"L{l}/N{n}" for l, n in drop_nodes),
        "clip_percentile": args.clip_percentile,
        "n_edges_substrate_collapsed": len(a),
        "n_edges_gate_collapsed": len(b),
        "n_edges_union_after_infra_drop": len(diff_edges),
        "kind_counts": dict(counts),
        "sign_flip_edges": [
            {"src": f"L{e['src'][0]}/N{e['src'][1]}",
             "dst": f"L{e['dst'][0]}/N{e['dst'][1]}",
             "w_substrate": round(e["w_substrate"], 5),
             "w_gate": round(e["w_gate"], 5),
             "delta": round(e["delta"], 5)}
            for e in flips
        ],
        "top_gate_specific": sorted(
            ({"src": f"L{e['src'][0]}/N{e['src'][1]}", "dst": f"L{e['dst'][0]}/N{e['dst'][1]}",
              "delta": round(e["delta"], 5)} for e in diff_edges if e["kind"] == "gate_only"),
            key=lambda d: -d["delta"])[:15],
        "top_substrate_specific": sorted(
            ({"src": f"L{e['src'][0]}/N{e['src'][1]}", "dst": f"L{e['dst'][0]}/N{e['dst'][1]}",
              "delta": round(e["delta"], 5)} for e in diff_edges if e["kind"] == "substrate_only"),
            key=lambda d: d["delta"])[:15],
    }
    (args.out_dir / "differential_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}/differential_summary.json")
    print(f"\nkind counts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
