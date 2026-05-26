"""Analyze a role table: pairwise rank correlations + role clustering.

Two views are reported:

1. Position-preserving view, primary key (layer, position, neuron). Same row count
   as the role table. Useful for the edges/attribution axes, which are inherently
   position-aware.

2. Collapsed view, primary key (layer, neuron). Intervention data
   (necessity, sufficiency) is position-collapsed in the source data, so reporting
   correlations at the position-preserved level over-counts intervention-tested
   neurons across their positions. The collapsed view is the honest unit for
   intervention-axis correlations.

Falsifier: pairwise rank correlation across the four primary axes (attribution,
edges, necessity, sufficiency) exceeds 0.85 in the collapsed view -> role
decomposition is degenerate to a single ranking.

Apparatus-asymmetric roles are first-class:
- reader-only          : nec_sigma > +5, |suff_dS| < 1.0
- writer-only          : |nec_sigma| <= +5, suff_dS > +1.0
- reader-writer        : nec_sigma > +5, suff_dS > +1.0
- suppressor-consistent: nec_sigma < -5 AND suff_dS < -0.5
- suppressor-transplant: |nec_sigma| <= 5, suff_dS < -0.5     (signal-bound)
- suppressor-ablation  : nec_sigma < -5, |suff_dS| <= 0.5     (context-dependent)
- mixed                : doesn't match the above
- unclassified         : no intervention data
- infrastructure       : super-weight
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spearman(x, y) -> tuple[float, int]:
    """Spearman rank correlation. Returns (rho, n_pairs).

    Accepts lists that may contain None; only complete pairs are used.
    """
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    def rank(v):
        order = v.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros_like(counts, dtype=float)
        for i, r in zip(inv, ranks):
            sums[i] += r
        means = sums / counts
        return means[inv]
    ra, rb = rank(a), rank(b)
    rho = np.corrcoef(ra, rb)[0, 1]
    return float(rho), len(pairs)


def pairwise_correlations(rows: list[dict], cols: list[str]) -> dict:
    series = {c: [r.get(c) for r in rows] for c in cols}
    result = {}
    for i, c1 in enumerate(cols):
        for c2 in cols[i:]:
            rho, n = spearman(series[c1], series[c2])
            result[f"{c1} ~ {c2}"] = {"rho": rho, "n_pairs": n}
    return result


def collapse_to_layer_neuron(rows: list[dict]) -> list[dict]:
    """Collapse position-preserving rows to (layer, neuron).

    For position-aware fields (attribution, edge_*), aggregate across positions:
    - attribution: max-magnitude across positions (preserves sign of the dominant
      position).
    - edge_in_*, edge_out_*: sum across positions.
    For position-collapsed fields (necessity, sufficiency, is_super_weight,
    is_bottleneck_candidate), take any one — they're identical across positions.
    """
    by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["layer"], r["neuron"])].append(r)

    out = []
    for (layer, neuron), group in by_key.items():
        # dominant-magnitude attribution preserves sign
        attrs = [g["attribution"] for g in group]
        dom = max(attrs, key=abs)

        merged = {
            "layer": layer,
            "neuron": neuron,
            "n_positions": len(group),
            "attribution": dom,
            "edge_in_count": sum(g.get("edge_in_count", 0) for g in group),
            "edge_in_weight_signed": sum(g.get("edge_in_weight_signed", 0.0) for g in group),
            "edge_in_weight_abs": sum(g.get("edge_in_weight_abs", 0.0) for g in group),
            "edge_out_count": sum(g.get("edge_out_count", 0) for g in group),
            "edge_out_weight_signed": sum(g.get("edge_out_weight_signed", 0.0) for g in group),
            "edge_out_weight_abs": sum(g.get("edge_out_weight_abs", 0.0) for g in group),
            # Intervention data: position-collapsed in source; pick from any row
            "necessity_dMargin": group[0].get("necessity_dMargin"),
            "necessity_fraction": group[0].get("necessity_fraction"),
            "necessity_sigma": group[0].get("necessity_sigma"),
            "sufficiency_dS": group[0].get("sufficiency_dS"),
            "sufficiency_sigma": group[0].get("sufficiency_sigma"),
            "is_super_weight": group[0].get("is_super_weight", False),
            "is_bottleneck_candidate": group[0].get("is_bottleneck_candidate", False),
        }
        out.append(merged)
    return out


# Role thresholds. Apparatus-asymmetric roles are first-class.
NEC_SIGMA_POS = 5.0
NEC_SIGMA_NEG = -5.0
SUFF_DS_POS = 1.0
SUFF_DS_NEG = -0.5
SUFF_DS_NEAR_ZERO = 0.5
SUFF_DS_NEAR_ZERO_LARGE = 1.0  # for reader: must be near-zero on a wider band


def assign_role(row: dict) -> str:
    """Assign a role using the apparatus-asymmetric scheme.

    Order matters: infrastructure first, then suppressors, then writers/readers,
    then catch-all.
    """
    if row.get("is_super_weight"):
        return "infrastructure"
    if row.get("necessity_dMargin") is None:
        return "unclassified"

    nec = row.get("necessity_sigma") or 0.0
    suff = row.get("sufficiency_dS") or 0.0

    # Suppressors first — they're the rarest, most diagnostic
    if nec < NEC_SIGMA_NEG and suff < SUFF_DS_NEG:
        return "suppressor-consistent"
    if abs(nec) <= NEC_SIGMA_POS and suff < SUFF_DS_NEG:
        return "suppressor-transplant"   # signal-bound
    if nec < NEC_SIGMA_NEG and abs(suff) <= SUFF_DS_NEAR_ZERO:
        return "suppressor-ablation"     # context-dependent

    # Reader/writer combinations
    is_nec_pos = nec >= NEC_SIGMA_POS
    is_suff_pos = suff >= SUFF_DS_POS

    if is_nec_pos and is_suff_pos:
        return "reader-writer"
    if is_nec_pos and abs(suff) < SUFF_DS_NEAR_ZERO_LARGE:
        return "reader-only"
    if not is_nec_pos and is_suff_pos:
        return "writer-only"

    return "mixed"


def role_summary(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        role = assign_role(r)
        r["_role"] = role
        out.setdefault(role, []).append(r)
    return out


def print_correlation_matrix(corrs: dict, view_name: str, n_rows: int) -> None:
    print(f"\n=== Pairwise Spearman correlations [{view_name}, n_rows={n_rows}] ===")
    print("(>0.85 in any non-trivial cross-axis cell would falsify the apparatus)\n")
    seen_pairs = set()
    for pair, v in corrs.items():
        a, b = pair.split(" ~ ")
        if a == b:
            continue
        if (b, a) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        rho = v["rho"]
        n = v["n_pairs"]
        flag = "  <-- HIGH" if not np.isnan(rho) and abs(rho) > 0.85 else ""
        print(f"  {pair:60s}  rho={rho:+.3f}  n={n}{flag}")


def print_role_summary(by_role: dict, fmt_key) -> None:
    print("\n=== Role assignments (collapsed view) ===")
    order = [
        "infrastructure",
        "reader-writer", "reader-only", "writer-only",
        "suppressor-consistent", "suppressor-transplant", "suppressor-ablation",
        "mixed", "unclassified",
    ]
    for role in order:
        rs = by_role.get(role, [])
        if not rs:
            continue
        print(f"\n[{role}] n={len(rs)}")
        rs_sorted = sorted(rs, key=lambda x: -abs(x.get("attribution", 0)))
        for r in rs_sorted[:10]:
            nec = r.get("necessity_dMargin")
            nec_s = r.get("necessity_sigma")
            suff = r.get("sufficiency_dS")
            ein = r.get("edge_in_count", 0)
            eout = r.get("edge_out_count", 0)
            attr = r.get("attribution", 0)
            nec_str = f"{nec:+.2f}" if nec is not None else "  -- "
            nec_sig_str = f"{nec_s:+.1f}σ" if nec_s is not None else " --  "
            suff_str = f"{suff:+.2f}" if suff is not None else "  -- "
            print(f"  {fmt_key(r):<25s}  attr={attr:+.3f}  "
                  f"nec={nec_str} ({nec_sig_str})  suff={suff_str}  "
                  f"edges(in/out)={ein}/{eout}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out-summary", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows(args.inp)
    print(f"Loaded {len(rows)} position-preserved rows from {args.inp}")

    # ----- position-preserved view -----
    pos_cols = [
        "attribution",
        "edge_in_count", "edge_in_weight_abs",
        "edge_out_count", "edge_out_weight_abs",
        "necessity_dMargin", "necessity_sigma",
        "sufficiency_dS", "sufficiency_sigma",
    ]
    pos_corrs = pairwise_correlations(rows, pos_cols)
    print_correlation_matrix(
        pos_corrs, view_name="position-preserved", n_rows=len(rows))

    # ----- collapsed view -----
    collapsed = collapse_to_layer_neuron(rows)
    print(f"\nCollapsed to {len(collapsed)} unique (layer, neuron)")
    coll_cols = pos_cols  # same axis names
    coll_corrs = pairwise_correlations(collapsed, coll_cols)
    print_correlation_matrix(
        coll_corrs, view_name="collapsed (layer, neuron)", n_rows=len(collapsed))

    # ----- role assignments on collapsed view -----
    by_role = role_summary(collapsed)
    print_role_summary(by_role, fmt_key=lambda r: f"L{r['layer']:02d}/N{r['neuron']:<5}")

    print("\n=== Role counts (collapsed) ===")
    order = [
        "infrastructure",
        "reader-writer", "reader-only", "writer-only",
        "suppressor-consistent", "suppressor-transplant", "suppressor-ablation",
        "mixed", "unclassified",
    ]
    for role in order:
        rs = by_role.get(role, [])
        if rs:
            print(f"  {role:24s}  n={len(rs)}")

    if args.out_summary:
        out = {
            "n_rows_position_preserved": len(rows),
            "n_rows_collapsed": len(collapsed),
            "correlations_position_preserved": pos_corrs,
            "correlations_collapsed": coll_corrs,
            "role_counts": {role: len(rs) for role, rs in by_role.items()},
        }
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(out, indent=2))
        print(f"\nWrote summary to {args.out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
