"""Analyze a role table: pairwise rank correlations + role clustering.

The first apparatus falsifier. If pairwise rank correlations across the four axes
(necessity, sufficiency_signed, edge_in, edge_out) exceed 0.85, there's no role
decomposition — it's a single ranking.

Operates on the role-table JSONL produced by role_table.py. Stdlib + numpy + scipy.
"""

from __future__ import annotations

import argparse
import json
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
    # rankdata-equivalent: argsort + average ranks for ties
    def rank(v):
        order = v.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        # tie-correct
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
    """Compute pairwise spearman correlations between the named columns."""
    series = {c: [r.get(c) for r in rows] for c in cols}
    result = {}
    for i, c1 in enumerate(cols):
        for c2 in cols[i:]:
            rho, n = spearman(series[c1], series[c2])
            result[f"{c1} ~ {c2}"] = {"rho": rho, "n_pairs": n}
    return result


def assign_role(row: dict) -> str:
    """Assign a tentative role to a single row using March-5 conventions.

    Rules (apply in order):
      1. super_weight -> 'infrastructure'
      2. has no intervention data -> 'unclassified'
      3. necessity_sigma <= -5 AND sufficiency_dS < -0.5 -> 'suppressor' (consistent counterforce)
      4. sufficiency_dS >= 1.0 -> 'writer' (drives behavior when transplanted)
      5. necessity_sigma >= 5 AND |sufficiency_dS| < 1.0 -> 'reader'
      6. else -> 'mixed'

    These thresholds are calibrated to the March 5 refusal data; the values are
    deliberately conservative and visible so we can tune them after looking.
    """
    if row.get("is_super_weight"):
        return "infrastructure"
    if row.get("necessity_dMargin") is None:
        return "unclassified"

    nec_sigma = row.get("necessity_sigma") or 0.0
    suff_dS = row.get("sufficiency_dS") or 0.0

    if nec_sigma <= -5 and suff_dS < -0.5:
        return "suppressor"
    if suff_dS >= 1.0:
        return "writer"
    if nec_sigma >= 5 and abs(suff_dS) < 1.0:
        return "reader"
    return "mixed"


def role_summary(rows: list[dict]) -> dict:
    """Group rows by assigned role and summarize."""
    out = {}
    for r in rows:
        role = assign_role(r)
        r["_role"] = role  # mutate for later
        out.setdefault(role, []).append(r)
    return out


def print_correlation_matrix(corrs: dict) -> None:
    print("\n=== Pairwise Spearman correlations ===")
    print("(>0.85 in any cell would falsify the role-decomposition apparatus)\n")
    for pair, v in corrs.items():
        rho = v["rho"]
        n = v["n_pairs"]
        flag = "  <-- HIGH" if not np.isnan(rho) and abs(rho) > 0.85 and not pair.split(" ~ ")[0] == pair.split(" ~ ")[1] else ""
        print(f"  {pair:60s}  rho={rho:+.3f}  n={n}{flag}")


def print_role_summary(by_role: dict) -> None:
    print("\n=== Role assignments ===")
    for role in ["infrastructure", "reader", "writer", "suppressor", "mixed", "unclassified"]:
        rs = by_role.get(role, [])
        if not rs:
            continue
        print(f"\n[{role}] n={len(rs)}")
        # show top 10 by attribution
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
            print(f"  L{r['layer']:02d}/P{r['position']:03d}/N{r['neuron']:<5}  "
                  f"attr={attr:+.3f}  nec={nec_str} ({nec_sig_str})  suff={suff_str}  "
                  f"edges(in/out)={ein}/{eout}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out-summary", type=Path, default=None,
                    help="Optional path to write the summary JSON")
    args = ap.parse_args()

    rows = load_rows(args.inp)
    print(f"Loaded {len(rows)} rows from {args.inp}")

    cols = [
        "attribution",
        "edge_in_count",
        "edge_in_weight_abs",
        "edge_out_count",
        "edge_out_weight_abs",
        "necessity_dMargin",
        "necessity_sigma",
        "sufficiency_dS",
        "sufficiency_sigma",
    ]
    corrs = pairwise_correlations(rows, cols)
    print_correlation_matrix(corrs)

    by_role = role_summary(rows)
    print_role_summary(by_role)

    print("\n=== Role counts ===")
    for role, rs in sorted(by_role.items(), key=lambda kv: -len(kv[1])):
        print(f"  {role:16s}  n={len(rs)}")

    if args.out_summary:
        out = {
            "n_rows": len(rows),
            "correlations": corrs,
            "role_counts": {role: len(rs) for role, rs in by_role.items()},
        }
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(out, indent=2))
        print(f"\nWrote summary to {args.out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
