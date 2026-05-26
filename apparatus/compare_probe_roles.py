"""Compare token-readout role table against hidden-probe role rows.

This is the Apparatus 2a readout-consistency report. It does not assign final
roles for the probe readout because probe-score units are layer-dependent; it
reports correlations and sign agreements for the two intervention axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from apparatus.analyze import collapse_to_layer_neuron, spearman


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sign(x: float, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def sign_agreement(a: list[float], b: list[float], eps_a: float = 1e-9, eps_b: float = 1e-9) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0, "agreement": float("nan"), "nonzero_agreement": float("nan")}
    signs = [(sign(x, eps_a), sign(y, eps_b)) for x, y in pairs]
    agreement = sum(1 for x, y in signs if x == y) / len(signs)
    nonzero = [(x, y) for x, y in signs if x != 0 and y != 0]
    nonzero_agreement = (
        sum(1 for x, y in nonzero if x == y) / len(nonzero)
        if nonzero else float("nan")
    )
    return {
        "n": len(pairs),
        "agreement": float(agreement),
        "nonzero_n": len(nonzero),
        "nonzero_agreement": float(nonzero_agreement),
    }


def compare(token_rows: list[dict], probe_rows: list[dict]) -> dict:
    collapsed = collapse_to_layer_neuron(token_rows)
    token_by_key = {(r["layer"], r["neuron"]): r for r in collapsed}
    probe_by_key = {(r["layer"], r["neuron"]): r for r in probe_rows}
    keys = sorted(set(token_by_key) & set(probe_by_key))

    joined = []
    for key in keys:
        t = token_by_key[key]
        p = probe_by_key[key]
        joined.append({
            "layer": key[0],
            "neuron": key[1],
            "upstream_of_probe": p["upstream_of_probe"],
            "token_necessity_sigma": t.get("necessity_sigma"),
            "token_sufficiency_dS": t.get("sufficiency_dS"),
            "probe_necessity_sigma": p.get("necessity_sigma"),
            "probe_sufficiency_dProbe": p.get("sufficiency_dProbe"),
            "probe_sufficiency_sigma": p.get("sufficiency_sigma"),
        })

    def col(name: str) -> list[float]:
        return [r.get(name) for r in joined]

    comparisons = {}
    for suffix, subset in {
        "all": joined,
        "upstream": [r for r in joined if r["upstream_of_probe"]],
        "downstream": [r for r in joined if not r["upstream_of_probe"]],
    }.items():
        if not subset:
            continue
        def scol(name: str) -> list[float]:
            return [r.get(name) for r in subset]
        nec_rho, nec_n = spearman(scol("token_necessity_sigma"), scol("probe_necessity_sigma"))
        suff_rho, suff_n = spearman(scol("token_sufficiency_dS"), scol("probe_sufficiency_dProbe"))
        cross_rho, cross_n = spearman(scol("probe_necessity_sigma"), scol("probe_sufficiency_dProbe"))
        comparisons[suffix] = {
            "n": len(subset),
            "token_necessity_vs_probe_necessity": {"rho": nec_rho, "n": nec_n},
            "token_sufficiency_vs_probe_sufficiency": {"rho": suff_rho, "n": suff_n},
            "probe_necessity_vs_probe_sufficiency": {"rho": cross_rho, "n": cross_n},
            "necessity_sign_agreement": sign_agreement(
                scol("token_necessity_sigma"), scol("probe_necessity_sigma"),
            ),
            "sufficiency_sign_agreement": sign_agreement(
                scol("token_sufficiency_dS"), scol("probe_sufficiency_dProbe"),
            ),
        }

    return {
        "n_joined": len(joined),
        "comparisons": comparisons,
        "joined_rows": joined,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare token and probe role tables")
    ap.add_argument("--token-role-table", required=True, type=Path)
    ap.add_argument("--probe-role-rows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    result = compare(load_jsonl(args.token_role_table), load_jsonl(args.probe_role_rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))

    print(f"joined rows: {result['n_joined']}")
    for name, comp in result["comparisons"].items():
        print(f"\n[{name}] n={comp['n']}")
        print("  token necessity ~ probe necessity:",
              f"rho={comp['token_necessity_vs_probe_necessity']['rho']:+.3f}",
              f"n={comp['token_necessity_vs_probe_necessity']['n']}")
        print("  token sufficiency ~ probe sufficiency:",
              f"rho={comp['token_sufficiency_vs_probe_sufficiency']['rho']:+.3f}",
              f"n={comp['token_sufficiency_vs_probe_sufficiency']['n']}")
        print("  probe necessity ~ probe sufficiency:",
              f"rho={comp['probe_necessity_vs_probe_sufficiency']['rho']:+.3f}",
              f"n={comp['probe_necessity_vs_probe_sufficiency']['n']}")
        print("  necessity sign agreement:",
              comp["necessity_sign_agreement"])
        print("  sufficiency sign agreement:",
              comp["sufficiency_sign_agreement"])
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
