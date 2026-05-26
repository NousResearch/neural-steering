"""Cross-probe-layer consistency check (Apparatus 2a follow-up).

Question: do the three probe layers (L18, L24, L28) agree about which upstream
neurons are probe-writers / probe-suppressors / probe-readers? If they do, we
have evidence that the substrate-level role structure is a property of the
model that any sufficient probe sees. If they disagree, the probe-layer choice
is itself load-bearing for what "substrate" means.

Inputs are the per-layer probe_role_rows_layer{L}.jsonl files produced by
probe_role_table.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apparatus.analyze import spearman


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pair(rows_a: list[dict], rows_b: list[dict], col: str,
         upstream_filter: bool = True):
    """Spearman between two probe layers' role rows on a column, joined by
    (layer, neuron). When upstream_filter=True, restrict to neurons upstream of
    BOTH probe layers (i.e. layer <= min(probe_a, probe_b)).
    """
    by_key_a = {(r["layer"], r["neuron"]): r for r in rows_a}
    by_key_b = {(r["layer"], r["neuron"]): r for r in rows_b}
    if rows_a and rows_b:
        probe_a = rows_a[0]["probe_layer"]
        probe_b = rows_b[0]["probe_layer"]
        min_probe = min(probe_a, probe_b)
    else:
        min_probe = 0
    keys = sorted(set(by_key_a) & set(by_key_b))
    if upstream_filter:
        keys = [k for k in keys if k[0] <= min_probe]
    xs = [by_key_a[k].get(col) for k in keys]
    ys = [by_key_b[k].get(col) for k in keys]
    return spearman(xs, ys), len(keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, type=Path,
                    help="Directory containing probe_role_rows_layerN.jsonl files")
    ap.add_argument("--layers", default="18,24,28")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]
    rows_by_layer = {
        L: load_jsonl(args.probe_dir / f"probe_role_rows_layer{L}.jsonl")
        for L in layers
    }
    for L, rs in rows_by_layer.items():
        print(f"L{L}: {len(rs)} role rows loaded")

    cols = ["necessity_sigma", "sufficiency_dProbe", "sufficiency_sigma"]
    pairs = []
    for i, a in enumerate(layers):
        for b in layers[i + 1:]:
            pairs.append((a, b))

    out = {"layers": layers, "comparisons": {}}
    print("\n=== Cross-probe-layer Spearman (upstream of min(probe_a, probe_b)) ===\n")
    for a, b in pairs:
        out["comparisons"][f"L{a}_vs_L{b}"] = {}
        for col in cols:
            (rho, _), n = pair(rows_by_layer[a], rows_by_layer[b], col,
                               upstream_filter=True)
            out["comparisons"][f"L{a}_vs_L{b}"][col] = {"rho": rho, "n": n}
            print(f"  L{a} vs L{b}  {col:22s}  rho={rho:+.3f}  n={n}")
        # Also all candidates (no upstream filter) for reference
        for col in cols:
            (rho, _), n = pair(rows_by_layer[a], rows_by_layer[b], col,
                               upstream_filter=False)
            out["comparisons"][f"L{a}_vs_L{b}"][f"{col}_all"] = {"rho": rho, "n": n}
        print()

    # Per-neuron drift summary: for the upstream-of-min-layer set, show each
    # neuron's sufficiency across requested probe layers.
    min_layer = min(layers)
    print(f"\n=== Per-neuron probe sufficiency across layers (upstream of L{min_layer}) ===\n")
    upstream_min = [r for r in rows_by_layer[min_layer] if r["upstream_of_probe"]]
    upstream_min_keys = {(r["layer"], r["neuron"]) for r in upstream_min}
    rows_by_key = {
        L: {(r["layer"], r["neuron"]): r for r in rows_by_layer[L]}
        for L in layers
    }
    drift_rows = []
    for k in sorted(upstream_min_keys, key=lambda x: x):
        vals = {L: rows_by_key[L].get(k, {}).get("sufficiency_dProbe") for L in layers}
        drift_rows.append({"layer_neuron": f"L{k[0]:02d}/N{k[1]:<5d}", **{f"L{L}": vals[L] for L in layers}})

    sort_layer = layers[len(layers) // 2]
    drift_rows.sort(key=lambda r: -(r[f"L{sort_layer}"] if r[f"L{sort_layer}"] is not None else 0.0))
    header = f"{'neuron':18s}" + "".join(f" L{L} suff".rjust(12) for L in layers)
    print(header)
    for r in drift_rows:
        def fmt(x):
            return f"{x:+10.4f}" if x is not None else "       N/A"
        print(f"{r['layer_neuron']:18s}" + "".join(f" {fmt(r[f'L{L}'])}" for L in layers))
    out[f"per_neuron_drift_upstream_of_L{min_layer}"] = drift_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
