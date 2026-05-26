"""Build the role table from existing March 5 JSON outputs.

CPU-only. No model loads, no GPU. The whole point is to validate the schema, the
joins, and the sign conventions against existing data before spending cluster time.

Usage:

    python -m apparatus.role_table \\
        --topology experiments/topology_llama8b_20260305_163508/relp-behavioral_refusal_kstar91 \\
        --surgical experiments/surgical_llama8b_20260305_175327/surgical_behavioral.json \\
        --sufficiency experiments/sufficiency_llama8b_20260305_183035/sufficiency_behavioral.json \\
        --out apparatus/output/role_table_refusal_8b_20260305.jsonl

Outputs a JSONL of RoleRow dicts (one row per circuit neuron, position-preserving).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from apparatus.schema import RoleRow, COLUMNS


SUPER_WEIGHT_NEURONS = {
    # Hardcoded from methods_supplement.md sec 8.3 — Llama-3.1-8B universal neurons
    (0, 491), (0, 8268), (1, 198), (1, 2427),
}


def parse_circuit_key(k: str) -> tuple[int, int, int]:
    """'1,1,2427' -> (1, 1, 2427)"""
    parts = k.split(",")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_circuit(topology_dir: Path) -> dict[tuple[int, int, int], float]:
    """Returns {(layer, position, neuron) -> attribution}."""
    p = topology_dir / "circuit.json"
    data = json.loads(p.read_text())
    neurons = data["neurons"]
    return {parse_circuit_key(k): float(v) for k, v in neurons.items()}


def load_edges(topology_dir: Path) -> list[dict]:
    """Returns the edge list."""
    p = topology_dir / "edges.json"
    return json.loads(p.read_text())["edges"]


def aggregate_edges(edges: list[dict]) -> tuple[dict, dict]:
    """Compute per-(layer, position, neuron) in/out aggregates.

    Returns (incoming_dict, outgoing_dict) where each maps
    (layer, position, neuron) -> dict(count, signed_sum, abs_sum).
    """
    incoming = defaultdict(lambda: {"count": 0, "signed_sum": 0.0, "abs_sum": 0.0})
    outgoing = defaultdict(lambda: {"count": 0, "signed_sum": 0.0, "abs_sum": 0.0})

    for e in edges:
        src = tuple(e["source"])  # (layer, position, neuron)
        tgt = tuple(e["target"])
        w = float(e["weight"])

        incoming[tgt]["count"] += 1
        incoming[tgt]["signed_sum"] += w
        incoming[tgt]["abs_sum"] += abs(w)

        outgoing[src]["count"] += 1
        outgoing[src]["signed_sum"] += w
        outgoing[src]["abs_sum"] += abs(w)

    return dict(incoming), dict(outgoing)


def load_surgical(p: Path) -> tuple[dict, set, float]:
    """Returns ({(layer, neuron) -> dict}, bottleneck_set, dMargin_full)."""
    data = json.loads(p.read_text())
    sn = data["single_neuron_ablation"]
    out = {}
    for row in sn:
        key = (row["layer"], row["neuron"])
        out[key] = {
            "dMargin": row["dMargin"],
            "fraction": row.get("fraction_of_full"),
            "sigma": row.get("sigma_above_random"),
        }
    bn = {(b["layer"], b["neuron"]) for b in data["bottleneck_neurons"]}
    return out, bn, data["dMargin_full"]


def load_sufficiency(p: Path) -> dict:
    """Returns {(layer, neuron) -> dict}."""
    data = json.loads(p.read_text())
    sn = data["single_neuron_transplant"]
    out = {}
    for row in sn:
        key = (row["layer"], row["neuron"])
        out[key] = {
            "dS": row["dSufficiency"],
            "sigma": row.get("sigma_above_random"),
        }
    return out


def build_role_table(
    topology_dir: Path,
    surgical_path: Path,
    sufficiency_path: Path,
) -> list[RoleRow]:
    circuit = load_circuit(topology_dir)
    edges = load_edges(topology_dir)
    incoming, outgoing = aggregate_edges(edges)
    surgical, bottleneck_set, _dMargin_full = load_surgical(surgical_path)
    sufficiency = load_sufficiency(sufficiency_path)

    rows: list[RoleRow] = []
    for (layer, position, neuron), attribution in circuit.items():
        key3 = (layer, position, neuron)
        key2 = (layer, neuron)

        inc = incoming.get(key3, {"count": 0, "signed_sum": 0.0, "abs_sum": 0.0})
        out = outgoing.get(key3, {"count": 0, "signed_sum": 0.0, "abs_sum": 0.0})

        surg = surgical.get(key2)
        suff = sufficiency.get(key2)

        row = RoleRow(
            layer=layer,
            position=position,
            neuron=neuron,
            attribution=attribution,
            edge_in_count=inc["count"],
            edge_in_weight_signed=inc["signed_sum"],
            edge_in_weight_abs=inc["abs_sum"],
            edge_out_count=out["count"],
            edge_out_weight_signed=out["signed_sum"],
            edge_out_weight_abs=out["abs_sum"],
            necessity_dMargin=surg["dMargin"] if surg else None,
            necessity_fraction=surg["fraction"] if surg else None,
            necessity_sigma=surg["sigma"] if surg else None,
            sufficiency_dS=suff["dS"] if suff else None,
            sufficiency_sigma=suff["sigma"] if suff else None,
            intervention_position_collapsed=True,
            is_bottleneck_candidate=(key2 in bottleneck_set),
            is_super_weight=(key2 in SUPER_WEIGHT_NEURONS),
        )
        rows.append(row)

    return rows


def write_jsonl(rows: list[RoleRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


def write_csv(rows: list[RoleRow], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True, type=Path,
                    help="Directory containing circuit.json + edges.json + analysis.json")
    ap.add_argument("--surgical", required=True, type=Path)
    ap.add_argument("--sufficiency", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output JSONL path (writes .csv alongside)")
    args = ap.parse_args()

    rows = build_role_table(args.topology, args.surgical, args.sufficiency)

    write_jsonl(rows, args.out)
    csv_path = args.out.with_suffix(".csv")
    write_csv(rows, csv_path)

    n_total = len(rows)
    n_with_intervention = sum(1 for r in rows if r.necessity_dMargin is not None)
    n_super = sum(1 for r in rows if r.is_super_weight)
    print(f"wrote {n_total} rows to {args.out}")
    print(f"  with intervention data: {n_with_intervention}")
    print(f"  super-weights: {n_super}")
    print(f"  csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
