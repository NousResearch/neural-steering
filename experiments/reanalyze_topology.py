#!/usr/bin/env python3
"""Reanalyze topology results with position-aware hub analysis.

Codex flagged that hub_analysis() in core.py groups by (layer, neuron),
collapsing across positions. This can inflate apparent hub degrees when
the same neuron appears at multiple token positions (from multi-prompt
discovery). This script reanalyzes the saved edge JSON files using
full (layer, position, neuron) identity to check if the bottleneck
findings hold.

Usage:
    python experiments/reanalyze_topology.py experiments/topology_llama8b_20260305_161327
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_edges(edges_path):
    with open(edges_path) as f:
        data = json.load(f)
    return data["edges"]


def analyze_hubs(edges, key_fn):
    """Compute source/target hub analysis with custom key function.

    key_fn: edge endpoint dict -> hashable key
    Returns (source_hubs, target_hubs) as sorted lists.
    """
    source_deg = defaultdict(list)
    target_deg = defaultdict(list)
    for e in edges:
        src_key = key_fn(e["source"])
        tgt_key = key_fn(e["target"])
        source_deg[src_key].append(e["weight"])
        target_deg[tgt_key].append(e["weight"])

    def rank(deg_map):
        ranked = []
        for key, weights in deg_map.items():
            ranked.append((key, len(weights), sum(abs(w) for w in weights)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    return rank(source_deg), rank(target_deg)


def bottleneck_analysis(source_hubs, target_hubs, top_frac=0.2):
    source_map = {k: deg for k, deg, _ in source_hubs}
    target_map = {k: deg for k, deg, _ in target_hubs}
    all_nodes = set(source_map) | set(target_map)

    max_out = max(source_map.values()) if source_map else 1
    max_in = max(target_map.values()) if target_map else 1
    out_thresh = max_out * top_frac
    in_thresh = max_in * top_frac

    bottlenecks = []
    for n in all_nodes:
        out_deg = source_map.get(n, 0)
        in_deg = target_map.get(n, 0)
        if out_deg >= out_thresh and in_deg >= in_thresh:
            bottlenecks.append((n, in_deg, out_deg))
    bottlenecks.sort(key=lambda x: min(x[1], x[2]), reverse=True)
    return bottlenecks


def key_collapsed(endpoint):
    """Original behavior: (layer, neuron), ignoring position."""
    return (endpoint[0], endpoint[2])


def key_full(endpoint):
    """Position-aware: (layer, position, neuron)."""
    return tuple(endpoint)


def format_key(key):
    if len(key) == 2:
        return f"L{key[0]:02d}/N{key[1]}"
    else:
        return f"L{key[0]:02d}/P{key[1]:03d}/N{key[2]}"


def unique_positions_per_neuron(edges):
    """Count how many distinct positions each (layer, neuron) appears at."""
    positions = defaultdict(set)
    for e in edges:
        for endpoint in [e["source"], e["target"]]:
            layer, pos, neuron = endpoint
            positions[(layer, neuron)].add(pos)
    return {k: len(v) for k, v in positions.items()}


def report(task_name, edges_path):
    print(f"\n{'='*70}")
    print(f"  {task_name}")
    print(f"{'='*70}")

    edges = load_edges(edges_path)
    print(f"  {len(edges)} edges")

    # Check position diversity
    pos_counts = unique_positions_per_neuron(edges)
    multi_pos = {k: v for k, v in pos_counts.items() if v > 1}

    if multi_pos:
        print(f"\n  Neurons appearing at MULTIPLE positions ({len(multi_pos)}):")
        for (layer, neuron), n_pos in sorted(multi_pos.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"    L{layer:02d}/N{neuron}: {n_pos} positions")
    else:
        print(f"\n  All neurons appear at exactly 1 position (no collapsing effect)")

    # Collapsed analysis (original)
    src_collapsed, tgt_collapsed = analyze_hubs(edges, key_collapsed)
    bn_collapsed = bottleneck_analysis(src_collapsed, tgt_collapsed)

    # Position-aware analysis
    src_full, tgt_full = analyze_hubs(edges, key_full)
    bn_full = bottleneck_analysis(src_full, tgt_full)

    # Compare top source hubs
    print(f"\n  Top SOURCE hubs — collapsed vs position-aware:")
    print(f"  {'Collapsed':<35s} {'Position-aware':<45s}")
    print(f"  {'-'*35} {'-'*45}")
    for i in range(min(10, len(src_collapsed))):
        c_key, c_deg, c_w = src_collapsed[i]
        c_str = f"{format_key(c_key)}: deg={c_deg}"
        if i < len(src_full):
            f_key, f_deg, f_w = src_full[i]
            f_str = f"{format_key(f_key)}: deg={f_deg}"
        else:
            f_str = ""
        print(f"  {c_str:<35s} {f_str:<45s}")

    # Compare top target hubs
    print(f"\n  Top TARGET hubs — collapsed vs position-aware:")
    print(f"  {'Collapsed':<35s} {'Position-aware':<45s}")
    print(f"  {'-'*35} {'-'*45}")
    for i in range(min(10, len(tgt_collapsed))):
        c_key, c_deg, c_w = tgt_collapsed[i]
        c_str = f"{format_key(c_key)}: deg={c_deg}"
        if i < len(tgt_full):
            f_key, f_deg, f_w = tgt_full[i]
            f_str = f"{format_key(f_key)}: deg={f_deg}"
        else:
            f_str = ""
        print(f"  {c_str:<35s} {f_str:<45s}")

    # Compare bottlenecks
    print(f"\n  BOTTLENECKS — collapsed vs position-aware:")
    print(f"  Collapsed ({len(bn_collapsed)}):")
    for key, in_deg, out_deg in bn_collapsed[:8]:
        print(f"    {format_key(key)}: in={in_deg}, out={out_deg}")
    print(f"  Position-aware ({len(bn_full)}):")
    for key, in_deg, out_deg in bn_full[:8]:
        print(f"    {format_key(key)}: in={in_deg}, out={out_deg}")

    # Quantify inflation
    print(f"\n  Degree inflation check (top collapsed hubs):")
    full_src_map = {k: deg for k, deg, _ in src_full}
    for key, c_deg, _ in src_collapsed[:5]:
        layer, neuron = key
        n_pos = pos_counts.get(key, 1)
        # Find max degree across positions for this neuron
        pos_degs = []
        for (fkey, fdeg, _) in src_full:
            if len(fkey) == 3 and fkey[0] == layer and fkey[2] == neuron:
                pos_degs.append((fkey[1], fdeg))
        if pos_degs:
            max_pos_deg = max(d for _, d in pos_degs)
            print(f"    {format_key(key)}: collapsed={c_deg}, positions={n_pos}, "
                  f"max_single_pos={max_pos_deg}, "
                  f"inflation={c_deg/max_pos_deg:.1f}x")
        else:
            print(f"    {format_key(key)}: collapsed={c_deg}, not found in full analysis")


def main():
    if len(sys.argv) < 2:
        print("Usage: python reanalyze_topology.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    # Find all kstar directories
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir() and "kstar" in subdir.name:
            edges_path = subdir / "edges.json"
            if edges_path.exists():
                report(subdir.name, edges_path)


if __name__ == "__main__":
    main()
