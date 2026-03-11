#!/usr/bin/env python3
"""Analyze v2 anti-slop experiment results across all methods."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "generations"


def parse_filename(fname: str):
    """Extract method, param, seed from filename.

    Patterns:
      baseline_42.jsonl
      prompt_engineering_42.jsonl
      temperature_0.3_42.jsonl
      logit_suppression_-10.0_42.jsonl
      neuron_steering_0.4_42.jsonl
      caa_repeng_0.5_42.jsonl
      random_ablation_sample0_42.jsonl
    """
    stem = fname.replace(".jsonl", "")

    # random_ablation_sample{N}_{seed}
    m = re.match(r"random_ablation_sample(\d+)_(\d+)$", stem)
    if m:
        return "random_ablation", f"sample{m.group(1)}", int(m.group(2))

    # baseline_{seed}
    m = re.match(r"baseline_(\d+)$", stem)
    if m:
        return "baseline", "-", int(m.group(1))

    # prompt_engineering_{seed}
    m = re.match(r"prompt_engineering_(\d+)$", stem)
    if m:
        return "prompt_eng", "-", int(m.group(1))

    # {method}_{param}_{seed}  — handles negative params like -10.0
    m = re.match(r"(.+?)_([-\d.]+)_(\d+)$", stem)
    if m:
        method = m.group(1)
        param = m.group(2)
        seed = int(m.group(3))
        return method, param, seed

    return None, None, None


def load_all():
    """Load all JSONL files, return list of (method, param, seed, records)."""
    results = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = DATA_DIR / fname
        # Skip empty files
        if fpath.stat().st_size == 0:
            print(f"  [SKIP] {fname} (0 bytes)")
            continue

        method, param, seed = parse_filename(fname)
        if method is None:
            print(f"  [WARN] Could not parse: {fname}")
            continue

        records = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except json.JSONDecodeError:
                    pass

        results.append((method, param, seed, records))
    return results


def aggregate(data):
    """Group by (method, param) and compute stats."""
    groups = defaultdict(list)

    for method, param, seed, records in data:
        key = (method, param)
        for rec in records:
            sm = rec.get("slop_metrics", {})
            # Some files have flat structure, some nested
            composite = sm.get("composite_slop", rec.get("composite_slop"))
            density = sm.get("slop_density", rec.get("slop_density"))
            wc = sm.get("word_count", rec.get("word_count"))

            if composite is not None:
                groups[key].append({
                    "composite_slop": composite,
                    "slop_density": density or 0.0,
                    "word_count": wc or 0,
                })

    return groups


def compute_stats(groups):
    """Compute mean/std for each group."""
    import statistics

    stats = {}
    for key, items in groups.items():
        n = len(items)
        composites = [x["composite_slop"] for x in items]
        densities = [x["slop_density"] for x in items]
        word_counts = [x["word_count"] for x in items]

        mean_comp = statistics.mean(composites) if composites else 0
        std_comp = statistics.stdev(composites) if len(composites) > 1 else 0
        mean_dens = statistics.mean(densities) if densities else 0
        mean_wc = statistics.mean(word_counts) if word_counts else 0

        stats[key] = {
            "n": n,
            "mean_composite": mean_comp,
            "std_composite": std_comp,
            "mean_slop_density": mean_dens,
            "mean_word_count": mean_wc,
        }

    return stats


def sort_key(method, param):
    """Sort order: baseline first, then alphabetically by method, then by param value."""
    order = {
        "baseline": 0,
        "prompt_eng": 1,
        "temperature": 2,
        "logit_suppression": 3,
        "neuron_steering": 4,
        "caa_repeng": 5,
        "random_ablation": 6,
    }
    method_rank = order.get(method, 99)
    try:
        param_val = float(param)
    except (ValueError, TypeError):
        param_val = 0.0
    return (method_rank, param_val)


def main():
    print("Loading JSONL files...\n")
    data = load_all()
    print(f"\nLoaded {len(data)} files.\n")

    groups = aggregate(data)
    stats = compute_stats(groups)

    # Get baseline mean composite for % change calculation
    baseline_key = ("baseline", "-")
    baseline_comp = stats.get(baseline_key, {}).get("mean_composite", 1.0)

    # Sort
    sorted_keys = sorted(stats.keys(), key=lambda k: sort_key(k[0], k[1]))

    # Aggregate random_ablation samples into one row too
    random_keys = [k for k in sorted_keys if k[0] == "random_ablation"]
    non_random_keys = [k for k in sorted_keys if k[0] != "random_ablation"]

    # Print header
    hdr = (
        f"{'Method':<22} {'Param':>8} {'N':>5} "
        f"{'Composite':>10} {'Std':>8} {'SlopDens':>10} "
        f"{'WordCnt':>8} {'%Chg':>8}  {'Flag'}"
    )
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    def print_row(key):
        method, param = key
        s = stats[key]
        pct_change = ((s["mean_composite"] - baseline_comp) / baseline_comp * 100) if baseline_comp != 0 else 0
        flag = "MODEL DEGRADED" if s["mean_word_count"] < 50 else ""
        print(
            f"{method:<22} {param:>8} {s['n']:>5} "
            f"{s['mean_composite']:>10.3f} {s['std_composite']:>8.3f} {s['mean_slop_density']:>10.4f} "
            f"{s['mean_word_count']:>8.1f} {pct_change:>+8.1f}%  {flag}"
        )

    for key in non_random_keys:
        print_row(key)

    # Print random ablation individual samples
    if random_keys:
        print("-" * len(hdr))
        for key in random_keys:
            print_row(key)

        # Also compute aggregate across all random samples
        all_random_items = []
        for key in random_keys:
            all_random_items.extend(groups[key])

        import statistics as st
        composites = [x["composite_slop"] for x in all_random_items]
        densities = [x["slop_density"] for x in all_random_items]
        word_counts = [x["word_count"] for x in all_random_items]
        n = len(all_random_items)
        mean_comp = st.mean(composites)
        std_comp = st.stdev(composites) if n > 1 else 0
        mean_dens = st.mean(densities)
        mean_wc = st.mean(word_counts)
        pct_change = ((mean_comp - baseline_comp) / baseline_comp * 100) if baseline_comp != 0 else 0
        flag = "MODEL DEGRADED" if mean_wc < 50 else ""

        print(
            f"{'random_abl (ALL)':<22} {'agg':>8} {n:>5} "
            f"{mean_comp:>10.3f} {std_comp:>8.3f} {mean_dens:>10.4f} "
            f"{mean_wc:>8.1f} {pct_change:>+8.1f}%  {flag}"
        )

    print("=" * len(hdr))

    # Summary insights
    print("\n--- KEY FINDINGS ---\n")

    # Best non-degraded method
    best_key = None
    best_comp = float("inf")
    for key in sorted_keys:
        s = stats[key]
        if s["mean_word_count"] >= 50 and s["mean_composite"] < best_comp:
            best_comp = s["mean_composite"]
            best_key = key

    if best_key:
        pct = ((best_comp - baseline_comp) / baseline_comp * 100) if baseline_comp != 0 else 0
        print(f"Best method (non-degraded): {best_key[0]} param={best_key[1]}")
        print(f"  composite_slop = {best_comp:.3f} ({pct:+.1f}% vs baseline {baseline_comp:.3f})")

    # Degraded conditions
    degraded = [(k, stats[k]) for k in sorted_keys if stats[k]["mean_word_count"] < 50]
    if degraded:
        print(f"\nDEGRADED conditions (word_count < 50):")
        for k, s in degraded:
            print(f"  {k[0]} param={k[1]}: word_count={s['mean_word_count']:.1f}")

    # Neuron steering trend
    ns_keys = sorted(
        [k for k in sorted_keys if k[0] == "neuron_steering"],
        key=lambda k: float(k[1])
    )
    if ns_keys:
        print(f"\nNeuron steering trend (alpha -> composite_slop):")
        for k in ns_keys:
            s = stats[k]
            pct = ((s["mean_composite"] - baseline_comp) / baseline_comp * 100) if baseline_comp != 0 else 0
            flag = " *** DEGRADED" if s["mean_word_count"] < 50 else ""
            print(f"  alpha={k[1]:>5}: composite={s['mean_composite']:.3f} ({pct:+.1f}%), wc={s['mean_word_count']:.0f}{flag}")

    # CAA trend
    caa_keys = sorted(
        [k for k in sorted_keys if k[0] == "caa_repeng"],
        key=lambda k: float(k[1])
    )
    if caa_keys:
        print(f"\nCAA/RepEng trend (alpha -> composite_slop):")
        for k in caa_keys:
            s = stats[k]
            pct = ((s["mean_composite"] - baseline_comp) / baseline_comp * 100) if baseline_comp != 0 else 0
            flag = " *** DEGRADED" if s["mean_word_count"] < 50 else ""
            print(f"  alpha={k[1]:>5}: composite={s['mean_composite']:.3f} ({pct:+.1f}%), wc={s['mean_word_count']:.0f}{flag}")


if __name__ == "__main__":
    main()
