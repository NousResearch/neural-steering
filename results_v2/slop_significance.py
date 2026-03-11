#!/usr/bin/env python3
"""Statistical significance tests for anti-slop experiment results.

Paired comparison of each method+param against baseline using:
- Wilcoxon signed-rank test (non-parametric)
- 95% bootstrap CI for mean difference
- Cohen's d effect size
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DATA_DIR = Path("/Users/karan/digi2digi/shared/projects/neuron-steering/neuron-circuits/experiments/results_v2/generations")
MIN_LINES = 200
N_BOOTSTRAP = 10000
RNG = np.random.default_rng(42)


def parse_filename(fname: str) -> tuple[str, str | None, str]:
    """Parse {method}_{param}_{seed}.jsonl or {method}_{seed}.jsonl.

    Returns (method, param_or_None, seed).

    Tricky cases:
    - baseline_42.jsonl -> ("baseline", None, "42")
    - prompt_engineering_42.jsonl -> ("prompt_engineering", None, "42")
    - neuron_steering_0.0_42.jsonl -> ("neuron_steering", "0.0", "42")
    - caa_repeng_1.0_42.jsonl -> ("caa_repeng", "1.0", "42")
    - logit_suppression_-10.0_42.jsonl -> ("logit_suppression", "-10.0", "42")
    - temperature_0.3_42.jsonl -> ("temperature", "0.3", "42")
    - random_ablation_sample0_42.jsonl -> ("random_ablation", "sample0", "42")
    """
    stem = fname.replace(".jsonl", "")

    # Known methods and their param patterns
    # Try to match from longest method name first
    known_no_param = ["baseline", "prompt_engineering"]
    known_with_param = [
        "neuron_steering", "caa_repeng", "logit_suppression",
        "temperature", "random_ablation"
    ]

    for method in known_no_param:
        if stem.startswith(method + "_"):
            rest = stem[len(method) + 1:]
            return method, None, rest

    for method in known_with_param:
        if stem.startswith(method + "_"):
            rest = stem[len(method) + 1:]
            # rest = "{param}_{seed}" - seed is last part after final underscore
            # But param can have negative sign, dots, etc.
            # seed is always one of: 42, 123, 456
            for seed in ["123", "456", "42"]:
                if rest.endswith("_" + seed):
                    param = rest[:-(len(seed) + 1)]
                    return method, param, seed

    raise ValueError(f"Cannot parse filename: {fname}")


def load_data() -> dict[tuple[str, str | None], dict[tuple[int, str], float]]:
    """Load all JSONL files, return {(method, param): {(prompt_idx, seed): composite_slop}}."""

    data = defaultdict(dict)

    for f in sorted(DATA_DIR.glob("*.jsonl")):
        # Count lines first
        with open(f) as fh:
            lines = fh.readlines()

        if len(lines) < MIN_LINES // 3:  # per-seed files have 75 lines each, skip if < ~67
            # We need at least 75 lines per seed file (225/3 = 75)
            # Skip files with fewer than 200/3 ~ 67 lines
            print(f"  SKIP {f.name} ({len(lines)} lines < threshold)")
            continue

        method, param, seed = parse_filename(f.name)

        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompt_idx = rec["prompt_idx"]
            composite_slop = rec["slop_metrics"]["composite_slop"]
            key = (prompt_idx, seed)
            data[(method, param)][key] = composite_slop

    return dict(data)


def cohens_d(diffs: np.ndarray) -> float:
    """Cohen's d for paired differences (mean / std)."""
    if np.std(diffs, ddof=1) == 0:
        return 0.0
    return np.mean(diffs) / np.std(diffs, ddof=1)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    """95% bootstrap CI for the mean difference."""
    means = np.array([
        RNG.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def main():
    print("Loading data...")
    data = load_data()

    # Get baseline
    baseline_key = ("baseline", None)
    if baseline_key not in data:
        print("ERROR: baseline data not found!")
        sys.exit(1)

    baseline = data[baseline_key]
    print(f"\nBaseline: {len(baseline)} prompt-seed pairs")
    print(f"Baseline mean composite_slop: {np.mean(list(baseline.values())):.3f}")
    print(f"Baseline median composite_slop: {np.median(list(baseline.values())):.3f}")

    # Compare each condition to baseline
    results = []

    for (method, param), condition_data in sorted(data.items()):
        if method == "baseline":
            continue

        # Find matching (prompt_idx, seed) pairs
        common_keys = set(baseline.keys()) & set(condition_data.keys())

        if len(common_keys) < 200:
            label = f"{method}({param})" if param else method
            print(f"  SKIP {label}: only {len(common_keys)} paired observations")
            continue

        # Compute paired differences (condition - baseline)
        # Negative = condition has LESS slop = GOOD
        diffs = np.array([
            condition_data[k] - baseline[k]
            for k in sorted(common_keys)
        ])

        n = len(diffs)
        mean_diff = np.mean(diffs)

        # Wilcoxon signed-rank test
        # Filter out zero differences (Wilcoxon can't handle them)
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) < 10:
            # Too few non-zero diffs for a meaningful test
            w_stat, p_val = np.nan, 1.0
        else:
            w_stat, p_val = stats.wilcoxon(nonzero_diffs, alternative='two-sided')

        # Bootstrap CI
        ci_lo, ci_hi = bootstrap_ci(diffs)

        # Cohen's d
        d = cohens_d(diffs)

        label = f"{method}({param})" if param else method

        results.append({
            "label": label,
            "method": method,
            "param": param,
            "n": n,
            "mean_baseline": np.mean([baseline[k] for k in sorted(common_keys)]),
            "mean_condition": np.mean([condition_data[k] for k in sorted(common_keys)]),
            "mean_diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_val": p_val,
            "d": d,
            "sig": sig_marker(p_val),
        })

    # Sort by absolute effect size (largest first)
    results.sort(key=lambda r: abs(r["d"]), reverse=True)

    # Print table
    print("\n" + "=" * 130)
    print(f"{'Method':<30} {'N':>4} {'Mean BL':>8} {'Mean Cond':>10} {'Mean Diff':>10} {'95% CI':>20} {'Wilcoxon p':>12} {'Cohen d':>8} {'Sig':>4}")
    print("-" * 130)

    for r in results:
        ci_str = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(
            f"{r['label']:<30} {r['n']:>4} {r['mean_baseline']:>8.3f} {r['mean_condition']:>10.3f} "
            f"{r['mean_diff']:>+10.3f} {ci_str:>20} {r['p_val']:>12.2e} {r['d']:>+8.3f} {r['sig']:>4}"
        )

    print("=" * 130)
    print("\nNotes:")
    print("  - Mean Diff = Condition - Baseline (negative = less slop = better)")
    print("  - Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
    print(f"  - Bootstrap CIs based on {N_BOOTSTRAP} resamples")
    print(f"  - Conditions with <200 paired observations skipped")

    # Sanity check: neuron_steering alpha=0.0 vs alpha=1.0
    print("\n" + "=" * 130)
    print("SANITY CHECK: neuron_steering alpha=0.0 (full ablation) vs baseline")
    print("=" * 130)

    ns_00 = data.get(("neuron_steering", "0.0"))
    if ns_00:
        common = set(baseline.keys()) & set(ns_00.keys())
        diffs_00 = np.array([ns_00[k] - baseline[k] for k in sorted(common)])
        print(f"  N = {len(diffs_00)}")
        print(f"  Mean diff (alpha=0.0 - baseline) = {np.mean(diffs_00):+.4f}")
        print(f"  Median diff = {np.median(diffs_00):+.4f}")
        nonzero = diffs_00[diffs_00 != 0]
        if len(nonzero) >= 10:
            _, p = stats.wilcoxon(nonzero, alternative='two-sided')
            print(f"  Wilcoxon p = {p:.4e}")
        print(f"  Cohen's d = {cohens_d(diffs_00):+.4f}")

    # Check if alpha=1.0 exists (should be no-op)
    ns_10 = data.get(("neuron_steering", "1.0"))
    if ns_10:
        print(f"\nneuron_steering alpha=1.0 (no-op) vs baseline:")
        common = set(baseline.keys()) & set(ns_10.keys())
        diffs_10 = np.array([ns_10[k] - baseline[k] for k in sorted(common)])
        print(f"  N = {len(diffs_10)}")
        print(f"  Mean diff = {np.mean(diffs_10):+.4f}")
        nonzero = diffs_10[diffs_10 != 0]
        if len(nonzero) >= 10:
            _, p = stats.wilcoxon(nonzero, alternative='two-sided')
            print(f"  Wilcoxon p = {p:.4e}")
        print(f"  Cohen's d = {cohens_d(diffs_10):+.4f}")
    else:
        print("\n  neuron_steering alpha=1.0 NOT FOUND (no no-op condition available)")

    # Direct comparison: neuron_steering 0.0 vs 0.6 (strongest vs lightest ablation)
    print("\n" + "=" * 130)
    print("DOSE-RESPONSE: neuron_steering across alpha values")
    print("=" * 130)

    ns_conditions = [(m, p) for (m, p) in data.keys() if m == "neuron_steering"]
    ns_conditions.sort(key=lambda x: float(x[1]) if x[1] else 0)

    for method, param in ns_conditions:
        cond = data[(method, param)]
        common = set(baseline.keys()) & set(cond.keys())
        if len(common) < 200:
            print(f"  alpha={param}: SKIPPED ({len(common)} pairs)")
            continue
        diffs = np.array([cond[k] - baseline[k] for k in sorted(common)])
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 10:
            _, p = stats.wilcoxon(nonzero, alternative='two-sided')
        else:
            p = 1.0
        print(
            f"  alpha={param:>4}: mean_diff={np.mean(diffs):+.3f}, "
            f"median_diff={np.median(diffs):+.3f}, "
            f"d={cohens_d(diffs):+.3f}, "
            f"p={p:.2e} {sig_marker(p)}"
        )

    # Random ablation control
    print("\n" + "=" * 130)
    print("CONTROL: random ablation samples vs baseline")
    print("=" * 130)

    ra_conditions = [(m, p) for (m, p) in data.keys() if m == "random_ablation"]
    ra_conditions.sort()

    ra_diffs_all = []
    for method, param in ra_conditions:
        cond = data[(method, param)]
        common = set(baseline.keys()) & set(cond.keys())
        if len(common) < 200:
            print(f"  {param}: SKIPPED ({len(common)} pairs)")
            continue
        diffs = np.array([cond[k] - baseline[k] for k in sorted(common)])
        ra_diffs_all.append(diffs)
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 10:
            _, p = stats.wilcoxon(nonzero, alternative='two-sided')
        else:
            p = 1.0
        print(
            f"  {param}: mean_diff={np.mean(diffs):+.3f}, "
            f"d={cohens_d(diffs):+.3f}, "
            f"p={p:.2e} {sig_marker(p)}"
        )

    if ra_diffs_all:
        pooled = np.concatenate(ra_diffs_all)
        print(f"\n  Pooled random ablation (N={len(pooled)}): "
              f"mean_diff={np.mean(pooled):+.3f}, d={cohens_d(pooled):+.3f}")


if __name__ == "__main__":
    main()
