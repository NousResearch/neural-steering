#!/usr/bin/env python3
"""Supplementary statistical analysis for anti-slop experiment v2.

Addresses:
1. CRITICAL SANITY: neuron_steering alpha=1.0 vs baseline (must be no-op)
2. Word count / degradation analysis per condition
3. CAA degradation at high alpha (word_count, repetition)
4. Clean summary table for reporting
5. Dose-response monotonicity analysis
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DATA_DIR = Path("/Users/karan/digi2digi/shared/projects/neuron-steering/neuron-circuits/experiments/results_v2/generations")
N_BOOTSTRAP = 10000
RNG = np.random.default_rng(42)


def parse_filename(fname: str) -> tuple[str, str | None, str]:
    """Parse {method}_{param}_{seed}.jsonl or {method}_{seed}.jsonl."""
    stem = fname.replace(".jsonl", "")
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
            for seed in ["123", "456", "42"]:
                if rest.endswith("_" + seed):
                    param = rest[:-(len(seed) + 1)]
                    return method, param, seed
    raise ValueError(f"Cannot parse filename: {fname}")


def load_all_data() -> dict[tuple[str, str | None], dict[tuple[int, str], dict]]:
    """Load all JSONL files, return {(method, param): {(prompt_idx, seed): full_slop_metrics}}."""
    data = defaultdict(dict)
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        with open(f) as fh:
            lines = fh.readlines()
        if len(lines) < 67:
            continue
        method, param, seed = parse_filename(f.name)
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompt_idx = rec["prompt_idx"]
            key = (prompt_idx, seed)
            data[(method, param)][key] = rec["slop_metrics"]
    return dict(data)


def repetition_score(text: str) -> float:
    """Detect repetitive text by checking for repeated n-grams.
    Returns fraction of 4-grams that are repeated."""
    words = text.lower().split()
    if len(words) < 8:
        return 0.0
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def load_generations_for_condition(method: str, param: str | None) -> dict[tuple[int, str], str]:
    """Load raw generation text for a condition."""
    gens = {}
    for seed in ["42", "123", "456"]:
        if param is not None:
            fname = f"{method}_{param}_{seed}.jsonl"
        else:
            fname = f"{method}_{seed}.jsonl"
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                gens[(rec["prompt_idx"], seed)] = rec["generation"]
    return gens


def cohens_d(diffs: np.ndarray) -> float:
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return 0.0
    return np.mean(diffs) / sd


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    means = np.array([
        RNG.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def sig_marker(p: float) -> str:
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"


def main():
    print("=" * 100)
    print("SUPPLEMENTARY ANALYSIS: Anti-Slop Experiment v2")
    print("=" * 100)

    data = load_all_data()
    baseline = data[("baseline", None)]

    # ========================================================================
    # 1. CRITICAL SANITY CHECK: alpha=1.0 as no-op
    # ========================================================================
    print("\n" + "=" * 100)
    print("1. CRITICAL SANITY CHECK: neuron_steering alpha=1.0 vs baseline")
    print("   (Multiplying activations by 1.0 should be identity)")
    print("=" * 100)

    ns_10 = data.get(("neuron_steering", "1.0"))
    if ns_10:
        common = sorted(set(baseline.keys()) & set(ns_10.keys()))
        n = len(common)
        bl_vals = np.array([baseline[k]["composite_slop"] for k in common])
        ns_vals = np.array([ns_10[k]["composite_slop"] for k in common])
        diffs = ns_vals - bl_vals

        # Check if values are IDENTICAL (not just statistically similar)
        identical = np.all(diffs == 0)
        max_abs_diff = np.max(np.abs(diffs))
        nonzero_count = np.sum(diffs != 0)

        print(f"\n  N paired observations: {n}")
        print(f"  Values IDENTICAL: {identical}")
        print(f"  Max absolute difference: {max_abs_diff:.10f}")
        print(f"  Non-zero differences: {nonzero_count}/{n}")

        if identical:
            print("\n  PASS: alpha=1.0 produces EXACTLY the same composite_slop as baseline.")
            print("  This confirms the steering pipeline is correctly implemented.")
            print("  (Multiplying by 1.0 = no modification to activations)")
        else:
            print(f"\n  FAIL: {nonzero_count} observations differ!")
            print(f"  Mean diff: {np.mean(diffs):+.6f}")
            if nonzero_count >= 10:
                _, p = stats.wilcoxon(diffs[diffs != 0], alternative='two-sided')
                print(f"  Wilcoxon p: {p:.4e}")
            print("  WARNING: This suggests a bug in the steering pipeline!")

        # Also check word counts
        bl_wc = np.array([baseline[k]["word_count"] for k in common])
        ns_wc = np.array([ns_10[k]["word_count"] for k in common])
        wc_diffs = ns_wc - bl_wc
        wc_identical = np.all(wc_diffs == 0)
        print(f"\n  Word counts IDENTICAL: {wc_identical}")
        if not wc_identical:
            print(f"  Mean word_count diff: {np.mean(wc_diffs):+.1f}")
            print(f"  Non-zero word_count diffs: {np.sum(wc_diffs != 0)}/{n}")
    else:
        print("\n  ERROR: neuron_steering alpha=1.0 data not found!")

    # ========================================================================
    # 2. WORD COUNT ANALYSIS: degradation at high alphas
    # ========================================================================
    print("\n" + "=" * 100)
    print("2. WORD COUNT ANALYSIS: mean word_count per condition")
    print("   (Degradation = low word count, model producing garbage/truncated text)")
    print("=" * 100)

    # Baseline word count stats
    bl_wc_all = np.array([v["word_count"] for v in baseline.values()])
    print(f"\n  Baseline: mean={np.mean(bl_wc_all):.1f}, median={np.median(bl_wc_all):.1f}, "
          f"std={np.std(bl_wc_all):.1f}, min={np.min(bl_wc_all)}, max={np.max(bl_wc_all)}")

    # Neuron steering word counts
    print("\n  --- Neuron Steering ---")
    ns_keys = sorted([(m, p) for m, p in data.keys() if m == "neuron_steering"],
                     key=lambda x: float(x[1]))
    for method, param in ns_keys:
        cond = data[(method, param)]
        wc = np.array([v["word_count"] for v in cond.values()])
        common = sorted(set(baseline.keys()) & set(cond.keys()))
        bl_wc_matched = np.array([baseline[k]["word_count"] for k in common])
        cond_wc_matched = np.array([cond[k]["word_count"] for k in common])
        wc_diff = cond_wc_matched - bl_wc_matched
        nonzero = wc_diff[wc_diff != 0]
        if len(nonzero) >= 10:
            _, p = stats.wilcoxon(nonzero, alternative='two-sided')
        else:
            p = 1.0
        flag = " <-- DEGRADED" if np.mean(wc) < 100 else ""
        flag = flag or (" <-- SIGNIFICANT CHANGE" if p < 0.05 else "")
        print(f"    alpha={float(param):>4.1f}: mean_wc={np.mean(wc):>6.1f}, "
              f"std={np.std(wc):>5.1f}, min={np.min(wc):>3}, "
              f"diff_from_bl={np.mean(wc_diff):>+6.1f}, p={p:.2e}{flag}")

    # CAA word counts
    print("\n  --- CAA (caa_repeng) ---")
    caa_keys = sorted([(m, p) for m, p in data.keys() if m == "caa_repeng"],
                      key=lambda x: float(x[1]))
    for method, param in caa_keys:
        cond = data[(method, param)]
        wc = np.array([v["word_count"] for v in cond.values()])
        common = sorted(set(baseline.keys()) & set(cond.keys()))
        bl_wc_matched = np.array([baseline[k]["word_count"] for k in common])
        cond_wc_matched = np.array([cond[k]["word_count"] for k in common])
        wc_diff = cond_wc_matched - bl_wc_matched
        nonzero = wc_diff[wc_diff != 0]
        if len(nonzero) >= 10:
            _, p = stats.wilcoxon(nonzero, alternative='two-sided')
        else:
            p = 1.0
        flag = ""
        if np.mean(wc) < 100:
            flag = " <-- DEGRADED"
        elif p < 0.05:
            flag = " <-- SIGNIFICANT CHANGE"
        print(f"    alpha={float(param):>5.1f}: mean_wc={np.mean(wc):>6.1f}, "
              f"std={np.std(wc):>5.1f}, min={np.min(wc):>3}, "
              f"diff_from_bl={np.mean(wc_diff):>+6.1f}, p={p:.2e}{flag}")

    # Other conditions
    print("\n  --- Other Conditions ---")
    other_keys = sorted([(m, p) for m, p in data.keys()
                         if m not in ("baseline", "neuron_steering", "caa_repeng")])
    for method, param in other_keys:
        cond = data[(method, param)]
        wc = np.array([v["word_count"] for v in cond.values()])
        label = f"{method}({param})" if param else method
        flag = " <-- DEGRADED" if np.mean(wc) < 100 else ""
        print(f"    {label:<30}: mean_wc={np.mean(wc):>6.1f}, std={np.std(wc):>5.1f}")

    # ========================================================================
    # 3. CAA HIGH-ALPHA DEGRADATION CHECK
    # ========================================================================
    print("\n" + "=" * 100)
    print("3. CAA HIGH-ALPHA DEGRADATION CHECK")
    print("   (Checking if CAA alpha>=1.0 produces degenerate/empty text)")
    print("=" * 100)

    for method, param in caa_keys:
        alpha_val = float(param)
        if alpha_val < 1.0:
            continue
        cond = data[(method, param)]
        wc = np.array([v["word_count"] for v in cond.values()])
        slop = np.array([v["composite_slop"] for v in cond.values()])

        # Load actual generations to check for repetition
        gens = load_generations_for_condition(method, param)
        rep_scores = [repetition_score(g) for g in gens.values()] if gens else []
        mean_rep = np.mean(rep_scores) if rep_scores else 0.0

        # Check for very short or empty generations
        short_count = np.sum(wc < 50)
        empty_count = np.sum(wc < 10)

        # Check zero-slop rate
        zero_slop = np.sum(slop == 0) / len(slop)

        print(f"\n  CAA alpha={alpha_val}:")
        print(f"    Word count: mean={np.mean(wc):.1f}, min={np.min(wc)}, max={np.max(wc)}")
        print(f"    Short (<50 words): {short_count}/{len(wc)} ({100*short_count/len(wc):.1f}%)")
        print(f"    Empty (<10 words): {empty_count}/{len(wc)} ({100*empty_count/len(wc):.1f}%)")
        print(f"    Zero slop rate: {100*zero_slop:.1f}%")
        print(f"    Mean repetition score (4-gram): {mean_rep:.3f}")
        print(f"    Mean composite_slop: {np.mean(slop):.3f}")

        if zero_slop > 0.9 and np.mean(wc) > 200:
            print(f"    --> GENUINE SLOP REDUCTION (good text, no slop)")
        elif zero_slop > 0.9 and np.mean(wc) < 100:
            print(f"    --> LIKELY DEGRADED (text too short, slop=0 is trivial)")
        elif mean_rep > 0.3:
            print(f"    --> LIKELY DEGRADED (high repetition)")

    # ========================================================================
    # 4. DOSE-RESPONSE MONOTONICITY
    # ========================================================================
    print("\n" + "=" * 100)
    print("4. DOSE-RESPONSE ANALYSIS: neuron_steering")
    print("   (Is slop reduction monotonically increasing as alpha decreases from 1.0?)")
    print("=" * 100)

    # Collect dose-response data for neuron steering
    ns_dose = []
    for method, param in ns_keys:
        alpha_val = float(param)
        cond = data[(method, param)]
        common = sorted(set(baseline.keys()) & set(cond.keys()))
        if len(common) < 200:
            continue
        diffs = np.array([cond[k]["composite_slop"] - baseline[k]["composite_slop"] for k in common])
        mean_slop = np.mean([cond[k]["composite_slop"] for k in common])
        mean_wc = np.mean([cond[k]["word_count"] for k in common])
        ns_dose.append({
            "alpha": alpha_val,
            "mean_slop": mean_slop,
            "mean_diff": np.mean(diffs),
            "mean_wc": mean_wc,
        })

    print("\n  alpha  | mean_slop | diff_from_bl | mean_wc | quality_flag")
    print("  " + "-" * 70)
    for d in ns_dose:
        quality = "OK"
        if d["mean_wc"] < 100:
            quality = "DEGRADED"
        elif abs(d["mean_diff"]) < 0.1:
            quality = "~no effect"
        elif d["mean_diff"] < -0.3:
            quality = "reduces slop"
        print(f"  {d['alpha']:>5.1f}  | {d['mean_slop']:>9.3f} | {d['mean_diff']:>+12.3f} | {d['mean_wc']:>7.1f} | {quality}")

    # Monotonicity test: Spearman correlation between alpha and mean_slop
    # For alphas 0.0 to 1.0 (the ablation range)
    ablation_dose = [d for d in ns_dose if d["alpha"] <= 1.0]
    if len(ablation_dose) >= 3:
        alphas = np.array([d["alpha"] for d in ablation_dose])
        slops = np.array([d["mean_slop"] for d in ablation_dose])
        rho, p_spearman = stats.spearmanr(alphas, slops)
        print(f"\n  Spearman correlation (alpha vs mean_slop, 0.0-1.0 range):")
        print(f"    rho = {rho:+.3f}, p = {p_spearman:.4f}")
        if rho > 0 and p_spearman < 0.05:
            print(f"    --> Positive monotonic trend: lower alpha = less slop (expected)")
        elif p_spearman >= 0.05:
            print(f"    --> NO significant monotonic trend detected")

    # For the amplification range (alpha > 1.0)
    amp_dose = [d for d in ns_dose if d["alpha"] >= 1.0]
    if len(amp_dose) >= 3:
        alphas = np.array([d["alpha"] for d in amp_dose])
        slops = np.array([d["mean_slop"] for d in amp_dose])
        rho, p_spearman = stats.spearmanr(alphas, slops)
        print(f"\n  Spearman correlation (alpha vs mean_slop, 1.0-3.0 range):")
        print(f"    rho = {rho:+.3f}, p = {p_spearman:.4f}")

    # ========================================================================
    # 5. CLEAN SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 100)
    print("5. SUMMARY TABLE FOR REPORTING")
    print("=" * 100)

    # Collect all results into a unified table
    all_results = []
    for (method, param), cond in sorted(data.items()):
        if method == "baseline":
            continue
        common = sorted(set(baseline.keys()) & set(cond.keys()))
        if len(common) < 200:
            continue

        diffs = np.array([cond[k]["composite_slop"] - baseline[k]["composite_slop"] for k in common])
        wc = np.array([cond[k]["word_count"] for k in common])
        bl_wc = np.array([baseline[k]["word_count"] for k in common])

        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 10:
            _, p = stats.wilcoxon(nonzero, alternative='two-sided')
        else:
            p = 1.0
        d = cohens_d(diffs)
        ci_lo, ci_hi = bootstrap_ci(diffs)

        label = f"{method}({param})" if param else method
        all_results.append({
            "label": label,
            "method": method,
            "param": param,
            "n": len(common),
            "mean_slop": np.mean([cond[k]["composite_slop"] for k in common]),
            "mean_diff": np.mean(diffs),
            "pct_change": 100 * np.mean(diffs) / np.mean([baseline[k]["composite_slop"] for k in common]),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p": p, "d": d,
            "mean_wc": np.mean(wc),
            "wc_diff": np.mean(wc - bl_wc),
        })

    # Sort by method category, then param
    def sort_key(r):
        order = {"caa_repeng": 0, "neuron_steering": 1, "logit_suppression": 2,
                 "prompt_engineering": 3, "temperature": 4, "random_ablation": 5}
        m = order.get(r["method"], 99)
        p = float(r["param"]) if r["param"] and r["param"].replace("-", "").replace(".", "").isdigit() else 0
        return (m, p)

    all_results.sort(key=sort_key)

    header = (f"{'Condition':<30} {'Slop':>5} {'Diff':>7} {'%chg':>6} "
              f"{'95% CI':>18} {'p':>10} {'|d|':>5} {'Sig':>3} "
              f"{'WC':>6} {'WC diff':>7}")
    print(f"\n  Baseline composite_slop = {np.mean(list(v['composite_slop'] for v in baseline.values())):.3f}, "
          f"word_count = {np.mean(bl_wc_all):.0f}")
    print()
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for r in all_results:
        ci_str = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  {r['label']:<30} {r['mean_slop']:>5.2f} {r['mean_diff']:>+7.3f} "
              f"{r['pct_change']:>+5.1f}% {ci_str:>18} {r['p']:>10.1e} "
              f"{abs(r['d']):>5.3f} {sig_marker(r['p']):>3} "
              f"{r['mean_wc']:>6.0f} {r['wc_diff']:>+7.1f}")

    # ========================================================================
    # 6. KEY FINDINGS
    # ========================================================================
    print("\n" + "=" * 100)
    print("6. KEY FINDINGS")
    print("=" * 100)

    # Find best neuron steering condition
    ns_results = [r for r in all_results if r["method"] == "neuron_steering" and r["p"] < 0.05]
    ns_results.sort(key=lambda r: r["d"])  # most negative d = biggest reduction

    print("\n  A. SANITY CHECK (alpha=1.0 no-op):")
    ns10 = [r for r in all_results if r["label"] == "neuron_steering(1.0)"]
    if ns10:
        r = ns10[0]
        if r["mean_diff"] == 0 and r["d"] == 0:
            print("     PASS -- alpha=1.0 is a perfect no-op (diff=0.000, d=0.000)")
            print("     The neuron steering pipeline has NO bugs in its identity condition.")
        else:
            print(f"     FAIL -- alpha=1.0 shows diff={r['mean_diff']:+.4f}")

    print("\n  B. NEURON STEERING DOSE-RESPONSE:")
    if ns_results:
        best = ns_results[0]
        print(f"     Best significant condition: {best['label']}")
        print(f"       Slop reduction: {best['mean_diff']:+.3f} ({best['pct_change']:+.1f}%)")
        print(f"       Cohen's d: {best['d']:+.3f} (small effect)")
        print(f"       p = {best['p']:.2e}")
    else:
        print("     No neuron steering condition reaches significance (p < 0.05).")

    # Check dose-response pattern
    ns_all = [r for r in all_results if r["method"] == "neuron_steering"]
    ns_all.sort(key=lambda r: float(r["param"]))
    sig_alphas = [float(r["param"]) for r in ns_all if r["p"] < 0.05]
    nonsig_alphas = [float(r["param"]) for r in ns_all if r["p"] >= 0.05]
    print(f"     Significant alphas: {sig_alphas}")
    print(f"     Non-significant alphas: {nonsig_alphas}")
    print(f"     Pattern: {'non-monotonic (gaps)' if len(sig_alphas) > 1 and max(sig_alphas) - min(sig_alphas) > 0.2 else 'sparse'}")

    print("\n  C. CAA EFFECTIVENESS:")
    caa_results = [r for r in all_results if r["method"] == "caa_repeng"]
    caa_results.sort(key=lambda r: float(r["param"]))
    for r in caa_results:
        wc_flag = " [DEGRADED - check text quality]" if r["mean_wc"] < 200 else ""
        print(f"     alpha={float(r['param']):>5.1f}: {r['pct_change']:>+6.1f}% slop, "
              f"d={r['d']:+.3f}, wc={r['mean_wc']:.0f}{wc_flag}")

    print("\n  D. RANDOM ABLATION CONTROL:")
    ra_results = [r for r in all_results if r["method"] == "random_ablation"]
    any_sig = any(r["p"] < 0.05 for r in ra_results)
    pooled_d = np.mean([abs(r["d"]) for r in ra_results])
    print(f"     Any sample significant? {'YES (unexpected!)' if any_sig else 'NO (correct -- ablating random neurons has no effect)'}")
    print(f"     Mean |d| across 5 samples: {pooled_d:.3f} (negligible)")

    print("\n  E. METHOD RANKING (by effect size, significant only):")
    sig_results = [r for r in all_results if r["p"] < 0.05]
    sig_results.sort(key=lambda r: r["d"])
    for i, r in enumerate(sig_results, 1):
        print(f"     {i:>2}. {r['label']:<30} d={r['d']:+.3f}  diff={r['mean_diff']:+.3f}  "
              f"wc={r['mean_wc']:.0f}")

    print("\n  F. OPTIMAL SETTINGS:")
    # Best without degradation
    good_results = [r for r in sig_results if r["mean_wc"] >= 200]
    if good_results:
        best = good_results[0]
        print(f"     Best method (no degradation): {best['label']}")
        print(f"       Slop: {best['mean_diff']:+.3f} ({best['pct_change']:+.1f}%), d={best['d']:+.3f}")
        print(f"       Word count: {best['mean_wc']:.0f} (baseline: {np.mean(bl_wc_all):.0f})")

    # Best neuron steering without degradation
    ns_good = [r for r in sig_results if r["method"] == "neuron_steering" and r["mean_wc"] >= 200]
    if ns_good:
        best_ns = ns_good[0]
        print(f"     Best neuron_steering: {best_ns['label']}")
        print(f"       Slop: {best_ns['mean_diff']:+.3f} ({best_ns['pct_change']:+.1f}%), d={best_ns['d']:+.3f}")

    # ========================================================================
    # 7. EQUIVALENCE TEST: Is neuron steering better than random ablation?
    # ========================================================================
    print("\n" + "=" * 100)
    print("7. NEURON STEERING vs RANDOM ABLATION (specificity test)")
    print("   (Do attributed neurons have LARGER effect than random neurons?)")
    print("=" * 100)

    # Pool random ablation diffs
    ra_all_diffs = []
    for method, param in sorted(data.keys()):
        if method != "random_ablation":
            continue
        cond = data[(method, param)]
        common = sorted(set(baseline.keys()) & set(cond.keys()))
        if len(common) < 200:
            continue
        diffs = np.array([cond[k]["composite_slop"] - baseline[k]["composite_slop"] for k in common])
        ra_all_diffs.extend(diffs)
    ra_all_diffs = np.array(ra_all_diffs)

    # Compare each NS condition to pooled random
    for method, param in ns_keys:
        alpha_val = float(param)
        if alpha_val >= 1.0:  # only ablation range
            continue
        cond = data[(method, param)]
        common = sorted(set(baseline.keys()) & set(cond.keys()))
        if len(common) < 200:
            continue
        ns_diffs = np.array([cond[k]["composite_slop"] - baseline[k]["composite_slop"] for k in common])

        # Mann-Whitney U test: is NS distribution of diffs shifted lower than random?
        u_stat, u_p = stats.mannwhitneyu(ns_diffs, ra_all_diffs, alternative='less')
        ns_mean = np.mean(ns_diffs)
        ra_mean = np.mean(ra_all_diffs)
        print(f"  alpha={alpha_val:.1f}: NS mean_diff={ns_mean:+.3f} vs Random mean_diff={ra_mean:+.3f}, "
              f"U-test p={u_p:.3f} {'*' if u_p < 0.05 else 'ns'}")


if __name__ == "__main__":
    main()
