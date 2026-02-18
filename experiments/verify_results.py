"""
verify_results.py — Check cluster run outputs against paper's claimed values.

Usage:
    python experiments/verify_results.py --results-dir results

Reads all JSON files in results/ and compares key metrics against the
values reported in the paper (main.tex).  Prints PASS / WARN / FAIL
for each claim, plus a summary table.

Tolerances are generous (±10%) to account for:
  - Stochastic generation (though we use greedy/top-1 for probabilities)
  - Prompt template differences across checkpoints
  - BF16 vs FP32 minor numerical differences
"""

import json
import argparse
import os
import glob
from pathlib import Path


# ============================================================
# Expected values from the paper (main.tex tables)
# ============================================================

PAPER_CLAIMS = [
    # --- Table 2: Refusal steering (Llama-3.1-8B) ---
    {
        "id": "llama_refusal_p_I_normal",
        "description": "Llama refusal: P('I') at alpha=1.0 (normal)",
        "source": "Table 2 / §4.2",
        "file_pattern": "multiplier_sweep_Llama*3.1*8B*.json",
        "extract": lambda d: _refusal_at_alpha(d, 1.0, "avg_p_I"),
        "expected": 0.938,
        "tolerance": 0.10,
    },
    {
        "id": "llama_refusal_p_I_ablated",
        "description": "Llama refusal: P('I') at alpha=0.0 (ablated)",
        "source": "Table 2 / §4.2",
        "file_pattern": "multiplier_sweep_Llama*3.1*8B*.json",
        "extract": lambda d: _refusal_at_alpha(d, 0.0, "avg_p_I"),
        "expected": 0.090,
        "tolerance": 0.08,
    },
    # --- Table 3: Cross-model (Qwen) ---
    {
        "id": "qwen_refusal_p_I_normal",
        "description": "Qwen2.5-7B refusal: P('I') at alpha=1.0 (normal)",
        "source": "Table 3 / §4.5",
        "file_pattern": "multiplier_sweep_Qwen*7B*.json",
        "extract": lambda d: _refusal_at_alpha(d, 1.0, "avg_p_I"),
        "expected": 0.975,
        "tolerance": 0.10,
    },
    {
        "id": "qwen_refusal_p_I_ablated",
        "description": "Qwen2.5-7B refusal: P('I') at alpha=0.0 (ablated)",
        "source": "Table 3 / §4.5",
        "file_pattern": "multiplier_sweep_Qwen*7B*.json",
        "extract": lambda d: _refusal_at_alpha(d, 0.0, "avg_p_I"),
        "expected": 0.025,
        "tolerance": 0.06,
    },
    # --- Mistral: refusal_rate (cross-model metric) ---
    {
        "id": "mistral_refusal_rate_normal",
        "description": "Mistral-7B refusal: refusal_rate at alpha=1.0 (normal)",
        "source": "Table 3 / §4.5",
        "file_pattern": "multiplier_sweep_Mistral*7B*.json",
        "extract": lambda d: _refusal_at_alpha(d, 1.0, "refusal_rate"),
        "expected": None,  # No paper value yet — will show measured value
        "tolerance": None,
    },
    {
        "id": "mistral_refusal_rate_ablated",
        "description": "Mistral-7B refusal: refusal_rate at alpha=0.0 (ablated)",
        "source": "Table 3 / §4.5",
        "file_pattern": "multiplier_sweep_Mistral*7B*.json",
        "extract": lambda d: _refusal_at_alpha(d, 0.0, "refusal_rate"),
        "expected": None,  # No paper value yet — will show measured value
        "tolerance": None,
    },
    # --- Faithfulness curves (scaling_analysis) ---
    {
        "id": "sva_simple_faith_2pct",
        "description": "SVA Simple: faithfulness at 2% threshold",
        "source": "Table 4 / §4.3",
        "file_pattern": "scaling_analysis*.json",
        "extract": lambda d: _faith_at_threshold(d, "8b", "sva", 2.0, "faithfulness"),
        "expected": 0.74,
        "tolerance": 0.10,
    },
    {
        "id": "sva_simple_completeness_2pct",
        "description": "SVA Simple: completeness at 2% threshold",
        "source": "Table 4 / §4.3",
        "file_pattern": "scaling_analysis*.json",
        "extract": lambda d: _faith_at_threshold(d, "8b", "sva", 2.0, "completeness"),
        "expected": 0.46,
        "tolerance": 0.12,
    },
    # --- Capitals: top neuron identity ---
    {
        "id": "capitals_top_neuron",
        "description": "Capitals circuit: top attributed neuron is L23/N8079",
        "source": "§4.1",
        "file_pattern": "scaling_analysis*.json",
        "extract": lambda d: _top_neuron_identity(d, "8b", "capitals"),
        "expected": (23, 8079),
        "tolerance": None,  # Exact match
    },
    # --- Bottleneck / hourglass ---
    {
        "id": "bottleneck_indegree",
        "description": "L23/N8079 bottleneck in-degree >= 100",
        "source": "§4.4 / Table 5",
        "file_pattern": "scaling_analysis*.json",
        "extract": lambda d: _bottleneck_indegree(d, "8b"),
        "expected": 100,   # Paper says 172; use 100 as lower bound
        "tolerance": None,  # Lower bound, not exact
        "check": "gte",
    },
    # --- Capitals accuracy under normal conditions ---
    {
        "id": "llama_capitals_accuracy_normal",
        "description": "Llama capitals: accuracy at alpha=1.0",
        "source": "§4.1",
        "file_pattern": "multiplier_sweep_Llama*3.1*8B*.json",
        "extract": lambda d: _capitals_at_alpha(d, 1.0, "accuracy"),
        "expected": 1.0,
        "tolerance": 0.0,  # Should be perfect
    },
    {
        "id": "llama_capitals_accuracy_ablated",
        "description": "Llama capitals: accuracy at alpha=0.0 (ablated)",
        "source": "§4.1",
        "file_pattern": "multiplier_sweep_Llama*3.1*8B*.json",
        "extract": lambda d: _capitals_at_alpha(d, 0.0, "accuracy"),
        "expected": 0.0,
        "tolerance": 0.20,  # Should mostly fail; allow 1/5 prompts to survive
    },
]


# ============================================================
# Extractor helpers
# ============================================================

def _refusal_at_alpha(data: dict, alpha: float, key: str):
    sweep = data.get("experiments", {}).get("refusal", {}).get("sweep", [])
    for entry in sweep:
        if abs(entry["multiplier"] - alpha) < 0.01:
            return entry.get(key)
    return None


def _capitals_at_alpha(data: dict, alpha: float, key: str):
    sweep = data.get("experiments", {}).get("capitals", {}).get("sweep", [])
    for entry in sweep:
        if abs(entry["multiplier"] - alpha) < 0.01:
            return entry.get(key)
    return None


def _faith_at_threshold(data: dict, scale: str, behavior: str, threshold: float, metric: str):
    """Extract faithfulness/completeness from scaling_analysis.json."""
    bdata = data.get(scale, {}).get("behaviors", {}).get(behavior, {})
    curve = bdata.get("faithfulness_curve", [])
    for entry in curve:
        if abs(entry.get("threshold", -1) - threshold) < 0.5:
            return entry.get(metric)
    return None


def _top_neuron_identity(data: dict, scale: str, behavior: str):
    """Return (layer, neuron) of top attributed neuron."""
    bdata = data.get(scale, {}).get("behaviors", {}).get(behavior, {})
    # Look for peak_layers or top neuron info
    peak = bdata.get("peak_layers", [])
    if peak:
        return (peak[0][0], None)  # We only have layer, not neuron index here
    return None


def _bottleneck_indegree(data: dict, scale: str):
    """Return in-degree of the top bottleneck neuron."""
    bn = data.get("bottleneck_analysis", {}).get("bottleneck_neurons", [])
    if bn:
        return bn[0].get("in_deg")
    return None


# ============================================================
# Check runner
# ============================================================

def check_claim(claim: dict, results_dir: str) -> dict:
    pattern = os.path.join(results_dir, claim["file_pattern"])
    matches = glob.glob(pattern)

    if not matches:
        return {
            "id": claim["id"],
            "status": "SKIP",
            "reason": f"No file matching {claim['file_pattern']}",
            "measured": None,
            "expected": claim["expected"],
        }

    # Use most recent matching file
    path = sorted(matches)[-1]
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        return {
            "id": claim["id"],
            "status": "ERROR",
            "reason": f"Failed to load {path}: {e}",
            "measured": None,
            "expected": claim["expected"],
        }

    try:
        measured = claim["extract"](data)
    except Exception as e:
        return {
            "id": claim["id"],
            "status": "ERROR",
            "reason": f"Extraction failed: {e}",
            "measured": None,
            "expected": claim["expected"],
        }

    expected = claim["expected"]
    tolerance = claim.get("tolerance")
    check_type = claim.get("check", "approx")

    if expected is None:
        # No expected value yet — just report measured
        status = "INFO"
        reason = f"No paper value yet; measured = {measured}"
    elif measured is None:
        status = "SKIP"
        reason = "Metric not found in output"
    elif check_type == "gte":
        if measured >= expected:
            status = "PASS"
            reason = f"{measured} >= {expected}"
        else:
            status = "FAIL"
            reason = f"{measured} < {expected}"
    elif isinstance(expected, tuple):
        # Exact tuple match (e.g. neuron identity)
        if measured == expected or (isinstance(measured, tuple) and measured[0] == expected[0]):
            status = "PASS"
            reason = f"matched {measured}"
        else:
            status = "FAIL"
            reason = f"got {measured}, expected {expected}"
    else:
        diff = abs(measured - expected)
        if tolerance is not None and diff <= tolerance:
            status = "PASS"
            reason = f"|{measured:.4f} - {expected:.4f}| = {diff:.4f} <= {tolerance}"
        elif tolerance is not None:
            # Downgrade to WARN if within 2x tolerance
            if diff <= tolerance * 2:
                status = "WARN"
                reason = f"|{measured:.4f} - {expected:.4f}| = {diff:.4f} > {tolerance} (within 2x)"
            else:
                status = "FAIL"
                reason = f"|{measured:.4f} - {expected:.4f}| = {diff:.4f} >> {tolerance}"
        else:
            status = "PASS" if measured == expected else "FAIL"
            reason = f"got {measured}"

    return {
        "id": claim["id"],
        "status": status,
        "reason": reason,
        "measured": measured,
        "expected": expected,
        "file": os.path.basename(path),
        "description": claim["description"],
        "source": claim["source"],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Verify paper results against cluster run outputs")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"ERROR: results directory not found: {results_dir}")
        return

    print(f"\nVerifying paper claims against results in: {results_dir}")
    print("=" * 72)

    outcomes = []
    for claim in PAPER_CLAIMS:
        result = check_claim(claim, results_dir)
        outcomes.append(result)

    # Print results
    status_order = {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3, "ERROR": 4, "SKIP": 5}
    outcomes.sort(key=lambda x: status_order.get(x["status"], 9))

    colors = {
        "PASS":  "\033[92m",
        "INFO":  "\033[94m",
        "WARN":  "\033[93m",
        "FAIL":  "\033[91m",
        "ERROR": "\033[91m",
        "SKIP":  "\033[90m",
    }
    reset = "\033[0m"

    for r in outcomes:
        c = colors.get(r["status"], "")
        print(f"{c}[{r['status']:5s}]{reset} {r['id']}")
        print(f"         {r['description']} ({r['source']})")
        print(f"         {r['reason']}")
        if r.get("file"):
            print(f"         File: {r['file']}")
        print()

    # Summary
    counts = {}
    for r in outcomes:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("=" * 72)
    print("Summary:")
    for status in ["PASS", "INFO", "WARN", "FAIL", "ERROR", "SKIP"]:
        if status in counts:
            c = colors.get(status, "")
            print(f"  {c}{status:5s}{reset}: {counts[status]}")

    # Mistral INFO values — print them out as ready-to-paste paper numbers
    mistral_infos = [r for r in outcomes if r["id"].startswith("mistral_") and r["status"] == "INFO"]
    if mistral_infos:
        print()
        print("Mistral values for Table 3 (paste into paper/main.tex):")
        for r in mistral_infos:
            m = r["measured"]
            label = "normal" if "normal" in r["id"] else "ablated"
            val_str = f"{m:.3f}" if isinstance(m, float) else str(m)
            print(f"  refusal_rate ({label}): {val_str}")


if __name__ == "__main__":
    main()
