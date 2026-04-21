#!/usr/bin/env python3
"""
Autoreason-style batch runner for neural-steering experiments.
Follows the pattern from NousResearch/autoreason — batch processing,
retries, incremental saves, unattended operation.

Usage:
    cd neural-steering
    source neuralvenv/bin/activate
    python experiments/run_experiments.py                    # run 1B model
    python experiments/run_experiments.py --model both       # run both 1B and 8B
    python experiments/run_experiments.py --model 8b         # run 8B only
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_with_retry(func, max_retries=3, delay=10, *args, **kwargs):
    """Run a function with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  All {max_retries} attempts failed: {e}")
                traceback.print_exc()
                raise


def run_single_model(model_name: str, output_dir: str, top_k: int = 200):
    """Run layer localization for a single model with retry logic."""
    from experiments.layer_localization import run_experiment

    print(f"\n{'#'*70}")
    print(f"# MODEL: {model_name}")
    print(f"# TIME:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    results = run_with_retry(
        run_experiment,
        max_retries=3,
        delay=15,
        model_name=model_name,
        output_dir=output_dir,
        top_k=top_k,
    )

    return results


def generate_comparison_report(results_dir: str):
    """Generate a comparison report across all completed runs."""
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("layer_localization_*.json"))

    if len(json_files) < 2:
        print(f"\nOnly {len(json_files)} result file(s) found. Need 2+ for comparison.")
        return

    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON REPORT")
    print(f"{'='*70}\n")

    all_results = {}
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        model = data["model"].split("/")[-1]
        all_results[model] = data

    # Comparison table
    behaviors = None
    for model, data in all_results.items():
        if behaviors is None:
            behaviors = list(data["behaviors"].keys())
        print(f"\n--- {model} ({data.get('n_layers', '?')} layers) ---")
        print(f"{'Behavior':<15} {'Type':<12} {'Top3%':<10} {'Top25%':<10}")
        print("-" * 50)
        for b in behaviors:
            bd = data["behaviors"].get(b, {})
            if "error" in bd:
                print(f"{b:<15} {'-':<12} ERROR: {bd['error'][:40]}")
            elif "concentration_top3" in bd:
                cat = "behavioral" if bd["type"] == "contrastive" else "factual"
                print(f"{b:<15} {cat:<12} {bd['concentration_top3']:<10.1%} {bd['concentration_top_quarter']:<10.1%}")

    # Save combined report
    report_path = results_path / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Autoreason-style experiment runner")
    parser.add_argument("--model", default="1b", choices=["1b", "8b", "both"],
                        help="Which model(s) to run")
    parser.add_argument("--output-dir", default="experiments/results",
                        help="Output directory")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Number of neurons per circuit")
    args = parser.parse_args()

    models = {
        "1b": "meta-llama/Llama-3.2-1B-Instruct",
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
    }

    if args.model == "both":
        model_list = [models["1b"], models["8b"]]
    else:
        model_list = [models[args.model]]

    os.makedirs(args.output_dir, exist_ok=True)

    # Run log
    log_path = Path(args.output_dir) / "run_log.json"
    run_log = {
        "started": datetime.now().isoformat(),
        "models": model_list,
        "top_k": args.top_k,
        "completed": [],
        "failed": [],
    }

    all_results = {}
    for model_name in model_list:
        t0 = time.time()
        try:
            results = run_single_model(model_name, args.output_dir, args.top_k)
            elapsed = time.time() - t0
            all_results[model_name] = results
            run_log["completed"].append({
                "model": model_name,
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            elapsed = time.time() - t0
            run_log["failed"].append({
                "model": model_name,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            })

        # Save log incrementally
        with open(log_path, "w") as f:
            json.dump(run_log, f, indent=2)

    # Generate comparison if multiple models ran
    if len(all_results) > 1:
        generate_comparison_report(args.output_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("EXPERIMENT RUN COMPLETE")
    print(f"{'='*70}")
    print(f"Completed: {len(run_log['completed'])}")
    print(f"Failed:    {len(run_log['failed'])}")
    print(f"Results in: {args.output_dir}")
    print(f"Log: {log_path}")

    if run_log["failed"]:
        print("\nFailed models:")
        for f in run_log["failed"]:
            print(f"  {f['model']}: {f['error']}")


if __name__ == "__main__":
    main()
