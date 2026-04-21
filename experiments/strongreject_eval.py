"""
Re-score existing jailbreak benchmark results using StrongREJECT evaluator.
No regeneration needed — uses saved responses from the benchmark run.

Run: CUDA_VISIBLE_DEVICES=0 python -u experiments/strongreject_eval.py > experiments/results/strongreject_eval.log 2>&1
"""

import sys
sys.path.insert(0, ".")

import json
import torch
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset


def load_benchmark(path):
    """Load benchmark results JSON."""
    with open(path) as f:
        return json.load(f)


def responses_to_dataset(results):
    """Convert benchmark results to a HuggingFace dataset for StrongREJECT."""
    data = {"forbidden_prompt": [], "response": [], "condition": []}

    for r in results["baseline_results"]:
        data["forbidden_prompt"].append(r["prompt"])
        data["response"].append(r["response_start"])
        data["condition"].append("baseline")

    for r in results["ablated_results"]:
        data["forbidden_prompt"].append(r["prompt"])
        data["response"].append(r["response_start"])
        data["condition"].append("ablated")

    return Dataset.from_dict(data)


def main():
    print("=" * 60, flush=True)
    print("StrongREJECT Evaluation", flush=True)
    print("=" * 60, flush=True)

    models = {
        "Llama-3.2-1B-Instruct": "experiments/results/jailbreak_benchmark.json",
        "Qwen2.5-3B-Instruct": "experiments/results/jailbreak_benchmark_Qwen2_5_3B_Instruct.json",
    }

    for model_name, path in models.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        results = load_benchmark(path)
        dataset = responses_to_dataset(results)
        print(f"Loaded {len(dataset)} samples ({len(results['baseline_results'])} baseline + {len(results['ablated_results'])} ablated)", flush=True)

        # Run StrongREJECT fine-tuned evaluator
        print("Running strongreject_finetuned...", flush=True)
        try:
            eval_results = evaluate_dataset(dataset, ["strongreject_finetuned"])
        except Exception as e:
            print(f"Fine-tuned evaluator failed: {type(e).__name__}: {e}", flush=True)
            print("Trying rubric evaluator (needs OPENAI_API_KEY)...", flush=True)
            try:
                eval_results = evaluate_dataset(dataset, ["strongreject_rubric"])
            except Exception as e2:
                print(f"Rubric also failed: {type(e2).__name__}: {e2}", flush=True)
                continue

        # Extract scores
        df = eval_results.to_pandas()

        # Split by condition
        baseline_scores = df[df["condition"] == "baseline"]["score"]
        ablated_scores = df[df["condition"] == "ablated"]["score"]

        print(f"\nStrongREJECT Scores (0-1, higher = more harmful):", flush=True)
        print(f"  Baseline: mean={baseline_scores.mean():.3f}, median={baseline_scores.median():.3f}", flush=True)
        print(f"  Ablated:  mean={ablated_scores.mean():.3f}, median={ablated_scores.median():.3f}", flush=True)
        print(f"  Δ mean:   {baseline_scores.mean() - ablated_scores.mean():.3f}", flush=True)

        # Show some score distributions
        print(f"\nBaseline score distribution:", flush=True)
        for bucket in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            count = ((baseline_scores >= bucket[0]) & (baseline_scores < bucket[1])).sum()
            bar = "█" * int(count / 2)
            print(f"  [{bucket[0]:.1f}-{bucket[1]:.1f}): {count:3d} {bar}", flush=True)

        print(f"\nAblated score distribution:", flush=True)
        for bucket in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            count = ((ablated_scores >= bucket[0]) & (ablated_scores < bucket[1])).sum()
            bar = "█" * int(count / 2)
            print(f"  [{bucket[0]:.1f}-{bucket[1]:.1f}): {count:3d} {bar}", flush=True)

        # Save full scores
        output_path = f"experiments/results/strongreject_{model_name.replace('-', '_').replace('.', '')}.json"
        scores = {
            "model": model_name,
            "evaluator": "strongreject_finetuned",
            "baseline_mean": round(float(baseline_scores.mean()), 4),
            "baseline_median": round(float(baseline_scores.median()), 4),
            "ablated_mean": round(float(ablated_scores.mean()), 4),
            "ablated_median": round(float(ablated_scores.median()), 4),
            "delta_mean": round(float(baseline_scores.mean() - ablated_scores.mean()), 4),
            "baseline_scores": baseline_scores.tolist(),
            "ablated_scores": ablated_scores.tolist(),
        }
        with open(output_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nSaved to {output_path}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
