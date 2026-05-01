import json
import glob
import os
import re
from pathlib import Path


def ngram_repetition(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def sentence_repetition(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    reps = sum(
        1 for i in range(1, len(sentences))
        if sentences[i] == sentences[i - 1]
    )
    return reps / len(sentences)


def escape_ratio(text: str) -> float:
    # Fraction of tokens that are LaTeX-style escape sequences like \\ \[ \( \m etc.
    # Strong signal for CAA gibberish at high alpha.
    tokens = text.split()
    if not tokens:
        return 0.0
    escape_count = sum(1 for t in tokens if re.match(r'^\\+[\[\](){}a-zA-Z_\-]*$', t))
    return escape_count / len(tokens)


def unicode_noise_ratio(text: str) -> float:
    # Fraction of chars that are non-ASCII, catches CJK/noise injection.
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


def repetitive_word_ratio(tokens: list[str]) -> float:
    # Max single-token frequency / total tokens.
    # A looping response repeating 'prepare prepare prepare...' will spike this.
    if not tokens:
        return 0.0
    from collections import Counter
    counts = Counter(tokens)
    return counts.most_common(1)[0][1] / len(tokens)


def degradation_score(text: str) -> float:
    """
    Combined degradation score in [0, 1]. Higher = more degraded.
    Tuned to catch CAA repetition loops, LaTeX gibberish, and vocabulary collapse.
    """
    if not text or not text.strip():
        return 1.0

    tokens = text.lower().split()

    tri_rep   = ngram_repetition(tokens, 3)
    four_rep  = ngram_repetition(tokens, 4)
    ttr       = type_token_ratio(tokens)
    sent_rep  = sentence_repetition(text)
    esc       = escape_ratio(text)
    uni_noise = unicode_noise_ratio(text)
    word_dom  = repetitive_word_ratio(tokens)

    score = (
        tri_rep   * 0.20 +
        four_rep  * 0.20 +
        (1 - ttr) * 0.15 +
        sent_rep  * 0.10 +
        esc       * 0.20 +
        uni_noise * 0.05 +
        word_dom  * 0.10
    )
    return round(min(score, 1.0), 4)


def quality_score(text: str) -> float:
    return round(1.0 - degradation_score(text), 4)


def score_samples(samples: list[dict]) -> dict:
    scores = [quality_score(s["gen"]) for s in samples]
    degs   = [degradation_score(s["gen"]) for s in samples]
    return {
        "mean_quality":     round(sum(scores) / len(scores), 4),
        "mean_degradation": round(sum(degs) / len(degs), 4),
        "min_quality":      round(min(scores), 4),
        "pct_degraded_50":  round(sum(1 for d in degs if d > 0.5) / len(degs), 4),
        "pct_degraded_75":  round(sum(1 for d in degs if d > 0.75) / len(degs), 4),
        "per_sample": [
            {
                "prompt":      s["prompt"],
                "refusal":     s["refusal"],
                "quality":     quality_score(s["gen"]),
                "degradation": degradation_score(s["gen"]),
            }
            for s in samples
        ]
    }


def process_file(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    model      = data.get("model", Path(path).stem)
    n_layers   = data.get("n_layers")
    alphas     = data.get("alphas", [])
    n_prompts  = data.get("n_prompts")
    baseline   = data.get("baseline", {})

    result = {
        "model":     model,
        "n_layers":  n_layers,
        "n_prompts": n_prompts,
        "baseline_refusal_pct": baseline.get("pct"),
        "alphas": {}
    }

    for method_name, method_data in data.get("methods", {}).items():
        result["alphas"][method_name] = {}
        for alpha_str, alpha_data in method_data.items():
            samples = alpha_data.get("samples", [])
            if not samples:
                continue

            alpha_result = {
                "refusal_pct":     alpha_data.get("pct"),
                "n_refusals":      alpha_data.get("n_refusals"),
                "elapsed_seconds": alpha_data.get("elapsed_seconds"),
            }

            if samples:
                alpha_result.update(score_samples(samples))

            result["alphas"][method_name][alpha_str] = alpha_result

    # Flattened summary table — convenient for plotting
    result["summary"] = []
    for alpha_str in [str(a) for a in alphas]:
        row = {"alpha": float(alpha_str)}
        for method in ["relp", "caa"]:
            m = result["alphas"].get(method, {}).get(alpha_str, {})
            row[f"{method}_refusal_pct"]     = m.get("refusal_pct")
            row[f"{method}_mean_quality"]    = m.get("mean_quality")
            row[f"{method}_mean_degradation"] = m.get("mean_degradation")
            row[f"{method}_pct_degraded_75"] = m.get("pct_degraded_75")
        result["summary"].append(row)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute quality/degradation scores for steering experiment JSONs"
    )
    parser.add_argument(
        "input_glob",
        help="Glob pattern for input JSON files, e.g. 'results/*.json'"
    )
    parser.add_argument(
        "output_path",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Strip per-sample breakdowns from output (smaller file)"
    )
    args = parser.parse_args()

    input_files = sorted(glob.glob(args.input_glob))
    if not input_files:
        print(f"No files matched: {args.input_glob}")
        return

    print(f"Processing {len(input_files)} file(s)...")

    all_results = []
    for path in input_files:
        print(f"  {path}")
        try:
            r = process_file(path)
            if args.no_samples:
                for method_data in r["alphas"].values():
                    for alpha_data in method_data.values():
                        alpha_data.pop("per_sample", None)
            all_results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")

    output = {
        "n_models": len(all_results),
        "models": all_results
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {args.output_path}")

    # Quick sanity print
    for r in all_results:
        print(f"\n{r['model']}")
        print(f"  {'alpha':>6}  {'relp_refusal':>13}  {'relp_quality':>13}  {'caa_refusal':>12}  {'caa_quality':>12}  {'caa_deg75%':>11}")
        for row in r["summary"]:
            print(
                f"  {row['alpha']:>6.2f}  "
                f"{str(row.get('relp_refusal_pct', '')):>13}  "
                f"{str(row.get('relp_mean_quality', '')):>13}  "
                f"{str(row.get('caa_refusal_pct', '')):>12}  "
                f"{str(row.get('caa_mean_quality', '')):>12}  "
                f"{str(row.get('caa_pct_degraded_75', '')):>11}"
            )


if __name__ == "__main__":
    main()