import json
import numpy as np

# ============================================================
# PART 1: Inversion Ratio from existing dose-response data
# ============================================================

files = {
    'Qwen2.5-3B': 'binary_ablation_Qwen2_5_3B.json',
    'Qwen3-4B': 'binary_ablation_Qwen3_4B.json',
    'Llama-3.2-3B': 'binary_ablation_Llama_3.2_3B_Instruct.json',
    'Llama-3.2-1B': 'binary_ablation_Llama_3.2_1B_Instruct.json',
}

behaviors = ['refusal', 'sycophancy', 'belief', 'sentiment']
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

print("=" * 90)
print("PART 1: INVERSION RATIO ANALYSIS")
print("=" * 90)
print()
print("Inversion ratio = (P(Yes, α=3) - P(Yes, α=0)) / max(P(Yes, α=0), ε)")
print("Negative = inverted (amplification suppresses behavior)")
print("Positive = standard (amplification enhances behavior)")
print()

# Collect all ratios for the summary table
all_ratios = {}

for model_name, fname in files.items():
    path = f'/home/nightwing/Desktop/Projects/neural-steering/experiments/results/{fname}'
    with open(path) as f:
        data = json.load(f)

    print(f"{'=' * 70}")
    print(f"{model_name}")
    print(f"{'=' * 70}")

    model_ratios = {}
    for behavior in behaviors:
        dr = data['behaviors'][behavior]['dose_response']
        p_yes_0 = dr.get('0.0', {}).get('Yes', dr.get('0.0', {}).get(' Yes', 0))
        p_yes_3 = dr.get('3.0', {}).get('Yes', dr.get('3.0', {}).get(' Yes', 0))

        epsilon = 1e-10
        ratio = (p_yes_3 - p_yes_0) / max(p_yes_0, epsilon)

        # Also compute the full curve shape
        curve = []
        for a in alphas:
            p = dr.get(str(a), {}).get('Yes', dr.get(str(a), {}).get(' Yes', 0))
            curve.append(p)

        # Determine monotonicity
        diffs = [curve[i+1] - curve[i] for i in range(len(curve)-1)]
        n_up = sum(1 for d in diffs if d > epsilon)
        n_down = sum(1 for d in diffs if d < -epsilon)
        if n_down > n_up * 2:
            trend = "MONOTONIC ↓ (inverted)"
        elif n_up > n_down * 2:
            trend = "MONOTONIC ↑ (standard)"
        elif max(curve) > min(curve) * 10:
            trend = "NON-MONOTONIC (U/N-shaped)"
        else:
            trend = "FLAT"

        direction = "INVERTED" if ratio < -0.5 else ("STANDARD" if ratio > 0.5 else "WEAK/FLAT")

        model_ratios[behavior] = {
            'ratio': ratio,
            'p_yes_0': p_yes_0,
            'p_yes_3': p_yes_3,
            'trend': trend,
            'direction': direction,
            'curve': curve,
        }

        print(f"  {behavior:12s}: α=0: {p_yes_0:.2e} → α=3: {p_yes_3:.2e}  "
              f"ratio={ratio:+.2f}  [{direction}]  {trend}")

    all_ratios[model_name] = model_ratios

# Summary table
print(f"\n{'=' * 90}")
print("SUMMARY: Inversion Ratio Table (for paper)")
print(f"{'=' * 90}")
print()
print(f"{'Model':<16} {'Refusal':>10} {'Sycophancy':>12} {'Belief':>10} {'Sentiment':>11} {'Mean':>8}")
print("-" * 70)

for model_name, ratios in all_ratios.items():
    vals = [ratios[b]['ratio'] for b in behaviors]
    mean_val = np.mean(vals)
    row = f"{model_name:<16}"
    for v in vals:
        marker = "↓" if v < -0.5 else ("↑" if v > 0.5 else "·")
        row += f" {v:+.2f}{marker:>1}"
        row += " " * (10 - len(f"{v:+.2f}{marker}")) if len(behaviors) > 2 else ""
    row += f"  {mean_val:+.2f}"
    print(row)

# Architecture comparison
print(f"\n{'=' * 90}")
print("ARCHITECTURE COMPARISON")
print(f"{'=' * 90}")

qwen_models = ['Qwen2.5-3B', 'Qwen3-4B']
llama_models = ['Llama-3.2-3B', 'Llama-3.2-1B']

for behavior in behaviors:
    qwen_ratios = [all_ratios[m][behavior]['ratio'] for m in qwen_models]
    llama_ratios = [all_ratios[m][behavior]['ratio'] for m in llama_models]
    print(f"\n  {behavior}:")
    print(f"    Qwen mean:  {np.mean(qwen_ratios):+.2f} (range: {min(qwen_ratios):+.2f} to {max(qwen_ratios):+.2f})")
    print(f"    Llama mean: {np.mean(llama_ratios):+.2f} (range: {min(llama_ratios):+.2f} to {max(llama_ratios):+.2f})")
    print(f"    Delta:      {np.mean(llama_ratios) - np.mean(qwen_ratios):+.2f}")
