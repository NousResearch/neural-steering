import json
import numpy as np

files = {
    'Qwen2.5-3B': 'binary_ablation_Qwen2_5_3B.json',
    'Qwen3-4B': 'binary_ablation_Qwen3_4B.json',
    'Llama-3.2-3B': 'binary_ablation_Llama_3.2_3B_Instruct.json',
    'Llama-3.2-1B': 'binary_ablation_Llama_3.2_1B_Instruct.json',
}

behaviors = ['refusal', 'sycophancy', 'belief', 'sentiment']
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

print("=" * 90)
print("INVERSION ANALYSIS: Robust Metrics")
print("=" * 90)

all_results = {}

for model_name, fname in files.items():
    path = f'/home/nightwing/Desktop/Projects/neural-steering/experiments/results/{fname}'
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"{model_name}")
    print(f"{'=' * 70}")

    model_results = {}
    for behavior in behaviors:
        dr = data['behaviors'][behavior]['dose_response']
        curve = []
        for a in alphas:
            p = dr.get(str(a), {}).get('Yes', dr.get(str(a), {}).get(' Yes', 0))
            curve.append(max(p, 1e-12))  # floor to avoid log(0)

        curve = np.array(curve)

        # Metric 1: Log-ratio (robust to small baselines)
        log_ratio = np.log10(curve[-1]) - np.log10(curve[0])

        # Metric 2: Direction classification by pairwise slopes
        # Compare adjacent alpha points: how many show increase vs decrease?
        slopes = np.diff(curve)
        n_increase = np.sum(slopes > 1e-12)
        n_decrease = np.sum(slopes < -1e-12)

        if n_decrease > n_increase + 2:
            direction = "INVERTED"
            symbol = "↓↓"
        elif n_increase > n_decrease + 2:
            direction = "STANDARD"
            symbol = "↑↑"
        elif log_ratio < -1:
            direction = "INVERTED"
            symbol = "↓"
        elif log_ratio > 1:
            direction = "STANDARD"
            symbol = "↑"
        else:
            direction = "FLAT"
            symbol = "→"

        # Metric 3: Peak location (which alpha gives max P(Yes)?)
        peak_alpha = alphas[np.argmax(curve)]

        # Metric 4: Ratio of mid to start (α=1.0 / α=0.0) — less noisy than full sweep
        mid_ratio = curve[4] / max(curve[0], 1e-12)  # index 4 = α=1.0

        model_results[behavior] = {
            'log_ratio': log_ratio,
            'direction': direction,
            'peak_alpha': peak_alpha,
            'mid_ratio': mid_ratio,
            'curve': curve.tolist(),
        }

        print(f"  {behavior:12s}: log_ratio={log_ratio:+.1f}  mid_ratio={mid_ratio:.1f}x  "
              f"peak@α={peak_alpha:.1f}  [{direction}] {symbol}")

    all_results[model_name] = model_results

# Clean summary table for the paper
print(f"\n{'=' * 90}")
print("TABLE: Dose-Response Direction by Model × Behavior")
print(f"{'=' * 90}")
print()
print(f"{'':16s} {'Refusal':>12s} {'Sycophancy':>12s} {'Belief':>12s} {'Sentiment':>12s}")
print("-" * 66)

for model_name in files:
    row = f"{model_name:<16s}"
    for behavior in behaviors:
        r = all_results[model_name][behavior]
        row += f" {r['direction']:>11s}"
    print(row)

# Log-ratio table
print(f"\n{'=' * 90}")
print("TABLE: Log10(P_yes@α=3 / P_yes@α=0)")
print(f"{'=' * 90}")
print()
print(f"{'':16s} {'Refusal':>12s} {'Sycophancy':>12s} {'Belief':>12s} {'Sentiment':>12s} {'Mean':>8s}")
print("-" * 74)

for model_name in files:
    row = f"{model_name:<16s}"
    vals = []
    for behavior in behaviors:
        lr = all_results[model_name][behavior]['log_ratio']
        vals.append(lr)
        row += f" {lr:>+11.1f}"
    row += f" {np.mean(vals):>+7.1f}"
    print(row)

# Architecture-level summary
print(f"\n{'=' * 90}")
print("ARCHITECTURE-LEVEL SUMMARY")
print(f"{'=' * 90}")

qwen_models = ['Qwen2.5-3B', 'Qwen3-4B']
llama_models = ['Llama-3.2-3B', 'Llama-3.2-1B']

for behavior in behaviors:
    qwen_dirs = [all_results[m][behavior]['direction'] for m in qwen_models]
    llama_dirs = [all_results[m][behavior]['direction'] for m in llama_models]
    qwen_lr = [all_results[m][behavior]['log_ratio'] for m in qwen_models]
    llama_lr = [all_results[m][behavior]['log_ratio'] for m in llama_models]

    qwen_consensus = "INVERTED" if qwen_dirs.count("INVERTED") >= 2 else (
        "STANDARD" if qwen_dirs.count("STANDARD") >= 2 else "MIXED")
    llama_consensus = "STANDARD" if llama_dirs.count("STANDARD") >= 2 else (
        "INVERTED" if llama_dirs.count("INVERTED") >= 2 else "MIXED")

    print(f"\n  {behavior}:")
    print(f"    Qwen:  {qwen_consensus:10s} (log-ratio: {np.mean(qwen_lr):+.1f})")
    print(f"    Llama: {llama_consensus:10s} (log-ratio: {np.mean(llama_lr):+.1f})")
    agree = qwen_consensus != llama_consensus and qwen_consensus in ("INVERTED", "STANDARD") and llama_consensus in ("INVERTED", "STANDARD")
    print(f"    Divergent: {'YES ✓' if agree else 'no'}")
