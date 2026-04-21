import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

files = {
    'Qwen2.5-3B': 'binary_ablation_Qwen2_5_3B.json',
    'Qwen3-4B': 'binary_ablation_Qwen3_4B.json',
    'Llama-3.2-3B': 'binary_ablation_Llama_3.2_3B_Instruct.json',
    'Llama-3.2-1B': 'binary_ablation_Llama_3.2_1B_Instruct.json',
}

alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
RESULTS = '/home/nightwing/Desktop/Projects/neural-steering/experiments/results'

# Load all data
data = {}
for name, fname in files.items():
    with open(f'{RESULTS}/{fname}') as f:
        data[name] = json.load(f)

def get_yes(dr, alpha):
    entry = dr.get(str(alpha), {})
    return entry.get('Yes', entry.get(' Yes', 0))

# ============================================================
# FIGURE 1: Gate vs Driver — Sycophancy (best example)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Circuit Sign: Qwen (Gate) vs Llama (Driver)', fontsize=15, fontweight='bold')

behaviors_to_plot = ['sycophancy', 'belief', 'sentiment']

for idx, behavior in enumerate(behaviors_to_plot):
    ax = axes[idx]

    for model_name, model_data in data.items():
        dr = model_data['behaviors'][behavior]['dose_response']
        yes_curve = [get_yes(dr, a) for a in alphas]

        if 'Qwen' in model_name:
            color = '#2196F3' if '2.5' in model_name else '#FF5722'
            ls = '-' if '2.5' in model_name else '--'
        else:
            color = '#4CAF50' if '3B' in model_name else '#9C27B0'
            ls = '-' if '3B' in model_name else '--'

        ax.plot(alphas, yes_curve, 'o-', color=color, linestyle=ls,
                linewidth=2, markersize=5, label=model_name)

    ax.set_xlabel('Alpha (steering strength)', fontsize=11)
    ax.set_ylabel('P(Yes)', fontsize=11)
    ax.set_title(behavior.capitalize(), fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xticks(alphas)

    # Annotate gate vs driver
    ax.annotate('GATE\n(Qwen)', xy=(2.5, ax.get_ylim()[0]),
                fontsize=9, color='#1565C0', ha='center', style='italic')
    ax.annotate('DRIVER\n(Llama)', xy=(2.5, ax.get_ylim()[1]),
                fontsize=9, color='#2E7D32', ha='center', style='italic')

plt.tight_layout()
plt.savefig(f'{RESULTS}/gate_vs_driver.png', dpi=150, bbox_inches='tight')
print("Saved gate_vs_driver.png")

# ============================================================
# FIGURE 2: Refusal — Universal Gate
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.suptitle('Refusal: Universal Gate (All Models)', fontsize=14, fontweight='bold')

for model_name, model_data in data.items():
    dr = model_data['behaviors']['refusal']['dose_response']
    yes_curve = [get_yes(dr, a) for a in alphas]

    if 'Qwen' in model_name:
        color = '#2196F3' if '2.5' in model_name else '#FF5722'
    else:
        color = '#4CAF50' if '3B' in model_name else '#9C27B0'

    ax2.plot(alphas, yes_curve, 'o-', color=color, linewidth=2,
             markersize=6, label=model_name)

ax2.set_xlabel('Alpha (steering strength)', fontsize=12)
ax2.set_ylabel('P(Yes)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('symlog', linthresh=1e-8)
ax2.set_xticks(alphas)
ax2.annotate('All models: amplification\nsuppresses P(Yes)\n→ refusal is a GATE',
             xy=(2.0, 1e-6), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{RESULTS}/refusal_universal_gate.png', dpi=150, bbox_inches='tight')
print("Saved refusal_universal_gate.png")

# ============================================================
# FIGURE 3: Combined summary — 2x2 grid with circuit sign labels
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('Circuit Sign Analysis: Dose-Response by Behavior', fontsize=15, fontweight='bold')

behaviors_all = ['refusal', 'sycophancy', 'belief', 'sentiment']
sign_labels = {
    'refusal': {'qwen': 'GATE', 'llama': 'GATE'},
    'sycophancy': {'qwen': 'GATE', 'llama': 'DRIVER'},
    'belief': {'qwen': 'GATE', 'llama': 'DRIVER'},
    'sentiment': {'qwen': 'GATE', 'llama': 'DRIVER'},
}

for idx, behavior in enumerate(behaviors_all):
    ax = axes3[idx // 2][idx % 2]

    qwen_yes = []
    llama_yes = []

    for model_name, model_data in data.items():
        dr = model_data['behaviors'][behavior]['dose_response']
        yes_curve = [get_yes(dr, a) for a in alphas]

        if 'Qwen' in model_name:
            qwen_yes.append(yes_curve)
            color = '#2196F3' if '2.5' in model_name else '#FF5722'
            ls = '-'
        else:
            llama_yes.append(yes_curve)
            color = '#4CAF50' if '3B' in model_name else '#9C27B0'
            ls = '--'

        ax.plot(alphas, yes_curve, 'o-', color=color, linestyle=ls,
                linewidth=2, markersize=4, alpha=0.7, label=model_name)

    # Add shaded regions for Qwen and Llama average
    if qwen_yes:
        qwen_mean = np.mean(qwen_yes, axis=0)
        ax.fill_between(alphas,
                        np.min(qwen_yes, axis=0),
                        np.max(qwen_yes, axis=0),
                        color='#2196F3', alpha=0.1)
    if llama_yes:
        llama_mean = np.mean(llama_yes, axis=0)
        ax.fill_between(alphas,
                        np.min(llama_yes, axis=0),
                        np.max(llama_yes, axis=0),
                        color='#4CAF50', alpha=0.1)

    ax.set_xlabel('Alpha', fontsize=10)
    ax.set_ylabel('P(Yes)', fontsize=10)

    sl = sign_labels[behavior]
    title = f"{behavior.capitalize()}  [Qwen: {sl['qwen']}, Llama: {sl['llama']}]"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-7)
    ax.set_xticks(alphas)

plt.tight_layout()
plt.savefig(f'{RESULTS}/circuit_sign_summary.png', dpi=150, bbox_inches='tight')
print("Saved circuit_sign_summary.png")

print("\nDone! All figures saved.")
