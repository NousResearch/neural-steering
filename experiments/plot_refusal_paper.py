import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS = '/home/nightwing/Desktop/Projects/neural-steering/experiments/results'

files = {
    'Qwen2.5-3B': 'binary_ablation_Qwen2_5_3B.json',
    'Qwen3-4B': 'binary_ablation_Qwen3_4B.json',
    'Llama-3.2-3B': 'binary_ablation_Llama_3.2_3B_Instruct.json',
    'Llama-3.2-1B': 'binary_ablation_Llama_3.2_1B_Instruct.json',
    'Gemma-4-E2B': 'binary_ablation_gemma_4_E2B.json',
}

colors = {
    'Qwen2.5-3B': '#1565C0',
    'Qwen3-4B': '#1976D2',
    'Llama-3.2-3B': '#2E7D32',
    'Llama-3.2-1B': '#43A047',
    'Gemma-4-E2B': '#E65100',
}

linestyles = {
    'Qwen2.5-3B': '-',
    'Qwen3-4B': '--',
    'Llama-3.2-3B': '-',
    'Llama-3.2-1B': '--',
    'Gemma-4-E2B': '-',
}

alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

# Load data
data = {}
for name, fname in files.items():
    with open(f'{RESULTS}/{fname}') as f:
        data[name] = json.load(f)

def get_yes(dr, alpha):
    entry = dr.get(str(alpha), {})
    return entry.get('Yes', entry.get(' Yes', 0))

def get_no(dr, alpha):
    entry = dr.get(str(alpha), {})
    return entry.get('No', entry.get(' No', 0))

# ============================================================
# FIGURE 1: Refusal dose response — P(Yes) across all 5 models
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Refusal Circuit: Universal Gate Across Architectures', fontsize=15, fontweight='bold')

for name, model_data in data.items():
    dr = model_data['behaviors']['refusal']['dose_response']
    yes_curve = [get_yes(dr, a) for a in alphas]
    ax1.plot(alphas, yes_curve, 'o-', color=colors[name], linestyle=linestyles[name],
             linewidth=2.5, markersize=6, label=name)

ax1.set_xlabel('Alpha (steering strength)', fontsize=12)
ax1.set_ylabel('P(Yes) — probability of compliance', fontsize=12)
ax1.set_title('P(Yes) — Amplification Suppresses Compliance', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('symlog', linthresh=1e-6)
ax1.set_xticks(alphas)
ax1.annotate('All models: P(Yes) → 0\nas alpha increases',
             xy=(2.0, 1e-5), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# P(No) too
for name, model_data in data.items():
    dr = model_data['behaviors']['refusal']['dose_response']
    no_curve = [get_no(dr, a) for a in alphas]
    ax2.plot(alphas, no_curve, 's-', color=colors[name], linestyle=linestyles[name],
             linewidth=2.5, markersize=6, label=name)

ax2.set_xlabel('Alpha (steering strength)', fontsize=12)
ax2.set_ylabel('P(No) — probability of refusal', fontsize=12)
ax2.set_title('P(No) — Ablation Releases Refusal', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('symlog', linthresh=1e-6)
ax2.set_xticks(alphas)

plt.tight_layout()
plt.savefig(f'{RESULTS}/refusal_gate_all_models.png', dpi=150, bbox_inches='tight')
print("Saved refusal_gate_all_models.png")

# ============================================================
# FIGURE 2: Layer localization — where do refusal circuits live?
# ============================================================
fig2, ax3 = plt.subplots(figsize=(10, 5))

model_layers = {
    'Qwen2.5-3B': 36,
    'Qwen3-4B': 36,
    'Llama-3.2-3B': 28,
    'Llama-3.2-1B': 16,
    'Gemma-4-E2B': 35,
}

y_pos = 0
yticks = []
ylabels = []

for name, model_data in data.items():
    n_layers = model_layers[name]
    circuits = model_data['behaviors']['refusal']['circuit_layers']
    
    for layer_idx, count in circuits:
        pct = (layer_idx / (n_layers - 1)) * 100
        ax3.barh(y_pos, count * 0.3, left=pct, height=0.6,
                color=colors[name], alpha=0.7)
    
    # Mark the top layer
    top_pct = (circuits[0][0] / (n_layers - 1)) * 100
    ax3.plot(top_pct, y_pos, 'o', color=colors[name], markersize=10, zorder=5)
    
    yticks.append(y_pos)
    ylabels.append(f"{name}\n({n_layers}L)")
    y_pos += 1

ax3.set_yticks(yticks)
ax3.set_yticklabels(ylabels, fontsize=10)
ax3.set_xlabel('Layer Position (%)', fontsize=12)
ax3.set_title('Refusal Circuit Localization: All Models', fontsize=13, fontweight='bold')
ax3.set_xlim(0, 105)
ax3.grid(True, alpha=0.3, axis='x')
ax3.axvline(x=90, color='gray', linestyle=':', alpha=0.5, label='90% mark')
ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{RESULTS}/refusal_layers_all_models.png', dpi=150, bbox_inches='tight')
print("Saved refusal_layers_all_models.png")

# ============================================================
# FIGURE 3: Ablation effect — what happens when you remove the circuit
# ============================================================
fig3, ax4 = plt.subplots(figsize=(10, 5))

for name, model_data in data.items():
    tests = model_data['behaviors']['refusal']['tests']
    if tests:
        normal_yes = [t['probs_normal'].get('Yes', t['probs_normal'].get(' Yes', 0)) for t in tests]
        ablated_yes = [t['probs_ablated'].get('Yes', t['probs_ablated'].get(' Yes', 0)) for t in tests]
        
        x = range(len(tests))
        ax4.bar([xi - 0.2 for xi in x], normal_yes, 0.35, label=f'{name} (normal)',
                color=colors[name], alpha=0.8)
        ax4.bar([xi + 0.2 for xi in x], ablated_yes, 0.35, label=f'{name} (ablated)',
                color=colors[name], alpha=0.3, hatch='//')

ax4.set_xlabel('Test Prompt', fontsize=12)
ax4.set_ylabel('P(Yes)', fontsize=12)
ax4.set_title('Refusal Ablation: Normal vs Circuit Removed', fontsize=13, fontweight='bold')
ax4.set_xticks(range(3))
ax4.set_xticklabels(['Lock pick', 'Phishing', 'Hack account'], fontsize=10)
ax4.legend(fontsize=8, ncol=2, loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_yscale('symlog', linthresh=1e-6)

plt.tight_layout()
plt.savefig(f'{RESULTS}/refusal_ablation_comparison.png', dpi=150, bbox_inches='tight')
print("Saved refusal_ablation_comparison.png")

print("\nAll refusal-focused figures saved!")
