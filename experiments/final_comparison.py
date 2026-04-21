import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load ALL datasets
datasets = {}
files = {
    'Qwen2.5-3B': 'binary_ablation_Qwen2_5_3B.json',
    'Qwen3-4B': 'binary_ablation_Qwen3_4B.json',
    'Llama-3.2-3B': 'binary_ablation_Llama_3.2_3B_Instruct.json',
    'Llama-3.2-1B': 'binary_ablation_Llama_3.2_1B_Instruct.json',
}

for name, fname in files.items():
    path = f'/home/nightwing/Desktop/Projects/neural-steering/experiments/results/{fname}'
    with open(path) as f:
        datasets[name] = json.load(f)

behaviors = ['refusal', 'sycophancy', 'belief', 'sentiment']
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = {
    'Qwen2.5-3B': '#2196F3',
    'Qwen3-4B': '#FF5722',
    'Llama-3.2-3B': '#4CAF50',
    'Llama-3.2-1B': '#9C27B0'
}

# ---- DOSE RESPONSE (P(Yes)) ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Binary Ablation: P(Yes) Dose Response Across All Models', fontsize=16, fontweight='bold')

for idx, behavior in enumerate(behaviors):
    ax = axes[idx // 2][idx % 2]
    
    for model_name, model_data in datasets.items():
        dr = model_data['behaviors'][behavior]['dose_response']
        yes_probs = []
        for a in alphas:
            entry = dr.get(str(a), {})
            yes_probs.append(entry.get('Yes', entry.get(' Yes', 0)))
        
        ax.plot(alphas, yes_probs, 'o-', color=colors[model_name], linewidth=2, markersize=6, label=model_name)
    
    ax.set_xlabel('Alpha (steering strength)', fontsize=11)
    ax.set_ylabel('P(Yes)', fontsize=11)
    ax.set_title(behavior.capitalize(), fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xticks(alphas)

plt.tight_layout()
plt.savefig('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/all_models_p_yes.png', dpi=150, bbox_inches='tight')
print("Saved P(Yes) dose response!")

# ---- DOSE RESPONSE (P(No)) ----
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Binary Ablation: P(No) Dose Response Across All Models', fontsize=16, fontweight='bold')

for idx, behavior in enumerate(behaviors):
    ax = axes2[idx // 2][idx % 2]
    
    for model_name, model_data in datasets.items():
        dr = model_data['behaviors'][behavior]['dose_response']
        no_probs = []
        for a in alphas:
            entry = dr.get(str(a), {})
            no_probs.append(entry.get('No', entry.get(' No', 0)))
        
        ax.plot(alphas, no_probs, 's-', color=colors[model_name], linewidth=2, markersize=6, label=model_name)
    
    ax.set_xlabel('Alpha (steering strength)', fontsize=11)
    ax.set_ylabel('P(No)', fontsize=11)
    ax.set_title(behavior.capitalize(), fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xticks(alphas)

plt.tight_layout()
plt.savefig('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/all_models_p_no.png', dpi=150, bbox_inches='tight')
print("Saved P(No) dose response!")

# ---- CIRCUIT LOCALIZATION (% of layers) ----
fig3, ax3 = plt.subplots(figsize=(12, 6))
fig3.suptitle('Circuit Layer Localization (% of Total Layers)', fontsize=14, fontweight='bold')

model_total_layers = {
    'Qwen2.5-3B': 36,
    'Qwen3-4B': 36,
    'Llama-3.2-3B': 28,
    'Llama-3.2-1B': 16,
}

x_pos = 0
xticks = []
xlabels = []

for model_name, model_data in datasets.items():
    n_layers = model_total_layers[model_name]
    for beh_idx, behavior in enumerate(behaviors):
        circuits = model_data['behaviors'][behavior]['circuit_layers']
        # Top circuit layer as percentage
        top_pct = (circuits[0][0] / (n_layers - 1)) * 100
        bottom_pct = (circuits[-1][0] / (n_layers - 1)) * 100
        
        ax3.barh(x_pos, top_pct - bottom_pct, left=bottom_pct, height=0.6,
                color=colors[model_name], alpha=0.7)
        ax3.plot(top_pct, x_pos, 'o', color=colors[model_name], markersize=8)
        
        xticks.append(x_pos)
        xlabels.append(f"{model_name}\n{behavior}")
        x_pos += 1
    x_pos += 0.5  # Gap between models

ax3.set_yticks(xticks)
ax3.set_yticklabels(xlabels, fontsize=9)
ax3.set_xlabel('Layer Position (%)', fontsize=11)
ax3.set_xlim(0, 105)
ax3.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/all_models_layer_pct.png', dpi=150, bbox_inches='tight')
print("Saved layer percentage chart!")

# ---- FINAL SUMMARY ----
print("\n" + "="*90)
print("FINAL CROSS-MODEL COMPARISON")
print("="*90)

for model_name, model_data in datasets.items():
    n_layers = model_total_layers[model_name]
    print(f"\n{'='*70}")
    print(f"{model_name} ({n_layers} layers)")
    print(f"{'='*70}")
    
    for behavior in behaviors:
        bdata = model_data['behaviors'][behavior]
        circuits = bdata['circuit_layers']
        top_layer = circuits[0][0]
        pct = (top_layer / (n_layers - 1)) * 100
        
        dr = bdata['dose_response']
        p_yes_0 = dr.get('0.0', {}).get('Yes', dr.get('0.0', {}).get(' Yes', 0))
        p_yes_3 = dr.get('3.0', {}).get('Yes', dr.get('3.0', {}).get(' Yes', 0))
        p_no_0 = dr.get('0.0', {}).get('No', dr.get('0.0', {}).get(' No', 0))
        p_no_3 = dr.get('3.0', {}).get('No', dr.get('3.0', {}).get(' No', 0))
        
        # Direction
        yes_dir = "↑" if p_yes_3 > p_yes_0 * 1.5 else ("↓" if p_yes_3 < p_yes_0 * 0.5 else "→")
        no_dir = "↑" if p_no_3 > p_no_0 * 1.5 else ("↓" if p_no_3 < p_no_0 * 0.5 else "→")
        
        print(f"  {behavior:12s}: layer {top_layer}/{n_layers-1} ({pct:.0f}%)  "
              f"Yes: {p_yes_0:.1e}→{p_yes_3:.1e} {yes_dir}  "
              f"No: {p_no_0:.1e}→{p_no_3:.1e} {no_dir}")
