import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load all datasets
datasets = {}
files = {
    'Qwen2.5-3B': 'binary_ablation_Qwen2_5_3B.json',
    'Qwen3-4B': 'binary_ablation_Qwen3_4B.json',
    'Llama-3.2-3B': 'binary_ablation_Llama_3.2_3B_Instruct.json',
    'Llama-3.2-1B': 'binary_ablation_Llama_3.2_1B_Instruct.json',
}

for name, fname in files.items():
    path = f'/home/nightwing/Desktop/Projects/neural-steering/experiments/results/{fname}'
    try:
        with open(path) as f:
            datasets[name] = json.load(f)
    except FileNotFoundError:
        print(f"Missing: {name}")

behaviors = ['refusal', 'sycophancy', 'belief', 'sentiment']
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = {'Qwen2.5-3B': '#2196F3', 'Qwen3-4B': '#FF5722', 'Llama-3.2-3B': '#4CAF50', 'Llama-3.2-1B': '#9C27B0'}

# ---- DOSE RESPONSE COMPARISON ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Binary Ablation Dose Response: Cross-Model Comparison', fontsize=16, fontweight='bold')

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
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xticks(alphas)

plt.tight_layout()
plt.savefig('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/cross_model_dose_response.png', dpi=150, bbox_inches='tight')
print("Saved dose response comparison!")

# ---- CIRCUIT LOCALIZATION ----
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('Circuit Layer Localization: Cross-Model Comparison', fontsize=14, fontweight='bold')

model_layers = {
    'Qwen2.5-3B': 36,
    'Qwen3-4B': 36,
    'Llama-3.2-3B': 28,
    'Llama-3.2-1B': 16,
}

for model_idx, (model_name, model_data) in enumerate(datasets.items()):
    ax = axes2[model_idx // 2][model_idx % 2]
    n_layers = model_layers[model_name]
    
    for beh_idx, behavior in enumerate(behaviors):
        circuits = model_data['behaviors'][behavior]['circuit_layers']
        layers = [c[0] for c in circuits]
        counts = [c[1] for c in circuits]
        ax.scatter(layers, [behavior] * len(layers), s=[c*2 for c in counts], alpha=0.7, label=behavior)
    
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_title(f'{model_name}\n({n_layers} layers)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/cross_model_layers.png', dpi=150, bbox_inches='tight')
print("Saved layer comparison!")

# ---- SUMMARY TABLE ----
print("\n" + "="*80)
print("CROSS-MODEL SUMMARY")
print("="*80)

for model_name, model_data in datasets.items():
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    for behavior in behaviors:
        bdata = model_data['behaviors'][behavior]
        circuits = bdata['circuit_layers']
        top_layer = circuits[0][0]
        n_total_layers = model_layers[model_name]
        pct = (top_layer / n_total_layers) * 100
        
        dr = bdata['dose_response']
        p_yes_0 = dr.get('0.0', {}).get('Yes', dr.get('0.0', {}).get(' Yes', 0))
        p_yes_3 = dr.get('3.0', {}).get('Yes', dr.get('3.0', {}).get(' Yes', 0))
        
        print(f"  {behavior:12s}: top_layer={top_layer}/{n_total_layers} ({pct:.0f}%), "
              f"P(Yes) α=0: {p_yes_0:.2e}, α=3: {p_yes_3:.2e}")
    
    print(f"  Controls:")
    for q, delta in model_data['control'].items():
        print(f"    {q[:45]}: {delta:.6f}")
