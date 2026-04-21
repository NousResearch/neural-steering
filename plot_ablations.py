import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('experiments/results/refusal_ablations.json') as f:
    d = json.load(f)

models = ['Qwen2.5-3B', 'Llama-3.2-1B', 'Gemma-4-E2B']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, model in enumerate(models):
    ax = axes[i]
    md = d[model]
    
    # Actual circuit dose response (extended alphas)
    dr_actual = md['layer_locality']['top_200_actual']['dose_response']
    alphas = sorted(dr_actual.keys(), key=float)
    yes_actual = [dr_actual[a]['Yes'] for a in alphas]
    
    # Random top layers control
    random_region = [r for r in md['layer_locality'] if 'top_layers' in r][0]
    dr_random = md['layer_locality'][random_region]['dose_response']
    yes_random = [dr_random[a]['Yes'] for a in alphas]
    
    alpha_floats = [float(a) for a in alphas]
    
    ax.plot(alpha_floats, yes_actual, 'o-', color='#FF5722', linewidth=2.5, markersize=8, label='Actual circuit (200 neurons)')
    ax.plot(alpha_floats, yes_random, 's--', color='#9E9E9E', linewidth=2, markersize=7, label='Random top layers (200 neurons)')
    
    ax.set_xlabel('Steering strength (α)', fontsize=12)
    ax.set_ylabel('P(Yes)', fontsize=12)
    ax.set_title(f'{model}', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks([0, 1, 2, 3, 5, 7, 10])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add suppression annotation
    suppression = yes_actual[0] / yes_actual[-1]
    ax.annotate(f'{suppression:.0f}× suppression', xy=(8, yes_actual[-2]), fontsize=10, 
                color='#FF5722', fontweight='bold',
                xytext=(6, yes_actual[3]), arrowprops=dict(arrowstyle='->', color='#FF5722'))

plt.suptitle('Extended Dose Response: Actual Circuit vs Random Neurons', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('experiments/results/extended_alpha_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to experiments/results/extended_alpha_comparison.png")

# Also make a combined sparsity + locality + extended figure
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 15))

for row, model in enumerate(models):
    md = d[model]
    
    # --- Plot 1: Prompt Robustness ---
    ax = axes2[row, 0]
    prompts = list(md['prompt_robustness'].keys())
    labels = [p.replace('Can you help me ', '').replace('Can you ', '')[:22] for p in prompts]
    normal_yes = [md['prompt_robustness'][p]['normal']['Yes'] for p in prompts]
    ablated_yes = [md['prompt_robustness'][p]['ablated']['Yes'] for p in prompts]
    amplified_yes = [md['prompt_robustness'][p]['amplified']['Yes'] for p in prompts]
    
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, normal_yes, w, label='Normal (α=1)', color='#4CAF50')
    ax.bar(x, ablated_yes, w, label='Ablated (α=0)', color='#FF5722')
    ax.bar(x + w, amplified_yes, w, label='Amplified (α=3)', color='#2196F3')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('P(Yes)')
    ax.set_title(f'{model} — Prompt Robustness')
    ax.legend(fontsize=6, loc='upper right')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-7)
    
    # --- Plot 2: Extended Dose Response ---
    ax = axes2[row, 1]
    dr_actual = md['layer_locality']['top_200_actual']['dose_response']
    alphas = sorted(dr_actual.keys(), key=float)
    alpha_floats = [float(a) for a in alphas]
    yes_actual = [dr_actual[a]['Yes'] for a in alphas]
    
    random_region = [r for r in md['layer_locality'] if 'top_layers' in r][0]
    dr_random = md['layer_locality'][random_region]['dose_response']
    yes_random = [dr_random.get(a, {}).get('Yes', np.nan) for a in alphas]
    
    ax.plot(alpha_floats, yes_actual, 'o-', color='#FF5722', linewidth=2.5, markersize=8, label='Actual circuit')
    ax.plot(alpha_floats, yes_random, 's--', color='#9E9E9E', linewidth=2, markersize=7, label='Random top layers')
    ax.set_xlabel('α')
    ax.set_ylabel('P(Yes)')
    ax.set_title(f'{model} — Extended Dose Response')
    ax.set_yscale('log')
    ax.set_xticks([0, 1, 3, 5, 7, 10])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: Layer Locality ---
    ax = axes2[row, 2]
    regions = list(md['layer_locality'].keys())
    reg_labels = ['Actual\ncircuit', 'Bottom\nlayers', 'Middle\nlayers', 'Top\nlayers']
    for j, (alpha, color, label) in enumerate(zip(['0.0', '1.0', '3.0'], ['#FF5722', '#4CAF50', '#2196F3'], ['α=0 (ablate)', 'α=1 (normal)', 'α=3 (amplify)'])):
        vals = [md['layer_locality'][r]['dose_response'][alpha]['Yes'] for r in regions]
        x = np.arange(len(regions))
        ax.bar(x + j*0.25, vals, 0.25, label=label, color=color)
    ax.set_xticks(np.arange(len(regions)) + 0.25)
    ax.set_xticklabels(reg_labels, fontsize=9)
    ax.set_ylabel('P(Yes)')
    ax.set_title(f'{model} — Layer Locality')
    ax.set_yscale('log')
    ax.legend(fontsize=6, loc='upper right')
    ax.set_ylim(bottom=1e-7)

plt.suptitle('Refusal Ablation Experiments — 3 Architectures (Extended α)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('experiments/results/ablation_experiments_extended.png', dpi=150, bbox_inches='tight')
print("Saved to experiments/results/ablation_experiments_extended.png")
