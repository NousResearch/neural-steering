import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load both datasets
with open('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/binary_ablation_Qwen2_5_3B.json') as f:
    qwen3b = json.load(f)

with open('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/binary_ablation_Qwen3_4B.json') as f:
    qwen4b = json.load(f)

behaviors = ['refusal', 'sycophancy', 'belief', 'sentiment']
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Binary Ablation: Qwen2.5-3B vs Qwen3-4B', fontsize=16, fontweight='bold')

for idx, behavior in enumerate(behaviors):
    ax = axes[idx // 2][idx % 2]
    
    # Qwen2.5-3B data
    dr3b = qwen3b['behaviors'][behavior]['dose_response']
    yes_3b = [dr3b[str(a)].get(' Yes', dr3b[str(a)].get('Yes', 0)) for a in alphas]
    
    # Qwen3-4B data
    dr4b = qwen4b['behaviors'][behavior]['dose_response']
    # Check format
    sample_key = list(dr4b.keys())[0]
    if isinstance(dr4b[sample_key], dict):
        yes_4b = [dr4b[str(a)].get(' Yes', dr4b[str(a)].get('Yes', 0)) for a in alphas]
    else:
        # might be plain list
        yes_4b = dr4b if isinstance(dr4b, list) else [0] * len(alphas)
    
    ax.plot(alphas, yes_3b, 'o-', color='#2196F3', linewidth=2, markersize=6, label='Qwen2.5-3B')
    ax.plot(alphas, yes_4b, 's-', color='#FF5722', linewidth=2, markersize=6, label='Qwen3-4B')
    
    ax.set_xlabel('Alpha (steering strength)', fontsize=11)
    ax.set_ylabel('P(Yes)', fontsize=11)
    ax.set_title(behavior.capitalize(), fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xticks(alphas)

plt.tight_layout()
plt.savefig('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/comparison_qwen3b_vs_4b_dose_response.png', dpi=150, bbox_inches='tight')
print("Saved dose response comparison chart!")

# Now let's also compare circuit layer localization
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Circuit Layer Localization: Qwen2.5-3B vs Qwen3-4B', fontsize=14, fontweight='bold')

for model_idx, (model_data, model_name, n_layers) in enumerate([
    (qwen3b, 'Qwen2.5-3B (36 layers)', 36),
    (qwen4b, 'Qwen3-4B (36 layers)', 36)
]):
    ax = axes2[model_idx]
    for beh_idx, behavior in enumerate(behaviors):
        circuits = model_data['behaviors'][behavior]['circuit_layers']
        # Each circuit is [position, neuron] - position is the layer
        layers = [c[0] for c in circuits]
        weights = list(range(len(layers), 0, -1))  # top circuit has highest weight
        ax.scatter(layers, [behavior] * len(layers), s=[w*40 for w in weights], alpha=0.7, label=behavior)
    
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/nightwing/Desktop/Projects/neural-steering/experiments/results/comparison_qwen3b_vs_4b_layers.png', dpi=150, bbox_inches='tight')
print("Saved layer localization comparison chart!")

# Summary table
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

for behavior in behaviors:
    print(f"\n--- {behavior.upper()} ---")
    
    # Circuit layers
    layers_3b = qwen3b['behaviors'][behavior]['circuit_layers']
    layers_4b = qwen4b['behaviors'][behavior]['circuit_layers']
    top_layer_3b = layers_3b[0][0]
    top_layer_4b = layers_4b[0][0]
    print(f"  Top circuit layer: Qwen2.5-3B={top_layer_3b}, Qwen3-4B={top_layer_4b}")
    
    # Dose response
    dr3b = qwen3b['behaviors'][behavior]['dose_response']
    dr4b = qwen4b['behaviors'][behavior]['dose_response']
    
    p_yes_0_3b = dr3b['0.0'].get(' Yes', dr3b['0.0'].get('Yes', 0))
    p_yes_3_3b = dr3b['3.0'].get(' Yes', dr3b['3.0'].get('Yes', 0))
    
    sample_key = list(dr4b.keys())[0]
    if isinstance(dr4b[sample_key], dict):
        p_yes_0_4b = dr4b['0.0'].get(' Yes', dr4b['0.0'].get('Yes', 0))
        p_yes_3_4b = dr4b['3.0'].get(' Yes', dr4b['3.0'].get('Yes', 0))
    else:
        p_yes_0_4b = 0
        p_yes_3_4b = 0
    
    print(f"  P(Yes) at α=0:  Qwen2.5-3B={p_yes_0_3b:.2e}, Qwen3-4B={p_yes_0_4b:.2e}")
    print(f"  P(Yes) at α=3:  Qwen2.5-3B={p_yes_3_3b:.2e}, Qwen3-4B={p_yes_3_4b:.2e}")

# Controls comparison
print("\n--- CONTROLS P(Yes) ---")
for q in qwen3b['control']:
    p3b = qwen3b['control'][q]
    p4b = qwen4b['control'].get(q, 'N/A')
    p4b_str = f"{p4b:.2e}" if isinstance(p4b, (int, float)) else str(p4b)
    print(f"  {q[:50]}: 3B={p3b:.2e}, 4B={p4b_str}")
