import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS = '/home/nightwing/Desktop/Projects/neural-steering/experiments/results'

with open(f'{RESULTS}/refusal_extended_alphas.json') as f:
    data = json.load(f)

colors = {
    'Qwen2.5-3B': '#1565C0',
    'Llama-3.2-1B': '#2E7D32',
    'Gemma-4-E2B': '#E65100',
}

alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

fig, ax = plt.subplots(figsize=(10, 6))

for name, result in data.items():
    dr = result['dose_response']
    yes_curve = [dr[str(a)]['Yes'] for a in alphas]
    ax.plot(alphas, yes_curve, 'o-', color=colors[name], linewidth=2.5,
            markersize=7, label=f'{name} ({result["n_neurons"]} neurons)')

ax.set_xlabel('Steering Strength (α)', fontsize=13)
ax.set_ylabel('P(Yes) — probability of compliance', fontsize=13)
ax.set_title('Refusal Circuit: Amplification Suppresses Compliance\n(Gate behavior across 3 architectures)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.set_xticks(alphas)

# Annotate the suppression
ax.annotate('Qwen: 50,000×\nsuppression',
            xy=(8, 3e-10), fontsize=9, color='#1565C0', ha='center',
            arrowprops=dict(arrowstyle='->', color='#1565C0'),
            xytext=(7, 1e-8))
ax.annotate('Llama: 2,600×\nsuppression',
            xy=(8, 4.5e-6), fontsize=9, color='#2E7D32', ha='center',
            arrowprops=dict(arrowstyle='->', color='#2E7D32'),
            xytext=(7, 3e-5))
ax.annotate('Gemma: 2,500×\nsuppression',
            xy=(8, 2e-6), fontsize=9, color='#E65100', ha='center',
            arrowprops=dict(arrowstyle='->', color='#E65100'),
            xytext=(7, 1e-5))

plt.tight_layout()
plt.savefig(f'{RESULTS}/refusal_extended_gate.png', dpi=150, bbox_inches='tight')
print("Saved refusal_extended_gate.png")
