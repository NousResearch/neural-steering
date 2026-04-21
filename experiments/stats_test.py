import json
import numpy as np
from scipy import stats

files = {
    'Llama-3.2-3B': 'layer_localization_Llama_3_2_3B_Instruct.json',
    'Llama-3.2-1B': 'layer_localization_Llama_3_2_1B_Instruct.json',
    'Qwen2.5-3B': 'layer_localization_Qwen2_5_3B_Instruct.json',
    'Qwen3-4B': 'layer_localization_Qwen3_4B.json',
}

behavioral_behaviors = ['refusal', 'sycophancy', 'sentiment', 'belief']
factual_behaviors = ['capitals', 'sva']

all_behavioral = []
all_factual = []

print("="*80)
print("STATISTICAL ANALYSIS: Behavioral vs Factual Layer Concentration")
print("="*80)

for model_name, fname in files.items():
    path = f'/home/nightwing/Desktop/Projects/neural-steering/experiments/results/{fname}'
    with open(path) as f:
        data = json.load(f)
    
    n_layers = data['n_layers']
    print(f"\n{'='*60}")
    print(f"{model_name} ({n_layers} layers)")
    print(f"{'='*60}")
    
    beh_vals = []
    fact_vals = []
    
    for beh in behavioral_behaviors:
        c = data['behaviors'][beh]['concentration_top3']
        beh_vals.append(c)
        all_behavioral.append(c)
        print(f"  {beh:12s}: {c:.3f} ({c*100:.1f}% in top 3 layers)")
    
    for beh in factual_behaviors:
        c = data['behaviors'][beh]['concentration_top3']
        fact_vals.append(c)
        all_factual.append(c)
        print(f"  {beh:12s}: {c:.3f} ({c*100:.1f}% in top 3 layers)")
    
    # Per-model t-test
    t_stat, p_val = stats.ttest_ind(beh_vals, fact_vals)
    print(f"\n  Per-model t-test: t={t_stat:.3f}, p={p_val:.4e}")
    print(f"  Behavioral mean: {np.mean(beh_vals):.3f} +/- {np.std(beh_vals):.3f}")
    print(f"  Factual mean:    {np.mean(fact_vals):.3f} +/- {np.std(fact_vals):.3f}")
    print(f"  Effect size (Cohen's d): {(np.mean(beh_vals) - np.mean(fact_vals)) / np.std(beh_vals + fact_vals):.3f}")

# Overall pooled test
print(f"\n{'='*80}")
print("POOLED ACROSS ALL MODELS")
print(f"{'='*80}")

t_stat, p_val = stats.ttest_ind(all_behavioral, all_factual)
print(f"  Behavioral: n={len(all_behavioral)}, mean={np.mean(all_behavioral):.3f} +/- {np.std(all_behavioral):.3f}")
print(f"  Factual:    n={len(all_factual)}, mean={np.mean(all_factual):.3f} +/- {np.std(all_factual):.3f}")
print(f"  t-test: t={t_stat:.3f}, p={p_val:.4e}")
cohens_d = (np.mean(all_behavioral) - np.mean(all_factual)) / np.sqrt((np.std(all_behavioral)**2 + np.std(all_factual)**2) / 2)
print(f"  Cohen's d: {cohens_d:.3f}")

# Mann-Whitney U (non-parametric alternative)
u_stat, p_val_u = stats.mannwhitneyu(all_behavioral, all_factual, alternative='greater')
print(f"  Mann-Whitney U: U={u_stat:.1f}, p={p_val_u:.4e}")

# Per-behavior vs factual
print(f"\n{'='*80}")
print("PER-BEHAVIOR vs FACTUAL (pooled across models)")
print(f"{'='*80}")

for beh in behavioral_behaviors:
    beh_vals = []
    for model_name, fname in files.items():
        path = f'/home/nightwing/Desktop/Projects/neural-steering/experiments/results/{fname}'
        with open(path) as f:
            data = json.load(f)
        beh_vals.append(data['behaviors'][beh]['concentration_top3'])
    
    t_stat, p_val = stats.ttest_ind(beh_vals, all_factual)
    cohens_d = (np.mean(beh_vals) - np.mean(all_factual)) / np.sqrt((np.std(beh_vals)**2 + np.std(all_factual)**2) / 2)
    print(f"  {beh:12s}: mean={np.mean(beh_vals):.3f}, vs factual t={t_stat:.2f}, p={p_val:.4e}, d={cohens_d:.2f}")

print(f"\n{'='*80}")
print("VERDICT")
print(f"{'='*80}")
print(f"  Behavioral circuits concentrate {np.mean(all_behavioral)*100:.1f}% of neurons in top 3 layers")
print(f"  Factual circuits concentrate {np.mean(all_factual)*100:.1f}% of neurons in top 3 layers")
print(f"  Difference: {(np.mean(all_behavioral) - np.mean(all_factual))*100:.1f} percentage points")
print(f"  This difference is {'HIGHLY ' if p_val < 0.001 else ''}statistically significant (p={p_val:.4e})")
print(f"  Effect size is {'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'} (Cohen's d={cohens_d:.3f})")
