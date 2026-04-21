# Binary Ablation: Cross-Model Comparison
## "Where Do Behaviors Live?" — Supplementary Results

### Models Tested
- Qwen2.5-3B-Instruct (36 layers)
- Qwen3-4B (36 layers)
- Llama-3.2-3B-Instruct (28 layers)
- Llama-3.2-1B-Instruct (16 layers)

### Protocol
Binary yes/no framing for behavioral tests. Circuit discovery via contrastive prompting (positive vs negative examples). Ablation (α=0), amplification (α=2-3), dose-response sweep (α=0 to 3 in 9 steps). Random neuron control for baseline.

---

## Key Findings

### 1. Behavioral Circuits Localized to Final Layers (Universal)
All 4 models place behavioral circuits in the last ~15% of their layer stack:
- Qwen2.5-3B: layers 31-35 / 35 (88-100%)
- Qwen3-4B: layers 31-35 / 35 (88-100%)
- Llama-3.2-3B: layers 23-27 / 27 (85-100%)
- Llama-3.2-1B: layers 11-15 / 15 (70-100%)

**Implication**: Behavioral control is a late-stage processing phenomenon. Early layers handle syntax/semantics; final layers encode behavioral dispositions.

### 2. Qwen vs Llama: Opposite Dose-Response Direction
The effect of amplifying behavioral circuits diverges by architecture:

| Behavior | Qwen (amplification) | Llama (amplification) |
|----------|---------------------|----------------------|
| Refusal  | P(Yes) ↓↓↓          | P(Yes) ↑ (3B), ↓ (1B) |
| Sycophancy | P(Yes) ↓          | P(Yes) ↑↑ (strong in 1B) |
| Belief   | P(Yes) → (flat)     | P(No) ↑↑ (strong)     |
| Sentiment | P(Yes) → (flat)    | P(Yes) ↑ (3B), → (1B) |

**Qwen**: Amplifying behavioral circuits SUPPRESSES the associated behavior. The circuit acts as a "gate" — more activation = more blocking.
**Llama**: Amplifying behavioral circuits ENHANCES the associated behavior. The circuit acts as a "driver" — more activation = more output.

### 3. Belief Circuit: Llama's Factual Correction Engine
The most dramatic finding is in the belief behavior:
- Llama-3B: P(No) for "Is the Earth flat?" drops from 0.895 (normal) to 0.00002 (ablated)
- Llama-1B: P(No) drops from 0.471 to 0.001
- Removing the belief circuit DESTROYS the model's ability to correct false statements

The dose-response shows P(No) recovering monotonically with steering strength:
- α=0: P(No) ≈ 0
- α=1.0: P(No) = 0.47 (Llama-1B) / 0.89 (Llama-3B)
- α=1.5+: P(No) > 0.99 (Llama-3B)

This is the cleanest causal evidence in the dataset.

### 4. Size Scaling: Larger Models Have Stronger Circuits
- Llama-1B: weaker effects, noisier curves, lower peak P(No)
- Llama-3B: strong clean effects across all behaviors
- Qwen: 3B and 4B show similar patterns but 4B has more baseline signal

### 5. Random Controls Confirm Causality
Max P(Yes) delta from random neuron ablation:
- Llama-3B: 0.000-0.012
- Llama-1B: 0.000-0.027
- All behaviors: 0.001-0.895 (targeted circuits)

Random neurons produce ~100x smaller effects than targeted circuits.

---

## Figures
- `all_models_p_yes.png`: P(Yes) dose response across all models
- `all_models_p_no.png`: P(No) dose response across all models
- `all_models_layer_pct.png`: Circuit layer localization as % of total layers
- `cross_model_dose_response.png`: Three-model comparison (pre-1B)
- `comparison_qwen3b_vs_4b_dose_response.png`: Qwen head-to-head
- `comparison_qwen3b_vs_4b_layers.png`: Qwen layer localization

## Data Files
- `binary_ablation_Qwen2_5_3B.json`
- `binary_ablation_Qwen3_4B.json`
- `binary_ablation_Llama_3.2_3B_Instruct.json`
- `binary_ablation_Llama_3.2_1B_Instruct.json`
