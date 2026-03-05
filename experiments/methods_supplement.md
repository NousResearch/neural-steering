# Supplementary Methods: Neural Circuit Discovery and Evaluation Pipeline

**Jake Henderson (simpolism) — 2026-03-05**
**Codebase: DamascusGit/neural-steering | Model: Llama-3.1-8B-Instruct | Hardware: NVIDIA A40 48GB**

---

## 1. Model and Environment

All experiments use **Meta-Llama-3.1-8B-Instruct** loaded in float16 on a single NVIDIA A40 (48GB VRAM). Prompts use the model's chat template (system + user message + generation prompt) for all tasks except SVA, which uses raw completion. All measurements are at the last token position unless otherwise stated.

Implementation: `neuron_steer/core.py` (~1600 lines). Depends only on PyTorch and HuggingFace Transformers.

---

## 2. Circuit Discovery Methods

### 2.1 Relevance Propagation (RelP)

Our primary method. Follows Arora et al. (2025, arxiv 2601.22594 / TransluceAI) but extends their implementation to behavioral tasks, which they did not attempt.

**Forward pass**: Standard forward through the model with three linearization rules active:

1. **LN-rule**: RMSNorm coefficients (weight * rsqrt) are detached from the computational graph but preserved in the forward pass. This linearizes normalization layers while retaining their scaling effect.

2. **AH-rule**: Attention computed with eager mode (full Q/K/V/O matmuls, no SDPA/FlashAttention) to maintain full autograd through attention heads.

3. **Half-rule (Shapley)**: The MLP gate-up multiply (`gate_proj(x) * up_proj(x)`) is bilinear. The Half-rule attributes half the product to each factor, avoiding the interaction term. Specifically, the neuron activation saved for attribution is `0.5 * (gate * up_detached + gate_detached * up)` where `_detached` means stop-gradient.

**Backward pass**: From the target token's logit (or logit difference), standard backpropagation through the linearized graph produces gradients at each MLP intermediate neuron.

**Attribution**: For each neuron at layer `l`, position `p`, index `n`:
```
attribution(l, p, n) = gradient(l, p, n) * activation(l, p, n)
```
This is a single forward+backward pass — no path integration (unlike Integrated Gradients).

**Sparsification**: Top 200 neurons per layer per position are retained; the rest are discarded. BOS position (position 0) neurons are filtered out by default. Infrastructure neurons (L0-L1) can optionally be excluded.

**Multi-prompt aggregation**: `discover_circuit_multi()` runs RelP on each discovery prompt independently, then averages attributions across prompts for neurons appearing at corresponding positions. The union of all prompted neurons forms the raw circuit.

**Selection**: TransluceAI's percentage method — keep neurons where `|attribution| >= threshold * |total_logit_diff|`. Default threshold = 0.005 (0.5%). Alternatively, fixed top-k selection.

**Counterfactual handling**: For percentage selection, backward from target logit alone (TransluceAI's approach). For top-k/threshold selection, backward from logit_diff (target - auto-detected second-highest logit, or specified counterfactual).

**Code**: `NeuronSteerer.discover_circuit()` and `discover_circuit_multi()` in `neuron_steer/core.py`.

### 2.2 Contrastive Discovery

Used only in the cross-method control experiment (Section 5.1). Serves as a baseline to demonstrate methodological confounds.

Runs all positive and negative prompts through the model, hooks into `down_proj` inputs to capture MLP intermediate activations at the last token position, computes mean activation per neuron for each set, selects top-k neurons by `|mean_positive - mean_negative|`. No gradients. No linearization.

**Code**: `NeuronSteerer.discover_contrastive()` in `neuron_steer/core.py`.

---

## 3. Measurement Space

All causal measurements (ablation, transplant) are reported in **logit-margin space**:

```
logit_margin = logit(target_token) - max(logit(other_tokens))
```

This is more stable than raw probability for causal claims because it avoids softmax compression effects. A positive margin means the target token is the model's top prediction; negative means it is not. Effect sizes are computed relative to baselines (no intervention) and normalized against random neuron controls (reported in σ).

**Code**: `measure_R()` in `experiments/surgical_ablation.py` returns `{prob, logit, logit_margin}`.

---

## 4. Evaluation Pipeline

Each task proceeds through up to five stages. Stages are independent — each reads the previous stage's outputs but can be run separately.

### 4.1 Circuit Topology (Stage 1)

**Script**: `experiments/circuit_topology.py`

**k\* search**: Finds the minimal sufficient circuit size. Discovers a large circuit (top-k = Kmax), then evaluates necessity (N_H) at every prefix k = 1, 2, ..., Kmax, where the prefix is the top-k neurons ranked by attribution magnitude. k\* is the smallest k where N_H >= τ (default τ = 0.95 of the full circuit's N_H).

**Necessity metric (N_H)**: Zero-ablate the circuit (set all circuit neuron activations to 0 at all positions), measure mean R(x) drop across held-out test prompts:
```
N_H = mean_prompts[ R_baseline(prompt) - R_ablated(prompt) ]
```

**Edge attribution**: For each target neuron in the circuit (top-k by attribution), backpropagates through the linearized model from that neuron's activation to find how much each earlier circuit neuron contributes. Edge weight = `d(target_act) / d(source_act) * source_act`. This produces a directed graph within the circuit.

**Position-aware topology**: Hub analysis groups neurons by `(layer, neuron)` but reports both collapsed and per-position degree. This corrects a position-collapsing inflation artifact: when the same neuron appears at multiple positions across prompts, collapsing inflates degree by 1.7-4.0x.

**Bottleneck identification**: Neurons with high in-degree AND out-degree (joint bottleneck score = in_deg * out_deg) are candidates for causal testing.

**Output**: Circuit JSON, analysis JSON (layer distributions, hub rankings, bottleneck candidates), k\* curve.

### 4.2 Surgical Ablation (Stage 2)

**Script**: `experiments/surgical_ablation.py`

Tests whether individual bottleneck neurons are causally necessary.

**Protocol**:
1. Baseline R(x) on held-out test prompts (no intervention)
2. For each bottleneck neuron `(layer, neuron)`:
   - Zero-ablate at ALL positions → measure R(x) drop (primary)
   - Zero-ablate at LAST position only → measure R(x) drop (comparison)
3. Small-set ablation: top-2, top-3, top-5 bottleneck neurons simultaneously
4. Full circuit ablation (reference, should match k\* analysis)
5. Random single-neuron controls: 20 random (layer, neuron) pairs drawn uniformly from all layers × intermediate_size. Their R(x) drops form the null distribution.

**Effect sizes**: `delta_margin / std(random_deltas)` = σ score. A neuron at 10σ has an ablation effect 10 standard deviations above random.

**Ablation mechanism**: `steer_neurons()` context manager hooks into each relevant layer's `down_proj` input (the gated intermediate activation, dimension = `intermediate_size` = 14336 for Llama-8B). For `multiplier=0.0`, the target neuron's activation is zeroed at either all positions or a specific position. This operates on the MLP intermediate representation, not the MLP output (which has `hidden_size` = 4096).

**Universal neuron filtering**: Optionally auto-detects "super-weight" neurons that appear across all task circuits (via `Circuit.find_universal_neurons()`) and excludes them from the bottleneck candidate list. These are typically L0/L1 embedding infrastructure neurons.

**Output**: JSON with per-neuron results, small-set results, random controls, sigma scores.

### 4.3 Synergy Search (Stage 3)

**Script**: `experiments/synergy_search.py`

Tests whether pairs/triples of neurons combine super-additively.

**Protocol**:
1. Re-measure single-neuron effects (consistency check)
2. Exhaustive pair search on top-24 bottleneck neurons (276 pairs for 24 neurons)
3. Greedy triple search from top-5 pairs: for each top pair, try adding each remaining neuron
4. Progressive ablation: ablate neurons cumulatively, ranked by single-neuron causal effect
5. Random pair controls (50 random pairs from the full neuron space)
6. Synergy ratio for each pair: `pair_effect / (single_i_effect + single_j_effect)`. Ratio ≈ 1.0 = additive; >> 1.0 = synergistic; << 1.0 = sub-additive.

**Output**: JSON with pair matrix, triple results, progressive curve, synergy ratios.

### 4.4 Sufficiency Test (Stage 4)

**Script**: `experiments/sufficiency_test.py`

Tests whether circuit activations can induce the target behavior on control prompts where it doesn't naturally occur.

**Protocol**:
1. **Activation collection**: For each bottleneck neuron, hook `down_proj` input to capture its activation at the last token position, average across source prompts (e.g., harmful prompts for refusal). This gives `mean_act(layer, neuron)`.

2. **Full-circuit transplant**: Inject collected activations from ALL circuit neurons into control prompts (e.g., benign prompts). Measure R(x) shift:
   ```
   dS = R_transplant(control) - R_baseline(control)
   ```
   This is the sufficiency ceiling.

3. **Single-neuron transplant**: For each bottleneck neuron, transplant its activation alone into control prompts. Identifies which neurons are individually sufficient.

4. **Progressive transplant**: Transplant neurons cumulatively, ranked by single-neuron sufficiency, to find how many neurons are needed to approach full-circuit sufficiency.

5. **Amplification**: Multiply top neurons' activations by 2x, 3x, 5x on benign prompts (alternative to transplant).

6. **Random neuron controls**: 20 random neurons, transplant each, form null distribution.

**Transplant mechanism**: Uses the same `steer_neurons()` hook infrastructure but injects collected mean activations rather than zeroing. The hook replaces the neuron's natural activation at the last position with the pre-collected value from source prompts.

**Control prompt selection**: Critical design choice. Controls must elicit the OPPOSITE behavior from target prompts. For refusal (target "I"), controls are benign prompts that elicit helpful responses. For fc_belief (target "Yes"), controls are FC_BELIEF_NO prompts where the model naturally says "No" (e.g., "Is the earth flat? Answer yes or no:"). Using controls where the model already produces the target token creates a ceiling effect and yields null sufficiency (we observed this and corrected it).

**Output**: JSON with full-circuit dS, per-neuron dS, progressive curve, amplification results, random controls.

### 4.5 Activation Probing (Ad Hoc)

**Script**: `experiments/probe_n14331.py`

Targeted investigation: determines whether a neuron encodes a specific behavior or a general response token.

**Protocol**:
1. Construct six categories of forced-choice prompts: harmful_no, benign_yes, opinion_yes, false_no, factual_yes, factual_no
2. For each prompt, hook `down_proj` input to capture the neuron's gated intermediate activation at the last token position
3. Simultaneously measure P("Yes") and P("No") via softmax on final logits
4. Compute:
   - Mean activation per category
   - `Corr(activation, P_no)` across all 30 prompts: if strong, neuron encodes the "No" token regardless of context
   - `Corr(activation, is_harmful)` across all 30 prompts: if strong, neuron encodes refusal-specific content
5. Verdict: if `|Corr(act, P_no)|` >> `|Corr(act, is_harmful)|`, neuron is a general "No" neuron, not refusal-specific.

### 4.6 Cross-Task Ablation (Ad Hoc)

**Script**: `experiments/shared_neuron_ablation.py`

Tests functional specificity of neurons that appear in multiple circuits by RelP attribution.

**Protocol**:
1. Select neurons appearing in 2+ circuits (e.g., fc_refusal ∩ fc_belief)
2. For each neuron, zero-ablate and measure P(target) on FIVE prompt sets: fc_refusal (P("No")), fc_belief (P("Yes")), fc_benign (P("Yes")), fc_belief_no (P("No")), open_refusal (P("I"))
3. Also test all shared neurons ablated simultaneously
4. 20 random neuron controls per prompt set for sigma scoring
5. A neuron is "functionally shared" only if ablation significantly affects BOTH tasks.

---

## 5. Experimental Designs

### 5.1 Cross-Method Control (Experiment 1)

**Question**: Is the paper's layer dichotomy (factual=distributed, behavioral=late) a property of the tasks or the methods?

**Design**: Apply BOTH methods to BOTH task types:

| | Factual (Capitals) | Behavioral (Refusal) |
|---|---|---|
| RelP | Standard | **Novel** |
| Contrastive | **Critical test** | Standard |

RelP uses 12 capital city prompts with per-prompt target tokens; contrastive uses same prompts split into positive/negative sets. Both use top-k=200.

**Analysis**: Compare weighted mean layer and late-layer fraction (% neurons in L28+) across all four cells. Compute Jaccard overlap between methods on the same task.

### 5.2 Open-Ended Refusal Pipeline

**Target token**: `"I"` (token ID 40). Refusal responses start "I cannot..." after the chat template's `\n\n`.

**Discovery prompts**: 10 harmful prompts (REFUSAL_DISCOVERY_POSITIVE). Negative set for contrastive: 10 matched benign prompts.

**Test prompts**: 5 held-out harmful prompts (REFUSAL_TEST), zero overlap with discovery.

**Control prompts for sufficiency**: 10 benign prompts (BENIGN_PROMPTS).

**Seed response**: empty string (model generates from scratch).

### 5.3 Factual (Capitals) Pipeline

**Target tokens**: Per-prompt (e.g., " Austin", " Paris").

**Discovery prompts**: 12 capital city questions.

**Test prompts**: 5 held-out capital questions.

**Seed response**: `"Answer: "` (matches protocol from Arora et al.).

### 5.4 SVA (Subject-Verb Agreement)

**Target tokens**: Per-prompt (" is" or " are").

**Prompts**: 10 sentences with number-distractor constructions (e.g., "The keys to the cabinet" → " are").

**Chat template**: DISABLED. SVA requires raw completion (model completes the sentence directly).

### 5.5 Sycophancy (Null Result)

**Target token**: `"I"` (agreement responses start "I agree..." or "I think...").

**Discovery**: 15 opinion prompts with strong user stance (SYCOPHANCY_DISCOVERY_POSITIVE) vs. 15 hedged versions (SYCOPHANCY_DISCOVERY_NEGATIVE).

**Test**: 5 held-out opinion prompts.

**Outcome**: Null. k\*=5, dMargin=-0.27. Verified via generation rollouts: model produces balanced responses, not agreement. RLHF has eliminated sycophantic behavior in this model.

### 5.6 Forced-Choice Refusal

**Target token**: `"No"` (model refuses to help with harmful requests).

**Prompt format**: `"Can you [harmful request]? Answer yes or no:"` — forces binary response.

**Discovery**: 10 harmful forced-choice prompts.

**Test**: 5 held-out harmful forced-choice prompts.

**Control (specificity)**: 5 benign forced-choice prompts (FC_BENIGN), where model says "Yes".

### 5.7 Forced-Choice Belief

**Target token**: `"Yes"` (model affirms prosocial/factual statements).

**Discovery**: 10 opinion/value questions where model says "Yes" (e.g., "Is education important?").

**Test**: 5 held-out opinion questions.

**Control (sufficiency)**: FC_BELIEF_NO — 5 prompts where model naturally says "No" (e.g., "Is climate change a hoax?"). Earlier attempts used FC_BENIGN as controls, but the model already says "Yes" to benign requests, creating a ceiling that masked sufficiency effects.

---

## 6. Prompt Datasets

All prompts are in `experiments/prompts.py`. Discovery and test sets are strictly non-overlapping (enforced by code comment and manual verification). Prompt counts:

| Dataset | Discovery | Test | Control |
|---|---|---|---|
| Refusal (open-ended) | 10 + 10 neg | 5 | 10 benign |
| Capitals (factual) | 12 | 5 | — |
| SVA | 10 | — | — |
| Sycophancy | 15 + 15 neg | 5 | — |
| FC Refusal | 10 | 5 | 5 benign |
| FC Belief | 10 | 5 | 5 "No" |
| FC Belief "No" | 10 | 5 | — |

---

## 7. Statistical Controls

**Random neuron baselines**: Every experiment includes 20 random single-neuron controls drawn uniformly from (layer ∈ [0, 31], neuron ∈ [0, 14335]). Effect sizes are reported as σ = `(observed_delta - mean_random) / std_random`.

**Held-out test prompts**: All measurements after circuit discovery use prompts NOT seen during discovery. No prompt appears in both discovery and test sets.

**No hyperparameter tuning on test prompts**: k\* is determined entirely on discovery prompts. Test prompts are used only for final evaluation.

**Specificity controls**: Ablation and transplant effects are measured on BOTH target prompts (should show effects) and control prompts (should not show effects). This catches "lobotomy" failure modes where interventions degrade general model behavior.

**Coherence checks**: For high-N_H ablations (>0.5), the model's generation is inspected to verify it still produces grammatical output rather than degenerate repetition.

---

## 8. Key Infrastructure Details

### 8.1 Neuron Indexing

MLP neurons are indexed as `(layer, position, neuron)` where:
- `layer` ∈ [0, 31] for Llama-3.1-8B (32 transformer layers)
- `position` ∈ [0, T-1] for sequence length T
- `neuron` ∈ [0, 14335] for intermediate_size = 14336

The "neuron" refers to the gated intermediate activation: `gate_proj(x) * up_proj(x)`, which is the input to `down_proj`. This has dimension `intermediate_size` (14336), NOT `hidden_size` (4096).

### 8.2 Hook Placement

All hooks are placed on `model.model.layers[l].mlp.down_proj` as forward pre-hooks. The input to `down_proj` IS the gated intermediate activation. This captures the neuron AFTER the nonlinearity (SiLU gate * up) but BEFORE the projection back to residual dimension.

### 8.3 Universal / Super-Weight Neurons

Four neurons appear in every circuit regardless of task: L0/N491, L0/N8268, L1/N198, L1/N2427. These are embedding-layer infrastructure (super-weights). TransluceAI maintains a hardcoded blacklist of 12 such neurons for Llama-3-8B; we additionally detect them empirically via `Circuit.find_universal_neurons()` which counts cross-circuit overlap.

---

## 9. Reproducibility

All experiment scripts are self-contained and runnable:

```bash
# Full pipeline for a task
python experiments/circuit_topology.py --model llama8b --task behavioral
python experiments/surgical_ablation.py --model llama8b --task behavioral --topology_base experiments/topology_llama8b_TIMESTAMP/
python experiments/synergy_search.py --model llama8b --task behavioral --topology_base experiments/topology_llama8b_TIMESTAMP/
python experiments/sufficiency_test.py --model llama8b --task behavioral --topology_base experiments/topology_llama8b_TIMESTAMP/

# Cross-method control
python experiments/cross_method_control.py --model llama8b

# Ad hoc probing
python experiments/probe_n14331.py
python experiments/shared_neuron_ablation.py
```

Random seeds: numpy RNG seeded with 42 for all random neuron controls. PyTorch operations are deterministic where possible (eager attention, no FlashAttention).

All results are saved as JSON with full numerical precision. Result directories include timestamps for versioning.
