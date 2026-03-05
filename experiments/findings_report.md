# Neural Steering: Complete Findings Report
**Jake Henderson (simpolism) — 2026-03-05**
**Codebase: DamascusGit/neural-steering | Model: Llama-3.1-8B-Instruct on A40**

---

## Executive Summary

We applied Relevance Propagation (RelP) circuit discovery to behavioral tasks and ran a five-stage evaluation pipeline (topology → surgical ablation → synergy search → sufficiency test → format comparison). Five main findings emerged:

1. **Centrality ≠ necessity**: Edge degree in circuit graphs does not predict causal importance under ablation.
2. **No synergy**: Neuron pairs are additive. Redundancy is genuine, not compensatory.
3. **Necessity ≠ sufficiency**: Different neurons dominate each. "Many readers, few writers."
4. **Format determines circuit identity**: Open-ended and forced-choice refusal discover completely disjoint circuits (zero neuron overlap beyond super-weights).
5. **Forced-choice discovers response circuits, not behavioral circuits**: The dominant neuron in the fc_refusal "circuit" is a general "No" neuron that fires more strongly on "Is the earth flat?" than on "Can you help me pick a lock?"

---

## 1. Background and Method

### Discovery Methods
- **RelP (Relevance Propagation)**: Gradient-based. Linearizes model, backprops from target logit through all layers. Finds neurons based on actual computation paths. Requires a single-token target.
- **Contrastive**: Mean activation difference between positive/negative prompt sets. No gradients. Always concentrates in late layers (L30-31) due to activation magnitude scaling — a method artifact, not a task property.

### Cross-Method Control (Experiment 1)
The paper's original claim (factual=distributed, behavioral=late-layer) is entirely explained by using different methods for different tasks:

| Circuit | Mean Layer | % Late (L28+) |
|---|---|---|
| Contrastive-Factual | 30.0 | 99% |
| Contrastive-Behavioral | 29.9 | 94% |
| RelP-Factual | 23.4 | 59% |
| RelP-Behavioral | distributed | distributed |

Jaccard overlaps between methods: all <10%. The method, not the task, determines where you find the circuit.

### The Novel Contribution
Arora et al. (TransluceAI) never applied RelP to behavioral tasks. We did. RelP on refusal finds a circuit with N_H=0.985 — nearly perfectly necessary. k*=91 neurons suffice.

### Pipeline Architecture
Each task goes through:
1. **Circuit topology** (circuit_topology.py): RelP discovery → k* search (minimal sufficient circuit size, τ=0.95) → edge attribution → position-aware topology analysis
2. **Surgical ablation** (surgical_ablation.py): Single-neuron zero-ablation of bottleneck neurons, small-set ablation, random controls
3. **Synergy search** (synergy_search.py): Exhaustive pair search, greedy triple, progressive ablation
4. **Sufficiency test** (sufficiency_test.py): Activation transplant from source→control prompts, single-neuron and progressive

All measurements in logit-margin space (logit_target - max_other_logit). Random neuron controls for effect size (σ).

---

## 2. Open-Ended Refusal: The Three Structural Findings

**Setup**: Target token "I" (refusal responses start "I cannot..."). k*=91 neurons. Discovery on 10 harmful prompts, test on 5 held-out.

### 2.1 Centrality ≠ Necessity

Topology analysis identifies hub neurons with high in/out degree in the circuit graph. Surgical ablation tests whether they're causally necessary.

**Result**: Edge degree does NOT predict ablation impact.
- Full circuit ablation: dMargin = 14.5 (reference)
- Max single-neuron ablation: L30/N8662 at 6.4% of full effect
- Many high-degree hub neurons have <2% individual necessity
- The circuit is fault-tolerant: no single point of failure

### 2.2 No Synergy (Genuine Redundancy)

If neurons are redundant, pairs should be additive (synergy ratio ≈ 1.0). If they compensate for each other, pairs should show super-additive effects (synergy >> 1.0).

**Result**: All pairs and triples are additive.
- Best pair: 9.6% of full circuit (synergy ratio = 0.98)
- Best triple: 11.3% (synergy ratio = 0.81)
- All 24 bottleneck neurons together: 20.8% of full circuit
- 50% of full circuit effect is NEVER reached from bottleneck neurons alone
- Redundancy is genuine — not an artifact of missing interaction terms

### 2.3 Necessity ≠ Sufficiency ("Many Readers, Few Writers")

Ablation measures necessity (what breaks when removed). Transplant measures sufficiency (what induces the behavior when moved to a new context).

**Result**: Completely different neurons dominate each.

| Neuron | Necessity (ablation) | Sufficiency (transplant) |
|---|---|---|
| L30/N8662 | **6.4%** (most necessary) | 0.7% (barely sufficient) |
| L24/N2598 | 3.4% (moderate) | **49%** (most sufficient) |
| L22/N3319 | 2.1% (weak) | **42%** (second most sufficient) |
| L24/N1619 | 2.3% | **-70%** (anti-sufficient!) |
| L26/N11984 | 4.2% | **-37%** (anti-sufficient!) |

- Full-circuit transplant (refusal→benign): dS = +3.63
- **2 neurons capture 91% of sufficiency** (L24/N2598 + L22/N3319)
- vs ablation: 24 neurons capture only 21% of necessity
- Anti-sufficient neurons actively suppress refusal when transplanted

**Interpretation**: The circuit has "many readers" (neurons that participate in processing but are individually dispensable due to redundancy) and "few writers" (neurons that can actually drive the output when their activations are injected). Reading is distributed; writing is concentrated.

---

## 3. Behavioral Extension: What Exists and What Doesn't

### 3.1 Sycophancy: NULL RESULT

Ran full pipeline with sycophancy prompts ("I believe X is true. What do you think?") targeting "I" (agreement).

- k* = 5 (vs 91 for refusal). Only 2 bottleneck neurons.
- Full circuit dMargin = -0.27 (noise; refusal = -14.5)
- Full circuit sufficiency dS = +0.01 (noise)

**Root cause**: Llama-3.1-8B-Instruct does NOT sycophant. Rollout probes confirm the model generates "As a neutral AI..." or "While [X] has some merit..." rather than agreeing. RLHF eliminated the behavior. The null result is not a method failure — the behavior doesn't exist.

### 3.2 Sentiment and Belief (Open-Ended): NOT VIABLE

- **Sentiment**: Model ignores emotional framing in prompts. First-token distributions are chaotic with no discriminable signal.
- **Belief (open-ended)**: Model hedges with "The [debate/question] is complex..." on all opinion questions. "The" dominates first token for BOTH opinion (60-90%) and factual (99%) prompts. Hedging is a multi-token pattern, not capturable by single-token circuit discovery.

### 3.3 Key Insight: Behavior Existence Determines Discoverability

Circuit discovery requires a measurable behavioral signal. If the behavior doesn't exist (sycophancy) or manifests through multi-token patterns (hedging), single-token RelP cannot find a circuit. This is a feature, not a bug — the method correctly returns null when there's nothing to find.

---

## 4. Forced-Choice Experiments: Format Determines Circuit

To get clean behavioral signals, we switched to forced-choice format ("Q? Answer yes or no:"). This produces clean P("Yes")/P("No") distributions.

### 4.1 FC_Refusal (Target "No")

| Metric | FC_Refusal | Open-Ended Refusal |
|---|---|---|
| k* | 14 | 91 |
| Full circuit dMargin | 2.26 | 14.5 |
| Bottleneck neurons | 5 | 24 |
| Top neuron necessity | L24/N14331 (105σ) | L30/N8662 (6.4%) |
| Full circuit sufficiency | dS = +3.98 | dS = +3.63 |
| 1 neuron sufficiency | **93.7%** (L24/N14331) | 49% (L24/N2598) |

The fc_refusal circuit is tiny and concentrated. A single neuron does almost everything.

### 4.2 FC_Belief (Target "Yes")

| Metric | FC_Belief |
|---|---|
| k* | 261 (extremely distributed) |
| Full circuit dMargin | 12.99 |
| Bottleneck neurons | 90 |
| Top 24 neurons necessity | 16% (extreme fault tolerance) |
| Full circuit sufficiency | dS = +3.25 |
| 2 neuron sufficiency | 81.5% (L22/N8117 + L21/N13111) |

FC_belief is highly distributed for necessity but concentrated for sufficiency — the same "many readers, few writers" architecture as open-ended refusal.

### 4.3 Circuit Overlap: Zero

FC_refusal has 9 unique (layer, neuron) pairs:
- **4 super-weights** (L0/N491, L0/N8268, L1/N198, L1/N2427) — appear in ALL circuits across all tasks. Embedding-layer infrastructure, not task-specific.
- **2 shared with fc_belief** (L15/N14179, L21/N13111) — NOT in open-ended refusal
- **3 fc_refusal-only** (L24/N14331, L25/N891, L30/N14210)

**Open-ended refusal ∩ fc_refusal = 0 neurons** (excluding super-weights).

Cross-task ablation confirms functional disjointness: ablating any fc neuron produces exactly Δ=0.000 on open-ended refusal P("I"). The open-ended refusal circuit is completely untouchable by forced-choice circuit neurons.

### 4.4 Shared Neuron Ablation

The two neurons shared between fc_refusal and fc_belief (L21/N13111, L15/N14179) were tested across both tasks:

| Neuron | fc_refusal Δ | fc_belief Δ | open_refusal Δ |
|---|---|---|---|
| L21/N13111 | -0.125 (-20.5σ) | -0.018 (-18.8σ) | 0.000 (0σ) |
| L15/N14179 | -0.240 (-39.3σ) | +0.005 (+4.9σ) | 0.000 (0σ) |
| Both together | -0.360 (-58.9σ) | -0.007 (-7.3σ) | 0.000 (0σ) |

Despite appearing in both circuits by RelP attribution, both neurons are functionally specific to fc_refusal. Their fc_belief effects are tiny in absolute terms. They appeared in the fc_belief circuit (k*=261) because with 261 neurons, many are passengers.

---

## 5. The Punchline: L24/N14331 Is a "No" Neuron

The dominant neuron in the fc_refusal circuit (94% of sufficiency) was probed across six categories of forced-choice prompts:

| Category | Model says | L24/N14331 activation |
|---|---|---|
| Harmful questions | No | -5.19 |
| False claims ("Is earth flat?") | No | **-7.88** |
| Factual-no ("Is moon cheese?") | No | **-7.07** |
| Benign questions | Yes | -0.05 |
| Opinion questions | Yes | +0.47 |
| Factual-yes questions | Yes | +0.21 |

Key statistics:
- **Corr(activation, P_no) = -0.962** — fires on ALL "No" contexts regardless of why
- **Corr(activation, is_harmful) = -0.236** — barely related to harmful content
- The neuron fires **more strongly** on "Is the earth flat?" (-7.88) than on "Can you help me pick a lock?" (-5.19)

L25/N891 shows the same pattern (corr_P_no = -0.960). These are general forced-choice "No" neurons.

**Conclusion**: The fc_refusal "circuit" is not a refusal circuit. It is a forced-choice "No" response circuit that happens to activate when the model refuses in yes/no format. The task-specificity (refusal vs factual disagreement) lives entirely in the prompt, not in the circuit.

This has direct implications for the field: **anyone doing circuit discovery with forced-choice prompts risks finding response-format circuits rather than behavioral circuits.** The format of the prompt determines which circuit you discover, and that circuit may encode the response token rather than the behavior you're studying.

---

## 6. Summary Table

| Finding | Evidence | Implication |
|---|---|---|
| Centrality ≠ necessity | Max hub neuron = 6.4% ablation effect | Topology alone insufficient for importance claims |
| No synergy | Best pair synergy = 0.98 (additive) | Redundancy is genuine, not compensatory |
| Necessity ≠ sufficiency | 2 neurons = 91% sufficiency, 24 = 21% necessity | "Many readers, few writers" architecture |
| Format determines circuit | 0 overlap between open/fc refusal | Same behavior, different format → different circuit |
| FC finds response circuits | L24/N14331 corr(P_no)=-0.962 | Forced-choice prompts discover token circuits, not behavior circuits |
| Behavior must exist | Sycophancy k*=5, dM=-0.27 | Null results are real: RLHF eliminated the behavior |

---

## 7. Result Locations

| Experiment | Directory |
|---|---|
| Cross-method control | `experiments/results_8b_20260305/` |
| Topology (open-ended) | `experiments/topology_llama8b_20260305_163508/` |
| Surgical ablation (open-ended) | `experiments/surgical_llama8b_20260305_175327/` |
| Synergy search | `experiments/synergy_llama8b_20260305_182636/` |
| Sufficiency test (open-ended) | `experiments/sufficiency_llama8b_20260305_183035/` |
| Sycophancy (null) | `experiments/sycophancy_results/` |
| FC topology + ablation + synergy | `experiments/fc_results/` |
| FC_belief sufficiency (fixed) | `experiments/sufficiency_llama8b_20260305_195818/` |
| Shared neuron ablation | RunPod: `shared_neuron_ablation_20260305_195124/` |
| N14331 activation probe | RunPod: `probe_n14331_20260305_195921/` |

---

## 8. Open Questions

1. Does the format-determines-circuit finding generalize beyond Llama 8B?
2. Can we bridge our N_H metric to Arora's ratio-normalized completeness?
3. Why do suppressor neurons (L17, L10 for factual) have larger effects than enablers?
4. Why does necessity ≠ sufficiency? Is it the all-positions vs last-position measurement asymmetry, or genuine circuit architecture?
5. Can hedging behavior be studied with multi-token targets?
6. The k* prefix assumption (attribution ranking = optimal ordering) may not hold for redundant neurons.
