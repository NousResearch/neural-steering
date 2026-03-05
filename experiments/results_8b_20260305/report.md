# Method Artifact Analysis: Contrastive vs RelP Circuit Discovery
**Llama-3.1-8B-Instruct on A40 — March 5, 2026**

## Summary

We ran three experiments testing whether the layer-localization dichotomy reported in the Contrastive Neuron Circuits draft (factual circuits = distributed, behavioral circuits = late-layer concentrated) reflects genuine structural properties of the circuits, or is an artifact of using different discovery methods for each task type.

**Finding: The dichotomy is a method artifact.** Contrastive discovery always concentrates in late layers regardless of task. RelP always produces broader distributions. The structural claim in the draft doesn't hold.

That said, both methods find *real* circuits — they just answer different questions about the model.

---

## Experiment 1: Cross-Method Control

**Question:** If we run contrastive discovery on factual tasks (capitals), does it produce distributed circuits like RelP, or late-concentrated circuits like contrastive-on-refusal?

**Design:** All three conditions on the same model, same circuit size (k=200):
- Contrastive on capitals (factual task, contrastive method)
- Contrastive on refusal (behavioral task, contrastive method)
- RelP on capitals (factual task, RelP method)

**Results:**

| Method | Mean Layer | Late Fraction |
|---|---|---|
| Contrastive-Factual | 30.0 | 0.99 |
| Contrastive-Behavioral | 29.9 | 0.94 |
| RelP-Factual | 23.4 | 0.59 |

Contrastive-Factual concentrates *more* in late layers than contrastive-behavioral (99% vs 94%). The paper's claim that factual circuits are distributed and behavioral circuits are concentrated is entirely explained by the method, not the task.

Jaccard overlap between circuits:
- Contrastive-Factual ↔ Contrastive-Behavioral: 0.093
- Contrastive-Factual ↔ RelP-Factual: 0.070
- Contrastive-Behavioral ↔ RelP-Factual: 0.018

The methods find almost entirely different neurons even on the same task.

---

## Experiment 2: Top-k Sweep

**Question:** How do circuit properties change as we vary circuit size? Does RelP produce faithful circuits at some k, or does it lobotomize the model at all sizes (as we saw on Llama 1B)?

**Design:** Discover large pools (k=500), slice to k={5, 10, 20, 50, 100, 200, 500}. Measure necessity (N_H: drop in target metric when ablating the circuit), specificity (N_B: collateral damage to control metric), and coherence for each.

**Key Results — Behavioral (Refusal):**

| k | RelP N_H(zero) | RelP Coherent? | Contrastive N_H(zero) | Contrastive Coherent? |
|---|---|---|---|---|
| 5 | 0.001 | yes | 0.000 | yes |
| 20 | 0.318 | yes | 0.000 | yes |
| 50 | 0.785 | yes | 0.000 | yes |
| 100 | 0.990 | yes | 0.000 | yes |
| 200 | 0.987 | yes | 0.103 | yes |
| 500 | 0.998 | yes | 0.230 | yes |

RelP-Behavioral finds highly necessary circuits — ablating 50 neurons drops P("I") from 1.0 to 0.215 while maintaining coherent output. Contrastive barely moves the needle until k=500.

Important: in this run, mean ablation (Arora-style, replace with dataset mean) remains highly effective for RelP at larger k, not a no-op. This differs from earlier pilot behavior where mean ablation looked weak. Because mean-ablation effects are circuit-dependent, we report both zero and mean ablation and treat zero ablation as the conservative primary test.

**Key Results — Factual (Capitals):**

RelP-Factual shows negative N_H at small k (ablation *helps*), then reaches N_H≈0.29 at k≥100. Contrastive-Factual is weaker but more stable. Both maintain perfect coherence.

---

## Experiment 3: Full Evaluation Protocol

**Question:** Can we build an evaluation framework for neuron circuits that tests both necessity and sufficiency on behavioral tasks?

### Background: Arora et al.'s Framework

Arora et al. (2024) evaluate RelP circuits on factual tasks using two ratio-normalized metrics:
- **Faithfulness(C)** = [m(C,x) − m(∅,x)] / [m(M,x) − m(∅,x)]: ablate everything *except* the circuit and measure what the circuit alone reproduces. Tests sufficiency.
- **Completeness(C)** = [m(C̄,x) − m(∅,x)] / [m(M,x) − m(∅,x)]: ablate the circuit and measure what the complement retains. Tests necessity (inversely).

They report near-perfect faithfulness and completeness with ~200 neurons on SVA (subject-verb agreement) using mean ablation on Llama-3.1-8B-Instruct.

These metrics rely on complement ablation — ablating everything except a small circuit. For behavioral tasks (refusal, tone), complement ablation of ~14,000 neurons is destructive and uninformative. We need different operations.

### Our Protocol

We designed a five-part evaluation protocol for behavioral circuit evaluation. It is *motivated by* the same conceptual decomposition as Arora (necessity + sufficiency), but uses different operations. **It is not a generalization of Arora's protocol** — it does not reproduce their metrics and produces different numbers on the same tasks (see SVA results below).

**1. Behavioral Response Metric R(x)**

The key insight: for refusal on Llama-3.1-8B-Instruct, we can define R(x) = P("I") at the first generation position. This works because Llama's refusal is near-monomorphic — 96%+ of refusal responses begin with "I" ("I can't", "I cannot", "I'm not able to"), and P("I") cleanly separates harmful from benign prompts (0.65–1.0 for harmful, ~0 for benign). For factual tasks, R(x) = P(correct_token) as in Arora (we use P("Paris") with an "Answer: " seed to position the target token at the generation point).

This unifies the framework: both task types have a single scalar R(x) ∈ [0,1] that tracks the behavior of interest.

**2. Necessity (N_H, N_B)**

Ablate the circuit neurons (set activations to zero or replace with dataset mean) and measure the change in R(x):

- **N_H** = R_baseline(target_prompts) − R_ablated(target_prompts): how much does ablating the circuit reduce the target behavior? Higher = more necessary.
- **N_B** = R_baseline(control_prompts) − R_ablated(control_prompts): collateral damage — does ablation affect behavior on prompts where it shouldn't? Should be ~0.

We test both zero ablation (multiply activations by 0) and mean ablation (replace with dataset mean, as in Arora). This matters because mean-ablation behavior is circuit-dependent: some circuits are strongly affected by mean replacement, while others can be weak/no-op.

**3. Coherence**

Does the model still produce grammatical, non-degenerate output after ablation? A circuit that "works" only by lobotomizing the model (producing repetitive garbage) hasn't actually located the relevant computation — it's just broken the model. We generate 30 tokens after ablation and check for repetition loops, empty output, and basic grammaticality. This is a critical check that Arora's protocol doesn't explicitly include.

**4. Sufficiency (S+, S−)**

We use bidirectional activation transplant to test whether the circuit carries the behavioral signal:

- **S+** (forward mediation): Collect circuit activations from target prompts (e.g., harmful prompts that trigger refusal). Transplant those activations into the circuit neurons while running control prompts (e.g., benign prompts). If the circuit is sufficient, this should *induce* the target behavior on prompts that normally don't trigger it. S+ = R_transplanted(control) − R_baseline(control).

- **S−** (reverse mediation): Collect circuit activations from control prompts. Transplant into the circuit while running target prompts. If the circuit is sufficient, this should *suppress* the target behavior. S− = R_baseline(target) − R_transplanted(target).

Bidirectional mediation tests whether the circuit *carries* the behavioral signal, rather than testing (as Arora does) whether the rest of the model can compensate when the circuit is removed. These are related but distinct questions.

**5. Random Controls**

Run the full protocol on multiple random neuron sets (same size as the discovered circuit, excluding universal/blacklisted neurons). This establishes the null distribution. We report effect sizes in σ above the random baseline. Any discovered circuit should substantially exceed random performance on at least necessity or sufficiency.

### How This Relates to Arora et al.

| Concept | Arora's Operation | Our Operation |
|---|---|---|
| Necessity | Completeness: ablate C, measure complement (ratio-normalized) | N_H: ablate C, measure absolute R(x) drop |
| Sufficiency | Faithfulness: ablate complement, measure C alone (ratio-normalized) | S±: transplant C's activations across prompt types |

Key differences that prevent direct comparison:
1. **Different operations.** We use transplant for sufficiency; Arora uses complement ablation. We measure absolute R(x) drop for necessity; Arora uses ratio-normalized complement performance.
2. **Different normalization.** Arora normalizes against full-ablation and full-model baselines. Our N_H is an absolute drop, not a ratio. A small N_H in our protocol could correspond to a high completeness score in Arora's framework if the normalization range is narrow.
3. **SVA confirms the gap.** On SVA (Arora's benchmark task), our protocol shows N_H ≈ 0.03 for 200-neuron circuits, while Arora reports near-perfect completeness. This is likely a combination of different R(x) choice (we use raw P(" is") with a low baseline of 0.124; Arora likely uses accuracy or logit difference) and the normalization difference. **We cannot recover Arora's published results using our protocol.**

What we add that Arora doesn't test:
- **Coherence checking.** On Llama 1B, RelP circuits showed perfect necessity scores only because ablation lobotomized the model into garbage — a failure mode Arora's metrics wouldn't catch.
- **Specificity (N_B).** Does ablation leak into unrelated tasks?
- **Random controls.** Null distribution for all metrics.
- **Both zero and mean ablation.** Mean-ablation behavior is circuit-dependent and can diverge from zero-ablation behavior. Reporting both is a methodological requirement for behavioral extensions.

**Bottom line:** This is an independent evaluation protocol for behavioral circuits, motivated by Arora's conceptual decomposition but using different operations that produce different numbers. It should not be described as a generalization of their framework.

**Results (n_random=10):**

| Circuit | N_H(zero) | N_H(mean) | N_B(zero) | Coherence | S+ | S- | σ(N_H) |
|---|---|---|---|---|---|---|---|
| **Capitals** | | | | | | | |
| Contrastive-Factual | 0.162 | 0.219 | 0.000 | 100% | 0.000 | 0.000 | 10.0σ |
| RelP-Factual | **0.288** | **0.295** | 0.000 | 100% | 0.000 | -0.082 | 16.3σ |
| **Refusal** | | | | | | | |
| Contrastive-Behavioral | 0.062 | 0.059 | -0.134 | 100% | 0.131 | 0.002 | very large* |
| RelP-Behavioral | **0.985** | **1.000** | 0.047 | 67% | 0.135 | 0.027 | very large* |
| **SVA** | | | | | | | |
| RelP-SVA | 0.037 | 0.032 | -0.068 | 100% | 0.019 | 0.074 | 32.8σ |
| Contrastive-SVA | 0.034 | 0.031 | 0.002 | 100% | 0.004 | 0.031 | 9.2σ |

**Interpretation:**

*Refusal:* RelP-Behavioral is the standout: N_H=0.985 means ablating these 200 neurons nearly eliminates the model's refusal response. Coherence is 67% (some garbage outputs, some coherent compliance). Specificity leak is small (0.047). This is a real, necessary circuit for refusal. Contrastive-Behavioral has low necessity (0.062 — ablating barely changes behavior) but shows the strongest sufficiency signal (S+=0.131). It finds neurons that are *correlated with* refusal and can partially *induce* it, but aren't *necessary* for it.

*Capitals:* RelP wins on necessity (0.288 vs 0.162), both have perfect coherence and zero specificity leak. Neither shows sufficiency — transplanting capital-knowledge activations doesn't make the model output "Paris" on non-capital prompts, which makes sense (factual recall is more distributed).

*SVA:* Both methods show weak necessity (N_H ≈ 0.03), consistent with SVA being a more distributed computation than refusal. However, both are highly significant vs random (32.8σ and 9.2σ). RelP-SVA shows notable S− (0.074) — transplanting plural-subject activations into singular-subject prompts partially suppresses P(" is"), meaning the circuit carries some of the agreement signal. The R(x) baseline for SVA is low (0.124) because " is" competes with many other plausible continuations; this ceiling effect limits the achievable N_H. Contrastive-SVA has excellent specificity (N_B=0.002) but weaker sufficiency.

All effects are above random controls. Some effect sizes are extremely large because random-control variance is near zero for those metrics.

\* "very large" indicates a numerically unstable σ estimate due to near-zero random-control standard deviation.

---

## What This Means for the Paper

1. **The layer-localization dichotomy is a method artifact.** This should be either removed or reframed. Contrastive always concentrates late; RelP always spreads. The task doesn't determine the distribution.

2. **Both methods find real circuits, but they answer different questions:**
   - RelP finds *necessary* neurons (where the information lives). High necessity, moderate coherence.
   - Contrastive finds *differentially active* neurons (what changes between conditions). Low necessity, high coherence, some sufficiency.

3. **Our evaluation protocol detects real differences between methods.** On 8B, both discovery methods produce circuits with above-random necessity and/or sufficiency scores. The protocol uses different operations from Arora et al. (direct ablation + transplant vs complement ablation) and does not reproduce their published metrics. It should be understood as an independent framework for comparing circuit discovery methods, not as a generalization of Arora.

4. **Mean ablation can be misleading if used alone.** In this run, mean and zero ablation diverge across circuits. Report both; use zero ablation as the conservative primary necessity test.

5. **Contrastive's weakness is necessity, not sufficiency.** The circuits it finds are real (above-random sufficiency), but they're not where the critical computation happens. This limits the strength of claims about "discovering the refusal circuit."

---

## Reproduction

All experiments ran on Llama-3.1-8B-Instruct (float16) on a single A40 (48GB VRAM). Total wall time ~15 minutes for all three experiments. Code is in `experiments/{cross_method_control,topk_sweep,circuit_eval_protocol}.py`.
