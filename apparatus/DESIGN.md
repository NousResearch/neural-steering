# Apparatus Design

This document is the spec. The code in this directory implements it; if the code and
the spec diverge, the spec is the source of truth for *intent*, and the code is the
source of truth for *what we actually ran*.

## Frame

We are not trying to "find refusal circuits." We are constructing measurement
apparatus, and the role decomposition (reader / writer / suppressor / etc.) is the
smallest non-trivial energetic object the apparatus can produce.

A "circuit" is not a structure inside the model. It is a jointly-constructed object
produced by `model + method + format + intervention + metric`. Vary any axis and you
get a different cross-section. Multiple cross-sections compared is what gives us
purchase on what's model-property vs apparatus-artifact.

## H1: signed local flow is real

The load-bearing energetic claim. Given a scalar readout R, signed local flow
(grad × activation, or transplant effect, or ablation effect) is a real object — not
a metaphor. It supports decomposition by sign and by role.

H2 (rollout attractor geometry) and H3 (discharge dynamics) are demoted to empirical
hypotheses the apparatus should be able to detect, not presupposed claims.

## Apparatuses

### Apparatus 1: Role Table

**Input:** RelP topology (circuit + edges) + surgical ablation + sufficiency transplant
on a single behavior, single readout, single model.

**Output:** one row per `(layer, position, neuron)` with axes
`(attribution, edge_in, edge_out, necessity_sigma, sufficiency_dS)`.

**Falsifier (composite):**
- Pairwise rank correlation across the four primary axes (attribution, edges,
  necessity, sufficiency) exceeds 0.85 → role decomposition is degenerate to a single
  ranking. **Status (8B refusal, March-5 data, 47 neurons with intervention):**
  max non-trivial cross-axis ρ = 0.80. Falsifier did not fire.

**Role definitions (apparatus-symmetric):**
- **infrastructure**: super-weights (universal across tasks)
- **reader**: high necessity_sigma (>+5), near-zero sufficiency (|dS|<1.0)
- **writer**: high sufficiency (dS>+1.0)
- **suppressor (consistent)**: negative necessity_sigma (<-5) AND negative sufficiency (dS<-0.5)

**Apparatus-asymmetric roles (added after visualization revealed the structure):**
- **suppressor (transplant-only)**: dS < -0.5 but necessity_sigma in [-5, +5]. Carries
  suppression in its activation pattern but ablating it doesn't release refusal —
  the suppression is signal-bound, not flow-bound.
- **suppressor (ablation-only)**: necessity_sigma < -5 but |dS| < 0.5. Pushes against
  refusal contextually but transplanting it to benign prompts has no effect — the
  suppression is context-dependent, requiring an active refusal computation nearby.
- **writer-only** (already implicit): high sufficiency, low/negative necessity. Pure
  discharge with no detection role.

### Apparatus 2: Multi-readout consistency

**Input:** Apparatus 1 outputs computed with at least 3 different readouts R:
1. Token-margin readout (logit P("I") - max_other on refusal prompts)
2. BehaviorProbe scalar (LDA direction on hidden states, multi-token-capable)
3. (Later) reward-model or other differentiable readout

**Falsifier:** if no two readouts agree on which neurons are writers (or readers,
or suppressors), the role decomposition is purely apparatus-bound. Any subsequent
claim must be readout-relativized.

Codex's correction: StrongREJECT/judge scores are not attribution targets (not
differentiable through model activations). They are downstream evaluation of whether
interventions actually changed behavior.

### Apparatus 3: Displacement test

For each candidate writer from Apparatus 1: probe activation across
`(behavior present/absent) × (output token shared/different)`.

If activation tracks **behavior**: real writer.
If activation tracks **token**: displacement (cf. fc_refusal L24/N14331).

**Falsifier:** if all candidate writers turn out to be token-tracking, the entire
tokenwise-readout program is at risk of studying displaced signifiers.

### Apparatus 4: Suppressor characterization

Probe-style analysis on identified suppressors, asking inverse question: across what
categories of prompt does this neuron's activation cancel refusal?

The visualization showed at least two suppressor subtypes:
- **context-free**: signal-bound (L24/N1619, L26/N11984) — transplant carries the
  suppression
- **context-dependent**: flow-bound (L29/N4918, L21/N3057, L30/N864) — only modulate
  when refusal computation is active

These look like *different kinds of counterforce* — fixed inhibitory binding vs
dynamic regulation.

### Apparatus 5: Cross-behavior junction

Run Apparatus 1 on a second behavior. Compute role-by-role overlap:
- reader overlap (shared registration substrate?)
- writer overlap (shared discharge population, or behavior-specific?)
- suppressor overlap (shared counterforces?)

**Falsifier:** all three overlaps at chance → behaviors entirely siloed.

Hold off on AI-deflection until Apparatus 1-4 are stable on refusal at 8B.

## Pre-run checklist (Codex)

Before scaling to cluster / model matrix:

1. Validate behavior exists in rollouts (no point steering an absent behavior — cf.
   sycophancy null result on Llama-8B)
2. Validate the scalar readout R isn't a response-format proxy (cf. fc_refusal
   L24/N14331 — the "No neuron" that wasn't a refusal neuron)
3. Run CNA and RelP on the same prompts/model, compare necessity / sufficiency /
   signed edge flow
4. Only then scale to model matrix

## Scaling order

1. Fill coverage gap on Llama-3.1-8B refusal: intervention on all 91 circuit
   neurons, not just 24 bottleneck candidates (cluster, ~1 hour)
2. Apparatus 2 on same 91 neurons (cluster, ~few hours)
3. Apparatus 3 on writer subset (cluster, ~hour)
4. Apparatus 5 on second behavior (cluster, ~hours)
5. Cluster scale-out across models (Llama 1B/3B/8B/70B, Qwen 1.5B/3B/7B/72B)

Each step has its own falsifier; if one falsifier fires hard, we stop and rethink
before the next.
