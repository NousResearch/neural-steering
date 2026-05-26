# Apparatus Design

This document is the spec. The code in this directory implements it; if the code and
the spec diverge, the spec is the source of truth for *intent*, and the code is the
source of truth for *what we actually ran*. Results-to-date are tracked in §Results
at the bottom; the apparatus list above it is the design.

## Frame

We are not trying to "find refusal circuits." We are constructing measurement
apparatus, and the role decomposition (reader / writer / suppressor / gate
neuron / etc.) is the smallest non-trivial energetic object the apparatus can
produce.

A "circuit" is not a structure inside the model. It is a jointly-constructed
object produced by `model + method + format + intervention + metric + probe
layer`. Vary any axis and you get a different cross-section. Multiple
cross-sections compared is what gives us purchase on what's model-property vs
apparatus-artifact.

The interpretive grammar we are reaching for is energetic-dynamical rather than
propositional/feature-inventory. SAE-style decomposition asks "what features
are present?". We want to ask "what is moving, against what, at what stage of
the layer stack?" — and the apparatus we build should produce objects that
make this question legible without flattening it back to a feature list.

## H1: signed local flow is real

The load-bearing energetic claim. Given a scalar readout R, signed local flow
(grad × activation, ablation effect, transplant effect, edge weight) is a real
object — not a metaphor. It supports decomposition by sign and by role.

H2 (rollout attractor geometry) and H3 (discharge dynamics) are demoted to
empirical hypotheses the apparatus should be able to detect, not presupposed
claims.

H4 (new, from late-probe Apparatus 2a results): **role identity evolves across
layers as a *developmental trajectory*, not as a fixed assignment**. The same
neuron can be substrate-positive at its source layer and gate-suppressive at
the readout, with downstream layers doing the *transformation* between the
two. Substrate-vs-gate is a layer-band distinction, not a neuron-population
distinction.

## Apparatuses

### Apparatus 1: Role Table

**Input:** RelP topology (circuit + edges) + surgical ablation + sufficiency
transplant on a single behavior, single readout, single model.

**Output:** one row per `(layer, position, neuron)` with axes
`(attribution, edge_in, edge_out, necessity_sigma, sufficiency_dS)`.

**Falsifier (composite):** pairwise rank correlation across the four primary
axes exceeds 0.85 → role decomposition is degenerate to a single ranking.

**Role definitions (apparatus-symmetric):**
- **infrastructure**: super-weights (universal across tasks)
- **reader**: high necessity_sigma (>+5), near-zero sufficiency (|dS|<1.0)
- **writer**: high sufficiency (dS>+1.0)
- **suppressor (consistent)**: negative necessity_sigma (<-5) AND negative
  sufficiency (dS<-0.5)

**Apparatus-asymmetric roles (added after visualization revealed the structure):**
- **suppressor (transplant-only)**: dS < -0.5 but necessity_sigma in [-5, +5].
  Carries suppression in its activation pattern but ablating it doesn't
  release refusal — the suppression is signal-bound, not flow-bound.
- **suppressor (ablation-only)**: necessity_sigma < -5 but |dS| < 0.5. Pushes
  against refusal contextually but transplanting it to benign prompts has no
  effect — the suppression is context-dependent, requiring an active refusal
  computation nearby.
- **writer-only** (already implicit): high sufficiency, low/negative
  necessity. Pure discharge with no detection role.

### Apparatus 2: Multi-readout consistency

**Input:** Apparatus 1 outputs computed with at least 3 different readouts R:
1. Token-margin readout (logit P("I") − max_other on refusal prompts).
2. Hidden-state probe scalar at multiple layers, including the late-layer
   gate band (mean-difference probes at L18, L24, L28, L30, L32 on
   Llama-3.1-8B; L32 is post-final-norm, immediately pre-LM-head).
3. (Later) reward-model or other differentiable readout.

**Falsifier:** if no two readouts agree on which neurons are writers (or
readers, or suppressors), the role decomposition is purely apparatus-bound.
Any subsequent claim must be readout-relativized.

Codex's correction: StrongREJECT/judge scores are not attribution targets
(not differentiable through model activations). They are downstream
evaluation of whether interventions actually changed behavior.

**Late-layer probes are required, not optional.** Stopping probes at L28 in
the first run gave a misleading picture — the published-CNA discrimination
band sits at L29-L31 plus the post-final-norm readout stream. The
substrate→gate transformation we care about happens *in* that band, so the
probe layer set must cover it.

### Apparatus 3: Displacement test

For each candidate writer from Apparatus 1: probe activation across
`(behavior present/absent) × (output token shared/different)`.

If activation tracks **behavior**: real writer.
If activation tracks **token**: displacement (cf. fc_refusal L24/N14331).

**Falsifier:** if all candidate writers turn out to be token-tracking, the
entire tokenwise-readout program is at risk of studying displaced signifiers.

### Apparatus 4: Suppressor characterization

Probe-style analysis on identified suppressors, asking inverse question:
across what categories of prompt does this neuron's activation cancel refusal?

Three suppressor subtypes have emerged empirically:
- **context-free / signal-bound** (L24/N1619, L26/N11984): transplant carries
  the suppression.
- **context-dependent / flow-bound** (L29/N4918, L21/N3057, L30/N864): only
  modulate when refusal computation is active.
- **gate-only late-layer counterforces**: visible only at L30/L32 probes
  (cf. L24/N1619 final probe_suff = −0.74 at L32).

These look like *different kinds of counterforce* — fixed inhibitory binding,
dynamic regulation, and late-layer routing competitor.

### Apparatus 5: Cross-behavior junction

Run Apparatus 1 on a second behavior. Compute role-by-role overlap:
- reader overlap (shared registration substrate?)
- writer overlap (shared discharge population, or behavior-specific?)
- suppressor overlap (shared counterforces?)

**Falsifier:** all three overlaps at chance → behaviors entirely siloed.

Hold off on a second behavior until Apparatus 1-4 are stable on refusal at 8B
and the flow-field apparatus (below) has produced its first picture.

### Apparatus 6: Flow field (planned, not yet built)

The role table and the per-neuron scatters are still categorical reductions.
The actual energetic object — signed local flow under a scalar readout — is
already in the data (edges.json with 2704 directed signed weights) but has
never been *displayed* as a flow field.

**Phase A (static flow field):**
- Nodes positioned by (layer, position).
- Color/shape by role assignment from Apparatus 1.
- Edges drawn with thickness ∝ |weight| and color by sign.
- The picture shows where the energy flows from where to where, with role
  coloring giving "reader-to-writer", "writer-to-gate-to-readout" structure
  visible at a glance.

**Phase B (intervention-perturbed flow field):**
- Re-run RelP edge attribution under intervention (ablation of canonical
  writers, transplant of L24/N1619, etc.).
- Compare to baseline flow field: where does the energy redistribute?
- This is the first apparatus that asks "how does a perturbation propagate"
  rather than "what neurons matter."

**Phase C (rollout-as-trajectory, speculative):**
- Track the residual-stream trajectory across generation steps, not just one
  forward pass.
- Identify attractor regions empirically.
- Measure how interventions deform attractor geometry.
- This is where the apparatus stops being post-hoc circuit description and
  starts being dynamical-systems characterization of an autoregressive
  transformer. H2 and H3 become measurable here, not before.

## Pre-run checklist (Codex)

Before scaling to cluster / model matrix:

1. Validate behavior exists in rollouts (no point steering an absent behavior
   — cf. sycophancy null result on Llama-8B).
2. Validate the scalar readout R isn't a response-format proxy (cf.
   fc_refusal L24/N14331 — the "No neuron" that wasn't a refusal neuron).
3. Cover the late-layer band when measuring substrate→gate dynamics.
4. Run CNA and RelP on the same prompts/model, compare necessity /
   sufficiency / signed edge flow.
5. Only then scale to model matrix.

## Scaling order (revised after late-probe finding)

1. ✅ Fill coverage gap on Llama-3.1-8B refusal: intervention on all 91
   circuit neurons.
2. ✅ Apparatus 2 with mid-layer probes (L18, L24, L28).
3. ✅ Apparatus 2 late-layer probes (L30, L32) — required addition.
4. ✅ L24/N1619 dossier + transplant rollout (Apparatus 4 first pass).
5. **Next: Apparatus 6 Phase A (static flow field).** Builds first
   energetic display from existing edge data.
6. Apparatus 6 Phase B (intervention-perturbed flow field).
7. Probe orthogonalization against I-onset direction (resolves the
   substrate-vs-routing confound at probe@L24).
8. Apparatus 5 on a second behavior (AI-deflection candidate).
9. Cluster scale-out across models (Llama 1B/3B/8B/70B, Qwen
   1.5B/3B/7B/72B).
10. Apparatus 6 Phase C (rollout trajectory dynamics).

Each step has its own falsifier; if one fires hard, stop and rethink.

---

# Results

## Apparatus 1: Role Table on Llama-3.1-8B refusal

**Datasets:**
- March 5 partial (24 bottleneck neurons, 91 position rows). `apparatus/output/role_table_refusal_8b_20260305.{jsonl,csv}`.
- Full circuit (51 unique neurons, 91 position rows). `apparatus/output/role_table_refusal_8b_fullcircuit_20260526.{jsonl,csv}`.

**Falsifier outcome:** did not fire. Collapsed-view max non-trivial
cross-axis ρ = 0.523 (`attribution ~ necessity_sigma`). Necessity ~
sufficiency = +0.390 — meaningful but not degenerate.

**Role counts (collapsed, full-circuit):**
- infrastructure: 4 (universal super-weights L0/L1)
- reader-only: 12 (dominant population)
- reader-writer: 1
- writer-only: 4
- suppressor-transplant: 4 (signal-bound, including L24/N1619 and L30/N2653)
- suppressor-ablation: 8 (context-dependent, anti-necessary under ablation
  but probe-neutral under transplant)
- mixed: 18 (low-signal "passenger" population)

**The "anti-writer" quadrant (+necessity, −sufficiency) is empty of
substantial points** even at full coverage. There are no neurons that are
necessary for refusal *and* suppress refusal when transplanted.

**"Many readers, few writers":** confirmed at full coverage. 12 readers
versus 4 writer-only neurons. Single-neuron writer transplants do not
induce refusal-shaped output on benign prompts; refusal requires
multi-neuron contribution.

## Apparatus 2a: Multi-layer probe consistency

Mean-difference hidden-state probes fit on REFUSAL_DISCOVERY_POSITIVE vs
REFUSAL_DISCOVERY_NEGATIVE at last token position. Each probe held out cleanly:
**all five layers AUC = 1.0** on REFUSAL_TEST vs BENIGN_PROMPTS.

Probe heldout margins (larger → probe direction is more aligned with the
output readout's direction):

| Probe layer | Heldout margin | Heldout AUC |
|---|---|---|
| L18 | +7.18 | 1.0 |
| L24 | +12.80 | 1.0 |
| L28 | +18.10 | 1.0 |
| L30 | +22.43 | 1.0 |
| L32 (post-final-norm) | **+91.27** | 1.0 |

The growth of the margin from L18 → L32 is itself a finding: the harmful-vs-
benign direction *sharpens dramatically* in the last three layers. By L32
the probe is essentially measuring the readout's commitment to "I cannot..."
output token routing.

**Tokenwise role table ↔ probe role table comparison (upstream-only):**

| Probe layer | ρ(tok_nec, probe_nec) | ρ(tok_suff, probe_suff) | n |
|---|---|---|---|
| L18 | +0.731 | +0.526 | 13 |
| L24 | +0.608 | +0.255 | 23 |
| L28 | +0.656 | +0.506 | 32 |
| L30 | +0.095 | **+0.590** | 43 |
| L32 | +0.321 | **+0.686** | 51 |

**Two patterns:**
1. *Sufficiency alignment with tokenwise grows toward the readout.* By L32,
   ρ = +0.686 — the highest cross-readout alignment in the entire study.
2. *Necessity alignment doesn't follow the same pattern.* L30's tok_nec ~
   probe_nec ρ collapses to +0.095, then recovers to +0.321 at L32. The
   *necessity* (reader) population is partially substrate-flavored — its
   role identity transfers cleanly between mid-substrate probes (L18-L28)
   but is partly broken at the L30 transition. Worth understanding later.

**Cross-probe-layer consistency:**

| Pair | ρ(necessity) | ρ(sufficiency) | n |
|---|---|---|---|
| L18 vs L24 | +0.863 | +0.863 | 13 |
| L18 vs L28 | +0.824 | +0.610 | 13 |
| L24 vs L28 | +0.819 | +0.830 | 23 |
| L30 vs L32 | +0.603 | +0.882 | 43 |

Substrate-band (L18-L28) probes agree among themselves at ρ ≥ 0.82.
Gate-band (L30-L32) probes agree at ρ = 0.88 on sufficiency but only
ρ = 0.60 on necessity. The substrate and gate bands are *internally
consistent* but disagree about *role identity*, especially for individual
neurons that change role across the band transition.

## L24/N1619: the layered finding

The single most diagnostic neuron in the corpus. Per-layer probe sufficiency:

| Layer | probe_suff(L24/N1619) | Interpretation |
|---|---|---|
| L18 | 0.000 | (probe upstream of source; structural) |
| L24 | **+0.111** | substrate-positive (writes harm-content at source) |
| L28 | +0.046 | decaying |
| L30 | **−0.058** | sign flip — downstream layers have inverted it |
| L32 | **−0.740** | strong gate-level suppressor |

Tokenwise readout: token_suff = **−2.59** (most anti-sufficient neuron in
circuit, suppresses P("I") strongly under transplant).

Activation profile (per-prompt last-token activation):
- REFUSAL_DISCOVERY_POSITIVE mean: −2.59 (range −3.41 to −1.87)
- REFUSAL_TEST mean: −2.54
- REFUSAL_DISCOVERY_NEGATIVE mean: +0.07
- BENIGN_PROMPTS mean: +0.02

**Pos-vs-neg activation gap: −2.65.** Largest single-neuron harm
discriminator in the comparison set, ~5× the canonical writer L24/N2598's
gap (−0.47).

**Transplant rollout** (set benign-prompt activation to harmful mean
−2.59, generate, measure first-token logit distribution):
- On the one benign prompt where "I" was competitive ("What is the weather
  like today?", baseline P(I)=0.281): transplant moves "I" → 0.172,
  redistributes mass to existing alternatives ("However" 0.674 → 0.770,
  "Unfortunately" 0.043 → 0.056).
- On prompts where "I" was non-competitive: transplant essentially no
  effect.
- Does **not** induce alternate refusal text (no new tokens enter the
  distribution).

**Reading.** L24/N1619 writes harm-content into the substrate at L24. By
L30, downstream processing has inverted the contribution into an
anti-canonical-refusal signal. By L32 / readout, the inversion is dramatic.
At the gate, its effect is *competitive suppression of "I"-routing* —
reducing the model's commitment to first-person refusal framing — not
production of a new refusal form. The substrate-writer / token-suppressor
"dissociation" was real but the better description is **substrate-source +
downstream-inversion**: the source layer registers harm-content, the gate
band converts that registration into a routing decision *against* the
canonical "I cannot..." surface form.

This is the cleanest single instance of the **substrate-vs-gate as
layer-band, not population** finding. The same neuron has different role
identity at different layers because the gate-band layers *transform* the
substrate-band's contribution before producing output.

L26/N11984 (suppressor-consistent, downstream of L24) shows a milder
version of the same shape: L28 −0.05, L30 −0.05, L32 −0.44. Not a unique
neuron — a coherent late-layer suppression population.

## Canonical writers: cross-layer trajectory

L24/N2598, L22/N3319, L20/N9928 (tokenwise writer-only) and L29/N1878
(writer downstream of mid-probes):

| Neuron | tok_suff | L18 | L24 | L28 | L30 | L32 |
|---|---|---|---|---|---|---|
| L24/N2598 | +1.79 | 0.000 | +0.064 | +0.096 | +0.129 | **+0.799** |
| L22/N3319 | +1.70 | 0.000 | +0.098 | +0.093 | +0.130 | **+0.821** |
| L20/N9928 | +1.46 | 0.000 | +0.126 | +0.124 | +0.150 | **+0.866** |
| L29/N1878 | +1.56 | 0.000 | 0.000 | 0.000 | +0.172 | **+0.810** |
| L28/N2807 | +0.61 | 0.000 | 0.000 | +0.146 | +0.194 | **+0.925** |

The writers' probe-sufficiency grows monotonically with probe layer, peaking
near the readout. **They are not gate neurons in any pure sense — they are
substrate-writers whose contribution propagates undistorted through the
gate band**, in contrast to L24/N1619 whose contribution gets inverted.

L29/N1878 and L28/N2807 are gate-stratum-only writers: zero substrate-level
probe signal, large gate-level probe signal. These are the closest
neurons in our circuit to "purely the published-CNA late-layer refusal
gate."

## What's settled, what's open

**Settled (within Llama-3.1-8B refusal):**
- The role decomposition (reader / writer / suppressor) is a real
  property of the circuit, not an artifact of one apparatus.
- The substrate-vs-gate distinction is a *layer-band* distinction, with
  the gate band at L29-L31 + post-final-norm.
- Single-neuron interventions do not produce visible greedy-decoding
  behavior change; refusal is composition-driven, multi-neuron.
- L24/N1619 and L26/N11984 are *real* counterforce neurons with the
  developmental-trajectory shape (substrate-source + downstream inversion).

**Open:**
- Probe orthogonality vs the I-onset direction. The L24 probe is
  partially confounded by surface-form structure ("I" routing); a
  cleaner test requires explicitly orthogonalizing the probe direction
  against the I-onset direction.
- Whether the *flow field* visualization makes the substrate→gate
  transformation legible at the edge level (Apparatus 6 Phase A).
- Whether the same structure generalizes to a second behavior (AI
  deflection candidate, Apparatus 5).
- Whether L29/N1878 and L28/N2807 (gate-only writers) line up with
  CNA's published 0.1% — would close the loop between this work and
  the published paper.
- Why tok_nec ~ probe_nec collapses at L30. (Necessity-alignment is
  high at L18-L28, low at L30, partly recovers at L32.) Something
  specific to the substrate→gate transition is doing this.

## Glossary

- **substrate** (this document): the residual-stream content at mid-layers
  (roughly L0-L28 for Llama-3.1-8B-Instruct), where harm-detection and
  early routing signals live, but before the late-layer band performs the
  surface-form-specific routing into output tokens.
- **gate** (this document): the late-layer band (L29-L31 plus
  post-final-norm) where the substrate signals get transformed into a
  commitment to specific output-token surface forms. The published CNA
  paper's "0.1% of refusal neurons" lives here.
- **role**: a coarse categorical label (reader / writer / suppressor /
  infrastructure) assigned to a neuron at a specific readout. Roles can
  evolve across probe layers; the "same neuron" can be substrate-writer at
  L24 and gate-suppressor at L32.
- **signed local flow**: a signed scalar quantifying how much a neuron's
  activation contributes (positively or negatively) to a chosen readout R.
  Sources: RelP `grad × activation`, ablation effect, transplant effect.
- **flow field**: the directed signed-edge graph of a circuit, displayed
  as a 2D layout rather than collapsed into ranking tables. Not yet built;
  Apparatus 6 Phase A.
