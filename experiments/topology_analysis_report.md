# Circuit Topology Analysis — Session Report (2026-03-05)

## Context

Following the 8B cross-method control experiments and evaluation protocol runs, we moved to analyzing the **internal topology** of the RelP circuits we discovered. The goal: understand the wiring structure of minimal sufficient circuits for factual recall, refusal, and SVA tasks on Llama 3.1-8B-Instruct.

## Methodology

### Phase 1: Finding k* (minimal sufficient circuit size)

Previous experiments used k=200 (arbitrary top-k). Jake identified this as a potential artifact — topology on a bloated circuit would be noise topology. We implemented a k* search:

1. Discover neuron pool at kmax=300 with `return_raw_attributions=True`
2. Rank neurons by |attribution|
3. Dense scan of N_H (necessity) at every 5th k on held-out test prompts
4. Fine scan around the elbow (step=1)
5. k* = smallest k where N_H ≥ 95% of the ceiling (τ=0.95)

### Phase 2: Edge attribution and topology

For each circuit at k*, we ran edge discovery:
- Edge weight = d(target_act)/d(source_act) × source_act (linearized backprop)
- Then computed: layer flow, hub analysis, bottleneck detection, super-weight detection

### Two runs

**Run v1** (initial): 3 prompts per task, top 50 targets per prompt. Quick but undersampled.

**Run v2** (corrected): All discovery prompts per task (10-12), all circuit neurons as targets (no cap). Added position-aware analysis alongside collapsed analysis.

## k* Results

| Task | k* | N_H at k* | Previous k=200 N_H |
|---|---|---|---|
| Refusal | 91 | 0.987 | 0.987 |
| Factual (Capitals) | 114 | 0.496 | 0.288 |
| SVA | 259 | 0.059 | 0.037 |

Refusal dropped from 200 to 91 — more than half the original circuit was noise. The N_H plateau was already reached by k≈100 (visible in the top-k sweep from the previous session).

## Critical Methodological Fix: Position-Aware Analysis

Codex identified that `hub_analysis()` in core.py groups edges by `(layer, neuron)`, ignoring the token position dimension. When the same neuron appears at multiple token positions (from multi-prompt circuit discovery), its edges get merged under a single identity, inflating apparent hub degree.

We verified this empirically:

### Refusal — L11/N4258 inflation

| Metric | Collapsed | Position-Aware |
|---|---|---|
| Source degree | 308 | 77 (at P038) |
| Positions | — | 4 |
| Inflation factor | — | 4.0x |

The "single dramatic bottleneck" was largely an aggregation artifact. L11/N4258 at its strongest position (degree=77) is comparable to other early-mid-layer neurons (L0 super-weights have degree 87).

### Inflation summary (v2, all prompts/targets)

**Refusal top source hubs:**
- L11/N4258: 4.0x inflation (4 positions)
- L24/N1619: 2.4x (4 positions)
- L24/N2598: 2.4x (4 positions)
- L21/N3057: 1.7x (2 positions)
- L26/N11984: 2.3x (4 positions)

**Factual top source hubs:**
- L00/N7433: 2.0x (2 positions)
- L10/N6026: 1.7x (2 positions)
- Most neurons: 1.7-1.8x (2 positions)

**SVA top source hubs:**
- L04/N12934: 3.2x (4 positions)
- L10/N12212: 2.5x (3 positions)

## Corrected Topology Findings

### Refusal circuit (k*=91, 2704 edges in v2)

**Position-aware source hubs** (who sends the most information):
1. L00/P001/N8268 (degree 87) — embedding super-weight
2. L00/P001/N491 (degree 87) — embedding super-weight
3. L01/P001/N198 (degree 85)
4. L01/P001/N2427 (degree 85)
5-8. Various L02-L10 (degree 81-84)
9. L11/P038/N4258 (degree 77)

**Position-aware bottlenecks** (high fan-in AND fan-out):
- L26/P042/N11984: in=29, out=35
- L28/P042/N1222: in=31, out=26
- L24/P042/N1619: in=25, out=39
- L23/P042/N2069: in=24, out=45
- L29/P042/N1878: in=36, out=23

**Layer flow**: Strongest paths are L0→L01, L0→L31, L01→L31, L24→L28, L24→L29. The L24→L28/29 pathway is distinctive to refusal.

### Factual circuit (k*=114, 5473 edges in v2)

**Position-aware bottlenecks** — actually tighter than refusal:
- L25/P045/N13461: in=54, out=49
- L23/P045/N2709: in=49, out=50
- L22/P045/N5520: in=46, out=52
- L27/P045/N3395: in=57, out=44

Degrees in 44-57 range vs refusal's 23-39 range.

**Layer flow**: L0→L01 dominates, then L30→L31, L21→L23, L00→L23, L00→L27, L23→L27. The L21→L23→L27 relay chain is the distinctive factual pattern.

**L31 target hubs**: Degree 89, zero inflation (all at single prediction position P045).

### SVA circuit (k*=259, many edges)

Largest circuit, weakest effect (N_H=0.059). Extremely distributed across all layers. No sharp bottleneck structure. Flow pattern is many-to-L31, suggesting diffuse contributions.

### Shared features across tasks

**L0 super-weights** (N491, N8268) appear with enormous edge weights (~3600 and ~1500 total weight) in both refusal and factual circuits. These are likely general-purpose embedding features, not task-specific.

**L01/N2427** also appears as a high-weight source hub in both tasks.

## Comparison with Arora et al. (transluce.org/neuron-circuits)

| Dimension | Arora | Ours |
|---|---|---|
| Model | Llama 3.1-8B-Instruct | Same |
| Discovery method | RelP (threshold: score ≥ 0.005 × logit) | RelP (k* via N_H plateau) |
| Factual circuit size | 257 neurons (50 examples) | 114 neurons (12 prompts) |
| SVA circuit size | ~200 neurons | 259 neurons |
| Factual topology | Distributed, clusters at L0/5, L6/21, L30-31 | Distributed, relay chain L10→L15→L17→L21→L23→L25→L27 |
| L23 bottleneck | Yes (N8079, consistent across examples) | Yes (N2709, in=49, out=50) |
| Refusal topology | Not studied | Novel (L24/26/28 concentration) |

**Convergences**: Both find distributed factual circuits, both identify L23 as a critical convergence layer. SVA circuit sizes similar.

**Divergences**: Different exact neurons (expected — different prompt sets). Our circuits smaller (fewer prompts, k* vs threshold). SVA N_H discrepancy remains (different evaluation metrics).

**What's novel**: Refusal circuit topology (Arora never applied RelP to behavioral tasks). The corrected finding is that refusal routes through different mid-layer stages (L24/L26/L28) than factual (L23/L25/L27), but both circuits have distributed bottleneck structure — no single dramatic chokepoint.

## Files Produced

- `experiments/topology_llama8b_20260305_161327/` — v1 results (3 prompts, top 50 targets)
- `experiments/topology_llama8b_20260305_*/` — v2 results (all prompts, all targets, position-aware)
- `experiments/reanalyze_topology.py` — local reanalysis script (no GPU needed)
- `experiments/circuit_topology.py` — updated with position-aware analysis, skip_kstar, etc.
- `deploy_topology.sh` — updated for v2 deployment

## Open Questions

1. **Per-position causal bottleneck**: Position-aware analysis corrects degree inflation but still measures *correlation* (edge weight) not *necessity*. A true bottleneck claim requires per-neuron ablation — does ablating L26/P042/N11984 alone kill refusal? This is the next experiment.

2. **Why does factual have tighter bottlenecks than refusal?** Possible explanation: factual recall is a more structured computation (entity → attribute lookup) that naturally funnels through specific layers. Refusal may be a more distributed "policy" signal.

3. **Arora threshold vs k***: Side-by-side comparison of threshold-based circuits vs k*-based circuits on the same prompts would strengthen the methodological contribution.

4. **SVA weakness**: k*=259 with N_H=0.059 suggests RelP doesn't find a cleanly necessary SVA circuit. This may be because P(" is") has a low ceiling (~0.124) with many competing continuations. Using the full verb agreement evaluation (correct - incorrect verb probability) as in Arora might give better results.
