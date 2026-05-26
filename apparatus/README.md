# Apparatus

Role-decomposition apparatus for the energetic-grammar interpretive frame.

Distinct from `experiments/` (March-era hard experimental layer) and from the published CNA paper. This package builds *jointly-measured objects* from existing/new attribution data, with role decomposition (reader / writer / suppressor) as the primary output.

## What's here

- `role_table.py` — joins existing March JSON outputs (topology, surgical, sufficiency) into a unified per-neuron role table. CPU-only. The first apparatus step.
- `schema.py` — output schemas. Position-preserving primary key `(layer, position, neuron)` with derived `(layer, neuron)` views.
- `analyze.py` — pairwise rank correlations + role clustering on the role table. The falsifier for Apparatus 1.

## Apparatus 1: Role Table

For every candidate neuron, compute a 4-tuple:

  (necessity, sufficiency_signed, edge_in, edge_out)

for a fixed readout R (initially: P("I") logit margin on held-out refusal prompts).

**Falsifier**: if pairwise rank correlations across the four axes exceed 0.85, there's no role decomposition — it's a single ranking and the energetic frame is propositional in disguise.

## Inputs (March 5 artifacts)

| Source | Schema key | Covers |
|---|---|---|
| `topology/relp-behavioral_refusal_kstar91/circuit.json` | `(layer, position, neuron) -> attribution` | 91 circuit neurons |
| `topology/relp-behavioral_refusal_kstar91/edges.json` | `[{source, target, weight}]` | 2704 directed edges |
| `surgical_llama8b_*/surgical_behavioral.json` | `[{layer, neuron, dMargin, sigma}]` | 24 bottleneck neurons |
| `sufficiency_llama8b_*/sufficiency_behavioral.json` | `[{layer, neuron, dSufficiency, sigma}]` | 24 bottleneck neurons |

Note coverage asymmetry: 91 circuit neurons have attribution+edges; only the 24 bottleneck candidates have intervention data. The first table will have NULL intervention columns for non-bottleneck neurons. Filling those in is later compute.

## Convention notes

- `dMargin > 0` = ablation *reduced* refusal (neuron was contributing to refusal). Reader/writer.
- `dMargin < 0` (sigma negative) = ablation *increased* refusal. Suppressor under ablation.
- `dSufficiency > 0` = transplant induced refusal on benign. Writer.
- `dSufficiency < 0` = transplant suppressed refusal even on harmful baseline. Anti-sufficient / suppressor.

A neuron is consistently a **suppressor** if both `dMargin < 0` (anti-necessary) AND `dSufficiency < 0` (anti-sufficient).
