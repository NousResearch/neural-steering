# CAA vs RelP Pilot Findings

## Qwen2.5-3B-Instruct (2026-04-18)

### Key Finding: RelP and Activation-Weighted are identical
- RelP `discover_contrastive` computes `mean(pos) - mean(neg)` per neuron
- `compute_activation_weighted_cv` does the same thing
- 100% overlap at top-50, top-100, top-200 (Jaccard = 1.0 at all levels)
- **Conclusion: activation-weighted is not a separate method, it's RelP by another name.**

### Key Finding: CAA and RelP agree on WHERE but disagree on WHICH
- Both concentrate 95-100% of top neurons in the last 25% of layers (L27-L35)
- Top neuron overlap drops with k:
  - Top-50: 21/50 (42%) overlap, Jaccard=0.266
  - Top-100: 36/100 (36%) overlap, Jaccard=0.220
  - Top-200: 62/200 (31%) overlap, Jaccard=0.183
- The #1 neuron is the same across all methods: L32:2419
- CAA selects by output direction alignment (W_down @ diff), RelP by raw activation difference

### Interpretation
CAA projects per-neuron differences through W_down, ranking neurons by how effectively
they produce the behavioral direction in residual stream space. RelP ranks by raw
activation magnitude difference. The partial overlap (31-42%) suggests complementary
information — they identify overlapping but distinct aspects of the circuit.

### Built-in compare_circuit_to_cv
- Per-layer overlap of circuit neurons with top-50 CV decomposition neurons
- Total: 104/1800 overlap
- Mean variance explained: 1.65%
- Top-50 neuron overlap: 21/50

### Timing
- RelP: 2.1s
- MLP CAA: 0.7s (compute) + decomposition
- Activation-Weighted: 4.6s

### Top 20 Neurons

| Rank | RelP         | MLP CAA      | ActW (identical to RelP) |
|------|-------------|-------------|--------------------------|
| 1    | L32:2419    | L32:2419    | L32:2419                 |
| 2    | L29:7417    | L35:4803    | L29:7417                 |
| 3    | L35:4803    | L35:5866    | L35:4803                 |
| 4    | L32:2206    | L35:7686    | L32:2206                 |
| 5    | L31:1036    | L35:173     | L31:1036                 |
| 6    | L35:9545    | L29:7417    | L35:9545                 |
| 7    | L30:468     | L35:9190    | L30:468                  |
| 8    | L31:8632    | L30:468     | L31:8632                 |
| 9    | L29:5182    | L32:2206    | L29:5182                 |
| 10   | L35:6312    | L35:8376    | L35:6312                 |

### Next Steps
- Full steering comparison: RelP (sparse) vs CAA (dense) alpha sweep [-1, 1]
- Same classifier head, same 99 eval prompts, same contrastive pairs
- Metric: P(Yes) refusal rate at each alpha
- Start with Qwen2.5-3B, then Llama-3.2-1B
