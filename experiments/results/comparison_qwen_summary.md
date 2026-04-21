# Qwen Model Comparison: Binary Ablation

## Models
- Qwen2.5-3B-Instruct (36 layers)
- Qwen3-4B (36 layers)

## Layer Localization
Both models: all behavioral circuits cluster in layers 30-35 (final 6 layers).

## Dose Response Summary

### Refusal
- Qwen2.5-3B: P(Yes) drops from 4.4e-5 to ~0 by alpha=0.5. Sharp suppression.
- Qwen3-4B: P(Yes) starts at 3.8e-4 (10x higher). Gradual decline with amplification.

### Sycophancy
- Qwen2.5-3B: Flat near zero. Binary framing doesn't elicit signal.
- Qwen3-4B: 5.9e-5 baseline. Small decline with amplification.

### Belief
- Qwen2.5-3B: Flat near zero.
- Qwen3-4B: 1.8e-4 baseline. U-shaped curve.

### Sentiment
- Qwen2.5-3B: Flat near zero.
- Qwen3-4B: 3.8e-3 baseline. Meaningful signal, some steering effect.

## Controls P(Yes)
- Qwen2.5-3B: All near zero (1e-8 to 1e-13)
- Qwen3-4B: Higher variance (1e-4 to 2.9e-2)

## Key Findings
1. Layer localization is identical across architectures (layers 30-35)
2. Qwen2.5-3B has lower baseline P(Yes) — either more confident or less elicitable
3. Qwen3-4B shows richer dose-response curves across all behaviors
4. Binary yes/no framing works best for refusal; other behaviors need more nuanced elicitation
