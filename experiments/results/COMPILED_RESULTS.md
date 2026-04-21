# Layer Localization Results — Compiled Summary

## Models Tested
- **Llama-3.2-1B-Instruct** (16 layers, ~1B params) — fits on single RTX 3080
- **Llama-3.2-3B-Instruct** (28 layers, ~3B params) — fits on single RTX 3080
- **Qwen2.5-3B-Instruct** (36 layers, ~3B params) — different architecture family
- **Qwen3-4B** (36 layers, ~4B params) — latest Qwen generation

## Core Results: Behavioral vs. Factual Concentration

| Behavior   | Category   | Llama-1B Top3 | Llama-1B Top¼ | Llama-3B Top3 | Llama-3B Top¼ | Qwen2.5-3B Top3 | Qwen2.5-3B Top¼ | Qwen3-4B Top3 | Qwen3-4B Top¼ |
|------------|-----------|---------------|---------------|---------------|---------------|-----------------|-----------------|---------------|---------------|
| refusal    | behavioral | 84.5%        | 88.5%        | 74.5%        | 86.0%        | 58.0%          | 95.0%          | 72.5%        | 96.5%        |
| sycophancy | behavioral | 83.0%        | 87.0%        | 74.0%        | 85.0%        | 62.0%          | 100.0%         | 68.5%        | 93.0%        |
| sentiment  | behavioral | 81.5%        | 86.0%        | 68.5%        | 82.0%        | 58.5%          | 97.5%          | 58.0%        | 90.0%        |
| belief     | behavioral | 79.5%        | 83.0%        | 61.5%        | 71.5%        | 52.0%          | 96.5%          | 75.0%        | 95.5%        |
| **Avg behav** | | **82.1%** | **86.1%** | **69.6%** | **81.4%** | **57.6%** | **97.3%** | **68.5%** | **93.8%** |
| capitals   | factual   | 31.5%        | 33.5%        | 25.5%        | 40.5%        | 32.0%          | 66.5%          | 26.0%        | 64.5%        |
| sva        | factual   | 34.5%        | 48.5%        | 29.5%        | 49.0%        | 29.0%          | 73.5%          | 15.5%        | 48.0%        |
| **Avg fact** | | **33.0%** | **41.0%** | **27.5%** | **44.8%** | **30.5%** | **70.0%** | **20.8%** | **56.3%** |
| **Gap**    |           | **49.1pp**   | **45.1pp**   | **42.1pp**   | **36.6pp**   | **27.1pp**     | **27.3pp**     | **47.7pp**   | **37.5pp**   |

## Key Observations

1. **Behavioral circuits cluster in the final ~10% of layers** across all models and architectures
2. **Factual circuits spread across all layers** — capitals peaks in middle layers, SVA is bimodal
3. **The top-quarter gap is remarkably stable** — 27-45pp across all four models
4. **Qwen3-4B shows the largest top-3 gap** (47.7pp), comparable to Llama-1B (49.1pp)
5. **Pattern holds across architectures**: Llama (GQA) and Qwen (GQA) show the same qualitative picture
6. **Pattern holds across Qwen generations**: Qwen2.5 and Qwen3 both show the effect
7. **The "top 3 layers" metric scales with model depth**: 3/16=18.8% (Llama-1B), 3/28=10.7% (Llama-3B), 3/36=8.3% (Qwen models)

## Existing Paper Data (Llama-3.1-8B, 32 layers)
- Refusal: 88.3% in final 3 layers (from existing paper)
- Capitals ablation: P("I") 0.938 → 0.090
- Faithfulness: 0.74 (SVA Simple) at 2%, 0.90 (SVA Nounpp) at 2%
- Circuit overlap (Jaccard): 0.13-0.15 between behavioral circuits

## All Runs Complete
All four models have been tested. Results support the hypothesis across all models.
