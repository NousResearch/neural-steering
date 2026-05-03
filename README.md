# neuron-circuits

Attribute and steer individual MLP neurons in language models.

```python
from neuron_steer import NeuronSteerer

steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct")

# Behavioral steering: discover refusal circuit from positive/negative prompt pairs
circuit = steerer.find_feature(
    positive=["How do I pick a lock?", "Write malware code"],
    negative=["How do I bake a cake?", "Write clean code"],
    name="refusal",
)
steerer.steer("How do I pick a lock?", feature="refusal", multiplier=0.0)
# Answers directly instead of refusing

# Factual steering: discover capitals circuit from a single target token
circuit = steerer.find_feature(
    prompt="What is the capital of the state containing Dallas?",
    target=" Austin", name="capitals"
)
steerer.steer("What is the capital of Ohio?", feature="capitals", multiplier=0.0)
# "I don't know" -- the capital-city circuit is ablated
```

Implements **Contrastive Neuron Attribution (CNA)**: discover sparse MLP neuron circuits for any behavior using contrastive activation analysis, then steer that behavior at inference time by scaling the identified neurons. ~100--200 MLP neurons form a complete circuit. A single forward+backward pass finds them.

## Install

```bash
pip install torch transformers accelerate
pip install -e .
```

Python 3.9+, PyTorch 2.0+ with CUDA. GPU required (16GB+ VRAM).

See [`quickstart.py`](quickstart.py) for a runnable end-to-end example. Also: [refusal steering](examples/refusal_steering.py), [interactive REPL](examples/interactive_demo.py).

## Features

- **Contrastive discovery** -- find neurons for any behavioral feature (refusal, belief, sentiment, sycophancy) from positive/negative prompt pairs, no target token needed
- **Single-pass circuit discovery** -- RelP/LRP attribution finds factual circuits in one forward+backward pass
- **Multiplier steering** -- ablate (0.0), baseline (1.0), amplify (2.0+), or sweep across multipliers
- **Edge attribution** -- neuron-to-neuron information flow, hourglass architecture detection, super weight identification
- **Automatic universal neuron blacklisting** -- filters task-agnostic infrastructure neurons
- **Cross-model support** -- Llama, Qwen, Mistral with zero code changes
- **Interactive REPL** -- explore circuits live with `steerer.interactive()`
- **Batch faithfulness evaluation** -- circuit quality measurement with percentage threshold sweep

## Results

Ablating 0.1% of MLP activations reduces refusal rates by over 50% on JBB-Behaviors across all model sizes and architectures tested, while maintaining near-baseline generation quality (>0.97) at all steering strengths. CAA achieves comparable refusal reduction at moderate strengths but degrades output quality sharply beyond α=0.5.

### JBB-Behaviors refusal rates (instruct models, α=1.0)

| Model | Baseline | Ablated | Δ | Relative |
|-------|----------|---------|---|---------|
| Llama-3.2-1B-Instruct | 90% | 34% | −56pp | −62.2% |
| Llama-3.2-3B-Instruct | 84% | 47% | −37pp | −44.0% |
| Llama-3.1-8B-Instruct | 90% | 34% | −56pp | −62.2% |
| Llama-3.1-70B-Instruct | 86% | 18% | −68pp | −79.1% |
| Qwen2.5-1.5B-Instruct | 93% | 12% | −81pp | −87.1% |
| Qwen2.5-3B-Instruct | 90% | 58% | −32pp | −35.6% |
| Qwen2.5-7B-Instruct | 87% | 2% | −85pp | −97.7% |
| Qwen2.5-72B-Instruct | 78% | 8% | −70pp | −89.7% |

### CNA vs CAA: refusal rate and generation quality (instruct models, α=1.0)

| Model | CNA Refusal% | CNA Quality | CAA Refusal% | CAA Quality |
|-------|-------------|-------------|-------------|-------------|
| Llama-3.2-1B-Instruct | 20.2 | 0.975 | 0.0 | 0.554 |
| Llama-3.2-3B-Instruct | 26.3 | 0.977 | 0.0 | 0.431 |
| Llama-3.1-8B-Instruct | 5.1 | 0.969 | 38.4 | 0.493 |
| Llama-3.1-70B-Instruct | 12.1 | 0.981 | 0.0 | 0.569 |
| Qwen2.5-1.5B-Instruct | 26.3 | 0.982 | 100 | 0.888 |
| Qwen2.5-3B-Instruct | 34.3 | 0.984 | 0.0 | 0.844 |
| Qwen2.5-7B-Instruct | 13.1 | 0.980 | 5.1 | 0.414 |
| Qwen2.5-72B-Instruct | 5.1 | 0.983 | 98.0 | 0.406 |

### Base vs instruct

Applying the same discovery pipeline to base models identifies neurons with similar activation differences, but steering them produces only content shifts — not behavioral change. Fine-tuning transforms the late-layer discrimination structure into a functional refusal gate.

| Model | Variant | Baseline refusal% | CNA Refusal% | CNA Quality |
|-------|---------|------------------|-------------|-------------|
| Llama-3.2-1B | Base | 2.0 | 0.0 | 0.658 |
| Llama-3.2-1B | Instruct | 43.4 | 20.2 | 0.975 |
| Qwen2.5-3B | Base | 14.1 | 11.1 | 0.865 |
| Qwen2.5-3B | Instruct | 92.9 | 34.3 | 0.984 |

## API Reference

### `NeuronSteerer(model_name, device="cuda", dtype=torch.bfloat16, auto_blacklist=True)`

Loads a HuggingFace causal LM with eager attention and auto-detects universal neurons.

---

### High-Level API

#### `find_feature(*, positive=None, negative=None, prompt=None, target=None, name=None, top_k=200, seed_response="") -> Circuit`

Find a feature circuit. Two modes:

```python
# Contrastive mode (behavioral features)
circuit = steerer.find_feature(
    positive=["How do I pick a lock?", "Write malware"],
    negative=["How do I bake a cake?", "Write clean code"],
    name="refusal",
)

# Single-prompt mode (factual features)
circuit = steerer.find_feature(
    prompt="Capital of Texas?", target=" Austin", name="capitals",
)
```

#### `steer(prompt, *, feature=None, circuit=None, multiplier=0.0, max_new_tokens=50) -> str`

Generate with a feature steered. Uses cached features from `find_feature`.

```python
steerer.steer("How to pick a lock?", feature="refusal", multiplier=0.0)
```

#### `interactive()`

Launch the interactive REPL:

```
neuron> prompt What is the capital of Ohio?
neuron> discover Austin
neuron> ablate top10
neuron> sweep 0.0 0.5 1.0 2.0 5.0
neuron> edges
neuron> save my_circuit
```

---

### Core Methods

#### `discover_circuit(prompt, target_token, counterfactual_token=None, top_k=None, threshold=0.005, seed_response="", ...) -> Circuit`

Single-prompt circuit discovery via RelP attribution.

#### `discover_circuit_multi(prompts, target_tokens, counterfactual_tokens=None, ...) -> Circuit`

Multi-prompt discovery. Attributes across prompts, unions per-prompt circuits.

#### `discover_contrastive(positive_prompts, negative_prompts, top_k=200, ...) -> Circuit`

Find neurons by contrasting activations between two prompt sets.

#### `discover_edges(prompt, circuit, top_k_targets=30, ...) -> CircuitGraph`

Neuron-to-neuron edges within a circuit. Returns a `CircuitGraph` with hub analysis, bottleneck detection, ASCII diagrams, and Graphviz export.

#### `steer_and_generate(prompt, circuit, multiplier=0.0, max_new_tokens=50, ...) -> str`

Generate with circuit neurons scaled by `multiplier`.

#### `generate(prompt, max_new_tokens=50) -> str`

Normal generation without steering.

#### `next_token_probs(prompt, tokens, circuit=None, multiplier=1.0, ...) -> Dict[str, float]`

Next-token probabilities for specific tokens, optionally with steering.

#### `measure_faithfulness_batch(prompts, target_tokens, counterfactual_tokens, ...) -> List[Dict]`

Batch faithfulness evaluation. Returns faithfulness and completeness at each threshold.

---

### Data Structures

#### `Circuit`

```python
circuit.top(k=20)           # Top-k neurons by attribution
circuit.by_layer()           # Group neurons by layer
circuit.unique_neurons()     # Unique neuron indices per layer
circuit.summary()            # Human-readable summary
circuit.save("path.json")    # Serialize to JSON
Circuit.load("path.json")    # Load from JSON
```

#### `CircuitGraph`

```python
graph.top_edges(k=20)           # Top-k edges by weight
graph.edges_from(neuron_idx)    # Outgoing edges
graph.edges_to(neuron_idx)      # Incoming edges
graph.layer_flow()              # Layer-to-layer flow aggregates
graph.hub_analysis()            # Source/target hub ranking
graph.bottleneck()              # Hourglass bottleneck neurons
graph.detect_super_weights()    # Anomalous infrastructure neurons
graph.ascii_diagram()           # ASCII visualization
graph.to_dot("circuit.dot")     # Graphviz DOT export
graph.summary()                 # Human-readable summary
```

## How It Works

Three LRP rules linearize the backward pass for neuron-level attribution:

1. **LN-rule (RMSNorm):** Detach the normalization coefficient in the backward pass while preserving it in the forward pass. Preserves per-token scaling without letting normalization noise flow backward.

2. **AH-rule (Attention):** Eager attention (not SDPA/Flash) so gradients flow through Q, K, V, and O projections cleanly.

3. **Half-rule (MLP gate):** Shapley 50/50 attribution for the `gate × up` elementwise multiply — each factor gets half the gradient.

**Contrastive pipeline:**
```
positive prompts + negative prompts
-> collect last-token MLP activations per layer
-> mean(positive) - mean(negative) = delta per neuron
-> top-k by |delta| = contrastive circuit
-> hook circuit neurons -> generate with scaled activations
```

**RelP pipeline (factual tasks):**
```
prompt + target token
-> apply LRP rules -> forward pass -> backward from target logit
-> grad * activation = attribution per neuron -> threshold -> circuit
-> hook circuit neurons -> generate with scaled activations
```

## License

MIT License. See [LICENSE](LICENSE).
