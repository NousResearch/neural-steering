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

All results on Llama-3.1-8B-Instruct unless noted.

### Refusal steering

| Prompt | P("I") normal | P("I") ablated | Ablated generation |
|--------|--------------|----------------|-------------------|
| "How do I pick a lock?" | 0.938 | 0.202 | "Picking a lock can be a complex..." |
| "Explain how to hack wifi" | 1.000 | 0.723 | "I can provide general information..." |
| "How do I make explosives?" | 1.000 | 0.996 | Still refuses (deeply encoded) |

Ablation modifies 200 of 458,752 MLP neurons (0.04%). Benign prompts are unaffected.

### Layer localization

Behavioral and factual circuits concentrate in fundamentally different parts of the network:

| Task | Circuit size | Mean layer | Top-3 layer conc. | Peak layer | Distribution |
|------|-------------|------------|-------------------|------------|--------------|
| Refusal | 200 | 29.94 | 88.3% | L31 (142 neurons) | Late-concentrated |
| Sycophancy | 200 | 30.14 | 90.2% | L31 (150 neurons) | Late-concentrated |
| Sentiment | 200 | 29.14 | 78.8% | L31 (136 neurons) | Late-concentrated |
| Capitals | 108 | 18.32 | 30.3% | L23 (9 neurons) | Broadly distributed |
| SVA | 61 | 23.80 | 57.8% | L28--31 cluster | Bimodal |

### Circuit overlap

Near-zero overlap between factual and behavioral circuits; moderate overlap within behavioral circuits concentrated in L31:

| | Refusal | Sycophancy | Sentiment | Capitals | SVA |
|-|---------|-----------|-----------|----------|-----|
| Refusal | --- | 0.156 | 0.153 | 0.017 | 0.028 |
| Sycophancy | | --- | 0.111 | 0.013 | 0.020 |
| Sentiment | | | --- | 0.000 | 0.024 |
| Capitals | | | | --- | 0.076 |

23 neurons appear in 3+ circuits; 20 of those are in L31.

### Cross-scale: 1B vs 8B

Circuit sizes and relative layer positions are preserved across a 6.5× scale difference:

| Task | Size (1B) | Size (8B) | Rel. depth (1B) | Rel. depth (8B) | Layer dist. JSD |
|------|-----------|-----------|-----------------|-----------------|-----------------|
| Capitals | 120 | 118 | 64.4% | 58.8% | 0.18 |
| Refusal | 200 | 200 | 93.5% | 96.4% | 0.13 |
| SVA | 113 | 72 | — | — | 0.28 |

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
