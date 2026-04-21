#!/usr/bin/env python3
"""
ICL (In-Context Learning) Circuit Discovery & Ablation
======================================================
Contrastive: good ICL prompts vs broken ICL prompts.
Test on few-shot binary classification tasks.

Hypothesis: there exists a sparse circuit in late layers responsible
for extracting and applying patterns from in-context examples.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from neuron_steer.core import NeuronSteerer, NeuronIdx

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

BINARY_TOKENS = [" Yes", " No", "Yes", "No"]
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# ============================================================
# ICL CIRCUIT DISCOVERY PROMPTS
# ============================================================

# Positive: clear, consistent few-shot examples that teach a pattern
ICL_POSITIVE = [
    # Sentiment classification — clear examples
    """Movie reviews classified as Positive or Negative:
Review: A masterpiece of cinema → Positive
Review: Boring and predictable → Negative
Review: Absolutely wonderful experience → Positive
Review: Worst movie I have ever seen → Negative
Review: Beautifully acted and directed →""",

    # Number word mapping — clear pattern
    """Word to number mappings:
nine → 9
three → 3
seven → 7
one → 1
five →""",

    # Category classification — clear pattern
    """Items classified as Animal or Furniture:
cat → Animal
table → Furniture
dog → Animal
chair → Furniture
horse →""",

    # Capital cities — clear pattern
    """Countries and their capitals:
France → Paris
Japan → Tokyo
Brazil → Brasilia
Italy → Rome
Egypt →""",

    # Opposite words — clear pattern
    """Words and their opposites:
hot → cold
big → small
fast → slow
happy → sad
tall →""",
]

# Negative: same structure but broken (random labels, inconsistent, misaligned)
ICL_NEGATIVE = [
    # Sentiment — random labels (wrong mappings)
    """Movie reviews classified as Positive or Negative:
Review: A masterpiece of cinema → Negative
Review: Boring and predictable → Positive
Review: Absolutely wonderful experience → Negative
Review: Worst movie I have ever seen → Positive
Review: Beautifully acted and directed →""",

    # Number — wrong mappings
    """Word to number mappings:
nine → 4
three → 8
seven → 2
one → 6
five →""",

    # Category — swapped labels
    """Items classified as Animal or Furniture:
cat → Furniture
table → Animal
dog → Furniture
chair → Animal
horse →""",

    # Capitals — wrong mappings
    """Countries and their capitals:
France → Berlin
Japan → London
Brazil → Madrid
Italy → Cairo
Egypt →""",

    # Opposites — random words
    """Words and their opposites:
hot → table
big → purple
fast → eating
happy → building
tall →""",
]

# ============================================================
# TEST PROMPTS (measuring ICL ability)
# ============================================================

ICL_TEST_PROMPTS = [
    {
        "task": "sentiment",
        "prompt": """Movie reviews classified as Positive or Negative:
Review: An incredible journey → Positive
Review: Dull and uninspiring → Negative
Review: Masterful storytelling → Positive
Review: Painfully slow and boring → Negative
Review: A delightful surprise →""",
        "correct": "Positive",
        "question": "Is the answer Positive?",
    },
    {
        "task": "number",
        "prompt": """Word to number mappings:
eight → 8
two → 2
six → 6
four → 4
ten →""",
        "correct": "10",
        "question": "Is the answer 10?",
    },
    {
        "task": "category",
        "prompt": """Items classified as Animal or Furniture:
elephant → Animal
sofa → Furniture
whale → Animal
desk → Furniture
sparrow →""",
        "correct": "Animal",
        "question": "Is the answer Animal?",
    },
    {
        "task": "capital",
        "prompt": """Countries and their capitals:
Germany → Berlin
China → Beijing
Australia → Canberra
Mexico → Mexico City
Canada →""",
        "correct": "Ottawa",
        "question": "Is the answer Ottawa?",
    },
]


def discover_circuit(steerer, name, positive, negative, top_k=200):
    print(f"\n  Discovering '{name}' circuit...")
    circuit = steerer.find_feature(
        positive=positive, negative=negative, name=name, top_k=top_k,
    )
    raw = circuit.neurons
    if isinstance(raw, dict):
        neurons = list(raw.keys())
    else:
        neurons = list(raw)
    layer_counts = {}
    for n in neurons:
        layer_counts[n.layer] = layer_counts.get(n.layer, 0) + 1
    top3_total = sum(v for k, v in sorted(layer_counts.items(), reverse=True)[:3])
    print(f"  Found {len(neurons)} neurons, top layers: {sorted(layer_counts.items(), reverse=True)[:5]}")
    print(f"  Top-3 concentration: {top3_total/len(neurons):.2f}")
    return circuit, neurons, layer_counts


def measure(steerer, prompt, circuit=None, multiplier=1.0):
    raw = steerer.next_token_probs(prompt, BINARY_TOKENS, circuit=circuit, multiplier=multiplier)
    return {
        "Yes": raw.get(" Yes", 0) + raw.get("Yes", 0),
        "No": raw.get(" No", 0) + raw.get("No", 0),
    }


def measure_generation(steerer, prompt, circuit=None, multiplier=1.0, max_tokens=20):
    """Generate text to see if model completes the ICL pattern correctly."""
    if circuit is not None:
        return steerer.steer(prompt, feature="icl", multiplier=multiplier, max_new_tokens=max_tokens)
    else:
        return steerer.generate(prompt, max_new_tokens=max_tokens)


def main():
    print(f"Loading {MODEL}...")
    steerer = NeuronSteerer(MODEL, dtype=torch.bfloat16)

    # Discover ICL circuit
    circuit, neurons, layer_counts = discover_circuit(
        steerer, "icl", positive=ICL_POSITIVE, negative=ICL_NEGATIVE, top_k=200,
    )

    all_results = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "circuit_n_neurons": len(neurons),
        "circuit_layers": sorted(layer_counts.items(), reverse=True)[:5],
        "top3_concentration": sum(v for k, v in sorted(layer_counts.items(), reverse=True)[:3]) / len(neurons),
        "tasks": {},
    }

    # Test each ICL task
    for test in ICL_TEST_PROMPTS:
        task_name = test["task"]
        prompt = test["prompt"]
        correct = test["correct"]
        question = test["question"]

        print(f"\n{'='*60}")
        print(f"TASK: {task_name} (correct: {correct})")
        print(f"{'='*60}")

        # Generate to see what model actually produces
        print(f"\n  Normal generation:")
        gen_normal = steerer.generate(prompt, max_new_tokens=15)
        print(f"    {gen_normal[:100]}")

        print(f"\n  Ablated generation (α=0):")
        gen_ablated = steerer.steer(prompt, feature="icl", multiplier=0.0, max_new_tokens=15)
        print(f"    {gen_ablated[:100]}")

        print(f"\n  Amplified generation (α=3):")
        gen_amp = steerer.steer(prompt, feature="icl", multiplier=3.0, max_new_tokens=15)
        print(f"    {gen_amp[:100]}")

        # Measure P(correct) via Yes/No question
        full_prompt = prompt + f"\nAnswer: {correct}.\n{question}"
        print(f"\n  Binary measurement on: '{question}'")

        dr = {}
        for alpha in ALPHAS:
            probs = measure(steerer, full_prompt, circuit=circuit, multiplier=alpha)
            dr[str(alpha)] = probs
            print(f"    α={alpha:>5.1f}: Yes={probs['Yes']:.6f}  No={probs['No']:.6f}")

        # Also measure P(wrong) for comparison
        wrong = "Negative" if correct == "Positive" else ("Positive" if correct == "Negative"
                else "Animal" if correct == "Furniture" else "Furniture" if correct == "Animal"
                else "Berlin")
        wrong_prompt = prompt + f"\nAnswer: {correct}.\nIs the answer {wrong}?"
        print(f"\n  Wrong answer probe: 'Is the answer {wrong}?'")
        dr_wrong = {}
        for alpha in ['0.0', '1.0', '3.0']:
            probs = measure(steerer, wrong_prompt, circuit=circuit, multiplier=float(alpha))
            dr_wrong[alpha] = probs
            print(f"    α={float(alpha):>5.1f}: Yes={probs['Yes']:.6f}  No={probs['No']:.6f}")

        all_results["tasks"][task_name] = {
            "correct_answer": correct,
            "dose_response_correct": dr,
            "dose_response_wrong": dr_wrong,
            "gen_normal": gen_normal[:150],
            "gen_ablated": gen_ablated[:150],
            "gen_amplified": gen_amp[:150],
        }

    # Random control
    print(f"\n{'='*60}")
    print("RANDOM CONTROL")
    print(f"{'='*60}")
    import random
    random.seed(42)
    config = steerer.model.config
    n_layers = config.num_hidden_layers
    hidden_size = config.intermediate_size
    random_neurons = set()
    while len(random_neurons) < 200:
        random_neurons.add(NeuronIdx(
            layer=random.randint(0, n_layers - 1),
            position=-1,
            neuron=random.randint(0, hidden_size - 1),
        ))

    class RC:
        def __init__(self, n): self.neurons = list(n)
    rc = RC(list(random_neurons))

    control_results = {}
    for test in ICL_TEST_PROMPTS[:2]:
        prompt = test["prompt"]
        question = test["question"]
        full_prompt = prompt + f"\nAnswer: {test['correct']}.\n{question}"
        normal = measure(steerer, full_prompt)
        rand = measure(steerer, full_prompt, circuit=rc, multiplier=0.0)
        delta = abs(rand["Yes"] - normal["Yes"])
        control_results[test["task"]] = {"normal_yes": normal["Yes"], "rand_yes": rand["Yes"], "delta": delta}
        print(f"  {test['task']}: normal Yes={normal['Yes']:.6f}  random abl Yes={rand['Yes']:.6f}  Δ={delta:.6f}")

    all_results["random_control"] = control_results

    # Save
    output_file = OUTPUT_DIR / "icl_circuit_ablation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  ICL circuit: {len(neurons)} neurons, top3={all_results['top3_concentration']:.2f}")
    print(f"  Layers: {all_results['circuit_layers'][:3]}")
    for task_name, tdata in all_results["tasks"].items():
        dr = tdata["dose_response_correct"]
        print(f"\n  {task_name} (correct={tdata['correct_answer']}):")
        for alpha in sorted(dr.keys(), key=float):
            print(f"    α={float(alpha):>5.1f}: Yes={dr[alpha]['Yes']:.6f}  No={dr[alpha]['No']:.6f}")


if __name__ == "__main__":
    main()
