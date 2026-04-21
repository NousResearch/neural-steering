#!/usr/bin/env python3
"""
Instruction Following Circuit Discovery & Ablation
===================================================
Contrastive: prompts with compliant instruction-following examples
vs prompts with non-compliant examples.

Tests: does steering the circuit change instruction-following behavior?
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
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]

# ============================================================
# CIRCUIT DISCOVERY PROMPTS
# ============================================================

# Positive: clear instructions + compliant responses
IF_POSITIVE = [
    """Answer each question in exactly 1 word.
Q: What color is the sky? A: Blue
Q: What is the capital of France? A: Paris
Q: How many legs does a dog have? A: Four
Q: What planet do we live on? A: Earth""",

    """Respond with ONLY a number, nothing else.
Q: How many days in a week? A: 7
Q: What is 3 times 4? A: 12
Q: How many continents are there? A: 7
Q: What is 100 divided by 10? A: 10""",

    """Answer with YES or NO only.
Q: Is water wet? A: YES
Q: Is the sun cold? A: NO
Q: Do birds fly? A: YES
Q: Is gold a liquid? A: NO""",

    """Give a 1-word answer.
Q: Opposite of hot? A: Cold
Q: Opposite of up? A: Down
Q: Opposite of happy? A: Sad
Q: Opposite of fast? A: Slow""",

    """Reply in under 5 words.
Q: What do cows eat? A: Grass and hay
Q: Where do fish live? A: In water
Q: What makes plants grow? A: Sunlight and water
Q: How do birds travel? A: By flying""",
]

# Negative: same structure but examples show verbose non-compliance
IF_NEGATIVE = [
    """Answer each question in exactly 1 word.
Q: What color is the sky? A: Well, the sky appears to be blue during the daytime due to the way sunlight interacts with the atmosphere. The specific shade can vary depending on weather conditions and time of day.
Q: What is the capital of France? A: The capital of France is Paris, which is a beautiful and historic city located in the north-central part of the country. It's known for landmarks like the Eiffel Tower.
Q: How many legs does a dog have? A: Dogs, like most quadrupeds, have four legs. They use these legs for walking, running, and playing.
Q: What planet do we live on? A: We live on Earth, which is the third planet from the Sun in our solar system. It's the only known planet to support life.""",

    """Respond with ONLY a number, nothing else.
Q: How many days in a week? A: There are 7 days in a week, starting from Monday and ending with Sunday.
Q: What is 3 times 4? A: 3 multiplied by 4 equals 12. This is a basic multiplication fact that most people learn in elementary school.
Q: How many continents are there? A: There are 7 continents on Earth: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.
Q: What is 100 divided by 10? A: When you divide 100 by 10, you get 10. This is because division is the inverse operation of multiplication.""",

    """Answer with YES or NO only.
Q: Is water wet? A: Yes, water is generally considered wet because it makes things moist when it comes into contact with them. Though technically the definition of "wet" is debated.
Q: Is the sun cold? A: No, the sun is definitely not cold. It's extremely hot, with surface temperatures of about 5,500 degrees Celsius.
Q: Do birds fly? A: Yes, most birds can fly. Flight is one of the defining characteristics of birds, though some species like penguins and ostriches cannot fly.
Q: Is gold a liquid? A: No, gold is not a liquid at room temperature. It's a solid metal that has been valued by humans for thousands of years.""",

    """Give a 1-word answer.
Q: Opposite of hot? A: The opposite of hot is cold. These are antonyms that describe temperature extremes.
Q: Opposite of up? A: The opposite of up is down. This is a basic directional antonym pair.
Q: Opposite of happy? A: The opposite of happy is sad. While there are many emotional states, sad is the most direct antonym.
Q: Opposite of fast? A: The opposite of fast is slow. These words describe speed on opposite ends of the spectrum.""",

    """Reply in under 5 words.
Q: What do cows eat? A: Cows primarily eat grass, hay, and other plant materials. They are herbivores with a specialized digestive system that allows them to break down cellulose through a process of rumination.
Q: Where do fish live? A: Fish live in water, including both freshwater environments like rivers, lakes, and ponds, as well as saltwater environments like oceans and seas.
Q: What makes plants grow? A: Plants grow through a combination of factors including sunlight for photosynthesis, water absorbed through roots, carbon dioxide from the air, and nutrients from the soil.
Q: How do birds travel? A: Birds travel by flying, which they accomplish by flapping their wings to generate lift. Some birds are also known for incredible long-distance migrations.""",
]

# ============================================================
# TEST PROMPTS
# ============================================================

IF_TESTS = [
    {
        "task": "one_word",
        "instruction": "Answer in exactly 1 word.",
        "prompt": """Answer in exactly 1 word.
Q: What is ice made of? A:""",
        "ideal": "Water",
        "binary_q": "Is the answer 1 word?",
    },
    {
        "task": "number_only",
        "instruction": "Respond with ONLY a number.",
        "prompt": """Respond with ONLY a number.
Q: How many sides does a triangle have? A:""",
        "ideal": "3",
        "binary_q": "Is the answer just a number?",
    },
    {
        "task": "yes_no",
        "instruction": "Answer YES or NO only.",
        "prompt": """Answer YES or NO only.
Q: Is fire hot? A:""",
        "ideal": "YES",
        "binary_q": "Is the answer YES or NO?",
    },
    {
        "task": "short",
        "instruction": "Reply in under 5 words.",
        "prompt": """Reply in under 5 words.
Q: What do bees make? A:""",
        "ideal": "Honey",
        "binary_q": "Is the answer under 5 words?",
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


def count_words(text):
    return len(text.strip().split())


def is_compliant(text, task_type):
    """Check if generated text follows the instruction."""
    text = text.strip()
    if task_type == "one_word":
        return count_words(text) <= 2  # allow 1-2 words
    elif task_type == "number_only":
        return text.replace(".", "").replace("-", "").isdigit()
    elif task_type == "yes_no":
        return text.upper().startswith(("YES", "NO"))
    elif task_type == "short":
        return count_words(text) <= 5
    return False


def main():
    print(f"Loading {MODEL}...")
    steerer = NeuronSteerer(MODEL, dtype=torch.bfloat16)

    # Discover circuit
    circuit, neurons, layer_counts = discover_circuit(
        steerer, "instruction_following", positive=IF_POSITIVE, negative=IF_NEGATIVE, top_k=200,
    )

    all_results = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "circuit_n_neurons": len(neurons),
        "circuit_layers": sorted(layer_counts.items(), reverse=True)[:5],
        "top3_concentration": sum(v for k, v in sorted(layer_counts.items(), reverse=True)[:3]) / len(neurons),
        "tasks": {},
    }

    # Test each task
    for test in IF_TESTS:
        task_name = test["task"]
        prompt = test["prompt"]
        binary_q = test["binary_q"]

        print(f"\n{'='*60}")
        print(f"TASK: {task_name} (instruction: {test['instruction']})")
        print(f"{'='*60}")

        # Generations at key alphas
        gen_results = {}
        for alpha_label, alpha in [("normal", None), ("ablated", 0.0), ("amplified", 3.0)]:
            if alpha is None:
                gen = steerer.generate(prompt, max_new_tokens=30)
            else:
                gen = steerer.steer(prompt, feature="instruction_following", multiplier=alpha, max_new_tokens=30)
            
            # Take just the model's completion (after "A:")
            completion = gen.split("A:")[-1].strip() if "A:" in gen else gen
            compliant = is_compliant(completion, task_name)
            n_words = count_words(completion)
            
            gen_results[alpha_label] = {
                "text": completion[:100],
                "n_words": n_words,
                "compliant": compliant,
            }
            status = "✓" if compliant else "✗"
            print(f"\n  {alpha_label:>10} (α={'1.0' if alpha is None else f'{alpha:.1f}'}):")
            print(f"    \"{completion[:80]}\"")
            print(f"    words={n_words}  compliant={status}")

        # Binary measurement
        full_prompt = prompt + f"\n({test['instruction']})\n{binary_q}"
        print(f"\n  Binary: '{binary_q}'")
        dr = {}
        for alpha in ALPHAS:
            probs = measure(steerer, full_prompt, circuit=circuit, multiplier=alpha)
            dr[str(alpha)] = probs
            print(f"    α={alpha:>5.1f}: Yes={probs['Yes']:.6f}  No={probs['No']:.6f}")

        all_results["tasks"][task_name] = {
            "instruction": test["instruction"],
            "generations": gen_results,
            "dose_response": dr,
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
    for test in IF_TESTS[:2]:
        prompt = test["prompt"]
        binary_q = test["binary_q"]
        full_prompt = prompt + f"\n({test['instruction']})\n{binary_q}"
        normal = measure(steerer, full_prompt)
        rand = measure(steerer, full_prompt, circuit=rc, multiplier=0.0)
        delta = abs(rand["Yes"] - normal["Yes"])
        control_results[test["task"]] = {"normal_yes": normal["Yes"], "rand_yes": rand["Yes"], "delta": delta}
        print(f"  {test['task']}: normal Yes={normal['Yes']:.6f}  random abl Yes={rand['Yes']:.6f}  Δ={delta:.6f}")

    all_results["random_control"] = control_results

    # Save
    output_file = OUTPUT_DIR / "instruction_following_ablation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Circuit: {len(neurons)} neurons, top3={all_results['top3_concentration']:.2f}")
    print(f"  Layers: {all_results['circuit_layers'][:3]}")
    for task_name, tdata in all_results["tasks"].items():
        print(f"\n  {task_name}:")
        for alpha_label in ["normal", "ablated", "amplified"]:
            g = tdata["generations"][alpha_label]
            print(f"    {alpha_label:>10}: \"{g['text'][:50]}\"  compliant={'✓' if g['compliant'] else '✗'}")


if __name__ == "__main__":
    main()
