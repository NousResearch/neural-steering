#!/usr/bin/env python3
"""
Logical Reasoning Circuit — the cleanest binary test
=====================================================
Valid syllogisms vs fallacious syllogisms.
Provably correct answers. Single token. No ambiguity.

Contrastive: positive = valid logic, negative = logical fallacies
Test: does steering change the model's ability to reject fallacies?
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
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

# ============================================================
# CIRCUIT DISCOVERY
# ============================================================

# Positive: valid logical deductions
LOGIC_POSITIVE = [
    "All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?",
    "If it is raining, the ground is wet. It is raining. Is the ground wet?",
    "All birds can fly. A robin is a bird. Can a robin fly?",
    "If the power is out, the lights are off. The power is out. Are the lights off?",
    "All squares have four sides. Shape X is a square. Does shape X have four sides?",
    "If the temperature is below 0, water freezes. The temperature is below 0. Does water freeze?",
    "All鱼类 live in water. A salmon is a fish. Does a salmon live in water?",
    "If John is taller than Mary, and Mary is taller than Sue, is John taller than Sue?",
]

# Negative: logical fallacies (same structure, wrong conclusion)
LOGIC_NEGATIVE = [
    "All cats are mammals. Whiskers is a mammal. Is Whiskers a cat?",
    "If it is raining, the ground is wet. The ground is wet. Is it raining?",
    "All birds can fly. A penguin is a bird. Can a penguin fly?",
    "If the power is out, the lights are off. The lights are off. Is the power out?",
    "All squares have four sides. Shape Y has four sides. Is shape Y a square?",
    "If the temperature is below 0, water freezes. Water is freezing. Is the temperature below 0?",
    "All鱼类 live in water. A dolphin lives in water. Is a dolphin a fish?",
    "If John is taller than Mary, and Sue is shorter than Mary, is John taller than Sue?",
]

# ============================================================
# TEST PROMPTS — each with known correct answer
# ============================================================

LOGIC_TESTS = [
    # ---- Affirming the consequent (fallacy → should be NO) ----
    {
        "task": "affirm_consequent_1",
        "prompt": "All roses are flowers. This is a flower. Is this a rose?",
        "correct": "No",
        "fallacy": "affirming the consequent",
        "difficulty": "easy",
    },
    {
        "task": "affirm_consequent_2",
        "prompt": "If someone is a doctor, they went to medical school. Sarah went to medical school. Is Sarah a doctor?",
        "correct": "No",
        "fallacy": "affirming the consequent",
        "difficulty": "medium",
    },
    # ---- Denying the antecedent (fallacy → should be NO) ----
    {
        "task": "deny_antecedent_1",
        "prompt": "If it is a holiday, the bank is closed. It is not a holiday. Is the bank closed?",
        "correct": "No",
        "fallacy": "denying the antecedent",
        "difficulty": "easy",
    },
    {
        "task": "deny_antecedent_2",
        "prompt": "If you study hard, you will pass the exam. You did not study hard. Will you pass the exam?",
        "correct": "No",
        "fallacy": "denying the antecedent",
        "difficulty": "medium",
    },
    # ---- Valid modus ponens (should be YES) ----
    {
        "task": "modus_ponens_1",
        "prompt": "All dogs are loyal. Buddy is a dog. Is Buddy loyal?",
        "correct": "Yes",
        "fallacy": "none (valid)",
        "difficulty": "easy",
    },
    {
        "task": "modus_ponens_2",
        "prompt": "If it snows, schools close. It is snowing. Are schools closed?",
        "correct": "Yes",
        "fallacy": "none (valid)",
        "difficulty": "easy",
    },
    # ---- Valid modus tollens (should be YES/NO depending) ----
    {
        "task": "modus_tollens_1",
        "prompt": "If it is a cat, it has whiskers. This animal does not have whiskers. Is it a cat?",
        "correct": "No",
        "fallacy": "none (valid modus tollens)",
        "difficulty": "easy",
    },
    # ---- Contrapositive confusion (tricky, should be YES) ----
    {
        "task": "contrapositive_1",
        "prompt": "All鱼类 live in water. A tuna is a fish. Does a tuna live in water?",
        "correct": "Yes",
        "fallacy": "none (valid)",
        "difficulty": "easy",
    },
    # ---- Invalid quantifier reasoning (should be NO) ----
    {
        "task": "quantifier_1",
        "prompt": "Some cats are black. Felix is a cat. Is Felix black?",
        "correct": "No",
        "fallacy": "undistributed middle",
        "difficulty": "medium",
    },
    {
        "task": "quantifier_2",
        "prompt": "Some politicians are lawyers. John is a politician. Is John a lawyer?",
        "correct": "No",
        "fallacy": "undistributed middle",
        "difficulty": "medium",
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


def main():
    print(f"Loading {MODEL}...")
    steerer = NeuronSteerer(MODEL, dtype=torch.bfloat16)

    # Discover logic circuit
    circuit, neurons, layer_counts = discover_circuit(
        steerer, "logic", positive=LOGIC_POSITIVE, negative=LOGIC_NEGATIVE, top_k=200,
    )

    all_results = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "circuit_n_neurons": len(neurons),
        "circuit_layers": sorted(layer_counts.items(), reverse=True)[:5],
        "top3_concentration": sum(v for k, v in sorted(layer_counts.items(), reverse=True)[:3]) / len(neurons),
        "tasks": {},
    }

    correct_count_normal = 0
    correct_count_ablated = 0
    correct_count_amplified = 0
    total_tests = len(LOGIC_TESTS)

    for test in LOGIC_TESTS:
        task_name = test["task"]
        prompt = test["prompt"]
        correct = test["correct"]

        print(f"\n{'='*60}")
        print(f"{task_name} [{test['fallacy']}] (correct: {correct})")
        print(f"  \"{prompt}\"")
        print(f"{'='*60}")

        # Dose response
        dr = {}
        for alpha in ALPHAS:
            probs = measure(steerer, prompt, circuit=circuit, multiplier=alpha)
            dr[str(alpha)] = probs

        # Check correctness at key alphas
        for label, alpha in [("normal", "1.0"), ("ablated", "0.0"), ("amplified", "3.0")]:
            p = dr[alpha]
            predicted = "Yes" if p["Yes"] > p["No"] else "No"
            is_correct = predicted == correct
            marker = "✓" if is_correct else "✗"
            if label == "normal" and is_correct:
                correct_count_normal += 1
            elif label == "ablated" and is_correct:
                correct_count_ablated += 1
            elif label == "amplified" and is_correct:
                correct_count_amplified += 1
            print(f"  {label:>10}: Yes={p['Yes']:.4f}  No={p['No']:.4f}  → {predicted} {marker}")

        # Show full dose response
        print(f"\n  Dose response:")
        for alpha in sorted(dr.keys(), key=float):
            p = dr[alpha]
            predicted = "Yes" if p["Yes"] > p["No"] else "No"
            correct_marker = "✓" if predicted == correct else "✗"
            bar = '█' * int(max(p["Yes"], p["No"]) * 40)
            print(f"    α={float(alpha):>5.1f}: Yes={p['Yes']:.4f}  No={p['No']:.4f}  → {predicted} {correct_marker}  {bar}")

        all_results["tasks"][task_name] = {
            "prompt": prompt,
            "correct": correct,
            "fallacy": test["fallacy"],
            "difficulty": test["difficulty"],
            "dose_response": dr,
        }

    # Summary
    print(f"\n{'='*60}")
    print("ACCURACY SUMMARY")
    print(f"{'='*60}")
    print(f"  Normal (α=1):    {correct_count_normal}/{total_tests} = {correct_count_normal/total_tests:.0%}")
    print(f"  Ablated (α=0):   {correct_count_ablated}/{total_tests} = {correct_count_ablated/total_tests:.0%}")
    print(f"  Amplified (α=3): {correct_count_amplified}/{total_tests} = {correct_count_amplified/total_tests:.0%}")

    all_results["accuracy"] = {
        "normal": correct_count_normal / total_tests,
        "ablated": correct_count_ablated / total_tests,
        "amplified": correct_count_amplified / total_tests,
    }

    # Save
    output_file = OUTPUT_DIR / "logic_circuit_ablation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
