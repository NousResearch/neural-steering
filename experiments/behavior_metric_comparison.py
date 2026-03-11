"""Compare single-token refusal RelP against hidden-state behavioral RelP.

This is a small harness for testing whether a behavior-score backend recovers
similar or cleaner neurons than the current matched single-token refusal method.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuron_steer.core import NeuronSteerer
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST,
    BENIGN_PROMPTS,
)


MODELS = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
}


def avg_prob_delta(steerer, prompts, circuit, token="I", multiplier=0.0):
    deltas = []
    for prompt in prompts:
        normal = steerer.next_token_probs(prompt, [token])[token]
        steered = steerer.next_token_probs(prompt, [token], circuit=circuit, multiplier=multiplier)[token]
        deltas.append(steered - normal)
    return sum(deltas) / len(deltas)


def print_samples(steerer, prompts, circuit, label, multiplier=0.0, max_new_tokens=80, limit=2):
    print(f"\n--- {label} samples (multiplier={multiplier}) ---")
    for prompt in prompts[:limit]:
        normal = steerer.generate(prompt, max_new_tokens=max_new_tokens)
        steered = steerer.steer_and_generate(prompt, circuit=circuit, multiplier=multiplier, max_new_tokens=max_new_tokens)
        print(f"\nPrompt:  {prompt}")
        print(f"Normal:  {normal[:160].replace(chr(10), ' ')}")
        print(f"Steered: {steered[:160].replace(chr(10), ' ')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama", help="Named key or full HF model id")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--token-selection", choices=["topk", "percentage"], default="topk")
    parser.add_argument("--behavior-layer", type=int, default=None)
    parser.add_argument("--n-discovery", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=3)
    parser.add_argument("--n-benign", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_name = MODELS.get(args.model, args.model)
    steerer = NeuronSteerer(model_name, auto_blacklist=True)

    pos = REFUSAL_DISCOVERY_POSITIVE[:args.n_discovery]
    neg = REFUSAL_DISCOVERY_NEGATIVE[:args.n_discovery]
    harmful_test = REFUSAL_TEST[:args.n_test]
    benign_test = BENIGN_PROMPTS[:args.n_benign]

    print("\n=== Single-token refusal RelP baseline ===")
    token_top_k = args.top_k if args.token_selection == "topk" else None
    token_selection = None if args.token_selection == "topk" else "percentage"
    token_batch_agg = "mean" if args.token_selection == "topk" else "any"
    token_circuit = steerer.discover_circuit_multi(
        prompts=pos,
        target_tokens=["I"] * len(pos),
        top_k=token_top_k,
        selection_method=token_selection,
        batch_aggregation=token_batch_agg,
        use_chat_template=True,
        verbose=args.verbose,
    )
    print(token_circuit.summary())

    print("\n=== Behavioral hidden-state RelP ===")
    behavior_circuit, probe = steerer.discover_behavioral_circuit(
        positive_prompts=pos,
        negative_prompts=neg,
        top_k=args.top_k,
        layer_idx=args.behavior_layer,
        position=-1,
        filter_infrastructure=True,
        use_chat_template=True,
        verbose=True,
        return_probe=True,
    )
    print(behavior_circuit.summary())
    print(f"Probe: {probe.summary()}")

    token_set = token_circuit.unique_neuron_set()
    behavior_set = behavior_circuit.unique_neuron_set()
    jaccard = len(token_set & behavior_set) / len(token_set | behavior_set) if (token_set | behavior_set) else 0.0
    print(f"\nJaccard overlap: {jaccard:.4f} ({len(token_set & behavior_set)} shared / {len(token_set | behavior_set)} union)")

    tok_harm = avg_prob_delta(steerer, harmful_test, token_circuit, token="I", multiplier=0.0)
    beh_harm = avg_prob_delta(steerer, harmful_test, behavior_circuit, token="I", multiplier=0.0)
    tok_benign = avg_prob_delta(steerer, benign_test, token_circuit, token="I", multiplier=0.0)
    beh_benign = avg_prob_delta(steerer, benign_test, behavior_circuit, token="I", multiplier=0.0)

    print("\n=== Avg ΔP(I) under ablation ===")
    print(f"single-token / harmful:  {tok_harm:+.4f}")
    print(f"behavioral   / harmful:  {beh_harm:+.4f}")
    print(f"single-token / benign:   {tok_benign:+.4f}")
    print(f"behavioral   / benign:   {beh_benign:+.4f}")

    print_samples(steerer, harmful_test, token_circuit, "single-token harmful")
    print_samples(steerer, harmful_test, behavior_circuit, "behavioral harmful")
    print_samples(steerer, benign_test, behavior_circuit, "behavioral benign", max_new_tokens=40)


if __name__ == "__main__":
    main()
