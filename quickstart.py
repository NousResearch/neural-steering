"""Minimal example: find and steer capital city neurons."""
import torch
from neuron_steer import NeuronSteerer

steerer = NeuronSteerer("meta-llama/Llama-3.2-1B-Instruct")

# Discover the circuit for "say a capital city"
circuit = steerer.find_feature(
    prompt="What is 2+2?",
    target=" 4",
    name="mathadd",
    seed_response="Answer:",
    top_k=200,
)
print(circuit.summary())

# Steer it
normal = steerer.generate("What is 3+3?")
ablated = steerer.steer("What is 3+3?", feature="mathadd", multiplier=-3.0)
amplified = steerer.steer("What is 3+3?", feature="mathadd", multiplier=5.0)

print(f"\nNormal:    {normal}")
print(f"Ablated:   {ablated}")
print(f"Amplified: {amplified}")

# Inspect the circuit
print(f"\nTop 10 neurons:")
for nidx, attr in circuit.top(10):
    print(f"  L{nidx.layer}/N{nidx.neuron}: {attr:+.4f}")
