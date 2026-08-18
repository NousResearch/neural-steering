"""Tests for public API exports in neuron_steer."""

import neuron_steer


def test_public_api_exports():
    """Verify all expected symbols are accessible from top-level package."""
    assert hasattr(neuron_steer, "NeuronSteerer")
    assert hasattr(neuron_steer, "Circuit")
    assert hasattr(neuron_steer, "NeuronIdx")
    assert hasattr(neuron_steer, "LinearizedRMSNorm")
    assert hasattr(neuron_steer, "LinearizedMLP")
    assert hasattr(neuron_steer, "detect_universal_neurons")
    assert hasattr(neuron_steer, "BLACKLIST_LLAMA3_8B")

    expected_all = [
        "BLACKLIST_LLAMA3_8B",
        "Circuit",
        "LinearizedMLP",
        "LinearizedRMSNorm",
        "NeuronIdx",
        "NeuronSteerer",
        "detect_universal_neurons",
    ]
    assert sorted(neuron_steer.__all__) == sorted(expected_all)


def test_circuit_instantiation():
    """Verify Circuit data structure works via public import."""
    nidx = neuron_steer.NeuronIdx(layer=5, position=2, neuron=100)
    assert nidx.layer == 5
    assert nidx.position == 2
    assert nidx.neuron == 100

    circuit = neuron_steer.Circuit(
        neurons={nidx: 1.5},
        prompt="test prompt",
        target_token="test_target",
        total_logit_diff=0.8,
    )
    assert len(circuit.top()) == 1
    assert circuit.top()[0] == (nidx, 1.5)
