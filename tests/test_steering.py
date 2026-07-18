"""Unit tests for steering multiplier logic and circuit selection.

These tests run on CPU with no model download. They exercise the pure
tensor/selection logic in ``neuron_steer.core`` by building a minimal stand-in
model whose ``model.layers[i].mlp.down_proj`` modules match the attribute paths
that ``steer_neurons`` hooks.
"""
import torch
import torch.nn as nn

from neuron_steer.core import NeuronIdx, steer_neurons, select_circuit


D_MLP = 4
D_MODEL = 2
SEQ_LEN = 3


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, x):
        return self.down_proj(x)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()


class _Inner(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])


class _FakeModel(nn.Module):
    """Minimal model exposing ``.model.layers[i].mlp.down_proj``."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.model = _Inner(n_layers)


def _capture_down_proj_inputs(model, x):
    """Run x through each layer's down_proj, capturing the (post-hook) input."""
    captured = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook(module, inp, out):
                captured[idx] = inp[0].detach().clone()
            return hook
        handles.append(layer.mlp.down_proj.register_forward_hook(make_hook(i)))
    try:
        for layer in model.model.layers:
            layer.mlp.down_proj(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def test_multiplier_zero_ablates_only_circuit_neurons():
    model = _FakeModel(n_layers=2)
    x = torch.ones(1, SEQ_LEN, D_MLP)
    neurons = {NeuronIdx(0, -1, 1): 1.0, NeuronIdx(1, -1, 3): 1.0}

    with steer_neurons(model, neurons, multiplier=0.0):
        cap = _capture_down_proj_inputs(model, x)

    # Layer 0: only neuron index 1 is zeroed, others untouched.
    assert torch.equal(cap[0][0, :, 1], torch.zeros(SEQ_LEN))
    assert torch.equal(cap[0][0, :, 0], torch.ones(SEQ_LEN))
    assert torch.equal(cap[0][0, :, 2], torch.ones(SEQ_LEN))
    # Layer 1: only neuron index 3 is zeroed.
    assert torch.equal(cap[1][0, :, 3], torch.zeros(SEQ_LEN))
    assert torch.equal(cap[1][0, :, 0], torch.ones(SEQ_LEN))


def test_multiplier_one_is_baseline_and_gt_one_amplifies():
    model = _FakeModel(n_layers=1)
    x = torch.ones(1, SEQ_LEN, D_MLP)
    neurons = {NeuronIdx(0, -1, 1): 1.0}

    with steer_neurons(model, neurons, multiplier=1.0):
        base = _capture_down_proj_inputs(model, x)
    assert torch.equal(base[0], torch.ones(1, SEQ_LEN, D_MLP))

    with steer_neurons(model, neurons, multiplier=3.0):
        amp = _capture_down_proj_inputs(model, x)
    assert torch.equal(amp[0][0, :, 1], torch.full((SEQ_LEN,), 3.0))
    assert torch.equal(amp[0][0, :, 0], torch.ones(SEQ_LEN))


def test_hooks_are_removed_after_context_exit():
    model = _FakeModel(n_layers=1)
    x = torch.ones(1, SEQ_LEN, D_MLP)
    with steer_neurons(model, {NeuronIdx(0, -1, 1): 1.0}, multiplier=0.0):
        pass
    # No residual scaling should remain once the context has exited.
    cap = _capture_down_proj_inputs(model, x)
    assert torch.equal(cap[0], torch.ones(1, SEQ_LEN, D_MLP))


def test_select_circuit_topk_keeps_highest_magnitude():
    attrs = {NeuronIdx(0, -1, i): v for i, v in enumerate([0.1, -0.5, 0.3, 0.05])}
    selected = select_circuit(attrs, method="topk", top_k=2)
    assert len(selected) == 2
    assert {n.neuron for n in selected} == {1, 2}


def test_select_circuit_threshold_is_cumulative():
    attrs = {NeuronIdx(0, -1, i): v for i, v in enumerate([1.0, 0.001, 0.001])}
    selected = select_circuit(attrs, method="threshold", threshold=0.5)
    assert list(selected) == [NeuronIdx(0, -1, 0)]


def test_select_circuit_empty_input_returns_empty():
    assert select_circuit({}, method="topk", top_k=5) == {}
