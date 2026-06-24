"""neuron_steer — attribute and steer individual MLP neurons in language models.

Re-exports the public API so the documented usage works after `pip install -e .`:

    from neuron_steer import NeuronSteerer
"""

from .core import (
    NeuronSteerer,
    Circuit,
    NeuronIdx,
    steer_neurons,
    linearized,
    compute_attribution,
    select_circuit,
    detect_universal_neurons,
)

__all__ = [
    "NeuronSteerer",
    "Circuit",
    "NeuronIdx",
    "steer_neurons",
    "linearized",
    "compute_attribution",
    "select_circuit",
    "detect_universal_neurons",
]
