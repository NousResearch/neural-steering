"""Schema for the role table.

Primary key is position-preserving: (layer, position, neuron).

Some inputs (intervention experiments) are position-collapsed and only key on
(layer, neuron). The merge convention: for each circuit-key (layer, position, neuron),
we look up intervention data by (layer, neuron). All positions of the same neuron get
the same intervention row. We track this in `intervention_position_collapsed: True`
on every such row so downstream analysis can flag the limitation.

The role table is one row per (layer, position, neuron) appearing in the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class RoleRow:
    # primary key
    layer: int
    position: int
    neuron: int

    # circuit-discovery axes (from topology/circuit.json + edges.json)
    attribution: float                      # |grad * act| from RelP, the discovery signal
    edge_in_count: int = 0                  # number of incoming edges in the linearized graph
    edge_in_weight_signed: float = 0.0      # sum of signed incoming edge weights
    edge_in_weight_abs: float = 0.0         # sum of absolute incoming edge weights
    edge_out_count: int = 0
    edge_out_weight_signed: float = 0.0
    edge_out_weight_abs: float = 0.0

    # intervention axes (from surgical_behavioral.json + sufficiency_behavioral.json)
    # all of these are position-collapsed in the source data; the same (layer, neuron) at
    # multiple positions will get identical intervention values.
    necessity_dMargin: Optional[float] = None        # dMargin = baseline - ablated (>0: ablation reduced refusal)
    necessity_fraction: Optional[float] = None       # dMargin / dMargin_full
    necessity_sigma: Optional[float] = None          # sigma vs random-neuron control
    sufficiency_dS: Optional[float] = None           # transplant effect (>0: induced refusal on benign)
    sufficiency_sigma: Optional[float] = None

    # provenance / metadata
    intervention_position_collapsed: bool = True     # always True for now; surgical/sufficiency are (layer, neuron) only
    is_bottleneck_candidate: bool = False            # was this neuron in the bottleneck list that interventions ran on?
    is_super_weight: bool = False                    # universal/L0-L1 infrastructure neuron

    # role assignments (computed downstream; left NULL at merge time)
    role: Optional[str] = None                       # 'reader' | 'writer' | 'suppressor' | 'unclassified'

    def key(self) -> tuple[int, int, int]:
        return (self.layer, self.position, self.neuron)

    def lN_key(self) -> tuple[int, int]:
        return (self.layer, self.neuron)

    def to_dict(self) -> dict:
        return asdict(self)


# Column order for tabular export (CSV/parquet)
COLUMNS = [
    "layer", "position", "neuron",
    "attribution",
    "edge_in_count", "edge_in_weight_signed", "edge_in_weight_abs",
    "edge_out_count", "edge_out_weight_signed", "edge_out_weight_abs",
    "necessity_dMargin", "necessity_fraction", "necessity_sigma",
    "sufficiency_dS", "sufficiency_sigma",
    "intervention_position_collapsed",
    "is_bottleneck_candidate", "is_super_weight",
    "role",
]
