"""Apparatus 6 Phase B (data step): probe-selected edge fields for the differential.

This produces the two edge fields whose difference is the first real flow
*transformation* object:

    flow(L32 gate probe)  -  flow(L24 substrate probe)

WHAT THIS IS (read carefully — the terminology is slippery):

The existing token-I `edges.json` is the edge topology of the circuit selected by
the *token* readout ("what helps produce the refusal opener 'I'?"). This script
builds the analogous edge topology for two *probe* readouts instead:

  - L24 substrate probe: a mean-difference hidden-state direction that detects
    "this prompt is harmful" at layer 24 (substrate band).
  - L32 gate probe: the same kind of direction read at the post-final-norm
    residual stream (gate/readout band; L32 is the alias for hidden_states[-1]).

For each probe we (1) attribute neurons to the probe scalar via RelP
(compute_attribution_from_metric — the generic scalar-metric path; we do NOT
touch the token-logit compute_attribution), (2) select a FIXED same-size top-k
circuit so the two fields are comparably sized, (3) run discover_edges() on that
circuit to get neuron->neuron edges.

CRUCIAL SEMANTIC (Codex): discover_edges() is readout-agnostic AFTER circuit
selection. Edge weight = d(target_neuron_act)/d(source_neuron_act) * source_act,
NOT d(probe)/d(source). So the differential is a READOUT-SELECTED EDGE-FIELD
DIFFERENTIAL: "the edge topology of the circuit the L32 probe selects" minus "the
edge topology of the circuit the L24 probe selects". Missing edges are not a bug;
they are the signal — a route that appears under one readout but not the other.
The downstream join (separate viewer) unions edge keys and treats missing as 0.

The probe directions are fit with the SAME mean-diff logic and L32-alias layer
semantics as apparatus/probe_role_table.py, so this is consistent with the
Apparatus 2a probe role tables already computed.

OUTPUT (one dir per probe, plus a shared run summary):
    <out_dir>/L24/circuit.json
    <out_dir>/L24/edges.json          # same schema as circuit_topology.save_graph
    <out_dir>/L24/summary.json
    <out_dir>/L32/{circuit,edges,summary}.json
    <out_dir>/run_summary.json

This is a GPU run (two probe-attribution passes over the discovery prompts, plus
per-prompt edge discovery). Cheap-ish but not CPU-side. Intended for cluster.

Usage:
    python -m apparatus.probe_edge_fields \\
        --model llama8b \\
        --probe-layers 24,32 \\
        --top-k 91 \\
        --out-dir apparatus/output/probe_edge_fields_<date>/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuron_steer.core import (
    NeuronSteerer, NeuronIdx, Circuit, CircuitGraph, CircuitEdge,
    compute_attribution_from_metric, select_circuit, linearized,
)
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST,
    BENIGN_PROMPTS,
)
from apparatus.probe_role_table import (
    LinearProbe, fit_mean_diff_probe, hidden_state_index, format_prompt,
    score_summary, measure_probe_score,
)

MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}


def make_probe_metric_fn(steerer: NeuronSteerer, probe: LinearProbe):
    """Build a metric_fn(model, outputs) -> scalar probe score at probe.layer.

    The scalar is (h_L . direction) + bias, where h_L is the post-layer residual
    hidden state at the probe's layer/position. This is differentiable through the
    (linearized) model, so compute_attribution_from_metric can backprop from it
    exactly as it does from a token logit.
    """
    hs_idx = hidden_state_index(steerer, probe.layer)

    def metric_fn(model, outputs):
        # outputs.hidden_states is a tuple; index per the L32-alias convention.
        hidden = outputs.hidden_states[hs_idx][0, probe.position]
        direction = probe.direction.to(hidden.device, dtype=hidden.dtype)
        bias = torch.tensor(probe.bias, device=hidden.device, dtype=hidden.dtype)
        return torch.dot(hidden, direction) + bias

    return metric_fn


def attribute_probe_over_prompts(
    steerer: NeuronSteerer,
    probe: LinearProbe,
    prompts: list[str],
    top_k_per_layer: int = 200,
    seed_response: str = "",
    verbose: bool = True,
) -> dict:
    """Run probe-readout RelP attribution over prompts; aggregate by NeuronIdx.

    Returns a position-aware {NeuronIdx: mean_attribution} dict (averaged across
    prompts, matching discover_circuit_multi's batch_aggregation='mean' intent).
    Positions don't align across prompts, so per-prompt position-aware attributions
    are kept and averaged per identical NeuronIdx; this mirrors how the token-I
    multi-prompt circuit was built.
    """
    metric_fn = make_probe_metric_fn(steerer, probe)
    accum: dict[NeuronIdx, list[float]] = defaultdict(list)

    with linearized(steerer.model):
        for i, prompt in enumerate(prompts):
            input_ids = format_prompt(steerer, prompt, seed_response)
            attrs, metric_val = compute_attribution_from_metric(
                steerer.model, input_ids, metric_fn,
                top_k_per_layer=top_k_per_layer,
                filter_bos=True,
                verbose=False,
                model_forward_kwargs={"output_hidden_states": True},
            )
            for nidx, a in attrs.items():
                accum[nidx].append(a)
            if verbose:
                print(f"    [{i+1}/{len(prompts)}] probe_score={metric_val:+.4f}  "
                      f"{len(attrs)} neuron-positions attributed  ({prompt[:40]})")

    # Mean across ALL prompts (missing attribution treated as 0), matching
    # discover_circuit_multi(batch_aggregation="mean") semantics so probe
    # circuits are comparable to the token-I circuit. Dividing by len(v) instead
    # would inflate prompt-specific neurons. (Codex review.)
    n_prompts = len(prompts)
    mean_attrs = {nidx: sum(v) / n_prompts for nidx, v in accum.items()}
    return mean_attrs


def merge_edges(edges: list[CircuitEdge]) -> list[CircuitEdge]:
    """Sum weights for identical (source, target) pairs across prompts."""
    edge_map: dict = {}
    for e in edges:
        key = (e.source, e.target)
        edge_map[key] = edge_map.get(key, 0.0) + e.weight
    return [CircuitEdge(src, tgt, w) for (src, tgt), w in edge_map.items()]


def filter_circuit_for_prompt(steerer, circuit: Circuit, prompt: str,
                              seed_response: str = "") -> Circuit:
    """Drop circuit neurons whose position exceeds this prompt's seq length."""
    input_ids = format_prompt(steerer, prompt, seed_response)
    seq_len = input_ids.shape[1]
    filtered = {n: a for n, a in circuit.neurons.items() if n.position < seq_len}
    return Circuit(neurons=filtered, prompt=circuit.prompt,
                   target_token=circuit.target_token,
                   total_logit_diff=circuit.total_logit_diff)


def save_edge_field(out_dir: Path, probe: LinearProbe, graph: CircuitGraph,
                    train_summary: dict, heldout_summary: dict,
                    top_k: int, edge_prompts_used: int, edge_top_k: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    graph.circuit.save(str(out_dir / "circuit.json"))

    edges_data = [
        {
            "source": [e.source.layer, e.source.position, e.source.neuron],
            "target": [e.target.layer, e.target.position, e.target.neuron],
            "weight": e.weight,
        }
        for e in graph.edges
    ]
    (out_dir / "edges.json").write_text(
        json.dumps({"n_edges": len(edges_data), "edges": edges_data}, indent=2))

    summary = {
        "probe": {
            "layer": probe.layer,
            "position": probe.position,
            "method": probe.method,
            "bias": probe.bias,
            "hidden_state_alias": "hidden_states[-1] (post-final-norm)"
                                  if probe.layer == 32 else f"hidden_states[{probe.layer}+1]",
        },
        "probe_train_validation": train_summary,
        "probe_heldout_validation": heldout_summary,
        "top_k": top_k,
        "edge_top_k": edge_top_k,
        "n_circuit_neurons": len(graph.circuit.neurons),
        "n_edges": len(edges_data),
        "edge_prompts_used": edge_prompts_used,
        "sign_counts": {
            "positive": sum(1 for e in edges_data if e["weight"] >= 0),
            "negative": sum(1 for e in edges_data if e["weight"] < 0),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {out_dir}/  (circuit={len(graph.circuit.neurons)} neurons, "
          f"{len(edges_data)} edges)")
    return summary


def run_probe_field(
    steerer: NeuronSteerer,
    probe_layer: int,
    top_k: int,
    top_k_per_layer: int,
    edge_top_k: int,
    n_edge_prompts: int,
    out_dir: Path,
) -> dict:
    print(f"\n{'='*72}\nPROBE EDGE FIELD: L{probe_layer}\n{'='*72}")

    # 1) Fit the probe (same mean-diff logic + L32 alias as probe_role_table).
    probe, train_summary = fit_mean_diff_probe(
        steerer, layer=probe_layer,
        positive_prompts=REFUSAL_DISCOVERY_POSITIVE,
        negative_prompts=REFUSAL_DISCOVERY_NEGATIVE,
    )
    print(f"Probe L{probe_layer}: train margin={train_summary['margin']:+.4f} "
          f"auc={train_summary['auc']:.3f}")

    # Heldout validation (mirror probe_role_table): refusal-test vs benign.
    test_pos = [measure_probe_score(steerer, p, probe) for p in REFUSAL_TEST]
    test_neg = [measure_probe_score(steerer, p, probe)
                for p in BENIGN_PROMPTS[:len(REFUSAL_TEST)]]
    heldout = score_summary(test_pos, test_neg)
    print(f"Probe L{probe_layer}: heldout margin={heldout['margin']:+.4f} "
          f"auc={heldout['auc']:.3f}")
    if heldout["auc"] < 0.9:
        print(f"  WARNING: heldout AUC {heldout['auc']:.3f} < 0.9 — probe readout "
              f"may be weak; downstream attributions suspect.")

    # 2) Probe-readout attribution over the refusal discovery prompts.
    print(f"\n--- Probe-readout RelP attribution (L{probe_layer}) ---")
    mean_attrs = attribute_probe_over_prompts(
        steerer, probe, REFUSAL_DISCOVERY_POSITIVE,
        top_k_per_layer=top_k_per_layer,
    )
    print(f"Aggregated {len(mean_attrs)} unique (layer,position,neuron) attributions")

    # 3) Fixed same-size top-k selection (so L24 and L32 fields are comparable).
    selected = select_circuit(mean_attrs, method="topk", top_k=top_k)
    circuit = Circuit(
        neurons=selected,
        prompt=f"[probe_L{probe_layer}]",
        target_token=f"probe_L{probe_layer}",
        total_logit_diff=0.0,
    )
    n_unique = len({(n.layer, n.neuron) for n in selected})
    print(f"Selected top-{top_k}: {len(selected)} position rows, {n_unique} unique neurons")

    # 4) Edge discovery on the probe-selected circuit, per prompt, then merged.
    edge_prompts = REFUSAL_DISCOVERY_POSITIVE
    actual_n = n_edge_prompts if n_edge_prompts > 0 else len(edge_prompts)
    tgt_desc = "all" if edge_top_k == 0 else f"top {edge_top_k}"
    print(f"\n--- Edge discovery ({actual_n} prompts, {tgt_desc} targets) ---")
    all_edges: list[CircuitEdge] = []
    for i, prompt in enumerate(edge_prompts[:actual_n]):
        filtered = filter_circuit_for_prompt(steerer, circuit, prompt)
        # Match circuit_topology.py: edge_top_k=0 means all circuit neurons as
        # targets; otherwise cap at edge_top_k (top by attribution, since
        # discover_edges sorts targets internally). (Codex review.)
        n_targets = (len(filtered.neurons) if edge_top_k == 0
                     else min(edge_top_k, len(filtered.neurons)))
        graph = steerer.discover_edges(
            prompt=prompt,
            circuit=filtered,
            top_k_targets=n_targets,
            verbose=False,
        )
        all_edges.extend(graph.edges)
        print(f"    [{i+1}/{actual_n}] {len(graph.edges)} edges  ({prompt[:40]})")

    merged = merge_edges(all_edges)
    full_graph = CircuitGraph(circuit=circuit, edges=merged)
    print(f"Merged: {len(merged)} unique edges across {actual_n} prompts")

    return save_edge_field(out_dir, probe, full_graph, train_summary, heldout,
                           top_k, actual_n, edge_top_k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=list(MODELS))
    ap.add_argument("--probe-layers", default="24,32",
                    help="Comma-separated probe layers (32 = post-final-norm alias)")
    ap.add_argument("--top-k", type=int, default=91,
                    help="Fixed circuit size per probe (same for all, for comparability). "
                         "91 matches the token-I circuit's position-row budget.")
    ap.add_argument("--top-k-per-layer", type=int, default=200,
                    help="Per-layer-per-position sparsification in attribution")
    ap.add_argument("--edge-top-k", type=int, default=30,
                    help="Number of edge target neurons per prompt (0 = all circuit "
                         "neurons). Default 30 matches circuit_topology.py and the "
                         "token-I edges.json; 'all' is exhaustive and not comparable.")
    ap.add_argument("--n-edge-prompts", type=int, default=0,
                    help="Number of discovery prompts for edge passes (0 = all)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    layers = [int(x) for x in args.probe_layers.split(",") if x.strip()]
    model_name = MODELS[args.model]
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(f"apparatus/output/probe_edge_fields_{args.model}_{ts}")

    print(f"Loading {model_name}...")
    steerer = NeuronSteerer(model_name)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    field_summaries = {}
    for layer in layers:
        field_summaries[f"L{layer}"] = run_probe_field(
            steerer=steerer,
            probe_layer=layer,
            top_k=args.top_k,
            top_k_per_layer=args.top_k_per_layer,
            edge_top_k=args.edge_top_k,
            n_edge_prompts=args.n_edge_prompts,
            out_dir=args.out_dir / f"L{layer}",
        )

    run_summary = {
        "model": model_name,
        "probe_layers": layers,
        "top_k": args.top_k,
        "top_k_per_layer": args.top_k_per_layer,
        "edge_top_k": args.edge_top_k,
        "discovery_prompts": "REFUSAL_DISCOVERY_POSITIVE",
        "note": ("Readout-selected edge-field differential inputs. Differential = "
                 "flow(L32) - flow(L24) is computed downstream by the differential "
                 "viewer (union edge keys, missing = 0)."),
        "fields": field_summaries,
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=str))
    print(f"\nAll probe edge fields saved to {args.out_dir}")
    print(f"Next: differential join/viewer over L24/edges.json and L32/edges.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
