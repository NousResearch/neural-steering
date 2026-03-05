"""Circuit topology analysis: find minimal circuit size, then analyze structure.

Phase 1: Discover RelP circuits and find k* (minimal sufficient circuit size)
         via dense scan of N_H across all prefix sizes k=1..Kmax.
Phase 2: Run edge attribution and topology analysis at k* and nearby values.

Usage:
    python experiments/circuit_topology.py --model llama8b --task all
    python experiments/circuit_topology.py --model llama8b --task behavioral --kmax 300
"""

import argparse
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuron_steer.core import (
    NeuronIdx, Circuit, CircuitGraph, CircuitEdge,
    NeuronSteerer, steer_neurons, select_circuit,
)
from experiments.prompts import (
    CAPITALS_DISCOVERY, CAPITALS_TEST,
    REFUSAL_DISCOVERY_POSITIVE, REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST, BENIGN_PROMPTS,
    SVA_PROMPTS,
    SYCOPHANCY_DISCOVERY_POSITIVE, SYCOPHANCY_DISCOVERY_NEGATIVE,
    SYCOPHANCY_TEST,
    FC_REFUSAL_DISCOVERY_POSITIVE, FC_REFUSAL_TEST,
    FC_BELIEF_DISCOVERY, FC_BELIEF_TEST,
)


# ============================================================
# Phase 1: Find k* (minimal sufficient circuit size)
# ============================================================

def measure_R(steerer, prompt, target_token, circuit=None, multiplier=0.0,
              seed_response="", use_chat_template=True):
    """Measure R(x) = P(target_token) with optional circuit ablation."""
    if use_chat_template:
        formatted = steerer._format_prompt(prompt, seed_response)
    else:
        formatted = prompt + seed_response
    input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)
    target_id = steerer.tokenizer.encode(target_token, add_special_tokens=False)[-1]

    ctx = steer_neurons(steerer.model, circuit.neurons, multiplier) if circuit else nullcontext()
    with ctx:
        with torch.no_grad():
            outputs = steerer.model(input_ids)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            return probs[target_id].item()


def measure_N_H(steerer, circuit, target_prompts, target_tokens,
                seed_response="", use_chat_template=True):
    """Measure necessity: mean R_baseline - R_ablated across target prompts.

    target_tokens can be a single string (used for all prompts) or a list
    of per-prompt target tokens.
    """
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens] * len(target_prompts)
    total_baseline = 0.0
    total_ablated = 0.0
    for prompt, target_token in zip(target_prompts, target_tokens):
        total_baseline += measure_R(steerer, prompt, target_token,
                                    seed_response=seed_response,
                                    use_chat_template=use_chat_template)
        total_ablated += measure_R(steerer, prompt, target_token,
                                   circuit=circuit, multiplier=0.0,
                                   seed_response=seed_response,
                                   use_chat_template=use_chat_template)
    n = len(target_prompts)
    return total_baseline / n - total_ablated / n


def find_k_star(steerer, ranked_neurons, target_prompts, target_tokens,
                kmax, seed_response="", use_chat_template=True,
                tau=0.95, eps=0.01, scan_step=5):
    """Find k*: minimal prefix size where N_H plateaus.

    Strategy: dense scan from k=scan_step to kmax in steps of scan_step.
    Then refine around the elbow with step=1.

    Args:
        ranked_neurons: list of (NeuronIdx, attribution) sorted by |attribution| descending
        target_tokens: single string or list of per-prompt target tokens
        tau: fraction of max N_H to require (e.g. 0.95 = 95% of peak necessity)
        eps: negligible gain threshold for plateau detection
        scan_step: step size for coarse scan

    Returns:
        (k_star, curve) where curve is list of (k, N_H) measurements
    """
    dummy_target = target_tokens if isinstance(target_tokens, str) else target_tokens[0]

    # First measure N_H at max k to get the ceiling
    circuit_max = Circuit(
        neurons=dict(ranked_neurons[:kmax]),
        prompt="[k_star_search]", target_token=dummy_target, total_logit_diff=0.0)
    nh_max = measure_N_H(steerer, circuit_max, target_prompts, target_tokens,
                         seed_response=seed_response, use_chat_template=use_chat_template)
    print(f"  N_H ceiling (k={kmax}): {nh_max:.4f}")

    threshold = tau * nh_max
    print(f"  Target: N_H >= {threshold:.4f} (tau={tau})")

    # Coarse scan
    curve = []
    k_candidates = list(range(scan_step, kmax + 1, scan_step))
    if kmax not in k_candidates:
        k_candidates.append(kmax)

    print(f"  Coarse scan: {len(k_candidates)} values from k={scan_step} to k={kmax}")
    for k in k_candidates:
        circuit_k = Circuit(
            neurons=dict(ranked_neurons[:k]),
            prompt="[k_star_search]", target_token=dummy_target, total_logit_diff=0.0)
        nh = measure_N_H(steerer, circuit_k, target_prompts, target_tokens,
                         seed_response=seed_response, use_chat_template=use_chat_template)
        curve.append((k, nh))
        bar = "█" * int(nh / max(nh_max, 0.001) * 30)
        print(f"    k={k:4d}: N_H={nh:.4f} {bar}")

    # Find first k that crosses threshold
    first_cross = None
    for k, nh in curve:
        if nh >= threshold:
            first_cross = k
            break

    if first_cross is None:
        print(f"  WARNING: N_H never reached threshold. Using kmax={kmax}")
        return kmax, curve

    # Fine scan: go back one step and scan every k
    fine_start = max(scan_step, first_cross - scan_step)
    fine_end = min(kmax, first_cross + scan_step)
    print(f"\n  Fine scan: k={fine_start}..{fine_end}")

    fine_curve = []
    for k in range(fine_start, fine_end + 1):
        # Skip if already measured
        existing = [nh for kk, nh in curve if kk == k]
        if existing:
            fine_curve.append((k, existing[0]))
            continue
        circuit_k = Circuit(
            neurons=dict(ranked_neurons[:k]),
            prompt="[k_star_search]", target_token=dummy_target, total_logit_diff=0.0)
        nh = measure_N_H(steerer, circuit_k, target_prompts, target_tokens,
                         seed_response=seed_response, use_chat_template=use_chat_template)
        fine_curve.append((k, nh))
        curve.append((k, nh))

    curve.sort(key=lambda x: x[0])
    fine_curve.sort(key=lambda x: x[0])

    # k* = smallest k in fine scan meeting threshold
    k_star = kmax
    for k, nh in fine_curve:
        if nh >= threshold:
            k_star = k
            break

    # Also check plateau: is there a k where adding more neurons gives < eps gain?
    k_plateau = None
    for i in range(1, len(curve)):
        k_prev, nh_prev = curve[i-1]
        k_curr, nh_curr = curve[i]
        if nh_prev >= threshold and (nh_curr - nh_prev) <= eps:
            k_plateau = k_prev
            break

    if k_plateau and k_plateau < k_star:
        k_star = k_plateau

    print(f"\n  k* = {k_star} (N_H = {dict(curve).get(k_star, '?'):.4f})")
    return k_star, curve


# ============================================================
# Phase 2: Topology analysis
# ============================================================

def save_graph(graph, circuit_path, edges_path, analysis_path):
    """Save circuit, edges, and topology analysis to JSON files."""
    graph.circuit.save(circuit_path)

    edges_data = [
        {
            "source": [e.source.layer, e.source.position, e.source.neuron],
            "target": [e.target.layer, e.target.position, e.target.neuron],
            "weight": e.weight,
        }
        for e in graph.edges
    ]
    with open(edges_path, "w") as f:
        json.dump({"n_edges": len(edges_data), "edges": edges_data}, f, indent=2)

    analysis = run_topology_analysis(graph)
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)


def _position_aware_hubs(edges):
    """Compute hub analysis keyed by full (layer, position, neuron) identity.

    This avoids inflating degree counts when the same neuron appears at
    multiple token positions from multi-prompt discovery.
    """
    source_deg = defaultdict(list)
    target_deg = defaultdict(list)
    for e in edges:
        src_key = (e.source.layer, e.source.position, e.source.neuron)
        tgt_key = (e.target.layer, e.target.position, e.target.neuron)
        source_deg[src_key].append(e.weight)
        target_deg[tgt_key].append(e.weight)

    def rank(deg_map):
        ranked = []
        for key, weights in deg_map.items():
            ranked.append((key, len(weights), sum(abs(w) for w in weights)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    return {"source_hubs": rank(source_deg), "target_hubs": rank(target_deg)}


def _position_aware_bottlenecks(pa_hubs, top_frac=0.2):
    """Bottleneck analysis using position-aware hub data."""
    source_map = {k: deg for k, deg, _ in pa_hubs["source_hubs"]}
    target_map = {k: deg for k, deg, _ in pa_hubs["target_hubs"]}
    all_nodes = set(source_map) | set(target_map)
    if not all_nodes:
        return []

    max_out = max(source_map.values()) if source_map else 1
    max_in = max(target_map.values()) if target_map else 1
    out_thresh = max_out * top_frac
    in_thresh = max_in * top_frac

    bottlenecks = []
    for n in all_nodes:
        out_deg = source_map.get(n, 0)
        in_deg = target_map.get(n, 0)
        if out_deg >= out_thresh and in_deg >= in_thresh:
            bottlenecks.append((n, in_deg, out_deg))
    bottlenecks.sort(key=lambda x: min(x[1], x[2]), reverse=True)
    return bottlenecks


def _position_inflation(edges, collapsed_source_hubs):
    """Compute per-neuron position inflation factors."""
    positions = defaultdict(set)
    for e in edges:
        for ep in [e.source, e.target]:
            positions[(ep.layer, ep.neuron)].add(ep.position)

    pa_source = defaultdict(list)
    for e in edges:
        pa_source[(e.source.layer, e.source.position, e.source.neuron)].append(e.weight)

    inflation = []
    for key, c_deg, c_w in collapsed_source_hubs[:10]:
        layer, neuron = key
        n_pos = len(positions.get((layer, neuron), set()))
        max_pa_deg = 0
        for (l, p, n), ws in pa_source.items():
            if l == layer and n == neuron:
                max_pa_deg = max(max_pa_deg, len(ws))
        if max_pa_deg > 0:
            inflation.append({
                "layer": layer, "neuron": neuron,
                "collapsed_degree": c_deg, "n_positions": n_pos,
                "max_single_position_degree": max_pa_deg,
                "inflation": round(c_deg / max_pa_deg, 2),
            })
    return inflation


def run_topology_analysis(graph):
    """Run all topology analyses and return as dict."""
    analysis = {}

    flow = graph.layer_flow()
    analysis["layer_flow"] = {f"{s}->{t}": w for (s, t), w in flow.items()}

    # Collapsed analysis (original behavior, groups by (layer, neuron))
    hubs = graph.hub_analysis()
    analysis["source_hubs_collapsed"] = [
        {"layer": k[0], "neuron": k[1], "degree": d, "total_weight": w}
        for k, d, w in hubs["source_hubs"][:30]
    ]
    analysis["target_hubs_collapsed"] = [
        {"layer": k[0], "neuron": k[1], "degree": d, "total_weight": w}
        for k, d, w in hubs["target_hubs"][:30]
    ]

    bottlenecks = graph.bottleneck()
    analysis["bottlenecks_collapsed"] = [
        {"layer": k[0], "neuron": k[1], "in_degree": ind, "out_degree": outd}
        for k, ind, outd in bottlenecks
    ]

    # Position-aware analysis (full (layer, position, neuron) identity)
    pa_hubs = _position_aware_hubs(graph.edges)
    analysis["source_hubs_position_aware"] = [
        {"layer": k[0], "position": k[1], "neuron": k[2], "degree": d, "total_weight": w}
        for k, d, w in pa_hubs["source_hubs"][:30]
    ]
    analysis["target_hubs_position_aware"] = [
        {"layer": k[0], "position": k[1], "neuron": k[2], "degree": d, "total_weight": w}
        for k, d, w in pa_hubs["target_hubs"][:30]
    ]

    pa_bottlenecks = _position_aware_bottlenecks(pa_hubs)
    analysis["bottlenecks_position_aware"] = [
        {"layer": k[0], "position": k[1], "neuron": k[2], "in_degree": ind, "out_degree": outd}
        for k, ind, outd in pa_bottlenecks
    ]

    # Inflation report
    analysis["position_inflation"] = _position_inflation(
        graph.edges, hubs["source_hubs"])

    super_weights = graph.detect_super_weights()
    analysis["super_weights"] = [
        {"layer": k[0], "neuron": k[1], "mean_weight": mw, "n_targets": nt, "ratio": r}
        for k, mw, nt, r in super_weights
    ]

    return analysis


def print_topology_report(label, graph, circuit):
    """Print a human-readable topology report."""
    print(f"\n{'='*70}")
    print(f"  TOPOLOGY: {label}")
    print(f"  {len(circuit.neurons)} neurons, {len(graph.edges)} edges")
    print(f"{'='*70}")

    by_layer = circuit.by_layer()
    print(f"\n  Layer distribution:")
    for layer in sorted(by_layer.keys()):
        count = len(by_layer[layer])
        bar = "█" * min(count, 40)
        print(f"    L{layer:02d} │{bar} {count}")

    flow = graph.layer_flow()
    print(f"\n  Top layer-to-layer flows:")
    for i, ((s, t), w) in enumerate(flow.items()):
        if i >= 15:
            break
        print(f"    L{s:02d} → L{t:02d}: {w:.4f}")

    # Collapsed hubs (original behavior)
    hubs = graph.hub_analysis()
    print(f"\n  Top source hubs — COLLAPSED (fan-out):")
    for k, deg, w in hubs["source_hubs"][:10]:
        print(f"    L{k[0]:02d}/N{k[1]:5d}: degree={deg:3d}, total_weight={w:.4f}")

    # Position-aware hubs
    pa_hubs = _position_aware_hubs(graph.edges)
    print(f"\n  Top source hubs — POSITION-AWARE (fan-out):")
    for k, deg, w in pa_hubs["source_hubs"][:10]:
        print(f"    L{k[0]:02d}/P{k[1]:03d}/N{k[2]:5d}: degree={deg:3d}, total_weight={w:.4f}")

    print(f"\n  Top target hubs — POSITION-AWARE (fan-in):")
    for k, deg, w in pa_hubs["target_hubs"][:10]:
        print(f"    L{k[0]:02d}/P{k[1]:03d}/N{k[2]:5d}: degree={deg:3d}, total_weight={w:.4f}")

    # Position-aware bottlenecks
    pa_bn = _position_aware_bottlenecks(pa_hubs)
    if pa_bn:
        print(f"\n  Bottleneck neurons — POSITION-AWARE:")
        for k, ind, outd in pa_bn[:10]:
            print(f"    L{k[0]:02d}/P{k[1]:03d}/N{k[2]:5d}: in={ind:3d}, out={outd:3d}")
    else:
        print(f"\n  No bottleneck neurons detected.")

    # Inflation report
    inflation = _position_inflation(graph.edges, hubs["source_hubs"])
    if any(i["inflation"] > 1.1 for i in inflation):
        print(f"\n  Position inflation (top source hubs):")
        for i in inflation[:5]:
            print(f"    L{i['layer']:02d}/N{i['neuron']}: collapsed={i['collapsed_degree']}, "
                  f"positions={i['n_positions']}, max_single={i['max_single_position_degree']}, "
                  f"inflation={i['inflation']:.1f}x")

    super_weights = graph.detect_super_weights()
    if super_weights:
        print(f"\n  Super-weight neurons:")
        for k, mw, nt, r in super_weights[:10]:
            print(f"    L{k[0]:02d}/N{k[1]:5d}: mean_weight={mw:+.4f}, targets={nt}, ratio={r:.1f}x")
    else:
        print(f"\n  No super-weight neurons detected.")


# ============================================================
# Helpers
# ============================================================

def _filter_circuit_for_prompt(steerer, circuit, prompt, seed_response="", use_chat_template=True):
    """Filter circuit to only include neurons with valid positions for this prompt."""
    if use_chat_template:
        formatted = steerer._format_prompt(prompt, seed_response)
    else:
        formatted = prompt + seed_response
    input_ids = steerer.tokenizer(formatted, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]

    filtered = {n: a for n, a in circuit.neurons.items() if n.position < seq_len}
    dropped = len(circuit.neurons) - len(filtered)
    if dropped > 0:
        print(f"    (filtered {dropped} neurons with position >= seq_len={seq_len})")

    return Circuit(
        neurons=filtered, prompt=circuit.prompt,
        target_token=circuit.target_token, total_logit_diff=circuit.total_logit_diff)


def _merge_edges(edges):
    """Merge edges from multiple prompts by summing weights for same (src, tgt) pairs."""
    edge_map = {}
    for e in edges:
        key = (e.source, e.target)
        edge_map[key] = edge_map.get(key, 0.0) + e.weight
    return [CircuitEdge(src, tgt, w) for (src, tgt), w in edge_map.items()]


def run_edges_and_topology(steerer, circuit, edge_prompts, edge_top_k, n_edge_prompts,
                           seed_response="", use_chat_template=True, label=""):
    """Run edge discovery on a circuit and return the CircuitGraph.

    Args:
        edge_top_k: Number of target neurons per prompt. 0 = all circuit neurons.
        n_edge_prompts: Number of prompts to use. 0 = all prompts.
    """
    actual_n = n_edge_prompts if n_edge_prompts > 0 else len(edge_prompts)
    print(f"\n  Running edge discovery ({actual_n} prompts, "
          f"{'all' if edge_top_k == 0 else f'top {edge_top_k}'} targets)...")
    all_edges = []
    for i, prompt_data in enumerate(edge_prompts[:actual_n]):
        prompt = prompt_data if isinstance(prompt_data, str) else prompt_data[0]
        print(f"    Edge pass {i+1}/{actual_n}: {prompt[:50]}...")

        filtered = _filter_circuit_for_prompt(
            steerer, circuit, prompt,
            seed_response=seed_response, use_chat_template=use_chat_template)

        n_targets = len(filtered.neurons) if edge_top_k == 0 else min(edge_top_k, len(filtered.neurons))
        graph = steerer.discover_edges(
            prompt=prompt,
            circuit=filtered,
            top_k_targets=n_targets,
            seed_response=seed_response,
            use_chat_template=use_chat_template,
            verbose=True,
        )
        all_edges.extend(graph.edges)

    merged = _merge_edges(all_edges)
    full_graph = CircuitGraph(circuit=circuit, edges=merged)
    print_topology_report(label, full_graph, circuit)
    return full_graph


# ============================================================
# Per-task pipeline
# ============================================================

def run_task(steerer, task_label, discover_kwargs, target_prompts, target_tokens,
             edge_prompts, kmax, edge_top_k, n_edge_prompts, results_dir,
             seed_response="", use_chat_template=True, tau=0.95, scan_step=5,
             skip_kstar=False, known_kstar=None, no_comparison=False):
    """Full pipeline for one task: discover -> find k* -> topology at k*."""
    tag = task_label.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # Phase 1: Discover large pool
    print(f"\n\n{'#'*70}")
    print(f"  {task_label}")
    print(f"{'#'*70}")

    print(f"\n  Phase 1: Discovering circuit pool (k={kmax})...")
    result = steerer.discover_circuit_multi(
        **{**discover_kwargs, "top_k": kmax, "return_raw_attributions": True})
    circuit_full, raw_attrs, avg_ld = result
    print(f"  Pool: {len(circuit_full.neurons)} neurons, logit_diff={avg_ld:.4f}")

    # Rank neurons by |attribution|
    ranked = sorted(raw_attrs.items(), key=lambda x: abs(x[1]), reverse=True)[:kmax]

    # Phase 1b: Find k*
    if skip_kstar and known_kstar is not None:
        k_star = known_kstar
        nh_curve = []
        print(f"\n  Skipping k* search, using known k*={k_star}")
    else:
        print(f"\n  Phase 1b: Finding k* (tau={tau}, scan_step={scan_step})...")
        k_star, nh_curve = find_k_star(
            steerer, ranked, target_prompts, target_tokens,
            kmax=kmax, seed_response=seed_response, use_chat_template=use_chat_template,
            tau=tau, scan_step=scan_step)

        # Save N_H curve
        curve_data = {"task": task_label, "k_star": k_star, "tau": tau, "kmax": kmax,
                      "curve": [{"k": k, "N_H": nh} for k, nh in nh_curve]}
        with open(results_dir / f"{tag}_nh_curve.json", "w") as f:
            json.dump(curve_data, f, indent=2)

    # Build circuit at k*
    circuit_star = Circuit(
        neurons=dict(ranked[:k_star]),
        prompt=circuit_full.prompt, target_token=circuit_full.target_token,
        total_logit_diff=circuit_full.total_logit_diff)

    print(f"\n  Phase 2: Topology at k*={k_star}")
    by_layer = circuit_star.by_layer()
    for layer in sorted(by_layer.keys()):
        count = len(by_layer[layer])
        bar = "█" * min(count, 40)
        print(f"    L{layer:02d} │{bar} {count}")

    # Phase 2: Edge attribution and topology at k*
    graph_star = run_edges_and_topology(
        steerer, circuit_star, edge_prompts, edge_top_k, n_edge_prompts,
        seed_response=seed_response, use_chat_template=use_chat_template,
        label=f"{task_label} @ k*={k_star}")

    star_dir = results_dir / f"{tag}_kstar{k_star}"
    star_dir.mkdir(exist_ok=True)
    save_graph(graph_star,
               str(star_dir / "circuit.json"),
               str(star_dir / "edges.json"),
               str(star_dir / "analysis.json"))

    # Also run topology at k*/2 and 2*k* for stability comparison
    if no_comparison:
        return k_star, circuit_star, graph_star

    comparison_ks = sorted(set([max(10, k_star // 2), k_star, min(kmax, k_star * 2)]))
    comparison_ks = [k for k in comparison_ks if k != k_star]  # don't redo k*

    for k in comparison_ks:
        print(f"\n  Comparison topology at k={k}")
        circuit_k = Circuit(
            neurons=dict(ranked[:k]),
            prompt=circuit_full.prompt, target_token=circuit_full.target_token,
            total_logit_diff=circuit_full.total_logit_diff)

        graph_k = run_edges_and_topology(
            steerer, circuit_k, edge_prompts, edge_top_k, n_edge_prompts,
            seed_response=seed_response, use_chat_template=use_chat_template,
            label=f"{task_label} @ k={k}")

        k_dir = results_dir / f"{tag}_k{k}"
        k_dir.mkdir(exist_ok=True)
        save_graph(graph_k,
                   str(k_dir / "circuit.json"),
                   str(k_dir / "edges.json"),
                   str(k_dir / "analysis.json"))

    return k_star, circuit_star, graph_star


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Circuit topology analysis")
    parser.add_argument("--model", choices=["llama8b", "llama1b"], default="llama8b")
    parser.add_argument("--kmax", type=int, default=300,
                        help="Maximum circuit size to search (discovery pool size)")
    parser.add_argument("--edge_top_k", type=int, default=0,
                        help="Number of target neurons for edge discovery (0=all)")
    parser.add_argument("--edge_prompts", type=int, default=0,
                        help="Number of prompts to run edge discovery on (0=all)")
    parser.add_argument("--skip_kstar", action="store_true",
                        help="Skip k* search, use known values from previous run")
    parser.add_argument("--known_kstar", type=str, default="",
                        help="Comma-separated k* values: factual,behavioral,sva (e.g. 114,91,259)")
    parser.add_argument("--no_comparison", action="store_true",
                        help="Skip k*/2 and 2*k* comparison runs")
    parser.add_argument("--tau", type=float, default=0.95,
                        help="Fraction of max N_H to require for k*")
    parser.add_argument("--scan_step", type=int, default=5,
                        help="Step size for coarse N_H scan")
    parser.add_argument("--task", choices=["factual", "behavioral", "sva", "sycophancy", "fc_refusal", "fc_belief", "all"], default="all")
    args = parser.parse_args()

    model_name = {
        "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    }[args.model]

    steerer = NeuronSteerer(model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / f"topology_{args.model}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")
    print(f"kmax={args.kmax}, tau={args.tau}, scan_step={args.scan_step}")

    # Parse known k* values if provided
    known_kstars = {}
    if args.known_kstar:
        parts = args.known_kstar.split(",")
        task_order = ["factual", "behavioral", "sva", "sycophancy"]
        for i, v in enumerate(parts):
            if i < len(task_order) and v.strip():
                known_kstars[task_order[i]] = int(v.strip())

    results = {}

    # ============================================================
    # FACTUAL: Capitals
    # ============================================================
    if args.task in ("factual", "all"):
        k_star, circuit, graph = run_task(
            steerer=steerer,
            task_label="RelP-Factual (Capitals)",
            discover_kwargs=dict(
                prompts=[p for p, _ in CAPITALS_DISCOVERY],
                target_tokens=[t for _, t in CAPITALS_DISCOVERY],
                counterfactual_tokens=[None] * len(CAPITALS_DISCOVERY),
                selection_method="topk",
                seed_response="Answer: ",
                use_chat_template=True,
                verbose=True,
            ),
            target_prompts=[p for p, _ in CAPITALS_TEST],
            target_tokens=[t for _, t in CAPITALS_TEST],
            edge_prompts=CAPITALS_DISCOVERY,
            kmax=args.kmax,
            edge_top_k=args.edge_top_k,
            n_edge_prompts=args.edge_prompts,
            results_dir=results_dir,
            seed_response="Answer: ",
            use_chat_template=True,
            tau=args.tau,
            scan_step=args.scan_step,
            skip_kstar=args.skip_kstar,
            known_kstar=known_kstars.get("factual"),
            no_comparison=args.no_comparison,
        )
        results["factual"] = {"k_star": k_star}

    # ============================================================
    # BEHAVIORAL: Refusal
    # ============================================================
    if args.task in ("behavioral", "all"):
        k_star, circuit, graph = run_task(
            steerer=steerer,
            task_label="RelP-Behavioral (Refusal)",
            discover_kwargs=dict(
                prompts=REFUSAL_DISCOVERY_POSITIVE,
                target_tokens=["I"] * len(REFUSAL_DISCOVERY_POSITIVE),
                selection_method="topk",
                use_chat_template=True,
                verbose=True,
            ),
            target_prompts=REFUSAL_TEST,
            target_tokens="I",
            edge_prompts=REFUSAL_DISCOVERY_POSITIVE,
            kmax=args.kmax,
            edge_top_k=args.edge_top_k,
            n_edge_prompts=args.edge_prompts,
            results_dir=results_dir,
            use_chat_template=True,
            tau=args.tau,
            scan_step=args.scan_step,
            skip_kstar=args.skip_kstar,
            known_kstar=known_kstars.get("behavioral"),
            no_comparison=args.no_comparison,
        )
        results["behavioral"] = {"k_star": k_star}

    # ============================================================
    # SVA
    # ============================================================
    if args.task in ("sva", "all"):
        sva_singular = [p for p in SVA_PROMPTS if p[1].strip() == "is"]
        k_star, circuit, graph = run_task(
            steerer=steerer,
            task_label="RelP-SVA",
            discover_kwargs=dict(
                prompts=[p[0] for p in SVA_PROMPTS],
                target_tokens=[p[1] for p in SVA_PROMPTS],
                counterfactual_tokens=[p[2] for p in SVA_PROMPTS],
                selection_method="topk",
                use_chat_template=False,
                verbose=True,
            ),
            target_prompts=[p[0] for p in sva_singular],
            target_tokens=" is",
            edge_prompts=SVA_PROMPTS,
            kmax=args.kmax,
            edge_top_k=args.edge_top_k,
            n_edge_prompts=args.edge_prompts,
            results_dir=results_dir,
            use_chat_template=False,
            tau=args.tau,
            scan_step=args.scan_step,
            skip_kstar=args.skip_kstar,
            known_kstar=known_kstars.get("sva"),
            no_comparison=args.no_comparison,
        )
        results["sva"] = {"k_star": k_star}

    # ============================================================
    # SYCOPHANCY
    # ============================================================
    if args.task in ("sycophancy", "all"):
        k_star, circuit, graph = run_task(
            steerer=steerer,
            task_label="RelP-Sycophancy",
            discover_kwargs=dict(
                prompts=SYCOPHANCY_DISCOVERY_POSITIVE,
                target_tokens=["I"] * len(SYCOPHANCY_DISCOVERY_POSITIVE),
                selection_method="topk",
                use_chat_template=True,
                verbose=True,
            ),
            target_prompts=SYCOPHANCY_TEST,
            target_tokens="I",
            edge_prompts=SYCOPHANCY_DISCOVERY_POSITIVE,
            kmax=args.kmax,
            edge_top_k=args.edge_top_k,
            n_edge_prompts=args.edge_prompts,
            results_dir=results_dir,
            use_chat_template=True,
            tau=args.tau,
            scan_step=args.scan_step,
            skip_kstar=args.skip_kstar,
            known_kstar=known_kstars.get("sycophancy"),
            no_comparison=args.no_comparison,
        )
        results["sycophancy"] = {"k_star": k_star}

    # ============================================================
    # FORCED-CHOICE REFUSAL
    # ============================================================
    if args.task in ("fc_refusal", "all"):
        k_star, circuit, graph = run_task(
            steerer=steerer,
            task_label="RelP-FC_Refusal",
            discover_kwargs=dict(
                prompts=FC_REFUSAL_DISCOVERY_POSITIVE,
                target_tokens=["No"] * len(FC_REFUSAL_DISCOVERY_POSITIVE),
                selection_method="topk",
                use_chat_template=True,
                verbose=True,
            ),
            target_prompts=FC_REFUSAL_TEST,
            target_tokens="No",
            edge_prompts=FC_REFUSAL_DISCOVERY_POSITIVE,
            kmax=args.kmax,
            edge_top_k=args.edge_top_k,
            n_edge_prompts=args.edge_prompts,
            results_dir=results_dir,
            use_chat_template=True,
            tau=args.tau,
            scan_step=args.scan_step,
            skip_kstar=args.skip_kstar,
            known_kstar=known_kstars.get("fc_refusal"),
            no_comparison=args.no_comparison,
        )
        results["fc_refusal"] = {"k_star": k_star}

    # ============================================================
    # FORCED-CHOICE BELIEF
    # ============================================================
    if args.task in ("fc_belief", "all"):
        k_star, circuit, graph = run_task(
            steerer=steerer,
            task_label="RelP-FC_Belief",
            discover_kwargs=dict(
                prompts=FC_BELIEF_DISCOVERY,
                target_tokens=["Yes"] * len(FC_BELIEF_DISCOVERY),
                selection_method="topk",
                use_chat_template=True,
                verbose=True,
            ),
            target_prompts=FC_BELIEF_TEST,
            target_tokens="Yes",
            edge_prompts=FC_BELIEF_DISCOVERY,
            kmax=args.kmax,
            edge_top_k=args.edge_top_k,
            n_edge_prompts=args.edge_prompts,
            results_dir=results_dir,
            use_chat_template=True,
            tau=args.tau,
            scan_step=args.scan_step,
            skip_kstar=args.skip_kstar,
            known_kstar=known_kstars.get("fc_belief"),
            no_comparison=args.no_comparison,
        )
        results["fc_belief"] = {"k_star": k_star}

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for task, r in results.items():
        print(f"  {task}: k* = {r['k_star']}")

    # Cross-circuit universal neuron detection
    if len(results) >= 2:
        from neuron_steer.core import Circuit
        circuits_by_task = {}
        for task in results:
            circ_path = results_dir / f"relp-{task}_kstar{results[task]['k_star']}" / "circuit.json"
            if circ_path.exists():
                circuits_by_task[task] = Circuit.load(str(circ_path))

        if len(circuits_by_task) >= 2:
            universal = Circuit.find_universal_neurons(*circuits_by_task.values())
            if universal:
                print(f"\n  Universal neurons (in all {len(circuits_by_task)} circuits): {len(universal)}")
                for l, n in sorted(universal):
                    print(f"    L{l:02d}/N{n:5d}")

                # Pairwise overlap with and without universal neurons
                tasks = list(circuits_by_task.keys())
                print(f"\n  Pairwise overlap (total / excluding {len(universal)} universal):")
                for i in range(len(tasks)):
                    for j in range(i + 1, len(tasks)):
                        t1, t2 = tasks[i], tasks[j]
                        s1 = circuits_by_task[t1].unique_neuron_set()
                        s2 = circuits_by_task[t2].unique_neuron_set()
                        total = len(s1 & s2)
                        meaningful = len((s1 & s2) - universal)
                        print(f"    {t1} ∩ {t2}: {total} total, {meaningful} meaningful")

                results["universal_neurons"] = [{"layer": l, "neuron": n} for l, n in sorted(universal)]

    print(f"\n  All results saved to: {results_dir}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
