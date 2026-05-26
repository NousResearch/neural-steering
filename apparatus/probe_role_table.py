"""Apparatus 2a: role decomposition under hidden-state probe readouts.

This holds the candidate neuron set fixed (usually the full 91-row RelP refusal
circuit collapsed to unique (layer, neuron) pairs) and swaps the scalar readout:
instead of measuring P("I")/logit-margin, we measure a linear hidden-state probe
at one or more layers.

The point is readout consistency, not rediscovery. If the tokenwise role-space
geometry survives under probe readouts, the role structure is less likely to be a
token-readout artifact. If it changes, the role claims must be readout-relative.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuron_steer.core import NeuronSteerer, NeuronIdx, steer_neurons
from experiments.prompts import (
    REFUSAL_DISCOVERY_POSITIVE,
    REFUSAL_DISCOVERY_NEGATIVE,
    REFUSAL_TEST,
    BENIGN_PROMPTS,
)
from experiments.surgical_ablation import (
    load_circuit,
    unique_circuit_neurons,
    make_single_neuron_ctx,
)
from experiments.sufficiency_test import (
    collect_mean_neuron_activation,
    make_transplant_single_ctx,
)


@dataclass
class LinearProbe:
    layer: int
    position: int
    direction: torch.Tensor
    bias: float
    method: str = "mean_diff"

    def score_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        direction = self.direction.to(hidden.device, dtype=hidden.dtype)
        bias = torch.tensor(self.bias, device=hidden.device, dtype=hidden.dtype)
        return torch.dot(hidden.float(), direction.float()) + bias.float()


def parse_layers(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def format_prompt(steerer: NeuronSteerer, prompt: str, seed_response: str = "") -> torch.Tensor:
    formatted = steerer._format_prompt(prompt, seed_response)
    return steerer.tokenizer(formatted, return_tensors="pt").input_ids.to(steerer.device)


def collect_hidden_states(
    steerer: NeuronSteerer,
    prompts: list[str],
    layer: int,
    position: int = -1,
    seed_response: str = "",
) -> torch.Tensor:
    """Collect post-layer residual hidden states: hidden_states[layer + 1]."""
    states = []
    for prompt in prompts:
        input_ids = format_prompt(steerer, prompt, seed_response)
        with torch.no_grad():
            outputs = steerer.model(input_ids, output_hidden_states=True)
        states.append(outputs.hidden_states[layer + 1][0, position].detach().float().cpu())
    return torch.stack(states)


def fit_mean_diff_probe(
    steerer: NeuronSteerer,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    position: int = -1,
    seed_response: str = "",
) -> tuple[LinearProbe, dict]:
    """Fit a difference-of-means residual stream direction."""
    pos = collect_hidden_states(steerer, positive_prompts, layer, position, seed_response)
    neg = collect_hidden_states(steerer, negative_prompts, layer, position, seed_response)

    mu_pos = pos.mean(dim=0)
    mu_neg = neg.mean(dim=0)
    direction = mu_pos - mu_neg
    direction = direction / (direction.norm() + 1e-8)
    bias = -0.5 * torch.dot(direction, mu_pos + mu_neg).item()
    probe = LinearProbe(
        layer=layer,
        position=position,
        direction=direction.cpu(),
        bias=float(bias),
    )

    pos_scores = (pos @ probe.direction + probe.bias).numpy()
    neg_scores = (neg @ probe.direction + probe.bias).numpy()
    summary = score_summary(pos_scores, neg_scores)
    summary.update({
        "layer": layer,
        "position": position,
        "method": probe.method,
        "n_positive": len(positive_prompts),
        "n_negative": len(negative_prompts),
    })
    return probe, summary


def score_summary(pos_scores: Iterable[float], neg_scores: Iterable[float]) -> dict:
    pos = np.array(list(pos_scores), dtype=float)
    neg = np.array(list(neg_scores), dtype=float)
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    # Mann-Whitney interpretation of AUC, no sklearn dependency.
    greater = 0
    total = 0
    for ps in pos:
        for ns in neg:
            greater += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
            total += 1
    auc = greater / total if total else float("nan")
    return {
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std()),
        "neg_mean": float(neg.mean()),
        "neg_std": float(neg.std()),
        "margin": float(pos.mean() - neg.mean()),
        "auc": float(auc),
        "all_mean": float(all_scores.mean()),
        "label_score_corr": float(np.corrcoef(labels, all_scores)[0, 1]) if len(all_scores) > 2 else float("nan"),
    }


def measure_probe_score(
    steerer: NeuronSteerer,
    prompt: str,
    probe: LinearProbe,
    make_ctx: Callable | None = None,
    seed_response: str = "",
) -> float:
    input_ids = format_prompt(steerer, prompt, seed_response)
    ctx = make_ctx() if make_ctx is not None else nullcontext()
    with ctx:
        with torch.no_grad():
            outputs = steerer.model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[probe.layer + 1][0, probe.position]
        return float(probe.score_hidden(hidden).item())


def measure_probe_batch(
    steerer: NeuronSteerer,
    prompts: list[str],
    probe: LinearProbe,
    make_ctx: Callable | None = None,
    seed_response: str = "",
) -> tuple[float, list[float]]:
    scores = [
        measure_probe_score(steerer, p, probe, make_ctx=make_ctx, seed_response=seed_response)
        for p in prompts
    ]
    return float(np.mean(scores)), scores


def run_probe_layer(
    steerer: NeuronSteerer,
    layer: int,
    candidates: list[tuple[int, int]],
    circuit_path: Path,
    n_random: int,
    output_dir: Path,
    position: int = -1,
    seed: int = 42,
    validate_only: bool = False,
) -> dict:
    print(f"\n{'=' * 72}")
    print(f"PROBE ROLE TABLE: layer {layer}")
    print(f"{'=' * 72}")

    probe, train_summary = fit_mean_diff_probe(
        steerer,
        layer=layer,
        positive_prompts=REFUSAL_DISCOVERY_POSITIVE,
        negative_prompts=REFUSAL_DISCOVERY_NEGATIVE,
        position=position,
    )
    print(f"Probe train margin={train_summary['margin']:+.4f} auc={train_summary['auc']:.3f}")

    test_pos_scores = [
        measure_probe_score(steerer, p, probe)
        for p in REFUSAL_TEST
    ]
    test_neg_scores = [
        measure_probe_score(steerer, p, probe)
        for p in BENIGN_PROMPTS[:len(REFUSAL_TEST)]
    ]
    validation = score_summary(test_pos_scores, test_neg_scores)
    print(f"Probe heldout margin={validation['margin']:+.4f} auc={validation['auc']:.3f}")

    target_baseline, target_per = measure_probe_batch(steerer, REFUSAL_TEST, probe)
    control_baseline, control_per = measure_probe_batch(steerer, BENIGN_PROMPTS, probe)
    print(f"Target baseline probe score:  {target_baseline:+.4f}")
    print(f"Control baseline probe score: {control_baseline:+.4f}")

    if validate_only:
        result = {
            "probe": {
                "layer": layer,
                "position": position,
                "method": probe.method,
                "bias": probe.bias,
            },
            "circuit_path": str(circuit_path),
            "n_candidates": len(candidates),
            "train_validation": train_summary,
            "heldout_validation": validation,
            "target_baseline": target_baseline,
            "target_scores": target_per,
            "control_baseline": control_baseline,
            "control_scores": control_per,
            "validate_only": True,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / f"probe_validation_layer{layer}.json"
        out_json.write_text(json.dumps(result, indent=2, default=str))
        print(f"Validate-only mode; wrote {out_json}")
        return result

    # Necessity: ablate candidate neuron, measure score drop on refusal prompts.
    print(f"\n--- Single-neuron ablation on probe score ({len(candidates)} candidates) ---")
    single_ablation = []
    for i, (l, n) in enumerate(candidates):
        ctx = make_single_neuron_ctx(steerer.model, l, n)
        abl_mean, _ = measure_probe_batch(steerer, REFUSAL_TEST, probe, make_ctx=ctx)
        d_score = target_baseline - abl_mean
        single_ablation.append({
            "layer": l,
            "neuron": n,
            "abl_probe_score": abl_mean,
            "dProbe": d_score,
            "upstream_of_probe": l <= layer,
        })
        if i < 10 or (i + 1) % 10 == 0:
            print(f"  [{i+1:2d}] L{l:02d}/N{n:5d}  score={abl_mean:+.4f}  dProbe={d_score:+.4f}")

    rng = random.Random(seed)
    d_mlp = steerer.model.config.intermediate_size
    candidate_layers = sorted(set(l for l, _ in candidates))
    candidate_set = set(candidates)
    random_ablation = []
    for _ in range(n_random):
        rand_layer = rng.choice(candidate_layers)
        rand_neuron = rng.randint(0, d_mlp - 1)
        if (rand_layer, rand_neuron) in candidate_set:
            rand_neuron = (rand_neuron + 1) % d_mlp
        ctx = make_single_neuron_ctx(steerer.model, rand_layer, rand_neuron)
        abl_mean, _ = measure_probe_batch(steerer, REFUSAL_TEST, probe, make_ctx=ctx)
        random_ablation.append(target_baseline - abl_mean)
    abl_mean = float(np.mean(random_ablation))
    abl_std = float(np.std(random_ablation))
    for row in single_ablation:
        row["sigma_above_random"] = (
            (row["dProbe"] - abl_mean) / abl_std if abl_std > 1e-10 else 0.0
        )

    # Sufficiency: transplant refusal activation into benign prompts, measure score rise.
    print(f"\n--- Single-neuron transplant on probe score ({len(candidates)} candidates) ---")
    source_acts = {}
    for i, (l, n) in enumerate(candidates):
        source_acts[(l, n)] = collect_mean_neuron_activation(
            steerer, l, n, REFUSAL_DISCOVERY_POSITIVE,
        )

    single_transplant = []
    for i, (l, n) in enumerate(candidates):
        ctx = make_transplant_single_ctx(steerer.model, l, n, source_acts[(l, n)])
        tx_mean, _ = measure_probe_batch(steerer, BENIGN_PROMPTS, probe, make_ctx=ctx)
        d_score = tx_mean - control_baseline
        single_transplant.append({
            "layer": l,
            "neuron": n,
            "source_activation": source_acts[(l, n)],
            "tx_probe_score": tx_mean,
            "dProbeSufficiency": d_score,
            "upstream_of_probe": l <= layer,
        })
        if i < 10 or (i + 1) % 10 == 0:
            print(f"  [{i+1:2d}] L{l:02d}/N{n:5d}  score={tx_mean:+.4f}  dProbeS={d_score:+.4f}")

    random_transplant = []
    rng.seed(seed)
    for _ in range(n_random):
        rand_layer = rng.choice(candidate_layers)
        rand_neuron = rng.randint(0, d_mlp - 1)
        rand_src = collect_mean_neuron_activation(
            steerer, rand_layer, rand_neuron, REFUSAL_DISCOVERY_POSITIVE[:3],
        )
        ctx = make_transplant_single_ctx(steerer.model, rand_layer, rand_neuron, rand_src)
        tx_mean, _ = measure_probe_batch(steerer, BENIGN_PROMPTS, probe, make_ctx=ctx)
        random_transplant.append(tx_mean - control_baseline)
    tx_mean = float(np.mean(random_transplant))
    tx_std = float(np.std(random_transplant))
    for row in single_transplant:
        row["sigma_above_random"] = (
            (row["dProbeSufficiency"] - tx_mean) / tx_std if tx_std > 1e-10 else 0.0
        )

    by_key_ablation = {(r["layer"], r["neuron"]): r for r in single_ablation}
    by_key_transplant = {(r["layer"], r["neuron"]): r for r in single_transplant}
    rows = []
    for l, n in candidates:
        a = by_key_ablation[(l, n)]
        t = by_key_transplant[(l, n)]
        rows.append({
            "layer": l,
            "neuron": n,
            "probe_layer": layer,
            "upstream_of_probe": l <= layer,
            "necessity_dProbe": a["dProbe"],
            "necessity_sigma": a["sigma_above_random"],
            "sufficiency_dProbe": t["dProbeSufficiency"],
            "sufficiency_sigma": t["sigma_above_random"],
            "source_activation": t["source_activation"],
        })

    result = {
        "probe": {
            "layer": layer,
            "position": position,
            "method": probe.method,
            "bias": probe.bias,
        },
        "circuit_path": str(circuit_path),
        "n_candidates": len(candidates),
        "candidates": [{"layer": l, "neuron": n} for l, n in candidates],
        "train_validation": train_summary,
        "heldout_validation": validation,
        "target_baseline": target_baseline,
        "target_scores": target_per,
        "control_baseline": control_baseline,
        "control_scores": control_per,
        "single_neuron_ablation": single_ablation,
        "single_neuron_transplant": single_transplant,
        "random_ablation": {
            "n": n_random,
            "mean": abl_mean,
            "std": abl_std,
            "values": random_ablation,
        },
        "random_transplant": {
            "n": n_random,
            "mean": tx_mean,
            "std": tx_std,
            "values": random_transplant,
        },
        "role_rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"probe_roles_layer{layer}.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))
    out_jsonl = output_dir / f"probe_role_rows_layer{layer}.jsonl"
    with out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_jsonl}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Run hidden-state probe role decomposition for fixed circuit candidates")
    ap.add_argument("--model", default="llama8b", choices=["llama8b"])
    ap.add_argument("--circuit", required=True, type=Path,
                    help="Path to position-aware circuit.json")
    ap.add_argument("--layers", default="18,24,28",
                    help="Comma-separated probe layers")
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--validate_only", action="store_true",
                    help="Fit and validate probes, then stop before intervention sweeps")
    args = ap.parse_args()

    model_name = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}[args.model]
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"apparatus/output/probe_roles_{args.model}_{timestamp}")

    circuit = load_circuit(str(args.circuit))
    candidates = unique_circuit_neurons(circuit)
    print(f"Loaded circuit: {len(circuit.neurons)} position rows, {len(candidates)} unique candidates")
    print(f"Probe layers: {parse_layers(args.layers)}")

    steerer = NeuronSteerer(model_name)

    all_results = {}
    for layer in parse_layers(args.layers):
        all_results[str(layer)] = run_probe_layer(
            steerer=steerer,
            layer=layer,
            candidates=candidates,
            circuit_path=args.circuit,
            n_random=args.n_random,
            output_dir=args.output_dir,
            validate_only=args.validate_only,
        )

    summary = {
        "model": model_name,
        "circuit": str(args.circuit),
        "layers": parse_layers(args.layers),
        "n_candidates": len(candidates),
        "layer_summaries": {
            layer: {
                "train_validation": result["train_validation"],
                "heldout_validation": result["heldout_validation"],
                "target_baseline": result["target_baseline"],
                "control_baseline": result["control_baseline"],
                "validate_only": result.get("validate_only", False),
            }
            for layer, result in all_results.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nAll probe-role outputs saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
