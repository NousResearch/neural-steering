#!/bin/bash
# Run the full experimental suite on Llama-3.1-8B-Instruct
#
# This runs three experiments in sequence:
#   1. top_k sweep (the key Arora comparison)
#   2. cross-method control (contrastive on factual tasks)
#   3. full evaluation protocol (necessity, sufficiency, mediation)
#
# Total runtime estimate: ~30-60 min on A100 40GB

set -e
cd "$(dirname "$0")"

MODEL=llama8b
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiments/results_8b_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "=== Neural Steering 8B Experiments ==="
echo "Results will be saved to $RESULTS_DIR"
echo ""

# 1. Top-k sweep — the critical Arora comparison
echo "=== [1/3] Top-k Sweep ==="
python3 experiments/topk_sweep.py --model $MODEL --task both 2>&1 | tee "$RESULTS_DIR/topk_sweep.txt"

echo ""
echo "=== [2/3] Cross-Method Control ==="
python3 experiments/cross_method_control.py --model $MODEL 2>&1 | tee "$RESULTS_DIR/cross_method_control.txt"

echo ""
echo "=== [3/3] Full Evaluation Protocol ==="
python3 experiments/circuit_eval_protocol.py --model $MODEL --task both --n_random 5 2>&1 | tee "$RESULTS_DIR/eval_protocol.txt"

# Copy JSON results
cp -f experiments/eval_results.json "$RESULTS_DIR/" 2>/dev/null || true

echo ""
echo "=== All experiments complete ==="
echo "Results saved to $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
