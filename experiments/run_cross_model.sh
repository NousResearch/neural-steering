#!/bin/bash
#SBATCH --job-name=cross_model
#SBATCH --output=logs/cross_model_%j.out
#SBATCH --error=logs/cross_model_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --partition=batch

# Cross-Model Generalization Experiment
# Runs multiplier sweep on Qwen2.5-7B and Mistral-7B to validate Table 3.
# Each model is run sequentially; GPU memory is freed between runs.

set -euo pipefail

echo "=== Cross-Model Generalization Experiment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

source ~/cc/env/bin/activate
cd ~/cc/neuron-steering
mkdir -p results logs

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- Running: $MODEL ---"
    python experiments/multiplier_sweep.py \
        --model "$MODEL" \
        --output-dir results
    echo "--- Done: $MODEL ---"
done

echo ""
echo "=== All cross-model runs complete: $(date) ==="
