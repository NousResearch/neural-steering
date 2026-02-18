#!/bin/bash
#SBATCH --job-name=sentiment-steer
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sentiment_steering_%j.out
#SBATCH --error=logs/sentiment_steering_%j.err

set -euo pipefail
source ~/cc/env/bin/activate
cd ~/cc/neuron-steering
mkdir -p results logs

echo "=== Sentiment Steering Experiment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

python experiments/sentiment_steering.py \
    --model "${1:-meta-llama/Llama-3.1-8B-Instruct}" \
    --top-k "${2:-200}" \
    --output-dir results/sentiment_steering

echo "=== Done: $(date) ==="
