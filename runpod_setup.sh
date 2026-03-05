#!/bin/bash
# RunPod setup for neural-steering experiments on Llama-3.1-8B-Instruct
#
# Requirements: A100 40GB+ (80GB preferred for comfortable RelP headroom)
# Template: RunPod PyTorch 2.x / CUDA 12.x
#
# STEP 1 (from your local machine):
#   rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
#     ~/github/neural-steering/ runpod:/workspace/neural-steering/
#
# STEP 2 (on RunPod):
#   cd /workspace/neural-steering && bash runpod_setup.sh

set -e

echo "=== Neural Steering — RunPod Setup ==="

# Install dependencies
pip install -q torch transformers accelerate

# HuggingFace login for gated Llama access
if ! huggingface-cli whoami &>/dev/null; then
    echo ""
    echo "=== HuggingFace Login Required ==="
    echo "Llama 3.1 8B is a gated model. You need a HF token with access."
    echo "Get one at: https://huggingface.co/settings/tokens"
    echo "Accept the license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    huggingface-cli login
fi

# Verify GPU
echo ""
echo "=== GPU Info ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Pre-download model
echo ""
echo "=== Downloading Llama-3.1-8B-Instruct ==="
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', torch_dtype='auto')
print('Done!')
"

echo ""
echo "=== Setup Complete ==="
echo "Run experiments with: bash run_8b_experiments.sh"
