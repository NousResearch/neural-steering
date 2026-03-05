#!/bin/bash
# Deploy and run neural-steering experiments on a RunPod instance.
#
# Usage:
#   bash deploy_runpod.sh <host> <port> <hf_token>
#
# Example:
#   bash deploy_runpod.sh root@1.2.3.4 12345 hf_xxxx
#
# What it does:
#   1. rsync the repo to the RunPod instance
#   2. Install dependencies + login to HuggingFace
#   3. Download Llama-3.1-8B-Instruct
#   4. Run all three experiments (topk sweep, cross-method, full eval)
#   5. Pull results back to local machine

set -e

HOST="${1:?Usage: deploy_runpod.sh <host> <port> <hf_token>}"
PORT="${2:?Usage: deploy_runpod.sh <host> <port> <hf_token>}"
HF_TOKEN="${3:?Usage: deploy_runpod.sh <host> <port> <hf_token>}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/workspace/neural-steering"
SSH_KEY="$HOME/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no"
SCP="scp -i $SSH_KEY -P $PORT -o StrictHostKeyChecking=no"
RSYNC="rsync -avz --no-owner --no-group -e 'ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no'"

echo "=== Neural Steering — RunPod Deploy ==="
echo "Host: $HOST:$PORT"
echo "Repo: $REPO_DIR"
echo ""

# Step 1: rsync repo
echo "=== [1/5] Syncing repo to RunPod ==="
eval $RSYNC \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'experiments/results_*' \
    "$REPO_DIR/" "$HOST:$REMOTE_DIR/"

# Step 2: Install deps + HF login
echo ""
echo "=== [2/5] Installing dependencies ==="
$SSH $HOST << SETUP
set -e
pip install -q torch transformers accelerate huggingface_hub
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
SETUP

# Step 3: Check GPU + download model
echo ""
echo "=== [3/5] Checking GPU and downloading model ==="
$SSH $HOST << 'DOWNLOAD'
set -e
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'meta-llama/Llama-3.1-8B-Instruct'
print(f'Downloading {name}...')
AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained(name, torch_dtype='auto')
print('Model downloaded.')
"
DOWNLOAD

# Step 4: Run experiments
echo ""
echo "=== [4/5] Running experiments ==="
$SSH $HOST << EXPERIMENTS
set -e
cd $REMOTE_DIR

TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiments/results_8b_\${TIMESTAMP}"
mkdir -p "\$RESULTS_DIR"

echo "Results dir: \$RESULTS_DIR"

echo ""
echo "--- Top-k Sweep ---"
python3 experiments/topk_sweep.py --model llama8b --task both 2>&1 | tee "\$RESULTS_DIR/topk_sweep.txt"

echo ""
echo "--- Cross-Method Control ---"
python3 experiments/cross_method_control.py --model llama8b 2>&1 | tee "\$RESULTS_DIR/cross_method_control.txt"

echo ""
echo "--- Full Evaluation Protocol ---"
python3 experiments/circuit_eval_protocol.py --model llama8b --task both --n_random 5 2>&1 | tee "\$RESULTS_DIR/eval_protocol.txt"

cp -f experiments/eval_results.json "\$RESULTS_DIR/" 2>/dev/null || true

echo ""
echo "=== Experiments complete ==="
echo "\$RESULTS_DIR" > /tmp/latest_results_dir.txt
EXPERIMENTS

# Step 5: Pull results back
echo ""
echo "=== [5/5] Pulling results back ==="
REMOTE_RESULTS=$($SSH $HOST "cat /tmp/latest_results_dir.txt")
LOCAL_RESULTS="$REPO_DIR/$REMOTE_RESULTS"
mkdir -p "$LOCAL_RESULTS"
eval $RSYNC "$HOST:$REMOTE_DIR/$REMOTE_RESULTS/" "$LOCAL_RESULTS/"

echo ""
echo "=== Done ==="
echo "Results saved to: $LOCAL_RESULTS"
ls -la "$LOCAL_RESULTS"
