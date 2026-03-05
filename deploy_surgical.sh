#!/bin/bash
# Deploy and run surgical single-neuron ablation on a RunPod instance.
#
# Usage:
#   bash deploy_surgical.sh <host> <port>
#
# Example:
#   bash deploy_surgical.sh root@1.2.3.4 12345

set -e

HOST="${1:?Usage: deploy_surgical.sh <host> <port>}"
PORT="${2:?Usage: deploy_surgical.sh <host> <port>}"

# HF token from env file
HF_TOKEN=$(grep HF_TOKEN /home/jake/claude/greenhouse/refusal-invariance/runpod-env | cut -d= -f2)

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/workspace/neural-steering"
SSH_KEY="/home/jake/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no"
RSYNC="rsync -avz --no-owner --no-group -e 'ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no'"

echo "=== Surgical Ablation — RunPod Deploy ==="
echo "Host: $HOST:$PORT"
echo ""

# Step 1: rsync repo (including topology results for bottleneck candidates)
echo "=== [1/4] Syncing repo ==="
eval $RSYNC \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'experiments/results_*' \
    "$REPO_DIR/" "$HOST:$REMOTE_DIR/"

# Step 2: Install deps + HF login
echo ""
echo "=== [2/4] Installing dependencies ==="
$SSH $HOST << SETUP
set -e
pip install -q torch transformers accelerate huggingface_hub
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
echo "Checking GPU..."
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
SETUP

# Step 3: Run surgical ablation
echo ""
echo "=== [3/4] Running surgical ablation ==="
$SSH $HOST << 'RUN'
set -e
cd /workspace/neural-steering

# Find the v2 topology results (position-aware, most recent)
TOPO_DIR=$(ls -td experiments/topology_llama8b_* | head -1)
echo "Using topology results from: $TOPO_DIR"

python3 experiments/surgical_ablation.py \
    --model llama8b \
    --task all \
    --topology_dir "$TOPO_DIR" \
    --n_random 20 \
    2>&1 | tee /tmp/surgical_output.txt

# Find the results directory (must be a directory, not the .py file)
RESULT_DIR=$(ls -td experiments/surgical_llama8b_* 2>/dev/null | head -1)
echo "$RESULT_DIR" > /tmp/latest_surgical_dir.txt
echo ""
echo "=== Surgical ablation complete ==="
echo "Results in: $RESULT_DIR"
RUN

# Step 4: Pull results back
echo ""
echo "=== [4/4] Pulling results ==="
REMOTE_RESULT=$($SSH $HOST "cat /tmp/latest_surgical_dir.txt")
LOCAL_RESULT="$REPO_DIR/$REMOTE_RESULT"
mkdir -p "$LOCAL_RESULT"
eval $RSYNC "$HOST:$REMOTE_DIR/$REMOTE_RESULT/" "$LOCAL_RESULT/"

# Also grab the raw output log
$SSH $HOST "cat /tmp/surgical_output.txt" > "$LOCAL_RESULT/surgical_output.txt"

echo ""
echo "=== Done ==="
echo "Results saved to: $LOCAL_RESULT"
ls -la "$LOCAL_RESULT"
