#!/bin/bash
# Deploy and run circuit topology analysis on a RunPod instance.
#
# Assumes rsync is already installed on the pod (apt-get install -y rsync).
#
# Usage:
#   bash deploy_topology.sh <host> <port>
#
# Example:
#   bash deploy_topology.sh root@1.2.3.4 12345

set -e

HOST="${1:?Usage: deploy_topology.sh <host> <port>}"
PORT="${2:?Usage: deploy_topology.sh <host> <port>}"

# HF token from env file
HF_TOKEN=$(grep HF_TOKEN /home/jake/claude/greenhouse/refusal-invariance/runpod-env | cut -d= -f2)

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/workspace/neural-steering"
SSH_KEY="/home/jake/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no"
RSYNC="rsync -avz --no-owner --no-group -e 'ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no'"

echo "=== Circuit Topology — RunPod Deploy ==="
echo "Host: $HOST:$PORT"
echo ""

# Step 1: rsync repo
echo "=== [1/4] Syncing repo ==="
eval $RSYNC \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'experiments/results_*' \
    --exclude 'experiments/topology_*' \
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

# Step 3: Run topology analysis
echo ""
echo "=== [3/4] Running topology analysis ==="
$SSH $HOST << 'RUN'
set -e
cd /workspace/neural-steering

echo "--- Circuit Topology v2: known k*, all prompts, all targets, position-aware ---"
python3 experiments/circuit_topology.py \
    --model llama8b \
    --task all \
    --kmax 300 \
    --skip_kstar \
    --known_kstar 114,91,259 \
    --no_comparison \
    --edge_top_k 0 \
    --edge_prompts 0 \
    2>&1 | tee /tmp/topology_output.txt

# Find the results directory (created by the script)
TOPO_DIR=$(ls -td experiments/topology_* | head -1)
echo "$TOPO_DIR" > /tmp/latest_topo_dir.txt
echo ""
echo "=== Topology analysis complete ==="
echo "Results in: $TOPO_DIR"
RUN

# Step 4: Pull results back
echo ""
echo "=== [4/4] Pulling results ==="
REMOTE_TOPO=$($SSH $HOST "cat /tmp/latest_topo_dir.txt")
LOCAL_TOPO="$REPO_DIR/$REMOTE_TOPO"
mkdir -p "$LOCAL_TOPO"
eval $RSYNC "$HOST:$REMOTE_DIR/$REMOTE_TOPO/" "$LOCAL_TOPO/"

# Also grab the raw output log
$SSH $HOST "cat /tmp/topology_output.txt" > "$LOCAL_TOPO/topology_output.txt"

echo ""
echo "=== Done ==="
echo "Results saved to: $LOCAL_TOPO"
ls -la "$LOCAL_TOPO"
