#!/bin/bash
# Deploy and run synergy search + sufficiency test on a RunPod instance.
#
# Usage:
#   bash deploy_synergy_sufficiency.sh <host> <port>
#
# Example:
#   bash deploy_synergy_sufficiency.sh root@1.2.3.4 12345

set -e

HOST="${1:?Usage: deploy_synergy_sufficiency.sh <host> <port>}"
PORT="${2:?Usage: deploy_synergy_sufficiency.sh <host> <port>}"

# HF token from env file
HF_TOKEN=$(grep HF_TOKEN /home/jake/claude/greenhouse/refusal-invariance/runpod-env | cut -d= -f2)

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/workspace/neural-steering"
SSH_KEY="/home/jake/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no"
RSYNC="rsync -avz --no-owner --no-group -e 'ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no'"

echo "=== Synergy + Sufficiency — RunPod Deploy ==="
echo "Host: $HOST:$PORT"
echo ""

# Step 1: rsync repo (include topology + surgical results for reference)
echo "=== [1/5] Syncing repo ==="
# Ensure rsync is installed on pod
$SSH $HOST "which rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y rsync)"

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
echo "Checking GPU..."
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
SETUP

# Step 3: Run synergy search
echo ""
echo "=== [3/5] Running synergy search ==="
$SSH $HOST << 'RUN_SYNERGY'
set -e
cd /workspace/neural-steering

TOPO_DIR=$(ls -td experiments/topology_llama8b_* | head -1)
echo "Using topology: $TOPO_DIR"

python3 experiments/synergy_search.py \
    --model llama8b \
    --task all \
    --topology_dir "$TOPO_DIR" \
    --max_neurons 24 \
    --n_random_pairs 50 \
    2>&1 | tee /tmp/synergy_output.txt

SYNERGY_DIR=$(ls -td experiments/synergy_llama8b_* 2>/dev/null | head -1)
echo "$SYNERGY_DIR" > /tmp/latest_synergy_dir.txt
echo "=== Synergy search complete: $SYNERGY_DIR ==="
RUN_SYNERGY

# Step 4: Run sufficiency test (reuses loaded model if cached)
echo ""
echo "=== [4/5] Running sufficiency test ==="
$SSH $HOST << 'RUN_SUFF'
set -e
cd /workspace/neural-steering

TOPO_DIR=$(ls -td experiments/topology_llama8b_* | head -1)
echo "Using topology: $TOPO_DIR"

python3 experiments/sufficiency_test.py \
    --model llama8b \
    --task behavioral \
    --topology_dir "$TOPO_DIR" \
    --max_neurons 24 \
    --n_random 20 \
    2>&1 | tee /tmp/sufficiency_output.txt

SUFF_DIR=$(ls -td experiments/sufficiency_llama8b_* 2>/dev/null | head -1)
echo "$SUFF_DIR" > /tmp/latest_sufficiency_dir.txt
echo "=== Sufficiency test complete: $SUFF_DIR ==="
RUN_SUFF

# Step 5: Pull results back
echo ""
echo "=== [5/5] Pulling results ==="

REMOTE_SYNERGY=$($SSH $HOST "cat /tmp/latest_synergy_dir.txt")
LOCAL_SYNERGY="$REPO_DIR/$REMOTE_SYNERGY"
mkdir -p "$LOCAL_SYNERGY"
eval $RSYNC "$HOST:$REMOTE_DIR/$REMOTE_SYNERGY/" "$LOCAL_SYNERGY/"
$SSH $HOST "cat /tmp/synergy_output.txt" > "$LOCAL_SYNERGY/synergy_output.txt"

REMOTE_SUFF=$($SSH $HOST "cat /tmp/latest_sufficiency_dir.txt")
LOCAL_SUFF="$REPO_DIR/$REMOTE_SUFF"
mkdir -p "$LOCAL_SUFF"
eval $RSYNC "$HOST:$REMOTE_DIR/$REMOTE_SUFF/" "$LOCAL_SUFF/"
$SSH $HOST "cat /tmp/sufficiency_output.txt" > "$LOCAL_SUFF/sufficiency_output.txt"

echo ""
echo "=== Done ==="
echo "Synergy results: $LOCAL_SYNERGY"
ls -la "$LOCAL_SYNERGY"
echo ""
echo "Sufficiency results: $LOCAL_SUFF"
ls -la "$LOCAL_SUFF"
