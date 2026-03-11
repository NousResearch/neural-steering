#!/bin/bash
# Deploy and run Arora et al. matched comparison on RunPod
set -e

POD="root@69.30.85.101"
PORT=22118
SSH_KEY="/home/jake/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $POD"
SCP="$SCP -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
REMOTE_DIR="/root/neural-steering"

echo "=== Step 1: Sync repo ==="
$SSH "which rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)" 2>/dev/null
RSYNC_SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
rsync -avz --no-owner --no-group \
    -e "$RSYNC_SSH" \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'output/' --exclude 'experiments/results/' \
    /home/jake/github/neural-steering/ $POD:$REMOTE_DIR/

# Also sync core.py to the right place
rsync -avz --no-owner --no-group \
    -e "$RSYNC_SSH" \
    /home/jake/github/neural-steering/neuron_steer/ $POD:$REMOTE_DIR/neuron_steer/

echo "=== Step 2: Install deps ==="
$SSH "cd $REMOTE_DIR && pip install -q -e . 2>&1 | tail -3"

echo "=== Step 3: Find topology directories ==="
# We have two topology bases: the original and the forced-choice
# Run on original (behavioral + factual)
TOPO_V1=$($SSH "ls -d $REMOTE_DIR/experiments/topology_llama8b_20260305_163508 2>/dev/null || echo ''")
TOPO_FC=$($SSH "ls -d $REMOTE_DIR/experiments/fc_results/topology 2>/dev/null || echo ''")

echo "  Original topology: $TOPO_V1"
echo "  FC topology: $TOPO_FC"

echo "=== Step 4: Run Arora comparison ==="
# Run on behavioral + factual (original circuits)
if [ -n "$TOPO_V1" ]; then
    echo "--- Running on behavioral + factual ---"
    $SSH "cd $REMOTE_DIR && python experiments/arora_comparison.py \
        --topology_base experiments/topology_llama8b_20260305_163508 \
        --task all \
        --n_random 20 \
        --output_dir experiments/arora_comparison_original" 2>&1 | tee /tmp/arora_comparison.log
fi

# Run on fc_refusal (forced-choice circuit)
if [ -n "$TOPO_FC" ]; then
    echo "--- Running on fc_refusal ---"
    $SSH "cd $REMOTE_DIR && python experiments/arora_comparison.py \
        --topology_base experiments/fc_results/topology \
        --task fc_refusal \
        --n_random 20 \
        --output_dir experiments/arora_comparison_fc" 2>&1 | tee -a /tmp/arora_comparison.log
fi

echo "=== Step 5: Pull results ==="
mkdir -p /home/jake/github/neural-steering/experiments/arora_comparison_original
mkdir -p /home/jake/github/neural-steering/experiments/arora_comparison_fc
$SCP "$POD:$REMOTE_DIR/experiments/arora_comparison_original/*" \
    /home/jake/github/neural-steering/experiments/arora_comparison_original/ 2>/dev/null || true
$SCP "$POD:$REMOTE_DIR/experiments/arora_comparison_fc/*" \
    /home/jake/github/neural-steering/experiments/arora_comparison_fc/ 2>/dev/null || true

echo "=== Done ==="
echo "Results in experiments/arora_comparison_original/ and experiments/arora_comparison_fc/"
