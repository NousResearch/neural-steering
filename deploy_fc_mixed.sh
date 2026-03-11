#!/bin/bash
# Deploy and run forced-choice mixed-target refusal discovery on RunPod
# Same harmful content, half phrased so refusal="No", half so refusal="Yes"
# Token signal cancels; refusal-specific signal survives
set -e

POD="root@69.30.85.101"
PORT=22118
SSH_KEY="/home/jake/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $POD"
SCP="scp -i $SSH_KEY -P $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
RSYNC_SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
REMOTE_DIR="/root/neural-steering"

echo "=== Step 1: Sync repo ==="
rsync -avz --no-owner --no-group \
    -e "$RSYNC_SSH" \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'output/' \
    /home/jake/github/neural-steering/experiments/ $POD:$REMOTE_DIR/experiments/

rsync -avz --no-owner --no-group \
    -e "$RSYNC_SSH" \
    /home/jake/github/neural-steering/neuron_steer/ $POD:$REMOTE_DIR/neuron_steer/

echo "=== Step 2: Run fc_refusal_mixed topology ==="
$SSH "cd $REMOTE_DIR && python experiments/circuit_topology.py \
    --model llama8b \
    --task fc_refusal_mixed \
    --kmax 300 \
    --output_dir experiments/fc_mixed_results" 2>&1 | tee /tmp/fc_mixed.log

echo "=== Step 3: Pull results ==="
mkdir -p /home/jake/github/neural-steering/experiments/fc_mixed_results
$SCP "$POD:$REMOTE_DIR/experiments/fc_mixed_results/*" \
    /home/jake/github/neural-steering/experiments/fc_mixed_results/ 2>/dev/null || true
# Also get subdirectories
$SSH "find $REMOTE_DIR/experiments/fc_mixed_results -type d" 2>/dev/null | while read dir; do
    local_dir="/home/jake/github/neural-steering/${dir#$REMOTE_DIR/}"
    mkdir -p "$local_dir"
    $SCP "$POD:$dir/*" "$local_dir/" 2>/dev/null || true
done

echo "=== Done ==="
