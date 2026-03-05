#!/bin/bash
# Deploy and run full sycophancy pipeline on RunPod
# Steps: sync → topology (discovery + k* + edges) → ablation → synergy → sufficiency
set -e

POD="root@69.30.85.9"
PORT=22029
SSH_KEY="$HOME/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT $POD"
SCP="scp -i $SSH_KEY -P $PORT"
REMOTE_DIR="/root/neural-steering"

echo "=== Step 1: Sync repo ==="
# Install rsync if needed
$SSH "which rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)" 2>/dev/null
rsync -avz --delete \
    -e "ssh -i $SSH_KEY -p $PORT" \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'output/' --exclude 'experiments/results/' \
    --exclude 'experiments/topology_llama*' \
    --exclude 'experiments/surgical_llama*' \
    --exclude 'experiments/synergy_llama*' \
    --exclude 'experiments/sufficiency_llama*' \
    --exclude 'experiments/sycophancy_results/' \
    /home/jake/github/neural-steering/ $POD:$REMOTE_DIR/

echo "=== Step 2: Install deps ==="
$SSH "cd $REMOTE_DIR && pip install -q -e . 2>&1 | tail -3"

echo "=== Step 3: Run topology (discovery + k* + edges) for sycophancy ==="
$SSH "cd $REMOTE_DIR && python experiments/circuit_topology.py \
    --model llama8b \
    --task sycophancy \
    --kmax 300 \
    --no_comparison \
    2>&1" | tee /tmp/sycophancy_topology.txt

# Extract topology dir from output
TOPO_DIR=$($SSH "ls -dt $REMOTE_DIR/experiments/topology_llama8b_* 2>/dev/null | head -1")
echo "Topology dir: $TOPO_DIR"

echo "=== Step 4: Run surgical ablation ==="
$SSH "cd $REMOTE_DIR && python experiments/surgical_ablation.py \
    --task sycophancy \
    --topology_dir $TOPO_DIR \
    2>&1" | tee /tmp/sycophancy_ablation.txt

echo "=== Step 5: Run synergy search ==="
$SSH "cd $REMOTE_DIR && python experiments/synergy_search.py \
    --task sycophancy \
    --topology_dir $TOPO_DIR \
    2>&1" | tee /tmp/sycophancy_synergy.txt

echo "=== Step 6: Run sufficiency test ==="
$SSH "cd $REMOTE_DIR && python experiments/sufficiency_test.py \
    --task sycophancy \
    --topology_dir $TOPO_DIR \
    2>&1" | tee /tmp/sycophancy_sufficiency.txt

echo "=== Step 7: Pull results ==="
mkdir -p /home/jake/github/neural-steering/experiments/sycophancy_results
rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
    $POD:$REMOTE_DIR/experiments/topology_llama8b_*/relp-sycophancy_kstar*/ \
    /home/jake/github/neural-steering/experiments/sycophancy_results/topology/
rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
    $POD:$REMOTE_DIR/experiments/surgical_llama8b_*/surgical_sycophancy.json \
    /home/jake/github/neural-steering/experiments/sycophancy_results/ 2>/dev/null || true
rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
    $POD:$REMOTE_DIR/experiments/synergy_llama8b_*/synergy_sycophancy.json \
    /home/jake/github/neural-steering/experiments/sycophancy_results/ 2>/dev/null || true
rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
    $POD:$REMOTE_DIR/experiments/sufficiency_llama8b_*/sufficiency_sycophancy.json \
    /home/jake/github/neural-steering/experiments/sycophancy_results/ 2>/dev/null || true

echo "=== Done! Results in experiments/sycophancy_results/ ==="
