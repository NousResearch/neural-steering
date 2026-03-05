#!/bin/bash
# Deploy and run forced-choice refusal + belief pipeline on RunPod
# Runs both tasks sequentially: fc_refusal then fc_belief
# Each: topology → ablation → synergy → sufficiency
set -e

POD="root@69.30.85.9"
PORT=22029
SSH_KEY="/home/jake/.ssh/id_ed25519_simpolism"
SSH="ssh -i $SSH_KEY -p $PORT $POD"
REMOTE_DIR="/root/neural-steering"

echo "=== Step 1: Sync repo ==="
$SSH "which rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)" 2>/dev/null
rsync -avz --delete --no-owner --no-group \
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

# Run each forced-choice task through the full pipeline
for TASK in fc_refusal fc_belief; do
    echo ""
    echo "============================================================"
    echo "  RUNNING: $TASK"
    echo "============================================================"

    echo "=== $TASK: Topology (discovery + k* + edges) ==="
    $SSH "cd $REMOTE_DIR && python experiments/circuit_topology.py \
        --model llama8b \
        --task $TASK \
        --kmax 300 \
        --no_comparison \
        2>&1" | tee /tmp/${TASK}_topology.txt

    TOPO_DIR=$($SSH "ls -dt $REMOTE_DIR/experiments/topology_llama8b_* 2>/dev/null | head -1")
    echo "Topology dir: $TOPO_DIR"

    echo "=== $TASK: Surgical ablation ==="
    $SSH "cd $REMOTE_DIR && python experiments/surgical_ablation.py \
        --task $TASK \
        --topology_dir $TOPO_DIR \
        2>&1" | tee /tmp/${TASK}_ablation.txt

    echo "=== $TASK: Synergy search ==="
    $SSH "cd $REMOTE_DIR && python experiments/synergy_search.py \
        --task $TASK \
        --topology_dir $TOPO_DIR \
        2>&1" | tee /tmp/${TASK}_synergy.txt

    echo "=== $TASK: Sufficiency test ==="
    $SSH "cd $REMOTE_DIR && python experiments/sufficiency_test.py \
        --task $TASK \
        --topology_dir $TOPO_DIR \
        2>&1" | tee /tmp/${TASK}_sufficiency.txt
done

echo "=== Pull results ==="
mkdir -p /home/jake/github/neural-steering/experiments/fc_results
for DIR in topology surgical synergy sufficiency; do
    rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
        $POD:$REMOTE_DIR/experiments/${DIR}_llama8b_*/ \
        /home/jake/github/neural-steering/experiments/fc_results/${DIR}/ 2>/dev/null || true
done

echo "=== Done! Results in experiments/fc_results/ ==="
