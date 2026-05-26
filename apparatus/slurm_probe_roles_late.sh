#!/bin/bash
# Late-layer Apparatus 2a probe extension.
#
# Runs hidden probes at L30 and L32. L32 is the final residual stream after all
# decoder layers/final norm, supported by probe_role_table.py.
#
# Submit from the cluster checkout:
#   sbatch apparatus/slurm_probe_roles_late.sh

#SBATCH --job-name=ns-probe-late
#SBATCH --partition=batch
#SBATCH --qos=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jake/neural-steering-role-decomp/slurm_logs/%x-%j.out
#SBATCH --error=/home/jake/neural-steering-role-decomp/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/jake/neural-steering-role-decomp
source .venv/bin/activate

export HF_HOME=/home/jake/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/jake/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="apparatus/output/probe_roles_late_llama8b_${STAMP}"
PROBE_LAYERS="30,32"
TOKEN_TABLE="apparatus/output/role_table_refusal_8b_fullcircuit_20260526.jsonl"
CIRCUIT="experiments/topology_llama8b_20260305_163508/relp-behavioral_refusal_kstar91/circuit.json"

echo "host=$(hostname)"
echo "qos=p3"
echo "probe_layers=${PROBE_LAYERS}"
echo "out_dir=${OUT_DIR}"

python -m apparatus.probe_role_table \
  --model llama8b \
  --circuit "${CIRCUIT}" \
  --layers "${PROBE_LAYERS}" \
  --n_random 20 \
  --output_dir "${OUT_DIR}"

for L in 30 32; do
  python -m apparatus.compare_probe_roles \
    --token-role-table "${TOKEN_TABLE}" \
    --probe-role-rows "${OUT_DIR}/probe_role_rows_layer${L}.jsonl" \
    --out "${OUT_DIR}/compare_token_probe_layer${L}.json"
done

python -m apparatus.probe_consistency \
  --probe-dir "${OUT_DIR}" \
  --layers "${PROBE_LAYERS}" \
  --out "${OUT_DIR}/cross_layer_consistency.json"

python -m apparatus.visualize_gate_substrate \
  --probe-dir "${OUT_DIR}" \
  --layers "${PROBE_LAYERS}" \
  --out "${OUT_DIR}/gate_substrate_late_refusal_8b.png"

echo "PROBE_ROLE_OUTPUT ${OUT_DIR}"
