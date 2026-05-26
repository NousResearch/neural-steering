#!/bin/bash
# Apparatus 2a cluster entrypoint.
#
# Submit from the cluster checkout:
#   sbatch apparatus/slurm_probe_roles.sh

#SBATCH --job-name=ns-probe-roles
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --qos=p3
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
OUT_DIR="apparatus/output/probe_roles_llama8b_${STAMP}"
PROBE_LAYERS="18,24,28"

echo "host=$(hostname)"
echo "qos=p3"
echo "probe_layers=${PROBE_LAYERS}"
echo "out_dir=${OUT_DIR}"

python -m apparatus.probe_role_table \
  --model llama8b \
  --circuit experiments/topology_llama8b_20260305_163508/relp-behavioral_refusal_kstar91/circuit.json \
  --layers "${PROBE_LAYERS}" \
  --n_random 20 \
  --output_dir "${OUT_DIR}"

echo "PROBE_ROLE_OUTPUT ${OUT_DIR}"
