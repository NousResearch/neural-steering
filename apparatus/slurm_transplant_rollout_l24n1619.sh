#!/bin/bash
# L24/N1619 transplant rollout test.
#
# Submit from the cluster checkout:
#   sbatch apparatus/slurm_transplant_rollout_l24n1619.sh

#SBATCH --job-name=ns-tx-1619
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
OUT_DIR="apparatus/output/transplant_rollout_L24N1619_${STAMP}"

echo "[$(hostname)] OUT_DIR=${OUT_DIR}"
echo "NEURONS=L24/N1619,L24/N2598,L22/N3319,L20/N9928,L26/N11984"

python -m apparatus.transplant_rollout_l24n1619 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir "${OUT_DIR}" \
  --neurons "L24/N1619,L24/N2598,L22/N3319,L20/N9928,L26/N11984"

echo "TRANSPLANT_ROLLOUT_OUTPUT ${OUT_DIR}"
