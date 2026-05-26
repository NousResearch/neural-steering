#!/bin/bash
# L24/N1619 dossier — activation profile + steering rollouts.
#
# Submit from the cluster checkout:
#   sbatch apparatus/slurm_dossier_l24n1619.sh

#SBATCH --job-name=ns-dossier-1619
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
OUT_DIR="apparatus/output/dossier_L24N1619_${STAMP}"

echo "[$(hostname)] OUT_DIR=${OUT_DIR}"

python -m apparatus.dossier_l24n1619 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir "${OUT_DIR}"

echo "DOSSIER_OUTPUT ${OUT_DIR}"
