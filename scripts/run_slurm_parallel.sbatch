#!/usr/bin/env bash
#SBATCH --job-name=xlstm-parallel
#SBATCH --partition=monstrao
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-parallel-%j.out
#SBATCH --error=slurm-parallel-%j.err
#
# Opção 1: paralelismo "embaraçoso" — 4 modelos por vez, um por GPU.
# Não muda código de treino. Cada modelo continua vendo 1 GPU.
#
# Submit:  sbatch scripts/run_slurm_parallel.sbatch
#
# Importante: monstrao tem limite de 20 min. Se as 2 ondas não couberem,
# divida em 2 submissões (edite run_parallel.sh pra rodar só wave 1, depois
# resubmeta pra wave 2 — os runs são idempotentes via checkpoint).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load cuda/12.8.1
source venv/bin/activate

echo "=== job info ==="
echo "  node:    $(hostname)"
echo "  jobid:   ${SLURM_JOB_ID:-n/a}"
echo "  gpus:    ${CUDA_VISIBLE_DEVICES:-n/a}"
nvidia-smi || true

PY="$(which python)" bash scripts/run_parallel.sh

echo "=== done ==="
