#!/usr/bin/env bash
#SBATCH --job-name=xlstm-ptbr-en
#SBATCH --partition=monstrao
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Submit with:   sbatch scripts/run_slurm.sbatch
# Check status:  squeue -u $USER
# Cancel:        scancel <jobid>
# Tail logs:     tail -f slurm-<jobid>.out
#
# Before submitting, fill in the REPLACE_ME_* lines above and check:
#   sinfo                         # list partitions + GPU availability
#   sinfo -o "%P %G %D %t"        # partitions with GRES (GPU) info
#   sacctmgr show assoc user=$USER format=account,partition,qos
#   module avail python           # see python/cuda modules available
#
# If your cluster uses a different GPU type request, adjust --gres
# (e.g. --gres=gpu:a100:1 to require an A100).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# --- environment ---------------------------------------------------------
# Pick ONE of the two blocks below depending on what your cluster offers.

# Insper supercomp usa Lmod. CUDA 12.8.1 está disponível como módulo.
module load cuda/12.8.1

# Ativa o venv do projeto (criado uma vez no login node — ver instruções abaixo).
source venv/bin/activate

# Some clusters don't expose internet on compute nodes. If `prepare_books`
# fails to download, run it on the login node BEFORE sbatch, or set:
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "=== job info ==="
echo "  node:    $(hostname)"
echo "  jobid:   ${SLURM_JOB_ID:-n/a}"
echo "  gpus:    ${CUDA_VISIBLE_DEVICES:-n/a}"
nvidia-smi || true

# --- pipeline ------------------------------------------------------------
# Reuses scripts/run_desktop.sh, which already orchestrates the 8 trainings
# + tokenizer + probe + scaling curve.
PY="$(which python)" bash scripts/run_desktop.sh

echo "=== done ==="
