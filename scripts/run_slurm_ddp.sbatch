#!/usr/bin/env bash
#SBATCH --job-name=xlstm-ddp
#SBATCH --partition=monstrao
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-ddp-%j.out
#SBATCH --error=slurm-ddp-%j.err
#
# Opção 2: DDP — treina UM modelo de cada vez, batch dividido entre 4 GPUs.
# Roda os 8 configs em sequência (cada um usando 4 GPUs via torchrun).
#
# Submit:  sbatch scripts/run_slurm_ddp.sbatch
#
# Importante: 20 min é pouco pros 8 configs em sequência. Pra benchmark,
# rode UM config primeiro (comente os outros) e veja quanto tempo leva
# o `large` com 4 GPUs vs 1 GPU (no run_slurm.sbatch original).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load cuda/12.8.1
source venv/bin/activate

echo "=== job info ==="
echo "  node:    $(hostname)"
echo "  jobid:   ${SLURM_JOB_ID:-n/a}"
nvidia-smi || true

NPROC=4
mkdir -p logs

run_ddp() {
    local label=$1
    local cfg=$2
    echo
    echo "=== DDP train: $label ($cfg) ==="
    local t0=$(date +%s)
    torchrun --nproc_per_node=$NPROC --standalone \
        -m scripts.train_local_ddp \
        --config "$cfg" \
        --train-file artifacts/train_books \
        --dev-file artifacts/dev/pt_dev.txt
    local t1=$(date +%s)
    echo "$label elapsed: $((t1-t0))s"
}

# Pra comparar com Opção 1, treine pelo menos o `large` aqui.
# Comente o que não precisa pra caber nos 20 min.

run_ddp "xLSTM large"         configs/train_large.yaml
# run_ddp "Transformer large"   configs/train_large_transformer.yaml
# run_ddp "xLSTM 10M"           configs/train_10m.yaml
# run_ddp "Transformer 10M"     configs/train_10m_transformer.yaml
# run_ddp "xLSTM medium"        configs/train_medium.yaml
# run_ddp "Transformer medium"  configs/train_medium_transformer.yaml
# run_ddp "xLSTM tiny"          configs/train_tiny.yaml
# run_ddp "Transformer tiny"    configs/train_tiny_transformer.yaml

echo "=== done ==="
