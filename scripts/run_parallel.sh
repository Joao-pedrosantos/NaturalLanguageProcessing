#!/usr/bin/env bash
# Roda os 8 treinos em 2 ondas de 4 (uma por GPU).
# Pré-requisito: as 4 GPUs visíveis (slurm aloca via --gres=gpu:v100-32gb:4).
#
# Uso:
#   PY=$(which python) bash scripts/run_parallel.sh
#
# Logs por modelo ficam em logs/<nome>.log.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-venv/bin/python}"
mkdir -p logs

run_on_gpu() {
    local gpu=$1
    local label=$2
    local cfg=$3
    local logfile=$4
    echo "[GPU $gpu] start: $label"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -m scripts.train_local \
        --config "$cfg" \
        --train-file artifacts/train_books \
        --dev-file artifacts/dev/pt_dev.txt \
        > "$logfile" 2>&1
    echo "[GPU $gpu] done:  $label"
}

echo "=== wave 1: tiny + medium (xLSTM e Transformer) ==="
t0=$(date +%s)
run_on_gpu 0 "xLSTM tiny"         configs/train_tiny.yaml               logs/xlstm_tiny.log &
run_on_gpu 1 "xLSTM medium"       configs/train_medium.yaml             logs/xlstm_medium.log &
run_on_gpu 2 "Transformer tiny"   configs/train_tiny_transformer.yaml   logs/transformer_tiny.log &
run_on_gpu 3 "Transformer medium" configs/train_medium_transformer.yaml logs/transformer_medium.log &
wait
t1=$(date +%s)
echo "wave 1 elapsed: $((t1-t0))s"

echo "=== wave 2: 10M + large (xLSTM e Transformer) ==="
run_on_gpu 0 "xLSTM 10M"           configs/train_10m.yaml                logs/xlstm_10m.log &
run_on_gpu 1 "xLSTM large"         configs/train_large.yaml              logs/xlstm_large.log &
run_on_gpu 2 "Transformer 10M"     configs/train_10m_transformer.yaml    logs/transformer_10m.log &
run_on_gpu 3 "Transformer large"   configs/train_large_transformer.yaml  logs/transformer_large.log &
wait
t2=$(date +%s)
echo "wave 2 elapsed: $((t2-t1))s"
echo "total elapsed: $((t2-t0))s"
