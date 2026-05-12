#!/usr/bin/env bash
# Roda apenas os transformers (tiny, medium, 10M) em sequência,
# seguidos de clean_log, compare_runs e probe_classify nos
# checkpoints medium/10M de cada arch.
#
# Uso:
#   bash scripts/run_transformers.sh
#
# Loga tudo em logs/run_transformers_<timestamp>.log.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs
LOG="logs/run_transformers_$(date +%Y%m%d_%H%M%S).log"

# Pega o python: prefere venv local, senão python do PATH.
if [ -z "${PY:-}" ]; then
    if [ -x "venv/Scripts/python.exe" ]; then
        PY="venv/Scripts/python.exe"
    elif [ -x "venv/bin/python" ]; then
        PY="venv/bin/python"
    else
        PY="$(command -v python)"
    fi
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_step() {
    local label="$1"; shift
    log "=== $label ==="
    "$PY" "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        log "FAIL: $label (exit $rc)"
        exit "$rc"
    fi
}

log "python: $PY"

# --- Treinos ---
run_step "Transformer tiny" \
    -m scripts.train_local \
    --config configs/train_tiny_transformer.yaml \
    --train-file artifacts/train_books \
    --dev-file artifacts/dev/pt_dev.txt

run_step "Transformer medium" \
    -m scripts.train_local \
    --config configs/train_medium_transformer.yaml \
    --train-file artifacts/train_books \
    --dev-file artifacts/dev/pt_dev.txt

run_step "Transformer 10M" \
    -m scripts.train_local \
    --config configs/train_10m_transformer.yaml \
    --train-file artifacts/train_books \
    --dev-file artifacts/dev/pt_dev.txt

# --- Clean logs (mantém só xLSTM + Transformer existentes até 10M) ---
LOGS_TO_CLEAN=()
for f in \
    runs/xlstm_tiny/log.jsonl \
    runs/xlstm_medium/log.jsonl \
    runs/xlstm_10m/log.jsonl \
    runs/transformer_tiny/log.jsonl \
    runs/transformer_medium/log.jsonl \
    runs/transformer_10m/log.jsonl
do
    [ -f "$f" ] && LOGS_TO_CLEAN+=("$f")
done

if [ "${#LOGS_TO_CLEAN[@]}" -gt 0 ]; then
    run_step "clean_log" -m scripts.clean_log "${LOGS_TO_CLEAN[@]}"
fi

# --- Scaling curve (só runs com log.jsonl presente) ---
RUN_DIRS=()
RUN_LABELS=()
add_run() {
    local dir="$1"; local label="$2"
    if [ -f "$dir/log.jsonl" ]; then
        RUN_DIRS+=("$dir")
        RUN_LABELS+=("$label")
    fi
}
add_run runs/xlstm_tiny        "xLSTM tiny"
add_run runs/xlstm_medium      "xLSTM medium"
add_run runs/xlstm_10m         "xLSTM 10M"
add_run runs/transformer_tiny  "Transformer tiny"
add_run runs/transformer_medium "Transformer medium"
add_run runs/transformer_10m   "Transformer 10M"

if [ "${#RUN_DIRS[@]}" -gt 0 ]; then
    run_step "compare_runs" \
        -m scripts.compare_runs \
        --runs "${RUN_DIRS[@]}" \
        --labels "${RUN_LABELS[@]}" \
        --out artifacts/scaling_curve.png
    if [ -f artifacts/scaling_curve.png ]; then
        cp artifacts/scaling_curve.png ./scaling_curve.png
    fi
fi

# --- Probe (author classification) nos medium/10M de cada arch ---
for ARCH_SCALE in xlstm_medium xlstm_10m transformer_medium transformer_10m; do
    arch="${ARCH_SCALE%%_*}"
    cfg="configs/${ARCH_SCALE}.yaml"
    ckpt="runs/${ARCH_SCALE}/best.pt"
    out="artifacts/probe_${ARCH_SCALE}.json"
    if [ -f "$ckpt" ]; then
        run_step "probe ${ARCH_SCALE}" \
            -m scripts.probe_classify \
            --ckpt "$ckpt" --config "$cfg" --arch "$arch" \
            --out "$out"
    else
        log "(skip probe ${ARCH_SCALE}: $ckpt ausente)"
    fi
done

log "feito."
