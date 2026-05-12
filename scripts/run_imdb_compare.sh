#!/usr/bin/env bash
# Traduz subset do IMDB (EN -> PT) com MarianMT e avalia probe de
# sentimento sobre xLSTM 10M vs Transformer 10M.
#
# Uso:
#   PY=/c/Users/jpthe/miniconda3/envs/xlstm/python.exe \
#       bash scripts/run_imdb_compare.sh

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs artifacts
LOG="logs/imdb_compare_$(date +%Y%m%d_%H%M%S).log"

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

IMDB_JSONL="artifacts/imdb_pt_marianmt.jsonl"
MAX_PER_CLASS="${MAX_PER_CLASS:-1000}"  # 2000 reviews total (balanceado)

# --- Traduz (skip se já existir) ---
if [ -f "$IMDB_JSONL" ]; then
    EXISTING=$(wc -l < "$IMDB_JSONL" | tr -d ' ')
    log "translation already exists ($EXISTING lines): $IMDB_JSONL"
else
    run_step "translate_imdb (MarianMT, max-per-class=$MAX_PER_CLASS)" \
        -m scripts.translate_imdb \
        --max-per-class "$MAX_PER_CLASS" \
        --out "$IMDB_JSONL"
fi

# --- Probes 10M ---
for ARCH_SCALE in xlstm_10m transformer_10m; do
    arch="${ARCH_SCALE%%_*}"
    cfg="configs/${ARCH_SCALE}.yaml"
    ckpt="runs/${ARCH_SCALE}/best.pt"
    out="artifacts/sentiment_${ARCH_SCALE}.json"
    if [ -f "$ckpt" ]; then
        run_step "probe_sentiment ${ARCH_SCALE}" \
            -m scripts.probe_sentiment \
            --ckpt "$ckpt" --config "$cfg" --arch "$arch" \
            --local-jsonl "$IMDB_JSONL" \
            --out "$out"
    else
        log "(skip ${ARCH_SCALE}: $ckpt ausente)"
    fi
done

# --- Resumo lado-a-lado ---
log "=== resumo ==="
"$PY" - <<'PYEOF' | tee -a "$LOG"
import json
from pathlib import Path

rows = []
for name in ("xlstm_10m", "transformer_10m"):
    p = Path(f"artifacts/sentiment_{name}.json")
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    rows.append((name, r))

if not rows:
    print("nenhum resultado encontrado.")
else:
    print(f"{'model':>18} {'test_acc':>10} {'macro_f1':>10} {'f1_neg':>10} {'f1_pos':>10}")
    for name, r in rows:
        print(f"{name:>18} {r['test_acc']*100:>9.2f}% "
              f"{r['macro_f1']*100:>9.2f}% "
              f"{r['per_class_f1'][0]*100:>9.2f}% "
              f"{r['per_class_f1'][1]*100:>9.2f}%")
    if len(rows) == 2:
        a = rows[0][1]['test_acc'] * 100
        b = rows[1][1]['test_acc'] * 100
        winner = rows[0][0] if a > b else rows[1][0]
        print(f"\nvencedor (test_acc): {winner}")
PYEOF

log "feito."
