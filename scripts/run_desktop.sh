#!/usr/bin/env bash
# One-shot pipeline pra rodar no desktop com GPU.
#
# Pré-requisitos (faça antes de rodar):
#   1) Driver NVIDIA recente (>= 560) e CUDA 12.8.
#   2) Ambiente com PyTorch nightly cu128 e xlstm instalados:
#        python -m venv venv && source venv/bin/activate
#        pip install --pre torch --index-url \
#            https://download.pytorch.org/whl/nightly/cu128
#        pip install -e .
#   3) (Opcional) Editar configs/books.yaml e adicionar mais livros.
#
# O que faz, em ordem:
#   - sanity-check da GPU
#   - baixa/atualiza livros do Gutenberg (idempotente)
#   - retreina o tokenizer (vocab=8000) sobre o corpus atual
#   - arquiva runs/ antigos (se forem do CPU) em runs.cpu_backup/
#   - treina 6 modelos: xLSTM e Transformer × tiny/medium/large
#   - limpa logs e regenera a scaling curve com os 6 runs
#
# Uso:
#   bash scripts/run_desktop.sh
#
# Pra retomar de um checkpoint, basta re-rodar (o train loop carrega
# o último step_*.pt em runs/<name>/).

set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-venv/bin/python}"

echo "=== sanity-check GPU ==="
"$PY" - <<'PYEOF'
import torch
ok = torch.cuda.is_available()
print(f"  cuda available: {ok}")
if ok:
    print(f"  device: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  compute capability: sm_{cap[0]}{cap[1]}")
    if cap[0] < 8:
        raise SystemExit("CC<8.0; xlstm cuda backend exige >=8.0")
else:
    raise SystemExit("CUDA indisponível — rode em máquina com GPU ou ajuste o backend para vanilla")
PYEOF

echo
echo "=== prepare books ==="
"$PY" -m scripts.prepare_books

echo
echo "=== train tokenizer (vocab=8000, corpus expandido) ==="
# Sempre retreina: o corpus em artifacts/train_books/ é a fonte de verdade.
if [ -f "artifacts/tokenizer.json" ]; then
    cp artifacts/tokenizer.json artifacts/tokenizer.json.bak
fi
"$PY" -m scripts.train_tokenizer \
    --corpus-dir artifacts/train_books \
    --vocab-size 8000 \
    --out artifacts/tokenizer.json

echo
echo "=== archive old CPU runs (se ainda não arquivados) ==="
# Os runs antigos (CPU, vocab=4000, corpus contaminado) ficam preservados
# em runs.cpu_backup/ pra histórico, mas NÃO entram na scaling curve nova.
if [ -d "runs" ] && [ ! -d "runs.cpu_backup" ]; then
    mv runs runs.cpu_backup
    echo "  runs/ -> runs.cpu_backup/"
fi
mkdir -p runs

echo
echo "=== count params ==="
for ARCH_CONFIG in xlstm_tiny xlstm_medium xlstm_10m xlstm_large; do
    "$PY" -m scripts.count_params --config "configs/${ARCH_CONFIG}.yaml" || true
done
for ARCH_CONFIG in transformer_tiny transformer_medium transformer_10m transformer_large; do
    "$PY" -m scripts.count_params --config "configs/${ARCH_CONFIG}.yaml" --arch transformer || true
done

run_one() {
    local label="$1"
    local cfg="$2"
    echo
    echo "=== train ${label} ==="
    "$PY" -m scripts.train_local \
        --config "${cfg}" \
        --train-file artifacts/train_books \
        --dev-file artifacts/dev/pt_dev.txt
}

# xLSTM family
run_one "xLSTM tiny"        configs/train_tiny.yaml
run_one "xLSTM medium"      configs/train_medium.yaml
run_one "xLSTM 10M"         configs/train_10m.yaml
run_one "xLSTM large"       configs/train_large.yaml

# Transformer family
run_one "Transformer tiny"   configs/train_tiny_transformer.yaml
run_one "Transformer medium" configs/train_medium_transformer.yaml
run_one "Transformer 10M"    configs/train_10m_transformer.yaml
run_one "Transformer large"  configs/train_large_transformer.yaml

echo
echo "=== clean logs (caso algum run tenha sido retomado) ==="
"$PY" -m scripts.clean_log \
    runs/xlstm_tiny/log.jsonl \
    runs/xlstm_medium/log.jsonl \
    runs/xlstm_10m/log.jsonl \
    runs/xlstm_large/log.jsonl \
    runs/transformer_tiny/log.jsonl \
    runs/transformer_medium/log.jsonl \
    runs/transformer_10m/log.jsonl \
    runs/transformer_large/log.jsonl

echo
echo "=== regenerate scaling curve ==="
"$PY" -m scripts.compare_runs \
    --runs runs/xlstm_tiny runs/xlstm_medium runs/xlstm_10m runs/xlstm_large \
           runs/transformer_tiny runs/transformer_medium runs/transformer_10m runs/transformer_large \
    --labels "xLSTM tiny" "xLSTM medium" "xLSTM 10M" "xLSTM large" \
             "Transformer tiny" "Transformer medium" "Transformer 10M" "Transformer large" \
    --out artifacts/scaling_curve.png

cp artifacts/scaling_curve.png ./scaling_curve.png

echo
echo "feito. tabela final está no output do compare_runs acima."
echo "próximo passo: atualizar Tabela 1 no main.tex e rodar pdflatex."
