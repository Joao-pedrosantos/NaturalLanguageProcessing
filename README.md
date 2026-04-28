# xLSTM-PTBR-EN

Pré-treino de um xLSTM em texto multilíngue, com fine-tuning downstream.
Projeto final da cadeira de NLP.

## Modos de uso

- **POC** (`configs/*_tiny.yaml`): ~1.5M params, livros locais, ~10-15 min CPU.
- **Medium** (`configs/*_medium.yaml`): ~4.5M params, livros locais, ~20 min CPU.
  Os números do paper saem desses runs.
- **Large** (`configs/*_large.yaml`): ~12-29M params, livros locais, **GPU**,
  algumas horas. Roda via `bash scripts/run_desktop.sh`. Veja seção
  *Desktop GPU run* abaixo.
- **Treino real** (`configs/*_50m.yaml`): ~50M params, Wikipedia PT+EN
  streaming, 24-36h na RTX 5080. Não usado no paper atual.

## Desktop GPU run (large)

Pré-requisitos no desktop:

```bash
# driver NVIDIA >= 560, CUDA 12.8
python -m venv venv && source venv/bin/activate
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -e .
```

Adicione mais livros em `configs/books.yaml` (descomente os candidatos
e preencha o `id` consultando o catálogo do Project Gutenberg). Depois:

```bash
bash scripts/run_desktop.sh
```

O script: confere GPU, baixa livros novos, (re)treina tokenizer se
necessário, treina xLSTM large e Transformer large, regenera
`scaling_curve.png` com tiny + medium + large e copia pra raiz pro
`pdflatex main.tex` consumir.

---

## POC: passo a passo

### 1. Setup

```bash
pip install -e .
```

Se a instalação do `xlstm` reclamar de kernels CUDA, tudo bem — o config
tiny usa `backend: vanilla` (PyTorch puro) e ignora os kernels.

### 2. Baixar um livro

Qualquer `.txt` UTF-8 serve. Sugestão: Dom Casmurro do Projeto Gutenberg.

```bash
mkdir -p artifacts/tokenizer_corpus artifacts/dev

# Linux/Mac:
curl -L -o artifacts/tokenizer_corpus/livro.txt \
  https://www.gutenberg.org/cache/epub/55752/pg55752.txt

# Windows PowerShell:
# Invoke-WebRequest -Uri "https://www.gutenberg.org/cache/epub/55752/pg55752.txt" `
#   -OutFile "artifacts/tokenizer_corpus/livro.txt"
```

> Se o link 404, escolhe outro livro PT em https://www.gutenberg.org/browse/languages/pt
> e baixa o `.txt` (UTF-8).

Os 200KB iniciais viram dev set:

```bash
# Linux/Mac:
head -c 200000 artifacts/tokenizer_corpus/livro.txt > artifacts/dev/pt_dev.txt

# Windows PowerShell:
# (Get-Content artifacts/tokenizer_corpus/livro.txt -Raw).Substring(0, 200000) | `
#   Out-File -Encoding utf8 artifacts/dev/pt_dev.txt
```

### 3. Treinar tokenizer

```bash
python -m scripts.train_tokenizer \
    --corpus-dir artifacts/tokenizer_corpus \
    --vocab-size 4000
```

Saída: `artifacts/tokenizer.json`. Demora ~30s.

### 4. Verificar tamanho do modelo

```bash
python -m scripts.count_params --config configs/xlstm_tiny.yaml
```

Deve imprimir ~1-2M params.

### 5. Treinar

```bash
python -m scripts.train_local \
    --config configs/train_tiny.yaml \
    --train-file artifacts/tokenizer_corpus/livro.txt \
    --dev-file artifacts/dev/pt_dev.txt
```

500 steps, ~10-15 min em CPU moderna. Logs em `runs/tiny/log.jsonl`,
checkpoint em `runs/tiny/best.pt`.

### 6. Gerar amostras

```bash
python -m scripts.generate \
    --ckpt runs/tiny/best.pt \
    --config configs/xlstm_tiny.yaml \
    --prompt "Capitu olhou para mim"
```

---

## Sinais de POC bem-sucedida

- **Loss inicial:** ~8.3 (≈ log(4000), entropia uniforme do vocab)
- **Loss final:** entre 4 e 6 dependendo do livro
- **Gerações:** fragmentos de palavras reais misturados com lixo. Você
  pode ver pedaços ("ela", "que", "para") aparecerem no meio. Não espera
  frases coerentes — o modelo é minúsculo e treinou pouquíssimo.

Se a loss não desce ou gera só `<unk>`, algo quebrou. Os suspeitos
mais comuns: vocab incompatível entre tokenizer e config, arquivo de
treino vazio, ou seq_len > context_length.

---

## Treino real (depois que a POC passar)

```bash
python -m scripts.sample_for_tokenizer --target-mb 100
python -m scripts.train_tokenizer --vocab-size 32000
python -m scripts.prepare_eval_set --target-mb 5
python -m scripts.count_params --config configs/xlstm_50m.yaml
python -m scripts.train --config configs/train_50m.yaml
```

Pode trocar `backend: vanilla` por `backend: cuda` em
`configs/xlstm_50m.yaml` na 5080 — vai ser muito mais rápido.

---

## Estrutura do repo

```
xlstm-ptbr-en/
├── pyproject.toml
├── README.md
├── configs/
│   ├── xlstm_tiny.yaml       # POC: 1-2M params
│   ├── train_tiny.yaml
│   ├── xlstm_50m.yaml        # treino real: 50M params
│   └── train_50m.yaml
├── src/
│   ├── model.py              # factory + contagem
│   ├── training.py           # loop, optim, eval, ckpt
│   └── data/
│       ├── streaming.py      # mix Wikipedia PT+EN
│       ├── tokenizer.py      # BPE byte-level
│       └── eval_dataset.py   # dev set file-based
├── scripts/
│   ├── sample_for_tokenizer.py
│   ├── train_tokenizer.py
│   ├── prepare_eval_set.py
│   ├── count_params.py
│   ├── test_pipeline.py      # sanity check do streaming
│   ├── train.py              # treino real (Wikipedia streaming)
│   ├── train_local.py        # POC (arquivo único)
│   └── generate.py           # gerar texto de checkpoint
└── artifacts/                # tokenizer.json, corpora, dev sets
```
