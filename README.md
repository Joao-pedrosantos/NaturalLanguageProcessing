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

## Desktop GPU run (tiny + medium + 10M + large, ambas as arquiteturas)

### Pré-requisitos

- Driver NVIDIA $\ge$ 560, CUDA 12.8 (RTX 5080 / Blackwell exige isso).
- Python 3.10+.
- (Windows) Git for Windows instalado, pra ter `bash` disponível via Git Bash.

### Quickstart no desktop

**Linux/macOS:**

```bash
git clone https://github.com/Joao-pedrosantos/NaturalLanguageProcessing.git
cd NaturalLanguageProcessing
python -m venv venv && source venv/bin/activate
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -e .
bash scripts/run_desktop.sh
```

**Windows (Git Bash — recomendado):**

```bash
git clone https://github.com/Joao-pedrosantos/NaturalLanguageProcessing.git
cd NaturalLanguageProcessing
python -m venv venv
source venv/Scripts/activate          # Git Bash usa Scripts/, não bin/
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -e .
bash scripts/run_desktop.sh
```

**Windows (PowerShell, equivalente):**

```powershell
git clone https://github.com/Joao-pedrosantos/NaturalLanguageProcessing.git
cd NaturalLanguageProcessing
python -m venv venv
venv\Scripts\Activate.ps1
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -e .
# Para rodar o pipeline completo, abra Git Bash e execute:
#   bash scripts/run_desktop.sh
# (As etapas individuais estão listadas abaixo se preferir rodar à mão.)
```

### O que `run_desktop.sh` faz

1. Sanity-check da GPU (CUDA disponível, compute capability $\ge$ 8.0).
2. Baixa livros novos do Project Gutenberg (idempotente — pula os já em `artifacts/train_books/`).
3. Retreina o tokenizer (vocab=8000, byte-level BPE) sobre o corpus atual.
4. Arquiva `runs/` antigo em `runs.cpu_backup/` na primeira execução.
5. Conta params dos 8 modelos (xLSTM e Transformer × tiny/medium/10M/large).
6. Treina os 8 modelos em sequência via `scripts/train_local.py`.
7. Limpa logs concatenados (caso algum run tenha sido retomado).
8. Regenera `scaling_curve.png` com os 8 runs e copia pra raiz pro `pdflatex` consumir.

Tempo estimado total na RTX 5080: ~6–10h. O runner é reentrante — re-executar resume do último checkpoint.

### Etapas à mão (se precisar rodar isolado)

```bash
# Confere GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Baixa corpus
python -m scripts.prepare_books

# Treina tokenizer (vocab 8k, byte-level BPE)
python -m scripts.train_tokenizer --corpus-dir artifacts/train_books --vocab-size 8000 --out artifacts/tokenizer.json

# Conta params de uma config
python -m scripts.count_params --config configs/xlstm_10m.yaml

# Treina uma config específica
python -m scripts.train_local --config configs/train_10m.yaml --train-file artifacts/train_books --dev-file artifacts/dev/pt_dev.txt

# Compara runs e regenera figura
python -m scripts.compare_runs --runs runs/xlstm_tiny runs/xlstm_medium runs/xlstm_10m runs/xlstm_large runs/transformer_tiny runs/transformer_medium runs/transformer_10m runs/transformer_large --labels "xLSTM tiny" "xLSTM medium" "xLSTM 10M" "xLSTM large" "Transformer tiny" "Transformer medium" "Transformer 10M" "Transformer large" --out artifacts/scaling_curve.png
```

### 100M (Wikipedia streaming, infraestrutura separada)

Os 100M usam o pipeline de streaming Wikipedia (`scripts/train.py`,
vocab=32k, ctx=1024) e NÃO entram no `run_desktop.sh`. Quando estiver
pronto pra rodar:

```bash
python -m scripts.sample_for_tokenizer --target-mb 100
python -m scripts.train_tokenizer --vocab-size 32000
python -m scripts.prepare_eval_set --target-mb 5
python -m scripts.train --config configs/train_100m.yaml
python -m scripts.train --config configs/train_100m_transformer.yaml
```

Tempo estimado: ~12–18h cada na RTX 5080 (Chinchilla-optimal de ~2B tokens).

### Conversar com um modelo treinado

```bash
python -m scripts.chat --ckpt runs/xlstm_10m/best.pt --config configs/xlstm_10m.yaml --arch xlstm
python -m scripts.chat --ckpt runs/transformer_10m/best.pt --config configs/transformer_10m.yaml --arch transformer
```

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
