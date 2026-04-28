# PROJECT_NOTES.md

> Handoff note from a long claude.ai conversation. Save this in the
> project root. When starting a new Claude Code session, point Claude
> at this file first: *"Read PROJECT_NOTES.md and `paper/main.tex`,
> then help me with [task]."*

---

## TL;DR

Final project for the **Natural Language Processing** class at Insper
(graduating June 2026). The deliverable is an **English-language paper**
(LaTeX → PDF, Nature-style) comparing **xLSTM** and **Transformer**
language models trained from scratch on **19th-century Portuguese
literature**. Pipeline is reproducible and open-source. Empirical
finding: **crossover** between architectures across two parameter
scales. Paper draft exists at `paper/main.tex`. Three placeholder edits
remain plus one figure to regenerate.

---

## Project structure (as of handoff)

```
NaturalLanguageProcessing/
├── PROJECT_NOTES.md            # this file
├── README.md                   # repo README (project overview)
├── pyproject.toml              # `pip install -e .`
├── requirements.txt
├── configs/
│   ├── xlstm_tiny.yaml         # ~1.5M params
│   ├── xlstm_medium.yaml       # ~4.5M params
│   ├── xlstm_50m.yaml          # not yet trained — for future GPU run
│   ├── transformer_tiny.yaml
│   ├── transformer_medium.yaml
│   ├── train_tiny.yaml
│   ├── train_tiny_transformer.yaml
│   ├── train_medium.yaml
│   ├── train_medium_transformer.yaml
│   └── train_50m.yaml
├── src/
│   ├── data/
│   │   ├── streaming.py            # streaming Wikipedia (for 50m future run)
│   │   ├── eval_dataset.py         # deterministic file-based eval
│   │   ├── multi_file_dataset.py   # concat books for tiny/medium
│   │   └── tokenizer.py            # byte-level BPE wrapper
│   ├── model.py                # xLSTM factory
│   ├── transformer.py          # GPT-style decoder (PyTorch native)
│   ├── model_factory.py        # arch dispatcher (xlstm vs transformer)
│   └── training.py             # shared training loop
├── scripts/
│   ├── prepare_books.py        # downloads + cleans Gutenberg books
│   ├── train_tokenizer.py
│   ├── sample_for_tokenizer.py
│   ├── prepare_eval_set.py     # for the future 50M wikipedia run
│   ├── count_params.py
│   ├── train_local.py          # main entry point: tiny/medium runs
│   ├── train.py                # main entry for the 50M wikipedia run
│   ├── test_pipeline.py
│   ├── generate.py             # text generation from checkpoint
│   ├── compare_runs.py         # parses log.jsonl, plots scaling_curve.png
│   ├── clean_log.py            # de-dups concatenated runs in log files
│   └── relatorio_assets.py     # extracts numbers for the report
├── artifacts/
│   ├── tokenizer.json          # 4000-vocab byte-level BPE
│   ├── tokenizer_corpus/       # text used to train the tokenizer
│   ├── train_books/            # 6 cleaned .txt training novels
│   ├── dev/pt_dev.txt          # holdout: Brás Cubas
│   └── scaling_curve.png       # ⚠️ has visual bug, needs regen
└── runs/
    ├── xlstm_tiny/             # log.jsonl + best.pt + final.pt + step_*.pt
    ├── xlstm_medium/
    ├── transformer_tiny/
    └── transformer_medium/
└── paper/
    ├── main.tex                # 7-page LaTeX paper, English, Nature-ish
    ├── main.pdf                # compiled (with placeholders to fix)
    ├── scaling_curve.png       # placeholder copy of the buggy figure
    └── README.md               # how to recompile
```

---

## What's done

### Pipeline (the methodology contribution)

End-to-end, isolates architecture as the sole variable:

1. **Data**: `prepare_books.py` downloads 6 books from Project Gutenberg
   and one holdout, strips the standard `*** START` / `*** END`
   boilerplate (which is in English and would otherwise pollute
   training).
2. **Tokenizer**: byte-level BPE, 4000 vocab, trained once on the
   training books. Byte-level chosen so accented Portuguese never hits
   `<unk>`.
3. **Model factory**: `src/model_factory.py` dispatches on a single
   `arch` flag (`"xlstm"` or `"transformer"`). Every other config field
   is shared.
4. **Training loop**: `src/training.py` — AdamW (β=0.9,0.95), cosine LR
   schedule with linear warmup, weight decay 0.1 (only on tensors of
   rank ≥ 2, Chinchilla convention), gradient clipping at 1.0,
   deterministic-on-the-same-seed.
5. **Eval**: `src/data/eval_dataset.py` — deterministic, file-based,
   identical chunks every time. Used during training every N steps.
6. **Logging**: each run writes `runs/<name>/log.jsonl` with one record
   per logged step plus eval records. `scripts/compare_runs.py` parses
   these to produce the comparison plot and table.

### Trained models (4 final runs)

All trained on **CPU**, fp32, on João's laptop (RTX 5080 was unusable
because the NVIDIA driver was too old for the installed PyTorch).

| Run | Params | Steps | Best dev PPL | Wall time |
|-----|-------:|------:|-------------:|----------:|
| xLSTM tiny         | 1.46M | 1500 | 461.9 | ~10 min |
| Transformer tiny   | 1.45M | 1500 | 332.5 | ~3 min  |
| xLSTM medium       | 4.55M | 2000 | 229.9 | ~19 min (with one hibernation interruption — see issues) |
| Transformer medium | 4.22M | 2000 | 235.6 | ~4 min  |

### Empirical observations

- **Crossover with scale.** At tiny scale, Transformer wins by 28%. At
  medium scale, xLSTM marginally wins by 2.5%. Consistent with the
  hypothesis from the xLSTM paper that its advantages emerge with
  scale, but the medium-scale gap is too small to claim with one seed.
- **Both medium runs are undertrained.** Both reached `best == final`,
  indicating the LR schedule cosine-decayed before perplexity
  plateaued. More steps (e.g., 4000) would likely push both lower and
  could shift the ranking.
- **Throughput differences are confounded.** xLSTM trains 5–8× slower
  than Transformer in CPU because we used the `vanilla` (pure PyTorch)
  xLSTM backend. The `cuda` backend would close most of this gap on GPU.
  The paper explicitly flags this as an implementation, not
  architectural, cost.
- **Qualitative samples** at medium scale reproduce 19th-century
  Portuguese orthography (`pae`, `n'um`, em-dash dialogue). xLSTM
  samples feel slightly more coherent than Transformer samples to a
  human reader, even though their PPLs are nearly identical.

### Paper

`paper/main.tex` is a **7-page LaTeX paper, single-column, Times-like,
in English**, written in a "methodology-first" tone (pipeline is the
contribution; crossover is illustration). It already compiles with
`pdflatex main.tex` (two passes). It uses only standard TeXLive
packages — no special class file. The current PDF (`paper/main.pdf`)
embeds a buggy version of the scaling curve.

Sections: Abstract, Introduction, Related Work, Methodology
(the main contribution), Results (Table 1 + Figure 1 + throughput
caveat), Discussion, Limitations, Conclusion, Code & Data Availability,
References (8 cites), Appendix A (samples).

---

## Known issues to fix before submitting

### 1. `scaling_curve.png` has a visual bug

**Cause.** During the xLSTM medium run, the laptop hibernated near
step 1740 (the log shows a 22000-second `wall_s` jump on a single
step). João re-ran the training from scratch. Both runs were appended
to `runs/xlstm_medium/log.jsonl`, so the file now contains two
concatenated runs.

When `compare_runs.py` reads this, it plots both runs in sequence,
which on the log-scale y-axis looks like a near-straight diagonal
line for `xLSTM 4.5M`. **The numbers in Table 1 are correct** (they
come from the last eval, which was the second run); only the figure
is wrong.

**Fix.** A `scripts/clean_log.py` was written to detect concatenated
runs (looks for a `step` value that decreases) and keep only the last
complete run. It writes a `.bak` backup before modifying anything.

```bash
python -m scripts.clean_log runs/xlstm_tiny/log.jsonl \
                            runs/transformer_tiny/log.jsonl \
                            runs/xlstm_medium/log.jsonl \
                            runs/transformer_medium/log.jsonl

python -m scripts.compare_runs \
    --runs runs/xlstm_tiny runs/xlstm_medium \
           runs/transformer_tiny runs/transformer_medium \
    --labels "xLSTM 1.5M" "xLSTM 4.5M" "Transformer 1.5M" "Transformer 4.2M" \
    --out artifacts/scaling_curve.png

cp artifacts/scaling_curve.png paper/scaling_curve.png
cd paper && pdflatex main.tex && pdflatex main.tex
```

Only `xlstm_medium/log.jsonl` is expected to actually need cleaning,
but running on all four is harmless (single-run logs are unchanged).

### 2. Three placeholders in `main.tex`

- `\author[1]{João~[Last~Name]}` — line ~76. **Replace `[Last Name]`
  with the actual last name.**
- `\url{https://github.com/[username]/xlstm-ptbr-en}` in the
  "Code and Data Availability" section. **Replace with actual GitHub URL.**
- The figure file `scaling_curve.png` itself (see issue #1).

### 3. (Optional) Title tone

Current title emphasises methodology:
> *"A Reproducible Pipeline for Architecture Comparison in Low-Resource
> Language Modelling: A Case Study on xLSTM versus Transformer"*

If the professor prefers an empirically-driven title, swap to:
> *"An xLSTM-vs-Transformer Crossover at Sub-5M Parameters: Empirical
> Observations from a Reproducible Pipeline"*

---

## Open questions / decisions deferred

### Should we run multiple seeds before submitting?

The medium-scale gap (2.5%) is within plausible seed variance for
small-scale LM. The paper currently flags this as a Limitation. If time
allows, **3 seeds × 4 configs = 12 runs ≈ 4–5 hours total wall-time on
CPU** would convert "suggestive observation" into a defensible claim
with error bars.

Not strictly needed for the class deliverable; would meaningfully
strengthen the paper. Decision deferred to João.

### Should we run the 50M-param GPU experiment?

`configs/xlstm_50m.yaml` and `configs/train_50m.yaml` exist as a
target for the RTX 5080. Setup required:

1. Update NVIDIA driver to ≥ 560 (current is 12020, too old).
2. Install PyTorch nightly with CUDA 12.8:
   `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`
3. Switch the xLSTM `slstm.backend` from `vanilla` to `cuda` in the
   config (RTX 5080 is sm_120, supports CC ≥ 8.0).
4. Run `scripts.prepare_eval_set` and `scripts.sample_for_tokenizer` to
   build a Wikipedia subset (these download from HuggingFace).
5. Train: `python -m scripts.train --config configs/train_50m.yaml`.

Estimated wall time on RTX 5080: 24–36 hours of training. Decision
deferred — depends on whether the class allows further work past the
deadline, and whether the paper revision benefits from larger-scale
results.

### Do we want to add a Bibliography file?

References are currently `\bibitem` entries inline in `main.tex`
(8 entries). If switching to a `.bib` file with biblatex/natbib would
help future revisions, that's a 10-minute refactor. Not strictly
needed.

---

## Things tried that didn't work / are not in the final repo

- **Web scraper instead of dataset.** Initially I considered a live
  scraper feeding text to the model and discarding it. Was talked out
  of it in conversation: GPU would idle waiting for I/O, results would
  be irreproducible, and the data sizes involved ("terabytes") were a
  mirage — Wikipedia PT is ~5GB and HuggingFace serves it in streaming
  mode without ever materialising the full corpus to disk.
- **Single-book POC training.** The first run used Dom Casmurro alone
  and the dev set was the first 200KB of the same book. Loss descended
  but generalisation could not be measured. Replaced by the 6-book +
  disjoint-holdout setup.
- **bf16 / autocast on CPU.** Tried `dtype: bfloat16` initially; CPU
  doesn't support it reliably and emitted a warning every step. The
  training loop now silently skips `torch.autocast` when dtype is fp32.
- **CUDA xLSTM backend.** Tried briefly; failed at compile time on the
  laptop because the NVIDIA driver was too old. Fell back to the
  `vanilla` (PyTorch) backend, which is what all four trained runs use.

---

## How to verify everything still works

If returning to this project after a gap, sanity-check before doing
anything else:

```bash
# 1. Repository is in a sane state
python -m scripts.count_params --config configs/xlstm_tiny.yaml
python -m scripts.count_params --config configs/transformer_tiny.yaml --arch transformer

# 2. Logs parse and produce the expected numbers
python -m scripts.compare_runs \
    --runs runs/xlstm_tiny runs/xlstm_medium \
           runs/transformer_tiny runs/transformer_medium \
    --labels "xLSTM 1.5M" "xLSTM 4.5M" "Transformer 1.5M" "Transformer 4.2M" \
    --out /tmp/check.png

# Expected table:
# xLSTM 1.5M           best PPL  461.9
# xLSTM 4.5M           best PPL  229.9
# Transformer 1.5M     best PPL  332.5
# Transformer 4.2M     best PPL  235.6

# 3. Paper compiles
cd paper && pdflatex main.tex > /dev/null && pdflatex main.tex > /dev/null
ls -la main.pdf
```

If any of these fail, something in the environment changed; debug
before continuing.

---

## Style notes for whoever continues this (Claude Code or future self)

- Code style is **PyTorch-native, no Lightning, no Hydra**. OmegaConf
  for YAML loading, `dacite` to validate into dataclasses. Keep it
  that way for consistency.
- All scripts are **module entry points** (`python -m scripts.foo`),
  not loose `.py` files run with `python scripts/foo.py`. This matters
  because they import from `src.*` which expects the package to be
  installed in editable mode (`pip install -e .`).
- Comments and docstrings are in **a mix of Portuguese and English**.
  Prefer English for new code (the paper is in English, the
  codebase will eventually go on a public GitHub repo). Existing
  Portuguese comments are fine, no need to translate them
  retroactively.
- The xLSTM package version installed is whatever pip resolved when
  João first did `pip install -e .` — pinning was deferred. If
  reproducibility outside the laptop becomes important, freeze
  versions into `requirements.txt`.

---

## Contact-tone note

João is a Computer Engineering student at Insper, fluent in Portuguese
and English. Prior conversation was bilingual, lean toward Portuguese
in casual exchanges and English in things meant for the paper or wider
audience. Direct, concrete answers preferred over hand-holding;
willing to push back on suggestions and iterate.

---

*Handoff written April 2026. If this file is more than a few weeks
old when you read it, things may have moved.*

---

## Update — 2026-04-28 (parte 2): full re-run on GPU (tiny + medium + large)

Decisão: re-treinar TODAS as 6 configurações (xLSTM e Transformer ×
tiny/medium/large) na GPU do desktop sobre o corpus expandido e limpo.

**Mudanças adicionais nesse update:**

- Bug do `o_cortico.txt` (era *Roy Blakeley's Silver Fox Patrol* em
  inglês, id 43011) e do `o_primo_basilio.txt` (era *O Mandarim*,
  id 16384) corrigidos.
- Corpus expandido de 6 → 19 livros (~1.7MB → ~7.5MB, ~1.9M tokens),
  todos PT 19c verificados manualmente no PG. Lista em `configs/books.yaml`.
- Tokenizer: vocab subiu de 4000 → 8000 (corpus 4× maior justifica
  vocab 2× maior). Todos os configs de modelo foram atualizados.
- Configs `tiny` e `medium` migraram pra GPU: `dtype: bfloat16`,
  `batch_size: 32`, `num_workers: 4`, `slstm.backend: cuda`. O `medium`
  ganhou orçamento estendido (`total_steps: 4000`, `warmup_steps: 400`)
  pra sair do regime undertrained (best=final no laptop CPU).
- `run_desktop.sh` agora arquiva `runs/` antigo em `runs.cpu_backup/`
  na primeira execução, treina 6 modelos em sequência, e regenera a
  scaling curve com tiny+medium+large pra ambas as arquiteturas.
- `main.tex` §3.1 reescrito com a lista nova de 19 livros, 6 autores;
  abstract atualizada (`six` → `nineteen`); §Limitations §holdout
  ajustado (overlap "two of six" → "six of nineteen").
- Tabela 1 e Figura 1 ficarão obsoletas até o desktop terminar; depois,
  precisa atualizar com os números do `compare_runs.py` que o runner
  imprime no final.

Reverter pra CPU (caso necessário em outra máquina):
- `xlstm_tiny.yaml` e `xlstm_medium.yaml`: trocar `backend: cuda` por
  `backend: vanilla`.
- `train_*.yaml`: trocar `dtype: bfloat16` por `float32`,
  `batch_size: 32` por `4`, `num_workers: 4` por `0`,
  `compile: true` por `false` no Transformer.

---

## Update — 2026-04-28 (parte 1): large-scale desktop GPU run prepared

Mudanças desde o handoff original:

- **Logs limpos.** `scripts/clean_log.py` foi escrito e rodado;
  `xlstm_medium/log.jsonl` e `transformer_tiny/log.jsonl` perderam runs
  concatenadas (backup em `.bak`).
- **`scaling_curve.png` regenerado** — sem o artefato diagonal. Cópia
  na raiz (`./scaling_curve.png`) pro `pdflatex main.tex` consumir.
- **Placeholders do paper preenchidos** (autor + URL do GitHub).
- **Lista de livros externalizada** pra `configs/books.yaml`. Edite o
  YAML pra adicionar novos livros (não precisa mexer em código).
  `prepare_books.py` lê do YAML, é idempotente, tolera 404 individuais.
- **Configs `large` criadas** pra desktop GPU:
  - `configs/xlstm_large.yaml` — emb=512, 12 blocos, ctx=512, dois sLSTM
    (`slstm_at: [1,6]`), backend `cuda`. Estimativa ~12-25M params.
  - `configs/transformer_large.yaml` — emb=512, 8 blocos, ctx=512,
    `mlp_expansion=4`. Estimativa ~29M params.
  - `configs/train_large.yaml` + `configs/train_large_transformer.yaml`
    — bf16, batch 32, 12k steps, 1k warmup, lr 6e-4, eval a cada 500.
- **`scripts/run_desktop.sh`** roda o pipeline ponta-a-ponta no desktop:
  sanity GPU → prepare_books → tokenizer (se faltar) → xLSTM large →
  Transformer large → clean_log → scaling curve. Idempotente.

Pendências reais do desktop (que NÃO posso testar no laptop):

1. Verificar se o `xlstm` cuda backend compila na máquina (driver e
   PyTorch nightly precisam estar OK — `torch.cuda.is_available()` e
   compute capability >= 8.0 já são checados pelo runner).
2. Decidir se vale retreinar o tokenizer com vocab maior (8k ou 16k)
   se o corpus expandido. O runner mantém o `tokenizer.json` atual se
   ele já existir.
3. Compilar o PDF: `pdflatex main.tex && pdflatex main.tex` (TeXLive
   não estava instalado no laptop).

