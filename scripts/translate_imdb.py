"""Translate the original English IMDB dataset to Portuguese using a
neural MT model (MarianMT, transformer encoder-decoder).

This is the architecture Google deployed for production translation in
the post-GNMT period (~2017--2020): an attention-based seq2seq model.
We use the publicly available checkpoint Helsinki-NLP/opus-mt-tc-big-en-pt
distributed by the OPUS-MT project. The script samples a balanced subset
of stanfordnlp/imdb, translates each review sentence-by-sentence, and
writes one JSON line per review to a local file. The output is consumed
by scripts/probe_sentiment.py via --local-jsonl.

Usage:
    python -m scripts.translate_imdb \\
        --max-per-class 2000 --out artifacts/imdb_pt_marianmt.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """Cheap sentence splitter; further splits any chunk that is still
    too long for the MT model. NMT quality degrades on very long inputs,
    so we cap chunk length conservatively."""
    text = text.replace("<br />", " ").replace("<br>", " ")
    text = re.sub(r"\s+", " ", text).strip()
    sents = SENT_SPLIT_RE.split(text)
    out: list[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def translate_batch(
    texts: list[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    device: torch.device,
    max_length: int = 512,
) -> list[str]:
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        gen = model.generate(
            **enc, max_length=max_length, num_beams=1,
            no_repeat_ngram_size=3,
        )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="Helsinki-NLP/opus-mt-tc-big-en-pt",
        help="MarianMT en->pt checkpoint",
    )
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument(
        "--sentence-batch-size", type=int, default=16,
        help="how many sentences to translate in a single forward pass",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", default="artifacts/imdb_pt_marianmt.jsonl",
        help="output JSONL: one {'texto', 'sentimento'} object per line",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading {args.model}...")
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).to(device).eval()

    print("loading stanfordnlp/imdb...")
    ds = load_dataset("stanfordnlp/imdb")["train"]

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[int, int] = defaultdict(int)
    target = args.max_per_class
    n_total = 2 * target

    with out_path.open("w", encoding="utf-8") as f_out:
        n_written = 0
        for i in indices:
            row = ds[i]
            y = int(row["label"])
            if counts[y] >= target:
                if all(counts[c] >= target for c in (0, 1)):
                    break
                continue

            sents = split_sentences(row["text"])
            translated: list[str] = []
            for s_start in range(0, len(sents), args.sentence_batch_size):
                batch = sents[s_start : s_start + args.sentence_batch_size]
                translated.extend(
                    translate_batch(batch, model, tokenizer, device)
                )
            pt_text = " ".join(translated)

            f_out.write(json.dumps(
                {"texto": pt_text, "sentimento": y},
                ensure_ascii=False,
            ) + "\n")
            counts[y] += 1
            n_written += 1
            if n_written % 25 == 0:
                print(
                    f"  [{n_written}/{n_total}] "
                    f"neg={counts[0]} pos={counts[1]}"
                )

    print(f"\ndone. wrote {n_written} translations to {out_path}")
    print(f"  negative: {counts[0]}")
    print(f"  positive: {counts[1]}")


if __name__ == "__main__":
    main()
