"""Train a byte-level BPE tokenizer on the sampled corpus.

Usage:
    python -m scripts.train_tokenizer --vocab-size 32000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.tokenizer import train_tokenizer


def read_lines(paths: list[Path]):
    """Yield non-empty lines from each file, in order."""
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="artifacts/tokenizer_corpus")
    ap.add_argument("--vocab-size", type=int, default=32_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="artifacts/tokenizer.json")
    args = ap.parse_args()

    paths = sorted(Path(args.corpus_dir).glob("*.txt"))
    if not paths:
        raise SystemExit(
            f"no .txt files in {args.corpus_dir}. "
            f"run `python -m scripts.sample_for_tokenizer` first."
        )
    print(f"training BPE on: {[str(p) for p in paths]}")

    tok = train_tokenizer(
        read_lines(paths),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_path=args.out,
    )
    print(f"vocab size: {tok.get_vocab_size()}")
    print(f"saved to:   {args.out}")

    # Quick smoke test on a mixed sample.
    sample = (
        "A inteligência artificial transforma a sociedade. "
        "Artificial intelligence is transforming society."
    )
    enc = tok.encode(sample)
    print(f"\nsample: {sample!r}")
    print(f"  ids (first 30): {enc.ids[:30]}")
    print(f"  round-trip:     {tok.decode(enc.ids)!r}")
