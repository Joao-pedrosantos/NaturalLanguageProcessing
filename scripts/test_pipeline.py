"""End-to-end sanity check of the data pipeline.

Loads the trained tokenizer, builds the PT+EN streaming mix, wraps it in the
PackedLMDataset, and pulls a few batches through a DataLoader. Prints shapes
and decoded previews so you can eyeball that the tokens look right.

Usage:
    python -m scripts.test_pipeline
"""

from __future__ import annotations

import argparse
import time

from torch.utils.data import DataLoader

from src.data.streaming import (
    PackedLMDataset,
    build_streaming_wiki,
    collate_lm,
)
from src.data.tokenizer import load_tokenizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    print(f"tokenizer vocab: {tok.get_vocab_size()}")

    stream, probs = build_streaming_wiki(alpha=args.alpha)
    print(f"mixing probs: pt={probs[0]:.3f}, en={probs[1]:.3f}")

    ds = PackedLMDataset(stream, tok, seq_len=args.seq_len)
    # num_workers=0 keeps behavior deterministic for this smoke test.
    # For real training, 2–4 workers usually helps.
    loader = DataLoader(
        ds, batch_size=args.batch_size, collate_fn=collate_lm, num_workers=0
    )

    t0 = time.time()
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        total_tokens += x.numel()
        print(f"\nbatch {i}: x={tuple(x.shape)}, y={tuple(y.shape)}")
        preview = tok.decode(x[0, :80].tolist())
        print(f"  preview: {preview[:200]!r}")
        if i + 1 >= args.n_batches:
            break
    dt = time.time() - t0
    print(
        f"\ntotal: {total_tokens} tokens in {dt:.2f}s "
        f"({total_tokens / dt:,.0f} tok/s through the pipeline)"
    )
