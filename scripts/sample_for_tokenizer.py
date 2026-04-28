"""Sample a bounded text subset for tokenizer training.

Reasoning: the BPE trainer needs to see the corpus (it computes merge
frequencies). Streaming one-shot iterators are awkward for trainers that
may want to iterate multiple times internally, and we also want the
tokenizer corpus to be fixed and reproducible across runs.

So we stream from HF once, dump ~N MB per language to plain .txt files,
and train the tokenizer on those files. The rest of the pre-training
can still use pure streaming — only the tokenizer corpus is materialized.

Usage:
    python -m scripts.sample_for_tokenizer --target-mb 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def dump(config: str, out: Path, target_mb: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True
    )
    target_bytes = target_mb * 1024 * 1024
    written = 0
    with out.open("w", encoding="utf-8") as f:
        pbar = tqdm(ds, desc=f"sampling {config}", unit=" articles")
        for sample in pbar:
            text = sample.get("text") or ""
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            written += len(text.encode("utf-8"))
            pbar.set_postfix(mb=f"{written / 1024 / 1024:.1f}")
            if written >= target_bytes:
                break
    print(f"wrote {written / 1024 / 1024:.1f} MB to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="20231101")
    ap.add_argument(
        "--target-mb", type=int, default=100,
        help="target MB of text per language",
    )
    ap.add_argument("--out-dir", default="artifacts/tokenizer_corpus")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    dump(f"{args.snapshot}.pt", out_dir / "pt.txt", args.target_mb)
    dump(f"{args.snapshot}.en", out_dir / "en.txt", args.target_mb)
