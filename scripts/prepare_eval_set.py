"""Prepare small fixed dev sets for per-language perplexity eval.

Dumps ~5MB of text per language to artifacts/dev/. These are sampled from
the same streaming dump as training — in a proper research setup you'd
want a disjoint split, but for a class project the overlap risk is
minimal given we only materialize ~5MB vs. billions of tokens seen
during training.

Usage:
    python -m scripts.prepare_eval_set --target-mb 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def dump(config: str, out: Path, target_mb: int, skip: int) -> None:
    """Dump target_mb of text to `out`, skipping the first `skip` articles
    (to avoid overlap with the tokenizer corpus which samples from the
    beginning of the stream)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True
    )
    target_bytes = target_mb * 1024 * 1024
    written = 0
    seen = 0
    with out.open("w", encoding="utf-8") as f:
        pbar = tqdm(ds, desc=f"eval set {config}", unit=" articles")
        for sample in pbar:
            seen += 1
            if seen <= skip:
                continue
            text = sample.get("text") or ""
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            written += len(text.encode("utf-8"))
            pbar.set_postfix(mb=f"{written / 1024 / 1024:.2f}")
            if written >= target_bytes:
                break
    print(f"wrote {written / 1024 / 1024:.2f} MB to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="20231101")
    ap.add_argument("--target-mb", type=int, default=5)
    ap.add_argument(
        "--skip", type=int, default=50_000,
        help="skip first N articles (should exceed tokenizer-corpus range)",
    )
    ap.add_argument("--out-dir", default="artifacts/dev")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    dump(f"{args.snapshot}.pt", out_dir / "pt_dev.txt",
         args.target_mb, args.skip)
    dump(f"{args.snapshot}.en", out_dir / "en_dev.txt",
         args.target_mb, args.skip)
