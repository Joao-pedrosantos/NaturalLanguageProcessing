"""Linear probe for binary sentiment on top of frozen LM hidden states.

Mirrors scripts/probe_classify.py (author probe), but the task is binary
sentiment classification on IMDB reviews translated to Portuguese
(celsowm/imdb-reviews-pt-br on the Hugging Face Hub).

Each review is tokenised with our byte-level BPE, truncated to seq_len
from the start (one feature vector per review), passed through the
frozen LM, mean-pooled over the time axis, and fed to a single
nn.Linear(D, 2) trained with cross-entropy.

Usage:
    python -m scripts.probe_sentiment \\
        --ckpt runs-gpu/xlstm_medium/best.pt \\
        --config configs/xlstm_medium.yaml \\
        --arch xlstm
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tokenizer import load_tokenizer
from src.model_factory import build_model_for_arch
from src.training import load_checkpoint
from scripts.probe_classify import (
    extract_features,
    stratified_split,
    train_probe,
)


def _iter_imdb_pt(
    local_jsonl: str | None,
) -> "list | object":
    """Return an iterable of {'texto', 'sentimento'} rows. If
    local_jsonl is given, read from disk; otherwise download
    celsowm/imdb-reviews-pt-br from the Hub."""
    if local_jsonl:
        rows = []
        with open(local_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    from datasets import load_dataset
    return load_dataset("celsowm/imdb-reviews-pt-br")["train"]


def load_imdb_pt(
    tokenizer,
    seq_len: int,
    max_per_class: int,
    seed: int,
    local_jsonl: str | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Tokenise IMDB-pt reviews, truncate/pad to seq_len, cap at
    max_per_class per class. Source is either the celsowm Hub dataset
    (default) or a local JSONL produced by scripts/translate_imdb.py."""
    ds = _iter_imdb_pt(local_jsonl)
    pad_id = 0  # byte-level BPE: id 0 is unused for content; safe as pad

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    passages: list[list[int]] = []
    labels: list[int] = []
    counts: dict[int, int] = defaultdict(int)
    for i in indices:
        row = ds[i]
        y = int(row["sentimento"])
        if counts[y] >= max_per_class:
            if all(counts[c] >= max_per_class for c in (0, 1)):
                break
            continue
        ids = tokenizer.encode(row["texto"]).ids
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [pad_id] * (seq_len - len(ids))
        passages.append(ids)
        labels.append(y)
        counts[y] += 1
    return passages, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--arch", default="xlstm", choices=["xlstm", "transformer"],
    )
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument(
        "--max-per-class", type=int, default=2000,
        help="cap of reviews per sentiment class (balances dataset and "
             "keeps feature extraction tractable on CPU)",
    )
    ap.add_argument(
        "--local-jsonl", default=None,
        help="path to a JSONL produced by scripts/translate_imdb.py "
             "(one {'texto', 'sentimento'} per line). If omitted, "
             "loads celsowm/imdb-reviews-pt-br from the Hub.",
    )
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--probe-epochs", type=int, default=200)
    ap.add_argument("--feat-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", default=None,
        help="JSON output path (default: stdout only)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = load_tokenizer(args.tokenizer)

    src = args.local_jsonl or "celsowm/imdb-reviews-pt-br"
    print(f"\nloading IMDB pt-BR ({src})...")
    passages, all_labels = load_imdb_pt(
        tokenizer, args.seq_len, args.max_per_class, args.seed,
        local_jsonl=args.local_jsonl,
    )
    counts = Counter(all_labels)
    print(f"total passages: {len(passages)}")
    print(f"  negative (0): {counts[0]:>5d}")
    print(f"  positive (1): {counts[1]:>5d}")

    print("\nloading model...")
    model = build_model_for_arch(args.arch, args.config).to(device)
    load_checkpoint(Path(args.ckpt), model, map_location=device)

    print("\nextracting features...")
    feats = extract_features(
        model, args.arch, passages, device,
        batch_size=args.feat_batch_size,
    )
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"feature shape: {tuple(feats.shape)}")

    train_idx, test_idx = stratified_split(
        all_labels, args.frac_train, args.seed,
    )
    X_train, y_train = feats[train_idx], labels[train_idx]
    X_test, y_test = feats[test_idx], labels[test_idx]

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"\ntraining linear probe ({len(train_idx)} train / "
          f"{len(test_idx)} test)...")
    metrics = train_probe(
        X_train, y_train, X_test, y_test,
        n_classes=2,
        n_epochs=args.probe_epochs,
        device=device,
    )

    majority = max(counts.values()) / sum(counts.values())

    print(f"\n=== {args.arch} sentiment-probe results ===")
    print(f"  train accuracy:          {metrics['train_acc']*100:.2f}%")
    print(f"  test  accuracy:          {metrics['test_acc']*100:.2f}%")
    print(f"  test  macro-F1:          {metrics['macro_f1']*100:.2f}%")
    print(f"  majority-class baseline: {majority*100:.2f}%")
    print(f"  per-class F1: neg={metrics['per_class_f1'][0]*100:.2f}% "
          f"pos={metrics['per_class_f1'][1]*100:.2f}%")
    print(f"  confusion (rows=true, cols=pred): {metrics['confusion']}")

    record = dict(
        arch=args.arch,
        ckpt=str(args.ckpt),
        config=str(args.config),
        dataset=(args.local_jsonl or "celsowm/imdb-reviews-pt-br"),
        seq_len=args.seq_len,
        max_per_class=args.max_per_class,
        n_classes=2,
        class_names=["negative", "positive"],
        n_passages_total=len(passages),
        n_train=len(train_idx),
        n_test=len(test_idx),
        majority_baseline=majority,
        **metrics,
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
