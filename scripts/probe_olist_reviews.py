"""Linear probe for review-score classification on top of frozen LM
hidden states, using the Olist e-commerce reviews dataset (PT-BR).

Mirrors scripts/probe_sentiment.py: each review_comment_message is
tokenised with our byte-level BPE, padded/truncated to seq_len, passed
through the frozen LM, mean-pooled over the time axis, and fed to a
single nn.Linear(D, n_classes) trained with cross-entropy.

Two label modes are supported:
  - binary (default): drop score=3 (ambiguous), label 1-2 -> 0 (neg),
    4-5 -> 1 (pos). Directly comparable to the IMDB sentiment probe.
  - multiclass: keep all five scores 1..5 as classes 0..4. Harder; lets
    us see whether representations capture review intensity, not just
    polarity.

Usage:
    python -m scripts.probe_olist_reviews \\
        --ckpt runs-gpu/xlstm_medium/best.pt \\
        --config configs/xlstm_medium.yaml \\
        --arch xlstm

    python -m scripts.probe_olist_reviews \\
        --ckpt runs-gpu/xlstm_medium/best.pt \\
        --config configs/xlstm_medium.yaml \\
        --arch xlstm --multiclass
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src.data.tokenizer import load_tokenizer
from src.model_factory import build_model_for_arch
from src.training import load_checkpoint
from scripts.probe_classify import (
    extract_features,
    stratified_split,
    train_probe,
)


def _read_olist_reviews(csv_path: Path) -> list[tuple[str, int]]:
    """Return (message, review_score) pairs with non-empty messages."""
    rows: list[tuple[str, int]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            msg = (row.get("review_comment_message") or "").strip()
            if not msg:
                continue
            try:
                score = int(row["review_score"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((msg, score))
    return rows


def _to_label(score: int, multiclass: bool) -> int | None:
    """Map a 1..5 score to a class id, or None to drop the row."""
    if multiclass:
        return score - 1  # 1..5 -> 0..4
    if score == 3:
        return None  # drop neutral in binary mode
    return 0 if score <= 2 else 1


def load_olist_reviews(
    csv_path: Path,
    tokenizer,
    seq_len: int,
    max_per_class: int,
    multiclass: bool,
    seed: int,
) -> tuple[list[list[int]], list[int], list[str]]:
    """Tokenise + pad/truncate to seq_len; cap at max_per_class per class.

    Returns (passages, labels, class_names)."""
    raw = _read_olist_reviews(csv_path)

    rng = random.Random(seed)
    rng.shuffle(raw)

    pad_id = 0  # byte-level BPE: id 0 is unused for content; safe as pad

    n_classes = 5 if multiclass else 2
    class_names = (
        ["1", "2", "3", "4", "5"] if multiclass else ["negative", "positive"]
    )

    passages: list[list[int]] = []
    labels: list[int] = []
    counts: dict[int, int] = defaultdict(int)
    target_total = max_per_class * n_classes

    for msg, score in raw:
        y = _to_label(score, multiclass)
        if y is None:
            continue
        if counts[y] >= max_per_class:
            if all(counts[c] >= max_per_class for c in range(n_classes)):
                break
            continue
        ids = tokenizer.encode(msg).ids
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [pad_id] * (seq_len - len(ids))
        passages.append(ids)
        labels.append(y)
        counts[y] += 1
        if len(passages) >= target_total:
            break

    return passages, labels, class_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--arch", default="xlstm", choices=["xlstm", "transformer"],
    )
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument(
        "--reviews-csv",
        default="olist/olist_order_reviews_dataset.csv",
    )
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument(
        "--max-per-class", type=int, default=2000,
        help="cap of reviews per class (balances dataset and keeps "
             "feature extraction tractable on CPU)",
    )
    ap.add_argument(
        "--multiclass", action="store_true",
        help="keep all five scores as separate classes (default: drop "
             "score=3 and binarize 1-2 vs 4-5)",
    )
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--probe-epochs", type=int, default=200)
    ap.add_argument("--feat-batch-size", type=int, default=8)
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="one or more seeds; each seed = full re-run "
             "(re-sample data, re-split, re-init probe). Mean/std "
             "across seeds is reported when more than one is given.",
    )
    ap.add_argument(
        "--out", default=None,
        help="JSON output path (default: stdout only)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = load_tokenizer(args.tokenizer)

    print("\nloading model...")
    model = build_model_for_arch(args.arch, args.config).to(device)
    load_checkpoint(Path(args.ckpt), model, map_location=device)

    mode = "multiclass (1..5)" if args.multiclass else "binary (1-2 vs 4-5)"
    print(f"label mode = {mode}")

    runs: list[dict] = []
    majority_baseline: float | None = None
    n_classes_run: int | None = None
    class_names_run: list[str] | None = None

    for seed in args.seeds:
        print(f"\n========== seed {seed} ==========")
        random.seed(seed)
        torch.manual_seed(seed)

        print(f"loading Olist reviews ({args.reviews_csv})...")
        passages, all_labels, class_names = load_olist_reviews(
            Path(args.reviews_csv), tokenizer, args.seq_len,
            args.max_per_class, args.multiclass, seed,
        )
        counts = Counter(all_labels)
        print(f"total passages: {len(passages)}")
        for i, name in enumerate(class_names):
            print(f"  {name:>10s} ({i}): {counts.get(i, 0):>5d}")

        print("extracting features...")
        t0 = time.perf_counter()
        feats = extract_features(
            model, args.arch, passages, device,
            batch_size=args.feat_batch_size,
        )
        feat_time_s = time.perf_counter() - t0
        labels = torch.tensor(all_labels, dtype=torch.long)

        train_idx, test_idx = stratified_split(
            all_labels, args.frac_train, seed,
        )
        X_train, y_train = feats[train_idx], labels[train_idx]
        X_test, y_test = feats[test_idx], labels[test_idx]

        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        n_classes = len(class_names)
        print(f"training probe ({len(train_idx)} train / "
              f"{len(test_idx)} test, {n_classes} classes)...")
        t0 = time.perf_counter()
        metrics = train_probe(
            X_train, y_train, X_test, y_test,
            n_classes=n_classes,
            n_epochs=args.probe_epochs,
            device=device,
        )
        probe_time_s = time.perf_counter() - t0

        majority = max(counts.values()) / sum(counts.values())
        majority_baseline = majority
        n_classes_run = n_classes
        class_names_run = class_names

        feat_throughput = len(passages) / feat_time_s if feat_time_s > 0 else 0.0
        print(f"  test acc = {metrics['test_acc']*100:.2f}%  "
              f"macro-F1 = {metrics['macro_f1']*100:.2f}%  "
              f"feat={feat_time_s:.2f}s ({feat_throughput:.1f} pas/s)  "
              f"probe={probe_time_s:.2f}s")

        runs.append(dict(
            seed=seed,
            n_passages_total=len(passages),
            n_train=len(train_idx),
            n_test=len(test_idx),
            feat_time_s=feat_time_s,
            feat_throughput_pas_per_s=feat_throughput,
            probe_time_s=probe_time_s,
            **metrics,
        ))

    def _agg(key: str) -> dict:
        vals = [r[key] for r in runs]
        return dict(
            mean=statistics.mean(vals),
            std=(statistics.stdev(vals) if len(vals) > 1 else 0.0),
            values=vals,
        )

    aggregate = dict(
        test_acc=_agg("test_acc"),
        train_acc=_agg("train_acc"),
        macro_f1=_agg("macro_f1"),
        feat_time_s=_agg("feat_time_s"),
        feat_throughput_pas_per_s=_agg("feat_throughput_pas_per_s"),
    )

    print(f"\n=== {args.arch} olist-review-probe results ({mode}) ===")
    print(f"  seeds: {args.seeds}  device: {device}")
    print(f"  test  accuracy:  "
          f"{aggregate['test_acc']['mean']*100:.2f}% "
          f"± {aggregate['test_acc']['std']*100:.2f}%")
    print(f"  test  macro-F1:  "
          f"{aggregate['macro_f1']['mean']*100:.2f}% "
          f"± {aggregate['macro_f1']['std']*100:.2f}%")
    print(f"  train accuracy:  "
          f"{aggregate['train_acc']['mean']*100:.2f}% "
          f"± {aggregate['train_acc']['std']*100:.2f}%")
    print(f"  feat-extract:    "
          f"{aggregate['feat_time_s']['mean']:.2f}s "
          f"± {aggregate['feat_time_s']['std']:.2f}s "
          f"({aggregate['feat_throughput_pas_per_s']['mean']:.1f} pas/s)")
    print(f"  majority-class baseline: {majority_baseline*100:.2f}%")
    print("  per-seed test acc: " + ", ".join(
        f"{r['seed']}={r['test_acc']*100:.2f}%" for r in runs
    ))

    record = dict(
        arch=args.arch,
        ckpt=str(args.ckpt),
        config=str(args.config),
        dataset=str(args.reviews_csv),
        label_mode=("multiclass" if args.multiclass else "binary"),
        seq_len=args.seq_len,
        max_per_class=args.max_per_class,
        n_classes=n_classes_run,
        class_names=class_names_run,
        majority_baseline=majority_baseline,
        seeds=args.seeds,
        runs=runs,
        aggregate=aggregate,
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
