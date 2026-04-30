"""Linear probe for author classification on top of frozen LM hidden states.

Standard downstream eval: freeze the trained LM, extract mean-pooled
hidden states for fixed-length passages, train a single linear layer to
predict the author. Gives a comparable scalar (accuracy + macro-F1)
across architectures that does not depend on perplexity calibration.

The probe is a fair common protocol because:
  - the same tokenizer + same passages are used across architectures
  - only the final linear layer is trained (one optim run per arch)
  - macro-F1 is reported in addition to accuracy to handle the class
    imbalance from Machado de Assis having more books than other authors

Usage:
    python -m scripts.probe_classify \\
        --ckpt runs/xlstm_medium/best.pt \\
        --config configs/xlstm_medium_legacy.yaml \\
        --arch xlstm

    python -m scripts.probe_classify \\
        --ckpt runs/transformer_medium/best.pt \\
        --config configs/transformer_medium_legacy.yaml \\
        --arch transformer

Output: a JSON line with all metrics, plus per-author confusion summary.
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
from omegaconf import OmegaConf

from src.data.tokenizer import load_tokenizer
from src.model_factory import build_model_for_arch
from src.training import load_checkpoint


def passage_features_xlstm(model, ids: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, D). Mean-pool over the sequence axis of the
    block-stack output (after the post-blocks norm)."""
    h = model.token_embedding(ids)
    h = model.emb_dropout(h)
    h = model.xlstm_block_stack(h)
    return h.mean(dim=1)


def passage_features_transformer(model, ids: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, D). Mean-pool over the sequence axis of the
    pre-head hidden state (after the final layer norm)."""
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device)
    h = model.tok_emb(ids) + model.pos_emb(pos)[None, :, :]
    h = model.drop(h)
    for block in model.blocks:
        h = block(h)
    h = model.ln_f(h)
    return h.mean(dim=1)


def load_books_authors(books_yaml: Path) -> dict[str, str]:
    """Map filename -> author from configs/books.yaml."""
    cfg = OmegaConf.to_container(OmegaConf.load(str(books_yaml)), resolve=True)
    return {entry["file"]: entry["author"] for entry in cfg.get("train", [])}


def passages_from_book(
    text: str, tokenizer, seq_len: int, stride: int,
) -> list[list[int]]:
    """Tokenise once, slide a window of seq_len with given stride.
    Drops the trailing partial window."""
    ids = tokenizer.encode(text).ids
    out = []
    for start in range(0, len(ids) - seq_len + 1, stride):
        out.append(ids[start : start + seq_len])
    return out


@torch.no_grad()
def extract_features(
    model,
    arch: str,
    passages: list[list[int]],
    device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Run all passages through the model and return (N, D) features."""
    model.eval()
    feat_fn = (
        passage_features_xlstm if arch == "xlstm"
        else passage_features_transformer
    )
    feats = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        x = torch.tensor(batch, dtype=torch.long, device=device)
        f = feat_fn(model, x)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def stratified_split(
    labels: list[int], frac_train: float, seed: int,
) -> tuple[list[int], list[int]]:
    """Per-class 80/20 split of indices."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    train, test = [], []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        cut = max(1, int(round(frac_train * len(idxs))))
        train.extend(idxs[:cut])
        test.extend(idxs[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def train_probe(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    n_classes: int, lr: float = 1.0e-2,
    weight_decay: float = 1.0e-3, n_epochs: int = 200,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train a single Linear layer with cross-entropy. Returns metrics
    and the (n_classes x n_classes) confusion matrix."""
    D = X_train.shape[1]
    probe = nn.Linear(D, n_classes).to(device)
    opt = torch.optim.AdamW(
        probe.parameters(), lr=lr, weight_decay=weight_decay,
    )
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for _ in range(n_epochs):
        opt.zero_grad()
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test)
        test_preds = test_logits.argmax(dim=-1)
        train_preds = probe(X_train).argmax(dim=-1)

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    # Macro-F1
    f1s = []
    for c in range(n_classes):
        tp = ((test_preds == c) & (y_test == c)).sum().item()
        fp = ((test_preds == c) & (y_test != c)).sum().item()
        fn = ((test_preds != c) & (y_test == c)).sum().item()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    # Confusion matrix (rows = true, cols = predicted)
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y_test.cpu().tolist(), test_preds.cpu().tolist()):
        confusion[t, p] += 1

    return dict(
        train_acc=train_acc,
        test_acc=test_acc,
        macro_f1=macro_f1,
        per_class_f1=f1s,
        confusion=confusion.tolist(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--arch", default="xlstm", choices=["xlstm", "transformer"],
    )
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--books-yaml", default="configs/books.yaml")
    ap.add_argument("--books-dir", default="artifacts/train_books")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max-passages-per-author", type=int, default=400)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--probe-epochs", type=int, default=200)
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
    file_to_author = load_books_authors(Path(args.books_yaml))
    authors = sorted(set(file_to_author.values()))
    author_to_id = {a: i for i, a in enumerate(authors)}
    print(f"authors ({len(authors)}): {authors}")

    print("\nbuilding passages...")
    all_passages: list[list[int]] = []
    all_labels: list[int] = []
    by_author = defaultdict(int)
    for fname, author in file_to_author.items():
        path = Path(args.books_dir) / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        ps = passages_from_book(text, tokenizer, args.seq_len, args.stride)
        for p in ps:
            if by_author[author] >= args.max_passages_per_author:
                break
            all_passages.append(p)
            all_labels.append(author_to_id[author])
            by_author[author] += 1
    counts = Counter(all_labels)
    print(f"total passages: {len(all_passages)}")
    for a, i in author_to_id.items():
        print(f"  {a:30s}  {counts.get(i, 0):>5d} passages")

    print("\nloading model...")
    model = build_model_for_arch(args.arch, args.config).to(device)
    load_checkpoint(Path(args.ckpt), model, map_location=device)

    print("\nextracting features...")
    feats = extract_features(
        model, args.arch, all_passages, device, batch_size=8,
    )
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"feature shape: {tuple(feats.shape)}")

    train_idx, test_idx = stratified_split(
        all_labels, args.frac_train, args.seed,
    )
    X_train, y_train = feats[train_idx], labels[train_idx]
    X_test, y_test = feats[test_idx], labels[test_idx]

    # Standardize features (mean/std from train only).
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"\ntraining linear probe ({len(train_idx)} train / "
          f"{len(test_idx)} test)...")
    metrics = train_probe(
        X_train, y_train, X_test, y_test,
        n_classes=len(authors),
        n_epochs=args.probe_epochs,
        device=device,
    )

    # Majority-class baseline for sanity.
    majority = max(counts.values()) / sum(counts.values())

    print(f"\n=== {args.arch} probe results ===")
    print(f"  train accuracy:        {metrics['train_acc']*100:.2f}%")
    print(f"  test  accuracy:        {metrics['test_acc']*100:.2f}%")
    print(f"  test  macro-F1:        {metrics['macro_f1']*100:.2f}%")
    print(f"  majority-class baseline: {majority*100:.2f}%")
    print()
    print("  per-author F1:")
    for a, f1 in zip(authors, metrics["per_class_f1"]):
        print(f"    {a:30s}  F1={f1*100:.2f}%")

    record = dict(
        arch=args.arch,
        ckpt=str(args.ckpt),
        config=str(args.config),
        seq_len=args.seq_len,
        stride=args.stride,
        n_authors=len(authors),
        authors=authors,
        n_passages_total=len(all_passages),
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
