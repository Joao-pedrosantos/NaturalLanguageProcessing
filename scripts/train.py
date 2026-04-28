"""Main training entry point: wires config, data, model, and trainer.

Usage:
    python -m scripts.train --config configs/train_50m.yaml
    python -m scripts.train --config configs/train_50m.yaml --resume runs/xlstm_50m/final.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.eval_dataset import FileEvalDataset
from src.data.streaming import (
    PackedLMDataset,
    build_streaming_wiki,
    collate_lm,
)
from src.data.tokenizer import load_tokenizer
from src.model import build_model, count_parameters, format_params, load_config
from src.training import TrainState, evaluate, load_checkpoint, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_eval_fn(
    path: Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    model,
    device,
    dtype: torch.dtype,
    max_batches: int,
) -> Callable[[], dict]:
    """Build a zero-arg callable that recomputes eval metrics on-demand.

    Re-creating the FileEvalDataset every call is intentional — it makes
    eval fully deterministic (same chunks, same order, every time).
    """
    def _eval() -> dict:
        ds = FileEvalDataset(path, tokenizer, seq_len=seq_len)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_lm)
        return evaluate(model, loader, device, dtype, max_batches=max_batches)
    return _eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_50m.yaml")
    ap.add_argument("--resume", default=None, help="path to checkpoint")
    ap.add_argument("--pt-dev", default="artifacts/dev/pt_dev.txt")
    ap.add_argument("--en-dev", default="artifacts/dev/en_dev.txt")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # ---- tokenizer + data ----
    tokenizer = load_tokenizer(cfg["tokenizer"])
    print(f"tokenizer vocab: {tokenizer.get_vocab_size()}")

    stream, probs = build_streaming_wiki(
        alpha=cfg["alpha"],
        shuffle_buffer=cfg["shuffle_buffer"],
        seed=cfg["seed"],
    )
    print(f"mixing probs: pt={probs[0]:.3f}, en={probs[1]:.3f}")

    train_ds = PackedLMDataset(stream, tokenizer, seq_len=cfg["seq_len"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        collate_fn=collate_lm,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- model ----
    model_cfg = load_config(cfg["model_config"])
    # Align context length between model and data loader.
    assert model_cfg.context_length >= cfg["seq_len"], (
        f"model context_length ({model_cfg.context_length}) must be "
        f">= train seq_len ({cfg['seq_len']})"
    )
    # Align vocab.
    assert model_cfg.vocab_size == tokenizer.get_vocab_size(), (
        f"model vocab_size ({model_cfg.vocab_size}) != tokenizer vocab "
        f"({tokenizer.get_vocab_size()}). Re-train tokenizer or fix config."
    )

    model = build_model(model_cfg).to(device)
    stats = count_parameters(model)
    print(f"model params: {format_params(stats['total'])} "
          f"({stats['total']:,})")

    if cfg.get("compile", False):
        print("compiling model with torch.compile...")
        model = torch.compile(model)

    # ---- eval fns (per language) ----
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["dtype"]]
    eval_batch = max(1, cfg["batch_size"] // 2)
    eval_fns = {}
    if Path(args.pt_dev).exists():
        eval_fns["pt_dev"] = make_eval_fn(
            Path(args.pt_dev), tokenizer, cfg["seq_len"],
            eval_batch, model, device, dtype, cfg["eval_max_batches"],
        )
    if Path(args.en_dev).exists():
        eval_fns["en_dev"] = make_eval_fn(
            Path(args.en_dev), tokenizer, cfg["seq_len"],
            eval_batch, model, device, dtype, cfg["eval_max_batches"],
        )
    if not eval_fns:
        print("WARNING: no dev sets found — run prepare_eval_set first.")

    # ---- resume ----
    resume_state = None
    if args.resume:
        print(f"resuming from {args.resume}")
        # Build optimizer/scheduler inside train() — just load model weights
        # and the TrainState here. We'll re-initialize optim from scratch;
        # for exact resumption, use the same path in src/training.py
        # load_checkpoint with optimizer/scheduler handles.
        ckpt = load_checkpoint(Path(args.resume), model, map_location=device)
        resume_state = TrainState(
            step=ckpt["step"],
            best_val=ckpt["best_val"],
        )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save resolved config alongside the run for reproducibility.
    OmegaConf.save(OmegaConf.create(cfg), out_dir / "train_config.yaml")

    train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        eval_fns=eval_fns or None,
        out_dir=out_dir,
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main()
