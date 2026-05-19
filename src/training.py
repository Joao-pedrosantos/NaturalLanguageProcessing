"""Training utilities for xLSTM LM pre-training.

Contains: optimizer builder (weight decay param groups), cosine schedule
with linear warmup, evaluation (perplexity), checkpoint save/load, and
the main training loop with mixed-precision, gradient accumulation, and
gradient clipping.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _unwrap(model: nn.Module) -> nn.Module:
    """Retorna o módulo interno, removendo o wrapper DDP se presente."""
    return model.module if hasattr(model, "module") else model


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    """Retorna context manager: autocast quando faz sentido, no-op caso contrário.

    fp32 não precisa de autocast (vira no-op com warning em CPU).
    Em CPU, autocast só suporta bf16/fp16 — em fp32 retornamos nullcontext
    pra evitar o aviso "CPU autocast disabled" que polui o log.
    """
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# -------------------- optimizer + schedule --------------------

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> AdamW:
    """AdamW with weight decay only on 2D+ tensors (weights, not biases/norms).

    This is the standard convention from Chinchilla / GPT-NeoX: weight
    decay on matmul weights, none on biases, layer-norm scales/biases,
    and 1D tensors.
    """
    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr, betas=betas)


def cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup then cosine decay to `min_lr_ratio * lr`."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# -------------------- evaluation --------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: Optional[int] = None,
) -> dict:
    """Compute average cross-entropy loss and perplexity on `loader`.

    Returns {"loss": float, "ppl": float, "n_tokens": int}.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast_ctx(device, dtype):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    avg = total_loss / max(1, total_tokens)
    return {
        "loss": avg,
        "ppl": math.exp(min(avg, 20)),  # clamp to avoid overflow early in training
        "n_tokens": total_tokens,
    }


# -------------------- checkpointing --------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    best_val: float,
    extra: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_val": best_val,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    map_location: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def prune_old_checkpoints(out_dir: Path, keep_last: int) -> None:
    """Keep only the most recent `keep_last` step-checkpoints. Preserves
    any file named `best.pt`."""
    step_ckpts = sorted(
        out_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for p in step_ckpts[:-keep_last]:
        p.unlink(missing_ok=True)


# -------------------- training loop --------------------

@dataclass
class TrainState:
    step: int = 0
    best_val: float = float("inf")
    start_time: float = field(default_factory=time.time)
    tokens_seen: int = 0


def train(
    model: nn.Module,
    train_loader: Iterable,
    cfg: dict,
    device: torch.device,
    eval_fns: Optional[dict[str, Callable[[], dict]]] = None,
    out_dir: Path = Path("runs/default"),
    resume_state: Optional[TrainState] = None,
    rank: int = 0,
) -> TrainState:
    """Main training loop.

    Parameters
    ----------
    model : xLSTMLMModel already on `device`.
    train_loader : yields (inputs, targets) both (B, T) long tensors.
    cfg : training config dict (see configs/train_50m.yaml).
    device : cuda or cpu device.
    eval_fns : optional dict mapping a name (e.g. "pt_dev") to a zero-arg
        callable that returns a metrics dict. Called every `eval_every`
        steps. The first entry drives best-checkpoint selection.
    out_dir : where checkpoints and log.jsonl are written.
    resume_state : to continue a run, pass the TrainState loaded from ckpt.
    """
    is_main = rank == 0
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    # --- optim / schedule ---
    optimizer = build_optimizer(
        model,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["betas"]),
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=cfg["warmup_steps"],
        total_steps=cfg["total_steps"],
        min_lr_ratio=cfg["min_lr_ratio"],
    )

    # --- precision ---
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["dtype"]]
    # bf16 does not need a GradScaler; fp16 would.
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler(device=device.type) if use_scaler else None

    state = resume_state or TrainState()
    accum = cfg["grad_accum_steps"]
    log_every = cfg["log_every"]
    eval_every = cfg["eval_every"]
    save_every = cfg["save_every"]
    total_steps = cfg["total_steps"]

    model.train()
    optimizer.zero_grad(set_to_none=True)

    micro_step = 0
    loss_accum = 0.0
    t_last_log = time.time()
    tokens_since_log = 0

    it = iter(train_loader)
    while state.step < total_steps:
        try:
            x, y = next(it)
        except StopIteration:
            # Streaming dataset exhausted (unlikely before `total_steps`
            # at this scale, but handle gracefully).
            it = iter(train_loader)
            x, y = next(it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with _autocast_ctx(device, dtype):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            loss = loss / accum

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_accum += loss.item()
        tokens_since_log += y.numel()
        state.tokens_seen += y.numel()
        micro_step += 1

        if micro_step % accum == 0:
            # optimizer step
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            state.step += 1

            # ---- logging ----
            if state.step % log_every == 0:
                now = time.time()
                dt = now - t_last_log
                tok_per_s = tokens_since_log / max(dt, 1e-6)
                lr = scheduler.get_last_lr()[0]
                record = {
                    "step": state.step,
                    "loss": loss_accum,  # loss_accum is already averaged
                    "ppl": math.exp(min(loss_accum, 20)),
                    "lr": lr,
                    "tok_per_s": tok_per_s,
                    "tokens_seen": state.tokens_seen,
                    "wall_s": now - state.start_time,
                }
                if is_main:
                    print(
                        f"step {state.step:6d} | loss {loss_accum:.4f} | "
                        f"ppl {record['ppl']:8.2f} | lr {lr:.2e} | "
                        f"{tok_per_s/1000:.1f}k tok/s"
                    )
                    with log_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
                t_last_log = now
                tokens_since_log = 0

            # ---- eval ---- (só rank 0 em DDP)
            if eval_fns and is_main and state.step % eval_every == 0:
                metrics = {}
                for name, fn in eval_fns.items():
                    m = fn()
                    metrics[name] = m
                    print(
                        f"  [eval {name}] loss {m['loss']:.4f} | "
                        f"ppl {m['ppl']:.2f} | "
                        f"{m['n_tokens']} tokens"
                    )
                # Use first eval for best-model selection.
                primary = next(iter(metrics.values()))
                eval_record = {
                    "step": state.step,
                    "eval": metrics,
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(eval_record) + "\n")
                if primary["loss"] < state.best_val:
                    state.best_val = primary["loss"]
                    save_checkpoint(
                        out_dir / "best.pt", model, optimizer, scheduler,
                        state.step, state.best_val,
                        extra={"metrics": metrics},
                    )
                    print(f"  new best: {state.best_val:.4f} -> best.pt")

            # ---- save ---- (só rank 0)
            if is_main and state.step % save_every == 0:
                ckpt_path = out_dir / f"step_{state.step}.pt"
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler,
                    state.step, state.best_val,
                )
                prune_old_checkpoints(out_dir, cfg["keep_last_ckpts"])

            loss_accum = 0.0

    # final save (só rank 0)
    if is_main:
        save_checkpoint(
            out_dir / "final.pt", model, optimizer, scheduler,
            state.step, state.best_val,
        )
    return state
