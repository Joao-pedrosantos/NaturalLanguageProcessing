"""xLSTM LM model factory and parameter accounting.

Wraps the NX-AI `xlstm` package so the rest of the codebase doesn't need
to know about OmegaConf + dacite plumbing.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf

from xlstm import xLSTMLMModel, xLSTMLMModelConfig


def load_config(path: str | Path) -> xLSTMLMModelConfig:
    """Load a YAML config file into an xLSTMLMModelConfig dataclass."""
    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    return from_dict(
        data_class=xLSTMLMModelConfig,
        data=raw,
        config=DaciteConfig(strict=True),
    )


def build_model(cfg: xLSTMLMModelConfig) -> xLSTMLMModel:
    """Build an xLSTMLMModel from config. Weights are freshly initialized."""
    return xLSTMLMModel(cfg)


def count_parameters(model: nn.Module) -> dict:
    """Return a breakdown of parameter counts.

    Returns dict with:
      total       — total parameter count
      trainable   — trainable parameter count
      by_module   — dict mapping top-level submodule name to param count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        by_module[top] = by_module.get(top, 0) + p.numel()
    return {
        "total": total,
        "trainable": trainable,
        "by_module": dict(sorted(
            by_module.items(), key=lambda kv: -kv[1]
        )),
    }


def format_params(n: int) -> str:
    """Human-readable parameter count, e.g. 50_123_456 -> '50.12M'."""
    for unit, scale in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)
