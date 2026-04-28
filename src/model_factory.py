"""Fábrica de modelos arch-agnostic. Despacha entre xLSTM e Transformer
baseado num campo `arch` lido do training config.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn


def build_model_for_arch(arch: str, model_config_path: str | Path) -> nn.Module:
    """Carrega o config de modelo e instancia a arquitetura correspondente.

    Parameters
    ----------
    arch : "xlstm" ou "transformer"
    model_config_path : caminho do .yaml com a config do modelo

    Returns
    -------
    nn.Module com:
      - .config.context_length (usado em truncamento durante geração)
      - .forward(input_ids) -> logits (B, T, V)
    """
    if arch == "xlstm":
        from src.model import build_model, load_config
        cfg = load_config(model_config_path)
        return build_model(cfg)
    elif arch == "transformer":
        from src.transformer import build_transformer, load_transformer_config
        cfg = load_transformer_config(model_config_path)
        return build_transformer(cfg)
    else:
        raise ValueError(
            f"arch desconhecida: {arch!r}. Use 'xlstm' ou 'transformer'."
        )
