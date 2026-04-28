"""Instancia um LM a partir de config e reporta contagem de parâmetros.

Use isso antes de comprometer-se a um treino — ajuste num_blocks /
embedding_dim se estiver fora do alvo.

Uso:
    python -m scripts.count_params --config configs/xlstm_tiny.yaml
    python -m scripts.count_params --config configs/transformer_tiny.yaml --arch transformer
"""

from __future__ import annotations

import argparse

from src.model import count_parameters, format_params
from src.model_factory import build_model_for_arch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xlstm_tiny.yaml")
    ap.add_argument(
        "--arch", default="xlstm", choices=["xlstm", "transformer"],
    )
    args = ap.parse_args()

    model = build_model_for_arch(args.arch, args.config)
    stats = count_parameters(model)
    cfg = model.config

    print(f"arch:            {args.arch}")
    print(f"config:          {args.config}")
    print(f"vocab_size:      {cfg.vocab_size}")
    print(f"embedding_dim:   {cfg.embedding_dim}")
    print(f"num_blocks:      {cfg.num_blocks}")
    print(f"context_length:  {cfg.context_length}")
    print()
    print(f"total params:     {format_params(stats['total']):>10s}"
          f"  ({stats['total']:,})")
    print(f"trainable params: {format_params(stats['trainable']):>10s}"
          f"  ({stats['trainable']:,})")
    print()
    print("top-level breakdown:")
    for name, n in stats["by_module"].items():
        share = 100 * n / stats["total"]
        print(f"  {name:30s}  {format_params(n):>10s}  ({share:5.1f}%)")
