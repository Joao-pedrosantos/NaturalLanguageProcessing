"""Treino local em arquivo único OU pasta de livros — POC sem Wikipedia.

Detecta automaticamente se --train-file é um arquivo .txt único ou uma
pasta contendo vários .txt. No segundo caso, todos os arquivos são
concatenados (com </s> entre eles) durante a iteração.

Uso (livro único):
    python -m scripts.train_local \\
        --config configs/train_tiny.yaml \\
        --train-file artifacts/tokenizer_corpus/livro.txt \\
        --dev-file artifacts/dev/pt_dev.txt

Uso (vários livros):
    python -m scripts.train_local \\
        --config configs/train_tiny.yaml \\
        --train-file artifacts/train_books \\
        --dev-file artifacts/dev/pt_dev.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from src.data.eval_dataset import FileEvalDataset
from src.data.multi_file_dataset import MultiFileDataset
from src.data.streaming import collate_lm
from src.data.tokenizer import load_tokenizer
from src.model import count_parameters, format_params
from src.model_factory import build_model_for_arch
from src.training import evaluate, train


class CyclingDataset(IterableDataset):
    """Loopa um IterableDataset finito infinitamente.

    O loop de treino consome batches até atingir total_steps; um corpus
    pequeno se esgota em poucos batches, então re-iniciamos quantas vezes
    forem necessárias. As múltiplas "épocas" são esperadas em POC.
    """

    def __init__(self, factory):
        """factory: callable que retorna um novo IterableDataset a cada chamada."""
        self.factory = factory

    def __iter__(self):
        while True:
            yield from self.factory()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_tiny.yaml")
    ap.add_argument(
        "--train-file", required=True,
        help="caminho do arquivo .txt para treino",
    )
    ap.add_argument(
        "--dev-file", default=None,
        help="caminho opcional do arquivo .txt para eval",
    )
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # --- tokenizer ---
    tokenizer = load_tokenizer(cfg["tokenizer"])
    print(f"tokenizer vocab: {tokenizer.get_vocab_size()}")

    # --- modelo ---
    arch = cfg.get("arch", "xlstm")
    print(f"arch: {arch}")
    model = build_model_for_arch(arch, cfg["model_config"]).to(device)

    # Validações comuns aos dois archs.
    if model.config.vocab_size != tokenizer.get_vocab_size():
        raise SystemExit(
            f"vocab_size do modelo ({model.config.vocab_size}) != tokenizer "
            f"({tokenizer.get_vocab_size()}). Ajuste o config ou re-treine "
            f"o tokenizer."
        )
    if model.config.context_length < cfg["seq_len"]:
        raise SystemExit(
            f"context_length ({model.config.context_length}) < seq_len "
            f"({cfg['seq_len']})"
        )

    stats = count_parameters(model)
    print(f"model params: {format_params(stats['total'])} "
          f"({stats['total']:,})")

    # --- data ---
    train_path = Path(args.train_file)
    if not train_path.exists():
        raise SystemExit(f"caminho de treino não encontrado: {train_path}")

    if train_path.is_dir():
        # pasta com vários livros: concatena todos com </s> entre eles
        n_files = len(list(train_path.glob("*.txt")))
        print(f"treino: pasta com {n_files} arquivo(s) .txt")
        train_ds = CyclingDataset(
            lambda: MultiFileDataset(train_path, tokenizer, cfg["seq_len"])
        )
    else:
        # arquivo único
        print(f"treino: arquivo único ({train_path})")
        train_ds = CyclingDataset(
            lambda: FileEvalDataset(train_path, tokenizer, cfg["seq_len"])
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        collate_fn=collate_lm,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # --- eval (opcional) ---
    eval_fns = None
    if args.dev_file:
        dev_path = Path(args.dev_file)
        if not dev_path.exists():
            print(f"AVISO: dev file não encontrado ({dev_path}), pulando eval")
        else:
            dtype = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[cfg["dtype"]]
            eval_batch = max(1, cfg["batch_size"] // 2)

            def _dev_eval():
                ds = FileEvalDataset(dev_path, tokenizer, seq_len=cfg["seq_len"])
                loader = DataLoader(
                    ds, batch_size=eval_batch, collate_fn=collate_lm
                )
                return evaluate(
                    model, loader, device, dtype,
                    max_batches=cfg["eval_max_batches"],
                )

            eval_fns = {"dev": _dev_eval}

    # --- treino ---
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), out_dir / "train_config.yaml")

    train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        eval_fns=eval_fns,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
