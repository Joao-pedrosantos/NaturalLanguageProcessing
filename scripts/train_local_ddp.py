"""Versão DDP do train_local — treina um único config em N GPUs.

Uso (com torchrun):
    torchrun --nproc_per_node=4 -m scripts.train_local_ddp \\
        --config configs/train_large.yaml \\
        --train-file artifacts/train_books \\
        --dev-file artifacts/dev/pt_dev.txt

torchrun seta LOCAL_RANK, RANK, WORLD_SIZE no env. Cada processo pega 1 GPU,
sharda o dataset (rank i pega samples i, i+W, i+2W, ...), e DDP sincroniza
gradientes a cada backward(). Só rank 0 imprime, escreve log e salva ckpts.
"""

from __future__ import annotations

import argparse
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from src.data.eval_dataset import FileEvalDataset
from src.data.multi_file_dataset import MultiFileDataset
from src.data.streaming import collate_lm
from src.data.tokenizer import load_tokenizer
from src.model import count_parameters, format_params
from src.model_factory import build_model_for_arch
from src.training import evaluate, train


class ShardedCyclingDataset(IterableDataset):
    """Loopa um IterableDataset infinitamente, shardando por rank.

    Rank i só emite o sample j quando j % world_size == i. Garante que
    cada rank vê dados disjuntos dentro de uma "época".
    """

    def __init__(self, factory, rank: int = 0, world_size: int = 1):
        self.factory = factory
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        i = 0
        while True:
            for sample in self.factory():
                if i % self.world_size == self.rank:
                    yield sample
                i += 1


def set_seed(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_tiny.yaml")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--dev-file", default=None)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    ddp = world_size > 1

    if ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    is_main = rank == 0

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    set_seed(cfg["seed"], rank=rank)

    if is_main:
        print(f"DDP: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"device: {device}")
        if device.type == "cuda":
            print(f"  {torch.cuda.get_device_name(local_rank)}")

    tokenizer = load_tokenizer(cfg["tokenizer"])
    if is_main:
        print(f"tokenizer vocab: {tokenizer.get_vocab_size()}")

    arch = cfg.get("arch", "xlstm")
    if is_main:
        print(f"arch: {arch}")
    model = build_model_for_arch(arch, cfg["model_config"]).to(device)

    if model.config.vocab_size != tokenizer.get_vocab_size():
        raise SystemExit(
            f"vocab_size do modelo ({model.config.vocab_size}) != tokenizer "
            f"({tokenizer.get_vocab_size()})"
        )
    if model.config.context_length < cfg["seq_len"]:
        raise SystemExit(
            f"context_length ({model.config.context_length}) < seq_len "
            f"({cfg['seq_len']})"
        )

    if is_main:
        stats = count_parameters(model)
        print(f"model params: {format_params(stats['total'])} "
              f"({stats['total']:,})")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # --- dados (sharded por rank) ---
    train_path = Path(args.train_file)
    if not train_path.exists():
        raise SystemExit(f"caminho de treino não encontrado: {train_path}")

    if train_path.is_dir():
        n_files = len(list(train_path.glob("*.txt")))
        if is_main:
            print(f"treino: pasta com {n_files} arquivo(s) .txt")
        train_ds = ShardedCyclingDataset(
            partial(MultiFileDataset, train_path, tokenizer, cfg["seq_len"]),
            rank=rank, world_size=world_size,
        )
    else:
        if is_main:
            print(f"treino: arquivo único ({train_path})")
        train_ds = ShardedCyclingDataset(
            partial(FileEvalDataset, train_path, tokenizer, cfg["seq_len"]),
            rank=rank, world_size=world_size,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        collate_fn=collate_lm,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # --- eval (só rank 0) ---
    eval_fns = None
    if args.dev_file and is_main:
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
                ds = FileEvalDataset(dev_path, tokenizer,
                                     seq_len=cfg["seq_len"])
                loader = DataLoader(ds, batch_size=eval_batch,
                                    collate_fn=collate_lm)
                model_eval = model.module if ddp else model
                return evaluate(model_eval, loader, device, dtype,
                                max_batches=cfg["eval_max_batches"])

            eval_fns = {"dev": _dev_eval}

    out_dir = Path(cfg["out_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(cfg), out_dir / "train_config.yaml")

    train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        eval_fns=eval_fns,
        out_dir=out_dir,
        rank=rank,
    )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
