"""Compara runs de treino lado a lado a partir dos log.jsonl.

Lê os logs de duas (ou mais) runs e plota as curvas de perplexidade no
dev set ao longo dos steps. Também imprime uma tabela com os números
finais — o que vai pro relatório.

Uso:
    python -m scripts.compare_runs \\
        --runs runs/xlstm_tiny runs/transformer_tiny \\
        --labels xLSTM Transformer \\
        --out artifacts/comparison.png

Se matplotlib não estiver instalado, só imprime a tabela.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_log(path: Path) -> dict:
    """Extrai pontos de treino e eval do log.jsonl.

    Retorna dict com:
      train_steps, train_loss      — listas paralelas
      eval_steps, eval_loss, eval_ppl — listas paralelas (do primeiro eval set)
    """
    train_steps, train_loss = [], []
    eval_steps, eval_loss, eval_ppl = [], [], []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "loss" in r and "eval" not in r:
                train_steps.append(r["step"])
                train_loss.append(r["loss"])
            elif "eval" in r:
                # pega o primeiro eval set (geralmente "dev")
                first_key = next(iter(r["eval"].keys()))
                m = r["eval"][first_key]
                eval_steps.append(r["step"])
                eval_loss.append(m["loss"])
                eval_ppl.append(m["ppl"])
    return dict(
        train_steps=train_steps, train_loss=train_loss,
        eval_steps=eval_steps, eval_loss=eval_loss, eval_ppl=eval_ppl,
    )


def print_summary(runs: list[tuple[str, dict]]) -> None:
    print(f"\n{'='*60}")
    print(f"{'run':<20s} {'final loss':>12s} {'final ppl':>12s} "
          f"{'best ppl':>12s}")
    print("=" * 60)
    for label, data in runs:
        if not data["eval_ppl"]:
            print(f"{label:<20s} (sem evals)")
            continue
        final_loss = data["eval_loss"][-1]
        final_ppl = data["eval_ppl"][-1]
        best_ppl = min(data["eval_ppl"])
        print(f"{label:<20s} {final_loss:>12.4f} {final_ppl:>12.2f} "
              f"{best_ppl:>12.2f}")
    print("=" * 60)


def plot(runs: list[tuple[str, dict]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib não instalado — pule a flag --out ou instale "
              "com `pip install matplotlib`)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, data in runs:
        if data["train_steps"]:
            ax1.plot(data["train_steps"], data["train_loss"],
                     label=label, alpha=0.6, linewidth=1)
        if data["eval_steps"]:
            ax2.plot(data["eval_steps"], data["eval_ppl"],
                     label=label, marker="o", linewidth=2)

    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss")
    ax1.set_title("Loss de treino")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("step")
    ax2.set_ylabel("dev perplexity")
    ax2.set_title("Perplexidade no dev set (holdout)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nplot salvo em {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", required=True,
        help="diretórios de run (ex: runs/xlstm_tiny runs/transformer_tiny)",
    )
    ap.add_argument(
        "--labels", nargs="+", default=None,
        help="rótulos correspondentes (default: usa o nome do diretório)",
    )
    ap.add_argument(
        "--out", default="artifacts/comparison.png",
        help="caminho do .png de saída",
    )
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise SystemExit("número de --labels precisa bater com --runs")
    labels = args.labels or [Path(r).name for r in args.runs]

    parsed = []
    for run_dir, label in zip(args.runs, labels):
        log_path = Path(run_dir) / "log.jsonl"
        if not log_path.exists():
            print(f"AVISO: {log_path} não existe, pulando")
            continue
        parsed.append((label, parse_log(log_path)))

    if not parsed:
        raise SystemExit("nenhum log encontrado")

    print_summary(parsed)
    plot(parsed, Path(args.out))
