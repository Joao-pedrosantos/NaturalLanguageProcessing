"""Generate the auxiliary figures for the paper:

- throughput_bars.png: tokens/s per architecture (log scale)
- best_ppl_vs_params.png: best dev PPL as a function of parameter count
- downstream_bars.png: author + IMDB-PT probe accuracies side-by-side

Run with:
    python -m scripts.make_paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


XLSTM_COLOR = "#1f77b4"
TX_COLOR = "#d62728"


def load_run_stats() -> dict[str, dict]:
    """Read runs/<name>/log.jsonl for the six GPU runs and return per-run
    {mean_tok_per_s, best_ppl, final_ppl, best_step}."""
    stats: dict[str, dict] = {}
    for name in (
        "xlstm_tiny", "xlstm_medium", "xlstm_10m",
        "transformer_tiny", "transformer_medium", "transformer_10m",
    ):
        p = Path(f"runs/{name}/log.jsonl")
        if not p.exists():
            continue
        tps: list[float] = []
        best_ppl = float("inf")
        best_step = None
        final_ppl = None
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "tok_per_s" in r:
                tps.append(r["tok_per_s"])
            if "eval" in r and "dev" in r["eval"]:
                ppl = r["eval"]["dev"]["ppl"]
                final_ppl = ppl
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_step = r["step"]
        tps_w = tps[1:] if len(tps) > 2 else tps
        stats[name] = dict(
            mean_tok_per_s=sum(tps_w) / len(tps_w) if tps_w else 0.0,
            best_ppl=best_ppl,
            final_ppl=final_ppl,
            best_step=best_step,
        )
    return stats


def plot_throughput(stats: dict[str, dict], out: Path) -> None:
    """Log-scale grouped bar chart of tokens/s per scale."""
    scales = ["tiny", "medium", "10m"]
    scale_labels = ["tiny (2.5M)", "medium (~6.5M)", "10M"]
    xlstm_tps = [stats[f"xlstm_{s}"]["mean_tok_per_s"] / 1000 for s in scales]
    tx_tps = [stats[f"transformer_{s}"]["mean_tok_per_s"] / 1000 for s in scales]

    x = np.arange(len(scales))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    b1 = ax.bar(x - width / 2, xlstm_tps, width,
                label="xLSTM (vanilla backend)", color=XLSTM_COLOR)
    b2 = ax.bar(x + width / 2, tx_tps, width,
                label="Transformer (SDPA / FlashAttention)", color=TX_COLOR)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_ylabel("Training throughput (k tokens / s, log scale)")
    ax.set_title("GPU training throughput by scale (NVIDIA RTX 5070 Ti)")
    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.95)

    # Annotate speedup ratios above transformer bars
    for i, (x_v, t_v) in enumerate(zip(xlstm_tps, tx_tps)):
        ratio = t_v / x_v if x_v > 0 else 0
        ax.text(x[i] + width / 2, t_v * 1.15,
                f"{ratio:.0f}×",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TX_COLOR)
        ax.text(x[i] - width / 2, x_v * 1.15, f"{x_v:.1f}k",
                ha="center", va="bottom", fontsize=8, color=XLSTM_COLOR)
        ax.text(x[i] + width / 2, t_v * 0.55, f"{t_v:.0f}k",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_best_ppl_vs_params(stats: dict[str, dict], out: Path) -> None:
    """Best dev PPL vs parameter count for both architectures."""
    from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter

    params = {
        "tiny": 2.5, "medium": 6.5, "10m": 10.2,
    }
    scales = ["tiny", "medium", "10m"]
    xlstm_ppl = [stats[f"xlstm_{s}"]["best_ppl"] for s in scales]
    tx_ppl = [stats[f"transformer_{s}"]["best_ppl"] for s in scales]
    xs = [params[s] for s in scales]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, xlstm_ppl, "o-", linewidth=2, markersize=10,
            color=XLSTM_COLOR, label="xLSTM")
    ax.plot(xs, tx_ppl, "s--", linewidth=2, markersize=10,
            color=TX_COLOR, label="Transformer")

    for x_v, y_v in zip(xs, xlstm_ppl):
        ax.annotate(f"{y_v:.0f}", (x_v, y_v), textcoords="offset points",
                    xytext=(10, 8), fontsize=10, color=XLSTM_COLOR,
                    fontweight="bold")
    for x_v, y_v in zip(xs, tx_ppl):
        ax.annotate(f"{y_v:.0f}", (x_v, y_v), textcoords="offset points",
                    xytext=(10, -16), fontsize=10, color=TX_COLOR,
                    fontweight="bold")

    # Highlight the crossover region with a faint band
    ax.axvspan(5.8, 7.5, color="gray", alpha=0.08)
    ax.text(6.65, 250, "crossover",
            ha="center", fontsize=9, color="gray", style="italic")

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Best held-out PPL")
    ax.set_title("Best dev perplexity vs. scale (GPU sweep)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xticks([2.5, 6.5, 10.2])
    ax.set_xticklabels(["2.5", "6.5", "10.2"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1.5, 11.5)
    ax.set_ylim(220, 770)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_downstream(stats: dict[str, dict], out: Path) -> None:
    """Grouped bar chart of downstream probe accuracies."""
    # Load probe JSONs
    def get_probe(arch: str, scale: str, dataset: str) -> float | None:
        if dataset == "author":
            p = Path(f"artifacts/probe_{arch}_{scale}.json")
        elif dataset == "imdb":
            p = Path(f"artifacts/sentiment_{arch}_{scale}.json")
        else:
            return None
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        return r["test_acc"] * 100

    groups = [
        ("Author probe\n(medium)", get_probe("xlstm", "medium", "author"),
         get_probe("transformer", "medium", "author"), 11.1),
        ("Author probe\n(10M)", get_probe("xlstm", "10m", "author"),
         get_probe("transformer", "10m", "author"), 11.1),
        ("IMDB-PT sentiment\n(10M)", get_probe("xlstm", "10m", "imdb"),
         get_probe("transformer", "10m", "imdb"), 50.0),
    ]

    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xlstm_vals = [g[1] for g in groups]
    tx_vals = [g[2] for g in groups]
    baselines = [g[3] for g in groups]

    b1 = ax.bar(x - width / 2, xlstm_vals, width,
                label="xLSTM", color=XLSTM_COLOR)
    b2 = ax.bar(x + width / 2, tx_vals, width,
                label="Transformer", color=TX_COLOR)

    # Baselines as horizontal dashed lines per group
    for i, base in enumerate(baselines):
        ax.hlines(base, x[i] - width, x[i] + width,
                  colors="gray", linestyles=":", linewidth=1.4)
        ax.text(x[i] + width + 0.04, base, f"{base:.1f}% baseline",
                fontsize=8, color="gray", va="center")

    for rect, val in list(zip(b1, xlstm_vals)) + list(zip(b2, tx_vals)):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.6,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Test accuracy (\\%)")
    ax.set_title("Downstream linear-probe accuracy (GPU checkpoints)")
    ax.set_ylim(0, max(max(xlstm_vals), max(tx_vals)) * 1.18)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading run stats...")
    stats = load_run_stats()
    for name, s in stats.items():
        print(f"  {name:>22} | best_ppl={s['best_ppl']:.2f} "
              f"| tok/s={s['mean_tok_per_s']/1000:.1f}k")

    print("\nplotting...")
    plot_throughput(stats, out_dir / "throughput_bars.png")
    plot_best_ppl_vs_params(stats, out_dir / "best_ppl_vs_params.png")
    plot_downstream(stats, out_dir / "downstream_bars.png")

    # Copy to root for pdflatex to pick up
    import shutil
    for fname in ("throughput_bars.png", "best_ppl_vs_params.png",
                  "downstream_bars.png"):
        shutil.copy(out_dir / fname, Path(".") / fname)
    print("\nfeito.")


if __name__ == "__main__":
    main()
