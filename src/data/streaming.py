"""Multilingual Wikipedia streaming pipeline for xLSTM pre-training.

Streams PT and EN Wikipedia from HuggingFace (no full download), mixes them
with temperature sampling, tokenizes on the fly, and packs into fixed-length
sequences for causal LM training.

Nothing is saved to disk in this module except what the OS chooses to cache
via the `datasets` streaming layer.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets


# Article counts in the wikimedia/wikipedia 20231101 dumps.
# These drive the temperature-based language mixing.
WIKI_PT_ARTICLES = 1_112_834
WIKI_EN_ARTICLES = 6_705_918


def temperature_probs(counts: list[int], alpha: float = 0.7) -> list[float]:
    """Temperature sampling a la XLM-R: p_i ∝ n_i ** alpha.

    alpha=1.0 -> proportional to size (EN dominates at ~86%).
    alpha=0.0 -> uniform across languages (50/50).
    alpha=0.7 -> soft upweight of smaller languages (good default for PT+EN).
    """
    weights = [c ** alpha for c in counts]
    total = sum(weights)
    return [w / total for w in weights]


def build_streaming_wiki(
    snapshot: str = "20231101",
    alpha: float = 0.7,
    seed: int = 42,
    shuffle_buffer: int = 10_000,
):
    """Build an interleaved, shuffled stream of PT+EN Wikipedia articles.

    Returns
    -------
    mixed : IterableDataset
        Yields dicts with keys {"text", "lang"}.
    probs : list[float]
        Sampling probabilities actually used (for logging).
    """
    ds_pt = load_dataset(
        "wikimedia/wikipedia", f"{snapshot}.pt",
        split="train", streaming=True,
    )
    ds_en = load_dataset(
        "wikimedia/wikipedia", f"{snapshot}.en",
        split="train", streaming=True,
    )

    # Tag language explicitly so it survives the interleave.
    # Using `remove_columns` to keep the output schema minimal.
    ds_pt = ds_pt.map(
        lambda x: {"text": x["text"], "lang": "pt"},
        remove_columns=[c for c in ds_pt.column_names if c not in ("text",)],
    )
    ds_en = ds_en.map(
        lambda x: {"text": x["text"], "lang": "en"},
        remove_columns=[c for c in ds_en.column_names if c not in ("text",)],
    )

    probs = temperature_probs(
        [WIKI_PT_ARTICLES, WIKI_EN_ARTICLES], alpha=alpha
    )
    mixed = interleave_datasets(
        [ds_pt, ds_en],
        probabilities=probs,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    # Streaming-compatible shuffle with a bounded memory buffer.
    mixed = mixed.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return mixed, probs


class PackedLMDataset(IterableDataset):
    """Tokenize streaming articles and pack into fixed-length chunks.

    Articles are concatenated with an EOS separator. Each yielded chunk has
    length `seq_len + 1`, so the trainer can split into (inputs, targets)
    with shifted positions for next-token prediction.

    This never materializes the whole corpus — only `seq_len + 1` tokens
    worth of buffer live in memory per worker.
    """

    def __init__(
        self,
        stream,
        tokenizer,
        seq_len: int = 1024,
        eos_token: str = "</s>",
        eos_token_id: Optional[int] = None,
    ):
        self.stream = stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else tokenizer.token_to_id(eos_token)
        )
        if self.eos_token_id is None:
            raise ValueError(
                f"Tokenizer has no token '{eos_token}'. Train it with "
                f"that special token or pass `eos_token_id` explicitly."
            )

    def __iter__(self) -> Iterator[torch.Tensor]:
        buf: list[int] = []
        chunk = self.seq_len + 1  # +1 so we can build shifted targets
        for sample in self.stream:
            text = sample.get("text") or ""
            if not text:
                continue
            ids = self.tokenizer.encode(text).ids
            ids.append(self.eos_token_id)
            buf.extend(ids)
            while len(buf) >= chunk:
                yield torch.tensor(buf[:chunk], dtype=torch.long)
                buf = buf[chunk:]


def collate_lm(batch: list[torch.Tensor]):
    """Stack packed chunks into (inputs, targets) for causal LM loss."""
    x = torch.stack(batch, dim=0)           # (B, seq_len + 1)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()
