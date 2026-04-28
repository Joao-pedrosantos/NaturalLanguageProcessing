"""File-based eval dataset for per-language perplexity measurement.

The streaming training pipeline packs PT and EN together, which is great
for training but makes per-language evaluation impossible (a single chunk
can span both languages). For eval we want fixed PT-only and EN-only
sequences, so we dump small deterministic dev sets to disk once and
tokenize them at eval time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


class FileEvalDataset(IterableDataset):
    """Load a text file, tokenize, and yield packed (seq_len+1)-token chunks.

    Deterministic: same file + same tokenizer always yields the same chunks.
    Use one instance per language (one file, one loader).
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        seq_len: int = 1024,
        eos_token: str = "</s>",
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.token_to_id(eos_token)
        if self.eos_id is None:
            raise ValueError(f"Tokenizer missing '{eos_token}'")

    def __iter__(self) -> Iterator[torch.Tensor]:
        buf: list[int] = []
        chunk = self.seq_len + 1
        # Read document-by-document, separated by blank lines in the dump.
        current: list[str] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    current.append(line)
                else:
                    if current:
                        text = "".join(current)
                        ids = self.tokenizer.encode(text).ids
                        ids.append(self.eos_id)
                        buf.extend(ids)
                        current = []
                    while len(buf) >= chunk:
                        yield torch.tensor(buf[:chunk], dtype=torch.long)
                        buf = buf[chunk:]
            # flush last doc
            if current:
                text = "".join(current)
                ids = self.tokenizer.encode(text).ids
                ids.append(self.eos_id)
                buf.extend(ids)
                while len(buf) >= chunk:
                    yield torch.tensor(buf[:chunk], dtype=torch.long)
                    buf = buf[chunk:]
