"""Byte-level BPE tokenizer for the multilingual corpus.

Byte-level BPE (GPT-2 style) has two big advantages for this project:
  1. No <unk> tokens possible — any unicode byte is always encodable.
  2. Handles mixed-script text (PT accents, EN ASCII) uniformly.

Special tokens follow the usual LM convention:
  <pad>  — padding for fixed-length batches
  <s>    — beginning of sequence
  </s>   — end of sequence (also doubles as document separator in packing)
  <unk>  — reserved but should be unused with byte-level BPE
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as BLPreTok
from tokenizers.decoders import ByteLevel as BLDecoder


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


def build_tokenizer() -> Tokenizer:
    """Empty byte-level BPE tokenizer, ready for training."""
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = BLPreTok(add_prefix_space=False)
    tok.decoder = BLDecoder()
    return tok


def train_tokenizer(
    iterator: Iterable[str],
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    save_path: str | Path = "artifacts/tokenizer.json",
) -> Tokenizer:
    """Train BPE on an iterator of text strings, save, and return."""
    tok = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=BLPreTok.alphabet(),
    )
    tok.train_from_iterator(iterator, trainer=trainer)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
