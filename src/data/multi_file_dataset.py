"""Dataset que concatena vários arquivos .txt e empacota como LM.

Análogo ao FileEvalDataset, mas opera sobre uma pasta inteira. Os livros
são lidos em ordem alfabética e separados por </s>, então o modelo
aprende a tratar fronteira entre livros como fronteira entre documentos.

Não é "infinite streaming" — usar com CyclingDataset wrapper se quiser
loopar pra POC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


class MultiFileDataset(IterableDataset):
    """Lê todos os .txt de uma pasta, tokeniza e empacota em chunks.

    Cada arquivo é tratado como um documento; entre documentos insere
    um </s>. Os chunks têm tamanho `seq_len + 1` (pra compor inputs e
    targets shifted no collate).
    """

    def __init__(
        self,
        folder: str | Path,
        tokenizer,
        seq_len: int = 256,
        eos_token: str = "</s>",
        pattern: str = "*.txt",
    ):
        self.folder = Path(folder)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.token_to_id(eos_token)
        if self.eos_id is None:
            raise ValueError(f"tokenizer não tem '{eos_token}'")
        self.files = sorted(self.folder.glob(pattern))
        if not self.files:
            raise SystemExit(
                f"nenhum arquivo {pattern} em {self.folder}"
            )

    def __iter__(self) -> Iterator[torch.Tensor]:
        buf: list[int] = []
        chunk = self.seq_len + 1
        for path in self.files:
            text = path.read_text(encoding="utf-8")
            ids = self.tokenizer.encode(text).ids
            ids.append(self.eos_id)
            buf.extend(ids)
            while len(buf) >= chunk:
                yield torch.tensor(buf[:chunk], dtype=torch.long)
                buf = buf[chunk:]
        # descarta o resto (< seq_len+1 tokens) — desperdício é desprezível
