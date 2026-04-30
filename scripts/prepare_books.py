"""Baixa livros do Projeto Gutenberg, limpa o boilerplate, e organiza
em train/dev sets disjuntos.

Lista de livros vem de `configs/books.yaml`. Pra adicionar novos livros,
edite o YAML — não precisa mexer aqui. Idempotente: pula livros já
baixados.

Uso:
    python -m scripts.prepare_books
    python -m scripts.prepare_books --books configs/books.yaml

Saída:
    artifacts/train_books/*.txt   — todos os livros do bloco `train`
    artifacts/dev/pt_dev.txt      — livro do bloco `holdout`
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from omegaconf import OmegaConf


START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)
END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)


def fetch(book_id: int) -> str:
    """Baixa o .txt do Gutenberg. Tenta três layouts conhecidos."""
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
    ]
    last_err: Exception | None = None
    for url in candidates:
        try:
            print(f"  fetching {url}")
            with urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError) as e:
            last_err = e
            print(f"    falhou ({e}), tentando próximo layout...")
    raise RuntimeError(f"todos os layouts falharam para id={book_id}: {last_err}")


def clean_gutenberg(text: str) -> str:
    """Remove cabeçalho/rodapé do Gutenberg, deixando só o livro."""
    m_start = START_RE.search(text)
    m_end = END_RE.search(text)
    if m_start:
        text = text[m_start.end():]
    if m_end:
        text = text[: m_end.start()]
    return text.strip()


def save_book(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    kb = path.stat().st_size / 1024
    print(f"  -> {path} ({kb:.1f} KB)")


def download_one(entry: dict, out_dir: Path) -> Path | None:
    """Baixa um livro se ainda não existir. Retorna o path se ok, None se falhou."""
    book_id = int(entry["id"])
    fname = entry["file"]
    title = entry.get("title", "?")
    author = entry.get("author", "?")
    print(f"\n{title} ({author}) [id={book_id}]")
    out = out_dir / fname
    if out.exists():
        kb = out.stat().st_size / 1024
        print(f"  já existe: {out} ({kb:.1f} KB) — pulando")
        return out
    try:
        raw = fetch(book_id)
    except Exception as e:
        print(f"  ERRO no download: {e}")
        return None
    cleaned = clean_gutenberg(raw)
    save_book(cleaned, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--books", default="configs/books.yaml",
        help="YAML com a lista de livros (default: configs/books.yaml)",
    )
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.books), resolve=True)
    train_books = cfg.get("train", [])
    holdout = cfg.get("holdout")
    if not train_books:
        raise SystemExit(f"{args.books} não tem livros em `train`.")
    if not holdout:
        raise SystemExit(f"{args.books} não tem `holdout`.")

    train_dir = Path("artifacts/train_books")
    dev_dir = Path("artifacts/dev")

    print(f"=== TRAIN ({len(train_books)} livros) ===")
    total_kb = 0.0
    n_ok = 0
    for entry in train_books:
        path = download_one(entry, train_dir)
        if path and path.exists():
            total_kb += path.stat().st_size / 1024
            n_ok += 1
    print(
        f"\ntotal treino: {total_kb:.1f} KB "
        f"(~{total_kb / 4:.0f}k tokens estimados, {n_ok}/{len(train_books)} ok)"
    )

    print("\n=== DEV (holdout) ===")
    download_one(holdout, dev_dir)

    print("\nfeito. próximos passos:")
    print("  1) (re)treinar tokenizer se mudou o corpus:")
    print("       python -m scripts.train_tokenizer")
    print("  2) treinar:")
    print("       bash scripts/run_desktop.sh    # GPU desktop")


if __name__ == "__main__":
    main()
