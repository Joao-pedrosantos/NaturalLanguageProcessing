"""Remove runs concatenadas em um log.jsonl, mantendo apenas a última.

Quando o treino é interrompido (e.g. hibernação) e re-executado sem
limpar o log, o arquivo passa a conter duas runs em sequência. Visualmente
isso aparece como um "step" que decresce no meio do arquivo. Este script
detecta esse decréscimo e mantém apenas os registros a partir da última
reinicialização — i.e., a run completa mais recente.

Uso:
    python -m scripts.clean_log runs/xlstm_medium/log.jsonl [outros...]

Para cada arquivo modificado, salva um backup `<path>.bak` antes de
escrever. Arquivos com uma única run são deixados intactos.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def find_last_run_start(records: list[dict]) -> int:
    """Índice do primeiro registro da última run no arquivo."""
    last_step = -1
    last_reset = 0
    for i, rec in enumerate(records):
        s = rec.get("step")
        if s is None:
            continue
        if s < last_step:
            last_reset = i
        last_step = s
    return last_reset


def clean(path: Path) -> tuple[int, int]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    start = find_last_run_start(records)
    if start == 0:
        return len(records), len(records)

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)

    kept = records[start:]
    with path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records), len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="arquivos log.jsonl a limpar")
    args = ap.parse_args()

    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"AVISO: {path} não existe, pulando")
            continue
        before, after = clean(path)
        if before == after:
            print(f"{path}: clean ({after} records, no concatenated runs)")
        else:
            backup = path.with_suffix(path.suffix + ".bak")
            print(f"{path}: cleaned {before} -> {after} records (backup: {backup})")


if __name__ == "__main__":
    main()
