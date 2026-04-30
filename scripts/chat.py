"""REPL pra mandar mensagens (prompts) e ver o que o modelo gera.

Importante: estes modelos são LMs autoregressivos treinados em português
do séc. XIX, NÃO são instruction-tuned. Eles continuam o texto que você
mandar, não respondem a perguntas. Mande um trecho que faça sentido como
abertura ("Capitu olhou para", "O sol nascia sobre o", etc.) em vez de
"Olá, tudo bem?".

Comandos especiais dentro do REPL:
    :temp <float>       muda temperatura (default 0.8)
    :topk <int>         muda top-k (default 40, 0 desliga)
    :tokens <int>       muda max_new_tokens (default 100)
    :seed <int>         fixa seed pra geração determinística
    :params             mostra hiperparâmetros atuais
    :quit / :q / Ctrl-D sai

Uso:
    python -m scripts.chat \\
        --ckpt runs/xlstm_10m/best.pt \\
        --config configs/xlstm_10m.yaml \\
        --arch xlstm

    python -m scripts.chat \\
        --ckpt runs/transformer_10m/best.pt \\
        --config configs/transformer_10m.yaml \\
        --arch transformer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.generate import generate
from src.data.tokenizer import load_tokenizer
from src.model_factory import build_model_for_arch
from src.training import load_checkpoint


def repl(model, tokenizer, device, temperature, top_k, max_new_tokens):
    print()
    print("Modelo carregado. Mande um prompt e veja a continuação.")
    print("Dica: estes modelos continuam texto, não respondem a perguntas.")
    print("Comandos: :temp, :topk, :tokens, :seed, :params, :quit (ou Ctrl-D).")
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue

        if line.startswith(":"):
            parts = line.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in (":quit", ":q", ":exit"):
                return
            elif cmd == ":temp":
                try:
                    temperature = float(arg)
                    print(f"  temperature = {temperature}")
                except ValueError:
                    print("  uso: :temp <float>")
            elif cmd == ":topk":
                try:
                    top_k = int(arg)
                    print(f"  top_k = {top_k} (0 = desligado)")
                except ValueError:
                    print("  uso: :topk <int>")
            elif cmd == ":tokens":
                try:
                    max_new_tokens = int(arg)
                    print(f"  max_new_tokens = {max_new_tokens}")
                except ValueError:
                    print("  uso: :tokens <int>")
            elif cmd == ":seed":
                try:
                    torch.manual_seed(int(arg))
                    print(f"  seed = {arg}")
                except ValueError:
                    print("  uso: :seed <int>")
            elif cmd == ":params":
                print(f"  temperature     = {temperature}")
                print(f"  top_k           = {top_k}")
                print(f"  max_new_tokens  = {max_new_tokens}")
            else:
                print(f"  comando desconhecido: {cmd}")
            continue

        out = generate(
            model, tokenizer, line,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            device=device,
        )
        print()
        print(out)
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--arch", default="xlstm", choices=["xlstm", "transformer"],
    )
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading tokenizer: {args.tokenizer}")
    tok = load_tokenizer(args.tokenizer)
    print(f"loading model: arch={args.arch}, config={args.config}")
    model = build_model_for_arch(args.arch, args.config).to(device)
    print(f"loading checkpoint: {args.ckpt}")
    load_checkpoint(Path(args.ckpt), model, map_location=device)
    model.eval()

    repl(
        model, tok, device,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
