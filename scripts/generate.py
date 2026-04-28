"""Gera amostras do modelo treinado pra inspecionar qualitativamente.

Útil pra POC: depois de 500 steps você não espera coerência, mas se o
modelo gera lixo aleatório vs. fragmentos de palavras reais, isso te
diz se algo está errado.

Funciona com xLSTM ou Transformer — passa --arch.

Uso:
    python -m scripts.generate \\
        --ckpt runs/xlstm_tiny/best.pt \\
        --config configs/xlstm_tiny.yaml \\
        --arch xlstm \\
        --prompt "Capitu olhou"

    python -m scripts.generate \\
        --ckpt runs/transformer_tiny/best.pt \\
        --config configs/transformer_tiny.yaml \\
        --arch transformer \\
        --prompt "Capitu olhou"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.tokenizer import load_tokenizer
from src.model_factory import build_model_for_arch
from src.training import load_checkpoint


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Geração autoregressiva token-a-token com top-k sampling."""
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # xLSTMLMModel processa toda a sequência de uma vez (sem cache);
        # pra POC tudo bem, geração fica O(n^2) mas n é pequeno.
        logits = model(x)
        logits = logits[0, -1, :] / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        x = torch.cat(
            [x, torch.tensor([[next_id]], device=device)], dim=1
        )

        # truncamento pelo context length do modelo
        if x.size(1) > model.config.context_length:
            x = x[:, -model.config.context_length:]

    return tokenizer.decode(x[0].tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--arch", default="xlstm", choices=["xlstm", "transformer"],
    )
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--prompt", default="O sol")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer)
    model = build_model_for_arch(args.arch, args.config).to(device)
    load_checkpoint(Path(args.ckpt), model, map_location=device)

    print(f"prompt: {args.prompt!r}\n")
    out = generate(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(out)
