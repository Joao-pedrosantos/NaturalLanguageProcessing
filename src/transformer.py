"""GPT-style decoder Transformer baseline para comparação com xLSTM.

Arquitetura padrão do nanoGPT/GPT-2 small:
- Embedding de token + embedding posicional aprendido
- Pre-norm (LayerNorm antes de attn/MLP)
- Multi-head attention causal via F.scaled_dot_product_attention
- MLP com expansão 4x e GELU
- Output head untied (matchando o xLSTM)

Dimensionado pra ~1.46M params com (vocab=4000, emb=128, blocks=2,
heads=4, ctx=256), batendo com o xLSTM tiny pra comparação justa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf


@dataclass
class TransformerConfig:
    vocab_size: int
    embedding_dim: int
    num_blocks: int
    num_heads: int
    context_length: int
    mlp_expansion: int = 4
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Atenção causal multi-head usando o kernel fundido do PyTorch.

    F.scaled_dot_product_attention escolhe automaticamente o caminho mais
    rápido disponível (FlashAttention em GPU CUDA, kernel matemático em
    CPU). is_causal=True aplica a máscara triangular sem custo extra.
    """

    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({emb_dim}) precisa ser divisível por "
                f"num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, emb_dim: int, expansion: int, dropout: float = 0.0):
        super().__init__()
        hidden = expansion * emb_dim
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self, emb_dim: int, num_heads: int,
        mlp_expansion: int, dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(emb_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, mlp_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: estabiliza gradientes em modelos profundos vs post-norm
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """LM autoregressivo causal. Interface compatível com xLSTMLMModel:
    forward(input_ids) -> logits de shape (B, T, V)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.embedding_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim, config.num_heads,
                config.mlp_expansion, config.dropout,
            )
            for _ in range(config.num_blocks)
        ])
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Inicialização à GPT-2: normal(0, 0.02) em pesos, zero em bias.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.config.context_length:
            raise ValueError(
                f"sequência de comprimento {T} excede context_length "
                f"({self.config.context_length})"
            )
        pos = torch.arange(T, device=input_ids.device)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


def load_transformer_config(path: str | Path) -> TransformerConfig:
    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    return from_dict(
        data_class=TransformerConfig,
        data=raw,
        config=DaciteConfig(strict=True),
    )


def build_transformer(cfg: TransformerConfig) -> TransformerLM:
    return TransformerLM(cfg)
