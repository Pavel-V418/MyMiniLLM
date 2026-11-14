"""
GPT-2-подобная модель (этап 0). На этапе 2 будем менять блоки по одному:
LayerNorm -> RMSNorm, GELU-MLP -> SwiGLU, позиционные эмбеддинги -> RoPE, MHA -> GQA.

Поток данных:
    id токенов (B, T)
      -> эмбеддинг токена + эмбеддинг позиции     (B, T, C)
      -> n_layer x Block:  x = x + Attn(LN(x));  x = x + MLP(LN(x))
      -> LayerNorm -> lm_head                      (B, T, vocab_size)  логиты следующего токена
B — batch, T — длина контекста, C — n_embd (ширина модели).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16000
    block_size: int = 256  # максимальная длина контекста
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False  # bias в Linear/LayerNorm; без него чуть быстрее и не хуже


class CausalSelfAttention(nn.Module):
    """Каждый токен смотрит на себя и на предыдущие токены (но не на будущие)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)  # Q, K, V одним матричным умножением
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): каждая голова работает со своим подпространством
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # softmax(Q K^T / sqrt(d) + causal_mask) V — та же формула, что в статье
        # «Attention is all you need»; PyTorch считает её оптимизированным ядром.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # склеиваем головы обратно
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Позиционно-независимая сеть: расширяем в 4 раза, GELU, сжимаем обратно."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN блок трансформера: нормализация перед подслоем, затем residual."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # токены обмениваются информацией
        x = x + self.mlp(self.ln2(x))  # каждый токен «обдумывает» полученное
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: входной эмбеддинг и выходной слой — одна матрица.
        # Экономит vocab_size * n_embd параметров (у нас 6M) и обычно улучшает качество.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2: выходные проекции residual-веток инициализируем меньше,
        # чтобы дисперсия не росла с глубиной (в residual складывается 2*n_layer веток).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # общая матрица считается один раз
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) id токенов. targets: (B, T) — те же id, сдвинутые на 1 вперёд."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"длина {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            # при генерации нужен только последний шаг — экономим вычисления
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        # cross-entropy = -log(вероятность правильного следующего токена), среднее по всем позициям
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Авторегрессия: предсказали токен -> дописали в контекст -> повторили."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size :]  # контекст не длиннее block_size
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # temperature < 1 — увереннее, > 1 — разнообразнее
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")  # оставляем только top_k кандидатов
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str):
        """AdamW: weight decay только для матриц (2D), не для LayerNorm и bias."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        extra = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)
