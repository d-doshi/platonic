"""Exact softmax-attention GPT architecture used by the paper checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    width: int = 512
    heads: int = 4
    layers: int = 8
    activation: str = "relu"
    weight_sharing: bool = True

    def validate(self) -> None:
        if min(self.vocab_size, self.context_length, self.width, self.heads, self.layers) < 1:
            raise ValueError("transformer dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransformerConfig":
        result = cls(**value)
        result.validate()
        return result


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.heads = config.heads
        self.width = config.width
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.projection = nn.Linear(config.width, config.width, bias=False)
        self.projection.scaled_residual_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        query, key, value = self.qkv(x).split(self.width, dim=2)
        head_width = width // self.heads
        query = query.view(batch, length, self.heads, head_width).transpose(1, 2)
        key = key.view(batch, length, self.heads, head_width).transpose(1, 2)
        value = value.view(batch, length, self.heads, head_width).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(output)


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.expand = nn.Linear(config.width, 4 * config.width, bias=False)
        self.activation = nn.ReLU() if config.activation == "relu" else nn.GELU()
        self.project = nn.Linear(4 * config.width, config.width, bias=False)
        self.project.scaled_residual_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.activation(self.expand(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.width, bias=False)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.width, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CausalTransformer(nn.Module):
    """NanoGPT-style model matching ``models/gpt2.py`` in the research repository."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width, bias=False)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        if config.weight_sharing:
            self.token_embedding.weight = self.output.weight
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "scaled_residual_init", False):
                std *= (2 * self.config.layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError("input exceeds context_length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        features = {"embedding": x} if return_features else {}
        for index, block in enumerate(self.blocks):
            x = block(x)
            if return_features:
                features[f"block.{index}"] = x
        logits = self.output(self.final_norm(x))
        return (logits, features) if return_features else logits

    def optimizer_groups(self, weight_decay: float):
        parameters = {name: value for name, value in self.named_parameters() if value.requires_grad}
        embedding_names = {"token_embedding.weight", "position_embedding.weight"}
        decay = [
            value
            for name, value in parameters.items()
            if value.dim() >= 2 and name not in embedding_names
        ]
        no_decay = [
            value
            for name, value in parameters.items()
            if value.dim() < 2 or name in embedding_names
        ]
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
