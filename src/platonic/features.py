"""Feature extraction shared by checkpoint metrics and latent probes."""
from __future__ import annotations

import numpy as np
import torch

from .model import CausalTransformer


def extract_transformer_features(
    model: CausalTransformer,
    tokens: np.ndarray,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Return every residual-stream stage as ``[examples, tokens, width]``."""
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(
                tokens[start : start + batch_size, :-1], dtype=torch.long, device=device
            )
            _, features = model(batch, return_features=True)
            for name, value in features.items():
                collected.setdefault(name, []).append(value.detach().cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in collected.items()}


def sentence_means(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.mean(axis=1) for name, value in features.items()}

