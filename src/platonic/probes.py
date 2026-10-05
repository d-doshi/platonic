"""Closed-form ridge probes for categorical MRHM latents."""
from __future__ import annotations

import numpy as np


def fit_ridge_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("features and labels are not aligned")
    centered = features - features.mean(axis=0, keepdims=True)
    targets = np.eye(num_classes, dtype=np.float64)[labels]
    target_mean = targets.mean(axis=0)
    gram = centered.T @ centered
    gram.flat[:: len(gram) + 1] += ridge * len(features)
    weight = np.linalg.solve(gram, centered.T @ (targets - target_mean))
    bias = target_mean - features.mean(axis=0) @ weight
    return weight, bias


def probe_accuracy(
    features: np.ndarray, labels: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> float:
    predictions = np.argmax(np.asarray(features) @ weight + bias, axis=1)
    return float(np.mean(predictions == np.asarray(labels)))
