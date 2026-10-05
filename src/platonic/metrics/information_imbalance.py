"""Directional and symmetric Information Imbalance of paired representations."""
from __future__ import annotations

import numpy as np
import torch


def information_imbalance(a, b, device="cpu", chunk=512):
    """Measure how well each space's nearest neighbours are preserved.

    delta_XY = (2 / n) * mean_i rank_Y(i, nearest_X(i)).
    The reverse direction exchanges X and Y; symmetric is their average.
    Smaller values indicate better alignment.

    Inputs are paired row matrices; their feature dimensions may differ.
    Squared Euclidean distances use float32. Self is excluded. Ranks are
    one plus the number of strictly smaller distances (minimum rank for
    ties); the source neighbour is chosen with PyTorch argmin.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or not a.shape[1] or not b.shape[1]:
        raise ValueError("inputs must be row matrices with nonempty features")
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("II requires at least two paired objects")
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("inputs must be finite")

    x, y = (torch.tensor(v, dtype=torch.float32, device=device) for v in (a, b))
    n = len(x)
    nx, ny = x.square().sum(1), y.square().sum(1)
    sums = [0, 0]
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        dx = (nx[start:stop, None] + nx[None, :] - 2 * x[start:stop] @ x.T).clamp_min_(0)
        dy = (ny[start:stop, None] + ny[None, :] - 2 * y[start:stop] @ y.T).clamp_min_(0)
        rows = torch.arange(stop - start, device=device)
        diagonal = torch.arange(start, stop, device=device)
        dx[rows, diagonal] = float("inf")
        dy[rows, diagonal] = float("inf")
        for direction, source, target in ((0, dx, dy), (1, dy, dx)):
            nearest = source.argmin(1)
            threshold = target[rows, nearest][:, None]
            ranks = 1 + (target < threshold).sum(1)
            sums[direction] += int(ranks.sum())
    ab, ba = (2.0 * value / (n * n) for value in sums)
    return {"delta_XY": ab, "delta_YX": ba, "symmetric": 0.5 * (ab + ba)}
