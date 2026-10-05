"""Mutual k-nearest-neighbour alignment of paired representations.

Distances use float32 squared Euclidean geometry.
"""
from __future__ import annotations

import numpy as np
import torch


def _neighbours(x, squares, start, stop, k):
    block = x[start:stop]
    distances = (
        squares[start:stop, None] + squares[None, :] - 2.0 * (block @ x.T)
    ).clamp_min_(0)
    rows = torch.arange(stop - start, device=x.device)
    distances[rows, torch.arange(start, stop, device=x.device)] = float("inf")
    values, indices = torch.topk(distances, k + 1, dim=1, largest=False, sorted=True)
    return indices[:, :k], values[:, k - 1] == values[:, k]


def mutual_knn(a, b, k=10, device="cpu", chunk=2048):
    """Return mean neighbour overlap and diagnostics for paired row matrices.

    The score is mean_i |N_a(i) intersect N_b(i)| / k, with self excluded.
    Higher means greater alignment; the chance level is k / (n - 1).
    Rows must describe the same objects in the same order. Feature dimensions
    may differ. At least k + 2 rows are needed to also diagnose boundary ties.
    Exact ties use PyTorch's topk ordering.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or not a.shape[1] or not b.shape[1]:
        raise ValueError("inputs must be row matrices with nonempty features")
    if len(a) != len(b):
        raise ValueError("inputs must contain the same number of paired objects")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if len(a) < k + 2:
        raise ValueError(f"k={k} requires at least {k + 2} objects")
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("inputs must be finite")

    n = len(a)
    x, y = (torch.tensor(v, dtype=torch.float32, device=device) for v in (a, b))
    nx, ny = x.square().sum(1), y.square().sum(1)
    overlap, tied = 0, [0, 0]
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        ia, ta = _neighbours(x, nx, start, stop, k)
        ib, tb = _neighbours(y, ny, start, stop, k)
        overlap += int((ia[:, :, None] == ib[:, None, :]).any(dim=2).sum())
        tied[0] += int(ta.sum())
        tied[1] += int(tb.sum())
    return {
        "mknn": overlap / (n * k),
        "k": k,
        "n": n,
        "metric": "euclidean",
        "chance": k / (n - 1),
        "shared_neighbours": overlap,
        "tied_boundary_rows": tied,
    }
