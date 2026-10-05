"""Adjacent-layer novelty with validation-MSE-selected affine ridge regression.

Regression calculations use float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


LAMBDAS = tuple(10.0 ** (-8 + 0.5 * i) for i in range(33))


@dataclass
class Statistics:
    """Unit-mass raw moments: E[X], E[Y], E[X'X], E[X'Y], E[Y**2]."""

    mx: np.ndarray
    my: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    y2: np.ndarray

    def centered(self):
        return self.xx - np.outer(self.mx, self.mx), self.xy - np.outer(self.mx, self.my)

    def mix(self, other, fraction):
        """Combine fit/validation moments using the fit fraction of sentences."""
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("fraction must be between zero and one")
        if any(getattr(self, name).shape != getattr(other, name).shape
               for name in self.__dataclass_fields__):
            raise ValueError("statistics must have matching dimensions")
        return Statistics(*(
            fraction * getattr(self, name) + (1 - fraction) * getattr(other, name)
            for name in self.__dataclass_fields__
        ))


class Accumulator:
    """Accumulate weighted regression moments from token representations.

    For S sentences, each valid token in a sentence of length T has weight
    1 / (S * T). For a shared bilingual map, S counts sentences from both
    languages. Add both languages to the same accumulator. Adjacent layers
    must have the same width. Supply float64 weights summing to one over
    all add() calls.
    """

    def __init__(self, width, device="cpu"):
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.device = device
        self.values = [
            torch.zeros(shape, dtype=torch.float64, device=device)
            for shape in ((width,), (width,), (width, width), (width, width), (width,))
        ]
        self.mass = 0.0

    def add(self, x, y, weights):
        x, y, weights = (
            torch.as_tensor(value, dtype=torch.float64, device=self.device)
            for value in (x, y, weights)
        )
        if (x.ndim != 2 or x.shape != y.shape or x.shape[1] != self.width
                or not len(x) or weights.shape != (len(x),)):
            raise ValueError("regression rows and weights must be aligned")
        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()
                    & torch.isfinite(weights).all() & (weights > 0).all()):
            raise ValueError("rows must be finite and weights positive and finite")
        mx, my, xx, xy, y2 = self.values
        mx += (x * weights[:, None]).sum(0)
        my += (y * weights[:, None]).sum(0)
        xx += x.T @ (x * weights[:, None])
        xy += x.T @ (y * weights[:, None])
        y2 += (y.square() * weights[:, None]).sum(0)
        self.mass += float(weights.sum())

    def finish(self):
        if abs(self.mass - 1.0) >= 1e-10:
            raise ValueError(f"weights sum to {self.mass}, expected one")
        return Statistics(*(value.cpu().numpy().copy() for value in self.values))


def fit_ridge(stats, ridge_lambda, device="cpu"):
    """Fit W, b minimizing E[||Y - XW - b||**2] + lambda * ||W||_F**2.

    The intercept is unpenalized. Moments have unit mass, so lambda is the
    penalty for the weighted mean loss. Returns float64 NumPy arrays (W, b).
    """
    if not np.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive and finite")
    gram, cross = stats.centered()
    shifted = torch.tensor(gram, dtype=torch.float64, device=device)
    shifted.diagonal().add_(ridge_lambda)
    factor, info = torch.linalg.cholesky_ex(shifted, check_errors=False)
    if int(info.max()) != 0:
        raise ValueError(f"ridge system is not positive definite at lambda={ridge_lambda}")
    del shifted
    weight = np.empty(cross.shape, dtype=np.float64)
    for start in range(0, cross.shape[1], 128):
        stop = min(cross.shape[1], start + 128)
        rhs = torch.tensor(cross[:, start:stop], dtype=torch.float64, device=device)
        weight[:, start:stop] = torch.cholesky_solve(rhs, factor).cpu().numpy()
    bias = stats.my - stats.mx @ weight
    if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
        raise ValueError("nonfinite ridge coefficients")
    return weight, bias


def validation_mse(stats, weight, bias, device="cpu"):
    """Weighted held-out prediction MSE, averaged over output coordinates.

    Uses sufficient statistics, with the same token weights as fitting.
    This equals sum_i weight_i * mean_j (Y_ij - prediction_ij)**2.
    """
    gram, cross = stats.centered()
    variance = np.maximum(stats.y2 - np.square(stats.my), 0.0)
    linear = np.sum(weight * cross, axis=0)
    quadratic = np.zeros(weight.shape[1], dtype=np.float64)
    wt = torch.tensor(weight, dtype=torch.float64, device=device)
    for start in range(0, len(weight), 2048):
        stop = min(start + 2048, len(weight))
        block = torch.tensor(gram[start:stop], dtype=torch.float64, device=device)
        product = block @ wt
        quadratic += (wt[start:stop] * product).sum(0).cpu().numpy()
    bias_error = np.square(stats.my - stats.mx @ weight - bias)
    error = variance - 2 * linear + quadratic + bias_error
    scale = np.maximum(variance + 2 * np.abs(linear) + np.abs(quadratic) + bias_error, 1e-24)
    if not (np.isfinite(error).all() and np.all(error >= -1e-9 * scale)):
        raise ValueError("invalid validation squared error")
    return float(np.maximum(error, 0.0).mean())


def select_mse(fit_stats, validation_stats, device="cpu", lambdas=LAMBDAS):
    """Select lambda by minimum validation MSE; exact ties use smaller lambda.

    Each candidate fits fit_stats and is evaluated on validation_stats.
    Returns the chosen candidate and the errors for the full penalty grid.
    """
    grid = tuple(sorted(float(value) for value in lambdas))
    if (not grid or len(grid) != len(set(grid))
            or not all(np.isfinite(value) and value > 0 for value in grid)):
        raise ValueError("lambdas must be distinct, positive, finite values")
    candidates = []
    for ridge_lambda in grid:
        weight, bias = fit_ridge(fit_stats, ridge_lambda, device)
        error = validation_mse(validation_stats, weight, bias, device)
        candidates.append({"ridge_lambda": ridge_lambda, "validation_mse": error})
    best = min(candidates, key=lambda row: (row["validation_mse"], row["ridge_lambda"]))
    selected = {
        **best,
        "at_lower_boundary": best["ridge_lambda"] == grid[0],
        "at_upper_boundary": best["ridge_lambda"] == grid[-1],
    }
    return {"selected": selected, "candidates": candidates}


def fit_novelty(fit_stats, validation_stats, *, fit_fraction, device="cpu", lambdas=LAMBDAS):
    """Select by validation MSE, then refit on fit plus validation moments.

    fit_fraction = N_fit / (N_fit + N_validation), using sentence counts.
    Returns (W, b, selection), with validation errors measured before refitting.
    """
    combined = fit_stats.mix(validation_stats, fit_fraction)
    selection = select_mse(fit_stats, validation_stats, device, lambdas)
    weight, bias = fit_ridge(combined, selection["selected"]["ridge_lambda"], device)
    return weight, bias, selection


def novelty_residuals(previous, current, weight, bias, scale, device="cpu", chunk=512):
    """Return (current - (previous @ W + b)) / scale as float32 rows.

    Apply the same fitted W and b to each language. scale is the positive
    target RMS for that language/layer from fit plus validation sentences.
    Use scale=1.0 for unnormalized residuals. After fitting on token rows,
    inputs can be sentence means: averaging commutes with the affine residual.
    """
    previous, current, weight, bias = (
        np.asarray(value) for value in (previous, current, weight, bias)
    )
    if (previous.ndim != 2 or current.ndim != 2 or not previous.size or not current.size
            or len(previous) != len(current)
            or weight.shape != (previous.shape[1], current.shape[1])
            or bias.shape != (current.shape[1],)):
        raise ValueError("representations and affine coefficients have incompatible shapes")
    if not all(np.isfinite(value).all() for value in (previous, current, weight, bias)):
        raise ValueError("representations and coefficients must be finite")
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite scalar")
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    wt, bt = (torch.as_tensor(value, dtype=torch.float64, device=device) for value in (weight, bias))
    result = np.empty(current.shape, dtype=np.float32)
    for start in range(0, len(previous), chunk):
        stop = min(len(previous), start + chunk)
        x = torch.tensor(previous[start:stop], dtype=torch.float64, device=device)
        y = torch.tensor(current[start:stop], dtype=torch.float64, device=device)
        result[start:stop] = ((y - (x @ wt + bt)) / scale).cpu().numpy()
    return result
