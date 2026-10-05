#!/usr/bin/env python3
"""Mutual-kNN alignment with an audit for exact neighborhood-boundary ties."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MutualKNNResult:
    mutual_knn: float
    standard_mutual_knn: float
    tie_averaging_used: bool
    boundary_tie_fraction_x: float
    boundary_tie_fraction_y: float
    boundary_tie_fraction_either: float
    max_boundary_tie_group_x: int
    max_boundary_tie_group_y: int
    k: int
    n: int
    chance_baseline: float
    tie_rtol: float
    tie_atol: float

    def to_dict(self):
        return asdict(self)


def _as_normalized_features(features):
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"Expected [N, D] features, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Features contain NaN or infinity")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return np.divide(value, norm, out=np.zeros_like(value), where=norm > 0.0)


def _neighbor_membership(features, k, tie_rtol, tie_atol):
    features = _as_normalized_features(features)
    n = features.shape[0]
    if not 1 <= int(k) < n:
        raise ValueError(f"k must lie in 1,...,N-1; found k={k}, N={n}")

    similarities = features @ features.T
    np.fill_diagonal(similarities, -np.inf)

    # This is the literal Platonic-Hypothesis implementation apart from using
    # argpartition rather than a full sort: choose exactly k largest entries.
    standard_indices = np.argpartition(similarities, -int(k), axis=1)[:, -int(k):]

    # The kth-largest value defines the neighborhood boundary.  If a tie group
    # crosses that boundary, give every member the probability with which it
    # would enter under an independent uniform random tie break.  The resulting
    # overlap is the exact expectation over random tie breaks in the two spaces.
    threshold = np.partition(similarities, -int(k), axis=1)[:, -int(k)]
    boundary_equal = np.isclose(
        similarities,
        threshold[:, None],
        rtol=float(tie_rtol),
        atol=float(tie_atol),
        equal_nan=False,
    )
    strictly_greater = (similarities > threshold[:, None]) & ~boundary_equal
    num_greater = strictly_greater.sum(axis=1)
    tie_group_size = boundary_equal.sum(axis=1)
    remaining = int(k) - num_greater
    if np.any(remaining < 1) or np.any(remaining > tie_group_size):
        raise AssertionError("Invalid kth-neighbor tie accounting")

    boundary_cut = tie_group_size > remaining
    tie_probability = remaining / tie_group_size
    membership = strictly_greater.astype(np.float32)
    membership += boundary_equal.astype(np.float32) * tie_probability[:, None]
    if not np.allclose(membership.sum(axis=1), float(k), rtol=0.0, atol=2e-5):
        raise AssertionError("Tie-averaged neighborhood mass does not sum to k")

    audit = {
        "boundary_cut": boundary_cut,
        "boundary_tie_fraction": float(boundary_cut.mean()),
        "max_boundary_tie_group": int(tie_group_size[boundary_cut].max())
        if np.any(boundary_cut)
        else 1,
    }
    return standard_indices, membership, audit


def mutual_knn(features_x, features_y, k=10, tie_rtol=1e-10, tie_atol=1e-12):
    """Mean k-neighborhood overlap, with exact averaging at tied boundaries.

    Rows of ``features_x`` and ``features_y`` must describe corresponding
    samples.  Each row is L2-normalized, self-neighbors are excluded, and
    cosine similarity determines the neighborhoods.
    """
    x = np.asarray(features_x)
    y = np.asarray(features_y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Expected paired [N, D] matrices, got {x.shape} and {y.shape}")
    n = int(x.shape[0])

    indices_x, membership_x, audit_x = _neighbor_membership(
        x, int(k), float(tie_rtol), float(tie_atol)
    )
    indices_y, membership_y, audit_y = _neighbor_membership(
        y, int(k), float(tie_rtol), float(tie_atol)
    )

    mask_x = np.zeros((n, n), dtype=bool)
    mask_y = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask_x[rows, indices_x] = True
    mask_y[rows, indices_y] = True
    standard = float(np.mean(np.sum(mask_x & mask_y, axis=1) / float(k)))

    expected = float(
        np.mean(np.sum(membership_x * membership_y, axis=1) / float(k))
    )
    boundary_either = audit_x["boundary_cut"] | audit_y["boundary_cut"]
    if not np.any(boundary_either):
        # Preserve the literal literature implementation exactly when its
        # no-ties assumption holds; fractional memberships are needed only for
        # anchors whose kth-neighbor boundary actually cuts a tie group.
        expected = standard
    return MutualKNNResult(
        mutual_knn=expected,
        standard_mutual_knn=standard,
        tie_averaging_used=bool(np.any(boundary_either)),
        boundary_tie_fraction_x=audit_x["boundary_tie_fraction"],
        boundary_tie_fraction_y=audit_y["boundary_tie_fraction"],
        boundary_tie_fraction_either=float(boundary_either.mean()),
        max_boundary_tie_group_x=audit_x["max_boundary_tie_group"],
        max_boundary_tie_group_y=audit_y["max_boundary_tie_group"],
        k=int(k),
        n=n,
        chance_baseline=float(k) / float(n - 1),
        tie_rtol=float(tie_rtol),
        tie_atol=float(tie_atol),
    )


def _self_test():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    identical_geometry = mutual_knn(x, x @ q, k=10)
    if not np.isclose(identical_geometry.mutual_knn, 1.0):
        raise AssertionError(identical_geometry)
    tied = mutual_knn(np.zeros((21, 3)), np.zeros((21, 5)), k=10)
    if not np.isclose(tied.mutual_knn, 10.0 / 20.0):
        raise AssertionError(tied)
    if not tied.tie_averaging_used:
        raise AssertionError("All-equal features must trigger tie averaging")
    print("mutual_knn self-test passed")


if __name__ == "__main__":
    _self_test()
