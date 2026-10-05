"""Shared II, mKNN, novelty, direct-difference, and latent-probe analysis."""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import DataConfig, token_latent_labels
from .metrics import (
    Accumulator,
    direct_subtraction,
    fit_novelty,
    information_imbalance,
    mutual_knn,
    novelty_residuals,
)
from .probes import fit_ridge_classifier, probe_accuracy


def _metric_rows(
    a: np.ndarray,
    b: np.ndarray,
    *,
    source: str,
    representation: str,
    stage: str,
    from_stage: str = "",
    k: int,
    device: str,
) -> list[dict[str, Any]]:
    ii = information_imbalance(a, b, device=device)
    knn = mutual_knn(a, b, k=k, device=device)
    common = {
        "source": source,
        "representation": representation,
        "from_stage": from_stage,
        "stage": stage,
    }
    return [
        {**common, "metric": "ii_a_to_b", "value": ii["delta_XY"]},
        {**common, "metric": "ii_b_to_a", "value": ii["delta_YX"]},
        {**common, "metric": "ii_symmetric", "value": ii["symmetric"]},
        {**common, "metric": "mknn", "value": knn["mknn"]},
        {**common, "metric": "mknn_chance", "value": knn["chance"]},
    ]


def _target_rms(fit: np.ndarray, validation: np.ndarray, fit_fraction: float) -> float:
    fit_mean = fit.mean(axis=1)
    validation_mean = validation.mean(axis=1)
    squared = fit_fraction * np.mean(np.sum(fit_mean * fit_mean, axis=1))
    squared += (1.0 - fit_fraction) * np.mean(
        np.sum(validation_mean * validation_mean, axis=1)
    )
    return float(np.sqrt(max(float(squared), 1e-24)))


def _moments(a_previous, a_current, b_previous, b_current, device):
    width = a_previous.shape[-1]
    accumulator = Accumulator(width, device=device)
    sentences = len(a_previous) + len(b_previous)
    for previous, current in (
        (a_previous, a_current),
        (b_previous, b_current),
    ):
        rows = previous.reshape(-1, width)
        targets = current.reshape(-1, width)
        weights = np.full(
            len(rows), 1.0 / (sentences * previous.shape[1]), dtype=np.float64
        )
        accumulator.add(rows, targets, weights)
    return accumulator.finish()


def analyze_feature_pairs(
    features_a: dict[str, np.ndarray],
    features_b: dict[str, np.ndarray],
    shared_levels: list[np.ndarray],
    config: DataConfig,
    *,
    source: str,
    fit_count: int,
    validation_count: int,
    test_count: int,
    k: int,
    device: str = "cpu",
    ridge: float = 1e-3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze aligned language-A/B features using deterministic contiguous splits."""
    names = list(features_a)
    if names != list(features_b) or not names:
        raise ValueError("the two feature dictionaries must have identical ordered stages")
    total = fit_count + validation_count + test_count
    if any(len(features_a[name]) != total for name in names):
        raise ValueError("feature count does not match the requested data splits")
    if test_count < k + 2:
        raise ValueError("test_count must be at least k + 2")
    fit = slice(0, fit_count)
    validation = slice(fit_count, fit_count + validation_count)
    test = slice(fit_count + validation_count, total)

    metric_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for name in names:
        pooled_a = features_a[name][test].mean(axis=1)
        pooled_b = features_b[name][test].mean(axis=1)
        metric_rows.extend(
            _metric_rows(
                pooled_a,
                pooled_b,
                source=source,
                representation="raw",
                stage=name,
                k=k,
                device=device,
            )
        )
        for latent_depth in range(config.shared_depth + 1):
            labels = token_latent_labels(shared_levels, config, latent_depth)[:, :-1]
            x_train = features_a[name][fit].reshape(-1, features_a[name].shape[-1])
            y_train = labels[fit].reshape(-1)
            weight, bias = fit_ridge_classifier(
                x_train, y_train, config.vocab_size, ridge=ridge
            )
            for language, values in (("a", features_a[name]), ("b", features_b[name])):
                accuracy = probe_accuracy(
                    values[test].reshape(-1, values.shape[-1]),
                    labels[test].reshape(-1),
                    weight,
                    bias,
                )
                probe_rows.append(
                    {
                        "source": source,
                        "stage": name,
                        "latent_depth": latent_depth,
                        "train_language": "a",
                        "test_language": language,
                        "accuracy": accuracy,
                        "chance": 1.0 / config.vocab_size,
                    }
                )

    fit_fraction = fit_count / (fit_count + validation_count)
    for previous_name, current_name in zip(names[:-1], names[1:]):
        pa, ca = features_a[previous_name], features_a[current_name]
        pb, cb = features_b[previous_name], features_b[current_name]
        if pa.shape[-1] != ca.shape[-1] or pb.shape[-1] != cb.shape[-1]:
            continue
        fit_stats = _moments(pa[fit], ca[fit], pb[fit], cb[fit], device)
        validation_stats = _moments(
            pa[validation], ca[validation], pb[validation], cb[validation], device
        )
        weight, bias, selection = fit_novelty(
            fit_stats,
            validation_stats,
            fit_fraction=fit_fraction,
            device=device,
        )
        selection_rows.append(
            {
                "source": source,
                "from_stage": previous_name,
                "stage": current_name,
                **selection["selected"],
            }
        )
        scale_a = _target_rms(ca[fit], ca[validation], fit_fraction)
        scale_b = _target_rms(cb[fit], cb[validation], fit_fraction)
        pa_test, ca_test = pa[test].mean(axis=1), ca[test].mean(axis=1)
        pb_test, cb_test = pb[test].mean(axis=1), cb[test].mean(axis=1)
        novelty_a = novelty_residuals(pa_test, ca_test, weight, bias, scale_a, device)
        novelty_b = novelty_residuals(pb_test, cb_test, weight, bias, scale_b, device)
        direct_a = direct_subtraction(pa_test, ca_test, scale_a)
        direct_b = direct_subtraction(pb_test, cb_test, scale_b)
        for representation, a, b in (
            ("novelty", novelty_a, novelty_b),
            ("direct_subtraction", direct_a, direct_b),
        ):
            metric_rows.extend(
                _metric_rows(
                    a,
                    b,
                    source=source,
                    representation=representation,
                    from_stage=previous_name,
                    stage=current_name,
                    k=k,
                    device=device,
                )
            )
    return metric_rows, probe_rows, selection_rows
