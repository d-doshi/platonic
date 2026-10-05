"""End-to-end training and checkpoint-analysis workflows."""
from __future__ import annotations

import copy
import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from .analysis import analyze_feature_pairs
from .bp.experiments import run_suite as run_bp_suite
from .data import sample_translation_pairs
from .features import extract_transformer_features
from .training import load_checkpoint, resolve_device, train_from_config


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_environment_manifest(path: Path, config: dict, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    path.write_text(json.dumps({
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint),
        "derived_seeds": payload["seeds"],
        "best_step": payload["best_step"],
        "config": config,
    }, indent=2, default=str) + "\n")


def analyze_checkpoint(
    checkpoint: str | Path,
    config: dict,
    output_dir: str | Path,
    *,
    device: str = "auto",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = config["analysis"]
    resolved_device = resolve_device(device)
    model, data_config, rules, payload = load_checkpoint(checkpoint, resolved_device)
    splits = [
        sample_translation_pairs(
            int(analysis[f"{name}_pairs"]), data_config, rules,
            int(analysis[f"{name}_seed"]),
        )
        for name in ("fit", "validation", "test")
    ]
    pairs = {
        "tokens_a": np.concatenate([part["tokens_a"] for part in splits]),
        "tokens_b": np.concatenate([part["tokens_b"] for part in splits]),
        "shared_levels": [
            np.concatenate([part["shared_levels"][level] for part in splits])
            for level in range(len(splits[0]["shared_levels"]))
        ],
    }
    kwargs = {
        "batch_size": int(analysis["feature_batch_size"]),
        "device": resolved_device,
    }
    features_a = extract_transformer_features(model, pairs["tokens_a"], **kwargs)
    features_b = extract_transformer_features(model, pairs["tokens_b"], **kwargs)
    metrics, probes, selections = analyze_feature_pairs(
        features_a, features_b, pairs["shared_levels"], data_config,
        source="transformer",
        fit_count=int(analysis["fit_pairs"]),
        validation_count=int(analysis["validation_pairs"]),
        test_count=int(analysis["test_pairs"]),
        k=int(analysis["k"]),
        device=str(analysis.get("metric_device", "cpu")),
        ridge=float(analysis.get("probe_ridge", 1e-3)),
    )
    _write_csv(output_dir / "metrics.csv", metrics)
    _write_csv(output_dir / "probes.csv", probes)
    _write_csv(output_dir / "novelty_selection.csv", selections)
    (output_dir / "manifest.json").write_text(json.dumps({
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_epoch": payload["epoch"],
        "analysis_seeds": {
            name: int(analysis[f"{name}_seed"])
            for name in ("fit", "validation", "test")
        },
        "stages": list(features_a),
    }, indent=2) + "\n")
    return output_dir


def run_one(config: dict, output_dir: str | Path, *, include_bp: bool = True) -> Path:
    output_dir = Path(output_dir)
    checkpoint = train_from_config(config, output_dir)
    analyze_checkpoint(checkpoint, config, output_dir / "transformer")
    if include_bp:
        run_bp_suite(output_dir / "bp", config)
    write_environment_manifest(output_dir / "run_manifest.json", config, checkpoint)
    return checkpoint


def reproduce(config: dict, output_dir: str | Path) -> list[Path]:
    sweep = config.get("sweep")
    if not sweep:
        return [run_one(config, output_dir)]
    checkpoints = []
    for shared_depth in sweep["shared_depths"]:
        for seed in sweep["random_seeds"]:
            resolved = copy.deepcopy(config)
            resolved.pop("sweep", None)
            resolved["data"]["shared_depth"] = int(shared_depth)
            resolved["random_seed"] = int(seed)
            run_dir = Path(output_dir) / f"shared_depth_{shared_depth}" / f"seed_{seed}"
            checkpoints.append(run_one(resolved, run_dir))
    return checkpoints
