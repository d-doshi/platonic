"""Paper-facing settled-BP experiments."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from . import core
from .mutual_knn import mutual_knn


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ii(
    output_dir: str | Path,
    *,
    shared_depth: int,
    seed: int,
    n: int = 2048,
    max_positions: int = 0,
    message_layout: str = core.DEFAULT_MESSAGE_LAYOUT,
) -> Path:
    """Run the complete paper BP Information-Imbalance sweep."""
    started = time.perf_counter()
    config = core.SweepConfig(seed=int(seed))
    core.validate_config(config, int(shared_depth))
    rng, rules = core.build_language_rules(config, int(shared_depth), seed_offset=0)
    trees = core.generate_translation_pair_trees(n, config, shared_depth, rules, rng)
    positions = core.select_positions(config, max_positions, seed + 991 * shared_depth)
    raw = {
        language: core.theoretical_atomic(
            trees[language][0], rules[language], config, positions,
            message_layout=message_layout,
        )
        for language in (0, 1)
    }
    rows = []
    for feature_encoding in core.FEATURE_ENCODINGS:
        for algorithm in core.ALGORITHMS:
            maps, alignment = core.encoding_maps(config, shared_depth, algorithm)
            for retention_mode in core.RETENTION_MODES:
                states = {
                    language: core.algorithm_states(
                        raw[language], maps[language], algorithm,
                        feature_encoding, retention_mode, config,
                    )
                    for language in (0, 1)
                }
                for stage_index, (state_a, state_b) in enumerate(
                    zip(states[0], states[1]), start=1
                ):
                    a = np.asarray(state_a["features"]).mean(axis=1)
                    b = np.asarray(state_b["features"]).mean(axis=1)
                    ab, ba, mean = core.stable_symmetric_ii(a, b)
                    rows.append({
                        "message_semantics": core.message_semantics_version(message_layout),
                        "message_layout": message_layout,
                        "algorithm": algorithm,
                        "top_alignment": alignment,
                        "feature_encoding": feature_encoding,
                        "retention_mode": retention_mode,
                        "L1": config.L - shared_depth,
                        "L2": shared_depth,
                        "bp_stage": stage_index,
                        "num_stages": len(states[0]),
                        "stage_name": state_a["stage_name"],
                        "delta_a_to_b": ab,
                        "delta_b_to_a": ba,
                        "mean_delta": mean,
                        "n": n,
                        "positions": len(positions),
                        "seed": seed,
                    })
    output_dir = Path(output_dir)
    csv_path = output_dir / "ii.csv"
    _write_rows(csv_path, rows)
    (output_dir / "ii_manifest.json").write_text(json.dumps({
        "shared_depth": shared_depth,
        "seed": seed,
        "n": n,
        "positions": len(positions),
        "message_order": list(core.MESSAGE_ORDER),
        "message_layout": message_layout,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")
    return csv_path


def _input_evidence_features(trees, positions, config):
    current = np.asarray(positions, dtype=np.int64) - 1
    eye = np.eye(int(config.v), dtype=np.float64)
    return {
        language: eye[np.asarray(trees[language][0])[:, current]].mean(axis=1)
        for language in (0, 1)
    }


def run_mknn(
    output_dir: str | Path,
    *,
    shared_depth: int,
    seed: int,
    n: int = 2048,
    k: int = 10,
    max_positions: int = 0,
    message_layout: str = core.DEFAULT_MESSAGE_LAYOUT,
) -> Path:
    """Run the paper's raw-probability, efficient-retention BP mKNN curve."""
    started = time.perf_counter()
    config = core.SweepConfig(seed=int(seed))
    core.validate_config(config, shared_depth)
    rng, rules = core.build_language_rules(config, shared_depth, seed_offset=0)
    trees = core.generate_translation_pair_trees(n, config, shared_depth, rules, rng)
    positions = core.select_positions(config, max_positions, seed + 991 * shared_depth)
    raw = {
        language: core.theoretical_atomic(
            trees[language][0], rules[language], config, positions,
            message_layout=message_layout,
        )
        for language in (0, 1)
    }
    maps, alignment = core.encoding_maps(config, shared_depth, "encoder-decoder")
    states = {
        language: core.algorithm_states(
            raw[language], maps[language], "encoder-decoder",
            "probability", "efficient", config,
        )
        for language in (0, 1)
    }
    u0 = _input_evidence_features(trees, positions, config)
    measurements = [(0, "u0", u0[0], u0[1])]
    measurements.extend(
        (index, a["stage_name"], np.asarray(a["features"]).mean(axis=1),
         np.asarray(b["features"]).mean(axis=1))
        for index, (a, b) in enumerate(zip(states[0], states[1]), start=1)
    )
    rows = []
    for stage, stage_name, features_a, features_b in measurements:
        result = mutual_knn(features_a, features_b, k=k)
        rows.append({
            "message_semantics": core.message_semantics_version(message_layout),
            "message_layout": message_layout,
            "algorithm": "encoder-decoder",
            "top_alignment": alignment,
            "feature_encoding": "probability",
            "retention_mode": "efficient",
            "L1": config.L - shared_depth,
            "L2": shared_depth,
            "bp_stage": stage,
            "stage_name": stage_name,
            "n": n,
            "k": k,
            "positions": len(positions),
            "seed": seed,
            **result.to_dict(),
        })
    output_dir = Path(output_dir)
    csv_path = output_dir / "mknn.csv"
    _write_rows(csv_path, rows)
    (output_dir / "mknn_manifest.json").write_text(json.dumps({
        "shared_depth": shared_depth,
        "seed": seed,
        "n": n,
        "k": k,
        "positions": len(positions),
        "message_order": ["u0", *core.MESSAGE_ORDER],
        "message_layout": message_layout,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")
    return csv_path


def run_suite(output_dir: str | Path, config: dict) -> tuple[Path, Path]:
    analysis = config["analysis"]
    common = {
        "shared_depth": int(config["data"]["shared_depth"]),
        "seed": int(config["random_seed"]),
        "n": int(analysis.get("bp_pairs", 2048)),
    }
    output_dir = Path(output_dir)
    return (
        run_ii(output_dir, **common),
        run_mknn(output_dir, k=int(analysis["k"]), **common),
    )
