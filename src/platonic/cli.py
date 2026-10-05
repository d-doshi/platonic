"""Command-line interface for all public experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bp.experiments import run_suite as run_bp_suite
from .experiments import analyze_checkpoint, load_config, reproduce
from .metrics import information_imbalance, mutual_knn
from .training import train_from_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="platonic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the tiny CPU integration experiment")
    smoke.add_argument("--config", default="configs/smoke.json")
    smoke.add_argument("--output", default="runs/smoke")

    train = subparsers.add_parser("train", help="train one transformer")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze a trained checkpoint")
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--device", default="auto")

    bp = subparsers.add_parser("bp", help="run the settled theoretical BP baselines")
    bp.add_argument("--config", required=True)
    bp.add_argument("--output", required=True)

    full = subparsers.add_parser("reproduce", help="run one config or its complete sweep")
    full.add_argument("--config", required=True)
    full.add_argument("--output", required=True)

    arrays = subparsers.add_parser("arrays", help="compare two aligned representation arrays")
    arrays.add_argument("--language-a", required=True)
    arrays.add_argument("--language-b", required=True)
    arrays.add_argument("--k", type=int, default=10)
    arrays.add_argument("--device", default="cpu")
    arrays.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        config = load_config(args.config)
        reproduce(config, args.output)
    elif args.command == "train":
        train_from_config(load_config(args.config), args.output)
    elif args.command == "analyze":
        analyze_checkpoint(
            args.checkpoint, load_config(args.config), args.output, device=args.device
        )
    elif args.command == "bp":
        run_bp_suite(args.output, load_config(args.config))
    elif args.command == "reproduce":
        reproduce(load_config(args.config), args.output)
    elif args.command == "arrays":
        a = np.load(args.language_a)
        b = np.load(args.language_b)
        payload = {
            "language_a": str(Path(args.language_a).resolve()),
            "language_b": str(Path(args.language_b).resolve()),
            "n": len(a),
            "information_imbalance": information_imbalance(a, b, device=args.device),
            "mutual_knn": mutual_knn(a, b, k=args.k, device=args.device),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
