"""Minimal reproduction code for the paper's MRHM experiments."""

from .data import DataConfig, Rules, build_rules, sample_corpus, sample_translation_pairs
from .model import CausalTransformer, TransformerConfig

__version__ = "0.1.0"

__all__ = [
    "CausalTransformer",
    "DataConfig",
    "Rules",
    "TransformerConfig",
    "build_rules",
    "sample_corpus",
    "sample_translation_pairs",
]
