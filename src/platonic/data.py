"""Exact two-language code-switch MRHM generator used for paper training."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DataConfig:
    vocab_size: int = 16
    num_classes: int = 16
    multiplicity: int = 4
    branching: int = 2
    depth: int = 4
    shared_depth: int = 1
    num_languages: int = 2
    switch_probability: float = 0.1

    def validate(self) -> None:
        if min(self.vocab_size, self.num_classes, self.multiplicity) < 1 or self.branching < 2:
            raise ValueError("invalid MRHM dimensions")
        if not 0 < self.shared_depth < self.depth:
            raise ValueError("shared_depth must lie strictly between zero and depth")
        if self.num_languages != 2:
            raise ValueError("the paper experiments use exactly two languages")
        if not 0.0 <= self.switch_probability <= 1.0:
            raise ValueError("switch_probability must be in [0, 1]")
        if self.num_classes * self.multiplicity > self.vocab_size ** self.branching:
            raise ValueError("not enough distinct tuples for the root production table")
        if self.vocab_size * self.multiplicity > self.vocab_size ** self.branching:
            raise ValueError("not enough distinct tuples for a production table")

    @property
    def sequence_length(self) -> int:
        return self.branching ** self.depth

    @property
    def surface_vocab_size(self) -> int:
        return self.num_languages * self.vocab_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataConfig":
        result = cls(**value)
        result.validate()
        return result


@dataclass
class Rules:
    """Shared and language-specific production tables in root-to-leaf order."""

    shared: list[np.ndarray]
    language: list[list[np.ndarray]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared": [table.tolist() for table in self.shared],
            "language": [[table.tolist() for table in tables] for tables in self.language],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Rules":
        return cls(
            shared=[np.asarray(table, dtype=np.int64) for table in value["shared"]],
            language=[
                [np.asarray(table, dtype=np.int64) for table in tables]
                for tables in value["language"]
            ],
        )


def _sample_table(
    parent_vocab: int,
    child_values: range,
    multiplicity: int,
    branching: int,
) -> np.ndarray:
    tuples = list(product(child_values, repeat=branching))
    chosen = random.sample(tuples, parent_vocab * multiplicity)
    return np.asarray(chosen, dtype=np.int64).reshape(
        parent_vocab, multiplicity, branching
    )


def build_rules(config: DataConfig, seed: int) -> Rules:
    """Match ``sample_rules_distinct_multilingual`` from the research code."""
    config.validate()
    random.seed(int(seed))
    shared = []
    for level in range(config.shared_depth):
        parent_vocab = config.num_classes if level == 0 else config.vocab_size
        shared.append(
            _sample_table(
                parent_vocab,
                range(config.vocab_size),
                config.multiplicity,
                config.branching,
            )
        )

    language = []
    for language_index in range(config.num_languages):
        random.seed(int(seed) + language_index + 1)
        tables = []
        surface_values = range(
            language_index * config.vocab_size,
            (language_index + 1) * config.vocab_size,
        )
        for _ in range(config.shared_depth, config.depth):
            tables.append(
                _sample_table(
                    config.vocab_size,
                    surface_values,
                    config.multiplicity,
                    config.branching,
                )
            )
        language.append(tables)
    return Rules(shared=shared, language=language)


def _expand(
    symbols: torch.Tensor,
    table: np.ndarray,
    generator: torch.Generator,
    language: int = 0,
    vocab_size: int = 0,
) -> torch.Tensor:
    table_tensor = torch.as_tensor(table, dtype=torch.long)
    indices = symbols - language * vocab_size if language else symbols
    choices = torch.randint(
        table_tensor.shape[1], indices.shape, generator=generator
    )
    return table_tensor[indices, choices].flatten(start_dim=1)


def sample_corpus(
    n: int,
    config: DataConfig,
    rules: Rules,
    seed: int,
) -> dict[str, Any]:
    """Sample the exact per-subtree code-switch training distribution."""
    if n < 1:
        raise ValueError("n must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    labels = torch.randint(config.num_classes, (n,), generator=generator)
    levels = [labels[:, None].clone()]
    for table in rules.shared:
        labels = _expand(labels, table, generator)
        levels.append(labels.clone())

    base_languages = torch.randint(
        config.num_languages, (n,), generator=generator
    )
    subtree_languages = base_languages[:, None].repeat(1, labels.shape[1])
    if config.switch_probability > 0:
        switch = torch.rand(subtree_languages.shape, generator=generator) < config.switch_probability
        switched = 1 - subtree_languages
        subtree_languages = torch.where(switch, switched, subtree_languages)

    symbol_languages = subtree_languages
    for lower_index, _ in enumerate(rules.language[0]):
        next_labels = torch.zeros(
            n, labels.shape[1] * config.branching, dtype=torch.long
        )
        next_languages = torch.zeros_like(next_labels)
        for language in range(config.num_languages):
            mask = symbol_languages == language
            if not mask.any():
                continue
            table = torch.as_tensor(rules.language[language][lower_index], dtype=torch.long)
            index_labels = labels[mask]
            if lower_index > 0:
                index_labels = index_labels - language * config.vocab_size
            choices = torch.randint(table.shape[1], index_labels.shape, generator=generator)
            expanded = table[index_labels, choices]
            next_labels.reshape(n, -1, config.branching)[mask] = expanded
            next_languages.reshape(n, -1, config.branching)[mask] = language
        labels = next_labels
        symbol_languages = next_languages
        levels.append(labels.clone())
    return {
        "tokens": labels.numpy(),
        "languages": base_languages.numpy(),
        "subtree_languages": subtree_languages.numpy(),
        "levels": [value.numpy() for value in levels],
    }


def sample_translation_pairs(
    n: int,
    config: DataConfig,
    rules: Rules,
    seed: int,
) -> dict[str, Any]:
    """Match the translated-pair sampler used by checkpoint analyses."""
    if n < 1:
        raise ValueError("n must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    labels = torch.randint(config.num_classes, (n,), generator=generator)
    shared_levels = [labels[:, None].clone()]
    for table in rules.shared:
        labels = _expand(labels, table, generator)
        shared_levels.append(labels.clone())

    current = {language: labels.clone() for language in range(config.num_languages)}
    all_levels = {
        language: [value.clone() for value in shared_levels]
        for language in range(config.num_languages)
    }
    for lower_index in range(len(rules.language[0])):
        for language in range(config.num_languages):
            current[language] = _expand(
                current[language],
                rules.language[language][lower_index],
                generator,
                language=language if lower_index > 0 else 0,
                vocab_size=config.vocab_size,
            )
            all_levels[language].append(current[language].clone())
    return {
        "tokens_a": current[0].numpy(),
        "tokens_b": current[1].numpy(),
        "levels_a": [value.numpy() for value in all_levels[0]],
        "levels_b": [value.numpy() for value in all_levels[1]],
        "shared_levels": [value.numpy() for value in shared_levels],
        "root": shared_levels[0][:, 0].numpy(),
    }


def token_latent_labels(
    levels: list[np.ndarray], config: DataConfig, depth: int
) -> np.ndarray:
    if not 0 <= depth <= config.depth:
        raise ValueError("latent depth is outside the tree")
    repeats = config.branching ** (config.depth - depth)
    return np.repeat(levels[depth], repeats, axis=1)
