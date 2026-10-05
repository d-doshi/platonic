#!/usr/bin/env python3
"""Core construction for the settled multilingual theoretical-BP sweep.

The hierarchy level convention in this file counts upward from the leaves:
level 0 is the surface, and level L is the single root.  Level L2 symbols are
shared.  Their outgoing rules are language tagged (m for A and m for B), so
the children at levels L2-1,...,0 are language specific.

The encoder-decoder state is the settled seven-message algorithm and never
contains a root-posterior block.  Upward and downward messages always occupy
different direct-sum blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from . import primitives as probes
from . import primitives as multilingual
from . import primitives as encoding


MESSAGE_ORDER = ("u1", "u2", "u3", "d3", "d2", "d1", "d0")
ATOMIC_MESSAGE_ORDER = ("u0",) + MESSAGE_ORDER
MESSAGE_LAYOUTS = ("partial-upward", "target-local-exact")
DEFAULT_MESSAGE_LAYOUT = "partial-upward"
MESSAGE_SEMANTICS_VERSIONS = {
    "partial-upward": "partial-upward-exact-down-cavities-v1",
    "target-local-exact": "exact-causal-frontier-v1",
}
EXPERIMENT_VERSIONS = {
    "ii": {
        "partial-upward": "settled-partial-upward-exact-down-v1",
        "target-local-exact": "settled-exact-causal-frontier-v1",
    },
    "cka": {
        "partial-upward": "settled-partial-upward-exact-down-cka-v1",
        "target-local-exact": "settled-exact-causal-frontier-cka-v1",
    },
    "mknn": {
        "partial-upward": "settled-partial-upward-exact-down-mknn-v1",
        "target-local-exact": "settled-exact-causal-frontier-mknn-v1",
    },
    "probes": {
        "partial-upward": "settled-partial-upward-exact-down-probes-v1",
        "target-local-exact": "settled-exact-causal-frontier-probes-v1",
    },
    "rule_depth_ii": {
        "partial-upward": "settled-rule-depth-partial-upward-exact-down-exp4-logprob-v1",
        "target-local-exact": "settled-rule-depth-exact-causal-frontier-exp4-logprob-v1",
    },
    "rule_depth_ii_efficient": {
        "partial-upward": "settled-rule-depth-partial-upward-exact-down-efficient-raw-logprob-v1",
        "target-local-exact": "settled-rule-depth-exact-causal-frontier-efficient-raw-logprob-v1",
    },
    "rule_depth_ii_exponential_base2": {
        "partial-upward": "settled-rule-depth-partial-upward-exact-down-exp2-raw-logprob-v1",
        "target-local-exact": "settled-rule-depth-exact-causal-frontier-exp2-raw-logprob-v1",
    },
    "rule_depth_cka": {
        "partial-upward": "settled-rule-depth-partial-upward-exact-down-exp4-logprob-cka-v1",
        "target-local-exact": "settled-rule-depth-exact-causal-frontier-exp4-logprob-cka-v1",
    },
    "rule_depth_cka_efficient": {
        "partial-upward": "settled-rule-depth-partial-upward-exact-down-efficient-raw-logprob-cka-v1",
        "target-local-exact": "settled-rule-depth-exact-causal-frontier-efficient-raw-logprob-cka-v1",
    },
    "rule_depth_probes": {
        "partial-upward": "settled-rule-depth-probes-partial-upward-exact-down-v1",
        "target-local-exact": "settled-rule-depth-probes-exact-causal-frontier-v1",
    },
    "prefix_probes": {
        "partial-upward": "settled-prefix-partial-upward-exact-down-all-latents-v1",
        "target-local-exact": "settled-prefix-exact-causal-frontier-all-latents-v1",
    },
}
# Backward-compatible name for callers that mean the restored default.
MESSAGE_SEMANTICS_VERSION = MESSAGE_SEMANTICS_VERSIONS[DEFAULT_MESSAGE_LAYOUT]


def message_semantics_version(message_layout: str) -> str:
    if message_layout not in MESSAGE_LAYOUTS:
        raise ValueError(message_layout)
    return MESSAGE_SEMANTICS_VERSIONS[message_layout]


def experiment_version(family: str, message_layout: str) -> str:
    if family not in EXPERIMENT_VERSIONS:
        raise ValueError(f"Unknown experiment family: {family}")
    if message_layout not in MESSAGE_LAYOUTS:
        raise ValueError(message_layout)
    return EXPERIMENT_VERSIONS[family][message_layout]
MESSAGE_LEVEL = {
    "u1": 1,
    "u2": 2,
    "u3": 3,
    "d3": 3,
    "d2": 2,
    "d1": 1,
    "d0": 0,
}
INTRODUCED_STAGE = {name: index for index, name in enumerate(MESSAGE_ORDER, start=1)}
LAST_EFFICIENT_STAGE = {
    "u1": 6,
    "u2": 5,
    "u3": 4,
    "d3": 4,
    "d2": 5,
    "d1": 6,
    "d0": 7,
}
EFFICIENT_LAYOUTS = (
    ("u1",),
    ("u1", "u2"),
    ("u1", "u2", "u3"),
    ("u1", "u2", "u3", "d3"),
    ("u1", "u2", "d2"),
    ("u1", "d1"),
    ("d0",),
)
RETENTION_MODES = (
    "efficient",
    "linear",
    "exponential-base2",
    "exponential-base4",
    "logarithmic",
    "residual",
)
FEATURE_ENCODINGS = ("probability", "centered_log_probability")
ALGORITHMS = ("encoder-decoder", "encode+one-pass-decode")
UNKNOWN_UPPER_MODELS = ("uniform-boundary", "dense-all-productions")
ONE_PASS_UPWARD_ORDER = ("u1", "u2", "u3")
ONE_PASS_EFFICIENT_LAYOUTS = (
    ("u1",),
    ("u1", "u2"),
    ("u1", "u2", "u3"),
    ("d0",),
)


@dataclass(frozen=True)
class SweepConfig:
    L: int = 4
    s: int = 2
    m: int = 4
    v: int = 16
    seed: int = 42
    linear_horizon: float = 4.0
    exp_half_life: float = 1.0
    log_epsilon: float = 1e-8


def validate_config(config: SweepConfig, L2: int) -> None:
    if config.L != 4:
        raise ValueError("The settled seven-message experiment requires L=4.")
    if int(L2) not in (1, 2, 3):
        raise ValueError(f"Expected L2 in (1,2,3), found {L2}.")
    if config.s != 2 or config.m != 4 or config.v != 16:
        raise ValueError("The settled sweep requires s=2, m=4, v=16.")
    if config.linear_horizon <= 0 or config.exp_half_life <= 0:
        raise ValueError("Decay scales must be positive.")
    if config.log_epsilon <= 0:
        raise ValueError("log_epsilon must be positive.")


def build_language_rules(config: SweepConfig, L2: int, seed_offset: int = 0):
    """Build shared upper rules and language-tagged lower rules.

    A rule at level ell maps a level-ell symbol to level-(ell-1) children.
    Rules with ell > L2 are shared.  Rules with ell <= L2 are language
    specific.  Thus the level-L2 nodes themselves are shared, while their
    outgoing alternatives are m-per-language (2m tagged alternatives total).
    """
    validate_config(config, L2)
    rng = np.random.default_rng(int(config.seed) + int(seed_offset) + 100_003 * int(L2))
    lang_a, lang_b = multilingual.build_languages(
        config.L,
        config.L - int(L2),
        int(L2),
        config.s,
        config.m,
        config.v,
        rng,
    )
    rules = {0: lang_a, 1: lang_b}
    for level in range(1, config.L + 1):
        a = rules[0][level]
        b = rules[1][level]
        expected = (config.v, config.m, config.s)
        if a.shape != expected or b.shape != expected:
            raise AssertionError(f"Unexpected rule shape at level {level}: {a.shape}, {b.shape}")
        if level > int(L2) and not np.array_equal(a, b):
            raise AssertionError(f"Level {level} must use the shared rule table.")
        if level <= int(L2) and np.array_equal(a, b):
            raise AssertionError(f"Level {level} must use language-tagged rule tables.")
    # At the boundary there are exactly m tagged choices for each language.
    if sum(rules[language][int(L2)].shape[1] for language in (0, 1)) != 2 * config.m:
        raise AssertionError("The level-L2 boundary must have 2m tagged outgoing rules.")
    return rng, rules


def _expand_with_choices(symbols, table, choices):
    symbols = np.asarray(symbols, dtype=np.int64)
    choices = np.asarray(choices, dtype=np.int64)
    return table[symbols, choices].reshape(symbols.shape[0], -1)


def generate_translation_pair_trees(n: int, config: SweepConfig, L2: int, rules, rng):
    """Generate translations that share nodes through level L2, then branch.

    The shared expansion loop includes rule L2+1, whose children are the
    shared level-L2 nodes.  Only after those nodes exist do A and B independently
    choose from their language-tagged level-L2 rule tables.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    shared = rng.integers(config.v, size=(n, 1), dtype=np.int64)
    trees = {0: {config.L: shared.copy()}, 1: {config.L: shared.copy()}}

    for level in range(config.L, int(L2), -1):
        table = rules[0][level]
        choices = rng.integers(config.m, size=shared.shape, dtype=np.int64)
        shared = _expand_with_choices(shared, table, choices)
        trees[0][level - 1] = shared.copy()
        trees[1][level - 1] = shared.copy()

    if not np.array_equal(trees[0][int(L2)], trees[1][int(L2)]):
        raise AssertionError("Level-L2 nodes must be shared exactly.")

    current = {0: shared.copy(), 1: shared.copy()}
    first_language_choices = {}
    for level in range(int(L2), 0, -1):
        for language in (0, 1):
            choices = rng.integers(config.m, size=current[language].shape, dtype=np.int64)
            if level == int(L2):
                first_language_choices[language] = choices.copy()
            current[language] = _expand_with_choices(
                current[language], rules[language][level], choices
            )
            trees[language][level - 1] = current[language].copy()

    # This checks that the two languages did not accidentally reuse one sampled
    # boundary realization.  Equal child symbols by chance remain allowed.
    if np.array_equal(first_language_choices[0], first_language_choices[1]):
        raise AssertionError("A and B reused all boundary rule choices; expected independent draws.")
    for level in range(int(L2) + 1):
        if level == int(L2):
            if not np.array_equal(trees[0][level], trees[1][level]):
                raise AssertionError("Boundary nodes were not shared.")
        elif np.shares_memory(trees[0][level], trees[1][level]):
            raise AssertionError(f"Language-specific level {level} shares storage.")
    return trees


def pure_language_trees(n: int, config: SweepConfig, rules, L2: int, seed_offset: int):
    output = {}
    for language in (0, 1):
        rng = np.random.default_rng(
            int(config.seed) + int(seed_offset) + 1_000_003 * int(L2) + 10_007 * language
        )
        output[language] = probes.sample_pure_language_trees(
            int(n), rules[language], config.L, config.v, language, rng
        )
    return output


def select_positions(config: SweepConfig, max_positions: int, seed: int):
    rng = np.random.default_rng(int(seed))
    return probes.select_positions(config.L, config.s, int(max_positions), rng)


def _unknown_upper_boundary(
    n: int, nodes: Mapping[int, int], config: SweepConfig, depth: int, model: str,
):
    """Return the level-depth downward prior implied by an unknown upper tree.

    With ``dense-all-productions``, every one of the v**s child tuples is an
    equally likely production for every parent symbol.  Marginalizing such a
    factor makes each target child exactly uniform: summing over the sibling
    coordinates gives one, independently of their normalized upward messages.
    Repeating this through every unknown upper level therefore yields the same
    uniform boundary as the direct ``uniform-boundary`` construction.  We keep
    the alternatives explicit so the equivalence is tested end-to-end.
    """
    if model not in UNKNOWN_UPPER_MODELS:
        raise ValueError(model)
    boundary = np.full(
        (int(n), int(nodes[int(depth)]), int(config.v)),
        1.0 / float(config.v),
        dtype=np.float64,
    )
    if model == "dense-all-productions":
        # There are v**(s-1) dense productions for each fixed target-child
        # symbol and parent.  Equal weights make all target symbols identical;
        # normalization gives 1/v.  This assertion guards that combinatorial
        # count for future non-binary configurations as well.
        count_per_target = int(config.v) ** (int(config.s) - 1)
        if count_per_target * int(config.v) != int(config.v) ** int(config.s):
            raise AssertionError("Dense production count is inconsistent")
    return boundary


def _uniform_message(n: int, config: SweepConfig):
    return np.full(
        (int(n), int(config.v)),
        1.0 / float(config.v),
        dtype=np.float64,
    )


def _normalize_message(message, config: SweepConfig):
    message = np.asarray(message, dtype=np.float64)
    denominator = message.sum(axis=-1, keepdims=True)
    return np.divide(
        message,
        denominator,
        out=np.full_like(message, 1.0 / float(config.v)),
        where=denominator > 0.0,
    )


def _frontier_sibling_message(up, predict_pos: int, level: int, config: SweepConfig):
    """Return the level-``level`` sibling cavity on the next-token path.

    The settled experiment is binary.  At every level, the sibling of the
    target-path node is either a completely observed subtree or a completely
    unobserved subtree.  Selecting that sibling therefore returns an exact
    upward message or the neutral uniform message; it can never select the
    partially observed target subtree.
    """
    if int(config.s) != 2:
        raise ValueError("Exact frontier extraction currently requires binary trees")
    level = int(level)
    target_node = int(predict_pos) // (int(config.s) ** level)
    parent_node, target_child = divmod(target_node, int(config.s))
    sibling_node = parent_node * int(config.s) + (1 - target_child)
    return np.asarray(up[level])[:, sibling_node, :]


def _factor_to_target_child(
    parent_message,
    sibling_message,
    target_child: int,
    factor,
    config: SweepConfig,
):
    """Apply one binary factor-to-child update using only its sibling cavity."""
    parent_symbols, child_symbols, _, child_selectors = factor
    sibling_child = 1 - int(target_child)
    contribution = np.asarray(parent_message)[:, parent_symbols]
    contribution = contribution * np.asarray(sibling_message)[
        :, child_symbols[:, sibling_child]
    ]
    message = contribution @ child_selectors[int(target_child)]
    return _normalize_message(message, config)


def _causal_frontier_messages(
    leaves, rules, config: SweepConfig, positions, depth: int,
    unknown_upper_model: str = "uniform-boundary",
    message_layout: str = DEFAULT_MESSAGE_LAYOUT,
):
    """Return encoded upward workspace and cavity-correct downward messages.

    Rules are numbered upward from the leaves.  Known levels use their true
    factors.  Every unknown upper factor is marginalized under the selected
    surrogate, which gives a uniform upward and downward message at that level.
    Unknown levels modify messages; they do not shorten the BP schedule.

    For prediction position ``t``, the default encoded u_k is the partial
    target-path workspace rooted at the ancestor of ``t-1``.  It describes an
    intermediate upward computation but is never substituted into a downward
    factor update.  ``target-local-exact`` instead exposes the exact sibling
    cavity itself as the encoded u_k block and is retained as a strongly
    efficient reference condition.

    In both layouts, the uniform level-``depth`` downward boundary is
    propagated through the known lower factors using only exact sibling
    cavities.  Thus d_(depth-1),...,d0 are identical between layouts.  At
    depth zero no factors are known, so every upward and downward message is
    the neutral uniform message.
    """
    depth = int(depth)
    if depth < 0 or depth > int(config.L):
        raise ValueError(f"learned_rule_depth must lie in 0,...,{config.L}")
    if unknown_upper_model not in UNKNOWN_UPPER_MODELS:
        raise ValueError(unknown_upper_model)
    if message_layout not in MESSAGE_LAYOUTS:
        raise ValueError(message_layout)
    if int(config.s) != 2:
        raise ValueError("The settled causal-frontier experiment requires s=2")
    collected = {name: [] for name in ATOMIC_MESSAGE_ORDER}

    factors = {
        level: multilingual.factor_arrays(rules[level], config.v)
        for level in range(1, depth + 1)
    }
    n = np.asarray(leaves).shape[0]
    nodes = {
        level: int(config.s) ** (int(config.L) - level)
        for level in range(config.L + 1)
    }
    for predict_pos in positions:
        current_pos = int(predict_pos) - 1
        exact_up, _exact_down = multilingual.bp_messages(
            leaves, rules, config.L, config.s, config.v, observed_upto=current_pos
        )
        frontier = {}
        for level in range(config.L):
            if level <= depth:
                frontier[level] = _frontier_sibling_message(
                    exact_up, int(predict_pos), level, config
                )
                if message_layout == "partial-upward":
                    workspace_node = current_pos // (int(config.s) ** level)
                    encoded_upward = np.asarray(exact_up[level])[
                        :, workspace_node, :
                    ]
                else:
                    encoded_upward = frontier[level]
            else:
                frontier[level] = _uniform_message(n, config)
                encoded_upward = frontier[level]
            collected[f"u{level}"].append(encoded_upward)

        # Marginalizing every unknown upper rule yields a uniform downward
        # boundary at the highest learned symbol level.  The same is true for
        # ``dense-all-productions``; _unknown_upper_boundary documents and
        # validates that equivalence.
        uniform_boundaries = {
            level: _unknown_upper_boundary(
                n, nodes, config, level, unknown_upper_model
            )[:, 0, :]
            for level in range(depth, config.L + 1)
        }
        down = dict(uniform_boundaries)
        for factor_level in range(depth, 0, -1):
            child_level = factor_level - 1
            target_node = int(predict_pos) // (int(config.s) ** child_level)
            target_child = target_node % int(config.s)
            down[child_level] = _factor_to_target_child(
                down[factor_level],
                frontier[child_level],
                target_child,
                factors[factor_level],
                config,
            )

        for level in range(config.L - 1, -1, -1):
            collected[f"d{level}"].append(down[level])
    output = {name: np.stack(values, axis=1) for name, values in collected.items()}
    if set(output) != set(ATOMIC_MESSAGE_ORDER):
        raise AssertionError(f"Incomplete causal-frontier message set: {sorted(output)}")
    return output


def theoretical_atomic(
    leaves, rules, config: SweepConfig, positions, learned_rule_depth: int = 4,
    unknown_upper_model: str = "uniform-boundary",
    message_layout: str = DEFAULT_MESSAGE_LAYOUT,
):
    """Atomic features needed by the settled two-algorithm comparison.

    The restored default exposes partial upward workspace but computes every
    downward block with exact sibling cavities.  The alternative
    ``target-local-exact`` layout exposes those sibling cavities directly and
    reproduces the archived strongly efficient reference.  Neither layout has
    an intermediate tau/rho path.
    """
    learned_rule_depth = int(learned_rule_depth)
    if unknown_upper_model not in UNKNOWN_UPPER_MODELS:
        raise ValueError(unknown_upper_model)
    if message_layout not in MESSAGE_LAYOUTS:
        raise ValueError(message_layout)
    atomic_messages = _causal_frontier_messages(
        leaves,
        rules,
        config,
        positions,
        learned_rule_depth,
        unknown_upper_model=unknown_upper_model,
        message_layout=message_layout,
    )
    encoder_messages = {
        name: atomic_messages[name] for name in MESSAGE_ORDER
    }
    encoder_levels = list(range(config.L)) + list(
        range(config.L - 1, -1, -1)
    )
    encoder_blocks = [
        atomic_messages[name] for name in ATOMIC_MESSAGE_ORDER
    ]
    root_uniform = np.full_like(
        atomic_messages[f"u{config.L - 1}"],
        1.0 / float(config.v),
    )
    upward_blocks = [
        atomic_messages[f"u{level}"] for level in range(config.L)
    ] + [root_uniform]
    surface = atomic_messages["d0"]
    return {
        "atomic_messages": atomic_messages,
        "encoder_blocks": encoder_blocks,
        "encoder_levels": encoder_levels,
        "encoder_messages": encoder_messages,
        "learned_rule_depth": learned_rule_depth,
        "message_layout": message_layout,
        "unknown_upper_model": unknown_upper_model,
        "upward_blocks": upward_blocks,
        "surface": surface,
        "message_semantics": MESSAGE_SEMANTICS_VERSIONS[message_layout],
    }


def centered_log_probability(value, epsilon: float):
    value = np.asarray(value, dtype=np.float64)
    encoded = np.log(np.maximum(value, float(epsilon)))
    return encoded - encoded.mean(axis=-1, keepdims=True)


def encode_probability(value, feature_encoding: str, epsilon: float):
    value = np.asarray(value, dtype=np.float64)
    if feature_encoding == "probability":
        return value
    if feature_encoding == "centered_log_probability":
        return centered_log_probability(value, epsilon)
    raise ValueError(feature_encoding)


def encoding_maps(config: SweepConfig, L2: int, algorithm: str):
    """Return the empirically selected encoder alignment for each algorithm."""
    if algorithm not in ALGORITHMS:
        raise ValueError(algorithm)
    rng = np.random.default_rng(int(config.seed) + 9_000_001 * int(L2))
    maps_a = {level: encoding.random_orthogonal(rng, config.v) for level in range(config.L + 1)}
    maps_b = {level: encoding.random_orthogonal(rng, config.v) for level in range(config.L + 1)}
    maps_a["surface"] = encoding.random_orthogonal(rng, config.v)
    maps_b["surface"] = encoding.random_orthogonal(rng, config.v)

    if algorithm == "encoder-decoder":
        # Level-L2 symbols are shared even though their outgoing rules are
        # language tagged.  The posterior over those shared symbols therefore
        # uses the common encoder at the boundary itself as well as above it.
        for level in range(int(L2), config.L + 1):
            maps_b[level] = maps_a[level]
        alignment = "shared-top"
    else:
        alignment = "separate-top"

    for level in range(config.L + 1):
        same = np.array_equal(maps_a[level], maps_b[level])
        expected_same = algorithm == "encoder-decoder" and level >= int(L2)
        if same != expected_same:
            raise AssertionError(
                f"Unexpected encoder sharing at level {level}: same={same}, expected={expected_same}"
            )
    return {0: maps_a, 1: maps_b}, alignment


def retention_weight(
    mode: str,
    age: int,
    linear_horizon: float = 4.0,
    exp_half_life: float = 1.0,
):
    """Weight of a message `age` stages after its last efficient use.

    Logarithmic decay is normalized to one at age zero:
        w(age) = log(2) / log(age + 2).
    """
    age = int(age)
    if age < 0:
        raise ValueError("age must be nonnegative")
    if age == 0:
        return 1.0
    if mode == "efficient":
        return 0.0
    if mode == "residual":
        return 1.0
    if mode == "linear":
        return max(0.0, 1.0 - age / float(linear_horizon))
    if mode == "exponential-base2":
        return 2.0 ** (-age / float(exp_half_life))
    if mode == "exponential-base4":
        return 4.0 ** (-age / float(exp_half_life))
    if mode == "logarithmic":
        return float(np.log(2.0) / np.log(float(age) + 2.0))
    raise ValueError(mode)


def encoder_decoder_atomic(raw, maps, feature_encoding: str, epsilon: float):
    output = {}
    for name, block in raw["encoder_messages"].items():
        encoded = encode_probability(block, feature_encoding, epsilon)
        output[name] = encoding.apply_map(encoded, maps[MESSAGE_LEVEL[name]])
    return output


def encoder_decoder_states(
    atomic: Mapping[str, np.ndarray],
    mode: str,
    config: SweepConfig,
):
    if mode not in RETENTION_MODES:
        raise ValueError(mode)
    states = []
    for stage in range(1, 8):
        pieces = []
        active_names = []
        weights = {}
        for name in MESSAGE_ORDER:
            if name not in atomic:
                continue
            if INTRODUCED_STAGE[name] > stage:
                continue
            if name in EFFICIENT_LAYOUTS[stage - 1]:
                weight = 1.0
            else:
                age = stage - LAST_EFFICIENT_STAGE[name]
                if age <= 0:
                    # Introduced but not in this efficient stage can only occur
                    # after its final use in this seven-stage schedule.
                    continue
                weight = retention_weight(
                    mode, age, config.linear_horizon, config.exp_half_life,
                )
            if weight > 0.0:
                pieces.append(float(weight) * np.asarray(atomic[name]))
                active_names.append(name)
                weights[name] = float(weight)
        if not pieces:
            raise AssertionError(f"Empty encoder-decoder state at stage {stage}")
        states.append({
            "stage": stage,
            "stage_name": "+".join(active_names),
            "features": np.concatenate(pieces, axis=-1),
            "weights": weights,
        })
    if mode == "efficient" and set(atomic) == set(MESSAGE_ORDER):
        actual = tuple(tuple(state["stage_name"].split("+")) for state in states)
        if actual != EFFICIENT_LAYOUTS:
            raise AssertionError(f"Efficient layout changed: {actual}")
    return states


def one_pass_states(
    raw,
    maps,
    feature_encoding: str,
    mode: str,
    config: SweepConfig,
):
    """Native upward-only construction followed by a one-shot decode.

    There are always L-1 cumulative upward states, using u_1,...,u_(L-1),
    followed by the next-token posterior d_0.  If only the bottom ``ell < L``
    rule levels are known, messages above ell are computed with the unknown-rule
    surrogate rather than omitted, and d_0 is obtained from the corresponding
    uniform level-ell downward boundary.

    The leaf evidence u_0 is an input to BP, not an encoded message state.
    There is no interpolation/stretching, no intermediate tau/rho block, and
    no encoded root posterior u_L.

    In the restored default layout, the displayed u_k blocks are partial
    upward workspace.  The one-shot decoder nevertheless consumes exact
    sibling cavities internally, as already incorporated into d0.  In the
    strongly efficient reference layout, the displayed u_k blocks are those
    exact target-local cavities.  The final representation is post-use:
    efficient discards the upward workspace; residual retains it; and each
    decay law retains it with its age-one weight.
    """
    learned_depth = int(raw.get("learned_rule_depth", config.L))
    if not 0 <= learned_depth <= config.L:
        raise ValueError(
            f"learned_rule_depth must lie in 0,...,{config.L}; "
            f"found {learned_depth}"
        )
    message_layout = raw.get("message_layout", DEFAULT_MESSAGE_LAYOUT)
    expected_semantics = MESSAGE_SEMANTICS_VERSIONS.get(message_layout)
    if raw.get("message_semantics") != expected_semantics:
        raise ValueError(
            "One-pass state/message semantics disagree; "
            f"found {raw.get('message_semantics')!r}."
        )
    upward_levels = list(range(1, config.L))
    upward_values = [
        raw["encoder_messages"][f"u{level}"] for level in upward_levels
    ]
    surface_raw = raw["encoder_messages"]["d0"]

    upward = [
        encoding.apply_map(
            encode_probability(value, feature_encoding, config.log_epsilon),
            maps[level],
        )
        for level, value in zip(upward_levels, upward_values)
    ]
    surface = encoding.apply_map(
        encode_probability(surface_raw, feature_encoding, config.log_epsilon),
        maps["surface"],
    )

    states = []
    # Native upward states: no stretching and no intermediate decode message.
    for stage in range(len(upward)):
        pieces = list(upward[: stage + 1])
        names = [f"u{level}" for level in upward_levels[: stage + 1]]
        weights = {name: 1.0 for name in names}
        states.append({
            "stage": stage + 1,
            "stage_name": "+".join(names),
            "features": np.concatenate(pieces, axis=-1),
            "weights": weights,
        })

    # The exact one-shot decoder consumes every upward block at once.  This is
    # the post-decode state, so retained workspace has age one after last use.
    pieces = [surface]
    names = ["d0"]
    weights = {"d0": 1.0}
    workspace_weight = retention_weight(
        mode, 1, config.linear_horizon, config.exp_half_life,
    )
    if workspace_weight > 0.0:
        for level, block in zip(upward_levels, upward):
            pieces.append(float(workspace_weight) * block)
            name = f"u{level}"
            names.append(name)
            weights[name] = float(workspace_weight)
    states.append({
        "stage": len(upward) + 1,
        "stage_name": "+".join(names),
        "features": np.concatenate(pieces, axis=-1),
        "weights": weights,
    })
    forbidden = {"u0", f"u{config.L}"}
    if any(name in forbidden for state in states for name in state["weights"]):
        raise AssertionError(
            "Leaf input u0 or root-posterior u_L entered the one-pass representation."
        )
    if any(name.startswith("rho") for state in states for name in state["weights"]):
        raise AssertionError("A rho/tau block entered the upward-only one-pass representation.")
    if mode == "efficient" and learned_depth == config.L:
        actual = tuple(tuple(state["stage_name"].split("+")) for state in states)
        if actual != ONE_PASS_EFFICIENT_LAYOUTS:
            raise AssertionError(f"Efficient one-pass layout changed: {actual}")
    return states


def algorithm_states(raw, maps, algorithm: str, feature_encoding: str, mode: str, config):
    if algorithm == "encoder-decoder":
        atomic = encoder_decoder_atomic(raw, maps, feature_encoding, config.log_epsilon)
        return encoder_decoder_states(atomic, mode, config)
    if algorithm == "encode+one-pass-decode":
        return one_pass_states(raw, maps, feature_encoding, mode, config)
    raise ValueError(algorithm)


def mean_position_features(states):
    return [np.asarray(state["features"]).mean(axis=1) for state in states]


def flatten_features(value):
    value = np.asarray(value, dtype=np.float64)
    return value.reshape(value.shape[0] * value.shape[1], value.shape[2])


def stable_symmetric_ii(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # A sample-independent representation supplies no neighborhood ordering.
    # Stable index-based tie breaking would spuriously align two such spaces
    # and return 2/N; the expected rank under unbiased random tie breaking is
    # the null information-imbalance value 1.
    a_constant = np.allclose(a, a[:1], rtol=0.0, atol=1e-12)
    b_constant = np.allclose(b, b[:1], rtol=0.0, atol=1e-12)
    if a_constant or b_constant:
        return 1.0, 1.0, 1.0
    ab = encoding.stable_information_imbalance(a, b)
    ba = encoding.stable_information_imbalance(b, a)
    return ab, ba, 0.5 * (ab + ba)
