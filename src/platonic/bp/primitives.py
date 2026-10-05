"""Small numerical primitives used by the settled causal-BP construction."""
from __future__ import annotations

import itertools

import numpy as np


def _make_level_rules(v: int, s: int, m: int, rng) -> np.ndarray:
    if v * m > v ** s:
        raise ValueError(f"need v*m <= v**s, got {v}*{m} > {v}^{s}")
    tuples = np.array(list(itertools.product(range(v), repeat=s)), dtype=np.int64)
    indices = rng.choice(len(tuples), size=v * m, replace=False)
    return tuples[indices].reshape(v, m, s)


def build_languages(L, L1, L2, s, m, v, rng):
    if L1 + L2 != L:
        raise ValueError(f"expected L1 + L2 == L, got {L1} + {L2} != {L}")
    shared = {level: _make_level_rules(v, s, m, rng) for level in range(L2 + 1, L + 1)}
    rules_a = {level: _make_level_rules(v, s, m, rng) for level in range(1, L2 + 1)}
    rules_b = {level: _make_level_rules(v, s, m, rng) for level in range(1, L2 + 1)}
    return {**rules_a, **shared}, {**rules_b, **shared}


def factor_arrays(table, v):
    parent_symbols = np.repeat(np.arange(v), table.shape[1])
    child_symbols = table.reshape(v * table.shape[1], table.shape[2])
    parent_selector = np.zeros((child_symbols.shape[0], v), dtype=np.float64)
    parent_selector[np.arange(child_symbols.shape[0]), parent_symbols] = 1.0
    child_selectors = []
    for child_index in range(table.shape[2]):
        selector = np.zeros((child_symbols.shape[0], v), dtype=np.float64)
        selector[np.arange(child_symbols.shape[0]), child_symbols[:, child_index]] = 1.0
        child_selectors.append(selector)
    return parent_symbols, child_symbols, parent_selector, child_selectors


def bp_messages(leaves, rules, L, s, v, observed_upto):
    n = leaves.shape[0]
    nodes = {level: s ** (L - level) for level in range(L + 1)}
    up = {0: np.empty((n, nodes[0], v), dtype=np.float64)}
    for position in range(nodes[0]):
        if position <= observed_upto:
            message = np.zeros((n, v), dtype=np.float64)
            message[np.arange(n), leaves[:, position]] = 1.0
            up[0][:, position, :] = message
        else:
            up[0][:, position, :] = 1.0 / v

    factors = {level: factor_arrays(rules[level], v) for level in range(1, L + 1)}
    for level in range(1, L + 1):
        _, child_symbols, parent_selector, _ = factors[level]
        child = up[level - 1].reshape(n, nodes[level], s, v)
        contribution = np.ones((n, nodes[level], len(child_symbols)), dtype=np.float64)
        for child_index in range(s):
            contribution *= child[:, :, child_index, :][:, :, child_symbols[:, child_index]]
        message = contribution @ parent_selector
        up[level] = message / message.sum(axis=-1, keepdims=True)

    down = {L: np.full((n, nodes[L], v), 1.0 / v, dtype=np.float64)}
    for level in range(L, 0, -1):
        parent_symbols, child_symbols, _, child_selectors = factors[level]
        child = up[level - 1].reshape(n, nodes[level], s, v)
        output = np.empty((n, nodes[level], s, v), dtype=np.float64)
        for target_child in range(s):
            contribution = down[level][:, :, parent_symbols]
            for sibling in range(s):
                if sibling != target_child:
                    contribution *= child[:, :, sibling, :][:, :, child_symbols[:, sibling]]
            message = contribution @ child_selectors[target_child]
            output[:, :, target_child, :] = message / message.sum(axis=-1, keepdims=True)
        down[level - 1] = output.reshape(n, nodes[level - 1], v)
    return up, down


def _expand_with_choices(symbols, table, rng):
    choices = rng.integers(table.shape[1], size=symbols.shape, dtype=np.int64)
    return table[symbols, choices].reshape(symbols.shape[0], -1)


def sample_pure_language_trees(n, rules, L, v, language, rng):
    del language
    root = rng.integers(v, size=(int(n), 1), dtype=np.int64)
    trees = {int(L): root}
    symbols = root
    for level in range(int(L), 0, -1):
        symbols = _expand_with_choices(symbols, rules[level], rng)
        trees[level - 1] = symbols
    return trees


def select_positions(L, s, max_positions, rng):
    positions = np.arange(1, int(s) ** int(L), dtype=np.int64)
    if int(max_positions) > 0 and len(positions) > int(max_positions):
        positions = np.sort(rng.choice(positions, size=int(max_positions), replace=False))
    return positions


def random_orthogonal(rng, dimension):
    matrix = rng.normal(size=(int(dimension), int(dimension)))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def apply_map(message, matrix):
    return np.asarray(message) @ np.asarray(matrix)


def _pairwise_squared_distances(values):
    squared = (values * values).sum(axis=1)
    return np.maximum(squared[:, None] + squared[None, :] - 2.0 * (values @ values.T), 0.0)


def stable_information_imbalance(x, y, decimals=11):
    n = x.shape[0]
    dx = np.round(_pairwise_squared_distances(x), decimals=decimals)
    dy = np.round(_pairwise_squared_distances(y), decimals=decimals)
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    nearest_x = np.argsort(dx, axis=1, kind="stable")[:, 0]
    order_y = np.argsort(dy, axis=1, kind="stable")
    rank_y = np.empty_like(order_y)
    rank_y[np.arange(n)[:, None], order_y] = np.arange(n)[None, :]
    return 2.0 * float((rank_y[np.arange(n), nearest_x] + 1).mean()) / n
