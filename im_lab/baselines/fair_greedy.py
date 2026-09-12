"""Welfare-weighted fair greedy baseline, in the style of Rahmattalabi et al. (2021,
"Fair Influence Maximization"): rather than maximizing raw total spread, maximize a
sum of concave per-group utilities

  W(seed_set) = sum_g  w_g * U_g(reach_g(seed_set)),   U_g(x) = log(1 + x)

The log concavity gives diminishing returns to over-serving any single group, which
is what produces more balanced allocations than plain spread-maximizing greedy; W is
still monotone submodular in the seed set (a non-negative concave transform of a
monotone submodular reach function is itself monotone submodular), so the same
greedy-with-MC-estimation recipe applies. Plain IC (beta=0, q=0), same as
kkt_greedy.py -- this is likewise a one-shot ablation baseline, without the
Bayesian-uncertainty or non-progressive/backfire relaxations.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .kkt_greedy import ic_cascade_once


def expected_reach_by_group(
    G: nx.DiGraph,
    seed_set,
    p_plus: dict,
    group_of: dict,
    num_sims: int,
    rng: np.random.Generator,
) -> dict:
    groups = set(group_of.values())
    totals = {g: 0.0 for g in groups}
    if not seed_set:
        return totals
    for _ in range(num_sims):
        active = ic_cascade_once(G, seed_set, p_plus, rng)
        for v in active:
            totals[group_of[v]] += 1.0
    return {g: totals[g] / num_sims for g in groups}


def welfare(reach_by_group: dict, weights: dict) -> float:
    return sum(weights[g] * np.log1p(reach_by_group.get(g, 0.0)) for g in weights)


def fair_welfare_greedy(
    G: nx.DiGraph,
    p_plus: dict,
    k: int,
    group_of: dict,
    weights: dict = None,
    num_sims: int = 100,
    rng: np.random.Generator = None,
) -> tuple[list, float, dict]:
    """Greedy seed selection maximizing sum_g w_g * log(1 + reach_g).

    Returns (seed_set, welfare_value, reach_by_group).
    """
    if rng is None:
        rng = np.random.default_rng()
    groups = set(group_of.values())
    if weights is None:
        weights = {g: 1.0 for g in groups}

    seed_set: list = []
    cur_reach = {g: 0.0 for g in groups}
    cur_welfare = welfare(cur_reach, weights)

    remaining = set(G.nodes())
    for _ in range(k):
        best_v, best_gain, best_reach, best_w = None, -np.inf, None, None
        for v in remaining:
            reach = expected_reach_by_group(
                G, seed_set + [v], p_plus, group_of, num_sims, rng
            )
            w = welfare(reach, weights)
            gain = w - cur_welfare
            if gain > best_gain:
                best_v, best_gain, best_reach, best_w = v, gain, reach, w
        if best_v is None:
            break
        seed_set.append(best_v)
        remaining.discard(best_v)
        cur_reach, cur_welfare = best_reach, best_w

    return seed_set, cur_welfare, cur_reach
