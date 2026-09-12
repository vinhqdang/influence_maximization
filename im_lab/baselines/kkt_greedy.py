"""Classical Kempe-Kleinberg-Tardos (2003) greedy under plain progressive IC
(beta=0, q=0: no backfire, no recovery -- once active, always active), with Monte
Carlo spread estimation and the CELF (Cost-Effective Lazy Forward) speedup of
Leskovec et al. (2007). This is the "ignore all four relaxations" ablation
baseline: no fairness, no uncertainty (true p_plus used directly), no non-progressive
dynamics, no backfire.

Submodularity of the expected-spread function under IC is exactly what makes greedy
a (1-1/e)-approximation (Kempe, Kleinberg & Tardos 2003; Nemhauser, Wolsey & Fisher
1978 for the general greedy-on-submodular guarantee); CELF exploits that
submodularity to skip most marginal-gain recomputations via lazy evaluation instead
of a proven closed-form shortcut.
"""

from __future__ import annotations

import heapq

import networkx as nx
import numpy as np


def ic_cascade_once(G: nx.DiGraph, seed_set, p_plus: dict, rng: np.random.Generator) -> set:
    """One Monte Carlo realization of progressive IC starting from seed_set."""
    active = set(seed_set)
    frontier = list(seed_set)
    while frontier:
        next_frontier = []
        for u in frontier:
            for v in G.successors(u):
                if v in active:
                    continue
                if rng.random() < p_plus[(u, v)]:
                    active.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
    return active


def expected_spread(
    G: nx.DiGraph, seed_set, p_plus: dict, num_sims: int, rng: np.random.Generator
) -> float:
    if not seed_set:
        return 0.0
    total = 0
    for _ in range(num_sims):
        total += len(ic_cascade_once(G, seed_set, p_plus, rng))
    return total / num_sims


def celf_greedy(
    G: nx.DiGraph, p_plus: dict, k: int, num_sims: int = 200, rng: np.random.Generator = None
) -> tuple[list, float]:
    """Lazy-forward greedy seed selection maximizing expected IC spread.

    Returns (seed_set, expected_spread_estimate).
    """
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G.nodes())

    # Initial marginal gains (singleton spreads); heap entries are
    # (-gain, node, "last round this gain was recomputed for").
    heap = []
    for v in nodes:
        gain = expected_spread(G, [v], p_plus, num_sims, rng)
        heap.append((-gain, v, 0))
    heapq.heapify(heap)

    seed_set: list = []
    cur_spread = 0.0
    while heap and len(seed_set) < k:
        neg_gain, v, last_round = heapq.heappop(heap)
        if last_round == len(seed_set):
            # Gain was already computed against the current seed set -- accept it.
            seed_set.append(v)
            cur_spread += -neg_gain
        else:
            new_spread = expected_spread(G, seed_set + [v], p_plus, num_sims, rng)
            new_gain = new_spread - cur_spread
            heapq.heappush(heap, (-new_gain, v, len(seed_set)))

    return seed_set, cur_spread
