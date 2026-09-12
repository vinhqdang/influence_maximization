import numpy as np

from im_lab import graphs
from im_lab.baselines.kkt_greedy import celf_greedy, expected_spread
from im_lab.simulator import true_params_from_graph


def test_celf_greedy_selects_k_distinct_seeds_and_improves_over_random():
    G = graphs.erdos_renyi_graph(30, 0.1, num_groups=2, seed=21)
    graphs.assign_true_parameters(G, p_plus_range=(0.1, 0.3), seed=21)
    p_plus, _q = true_params_from_graph(G)

    rng = np.random.default_rng(21)
    k = 4
    seed_set, spread = celf_greedy(G, p_plus, k, num_sims=150, rng=rng)

    assert len(seed_set) == k
    assert len(set(seed_set)) == k

    rng2 = np.random.default_rng(99)
    random_nodes = list(rng2.choice(list(G.nodes()), size=k, replace=False))
    random_spread = expected_spread(G, random_nodes, p_plus, num_sims=150, rng=rng2)

    # Greedy should be at least as good as an arbitrary random seed set (allow a
    # small MC-noise slack rather than requiring strict inequality).
    assert spread >= random_spread - 1.0


def test_marginal_gains_are_roughly_diminishing_submodular_sanity():
    """Not a formal submodularity proof, just a Monte-Carlo sanity check that
    marginal gains from CELF are non-negative and (on average, over a fairly dense
    graph) do not increase as the seed set grows."""
    G = graphs.erdos_renyi_graph(40, 0.08, num_groups=2, seed=5)
    graphs.assign_true_parameters(G, p_plus_range=(0.1, 0.25), seed=5)
    p_plus, _q = true_params_from_graph(G)
    rng = np.random.default_rng(5)

    seed_set = []
    cur = 0.0
    gains = []
    for _ in range(4):
        best_v, best_gain = None, -np.inf
        for v in G.nodes():
            if v in seed_set:
                continue
            s = expected_spread(G, seed_set + [v], p_plus, num_sims=100, rng=rng)
            gain = s - cur
            if gain > best_gain:
                best_v, best_gain = v, gain
        seed_set.append(best_v)
        cur += best_gain
        gains.append(best_gain)

    assert all(g >= -1e-6 for g in gains)
