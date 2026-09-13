import math

import networkx as nx
import numpy as np

from im_lab import graphs
from im_lab.baselines.imm import imm_select, node_selection, sample_rr_set
from im_lab.baselines.kkt_greedy import celf_greedy, expected_spread
from im_lab.simulator import true_params_from_graph


# ---------------------------------------------------------------------------
# 1. Hand-computable small graphs: IMM should pick the (obviously) best seed.
# ---------------------------------------------------------------------------


def test_imm_star_picks_the_hub():
    """Star graph: one hub with high-probability edges to several leaves, plus
    a couple of low-probability leaf-leaf edges. For k=1 the optimal seed is
    obviously the hub (it alone can reach every leaf directly; any single leaf
    can reach at most the hub plus, transitively, other leaves at a much lower
    combined probability)."""
    G = nx.DiGraph()
    hub = 0
    leaves = [1, 2, 3, 4, 5]
    p_plus = {}
    for leaf in leaves:
        G.add_edge(hub, leaf)
        p_plus[(hub, leaf)] = 0.9
        G.add_edge(leaf, hub)
        p_plus[(leaf, hub)] = 0.05
    rng = np.random.default_rng(0)
    seed_set = imm_select(G, p_plus, k=1, epsilon=0.3, ell=1.0, rng=rng)
    assert seed_set == [hub]


def test_imm_directed_path_picks_the_source():
    """Directed path 0 -> 1 -> 2 -> 3 -> 4 with a constant high edge
    probability: the only node that can (in expectation) reach the whole
    path is the source, node 0."""
    n = 5
    G = nx.DiGraph()
    p_plus = {}
    for i in range(n - 1):
        G.add_edge(i, i + 1)
        p_plus[(i, i + 1)] = 0.85
    for i in range(n):
        if i not in G.nodes():
            G.add_node(i)
    rng = np.random.default_rng(1)
    seed_set = imm_select(G, p_plus, k=1, epsilon=0.3, ell=1.0, rng=rng)
    assert seed_set == [0]


# ---------------------------------------------------------------------------
# 2. IMM's spread (via INDEPENDENT Monte Carlo simulation, not its own RR-set
#    estimate) should be close to a brute-force/CELF-greedy optimum -- this is
#    the check that IMM behaves like a genuine (1-1/e-epsilon)-approximate
#    algorithm rather than just "some algorithm that runs".
# ---------------------------------------------------------------------------


def _brute_force_best_seed_set(G, p_plus, k, num_sims, rng):
    import itertools

    best_set, best_spread = None, -1.0
    for combo in itertools.combinations(G.nodes(), k):
        s = expected_spread(G, list(combo), p_plus, num_sims, rng)
        if s > best_spread:
            best_set, best_spread = list(combo), s
    return best_set, best_spread


def test_imm_spread_close_to_brute_force_optimum_small_graph():
    """Small enough (8 nodes) that brute force over all size-2 seed sets is
    cheap. IMM's independently-simulated spread should land within a few
    percent of the true brute-force optimum's independently-simulated spread
    (allowing generous MC slack, since both sides are noisy estimates)."""
    G = graphs.erdos_renyi_graph(8, 0.35, num_groups=2, seed=7)
    graphs.assign_true_parameters(G, p_plus_range=(0.15, 0.35), seed=7)
    p_plus, _q = true_params_from_graph(G)
    k = 2

    rng_bf = np.random.default_rng(7)
    _best_set, best_spread = _brute_force_best_seed_set(G, p_plus, k, num_sims=400, rng=rng_bf)

    rng_imm = np.random.default_rng(70)
    imm_seeds = imm_select(G, p_plus, k, epsilon=0.3, ell=1.0, rng=rng_imm)
    assert len(imm_seeds) == k
    assert len(set(imm_seeds)) == k

    rng_eval = np.random.default_rng(700)
    imm_spread = expected_spread(G, imm_seeds, p_plus, num_sims=2000, rng=rng_eval)

    # Generous slack: within 15% of the brute-force optimum (both quantities
    # are themselves noisy MC estimates on an 8-node graph).
    assert imm_spread >= 0.85 * best_spread


def test_imm_spread_competitive_with_celf_greedy_on_moderate_graph():
    G = graphs.erdos_renyi_graph(25, 0.15, num_groups=2, seed=11)
    graphs.assign_true_parameters(G, p_plus_range=(0.1, 0.3), seed=11)
    p_plus, _q = true_params_from_graph(G)
    k = 3

    rng_celf = np.random.default_rng(11)
    celf_seeds, _ = celf_greedy(G, p_plus, k, num_sims=200, rng=rng_celf)

    rng_imm = np.random.default_rng(110)
    imm_seeds = imm_select(G, p_plus, k, epsilon=0.3, ell=1.0, rng=rng_imm)

    rng_eval = np.random.default_rng(1100)
    celf_spread = expected_spread(G, celf_seeds, p_plus, num_sims=1500, rng=rng_eval)
    imm_spread = expected_spread(G, imm_seeds, p_plus, num_sims=1500, rng=rng_eval)

    assert imm_spread >= 0.85 * celf_spread


# ---------------------------------------------------------------------------
# 3. RR-set sampling correctness in isolation, on a tiny fixed graph with
#    hand-computable reachability probabilities.
# ---------------------------------------------------------------------------


def test_rr_set_coverage_matches_hand_computed_probabilities():
    """Graph: 0 -> 1 -> 2, with p_plus[0,1] = 0.6, p_plus[1,2] = 0.5, and an
    isolated node 3. Fixing the target at node 2, the RR set (reverse
    reachability from 2) is:
      - always contains 2.
      - contains 1 iff the (1,2) trial succeeds: Pr = 0.5.
      - contains 0 iff (1,2) succeeds AND (0,1) succeeds: Pr = 0.5*0.6 = 0.3.
      - never contains 3 (no path from 3 to 2).
    We sample many RR sets with target forced to 2 and check the empirical
    inclusion frequencies against these exact probabilities within MC noise.
    """
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    p_plus = {(0, 1): 0.6, (1, 2): 0.5}

    rng = np.random.default_rng(42)
    num_samples = 20000
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(num_samples):
        rr = sample_rr_set(G, p_plus, rng, target=2)
        for v in rr:
            counts[v] += 1

    freq = {v: counts[v] / num_samples for v in counts}
    # 3-sigma-ish binomial tolerance for p in {0.3, 0.5, 1.0}: sqrt(p(1-p)/n)
    # with n=20000 gives at most ~0.0035, use a generous 0.02 absolute margin.
    assert freq[2] == 1.0
    assert abs(freq[1] - 0.5) < 0.02
    assert abs(freq[0] - 0.3) < 0.02
    assert freq[3] == 0.0


def test_rr_set_coverage_estimates_expected_spread_of_a_singleton():
    """sigma({v}) = n * E[1{R contains v}] (the RR-set duality this whole
    approach rests on -- Borgs et al. 2014). Check this numerically for a
    singleton seed on a small graph against the same quantity computed by
    direct IC Monte Carlo simulation (independent code path)."""
    G = graphs.erdos_renyi_graph(12, 0.3, num_groups=2, seed=3)
    graphs.assign_true_parameters(G, p_plus_range=(0.1, 0.3), seed=3)
    p_plus, _q = true_params_from_graph(G)
    n = G.number_of_nodes()
    v = 0

    rng_rr = np.random.default_rng(3)
    num_rr = 20000
    hits = 0
    for _ in range(num_rr):
        rr = sample_rr_set(G, p_plus, rng_rr)
        if v in rr:
            hits += 1
    rr_estimate = n * hits / num_rr

    rng_ic = np.random.default_rng(30)
    ic_estimate = expected_spread(G, [v], p_plus, num_sims=5000, rng=rng_ic)

    assert abs(rr_estimate - ic_estimate) < 0.75


def test_node_selection_picks_maximum_coverage_greedily():
    # 4 RR sets over 3 nodes: node "a" alone covers 3 of them.
    rr_sets = [{"a", "b"}, {"a"}, {"a", "c"}, {"b", "c"}]
    node_to_rrsets = {"a": [0, 1, 2], "b": [0, 3], "c": [2, 3]}
    seed_set, covered = node_selection(rr_sets, node_to_rrsets, k=1)
    assert seed_set == ["a"]
    assert covered == 3

    seed_set2, covered2 = node_selection(rr_sets, node_to_rrsets, k=2)
    assert set(seed_set2) == {"a", "b"} or set(seed_set2) == {"a", "c"}
    assert covered2 == 4


# ---------------------------------------------------------------------------
# 4. Scaling sanity check on the ~120-node SBM graph used elsewhere in this
#    repo (see im_lab/graphs.py, experiments/common.py): IMM should complete
#    quickly and be competitive with CELF-greedy on the same instance.
# ---------------------------------------------------------------------------


def test_imm_scales_to_sbm_graph_and_is_competitive_with_celf():
    import time

    sizes = [20, 40, 60]  # matches experiments/common.py's SIZES (120 nodes)
    G = graphs.stochastic_block_model_graph(sizes, p_in=0.07, p_out=0.008, seed=42)
    graphs.assign_true_parameters(G, p_plus_range=(0.05, 0.15), seed=42)
    p_plus, _q = true_params_from_graph(G)
    k = 20

    t0 = time.time()
    rng_imm = np.random.default_rng(123)
    imm_seeds = imm_select(G, p_plus, k, epsilon=0.4, ell=1.0, rng=rng_imm)
    imm_time = time.time() - t0

    assert imm_time < 60.0
    assert len(imm_seeds) == k
    assert len(set(imm_seeds)) == k

    rng_celf = np.random.default_rng(124)
    celf_seeds, _ = celf_greedy(G, p_plus, k, num_sims=40, rng=rng_celf)

    rng_eval = np.random.default_rng(125)
    imm_spread = expected_spread(G, imm_seeds, p_plus, num_sims=300, rng=rng_eval)
    celf_spread = expected_spread(G, celf_seeds, p_plus, num_sims=300, rng=rng_eval)

    # IMM should be in the same ballpark as CELF-greedy on this instance -- not
    # necessarily better, but not dramatically worse (which would indicate a
    # bug rather than approximation-ratio slack).
    assert imm_spread >= 0.75 * celf_spread
