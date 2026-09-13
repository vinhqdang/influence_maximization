"""Tests for im_lab/baselines/robust_kempe.py (He & Kempe 2016, "Robust
Influence Maximization", Saturate Greedy). See that module's docstring for
exactly which parts of the algorithm are verified against the paper's text.
"""

from __future__ import annotations

import itertools
import time

import networkx as nx
import numpy as np

from im_lab import graphs
from im_lab.baselines.kkt_greedy import celf_greedy, expected_spread
from im_lab.baselines.robust_kempe import robust_select, saturate_greedy
from im_lab.simulator import true_params_from_graph


# ---------------------------------------------------------------------------
# 1. Hand-constructed counterexample (mirrors the paper's own Section 4.2
#    construction: a directed bipartite "fan" per scenario, plus a
#    deterministic pendant pair per seed slot) demonstrating BOTH halves of
#    the paper's claim concretely:
#      (a) plain single-scenario greedy picks a seed set that is great under
#          the scenario it was optimized for but bad under the other, and
#      (b) the robust algorithm instead picks a seed set that is good under
#          both scenarios simultaneously.
# ---------------------------------------------------------------------------


def _build_two_scenario_fan_instance():
    """k=2 budget. Nodes: x1, x2 (bipartite hubs), y1..y6 (leaves reachable
    only through whichever x_i is "the good one" in a given scenario), and
    two deterministic pendant pairs (u1, v1), (u2, v2) that are always fully
    active in every scenario regardless of which node is seeded (u_i -> v_i
    always succeeds).

    Scenario A: x1 -> y_j has probability 1 for all j; x2 -> y_j has
    probability 0. Scenario B: the reverse (only x2 reaches the y's).

    With budget k=2:
    - {x1, x2} reaches all 6 y's under EITHER scenario (spread 8 under both:
      2 seeds + 6 leaves), so it is robust.
    - Greedy run on scenario A alone picks x1 (spread 7: itself + 6 leaves)
      then, since x2 contributes nothing more under A, prefers a pendant
      pair (u1, gain 2: itself + v1) over x2 (gain 0 under A) -- so
      single-scenario greedy on A returns {x1, u1}, spread 9 under A (its
      own optimum) but spread only 3 under B (x1's edges are dead under B,
      and u1/v1 are active regardless of scenario but contribute nothing
      about the leaves) -- a much worse worst-case than {x1, x2}'s 8.
    """
    G = nx.DiGraph()
    xs = ["x1", "x2"]
    ys = [f"y{i}" for i in range(1, 7)]
    G.add_nodes_from(xs + ys)
    for x in xs:
        for y in ys:
            G.add_edge(x, y)
    G.add_edge("u1", "v1")
    G.add_edge("u2", "v2")

    p_plus_A, p_plus_B = {}, {}
    for y in ys:
        p_plus_A[("x1", y)] = 1.0
        p_plus_A[("x2", y)] = 0.0
        p_plus_B[("x1", y)] = 0.0
        p_plus_B[("x2", y)] = 1.0
    for pendant_edge in [("u1", "v1"), ("u2", "v2")]:
        p_plus_A[pendant_edge] = 1.0
        p_plus_B[pendant_edge] = 1.0

    return G, p_plus_A, p_plus_B


def test_single_scenario_greedy_picks_the_wrong_seed_set():
    """Confirms half (a) of the paper's claim: greedy on one scenario alone
    finds ITS OWN optimum but performs badly on the other scenario (all
    probabilities here are 0 or 1, so spreads are exact -- no MC noise)."""
    G, p_plus_A, p_plus_B = _build_two_scenario_fan_instance()
    k = 2
    rng = np.random.default_rng(0)
    single_A_seeds, single_A_spread_reported = celf_greedy(G, p_plus_A, k, num_sims=50, rng=rng)

    assert set(single_A_seeds) == {"x1", "u1"} or set(single_A_seeds) == {"x1", "u2"}
    assert single_A_spread_reported == 9.0

    rng_eval = np.random.default_rng(1)
    spread_under_A = expected_spread(G, single_A_seeds, p_plus_A, 200, rng_eval)
    spread_under_B = expected_spread(G, single_A_seeds, p_plus_B, 200, rng_eval)
    assert spread_under_A == 9.0  # matches its own scenario's optimum exactly
    assert spread_under_B == 3.0  # but collapses under the other scenario
    worst_case_single = min(spread_under_A, spread_under_B)
    assert worst_case_single == 3.0


def test_saturate_greedy_beats_single_scenario_greedy_on_two_incompatible_scenarios():
    """Confirms half (b): the robust algorithm picks {x1, x2} (or a
    superset of it, since it is allowed up to beta*k seeds), whose
    worst-case spread across both scenarios is far better than either
    single-scenario greedy solution's worst-case spread."""
    G, p_plus_A, p_plus_B = _build_two_scenario_fan_instance()
    k = 2
    rng = np.random.default_rng(2)
    robust_seeds = robust_select(G, [p_plus_A, p_plus_B], k, gamma=0.2, num_sims=100, rng=rng)

    assert {"x1", "x2"}.issubset(set(robust_seeds))

    rng_eval = np.random.default_rng(3)
    spread_A = expected_spread(G, robust_seeds, p_plus_A, 200, rng_eval)
    spread_B = expected_spread(G, robust_seeds, p_plus_B, 200, rng_eval)
    worst_case_robust = min(spread_A, spread_B)
    assert worst_case_robust >= 8.0  # {x1,x2} alone already achieves this

    # And it strictly beats BOTH single-scenario greedy choices' worst case.
    for p_plus_single in (p_plus_A, p_plus_B):
        rng_g = np.random.default_rng(4)
        single_seeds, _ = celf_greedy(G, p_plus_single, k, num_sims=50, rng=rng_g)
        rng_eval2 = np.random.default_rng(5)
        wa = expected_spread(G, single_seeds, p_plus_A, 200, rng_eval2)
        wb = expected_spread(G, single_seeds, p_plus_B, 200, rng_eval2)
        assert worst_case_robust > min(wa, wb)


# ---------------------------------------------------------------------------
# 2. Empirical bicriteria-guarantee check on small random instances: measure
#    min-scenario spread of the returned (bicriteria-sized) seed set,
#    independently Monte-Carlo-evaluated, against a brute-force robust OPT
#    (over all size-k seed sets), and check the ratio is in the ballpark
#    Theorem 3 predicts: rho(S_hat) >= (1-1/e)*rho(S*) - gamma.
# ---------------------------------------------------------------------------


def _brute_force_robust_opt(G, scenarios, k, num_sims, rng):
    """rho(S) = min_sigma spread_sigma(S) / OPT_sigma, brute forced over all
    size-k seed sets AND (for the denominator) over all size-k seed sets per
    scenario. Only feasible for small graphs -- used here on tiny instances."""
    nodes = list(G.nodes())

    opt_per_scenario = []
    for p_plus_sigma in scenarios:
        best = 0.0
        for combo in itertools.combinations(nodes, k):
            s = expected_spread(G, list(combo), p_plus_sigma, num_sims, rng)
            best = max(best, s)
        opt_per_scenario.append(max(best, 1e-9))

    best_rho, best_S = -1.0, None
    for combo in itertools.combinations(nodes, k):
        combo = list(combo)
        ratios = []
        for p_plus_sigma, opt in zip(scenarios, opt_per_scenario):
            s = expected_spread(G, combo, p_plus_sigma, num_sims, rng)
            ratios.append(s / opt)
        rho = min(ratios)
        if rho > best_rho:
            best_rho, best_S = rho, combo
    return best_S, best_rho, opt_per_scenario


def test_bicriteria_guarantee_empirically_on_small_random_instance():
    """6-node graph, 2 scenarios, k=2: small enough to brute-force both the
    per-scenario optima and the true robust optimum rho(S*) directly, then
    check Saturate Greedy's independently-evaluated rho is within the
    ballpark of (1-1/e)*rho(S*) - gamma (allowing generous slack for the
    fact both sides are noisy Monte Carlo estimates on a small graph, and
    that gamma itself already bakes in some slack)."""
    G = graphs.erdos_renyi_graph(6, 0.5, num_groups=2, seed=13)
    # Ensure some structure: two different random probability assignments on
    # the SAME edge set act as the two scenarios.
    graphs.assign_true_parameters(G, p_plus_range=(0.1, 0.4), seed=13)
    p_plus_A, _q = true_params_from_graph(G)
    graphs.assign_true_parameters(G, p_plus_range=(0.1, 0.4), seed=99)
    p_plus_B, _q = true_params_from_graph(G)
    scenarios = [p_plus_A, p_plus_B]
    k = 2
    gamma = 0.2

    rng_bf = np.random.default_rng(13)
    _S_star, rho_star, opt_per_scenario = _brute_force_robust_opt(
        G, scenarios, k, num_sims=1500, rng=rng_bf
    )

    rng_alg = np.random.default_rng(14)
    S_hat = saturate_greedy(G, scenarios, k, gamma=gamma, num_sims=150, rng=rng_alg)

    m = len(scenarios)
    import math

    beta = 1.0 + math.log(m) + math.log(3.0 / gamma)
    assert len(S_hat) <= beta * k + 1e-6  # Theorem 3's budget bound

    rng_eval = np.random.default_rng(15)
    ratios = []
    for p_plus_sigma, opt in zip(scenarios, opt_per_scenario):
        s = expected_spread(G, S_hat, p_plus_sigma, 1500, rng_eval)
        ratios.append(s / opt)
    rho_hat = min(ratios)

    predicted_floor = (1.0 - 1.0 / math.e) * rho_star - gamma
    # Generous MC/small-instance slack on top of the theorem's own additive
    # gamma slack, since rho_star, opt_per_scenario, and rho_hat are all
    # themselves noisy Monte Carlo estimates.
    assert rho_hat >= predicted_floor - 0.15


# ---------------------------------------------------------------------------
# 3. Scaling/sanity test on the repo's actual 120-node SBM graph, matching
#    experiments/common.py's topology (SIZES=[20,40,60], p_in=0.07,
#    p_out=0.008) with two scenarios given by p_plus_range's low/high
#    endpoints (0.05, 0.15). Confirms reasonable runtime AND that the
#    robust seed set's worst-case (min over the two scenarios) spread beats
#    plain CELF-greedy run on either single scenario alone.
# ---------------------------------------------------------------------------


def test_robust_select_scales_to_sbm_graph_and_beats_single_scenario_celf():
    sizes = [20, 40, 60]  # matches experiments/common.py's SIZES (120 nodes)
    G = graphs.stochastic_block_model_graph(sizes, p_in=0.07, p_out=0.008, seed=42)

    # Per Lemma 1 of the paper, the worst case for an interval uncertainty
    # set lives at per-edge corner choices; use one random corner
    # assignment as scenario "A" and its edge-wise complement as scenario
    # "B" (rather than uniformly setting every edge to the same endpoint,
    # which -- as verified separately -- produces two scenarios that are
    # just a global rescaling of each other and barely disagree on which
    # nodes are influential).
    rng_mask = np.random.default_rng(42)
    edges = list(G.edges())
    low, high = 0.05, 0.15
    mask = {e: rng_mask.random() < 0.5 for e in edges}
    p_plus_low_corner = {e: (low if mask[e] else high) for e in edges}
    p_plus_high_corner = {e: (high if mask[e] else low) for e in edges}
    scenarios = [p_plus_low_corner, p_plus_high_corner]

    k = 5
    t0 = time.time()
    rng_robust = np.random.default_rng(123)
    robust_seeds = robust_select(G, scenarios, k, gamma=0.2, num_sims=100, rng=rng_robust)
    robust_time = time.time() - t0

    assert robust_time < 60.0
    assert len(robust_seeds) >= k  # bicriteria: at least k, possibly more
    assert len(set(robust_seeds)) == len(robust_seeds)  # no duplicates

    rng_A = np.random.default_rng(1)
    celf_A_seeds, _ = celf_greedy(G, p_plus_low_corner, k, num_sims=100, rng=rng_A)
    rng_B = np.random.default_rng(2)
    celf_B_seeds, _ = celf_greedy(G, p_plus_high_corner, k, num_sims=100, rng=rng_B)

    rng_eval = np.random.default_rng(999)

    def worst_case_spread(seed_set):
        s_a = expected_spread(G, seed_set, p_plus_low_corner, 1500, rng_eval)
        s_b = expected_spread(G, seed_set, p_plus_high_corner, 1500, rng_eval)
        return min(s_a, s_b)

    robust_worst = worst_case_spread(robust_seeds)
    celf_A_worst = worst_case_spread(celf_A_seeds)
    celf_B_worst = worst_case_spread(celf_B_seeds)

    assert robust_worst > celf_A_worst
    assert robust_worst > celf_B_worst
