import numpy as np
import networkx as nx

from im_lab import graphs
from im_lab.actions import Action, ACTION_COST
from im_lab.baselines.kkt_greedy import celf_greedy
from im_lab.bayes import BetaBernoulliTracker
from im_lab.mf_bwi_fair import MFBWIFair, run_mf_bwi_fair
from im_lab.simulator import (
    count_active,
    initial_state,
    seed_then_none_actions,
    simulate,
    simulate_step,
    true_params_from_graph,
)


def _avg_spread_in_sim(G, p_plus, q, seeds, T, reps=40, base_seed=3000):
    init = initial_state(G, seeds)
    acts = seed_then_none_actions(G, seeds, T)
    vals = []
    for r in range(reps):
        rng = np.random.default_rng(base_seed + r)
        traj, _obs = simulate(G, init, p_plus, q, 0.0, acts, rng)
        vals.append(count_active(traj[-1]))
    return float(np.mean(vals))


def test_regression_matches_kkt_greedy_when_reduced_to_classical_case():
    """beta=0, q=0, budget allows exactly k CONVERTs at round 0 and nothing after.

    Both the KKT-greedy seed set and MF-BWI-Fair's own round-0 seed choice are
    evaluated by running them through the SAME sequential simulator (seed once,
    then NONE), since that is the fair, apples-to-apples comparison the spec asks
    for ("to compare fairly in our sequential setting..."). MF-BWI-Fair's index is
    a one-step/mean-field heuristic (not a proven-optimal greedy), so we only check
    that its resulting spread is *directionally consistent* with (same order of
    magnitude as) classical greedy's, not that it is equal or better.
    """
    G = graphs.erdos_renyi_graph(40, 0.06, num_groups=2, seed=100)
    graphs.assign_true_parameters(G, p_plus_range=(0.05, 0.2), q_range=(0.0, 0.0), seed=100)
    p_plus, q = true_params_from_graph(G)

    k = 3
    T = 4

    rng = np.random.default_rng(100)
    seed_set_kkt, _spread_est = celf_greedy(G, p_plus, k, num_sims=300, rng=rng)

    policy = MFBWIFair(G, budget=5 * k, alpha_fair=0.0)
    state0 = {v: False for v in G.nodes()}
    actions0 = policy.choose_actions(state0, np.random.default_rng(1))
    seed_set_mf = [v for v, a in actions0.items() if a == Action.CONVERT]

    assert len(seed_set_mf) == k

    spread_kkt = _avg_spread_in_sim(G, p_plus, q, seed_set_kkt, T)
    spread_mf = _avg_spread_in_sim(G, p_plus, q, seed_set_mf, T)

    assert spread_kkt > k  # sanity: cascading actually happened
    assert spread_mf > k
    # Directionally consistent: same order of magnitude, not necessarily equal.
    assert 0.5 * spread_kkt <= spread_mf <= 2.0 * spread_kkt


def test_budget_invariant_never_violated_across_random_graphs_and_rounds():
    rng_master = np.random.default_rng(0)
    for trial in range(6):
        n = int(rng_master.integers(15, 35))
        p_edge = float(rng_master.uniform(0.08, 0.25))
        beta = float(rng_master.uniform(0.0, 0.6))
        alpha_fair = float(rng_master.uniform(0.0, 0.8))
        budget = float(rng_master.integers(3, 20))
        num_groups = int(rng_master.integers(2, 4))

        G = graphs.erdos_renyi_graph(n, p_edge, num_groups=num_groups, seed=trial)
        graphs.assign_true_parameters(G, seed=trial)

        result = run_mf_bwi_fair(
            G, true_beta=beta, T=10, budget=budget, alpha_fair=alpha_fair, seed=trial
        )
        for spend in result["budget_usage_log"]:
            assert spend <= budget + 1e-9
        for group_spend in result["group_spend_log"]:
            assert sum(group_spend.values()) <= budget + 1e-9


def test_fairness_floor_met_when_jointly_feasible():
    """Two equal-size groups, alpha small enough that floors sum well within
    budget (always true since sum_g floor_g = alpha*budget <= budget) and there
    are plenty of candidate nodes per group -- floors must be met every round."""
    sizes = [12, 12]
    G = graphs.stochastic_block_model_graph(sizes, p_in=0.3, p_out=0.05, seed=9)
    graphs.assign_true_parameters(G, seed=9)

    budget = 20.0
    alpha_fair = 0.6
    result = run_mf_bwi_fair(
        G, true_beta=0.1, T=6, budget=budget, alpha_fair=alpha_fair, seed=9
    )
    policy = result["policy"]

    for round_idx in range(len(result["group_spend_log"])):
        satisfied = policy.floor_satisfied(round_idx)
        assert all(satisfied.values()), (
            f"round {round_idx}: floors not met: {satisfied}, "
            f"spend={result['group_spend_log'][round_idx]}"
        )


def test_bayesian_posterior_converges_via_full_simulation_loop():
    """Consistency check: run many rounds of the real simulator (hidden true
    parameters) feeding observations into a BetaBernoulliTracker exactly as
    MFBWIFair does, and confirm the posterior mean approaches the true value."""
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1)
    for v in G.nodes():
        G.nodes[v]["group"] = 0
    true_p_plus = 0.65
    true_q = 0.2
    G.edges[0, 1]["p_plus"] = true_p_plus
    G.nodes[0]["q"] = true_q
    G.nodes[1]["q"] = true_q

    p_plus, q = true_params_from_graph(G)
    tracker = BetaBernoulliTracker()

    state = {0: True, 1: False}
    rng = np.random.default_rng(123)
    for _ in range(4000):
        actions = {0: Action.NONE, 1: Action.NONE}
        new_state, obs = simulate_step(G, state, p_plus, q, beta=0.0, actions=actions, rng=rng)
        for o in obs:
            if o["param"] == "q":
                key = ("q", o["v"])
            else:
                key = (o["param"], o["u"], o["v"])
            tracker.update(key, o["success"])
        # Re-seed node 1 back to inactive periodically so p_plus keeps getting
        # exercised (otherwise, once active, the p_plus edge stops being tested).
        state = new_state
        if state[1]:
            state = {0: True, 1: False}

    assert abs(tracker.mean(("p_plus", 0, 1)) - true_p_plus) < 0.07
