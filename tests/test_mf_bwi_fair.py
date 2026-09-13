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


def test_classical_reduction_sanity_check_beta0_q0():
    """beta=0, q=0 classical-IC reduction.

    NOTE on why this is NOT the same comparison as before the redesign: the old
    myopic index broke round-0 ties (every never-yet-active node with no active
    in-neighbors has zero one-step gain AND zero mean-field belief at round 0) with
    an explicit out-degree tie-breaker, so its round-0 CONVERT set was structurally
    informed even before any observation existed.

    The new Lagrangian index has no such tie-breaker: with a fresh Beta(1,1) prior,
    EVERY edge/node posterior mean is identical (0.5), and mean-field beliefs from
    an all-inactive state are identically 0 for every node (a trivial fixed point,
    same degenerate case the old docstring already noted) -- so p01(v)=0 and
    p10(v)=q_hat(v)=0.5 for literally every node, meaning every node's round-0
    Bellman value is IDENTICAL. The bisection's tie-break fill then picks whichever
    k nodes happen to sort first among exactly-tied gaps (i.e. essentially
    arbitrary/insertion-order), which carries no structural signal and is not a
    meaningful thing to compare against KKT-greedy's degree/spread-informed choice.

    So instead of comparing MF-BWI-Fair's round-0 seed set directly, we run the
    FULL T-round sequential policy (letting real observations differentiate nodes'
    Bayesian posteriors and mean-field beliefs from round 1 onward) and check that
    its final spread is (a) meaningfully above doing nothing, and (b) within a
    generous order-of-magnitude band of KKT-greedy's one-shot classical spread over
    the same horizon -- still a "directionally consistent, not necessarily
    equal/better" check, as before, just applied to the full sequential run rather
    than an isolated, structurally-uninformative round-0 comparison.
    """
    G = graphs.erdos_renyi_graph(40, 0.06, num_groups=2, seed=100)
    graphs.assign_true_parameters(G, p_plus_range=(0.05, 0.2), q_range=(0.0, 0.0), seed=100)
    p_plus, q = true_params_from_graph(G)

    k = 3
    T = 8
    budget = 5 * k

    rng = np.random.default_rng(100)
    seed_set_kkt, _spread_est = celf_greedy(G, p_plus, k, num_sims=300, rng=rng)
    spread_kkt = _avg_spread_in_sim(G, p_plus, q, seed_set_kkt, T)
    assert spread_kkt > k  # sanity: cascading actually happened

    result = run_mf_bwi_fair(G, true_beta=0.0, T=T, budget=budget, alpha_fair=0.0, seed=100)
    spread_mf = float(count_active(result["trajectory"][-1]))

    assert spread_mf > 0  # some diffusion happened at all
    # Generous directional-consistency band (looser than before the redesign,
    # given the round-0 signal loss explained above).
    assert 0.2 * spread_kkt <= spread_mf <= 3.0 * spread_kkt


def test_budget_invariant_never_violated_across_random_graphs_and_rounds():
    rng_master = np.random.default_rng(0)
    for trial in range(6):
        n = int(rng_master.integers(15, 35))
        p_edge = float(rng_master.uniform(0.08, 0.25))
        beta = float(rng_master.uniform(0.0, 0.6))
        alpha_fair = float(rng_master.uniform(-2.0, 1.0))
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


def test_tie_break_fill_reduces_leftover_budget_in_the_full_policy():
    """End-to-end (not synthetic-arm) version of the tie-break fill check: build a
    real MFBWIFair instance, compute the same (p01, p10, w) arrays choose_actions
    builds internally (via the same public helper functions), confirm the raw
    bisection alone would leave some budget unused on this instance, then confirm
    the policy's actually-logged spend (bisection + fill) is at least as large and
    still respects the budget."""
    from im_lab import lagrangian_index
    from im_lab.fairness import group_welfare_weights
    from im_lab.mf_bwi_fair import mean_field_beliefs, field_transition_probs

    G = graphs.stochastic_block_model_graph([10, 10], p_in=0.25, p_out=0.05, seed=11)
    graphs.assign_true_parameters(G, seed=11)
    budget = 23.0  # not a clean multiple of 5 or 1-heavy mixes -> likely leftover

    policy = MFBWIFair(G, budget=budget, alpha_fair=0.0)
    state = {v: (v % 3 == 0) for v in G.nodes()}  # arbitrary non-trivial state

    p_plus_hat, p_minus_hat, q_hat = policy.posterior_means()
    m = mean_field_beliefs(G, state, p_plus_hat, p_minus_hat, q_hat)
    p01, p10 = field_transition_probs(G, m, p_plus_hat, p_minus_hat, q_hat)
    policy.group_u_avg = policy._group_active_fraction(state)
    policy.group_round_count = {g: 1 for g in policy.group_sizes}
    w = group_welfare_weights(policy.group_sizes, policy.group_u_avg, policy.alpha_fair, policy.u_floor)

    s_arr = np.array([1.0 if state[v] else 0.0 for v in policy.nodes])
    p01_arr = np.array([p01[v] for v in policy.nodes])
    p10_arr = np.array([p10[v] for v in policy.nodes])
    w_arr = np.array([w[policy.group_of[v]] for v in policy.nodes])

    _lam, paid, raw_cost_arr, _gap, cost_if_paid = lagrangian_index.solve_lambda_bisection(
        s_arr, p01_arr, p10_arr, w_arr, budget, policy.gamma,
        float(ACTION_COST[Action.CONVERT]), float(ACTION_COST[Action.MAINTAIN]),
        n_bisect_iters=policy.n_bisect_iters, n_vi_sweeps=policy.n_vi_sweeps,
    )
    raw_cost = float(raw_cost_arr.sum())
    assert raw_cost <= budget + 1e-9
    raw_leftover = budget - raw_cost
    # Is there at least one un-paid arm whose paid-action cost fits the raw
    # leftover, i.e. a fill opportunity the bisection itself left on the table?
    fill_opportunity_exists = bool(
        np.any((~paid) & (cost_if_paid <= raw_leftover + 1e-9))
    )

    policy.choose_actions(state, np.random.default_rng(0))
    actual_spend = policy.budget_usage_log[-1]

    assert actual_spend <= budget + 1e-9
    assert actual_spend >= raw_cost - 1e-9
    if fill_opportunity_exists:
        assert actual_spend > raw_cost + 1e-9, (
            "tie-break fill did not use an available leftover-budget opportunity"
        )


def test_lower_alpha_shifts_realized_reach_toward_worst_off_group():
    """Replaces the old fairness-floor test (no longer meaningful: floors are
    gone). Checks the new mechanism's actual intended effect: decreasing
    alpha_fair (more inequality-averse) should not decrease -- and, on
    average, should increase -- the WORST-OFF group's realized reach share,
    averaged over enough trials to not be flaky.

    Which group is "worst-off" is determined per-seed from the alpha=1.0
    (utilitarian, no fairness weighting) baseline, NOT hardcoded to the
    smaller group by size. An earlier version of this test assumed the
    smaller group (by node count) is always the worst-off one -- true under
    the population-weighted welfare form (where the group-size factor
    actively suppresses a small group's weight), but this project's default
    is the population-UNweighted/egalitarian form, under which a smaller
    group can actually saturate FASTER under a fixed budget (more of it is
    covered per seed) and end up the BETTER-off one on a small graph -- a
    confirmed, non-buggy effect (see the diagnostic investigation in this
    project's history), not something this test should assume away.
    """
    sizes = [8, 16]
    n_trials = 20
    T, budget = 6, 8

    def group_reach(alpha_fair: float, seed: int) -> dict:
        G = graphs.stochastic_block_model_graph(sizes, p_in=0.3, p_out=0.03, seed=seed)
        graphs.assign_true_parameters(G, seed=seed)
        result = run_mf_bwi_fair(G, true_beta=0.1, T=T, budget=budget, alpha_fair=alpha_fair, seed=seed)
        policy = result["policy"]
        final_state = result["trajectory"][-1]
        return {
            g: sum(1 for v in nodes if final_state[v]) / len(nodes)
            for g, nodes in policy.nodes_by_group.items()
        }

    utilitarian_worst = []
    averse_same_group = []
    for seed in range(n_trials):
        reach_util = group_reach(1.0, seed)
        worst_g = min(reach_util, key=lambda g: reach_util[g])
        reach_averse = group_reach(-2.0, seed)
        utilitarian_worst.append(reach_util[worst_g])
        averse_same_group.append(reach_averse[worst_g])

    mean_utilitarian = float(np.mean(utilitarian_worst))
    mean_averse = float(np.mean(averse_same_group))

    assert mean_averse >= mean_utilitarian - 1e-9, (
        f"more inequality-averse alpha did not help the worst-off group: "
        f"utilitarian={mean_utilitarian}, inequality-averse={mean_averse}"
    )
    # Non-trivial effect, not just noise-level parity.
    assert mean_averse > mean_utilitarian


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
