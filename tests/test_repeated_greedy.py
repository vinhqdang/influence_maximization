import numpy as np

from im_lab import graphs
from im_lab.actions import ACTION_COST
from im_lab.baselines.repeated_greedy import RepeatedGreedy, run_repeated_greedy
from im_lab.simulator import (
    count_active,
    initial_state,
    simulate,
    true_params_from_graph,
)


def _all_none_actions(G, T):
    from im_lab.actions import Action

    return [{v: Action.NONE for v in G.nodes()} for _ in range(T)]


def test_budget_invariant_never_violated_across_random_graphs_and_rounds():
    rng_master = np.random.default_rng(0)
    for trial in range(6):
        n = int(rng_master.integers(12, 30))
        p_edge = float(rng_master.uniform(0.1, 0.3))
        beta = float(rng_master.uniform(0.0, 0.6))
        budget = float(rng_master.integers(3, 20))
        num_groups = int(rng_master.integers(2, 4))

        G = graphs.erdos_renyi_graph(n, p_edge, num_groups=num_groups, seed=trial)
        graphs.assign_true_parameters(G, seed=trial)

        result = run_repeated_greedy(G, true_beta=beta, T=6, budget=budget, seed=trial)

        assert len(result["budget_usage_log"]) == 6
        for spend in result["budget_usage_log"]:
            assert spend <= budget + 1e-9

        # Cross-check against the actual per-round action costs.
        for actions in result["actions_history"]:
            total = sum(ACTION_COST[a] for a in actions.values())
            assert total <= budget + 1e-9


def test_single_round_action_set_never_exceeds_budget_directly():
    """Directly exercise RepeatedGreedy.choose_actions (not just the T-round
    wrapper) across a range of budgets, including budgets too small to afford
    even one CONVERT."""
    G = graphs.stochastic_block_model_graph([8, 10], p_in=0.25, p_out=0.05, seed=3)
    graphs.assign_true_parameters(G, p_plus_range=(0.1, 0.3), q_range=(0.05, 0.2), seed=3)

    state = initial_state(G, [])
    rng = np.random.default_rng(11)

    for budget in (0.0, 1.0, 3.0, 4.0, 10.0, 50.0):
        rg = RepeatedGreedy(G, budget=budget, true_beta=0.2)
        actions = rg.choose_actions(state, rng)
        total = sum(ACTION_COST[a] for a in actions.values())
        assert total <= budget + 1e-9


def test_repeated_greedy_beats_doing_nothing_on_a_small_test_graph():
    """Sanity check: with a real budget every round, repeated_greedy should
    produce materially more spread over the horizon than a "do nothing ever"
    policy on a graph where the natural dynamics alone barely cascade."""
    G = graphs.stochastic_block_model_graph([15, 15], p_in=0.2, p_out=0.03, seed=17)
    graphs.assign_true_parameters(G, p_plus_range=(0.08, 0.2), q_range=(0.1, 0.2), seed=17)
    p_plus, q = true_params_from_graph(G)

    T = 12
    budget = 15.0
    beta = 0.1

    # Repeated-greedy, averaged over a few seeds to smooth MC noise.
    rg_finals = []
    for seed in range(4):
        result = run_repeated_greedy(G, true_beta=beta, T=T, budget=budget, seed=seed)
        rg_finals.append(count_active(result["trajectory"][-1]))
    rg_mean = float(np.mean(rg_finals))

    # Do-nothing baseline: no seeding at all, no actions ever, same dynamics.
    none_finals = []
    for seed in range(4):
        init = initial_state(G, [])
        acts = _all_none_actions(G, T)
        rng = np.random.default_rng(1000 + seed)
        traj, _obs = simulate(G, init, p_plus, q, beta, acts, rng)
        none_finals.append(count_active(traj[-1]))
    none_mean = float(np.mean(none_finals))

    assert rg_mean > none_mean + 1.0, (
        f"repeated_greedy ({rg_mean}) should clearly beat doing nothing ({none_mean})"
    )


def test_actions_only_ever_convert_inactive_and_maintain_active_nodes():
    """Structural sanity: repeated_greedy must respect the model's action
    validity rules (CONVERT only on inactive nodes, MAINTAIN only on active
    ones) at every round -- i.e. it never proposes a nonsensical action."""
    from im_lab.actions import Action, is_valid_action

    G = graphs.erdos_renyi_graph(25, 0.15, num_groups=2, seed=4)
    graphs.assign_true_parameters(G, seed=4)

    rng = np.random.default_rng(4)
    rg = RepeatedGreedy(G, budget=20.0, true_beta=0.2)
    state = initial_state(G, [])
    for _ in range(5):
        actions = rg.choose_actions(state, rng)
        for v, act in actions.items():
            assert is_valid_action(act, state[v])
        from im_lab.simulator import simulate_step

        p_plus, q = true_params_from_graph(G)
        state, _obs = simulate_step(G, state, p_plus, q, 0.2, actions, rng)
