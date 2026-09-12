import numpy as np

from im_lab import graphs
from im_lab.actions import Action
from im_lab.simulator import (
    count_active,
    initial_state,
    seed_then_none_actions,
    simulate,
    true_params_from_graph,
)


def test_progressive_ic_never_deactivates_when_beta_zero_and_q_zero():
    """With beta=0 (no backfire) and q=0 (no spontaneous recovery), and only NONE
    actions after seeding, once a node is active it must stay active forever --
    this is exactly plain progressive IC."""
    G = graphs.erdos_renyi_graph(20, 0.2, num_groups=2, seed=7)
    graphs.assign_true_parameters(G, q_range=(0.0, 0.0), seed=7)
    p_plus, q = true_params_from_graph(G)

    seeds = [0, 1]
    T = 6
    init = initial_state(G, seeds)
    actions_seq = seed_then_none_actions(G, seeds, T)
    rng = np.random.default_rng(0)

    trajectory, _obs = simulate(G, init, p_plus, q, beta=0.0, actions_sequence=actions_seq, rng=rng)

    active_counts = [count_active(s) for s in trajectory]
    # Monotone non-decreasing under progressive IC.
    for a, b in zip(active_counts, active_counts[1:]):
        assert b >= a


def test_backfire_and_recovery_can_deactivate_nodes():
    """With q>0 and beta>0, some previously-active nodes should eventually go
    inactive without MAINTAIN actions -- i.e. the process is genuinely
    non-progressive, unlike plain IC."""
    G = graphs.erdos_renyi_graph(25, 0.25, num_groups=2, seed=11)
    graphs.assign_true_parameters(G, p_plus_range=(0.2, 0.5), q_range=(0.3, 0.6), seed=11)
    p_plus, q = true_params_from_graph(G)

    seeds = list(range(10))
    T = 8
    init = initial_state(G, seeds)
    actions_seq = seed_then_none_actions(G, seeds, T)
    rng = np.random.default_rng(1)

    trajectory, _obs = simulate(G, init, p_plus, q, beta=0.5, actions_sequence=actions_seq, rng=rng)

    active_counts = [count_active(s) for s in trajectory]
    # At least one round should show a strict decrease vs. the previous round.
    assert any(b < a for a, b in zip(active_counts, active_counts[1:]))


def test_convert_action_forces_activation_with_no_observation():
    G = graphs.erdos_renyi_graph(5, 0.0, num_groups=1, seed=2)  # no edges
    graphs.assign_true_parameters(G, seed=2)
    p_plus, q = true_params_from_graph(G)
    init = initial_state(G, [])
    actions_seq = [{v: Action.CONVERT for v in G.nodes()}]
    rng = np.random.default_rng(3)

    trajectory, obs = simulate(G, init, p_plus, q, beta=0.0, actions_sequence=actions_seq, rng=rng)
    assert all(trajectory[1].values())
    # No trials were drawn for forced-active nodes.
    assert obs[0] == []


def test_maintain_action_forces_no_deactivation():
    G = graphs.erdos_renyi_graph(5, 0.0, num_groups=1, seed=5)
    graphs.assign_true_parameters(G, q_range=(1.0, 1.0), seed=5)  # certain recovery if not maintained
    p_plus, q = true_params_from_graph(G)
    init = initial_state(G, list(G.nodes()))
    actions_seq = [{v: Action.MAINTAIN for v in G.nodes()}]
    rng = np.random.default_rng(6)

    trajectory, obs = simulate(G, init, p_plus, q, beta=0.0, actions_sequence=actions_seq, rng=rng)
    assert all(trajectory[1].values())
    assert obs[0] == []
