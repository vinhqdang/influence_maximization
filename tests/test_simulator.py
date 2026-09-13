import networkx as nx
import numpy as np

from im_lab import graphs
from im_lab.actions import Action
from im_lab.simulator import (
    count_active,
    initial_state,
    seed_then_none_actions,
    simulate,
    simulate_step,
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


# ---------------------------------------------------------------------------
# Layered semantics (docs/theory.md Section 0): positive trials are drawn into
# active targets too, and a success rescues the target from recovery/backfire.
# ---------------------------------------------------------------------------


def _two_node_graph(p_plus_uv: float, q_v: float, q_u: float = 0.0):
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1)
    for v in G.nodes():
        G.nodes[v]["group"] = 0
    G.edges[0, 1]["p_plus"] = p_plus_uv
    G.nodes[0]["q"] = q_u
    G.nodes[1]["q"] = q_v
    return G


def test_positive_trial_rescues_active_node_from_certain_recovery():
    """u -> v with p_plus = 1, q[v] = 1, beta = 0, both active, no actions.

    Old rule (positive trials only into inactive targets): v recovers with
    certainty and is inactive next round.
    Layered rule: u's certain positive trial into v fires in the same round and
    rescues v, so v must be ACTIVE next round. The rescue must also be visible
    to the tracker as a recorded p_plus observation on (u, v).
    """
    G = _two_node_graph(p_plus_uv=1.0, q_v=1.0)
    p_plus, q = true_params_from_graph(G)
    state = initial_state(G, [0, 1])
    actions = {0: Action.NONE, 1: Action.NONE}

    for seed in range(5):
        rng = np.random.default_rng(seed)
        new_state, obs = simulate_step(G, state, p_plus, q, beta=0.0, actions=actions, rng=rng)
        assert new_state[1] is True
        # Recovery was drawn (and succeeded), yet v is active: that is the rescue.
        assert any(o["param"] == "q" and o["v"] == 1 and o["success"] for o in obs)
        assert any(
            o["param"] == "p_plus" and o["u"] == 0 and o["v"] == 1 and o["success"] for o in obs
        )


def test_no_rescue_without_a_successful_positive_trial():
    """Complement of the rescue test: with p_plus = 0 the positive trial can
    never fire, so certain recovery still deactivates v (the rescue clause
    only ever *adds* activations, it never suppresses a legitimate recovery)."""
    G = _two_node_graph(p_plus_uv=0.0, q_v=1.0)
    p_plus, q = true_params_from_graph(G)
    state = initial_state(G, [0, 1])
    actions = {0: Action.NONE, 1: Action.NONE}
    rng = np.random.default_rng(0)
    new_state, _obs = simulate_step(G, state, p_plus, q, beta=0.0, actions=actions, rng=rng)
    assert new_state[1] is False


def test_maintain_still_draws_no_trials_and_positive_trials_into_active_are_observed():
    """Observation bookkeeping under the layered rule: an un-MAINTAINed active
    node with an active in-neighbour yields exactly one q, one p_plus and one
    p_minus observation; a MAINTAINed node yields none."""
    G = _two_node_graph(p_plus_uv=0.5, q_v=0.5)
    p_plus, q = true_params_from_graph(G)
    state = initial_state(G, [0, 1])
    rng = np.random.default_rng(0)

    _ns, obs = simulate_step(
        G, state, p_plus, q, beta=0.3, actions={0: Action.NONE, 1: Action.NONE}, rng=rng
    )
    params_for_v = sorted(o["param"] for o in obs if o["v"] == 1)
    assert params_for_v == ["p_minus", "p_plus", "q"]

    _ns, obs = simulate_step(
        G, state, p_plus, q, beta=0.3, actions={0: Action.NONE, 1: Action.MAINTAIN}, rng=rng
    )
    assert [o for o in obs if o["v"] == 1] == []


def _all_trial_keys(G):
    keys = []
    for v in G.nodes():
        keys.append(("q", v))
        for u in G.predecessors(v):
            keys.append(("p_plus", u, v))
            keys.append(("p_minus", u, v))
    return keys


def test_one_step_map_is_monotone_under_shared_randomness_when_beta_zero():
    """The property Lemma 1.1 / Lemma 1.2 of docs/theory.md rest on: for beta = 0
    and FIXED randomness, A subset of A' implies step(A) subset of step(A').

    Approach: EXACT shared-randomness coupling, not monotonicity in
    expectation. The number and order of `rng.random()` calls in simulate_step
    depends on which nodes are active (trials are only drawn on edges out of
    active nodes), so seeding the rng identically for A and A' would NOT give
    the same draws to the same trials. Instead we use the `trial_draws` hook of
    simulate_step: one U[0,1) value is pre-drawn for every possible trial key
    ('q', v), ('p_plus', u, v), ('p_minus', u, v) and both runs read the uniform
    for a given trial from that shared table. Every trial that is drawn in both
    runs then has literally the same outcome, which is exactly the coupling
    used in Lemma 1.1 of docs/theory.md.

    Under the OLD simulator rule (no positive trials into active targets) this
    fails: if v in A recovers, v is inactive in step(A); but the same v with an
    extra active in-neighbour in A' would have been re-activated only if v had
    been inactive -- so the rescue clause is exactly what is being tested here.
    """
    G = graphs.erdos_renyi_graph(15, 0.3, num_groups=2, seed=21)
    graphs.assign_true_parameters(G, p_plus_range=(0.2, 0.9), q_range=(0.1, 0.8), seed=21)
    p_plus, q = true_params_from_graph(G)
    nodes = list(G.nodes())
    actions = {v: Action.NONE for v in nodes}
    keys = _all_trial_keys(G)

    rng = np.random.default_rng(2024)
    n_checks = 0
    n_strict = 0
    for _ in range(150):
        draws = {k: float(rng.random()) for k in keys}
        A = {v for v in nodes if rng.random() < 0.4}
        extra = {v for v in nodes if rng.random() < 0.4}
        A_sup = A | extra

        s_A = initial_state(G, A)
        s_sup = initial_state(G, A_sup)
        # The rng passed in is a fresh, unrelated generator: with a complete
        # trial_draws table it must never actually be consulted.
        new_A, _ = simulate_step(
            G, s_A, p_plus, q, 0.0, actions, np.random.default_rng(1), trial_draws=draws
        )
        new_sup, _ = simulate_step(
            G, s_sup, p_plus, q, 0.0, actions, np.random.default_rng(2), trial_draws=draws
        )
        act_A = {v for v in nodes if new_A[v]}
        act_sup = {v for v in nodes if new_sup[v]}
        assert act_A <= act_sup, (A, A_sup, act_A - act_sup)
        n_checks += 1
        n_strict += act_A < act_sup
    assert n_checks == 150
    # Sanity: the check is not vacuous (supersets frequently reach strictly more).
    assert n_strict > 50


def test_one_step_map_is_not_monotone_when_beta_positive_deterministic_witness():
    """Boundary of the theory, made explicit: with beta > 0 the one-step map is
    NOT monotone in A_t even for fixed randomness -- an extra active
    in-neighbour adds a backfire trial that can remove v. This is the
    neighbour-dependent deactivation that docs/theory.md Section 1 (Remark)
    identifies as the thing that destroys the coverage structure, and why the
    beta > 0 guarantees go through the sandwich coupling (Section 2) instead.

    Deterministic witness on u -> v: X_uv fails, Y_uv fires, v does not recover.
    A = {v}: v simply stays.  A' = {u, v}: v is backfired.  So step(A) is not a
    subset of step(A').
    """
    G = _two_node_graph(p_plus_uv=0.5, q_v=0.5)
    p_plus, q = true_params_from_graph(G)
    actions = {0: Action.NONE, 1: Action.NONE}
    beta = 0.5
    draws = {
        ("q", 0): 0.99,  # u does not recover
        ("q", 1): 0.99,  # v does not recover
        ("p_plus", 0, 1): 0.99,  # positive trial fails (0.99 >= 0.5)
        ("p_minus", 0, 1): 0.0,  # backfire fires (0.0 < beta * 0.5)
    }
    rng = np.random.default_rng(0)
    new_A, _ = simulate_step(G, initial_state(G, [1]), p_plus, q, beta, actions, rng, trial_draws=draws)
    new_sup, _ = simulate_step(
        G, initial_state(G, [0, 1]), p_plus, q, beta, actions, rng, trial_draws=draws
    )
    assert new_A[1] is True
    assert new_sup[1] is False
    # ... and the same witness with beta = 0 is monotone again (v stays in both).
    new_sup0, _ = simulate_step(
        G, initial_state(G, [0, 1]), p_plus, q, 0.0, actions, rng, trial_draws=draws
    )
    assert new_sup0[1] is True


def test_trial_draws_hook_reproduces_default_rng_path():
    """The coupling hook must not change the dynamics: feeding simulate_step the
    same uniforms the default path would draw from the rng (in its
    node/predecessor iteration order) reproduces the default result exactly."""
    G = graphs.erdos_renyi_graph(12, 0.3, num_groups=2, seed=8)
    graphs.assign_true_parameters(G, seed=8)
    p_plus, q = true_params_from_graph(G)
    state = initial_state(G, [0, 1, 2, 3])
    actions = {v: Action.NONE for v in G.nodes()}

    ref_state, ref_obs = simulate_step(G, state, p_plus, q, 0.3, actions, np.random.default_rng(99))

    # Replay the default draw order into a trial_draws table.
    rng = np.random.default_rng(99)
    draws = {}
    for v in G.nodes():
        if not state[v]:
            for u in G.predecessors(v):
                if state[u]:
                    draws[("p_plus", u, v)] = rng.random()
        else:
            draws[("q", v)] = rng.random()
            for u in G.predecessors(v):
                if state[u]:
                    draws[("p_plus", u, v)] = rng.random()
                    draws[("p_minus", u, v)] = rng.random()

    cpl_state, cpl_obs = simulate_step(
        G, state, p_plus, q, 0.3, actions, np.random.default_rng(0), trial_draws=draws
    )
    assert cpl_state == ref_state
    assert cpl_obs == ref_obs
