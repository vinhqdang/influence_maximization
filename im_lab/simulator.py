"""Discrete-round diffusion simulator for the generalized (non-progressive,
backfire-capable) independent-cascade model, under the *layered semantics* of
docs/theory.md Section 0.

Per round, independent randomness:
  for each edge (u,v) with u active:  positive trial   X_uv ~ Bern(p_plus[u,v])
                                      backfire trial   Y_uv ~ Bern(beta * p_plus[u,v])
  for each active node v:             recovery trial   R_v  ~ Bern(q[v])

Transition (per node v):

  v in A_{t+1}  iff  [ v in A_t  and  R_v = 0  and  no active in-neighbour u has Y_uv = 1 ]
                 or  [ some active in-neighbour u has X_uv = 1 ],

subject to the forced actions:
  CONVERT  on inactive v: v becomes active, no trials drawn / observed for v.
  MAINTAIN on active v:   v stays active,   no trials drawn / observed for v.

Consequences:
  inactive v: activates w.p. 1 - prod_{u active in-nbr}(1 - p_plus[u,v])  (classical IC).
  active v:   would naturally deactivate w.p. 1 - (1-q[v]) * prod_{u}(1 - beta*p_plus[u,v]),
              but is RESCUED (kept active) if any active in-neighbour's positive trial
              succeeds this round. This rescue clause is exactly what makes the one-step
              map A_t -> A_{t+1} monotone in A_t for fixed randomness, which is the
              property every theorem in docs/theory.md rests on (Lemma 1.1: layered
              live-edge representation; Lemma 1.2: submodularity for beta = 0).

Both formulas are realized as independent per-edge (and, for q, per-node) Bernoulli
trials rather than as a single aggregate coin flip, so that each trial can be
reported back as a distinct Beta-Bernoulli observation for the Bayesian updater in
bayes.py -- this is exactly the classical IC "each edge gets one independent
attempt" semantics, extended with a symmetric backfire/recovery side for
deactivation.

Observability caveat (identifiability, for the paper): in a real deployment a
positive trial into an *already-active* node is generally not observable (the
target is active either way, so success/failure of X_uv leaves no trace). In
simulation the outcome is nevertheless generated, and we follow the existing
convention that the Bayesian tracker sees every drawn trial, including these.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .actions import Action


def true_params_from_graph(G: nx.DiGraph) -> tuple[dict, dict]:
    """Extract the hidden ground-truth p_plus (dict[(u,v)]) and q (dict[v]) used by
    the simulator, from the attributes set by graphs.assign_true_parameters."""
    p_plus = {(u, v): G.edges[u, v]["p_plus"] for u, v in G.edges()}
    q = {v: G.nodes[v]["q"] for v in G.nodes()}
    return p_plus, q


def initial_state(G: nx.DiGraph, active_nodes=()) -> dict:
    active_set = set(active_nodes)
    return {v: (v in active_set) for v in G.nodes()}


def count_active(state: dict) -> int:
    return sum(1 for a in state.values() if a)


def _uniform(rng: np.random.Generator, trial_draws, key):
    """One U[0,1) draw for the trial identified by `key`.

    By default this is `rng.random()`. If `trial_draws` (a dict keyed by
    ('p_plus', u, v) / ('p_minus', u, v) / ('q', v)) is given and contains `key`,
    that value is used instead. This is a coupling hook: it lets callers run
    simulate_step from two different states under literally the same
    per-trial randomness (the number and order of rng draws otherwise depends on
    which nodes are active, so seeding the rng identically is not enough).
    Used in tests to check the monotonicity of the one-step map.
    """
    if trial_draws is not None and key in trial_draws:
        return float(trial_draws[key])
    return float(rng.random())


def simulate_step(
    G: nx.DiGraph,
    state: dict,
    p_plus: dict,
    q: dict,
    beta: float,
    actions: dict,
    rng: np.random.Generator,
    *,
    trial_draws: dict | None = None,
) -> tuple[dict, list]:
    """Advance the diffusion by exactly one round (layered semantics, see module
    docstring).

    Returns (new_state, observations) where observations is a list of dicts
    {'param': 'p_plus'|'p_minus'|'q', 'u': node_or_None, 'v': node, 'success': bool},
    one entry per stochastic trial actually drawn this round (forced nodes under
    CONVERT/MAINTAIN contribute no observations, since no trial was drawn for them).

    For an active, un-MAINTAINed node v, positive trials from its active
    in-neighbours are drawn (and recorded as 'p_plus' observations) in addition to
    the recovery and backfire trials; a successful positive trial rescues v from
    a recovery/backfire deactivation in the same round. Note that such a trial
    is generally not observable in a real deployment (see module docstring).

    `trial_draws` is an optional coupling hook (see _uniform); leave it as None
    for ordinary simulation, in which case rng consumption is exactly one
    `rng.random()` call per trial in node/predecessor iteration order.
    """
    new_state: dict = {}
    observations: list = []

    for v in G.nodes():
        act = actions.get(v, Action.NONE)
        active = state[v]

        if not active:
            if act == Action.CONVERT:
                new_state[v] = True
                continue
            pos_success = False
            for u in G.predecessors(v):
                if not state[u]:
                    continue
                p = p_plus[(u, v)]
                success = bool(_uniform(rng, trial_draws, ("p_plus", u, v)) < p)
                observations.append({"param": "p_plus", "u": u, "v": v, "success": success})
                if success:
                    pos_success = True
            new_state[v] = pos_success
        else:
            if act == Action.MAINTAIN:
                new_state[v] = True
                continue
            recovered = bool(_uniform(rng, trial_draws, ("q", v)) < q[v])
            observations.append({"param": "q", "u": None, "v": v, "success": recovered})
            deactivated_by_natural = recovered
            pos_success = False
            for u in G.predecessors(v):
                if not state[u]:
                    continue
                p = p_plus[(u, v)]
                # Positive trial: also drawn into an active target (layered
                # semantics); a success rescues v from deactivation this round.
                x = bool(_uniform(rng, trial_draws, ("p_plus", u, v)) < p)
                observations.append({"param": "p_plus", "u": u, "v": v, "success": x})
                if x:
                    pos_success = True
                # Backfire trial.
                y = bool(_uniform(rng, trial_draws, ("p_minus", u, v)) < beta * p)
                observations.append({"param": "p_minus", "u": u, "v": v, "success": y})
                if y:
                    deactivated_by_natural = True
            new_state[v] = (not deactivated_by_natural) or pos_success

    return new_state, observations


def simulate(
    G: nx.DiGraph,
    init_state: dict,
    p_plus: dict,
    q: dict,
    beta: float,
    actions_sequence: list,
    rng: np.random.Generator,
) -> tuple[list, list]:
    """Run T = len(actions_sequence) rounds from init_state.

    Returns (trajectory, observations_per_round):
      trajectory: list of T+1 state dicts, trajectory[0] == init_state.
      observations_per_round: list of T observation lists (see simulate_step).
    """
    state = dict(init_state)
    trajectory = [state]
    observations_per_round = []
    for actions in actions_sequence:
        state, obs = simulate_step(G, state, p_plus, q, beta, actions, rng)
        trajectory.append(state)
        observations_per_round.append(obs)
    return trajectory, observations_per_round


def seed_then_none_actions(G: nx.DiGraph, seed_set, T: int) -> list:
    """Build the "CONVERT the seeds at round 0, then NONE forever after" action
    sequence used to run one-shot classical baselines inside this sequential
    simulator (see baselines/kkt_greedy.py and baselines/fair_greedy.py)."""
    seed_set = set(seed_set)
    actions_sequence = []
    round0 = {v: (Action.CONVERT if v in seed_set else Action.NONE) for v in G.nodes()}
    actions_sequence.append(round0)
    for _ in range(1, T):
        actions_sequence.append({v: Action.NONE for v in G.nodes()})
    return actions_sequence
