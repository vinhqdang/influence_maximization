"""MF-BWI-Fair: Mean-Field Belief / Whittle-style Index policy under fairness floors.

Pipeline per round:
  1. Read current posterior-mean estimates of p_plus, p_minus, q from the Bayesian
     tracker (the algorithm never sees the simulator's hidden true parameters).
  2. Compute mean-field beliefs m_v (an approximate marginal P(active) for every
     node) via the fixed-point iteration described below.
  3. Compute a per-node myopic index for its one meaningful action (CONVERT if
     currently inactive, MAINTAIN if currently active).
  4. Allocate the round's budget across nodes by descending index, respecting
     per-group fairness floors first (fairness.allocate_with_fairness).
  5. After the simulator reports this round's realized transitions, update the
     Bayesian tracker (conjugate Beta-Bernoulli).

--- Mean-field belief fixed point ---
Exact per-node marginals require tracking correlations across the whole active-set
distribution, which is intractable at scale. We instead use the standard N-intertwined
/ individual-based mean-field approximation from network epidemic models (e.g. the
SIS mean-field of Van Mieghem et al.): each node's neighbors are replaced by their
*marginal* activation probabilities m_u, decoupling the joint distribution into a
product of independent Bernoullis. The steady-state self-consistency equation this
implies is:

  m_v = m_v * (1-q_v) * prod_u (1 - p_minus[u,v] * m_u)
        + (1-m_v) * (1 - prod_u (1 - p_plus[u,v] * m_u))

which we solve by fixed-point (Jacobi) iteration from an initialization at the
node's current observed state, to tolerance 1e-4 or 50 iterations, whichever first.
This gives each node a smoothed, network-aware estimate of "how likely am I to be
active going forward" that is less noisy than its single instantaneous 0/1 state.

--- Myopic index (NOT a proven Whittle index) ---
A fully general Whittle index for this non-progressive, non-monotone, fairness- and
budget-constrained restless-bandit-like problem would require proving indexability
(that the set of states where each per-node subproblem prefers "act" over "don't
act" is monotone in a Lagrange multiplier on the budget) -- a substantial separate
theoretical exercise that is out of scope here. Instead we implement a clearly
labeled one-step-lookahead / myopic index:

  index_CONVERT(v) = (1 - p_natural_activate(v)) * (1 + h(v)) * d(v) / cost(CONVERT)
  index_MAINTAIN(v) = p_natural_deactivate(v) * (1 + h(v)) * d(v) / cost(MAINTAIN)

where p_natural_activate/deactivate(v) is v's one-step transition probability under
the CURRENT actual neighbor states (using posterior-mean parameters) if no action is
taken -- i.e. exactly the immediate-round gain that forcing the action would buy --
and h(v) = 1 - |2*m_v - 1| in [0,1] is a mean-field "instability weight" that upweights
nodes whose long-run mean-field belief sits near 0.5 (where the node's fate is still
genuinely undecided and an action has lasting leverage) relative to nodes whose
mean-field belief already sits near 0 or 1 (where the node is mean-field-stable and
a one-round push will just wash out). This folds the fixed-point beliefs into the
index as a horizon/leverage correction without claiming a proven multi-step-optimal
index.

d(v) = 1 + out_degree(v) / max(1, max_out_degree) is a cheap structural tie-breaker.
It matters for exactly one degenerate case: starting from a fully-inactive graph,
every never-yet-active node with no active in-neighbors has p_natural_activate(v)=0
and mean-field belief m_v=0 identically (mean field with no active seed anywhere is
a trivial all-zero fixed point), so gain and h(v) alone cannot distinguish ANY two
candidate nodes at round 0 -- a fundamental limitation of any strictly "node's own
trajectory" one-step index, since it deliberately ignores the value a node's
activation has for its neighbors. Weighting by out-degree is the classical
"high-degree heuristic" for IM (a standard, much weaker cousin of greedy noted
already in Kempe-Kleinberg-Tardos 2003): it lets the index prefer structurally
better-connected nodes without computing a real multi-hop spread estimate, which
keeps this an O(1)-per-node, one-step-lookahead index rather than smuggling in a
CELF-style expected-cascade computation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .actions import Action, ACTION_COST
from .bayes import BetaBernoulliTracker
from .fairness import allocate_with_fairness
from .graphs import group_of_map, group_sizes as graph_group_sizes
from .simulator import simulate_step, true_params_from_graph


def mean_field_beliefs(
    G: nx.DiGraph,
    state: dict,
    p_plus_hat: dict,
    p_minus_hat: dict,
    q_hat: dict,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> dict:
    """Fixed-point mean-field marginals m_v, initialized at the current state."""
    m = {v: float(state[v]) for v in G.nodes()}
    preds = {v: list(G.predecessors(v)) for v in G.nodes()}

    for _ in range(max_iter):
        max_diff = 0.0
        new_m = {}
        for v in G.nodes():
            act_prod = 1.0
            deact_prod = 1.0
            for u in preds[v]:
                mu = m[u]
                act_prod *= 1.0 - p_plus_hat[(u, v)] * mu
                deact_prod *= 1.0 - p_minus_hat[(u, v)] * mu
            p_activate = 1.0 - act_prod
            p_stay_active = (1.0 - q_hat[v]) * deact_prod
            mv = m[v]
            new_mv = mv * p_stay_active + (1.0 - mv) * p_activate
            new_m[v] = new_mv
            max_diff = max(max_diff, abs(new_mv - mv))
        m = new_m
        if max_diff < tol:
            break
    return m


class MFBWIFair:
    """Stateful policy + Bayesian tracker for the MF-BWI-Fair algorithm."""

    def __init__(
        self,
        G: nx.DiGraph,
        budget: float,
        alpha_fair: float,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        mf_tol: float = 1e-4,
        mf_max_iter: int = 50,
    ):
        self.G = G
        self.budget = budget
        self.alpha_fair = alpha_fair
        self.mf_tol = mf_tol
        self.mf_max_iter = mf_max_iter

        self.group_of = group_of_map(G)
        self.group_sizes = graph_group_sizes(G)
        self.n = G.number_of_nodes()
        self.out_degree = dict(G.out_degree())
        self.max_out_degree = max(1, max(self.out_degree.values(), default=1))

        self.tracker = BetaBernoulliTracker(alpha0=alpha0, beta0=beta0)
        # Seed the tracker so every edge/node has a posterior even before any trial
        # is observed on it (mean() would lazily default anyway, but this makes the
        # estimate dicts below simple, total dict comprehensions).
        for u, v in G.edges():
            self.tracker._ensure(("p_plus", u, v))
            self.tracker._ensure(("p_minus", u, v))
        for v in G.nodes():
            self.tracker._ensure(("q", v))

        # Budget/fairness bookkeeping for the runtime invariant checks.
        self.budget_usage_log: list = []
        self.group_spend_log: list = []

    def posterior_means(self) -> tuple[dict, dict, dict]:
        p_plus_hat = {(u, v): self.tracker.mean(("p_plus", u, v)) for u, v in self.G.edges()}
        p_minus_hat = {(u, v): self.tracker.mean(("p_minus", u, v)) for u, v in self.G.edges()}
        q_hat = {v: self.tracker.mean(("q", v)) for v in self.G.nodes()}
        return p_plus_hat, p_minus_hat, q_hat

    def _natural_transition_probs(self, state: dict, p_plus_hat: dict, p_minus_hat: dict, q_hat: dict):
        """One-step natural (no-action) activate/deactivate probabilities per node,
        using ACTUAL current neighbor states (not mean-field), since these are known
        exactly and give the correct immediate-round gain from acting."""
        p_activate = {}
        p_deactivate = {}
        for v in self.G.nodes():
            preds_active = [u for u in self.G.predecessors(v) if state[u]]
            if not state[v]:
                prod = 1.0
                for u in preds_active:
                    prod *= 1.0 - p_plus_hat[(u, v)]
                p_activate[v] = 1.0 - prod
            else:
                prod = 1.0
                for u in preds_active:
                    prod *= 1.0 - p_minus_hat[(u, v)]
                p_deactivate[v] = 1.0 - (1.0 - q_hat[v]) * prod
        return p_activate, p_deactivate

    def choose_actions(self, state: dict, rng: np.random.Generator = None) -> dict:
        p_plus_hat, p_minus_hat, q_hat = self.posterior_means()
        m = mean_field_beliefs(
            self.G, state, p_plus_hat, p_minus_hat, q_hat, self.mf_tol, self.mf_max_iter
        )
        p_activate, p_deactivate = self._natural_transition_probs(
            state, p_plus_hat, p_minus_hat, q_hat
        )

        candidates = []
        for v in self.G.nodes():
            h = 1.0 - abs(2.0 * m[v] - 1.0)
            d = 1.0 + self.out_degree[v] / self.max_out_degree
            if not state[v]:
                gain = 1.0 - p_activate[v]
                cost = ACTION_COST[Action.CONVERT]
                index = gain * (1.0 + h) * d / cost
                candidates.append((v, Action.CONVERT, cost, index))
            else:
                gain = p_deactivate[v]
                cost = ACTION_COST[Action.MAINTAIN]
                index = gain * (1.0 + h) * d / cost
                candidates.append((v, Action.MAINTAIN, cost, index))

        selected, spend_by_group = allocate_with_fairness(
            candidates, self.group_of, self.group_sizes, self.n, self.budget, self.alpha_fair
        )

        total_spend = sum(cost for _, cost in selected.values())
        assert total_spend <= self.budget + 1e-9, "MF-BWI-Fair budget invariant violated"
        self.budget_usage_log.append(total_spend)
        self.group_spend_log.append(dict(spend_by_group))

        actions = {v: Action.NONE for v in self.G.nodes()}
        for v, (action, _cost) in selected.items():
            actions[v] = action
        return actions

    def observe(self, observations: list) -> None:
        """Beta-Bernoulli posterior update from one round's realized transitions."""
        for obs in observations:
            if obs["param"] == "q":
                key = ("q", obs["v"])
            else:
                key = (obs["param"], obs["u"], obs["v"])
            self.tracker.update(key, obs["success"])

    def floor_satisfied(self, round_idx: int) -> dict:
        """For diagnostics/tests: per-group bool of whether that round's spend met
        (or exceeded) its fairness floor."""
        from .fairness import group_floors

        floors = group_floors(self.group_sizes, self.n, self.budget, self.alpha_fair)
        spend = self.group_spend_log[round_idx]
        return {g: spend.get(g, 0.0) >= floors[g] - 1e-9 for g in self.group_sizes}


def run_mf_bwi_fair(
    G: nx.DiGraph,
    true_beta: float,
    T: int,
    budget: float,
    alpha_fair: float,
    init_active=(),
    seed=None,
) -> dict:
    """Full end-to-end run: policy + simulator + Bayesian updates, T rounds.

    true_beta and the graph's own 'p_plus'/'q' attributes (via
    simulator.true_params_from_graph) define the hidden ground truth used by the
    simulator; the policy only ever sees posterior means derived from observations.
    """
    rng = np.random.default_rng(seed)
    p_plus_true, q_true = true_params_from_graph(G)

    policy = MFBWIFair(G, budget=budget, alpha_fair=alpha_fair)
    state = {v: (v in set(init_active)) for v in G.nodes()}

    trajectory = [state]
    actions_history = []
    for _t in range(T):
        actions = policy.choose_actions(state, rng)
        new_state, observations = simulate_step(G, state, p_plus_true, q_true, true_beta, actions, rng)
        policy.observe(observations)

        trajectory.append(new_state)
        actions_history.append(actions)
        state = new_state

    return {
        "trajectory": trajectory,
        "actions_history": actions_history,
        "budget_usage_log": policy.budget_usage_log,
        "group_spend_log": policy.group_spend_log,
        "policy": policy,
    }
