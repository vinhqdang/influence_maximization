"""MF-BWI-Fair: Mean-Field Belief policy with a Lagrangian multi-action index and
isoelastic-welfare fairness reweighting.

Pipeline per round:
  1. Read current posterior-mean estimates of p_plus, p_minus, q from the Bayesian
     tracker (the algorithm never sees the simulator's hidden true parameters).
  2. Compute mean-field beliefs m_v (an approximate marginal P(active) for every
     node) via the fixed-point iteration described below -- UNCHANGED from the
     original design.
  3. Compute each group's current marginal welfare weight w_g from its tracked
     time-averaged realized reach u_g (im_lab/fairness.group_welfare_weights).
  4. Build each node's per-round 2-state (inactive=0/active=1) Lagrangian MDP using
     the mean-field belief vector as the frozen exogenous "field" for its
     neighbors' states, with reward r_v(s) = w_g * s, and solve the shared-budget
     Lagrangian relaxation (im_lab/lagrangian_index.py) via bisection on lambda.
  5. Fill any leftover budget (from the bisection's discreteness) by a greedy
     tie-break on each node's action-value gap at lambda*.
  6. Execute actions, observe transitions (unchanged), Bayesian-update (unchanged),
     and update each group's tracked u_g for next round's w_g computation.

--- Mean-field belief fixed point (unchanged) ---
Exact per-node marginals require tracking correlations across the whole active-set
distribution, which is intractable at scale. We use the standard N-intertwined /
individual-based mean-field approximation from network epidemic models (e.g. the
SIS mean-field of Van Mieghem et al.): each node's neighbors are replaced by their
*marginal* activation probabilities m_u, decoupling the joint distribution into a
product of independent Bernoullis. The steady-state self-consistency equation this
implies is:

  m_v = m_v * (1-q_v) * prod_u (1 - p_minus[u,v] * m_u)
        + (1-m_v) * (1 - prod_u (1 - p_plus[u,v] * m_u))

which we solve by fixed-point (Jacobi) iteration from an initialization at the
node's current observed state, to tolerance 1e-4 or 50 iterations, whichever first.

We separately confirmed (via a full read of Ou et al., AAMAS 2022, "Networked
Restless Multi-Armed Bandits for Mobile Interventions") that their alternative
concavity-based method for handling network coupling only applies to a
fundamentally different coupling type (deterministic action-vector coupling, e.g. a
commuting matrix) and does NOT apply to our bilinear neighbor-STATE coupling (an
active neighbor's actual random state affects another node's transition
probability). So this mean-field fixed point remains the right tool here and is
kept exactly as before; only what CONSUMES it (the index computation) changes.

--- Lagrangian multi-action index (replaces the old myopic/out-degree heuristic) ---
The previous index was an explicitly-labeled "NOT a proven Whittle index" one-step
myopic heuristic with an ad hoc out-degree tie-breaker. We replace it with the
Lagrangian-relaxation template of:

    Killian, Perrault, Tambe, "Beyond 'To Act or Not to Act': Fast Lagrangian
    Approaches to General Multi-Action Restless Bandits," AAMAS 2021.

which bypasses proving indexability for our 3-action (NONE/CONVERT/MAINTAIN)
setting (Killian et al. note this is "notoriously difficult" for M > 2 actions and
do not attempt it either) by instead finding the smallest shared Lagrange
multiplier lambda whose aggregate expected cost across all nodes is <= the round's
budget B. Originally this was done by bisection on lambda with value iteration per
node (im_lab/lagrangian_index.py, still available via solver="bisection"). Because
each node's MDP has only 2 states and 2 actions per state, its Lagrangian value is
a max of 4 affine functions of lambda, so the per-node switch price (index) and the
budget-matching lambda are available in closed form; im_lab/closed_form_index.py
does exactly that and is the default (solver="closed_form"). See that module's
docstring for the closed forms, what is proved (convexity in lambda, exactness of
the index scan) and what is only checked numerically (indexability, i.e. that
the paid-action region in lambda is a down-set). This module only wires the network
model (mean-field beliefs, posterior means, group welfare weights) into it.

--- Fairness (replaces the old per-group budget-floor heuristic) ---
The previous two-phase "meet each group's floor first" allocator was empirically
shown, in a prior experiment, to have only a weak effect. We replace it with the
welfare-OBJECTIVE-FORM reweighting of:

    Rahmattalabi et al., "Fair Influence Maximization: A Welfare Optimization
    Approach," AAAI 2021 (isoelastic/CES welfare over per-group reach).

See im_lab/fairness.py for the full derivation of the per-group marginal weight
w_g and the precise (re-documented) meaning of alpha_fair under this new mechanism.
There are no per-group floors or quotas anywhere in this design: fairness comes
entirely from reweighting each node's reward by its group's current marginal
welfare weight, which then competes on equal footing with every other node inside
the single shared-budget Lagrangian solve.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from . import closed_form_index, lagrangian_index
from .actions import Action, ACTION_COST
from .bayes import BetaBernoulliTracker
from .fairness import group_welfare_weights
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


def field_transition_probs(
    G: nx.DiGraph,
    m: dict,
    p_plus_hat: dict,
    p_minus_hat: dict,
    q_hat: dict,
) -> tuple[dict, dict]:
    """Per-node (p01, p10) "no-action" transition probabilities with the mean-field
    belief vector m plugged in as the frozen exogenous field for every neighbor --
    i.e. exactly the same formulas used inside mean_field_beliefs' fixed-point
    update, evaluated once at the CONVERGED m, treating m as consistent with the
    existing mean-field fixed-point framing (rather than the actual, un-smoothed
    current neighbor states). These are the transition probabilities fed into each
    node's Lagrangian MDP (im_lab/lagrangian_index.py) for BOTH branches (s=0 and
    s=1), since the MDP's value function needs the whole one-step kernel, not just
    the current-state gain.

      p01(v) = 1 - prod_u (1 - p_plus_hat[u,v] * m_u)         (activate, from s=0)
      p10(v) = 1 - (1-q_hat[v]) * prod_u (1 - p_minus_hat[u,v] * m_u)  (deactivate, from s=1)
    """
    p01: dict = {}
    p10: dict = {}
    for v in G.nodes():
        act_prod = 1.0
        deact_prod = 1.0
        for u in G.predecessors(v):
            mu = m[u]
            act_prod *= 1.0 - p_plus_hat[(u, v)] * mu
            deact_prod *= 1.0 - p_minus_hat[(u, v)] * mu
        p01[v] = 1.0 - act_prod
        p10[v] = 1.0 - (1.0 - q_hat[v]) * deact_prod
    return p01, p10


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
        gamma: float = 0.9,
        u_floor: float = 1e-3,
        n_bisect_iters: int = 40,
        n_vi_sweeps: int = 200,
        solver: str = "closed_form",
    ):
        if solver not in ("closed_form", "bisection"):
            raise ValueError(f"solver must be 'closed_form' or 'bisection', got {solver!r}")
        self.solver = solver
        self.G = G
        self.budget = budget
        # alpha_fair is now the isoelastic (CES) welfare exponent, NOT the old
        # per-group floor fraction. See im_lab/fairness.py's module docstring for
        # its full, redocumented meaning: 1 = utilitarian/size-only weighting,
        # 0 = proportional fairness, more negative = more leximin-like.
        self.alpha_fair = alpha_fair
        self.mf_tol = mf_tol
        self.mf_max_iter = mf_max_iter
        self.gamma = gamma
        self.u_floor = u_floor
        self.n_bisect_iters = n_bisect_iters
        self.n_vi_sweeps = n_vi_sweeps

        self.group_of = group_of_map(G)
        self.group_sizes = graph_group_sizes(G)
        self.n = G.number_of_nodes()
        self.nodes = list(G.nodes())
        self.nodes_by_group: dict = {g: [] for g in self.group_sizes}
        for v in self.nodes:
            self.nodes_by_group[self.group_of[v]].append(v)

        self.tracker = BetaBernoulliTracker(alpha0=alpha0, beta0=beta0)
        # Seed the tracker so every edge/node has a posterior even before any trial
        # is observed on it (mean() would lazily default anyway, but this makes the
        # estimate dicts below simple, total dict comprehensions).
        for u, v in G.edges():
            self.tracker._ensure(("p_plus", u, v))
            self.tracker._ensure(("p_minus", u, v))
        for v in G.nodes():
            self.tracker._ensure(("q", v))

        # Per-group time-averaged realized reach fraction u_g, used to compute this
        # round's welfare weight w_g (fairness.group_welfare_weights). Bootstrapped
        # from the actual state on the first choose_actions call, then updated after
        # every round from the realized post-transition state (update_group_reach).
        self.group_u_avg: dict = {}
        self.group_round_count: dict = {g: 0 for g in self.group_sizes}

        # Budget/diagnostics bookkeeping for the runtime invariant checks/tests.
        self.budget_usage_log: list = []
        self.group_spend_log: list = []
        self.lambda_star_log: list = []

    def posterior_means(self) -> tuple[dict, dict, dict]:
        p_plus_hat = {(u, v): self.tracker.mean(("p_plus", u, v)) for u, v in self.G.edges()}
        p_minus_hat = {(u, v): self.tracker.mean(("p_minus", u, v)) for u, v in self.G.edges()}
        q_hat = {v: self.tracker.mean(("q", v)) for v in self.G.nodes()}
        return p_plus_hat, p_minus_hat, q_hat

    def _group_active_fraction(self, state: dict) -> dict:
        frac = {}
        for g, nodes in self.nodes_by_group.items():
            if not nodes:
                frac[g] = 0.0
                continue
            frac[g] = sum(1 for v in nodes if state[v]) / len(nodes)
        return frac

    def update_group_reach(self, state: dict) -> None:
        """Update each group's time-averaged realized reach fraction u_g from a
        realized (actual, not mean-field) state -- called after each round's
        transition, so next round's welfare weight reflects it."""
        frac = self._group_active_fraction(state)
        for g in self.group_sizes:
            c = self.group_round_count.get(g, 0)
            avg = self.group_u_avg.get(g, frac[g])
            self.group_u_avg[g] = (avg * c + frac[g]) / (c + 1)
            self.group_round_count[g] = c + 1

    def choose_actions(self, state: dict, rng: np.random.Generator = None) -> dict:
        p_plus_hat, p_minus_hat, q_hat = self.posterior_means()
        m = mean_field_beliefs(
            self.G, state, p_plus_hat, p_minus_hat, q_hat, self.mf_tol, self.mf_max_iter
        )
        p01, p10 = field_transition_probs(self.G, m, p_plus_hat, p_minus_hat, q_hat)

        # Bootstrap u_g, on the very first round, from the actual current state
        # (there is no realized-reach history yet); every later round instead uses
        # the running average maintained by update_group_reach.
        if not self.group_u_avg:
            self.group_u_avg = self._group_active_fraction(state)
            self.group_round_count = {g: 1 for g in self.group_sizes}

        w = group_welfare_weights(self.group_sizes, self.group_u_avg, self.alpha_fair, self.u_floor)

        s_arr = np.array([1.0 if state[v] else 0.0 for v in self.nodes])
        p01_arr = np.array([p01[v] for v in self.nodes])
        p10_arr = np.array([p10[v] for v in self.nodes])
        w_arr = np.array([w[self.group_of[v]] for v in self.nodes])

        cost_convert = float(ACTION_COST[Action.CONVERT])
        cost_maintain = float(ACTION_COST[Action.MAINTAIN])
        if self.solver == "closed_form":
            lam_star, action_is_paid, cost_arr, gap_arr, cost_if_paid, index_arr = (
                closed_form_index.solve_lambda_closed_form(
                    s_arr, p01_arr, p10_arr, w_arr, self.budget, self.gamma,
                    cost_convert, cost_maintain, return_indices=True,
                )
            )
            # Fill priority: the closed-form index itself (the highest price at
            # which the node would still act). Among nodes the matching skipped it
            # is the natural ranking, and it is what the O(n log n) sort already
            # produced, so no extra Bellman evaluation is needed.
            fill_key = index_arr
        else:
            lam_star, action_is_paid, cost_arr, gap_arr, cost_if_paid = lagrangian_index.solve_lambda_bisection(
                s_arr, p01_arr, p10_arr, w_arr, self.budget, self.gamma,
                cost_convert, cost_maintain,
                n_bisect_iters=self.n_bisect_iters, n_vi_sweeps=self.n_vi_sweeps,
            )
            fill_key = gap_arr
        action_is_paid = action_is_paid.copy()
        total_spend = float(cost_arr.sum())

        # Tie-break fill (step 6 of the redesign): any leftover budget from the
        # matching's discreteness is filled greedily by descending fill_key
        # (closed-form index, or action-value gap at lambda* for the bisection
        # path), exactly like the "act or don't" tie-breaking Lagrangian index
        # methods use -- this keeps the hard budget invariant intact without
        # reintroducing any per-group floor/quota mechanism.
        leftover = self.budget - total_spend
        if leftover > 1e-9:
            not_paid_idx = np.where(~action_is_paid)[0]
            order = sorted(not_paid_idx.tolist(), key=lambda i: fill_key[i], reverse=True)
            for i in order:
                c = float(cost_if_paid[i])
                if c <= leftover + 1e-9:
                    action_is_paid[i] = True
                    total_spend += c
                    leftover -= c

        assert total_spend <= self.budget + 1e-9, "MF-BWI-Fair budget invariant violated"

        actions: dict = {}
        spend_by_group = {g: 0.0 for g in self.group_sizes}
        for i, v in enumerate(self.nodes):
            paid = bool(action_is_paid[i])
            if s_arr[i] == 0.0:
                actions[v] = Action.CONVERT if paid else Action.NONE
            else:
                actions[v] = Action.MAINTAIN if paid else Action.NONE
            if paid:
                spend_by_group[self.group_of[v]] += float(cost_if_paid[i])

        self.budget_usage_log.append(total_spend)
        self.group_spend_log.append(spend_by_group)
        self.lambda_star_log.append(float(lam_star))
        return actions

    def observe(self, observations: list) -> None:
        """Beta-Bernoulli posterior update from one round's realized transitions."""
        for obs in observations:
            if obs["param"] == "q":
                key = ("q", obs["v"])
            else:
                key = (obs["param"], obs["u"], obs["v"])
            self.tracker.update(key, obs["success"])

    def group_reach_estimates(self) -> dict:
        """For diagnostics/tests: current per-group time-averaged realized reach
        fraction u_g (the quantity fairness.group_welfare_weights consumes)."""
        return dict(self.group_u_avg)


def run_mf_bwi_fair(
    G: nx.DiGraph,
    true_beta: float,
    T: int,
    budget: float,
    alpha_fair: float,
    init_active=(),
    seed=None,
    **policy_kwargs,
) -> dict:
    """Full end-to-end run: policy + simulator + Bayesian updates, T rounds.

    true_beta and the graph's own 'p_plus'/'q' attributes (via
    simulator.true_params_from_graph) define the hidden ground truth used by the
    simulator; the policy only ever sees posterior means derived from observations.

    Any extra keyword arguments (gamma, u_floor, solver, n_bisect_iters,
    n_vi_sweeps, mf_tol, mf_max_iter, alpha0, beta0) are forwarded to MFBWIFair's constructor;
    existing callers that only pass the original positional/keyword arguments are
    unaffected.
    """
    rng = np.random.default_rng(seed)
    p_plus_true, q_true = true_params_from_graph(G)

    policy = MFBWIFair(G, budget=budget, alpha_fair=alpha_fair, **policy_kwargs)
    state = {v: (v in set(init_active)) for v in G.nodes()}

    trajectory = [state]
    actions_history = []
    for _t in range(T):
        actions = policy.choose_actions(state, rng)
        new_state, observations = simulate_step(G, state, p_plus_true, q_true, true_beta, actions, rng)
        policy.observe(observations)
        policy.update_group_reach(new_state)

        trajectory.append(new_state)
        actions_history.append(actions)
        state = new_state

    return {
        "trajectory": trajectory,
        "actions_history": actions_history,
        "budget_usage_log": policy.budget_usage_log,
        "group_spend_log": policy.group_spend_log,
        "lambda_star_log": policy.lambda_star_log,
        "policy": policy,
    }
