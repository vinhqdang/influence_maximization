"""Repeated-greedy baseline: acts EVERY round under the same per-round budget B
as MF-BWI-Fair, but with plain classical cost-effective greedy logic -- no
mean-field belief propagation, no Bayesian parameter estimation, no fairness
floors. It is handed the TRUE (ground-truth) p_plus/q/beta directly rather than
posterior means, which is a deliberate advantage: this baseline is meant to be
the strongest plausible "classical, non-fancy" competitor that still gets to
act every round, so that any remaining gap to MF-BWI-Fair can be attributed to
MF-BWI-Fair's fairness/uncertainty-handling machinery rather than to "gets to
spend budget every round" or "doesn't know the true parameters".

--- Per-round action selection: cost-effective ratio greedy over a short MC
    forward-rollout marginal-gain estimate ---

Every round, for every node v, exactly one action is meaningful given its
current state (CONVERT if inactive, MAINTAIN if active -- same convention as
MF-BWI-Fair). The marginal gain of forcing that action is estimated by:

  1. Running `NUM_SIMS` independent Monte Carlo forward rollouts of the REAL
     simulator (simulator.simulate_step, using the TRUE p_plus/q/beta) for
     `LOOKAHEAD_ROUNDS` rounds from the current state with NO actions applied
     anywhere ("do nothing" baseline), averaging the resulting active-node
     count -> base_value.
  2. For each candidate node v, running the same `NUM_SIMS` x `LOOKAHEAD_ROUNDS`
     rollout, but forcing v's candidate action at round 0 (and NONE elsewhere,
     exactly as in step 1) -> value(v). gain(v) = value(v) - base_value.
  3. Sorting all candidates by gain(v) / cost(v) descending (the standard
     cost-effective-greedy ratio, same quantity CELF/kkt_greedy.py uses to pick
     the next seed) and accepting nodes in that order while budget remains,
     stopping as soon as the ratio turns non-positive (no more nodes make
     forcing an action worthwhile -- the residual budget is simply left unspent,
     the same behavior as MF-BWI-Fair leaving nonnegative-but-zero-index
     candidates unfunded).

Why LOOKAHEAD_ROUNDS >= 2: under this simulator's action semantics, a forced
action at round 0 only fixes the ACTING node's own state for that round --
other nodes' round-0 transitions are computed from the state at the START of
the round (see simulator.simulate_step), so a 1-round lookahead would credit a
CONVERT/MAINTAIN action with *zero* network effect (no cascading credit to
neighbors) and collapse to a trivial single-node heuristic. A 2-round lookahead
is the minimum that lets a forced action's effect reach the acting node's
out-neighbors in the second round, which is exactly the "does this action
actually spread influence" signal a greedy IM baseline needs to be meaningful.
We use LOOKAHEAD_ROUNDS = 2 (not more) to keep this cheap -- see below.

--- Simplification vs. "true" CELF, documented as instructed ---

kkt_greedy.celf_greedy's lazy-forward trick (recompute a candidate's marginal
gain against the CURRENT partial seed set only when it reaches the top of a
heap, exploiting submodularity so most candidates are never recomputed) is
built for a ONE-SHOT problem picking a small k (here, k=20) out of n
candidates. Applied naively every round here, a single round can select up to
B/1 = 100 actions (mostly cheap MAINTAINs, cost 1) out of n=120 candidates --
i.e., nearly the ENTIRE candidate pool, every one of the T=30 rounds, for every
(trial, algorithm, swept-parameter) combination in the study. Re-deriving each
acceptance's marginal gain against the growing selected set (full adaptive
CELF) would need, in the worst case, O(n) recomputation passes for EACH of up
to ~100 acceptances per round -- i.e., a per-round cost that scales with
n^2, not n. That is not "comparable in cost to MF-BWI-Fair's per-round
mean-field fixed point + index computation" (see mf_bwi_fair.py), it is orders
of magnitude more.

So this implementation uses the simpler, well-known "one-pass cost-effective
greedy": marginal gains are estimated ONCE per round, against the EMPTY
round-action-set (not re-estimated against the growing selected set as more
nodes are picked), then nodes are accepted in one descending-ratio pass. This
is the standard practical relaxation of adaptive submodular greedy for large
per-step batches (freezing gains against the "no action" baseline rather than
adaptively re-evaluating them), and it makes this baseline's per-round cost
FIXED and predictable: exactly (n + 1) marginal-value estimates per round
(1 "do nothing" baseline + 1 per candidate node), each estimate being
`NUM_SIMS * LOOKAHEAD_ROUNDS` calls to simulate_step -- regardless of how many
actions end up selected. It does give up genuine adaptive diminishing-returns
correction WITHIN a round (e.g. two candidates whose 2-hop lookahead windows
overlap are each scored as if the other were not also being acted on) --
exactly the kind of simplification the task calls for documenting rather than
hiding. Diminishing returns ACROSS rounds are still captured correctly, since
the whole process re-runs from the real, freshly-observed state every round.

--- Compute budget vs. MF-BWI-Fair, benchmarked ---

NUM_SIMS=2, LOOKAHEAD_ROUNDS=2 were chosen after directly benchmarking both
policies' `choose_actions` call on this study's actual 120-node/422-edge SBM
graph (see experiments/RESULTS.md for the reported numbers). In *operation-
count* terms this is deliberately generous to repeated_greedy, not stingy:
each round costs (n + 1) marginal-value estimates x NUM_SIMS x LOOKAHEAD_ROUNDS
= 121 x 2 x 2 = 484 full-graph transition passes (O(n+m) each) -- roughly
484 x 542 ~= 2.6e5 node/edge touches -- versus MF-BWI-Fair's mean-field fixed
point (<=50 iterations, O(n+m) each, but empirically converges in far fewer
than 50 on this graph) plus its O(n) index pass, on the order of 1e4-3e4
node/edge touches: a single-digit-to-low-double-digit multiple, not orders of
magnitude, and in repeated_greedy's favor if anything. In *wall-clock* terms
the two are NOT at parity (repeated_greedy measured ~0.03-0.12s/round on this
graph vs. MF-BWI-Fair's ~0.001s/round), because MF-BWI-Fair's inner loop is
vectorizable-free pure dict arithmetic with no repeated Python function-call
overhead across independent rollouts, while repeated_greedy repeats a full
Python-level simulate_step call hundreds of times per round; closing that gap
would require rewriting the rollout in vectorized numpy (a real engineering
investment we did not judge worth making for a baseline whose whole point is
to be the "no fancy machinery" comparison point) or exploiting common-random-
number locality tricks. We chose to keep the operation-count budget comparable
(the dimension the task cares about for a fair algorithm-quality comparison)
and accept the wall-clock cost, since the full four-way study still completes
in single-digit minutes at N_TRIALS=15, T=30 (see RESULTS.md for the measured
total). Both NUM_SIMS and LOOKAHEAD_ROUNDS are intentionally small: this makes
each individual marginal-gain estimate noisy, but the noise is independent
across the ~120 candidates scored each round and the policy re-estimates from
scratch next round anyway, so it washes out in the many-round, many-trial
aggregate this study reports rather than needing a precise single-round
estimate.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..actions import Action, ACTION_COST
from ..simulator import count_active, simulate_step, true_params_from_graph

LOOKAHEAD_ROUNDS = 2
NUM_SIMS = 2


def _candidate_action(state: dict, v) -> Action:
    return Action.CONVERT if not state[v] else Action.MAINTAIN


def _rollout_value(
    G: nx.DiGraph,
    state: dict,
    round0_actions: dict,
    none_actions: dict,
    p_plus: dict,
    q: dict,
    beta: float,
    lookahead_rounds: int,
    num_sims: int,
    rng: np.random.Generator,
) -> float:
    """Average active-node count `lookahead_rounds` rounds ahead of `state`,
    over `num_sims` independent forward rollouts of the REAL simulator, forcing
    `round0_actions` at the first round and NONE (natural dynamics) after."""
    total = 0
    for _ in range(num_sims):
        s = state
        for t in range(lookahead_rounds):
            acts = round0_actions if t == 0 else none_actions
            s, _obs = simulate_step(G, s, p_plus, q, beta, acts, rng)
        total += count_active(s)
    return total / num_sims


def choose_actions_repeated_greedy(
    G: nx.DiGraph,
    state: dict,
    p_plus: dict,
    q: dict,
    beta: float,
    budget: float,
    rng: np.random.Generator,
    lookahead_rounds: int = LOOKAHEAD_ROUNDS,
    num_sims: int = NUM_SIMS,
) -> tuple[dict, float]:
    """One round of repeated-greedy action selection (see module docstring).

    Returns (actions: dict[node, Action], total_spend: float).
    """
    nodes = list(G.nodes())
    none_actions = {v: Action.NONE for v in nodes}

    base_value = _rollout_value(
        G, state, none_actions, none_actions, p_plus, q, beta, lookahead_rounds, num_sims, rng
    )

    candidates = []  # (ratio, gain, cost, node, action)
    for v in nodes:
        action = _candidate_action(state, v)
        cost = ACTION_COST[action]
        trial_actions = dict(none_actions)
        trial_actions[v] = action
        value_v = _rollout_value(
            G, state, trial_actions, none_actions, p_plus, q, beta, lookahead_rounds, num_sims, rng
        )
        gain = value_v - base_value
        ratio = gain / cost
        candidates.append((ratio, gain, cost, v, action))

    # One-pass cost-effective greedy: sort once by gain/cost, accept in
    # descending order while affordable, stop at the first non-positive ratio
    # (see module docstring for why this is not fully adaptive CELF).
    candidates.sort(key=lambda c: c[0], reverse=True)

    actions = dict(none_actions)
    remaining = float(budget)
    total_spend = 0.0
    for ratio, gain, cost, v, action in candidates:
        if ratio <= 0:
            break
        if cost > remaining:
            continue
        actions[v] = action
        remaining -= cost
        total_spend += cost

    assert total_spend <= budget + 1e-9, "repeated_greedy budget invariant violated"
    return actions, total_spend


class RepeatedGreedy:
    """Stateless-except-for-logging policy wrapper, mirroring MFBWIFair's shape
    for easy interchangeability in experiments/common.py."""

    def __init__(
        self,
        G: nx.DiGraph,
        budget: float,
        true_beta: float,
        lookahead_rounds: int = LOOKAHEAD_ROUNDS,
        num_sims: int = NUM_SIMS,
    ):
        self.G = G
        self.budget = budget
        self.true_beta = true_beta
        self.lookahead_rounds = lookahead_rounds
        self.num_sims = num_sims
        # TRUE parameters, used directly -- no Bayesian estimation (see module
        # docstring: this baseline is deliberately given every advantage other
        # than fairness- or uncertainty-awareness).
        self.p_plus, self.q = true_params_from_graph(G)

        self.budget_usage_log: list = []

    def choose_actions(self, state: dict, rng: np.random.Generator = None) -> dict:
        if rng is None:
            rng = np.random.default_rng()
        actions, total_spend = choose_actions_repeated_greedy(
            self.G, state, self.p_plus, self.q, self.true_beta, self.budget, rng,
            self.lookahead_rounds, self.num_sims,
        )
        self.budget_usage_log.append(total_spend)
        return actions


def run_repeated_greedy(
    G: nx.DiGraph,
    true_beta: float,
    T: int,
    budget: float,
    init_active=(),
    lookahead_rounds: int = LOOKAHEAD_ROUNDS,
    num_sims: int = NUM_SIMS,
    seed=None,
) -> dict:
    """Full end-to-end run: repeated-greedy policy + real sequential simulator,
    T rounds. Mirrors mf_bwi_fair.run_mf_bwi_fair's return shape."""
    rng = np.random.default_rng(seed)
    p_plus_true, q_true = true_params_from_graph(G)

    policy = RepeatedGreedy(G, budget=budget, true_beta=true_beta,
                            lookahead_rounds=lookahead_rounds, num_sims=num_sims)
    state = {v: (v in set(init_active)) for v in G.nodes()}

    trajectory = [state]
    actions_history = []
    for _t in range(T):
        actions = policy.choose_actions(state, rng)
        new_state, _observations = simulate_step(G, state, p_plus_true, q_true, true_beta, actions, rng)

        trajectory.append(new_state)
        actions_history.append(actions)
        state = new_state

    return {
        "trajectory": trajectory,
        "actions_history": actions_history,
        "budget_usage_log": policy.budget_usage_log,
        "policy": policy,
    }
