"""Generic Lagrangian relaxation solver for a per-arm 2-state, multi-action
restless-bandit subproblem.

This module has no knowledge of graphs, mean-field beliefs, or fairness -- it is a
small, reusable numerical building block: given, for each of n independent arms, a
"no-action" transition model (p01 = P(0->1), p10 = P(1->0)) and a per-arm reward
weight w (reward r(s) = w*s, s in {0,1}), plus two extra deterministic actions
CONVERT (0->1, cost c_convert, only from s=0) and MAINTAIN (1->1, cost c_maintain,
only from s=1), it finds the shared Lagrange multiplier lambda that makes the
aggregate expected cost of every arm's Lagrangian-optimal action match a shared
budget B as closely as possible from below.

This is a direct, minimal specialization of:

    Killian, J. A., Perrault, A., & Tambe, M. (2021). "Beyond 'To Act or Not to
    Act': Fast Lagrangian Approaches to General Multi-Action Restless Bandits."
    AAMAS 2021.

For M actions with an ascending cost vector C (c_0 = 0) and a shared budget B, they
solve, per arm i, the Lagrangian-relaxed per-state value function

    V^i(s^i, lambda) = max_{a_j in A} { r^i(s^i) - lambda*c_j
                                        + beta * sum_{s'} T(s^i, a_j, s') V^i(s', lambda) }

and prove (their Prop 4.1) that V^i(s^i, lambda) is CONVEX and NON-INCREASING in
lambda. A direct consequence is that each arm's Lagrangian-optimal action's cost is
non-increasing in lambda (a higher shadow price for acting can only make acting less
attractive), so the SUM of costs across arms -- the quantity solve_lambda_bisection
searches over -- is also non-increasing in lambda. This is exactly what makes
bisection ("BLam" in their paper) a valid way to find the smallest lambda whose
aggregate cost is <= B, which is what this module implements. We rely on (cite)
their Prop 4.1 rather than re-deriving it here; tests/test_lagrangian_index.py
includes an empirical regression check that this monotonicity actually holds for our
concrete transition/reward setup, as a sanity check on this implementation, not a
substitute for their proof.

We do not attempt to prove indexability (Whittle-style) for this >2-action setting --
Killian et al. explicitly note that is "notoriously difficult" for M > 2 actions and
avoid it via exactly this Lagrangian-bisection route instead of a closed-form index.

All arms share (gamma, c_convert, c_maintain); only (p01, p10, w, s) vary per arm.
Everything below is vectorized over arms with numpy for speed (a Python-level loop
per arm, times ~50 bisection steps, times ~200 value-iteration sweeps, would be slow
in pure Python for graphs of even a few hundred nodes).
"""

from __future__ import annotations

import numpy as np


def value_iteration(
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    lam: float,
    cost_convert: float,
    cost_maintain: float,
    sweeps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Converged (V(s=0, lam), V(s=1, lam)) for n independent 2-state MDPs.

    Bellman updates (r(0)=0, r(1)=w):
      V(0) = max( gamma*[(1-p01)*V(0) + p01*V(1)],                  # NONE
                  -lam*cost_convert + gamma*V(1) )                  # CONVERT (0->1)
      V(1) = max( w + gamma*[(1-p10)*V(1) + p10*V(0)],              # NONE
                  w - lam*cost_maintain + gamma*V(1) )              # MAINTAIN (1->1)

    This is a 2-state MDP per arm, so a fixed small number of synchronous sweeps
    (default 200) from a zero initialization converges comfortably given gamma<1;
    no need for anything fancier (e.g. policy iteration) at this scale.
    """
    n = p01.shape[0]
    V0 = np.zeros(n)
    V1 = np.zeros(n)
    for _ in range(sweeps):
        q0_none = gamma * ((1.0 - p01) * V0 + p01 * V1)
        q0_convert = -lam * cost_convert + gamma * V1
        new_V0 = np.maximum(q0_none, q0_convert)

        q1_none = w + gamma * ((1.0 - p10) * V1 + p10 * V0)
        q1_maintain = w - lam * cost_maintain + gamma * V1
        new_V1 = np.maximum(q1_none, q1_maintain)

        V0, V1 = new_V0, new_V1
    return V0, V1


def bellman_gaps_and_costs(
    s: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    lam: float,
    cost_convert: float,
    cost_maintain: float,
    V0: np.ndarray,
    V1: np.ndarray,
):
    """Policy extraction: one more Bellman step at each arm's ACTUAL current state s,
    using the converged V(., lam) (standard "extract a policy from a value function").

    Returns:
      action_is_paid: bool array, True where the Lagrangian-optimal action at (s, lam)
        is the paid one (CONVERT if s=0, MAINTAIN if s=1) rather than NONE.
      cost: float array, the cost actually incurred (0 where action_is_paid is False).
      gap: float array, Q(paid action) - Q(NONE) at (s, lam), for every arm -- used
        both for the action_is_paid decision (gap > 0) and, downstream, to rank arms
        for the residual-budget tie-break fill (mf_bwi_fair.py).
      cost_if_paid: float array, the cost of the (state-appropriate) paid action for
        every arm, whether or not it was actually chosen -- also needed by the fill.
    """
    q0_none = gamma * ((1.0 - p01) * V0 + p01 * V1)
    q0_convert = -lam * cost_convert + gamma * V1
    q1_none = w + gamma * ((1.0 - p10) * V1 + p10 * V0)
    q1_maintain = w - lam * cost_maintain + gamma * V1

    is_active = s.astype(bool)
    gap = np.where(is_active, q1_maintain - q1_none, q0_convert - q0_none)
    cost_if_paid = np.where(is_active, cost_maintain, cost_convert).astype(float)
    action_is_paid = gap > 0.0  # ties go to NONE (cheaper), an arbitrary but harmless tie-break
    cost = np.where(action_is_paid, cost_if_paid, 0.0)
    return action_is_paid, cost, gap, cost_if_paid


def aggregate_cost(
    s: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    lam: float,
    cost_convert: float,
    cost_maintain: float,
    sweeps: int = 200,
):
    """Total expected cost of every arm's Lagrangian-optimal action at this lam --
    the quantity Killian et al.'s bisection ("BLam") searches over."""
    V0, V1 = value_iteration(p01, p10, w, gamma, lam, cost_convert, cost_maintain, sweeps)
    action_is_paid, cost, gap, cost_if_paid = bellman_gaps_and_costs(
        s, p01, p10, w, gamma, lam, cost_convert, cost_maintain, V0, V1
    )
    return float(cost.sum()), action_is_paid, cost, gap, cost_if_paid


def lambda_upper_bound(w: np.ndarray, gamma: float, cost_min: float = 1.0, margin: float = 1e-6) -> float:
    """A lambda beyond which EVERY arm's Lagrangian-optimal action is NONE.

    Derivation (used to pick a "sane upper bound" per the design spec, rather than
    an arbitrary large constant): reward r(s) = w*s in {0, w}, and NONE (cost 0) is
    always available, so for every lam >= 0 the converged value obeys
    0 <= V(s, lam) <= w_max / (1 - gamma) (the value of collecting w_max every round
    forever, discounted -- an unconditional upper bound since subtracting lam*cost
    can only ever lower V relative to the free/cost-0 version of the same MDP).

    The benefit of the (state-appropriate) paid action over NONE, *excluding* its
    cost term, is bounded by gamma*(V(1) - V(0)) <= gamma * w_max / (1 - gamma) in
    both cases (CONVERT and MAINTAIN; see the algebra in the module tests). So once
    lam * cost_min >= w_max / (1 - gamma) (which, since gamma < 1, already implies
    lam * cost_min > gamma * w_max / (1 - gamma)), no paid action -- whose cost is at
    least cost_min -- can ever beat NONE, for any arm, at any state. We therefore use

        lam_max = w_max / (1 - gamma)

    (with cost_min = min(cost_convert, cost_maintain) = 1 here) as the safe upper
    bound, plus a tiny multiplicative margin for floating-point safety.
    """
    w_max = float(np.max(w)) if w.size else 0.0
    if w_max <= 0.0:
        return margin  # degenerate all-zero-weight case; any tiny lam already suffices
    return (1.0 + margin) * w_max / (1.0 - gamma) / max(cost_min, 1e-12)


def solve_lambda_bisection(
    s: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    budget: float,
    gamma: float,
    cost_convert: float,
    cost_maintain: float,
    lam_max: float | None = None,
    n_bisect_iters: int = 40,
    n_vi_sweeps: int = 200,
):
    """Smallest lam in [0, lam_max] whose aggregate expected cost is <= budget
    (Killian et al. 2021's "BLam", specialized to this 2-state/3-action setting).

    Returns (lam_star, action_is_paid, cost, gap, cost_if_paid) -- all but lam_star
    are the arrays from bellman_gaps_and_costs at lam_star. Bisection relies on
    aggregate_cost(lam) being non-increasing in lam (see module docstring); it is
    NOT re-verified here on every call (that would defeat the point of bisection) --
    tests/test_lagrangian_index.py checks it once, empirically, as a regression
    guard on this implementation.
    """
    if lam_max is None:
        lam_max = lambda_upper_bound(w, gamma, cost_min=min(cost_convert, cost_maintain))

    cost0, paid0, arr_cost0, gap0, cip0 = aggregate_cost(
        s, p01, p10, w, gamma, 0.0, cost_convert, cost_maintain, n_vi_sweeps
    )
    if cost0 <= budget:
        # Budget is generous enough that even lam=0 (every arm takes its best
        # action free of any shadow price) fits -- nothing to bisect.
        return 0.0, paid0, arr_cost0, gap0, cip0

    lo, hi = 0.0, lam_max
    cost_hi, paid_hi, arr_cost_hi, gap_hi, cip_hi = aggregate_cost(
        s, p01, p10, w, gamma, hi, cost_convert, cost_maintain, n_vi_sweeps
    )
    # By construction of lam_max, cost_hi should be 0 (<= budget for any budget >= 0);
    # this is our bisection invariant's base case (an always-feasible upper bracket).
    best = (hi, paid_hi, arr_cost_hi, gap_hi, cip_hi)

    for _ in range(n_bisect_iters):
        mid = 0.5 * (lo + hi)
        cost_mid, paid_mid, arr_cost_mid, gap_mid, cip_mid = aggregate_cost(
            s, p01, p10, w, gamma, mid, cost_convert, cost_maintain, n_vi_sweeps
        )
        if cost_mid <= budget:
            hi = mid
            best = (mid, paid_mid, arr_cost_mid, gap_mid, cip_mid)
        else:
            lo = mid

    return best
