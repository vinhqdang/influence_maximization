"""Tests for the generic Lagrangian multi-action solver (im_lab/lagrangian_index.py).

These use synthetic per-arm (p01, p10, w) data directly -- no graph/mean-field/
fairness machinery involved -- to isolate and regression-test the Killian et al.
2021-style bisection machinery itself. tests/test_mf_bwi_fair.py covers it wired
into the full MF-BWI-Fair policy.
"""

from __future__ import annotations

import numpy as np

from im_lab import lagrangian_index as li
from im_lab.actions import Action, ACTION_COST

COST_CONVERT = float(ACTION_COST[Action.CONVERT])
COST_MAINTAIN = float(ACTION_COST[Action.MAINTAIN])
GAMMA = 0.9


def _synthetic_arms(n=25, seed=0):
    """n arms with distinct, randomized (p01, p10, w) -- distinct enough that arms
    generically flip from NONE to their paid action at distinct lambda thresholds
    (no exact ties), which is what the "leftover < one action's cost" bisection
    property below relies on."""
    rng = np.random.default_rng(seed)
    p01 = rng.uniform(0.01, 0.4, size=n)
    p10 = rng.uniform(0.01, 0.4, size=n)
    w = rng.uniform(0.5, 5.0, size=n)
    s = (rng.random(n) < 0.5).astype(float)
    return s, p01, p10, w


def test_aggregate_cost_is_non_increasing_as_lambda_increases():
    """Regression check on Killian et al. 2021's Prop 4.1 (V convex & non-increasing
    in lambda), which is what makes bisection over lambda valid here: as lambda
    increases (acting becomes more expensive), the aggregate expected cost of every
    arm's Lagrangian-optimal action must never INCREASE. We check this on a
    concrete instance across a fine lambda grid, rather than re-deriving the proof.
    """
    s, p01, p10, w = _synthetic_arms(n=30, seed=1)
    lam_max = li.lambda_upper_bound(w, GAMMA, cost_min=min(COST_CONVERT, COST_MAINTAIN))

    lambdas = np.linspace(0.0, lam_max, 60)
    costs = []
    for lam in lambdas:
        cost, *_ = li.aggregate_cost(s, p01, p10, w, GAMMA, lam, COST_CONVERT, COST_MAINTAIN)
        costs.append(cost)

    diffs = np.diff(costs)
    assert np.all(diffs <= 1e-9), (
        f"aggregate cost increased somewhere as lambda increased: {costs}"
    )
    # Sanity: the property is non-trivial on this instance (cost actually varies).
    assert costs[0] > costs[-1]
    assert costs[-1] == 0.0  # at lam_max every arm's action must be NONE


def test_bisection_finds_feasible_lambda_star_with_bounded_leftover():
    """The bisection converges to a lambda* whose aggregate cost is <= budget, and
    (on an instance with no exact lambda-threshold ties between arms) leaves less
    than one paid action's worth of budget unused -- i.e. the discreteness gap the
    tie-break fill step in MFBWIFair.choose_actions is designed to mop up."""
    s, p01, p10, w = _synthetic_arms(n=40, seed=2)
    budget = 37.0  # deliberately not a clean multiple of either action's cost

    lam_star, action_is_paid, cost_arr, gap_arr, cost_if_paid = li.solve_lambda_bisection(
        s, p01, p10, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN, n_bisect_iters=50
    )

    total_cost = float(cost_arr.sum())
    assert total_cost <= budget + 1e-9
    leftover = budget - total_cost
    # Generic instance (no ties) => at most one arm's paid-action cost of slack.
    assert leftover < COST_CONVERT, f"leftover {leftover} too large for a single-arm gap"
    assert lam_star >= 0.0


def test_tie_break_style_fill_reduces_leftover_budget():
    """Directly exercises the same "rank remaining NONE arms by action-value gap,
    descending, fill while budget remains" rule MFBWIFair.choose_actions applies
    after the bisection, and checks it actually reduces leftover budget (without
    ever exceeding it) -- the property the redesign spec explicitly asks to
    verify. The bisection alone generically cannot spend the very last few units
    of budget (a discreteness artifact); this greedy fill is what recovers them.
    """
    s, p01, p10, w = _synthetic_arms(n=40, seed=3)
    budget = 41.0

    lam_star, action_is_paid, cost_arr, gap_arr, cost_if_paid = li.solve_lambda_bisection(
        s, p01, p10, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN, n_bisect_iters=50
    )
    action_is_paid = action_is_paid.copy()
    spend_before_fill = float(cost_arr.sum())
    leftover_before = budget - spend_before_fill

    not_paid_idx = np.where(~action_is_paid)[0]
    order = sorted(not_paid_idx.tolist(), key=lambda i: gap_arr[i], reverse=True)
    spend_after_fill = spend_before_fill
    leftover = leftover_before
    for i in order:
        c = float(cost_if_paid[i])
        if c <= leftover + 1e-9:
            action_is_paid[i] = True
            spend_after_fill += c
            leftover -= c

    assert spend_after_fill <= budget + 1e-9
    assert spend_after_fill >= spend_before_fill
    if leftover_before >= min(COST_CONVERT, COST_MAINTAIN):
        # There was room for at least one more cheap (MAINTAIN, cost 1) action --
        # the fill step must have actually used some of that leftover.
        assert spend_after_fill > spend_before_fill
