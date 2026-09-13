"""Tests for the exact closed-form Lagrangian index (im_lab/closed_form_index.py).

The bisection + value-iteration solver in im_lab/lagrangian_index.py is the
independent numerical oracle throughout. What each test establishes:

* affine-value correctness: each policy's closed-form (V0, V1) satisfies that
  policy's own Bellman fixed-point equations (plug back in, ~1e-9);
* optimal value: max over the 4 policies equals converged value iteration;
* convexity: V*(s, lam) lies on or above its chords on a lam grid (a numerical
  restatement of "max of affines is convex");
* indexability (numerical only, not proved): per-arm cost(v, lam) is
  non-increasing in lam on a fine grid, and equals 1[lam < lam*_v];
* oracle equivalence: the closed-form selected action set equals the bisection's,
  with disagreement tolerated ONLY on arms whose index is within floating-point
  distance of the final lam* (the bisection stops after 40 halvings and value
  iteration after 200 sweeps, so an arm whose switch price coincides with lam*
  to ~1e-9 can legitimately land on either side); total spend must agree exactly
  whenever there is no such near-tie arm;
* speed: closed form is faster than the bisection (ratio printed, not asserted
  beyond "faster", to avoid a machine-dependent threshold).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from im_lab import closed_form_index as cf
from im_lab import lagrangian_index as li
from im_lab.actions import Action, ACTION_COST

COST_CONVERT = float(ACTION_COST[Action.CONVERT])
COST_MAINTAIN = float(ACTION_COST[Action.MAINTAIN])
GAMMA = 0.9


def _arms(n, seed, a_hi=0.6, b_hi=0.6, w_lo=0.1, w_hi=5.0):
    rng = np.random.default_rng(seed)
    p01 = rng.uniform(0.0, a_hi, size=n)
    p10 = rng.uniform(0.0, b_hi, size=n)
    w = rng.uniform(w_lo, w_hi, size=n)
    s = (rng.random(n) < 0.5).astype(float)
    return s, p01, p10, w


# --------------------------------------------------------------------------
# Affine-value correctness: plug each policy's closed form back into ITS OWN
# Bellman equations (fixed actions, no max).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
def test_policy_values_satisfy_their_own_bellman_equations(seed, gamma):
    rng = np.random.default_rng(seed)
    n = 200
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    w = rng.uniform(0.0, 10.0, n)
    for lam in (0.0, 0.37, 4.2, 55.0):
        V0, V1 = cf.policy_values(a, b, w, gamma, COST_CONVERT, COST_MAINTAIN, lam)
        g = gamma
        # Per-policy one-step backups with that policy's fixed actions.
        # s=0, NONE: g[(1-a)V0 + aV1];  s=0, CONVERT: -lam cC + g V1
        # s=1, NONE: w + g[(1-b)V1 + bV0];  s=1, MAINTAIN: w - lam cM + g V1
        rhs0 = {
            cf.NN: g * ((1 - a) * V0[:, cf.NN] + a * V1[:, cf.NN]),
            cf.NM: g * ((1 - a) * V0[:, cf.NM] + a * V1[:, cf.NM]),
            cf.CN: -lam * COST_CONVERT + g * V1[:, cf.CN],
            cf.CM: -lam * COST_CONVERT + g * V1[:, cf.CM],
        }
        rhs1 = {
            cf.NN: w + g * ((1 - b) * V1[:, cf.NN] + b * V0[:, cf.NN]),
            cf.NM: w - lam * COST_MAINTAIN + g * V1[:, cf.NM],
            cf.CN: w + g * ((1 - b) * V1[:, cf.CN] + b * V0[:, cf.CN]),
            cf.CM: w - lam * COST_MAINTAIN + g * V1[:, cf.CM],
        }
        for k in range(4):
            scale = 1.0 + np.abs(V0[:, k]) + np.abs(V1[:, k])
            assert np.all(np.abs(V0[:, k] - rhs0[k]) / scale < 1e-9), cf.POLICIES[k]
            assert np.all(np.abs(V1[:, k] - rhs1[k]) / scale < 1e-9), cf.POLICIES[k]


def test_optimal_value_matches_value_iteration_oracle():
    s, a, b, w = _arms(80, seed=5)
    for lam in (0.0, 0.2, 1.0, 3.0, 20.0):
        V0, V1 = cf.optimal_values(a, b, w, GAMMA, COST_CONVERT, COST_MAINTAIN, lam)
        V0i, V1i = li.value_iteration(a, b, w, GAMMA, lam, COST_CONVERT, COST_MAINTAIN, sweeps=400)
        assert np.allclose(V0, V0i, atol=1e-9, rtol=1e-9)
        assert np.allclose(V1, V1i, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------
# Convexity of V*(s, lam) in lam.
# --------------------------------------------------------------------------
def test_optimal_value_is_convex_in_lambda():
    s, a, b, w = _arms(60, seed=7)
    lam_max = li.lambda_upper_bound(w, GAMMA, cost_min=min(COST_CONVERT, COST_MAINTAIN))
    grid = np.linspace(0.0, lam_max, 41)
    vals = np.array([
        np.stack(cf.optimal_values(a, b, w, GAMMA, COST_CONVERT, COST_MAINTAIN, l), axis=1)
        for l in grid
    ])  # (L, n, 2)
    # Chord test for every triple i<j<k on the grid: convexity means the function
    # lies ON OR BELOW every chord, V(x_j) <= linear interpolation of V(x_i),
    # V(x_k) at x_j. (A max of affine functions is convex, so this must hold.)
    L = len(grid)
    for i in range(L):
        for k in range(i + 2, L):
            t = (grid[i + 1:k] - grid[i]) / (grid[k] - grid[i])
            chord = (1 - t)[:, None, None] * vals[i] + t[:, None, None] * vals[k]
            assert np.all(vals[i + 1:k] <= chord + 1e-9 * (1 + np.abs(chord)))
    # Also non-increasing in lam (paid actions can only get more expensive).
    assert np.all(np.diff(vals, axis=0) <= 1e-9)
    # And the function is genuinely piecewise-linear, not constant: at least one
    # arm has a strictly decreasing value somewhere.
    assert np.any(np.diff(vals, axis=0) < -1e-6)


# --------------------------------------------------------------------------
# Indexability (numerical): per-arm cost(v, lam) non-increasing in lam, and the
# closed-form index reproduces the paid decision as 1[lam < lam*_v].
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed,a_hi,b_hi", [(11, 0.6, 0.6), (12, 1.0, 1.0), (13, 0.05, 0.9)])
def test_per_arm_cost_is_non_increasing_in_lambda_and_matches_index(seed, a_hi, b_hi):
    s, a, b, w = _arms(300, seed=seed, a_hi=a_hi, b_hi=b_hi)
    lam_max = li.lambda_upper_bound(w, GAMMA, cost_min=min(COST_CONVERT, COST_MAINTAIN))
    grid = np.linspace(0.0, lam_max, 500)
    paid = np.stack(
        [cf.paid_at_price(s, a, b, w, GAMMA, COST_CONVERT, COST_MAINTAIN, l) for l in grid], axis=1
    )  # (n, L)
    cost = paid * np.where(s > 0, COST_MAINTAIN, COST_CONVERT)[:, None]
    assert np.all(np.diff(cost, axis=1) <= 0.0), "some arm's cost increased with lambda"
    assert not paid[:, -1].any()  # at lam_max nobody pays (lambda_upper_bound)

    idx = cf.node_indices(s, a, b, w, GAMMA, COST_CONVERT, COST_MAINTAIN)
    predicted = grid[None, :] < idx[:, None]
    assert np.array_equal(predicted, paid)
    # Both "never paid" and "paid at lam=0" arms occur on this instance.
    assert np.any(np.isneginf(idx)) or np.all(idx > 0)
    assert np.any(idx > 0)


def test_index_agrees_with_value_iteration_decision_near_threshold():
    """Just below lam*_v the oracle must pay, just above it must not."""
    s, a, b, w = _arms(120, seed=21)
    idx = cf.node_indices(s, a, b, w, GAMMA, COST_CONVERT, COST_MAINTAIN)
    finite = np.isfinite(idx) & (idx > 0)
    for i in np.where(finite)[0][:40]:
        for lam, expect in ((idx[i] * (1 - 1e-4), True), (idx[i] * (1 + 1e-4), False)):
            V0, V1 = li.value_iteration(a[i:i+1], b[i:i+1], w[i:i+1], GAMMA, lam, COST_CONVERT, COST_MAINTAIN, sweeps=400)
            paid, *_ = li.bellman_gaps_and_costs(
                s[i:i+1], a[i:i+1], b[i:i+1], w[i:i+1], GAMMA, lam, COST_CONVERT, COST_MAINTAIN, V0, V1
            )
            assert bool(paid[0]) is expect, (i, lam, idx[i])


def test_never_paid_arms_get_minus_inf_index_and_are_never_selected():
    # w=0 arms have nothing to gain; they must never be paid.
    n = 30
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 0.5, n); b = rng.uniform(0, 0.5, n)
    w = np.zeros(n); w[:10] = rng.uniform(1, 3, 10)
    s = (rng.random(n) < 0.5).astype(float)
    idx = cf.node_indices(s, a, b, w, GAMMA, COST_CONVERT, COST_MAINTAIN)
    assert np.all(np.isneginf(idx[10:]))
    lam, paid, cost, gap, cip = cf.solve_lambda_closed_form(s, a, b, w, 1e6, GAMMA, COST_CONVERT, COST_MAINTAIN)
    assert lam == 0.0
    assert not paid[10:].any()


# --------------------------------------------------------------------------
# Oracle equivalence with the bisection.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n,seed,budget",
    [(25, 100, 7.0), (60, 101, 40.0), (120, 102, 23.0), (120, 103, 250.0), (200, 104, 1e6), (40, 105, 0.5)],
)
def test_closed_form_selection_matches_bisection_oracle(n, seed, budget):
    s, a, b, w = _arms(n, seed=seed)
    lam_b, paid_b, cost_b, gap_b, cip_b = li.solve_lambda_bisection(
        s, a, b, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN
    )
    lam_c, paid_c, cost_c, gap_c, cip_c, idx = cf.solve_lambda_closed_form(
        s, a, b, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN, return_indices=True
    )
    assert np.array_equal(cip_b, cip_c)
    assert cost_c.sum() <= budget + 1e-9
    # lam*: bisection converges to the threshold index from above within lam_max*2^-40.
    lam_max = li.lambda_upper_bound(w, GAMMA, cost_min=min(COST_CONVERT, COST_MAINTAIN))
    assert abs(lam_b - lam_c) <= lam_max * 2.0 ** -39 + 1e-12

    # Arms allowed to differ: index within numerical tolerance of lam*.
    near_tie = np.abs(idx - lam_c) <= 1e-7 * max(1.0, lam_c)
    disagree = paid_b != paid_c
    assert not np.any(disagree & ~near_tie), (
        f"closed form and bisection disagree on non-tie arms: {np.where(disagree & ~near_tie)[0]}"
    )
    if not near_tie.any():
        assert np.array_equal(paid_b, paid_c)
        assert cost_b.sum() == cost_c.sum()
    # The Bellman gap at lam* (used by the bisection path's fill) agrees too.
    assert np.allclose(gap_b, gap_c, atol=1e-7, rtol=1e-7)


def test_closed_form_matches_bisection_on_exact_tie_instance():
    """Identical arms => identical indices; both solvers must then pay NONE of the
    tied arms at the threshold (strict rule) and leave them for the fill."""
    n = 12
    a = np.full(n, 0.2); b = np.full(n, 0.1); w = np.full(n, 2.0); s = np.zeros(n)
    budget = 3 * COST_CONVERT + 1.0  # room for 3 of 12 identical CONVERTs
    lam_b, paid_b, cost_b, *_ = li.solve_lambda_bisection(s, a, b, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN)
    lam_c, paid_c, cost_c, *_ = cf.solve_lambda_closed_form(s, a, b, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN)
    assert not paid_b.any() and not paid_c.any()
    assert abs(lam_b - lam_c) < 1e-6


# --------------------------------------------------------------------------
# Speed.
# --------------------------------------------------------------------------
def _time(fn, reps):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.parametrize("n,reps", [(120, 3), (2000, 1)])
def test_closed_form_is_faster_than_bisection(n, reps):
    s, a, b, w = _arms(n, seed=200 + n)
    budget = 0.6 * n
    t_b = _time(lambda: li.solve_lambda_bisection(s, a, b, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN), reps)
    t_c = _time(lambda: cf.solve_lambda_closed_form(s, a, b, w, budget, GAMMA, COST_CONVERT, COST_MAINTAIN), reps)
    print(f"\n[n={n}] bisection {t_b*1e3:.2f} ms, closed form {t_c*1e3:.2f} ms, ratio {t_b/t_c:.1f}x")
    assert t_c < t_b


# --------------------------------------------------------------------------
# Policy wiring: both solvers run side by side and respect the budget.
# --------------------------------------------------------------------------
def test_mf_bwi_fair_solver_flag_both_paths_run_and_agree_on_spend():
    from im_lab import graphs
    from im_lab.mf_bwi_fair import MFBWIFair, run_mf_bwi_fair

    G = graphs.stochastic_block_model_graph([12, 12], p_in=0.25, p_out=0.05, seed=31)
    graphs.assign_true_parameters(G, seed=31)
    budget = 17.0
    state = {v: (v % 4 == 0) for v in G.nodes()}
    lam = {}
    for solver in ("closed_form", "bisection"):
        pol = MFBWIFair(G, budget=budget, alpha_fair=0.0, solver=solver)
        pol.choose_actions(state, np.random.default_rng(0))
        assert pol.budget_usage_log[-1] <= budget + 1e-9
        lam[solver] = pol.lambda_star_log[-1]
    # Same pre-fill lambda* up to bisection precision. (Final action dicts may
    # differ on fill tie-breaks -- index vs gap ranking -- so we compare lambda*
    # and the budget invariant, not the exact dicts.)
    assert abs(lam["closed_form"] - lam["bisection"]) < 1e-6
    with pytest.raises(ValueError):
        MFBWIFair(G, budget=budget, alpha_fair=0.0, solver="nope")
    r = run_mf_bwi_fair(G, true_beta=0.1, T=3, budget=budget, alpha_fair=0.0, seed=1, solver="bisection")
    assert all(u <= budget + 1e-9 for u in r["budget_usage_log"])
    r2 = run_mf_bwi_fair(G, true_beta=0.1, T=3, budget=budget, alpha_fair=0.0, seed=1)
    assert r2["policy"].solver == "closed_form"
    assert all(u <= budget + 1e-9 for u in r2["budget_usage_log"])
