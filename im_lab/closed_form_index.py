"""Exact closed-form Lagrangian index for the per-arm 2-state / 3-action MDP.

This module is the O(n log n)-per-round replacement for the bisection + value-
iteration machinery in im_lab/lagrangian_index.py (which is kept intact and now
serves as the test oracle). Same per-arm model as there: states s in {0 (inactive),
1 (active)}; reward r(s) = w*s; "no-action" transitions p01 = P(0->1) =: a,
p10 = P(1->0) =: b; paid actions CONVERT (cost c_C, forces 0->1, only at s=0) and
MAINTAIN (cost c_M, forces 1->1, only at s=1); discount g; Lagrange price lam per
unit cost, so a paid action's one-step Lagrangian penalty is lam*cost.

Why a closed form exists
------------------------
Each arm's MDP has 2 states and exactly 2 available actions per state, so there
are exactly 4 deterministic stationary policies, named by (action at s=0, action
at s=1):

    NN  never act;                 NM  maintain at s=1, never convert;
    CN  convert at s=0, never maintain;   CM  both paid actions.

For a FIXED policy pi, V_pi = (I - g*P_pi)^{-1} (r - lam*c_pi) is AFFINE in lam.
Solving each policy's 2x2 linear fixed-point system by hand, with
D := (1-g)*(1-g+g*(a+b)):

    NN:  V1 = w*(1-g+g*a) / D                 V0 = g*a*w / D
    NM:  V1 = (w - lam*c_M) / (1-g)           V0 = g*a*(w - lam*c_M) / ((1-g)*(1-g+g*a))
    CN:  V1 = (w - g*b*lam*c_C) / ((1-g)*(1+g*b))   V0 = -lam*c_C + g*V1
    CM:  V1 = (w - lam*c_M) / (1-g)           V0 = -lam*c_C + g*(w - lam*c_M)/(1-g)

(Derivation, e.g. CN: V0 = -lam*c_C + g*V1 and V1 = w + g*[(1-b)*V1 + b*V0];
substituting gives V1*[1 - g*(1-b) - g^2*b] = w - g*b*lam*c_C and
1 - g*(1-b) - g^2*b = (1-g)*(1+g*b). NN: V0*(1-g*(1-a)) = g*a*V1 and
V1*(1-g*(1-b)) = w + g*b*V0; eliminating gives (1-g+g*a)*(1-g+g*b) - g^2*a*b = D.)
tests/test_closed_form_index.py plugs each pair back into its own policy's Bellman
equations as a check on the algebra.

What is PROVED here (for this 2-state / 3-action case only)
-----------------------------------------------------------
* Optimal value. In a discounted finite MDP some stationary deterministic policy
  is optimal at every state simultaneously, so V*(s, lam) = max over the 4
  policies of V_pi(s, lam). Hence V*(s, .) is a maximum of 4 affine functions of
  lam: convex, piecewise-linear, with at most 3 breakpoints. (Killian et al. 2021,
  Prop 4.1, prove convexity for the general multi-action case; here it is a
  one-line consequence of the closed forms, no citation needed.)
* Optimal action = strict policy-value comparison. The Lagrangian-optimal action
  at s=0 is CONVERT iff max(V0_CN, V0_CM) > max(V0_NN, V0_NM) (strictly), and at
  s=1 is MAINTAIN iff max(V1_NM, V1_CM) > max(V1_NN, V1_CN). This is exactly the
  same decision as the strict Bellman-gap test Q*(s, paid) - Q*(s, NONE) > 0 that
  lagrangian_index.bellman_gaps_and_costs uses (ties -> NONE): if the Q-gap is
  strictly positive the optimal policy pays at s, so the best paid policy's value
  is V*(s), which strictly exceeds every unpaid policy's value (an unpaid policy's
  value at s is <= Q*(s, NONE) < Q*(s, paid) = V*(s)); conversely, if the best
  paid policy strictly beats every unpaid one but the Q-gap were zero, the policy
  that plays NONE at s and the optimal action elsewhere would be greedy w.r.t.
  V*, hence optimal, hence an unpaid policy attaining V*(s) -- contradiction.
* Index computation is exact. Because all four V(s, .) are affine, the paid/unpaid
  decision at a state can only change at one of the <= 6 pairwise crossing points
  of those four lines. Sorting the crossings that fall in [0, lam_max] and
  evaluating the decision at the midpoint of every resulting interval therefore
  determines the decision on ALL of [0, lam_max] exactly, in O(1) per arm. The
  index lam*_v is the supremum of the paid set (its upper endpoint), or -inf if
  the arm is never paid on [0, lam_max]. At lam = lam*_v itself the two competing
  values tie, so under the strict rule the arm is paid iff lam < lam*_v.
* Budget matching. Selecting arms in descending order of lam*_v until the next
  one no longer fits is exactly "the smallest lam whose aggregate cost <= B" that
  the bisection targets, PROVIDED the paid set of every arm is a down-set
  [0, lam*_v) (see indexability below): then cost(lam) = sum over arms with
  lam*_v > lam, a non-increasing step function whose steps are the sorted indices.

What is only CHECKED numerically (not proved)
---------------------------------------------
* Indexability, i.e. that each arm's paid set {lam : paid(v, lam)} is a down-set
  in lam. The paid policies' values have steeper negative slope in lam than the
  free NN policy, which makes the down-set structure the expected outcome, but the
  NM policy's slope can be steeper than CN's, so the pairwise ordering alone does
  not settle it. We therefore do NOT claim a proof. The index below is defined as
  the supremum of the paid set (robust to a non-down-set paid region), and
  tests/test_closed_form_index.py checks on random instances, over a fine lam
  grid, that cost(v, lam) is non-increasing in lam for every arm (the same
  empirical regression check tests/test_lagrangian_index.py runs for the aggregate
  cost). Nothing beyond this 2-state / {NONE, CONVERT, MAINTAIN} case is claimed.
* Agreement of the s=0 and s=1 argmax sets. Theory says the optimal policy attains
  the max at both states; node_indices asserts (with a tolerance, treating ties as
  sets) that the two argmax sets intersect, as a runtime sanity check on the
  closed forms.

The bisection oracle (lagrangian_index.solve_lambda_bisection) and this solver are
compared action-set-for-action-set in the tests, with disagreement allowed only on
arms whose index is within floating-point tolerance of the final lam*.
"""

from __future__ import annotations

import numpy as np

from .lagrangian_index import lambda_upper_bound

# Policy order used for every (n, 4) array in this module.
POLICIES = ("NN", "NM", "CN", "CM")
NN, NM, CN, CM = 0, 1, 2, 3
PAID_AT_0 = np.array([False, False, True, True])  # CONVERT at s=0 under CN, CM
PAID_AT_1 = np.array([False, True, False, True])  # MAINTAIN at s=1 under NM, CM


def policy_value_coefficients(
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    cost_convert: float,
    cost_maintain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Affine coefficients V_pi(s, lam) = intercept + slope*lam for all 4 policies.

    Returns (V0_int, V0_slope, V1_int, V1_slope), each of shape (n, 4) in POLICIES
    order. These are the closed forms from the module docstring with the lam
    dependence split out, so every evaluation at a price is one FMA.
    """
    a = np.asarray(p01, dtype=float)
    b = np.asarray(p10, dtype=float)
    w = np.asarray(w, dtype=float)
    g = float(gamma)
    cC = float(cost_convert)
    cM = float(cost_maintain)
    n = a.shape[0]

    one_m_g = 1.0 - g
    D = one_m_g * (one_m_g + g * (a + b))
    den_a = one_m_g * (one_m_g + g * a)  # (1-g)(1-g+ga)
    den_b = one_m_g * (1.0 + g * b)  # (1-g)(1+gb)

    V0_int = np.empty((n, 4))
    V0_slope = np.empty((n, 4))
    V1_int = np.empty((n, 4))
    V1_slope = np.empty((n, 4))

    # NN: independent of lam.
    V1_int[:, NN] = w * (one_m_g + g * a) / D
    V1_slope[:, NN] = 0.0
    V0_int[:, NN] = g * a * w / D
    V0_slope[:, NN] = 0.0

    # NM: V1 = (w - lam cM)/(1-g); V0 = g a (w - lam cM)/((1-g)(1-g+ga)).
    V1_int[:, NM] = w / one_m_g
    V1_slope[:, NM] = -cM / one_m_g
    V0_int[:, NM] = g * a * w / den_a
    V0_slope[:, NM] = -g * a * cM / den_a

    # CN: V1 = (w - g b lam cC)/((1-g)(1+gb)); V0 = -lam cC + g V1.
    V1_int[:, CN] = w / den_b
    V1_slope[:, CN] = -g * b * cC / den_b
    V0_int[:, CN] = g * V1_int[:, CN]
    V0_slope[:, CN] = -cC + g * V1_slope[:, CN]

    # CM: V1 = (w - lam cM)/(1-g); V0 = -lam cC + g V1.
    V1_int[:, CM] = w / one_m_g
    V1_slope[:, CM] = -cM / one_m_g
    V0_int[:, CM] = g * V1_int[:, CM]
    V0_slope[:, CM] = -cC + g * V1_slope[:, CM]

    return V0_int, V0_slope, V1_int, V1_slope


def policy_values(
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    cost_convert: float,
    cost_maintain: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(V0, V1), each (n, 4) in POLICIES order, for all 4 policies at price lam."""
    V0_int, V0_slope, V1_int, V1_slope = policy_value_coefficients(
        p01, p10, w, gamma, cost_convert, cost_maintain
    )
    return V0_int + V0_slope * lam, V1_int + V1_slope * lam


def optimal_values(
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    cost_convert: float,
    cost_maintain: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """V*(0, lam), V*(1, lam) = max over the 4 policies -- the exact fixed point
    lagrangian_index.value_iteration approximates."""
    V0, V1 = policy_values(p01, p10, w, gamma, cost_convert, cost_maintain, lam)
    return V0.max(axis=1), V1.max(axis=1)


def _paid_from_lines(intercept: np.ndarray, slope: np.ndarray, lam, paid_mask: np.ndarray) -> np.ndarray:
    """Strict rule: paid iff best paid-policy value > best unpaid-policy value.

    intercept/slope: (n, 4); lam: scalar or (n, k) broadcastable to (n, k, 1).
    Returns bool array of shape lam's (broadcast) shape.
    """
    lam = np.asarray(lam, dtype=float)
    if lam.ndim == 0:
        vals = intercept + slope * lam  # (n, 4)
    else:
        vals = intercept[:, None, :] + slope[:, None, :] * lam[..., None]  # (n, k, 4)
    best_paid = np.max(np.where(paid_mask, vals, -np.inf), axis=-1)
    best_free = np.max(np.where(~paid_mask, vals, -np.inf), axis=-1)
    return best_paid > best_free


def paid_at_price(
    s: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    cost_convert: float,
    cost_maintain: float,
    lam: float,
) -> np.ndarray:
    """bool (n,): does each arm's Lagrangian-optimal action at its current state s
    pay (CONVERT if s=0, MAINTAIN if s=1) at price lam? Strict rule, ties -> NONE,
    matching lagrangian_index.bellman_gaps_and_costs."""
    V0_int, V0_slope, V1_int, V1_slope = policy_value_coefficients(
        p01, p10, w, gamma, cost_convert, cost_maintain
    )
    is_active = np.asarray(s).astype(bool)
    paid0 = _paid_from_lines(V0_int, V0_slope, lam, PAID_AT_0)
    paid1 = _paid_from_lines(V1_int, V1_slope, lam, PAID_AT_1)
    return np.where(is_active, paid1, paid0)


def node_indices(
    s: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    cost_convert: float,
    cost_maintain: float,
    lam_max: float | None = None,
    check_argmax_agreement: bool = True,
    agreement_tol: float = 1e-7,
) -> np.ndarray:
    """Closed-form index lam*_v for every arm: the supremum of the set of prices
    lam in [0, lam_max] at which the arm's Lagrangian-optimal action at its current
    state s_v is the paid one; -inf if that set is empty (NONE already optimal at
    lam = 0, so the arm is never selected).

    Exact O(1)-per-arm interval scan (vectorized over arms): the decision can only
    flip at a pairwise crossing of the 4 affine V(s_v, .) lines, so we collect the
    <= 6 crossings inside (0, lam_max), sort them together with the endpoints 0
    and lam_max, evaluate the strict paid rule at every interval midpoint, and take
    the largest upper endpoint of a paid interval. No single-equation shortcut is
    used, so exact ties and coincident slopes are handled uniformly.

    check_argmax_agreement: sanity-assert the closed forms are self-consistent --
    the set of policies attaining max_pi V_pi(0, lam) must intersect the set
    attaining max_pi V_pi(1, lam) (the globally optimal policy lies in both); ties
    are treated as sets with tolerance agreement_tol (relative). Checked at every
    interval midpoint, i.e. on every linear piece.
    """
    s = np.asarray(s)
    n = s.shape[0]
    if lam_max is None:
        lam_max = lambda_upper_bound(np.asarray(w, dtype=float), gamma, cost_min=min(cost_convert, cost_maintain))
    lam_max = float(lam_max)

    V0_int, V0_slope, V1_int, V1_slope = policy_value_coefficients(
        p01, p10, w, gamma, cost_convert, cost_maintain
    )
    is_active = s.astype(bool)
    # Lines relevant to the decision at each arm's CURRENT state.
    intercept = np.where(is_active[:, None], V1_int, V0_int)
    slope = np.where(is_active[:, None], V1_slope, V0_slope)
    paid_mask_rows = np.where(is_active[:, None], PAID_AT_1[None, :], PAID_AT_0[None, :])

    # All 6 pairwise crossings lam_ij = (c_i - c_j) / (m_j - m_i).
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    cand = np.empty((n, len(pairs) + 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (i, j) in enumerate(pairs):
            dm = slope[:, j] - slope[:, i]
            x = (intercept[:, i] - intercept[:, j]) / dm
            # Parallel lines (no crossing) or crossings outside [0, lam_max] carry no
            # information about the decision inside the range: clamp them onto the
            # range boundary, which just creates a zero-width interval there.
            x = np.where(np.isfinite(x), x, lam_max)
            cand[:, k] = np.clip(x, 0.0, lam_max)
    cand[:, -2] = 0.0
    cand[:, -1] = lam_max
    cand.sort(axis=1)
    lo = cand[:, :-1]
    hi = cand[:, 1:]
    mids = 0.5 * (lo + hi)  # (n, 7)

    vals = intercept[:, None, :] + slope[:, None, :] * mids[..., None]  # (n, 7, 4)
    best_paid = np.max(np.where(paid_mask_rows[:, None, :], vals, -np.inf), axis=-1)
    best_free = np.max(np.where(~paid_mask_rows[:, None, :], vals, -np.inf), axis=-1)
    paid = best_paid > best_free  # (n, 7)

    if check_argmax_agreement:
        v0 = V0_int[:, None, :] + V0_slope[:, None, :] * mids[..., None]
        v1 = V1_int[:, None, :] + V1_slope[:, None, :] * mids[..., None]
        scale0 = np.maximum(np.abs(v0).max(axis=-1, keepdims=True), 1.0)
        scale1 = np.maximum(np.abs(v1).max(axis=-1, keepdims=True), 1.0)
        arg0 = v0 >= v0.max(axis=-1, keepdims=True) - agreement_tol * scale0
        arg1 = v1 >= v1.max(axis=-1, keepdims=True) - agreement_tol * scale1
        agree = np.any(arg0 & arg1, axis=-1)
        assert np.all(agree), (
            "closed-form self-consistency failure: the policy maximizing V(0, lam) "
            "does not also maximize V(1, lam) on some linear piece"
        )

    upper_if_paid = np.where(paid, hi, -np.inf)
    return upper_if_paid.max(axis=1)


def bellman_gaps_closed_form(
    s: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    gamma: float,
    lam: float,
    cost_convert: float,
    cost_maintain: float,
):
    """Same outputs as lagrangian_index.bellman_gaps_and_costs, but built from the
    exact V*(., lam) instead of a value-iteration approximation."""
    V0, V1 = optimal_values(p01, p10, w, gamma, cost_convert, cost_maintain, lam)
    a = np.asarray(p01, dtype=float)
    b = np.asarray(p10, dtype=float)
    w = np.asarray(w, dtype=float)
    q0_none = gamma * ((1.0 - a) * V0 + a * V1)
    q0_convert = -lam * cost_convert + gamma * V1
    q1_none = w + gamma * ((1.0 - b) * V1 + b * V0)
    q1_maintain = w - lam * cost_maintain + gamma * V1
    is_active = np.asarray(s).astype(bool)
    gap = np.where(is_active, q1_maintain - q1_none, q0_convert - q0_none)
    cost_if_paid = np.where(is_active, cost_maintain, cost_convert).astype(float)
    return gap, cost_if_paid


def solve_lambda_closed_form(
    s: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    w: np.ndarray,
    budget: float,
    gamma: float,
    cost_convert: float,
    cost_maintain: float,
    lam_max: float | None = None,
    return_indices: bool = False,
):
    """Drop-in for lagrangian_index.solve_lambda_bisection: O(n log n) budget
    matching by sorting arms on their closed-form index.

    Returns (lam_star, action_is_paid, cost, gap, cost_if_paid) with the same
    shapes/meanings as the bisection (gap is the exact Bellman gap at lam_star,
    computed from the closed-form V*), plus the (n,) index array as a 6th element
    when return_indices=True.

    Matching rule: sort arms by lam*_v descending, skip arms with lam*_v <= 0
    (never paid at any positive price, or paid only at lam=0 where the strict rule
    already says NONE), and take the longest prefix whose cumulative paid cost is
    <= budget. lam_star is 0.0 if every positive-index arm fits (the bisection's
    "nothing to bisect" branch), otherwise the index of the first arm that does
    not fit; action_is_paid is then exactly {lam*_v > lam_star}, which is what the
    bisection converges to from above (strict rule, so arms tied at lam_star are
    NOT paid -- same as the bisection, which lands on a price a hair above the tie).
    Any arm after the first non-fitting one that would still fit is deliberately
    left for the caller's leftover-budget fill, exactly as with the bisection.
    """
    idx = node_indices(s, p01, p10, w, gamma, cost_convert, cost_maintain, lam_max=lam_max)
    is_active = np.asarray(s).astype(bool)
    cost_if_paid = np.where(is_active, cost_maintain, cost_convert).astype(float)

    eligible = idx > 0.0
    order = np.argsort(-np.where(eligible, idx, -np.inf), kind="stable")
    order = order[eligible[order]]
    cum = np.cumsum(cost_if_paid[order])
    fits = cum <= budget + 1e-9
    if fits.all():
        lam_star = 0.0
    else:
        first_bad = int(np.argmin(fits))  # first False (prefix is all True)
        lam_star = float(idx[order[first_bad]])

    action_is_paid = idx > lam_star
    cost = np.where(action_is_paid, cost_if_paid, 0.0)
    gap, _ = bellman_gaps_closed_form(s, p01, p10, w, gamma, lam_star, cost_convert, cost_maintain)
    if return_indices:
        return lam_star, action_is_paid, cost, gap, cost_if_paid, idx
    return lam_star, action_is_paid, cost, gap, cost_if_paid
