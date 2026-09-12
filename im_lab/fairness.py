"""Isoelastic (CES) welfare reweighting shared by MF-BWI-Fair.

Replaces the old per-group budget-floor heuristic (empirically shown, in a prior
experiment, to have only a weak effect) with the objective form from:

    Rahmattalabi, A., Vayanos, P., Fulginiti, A., Rice, E., Wilder, B., Yadav, A.,
    & Tambe, M. (2021). "Fair Influence Maximization: A Welfare Optimization
    Approach." AAAI 2021.

They maximize an isoelastic welfare function over per-group expected reach u_c:

    W_alpha(u) = sum_c N_c * u_c^alpha / alpha         for alpha < 1, alpha != 0
    W_alpha(u) = sum_c N_c * log(u_c)                  for alpha == 0

alpha is a single inequality-aversion parameter: alpha -> 1 is utilitarian/no
fairness weighting (see below for the precise sense in which this is true here),
alpha == 0 is proportional fairness, and alpha -> -infinity approaches leximin
(all the weight concentrates on whichever group is currently worst-off).

We adopt Rahmattalabi et al.'s OBJECTIVE FORM (this reweighting), not their
(1-1/e) submodularity guarantee -- that guarantee needs u_c(A) itself monotone
submodular in the seed set A, which their proof gets from the classical
*progressive* IC model; it does not transfer to our non-progressive/backfire
model, and no equivalent guarantee is claimed or proven here.

--- From welfare objective to a per-round reward weight (this module) ---
MF-BWI-Fair does not re-solve a combinatorial welfare-maximization problem each
round; instead it takes the standard (and standard-to-derive) marginal/subgradient
view of concave welfare maximization: the marginal value, to the OVERALL welfare
objective W_alpha(u), of one more unit of expected reach in group g is

    dW_alpha/du_g = N_g * u_g^(alpha - 1)                            (alpha != 0)
    dW_alpha/du_g = N_g / u_g                                        (alpha == 0)

Reweighting each group's per-node reward by exactly this marginal utility --
"water-filling" the budget so that the group whose current reach u_g is lowest (and
whose marginal utility is therefore highest, since u^(alpha-1) is decreasing in u
for alpha<1) gets a disproportionately higher reward-per-active-node -- is a
standard subgradient scheme for concave welfare maximization; it is not a new
mechanism invented for this codebase. group_welfare_weights below computes exactly
w_g = dW_alpha/du_g, used as the per-node reward coefficient r_v(s) = w_g * s inside
each node's Lagrangian MDP (im_lab/lagrangian_index.py).

u_g is a small, epsilon-floored (avoids the alpha==0 and alpha<1 singularities at
u_g=0) time-averaged realized reach fraction for group g, tracked by
MFBWIFair.update_group_reach; see im_lab/mf_bwi_fair.py for how it is bootstrapped
and updated round to round.

--- On alpha=1 not quite meaning "no fairness weighting" ---
At alpha=1, w_g = N_g * u_g^0 = N_g: every group's nodes get a reward proportional
to raw GROUP SIZE only, not to the group's current (inverse) reach -- i.e. alpha=1
is the utilitarian baseline (all active nodes count equally regardless of group,
once you cancel the shared N_g/N_g proportionality across equally-sized groups),
not literally "u_g plays no role" (which would require reading u_g^0=1 as "u_g is
irrelevant", true, but the group-size factor N_g is still present and NOT
current-reach-dependent). Smaller and negative alpha increasingly favor
currently-under-reached groups.
"""

from __future__ import annotations


def group_welfare_weights(
    group_sizes: dict,
    u_g: dict,
    alpha_fair: float,
    u_floor: float = 1e-3,
) -> dict:
    """Per-group marginal welfare weight w_g = dW_alpha/du_g (see module docstring).

    group_sizes: group id -> N_g.
    u_g: group id -> current time-averaged realized reach fraction in (0, 1] (may be
        missing/0 for a group with no observed history yet; floored at u_floor).
    alpha_fair: the isoelastic inequality-aversion exponent (roughly (-inf, 1];
        1 = utilitarian/size-only weighting, 0 = proportional fairness, more
        negative = more leximin-like, always favoring the worse-off group more).
    """
    weights: dict = {}
    for g, size in group_sizes.items():
        u = max(u_g.get(g, u_floor), u_floor)
        if alpha_fair == 0.0:
            weights[g] = size / u
        else:
            weights[g] = size * (u ** (alpha_fair - 1.0))
    return weights
