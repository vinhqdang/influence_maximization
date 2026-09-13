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

--- Population-weighted vs. population-unweighted (egalitarian) welfare ---
The form above -- W_alpha(u) = sum_c N_c * u_c^alpha / alpha -- is POPULATION-
WEIGHTED: each group's contribution to total welfare scales with its size N_c, so
it is the right objective when the goal is genuinely aggregate/utilitarian-style
social welfare, i.e. when a unit of reach in a bigger group is legitimately worth
more simply because it reaches more people.

That population weighting has a side effect worth naming explicitly: it actively
works AGAINST protecting a small disadvantaged group. Since w_g = N_g * u_g^(alpha
- 1), group g's weight only exceeds group g''s once u_g'/u_g > (N_g'/N_g)^(1/(1-
alpha)) -- for alpha=0 that is simply u_g'/u_g > N_g'/N_g. On this project's own
3-group (20/40/60) SBM graph, group0 (N=20) stays under-served (reach ~0.45-0.6)
across most of the alpha_fair sweep while group2 (N=60) stays ~0.85-0.93: for
group0's weight to overtake group2's at alpha=0 requires u_g2/u_g0 > N_g2/N_g0 = 3,
but the realized ratio there was only ~1.5 -- so the population-weighted form does
not even start prioritizing group0 until alpha is pushed very negative. In other
words, N_g-dilution means a small group has to be reached proportionally MUCH less
than a large group before the weighting formula reacts at all.

The alternative implemented here (population_weighted=False) is the standard
EGALITARIAN/Rawlsian-style isoelastic welfare, counting each GROUP once regardless
of its population -- a well-established alternative form in the social-welfare and
fair-division literature, not something invented for this codebase:

    W_alpha(u) = sum_c u_c^alpha / alpha        (alpha != 0)
    W_alpha(u) = sum_c log(u_c)                  (alpha == 0)

giving the marginal weight

    w_g = u_g^(alpha - 1)      (alpha != 0)
    w_g = 1 / u_g              (alpha == 0)

i.e. exactly Rahmattalabi et al.'s formula with the N_g factor dropped from both
branches. Which form is "correct" depends on what fairness is meant to mean here:
- population_weighted=True is right when the objective genuinely IS aggregate
  social welfare -- more people benefiting is supposed to count for more, and a
  large group's members are not being penalized for existing in a large group.
- population_weighted=False (egalitarian) is right when the goal is each GROUP
  being treated fairly AS A GROUP regardless of its size -- which is what "don't
  let the small group be neglected" actually calls for, since under the
  population-weighted form a small group can be badly under-served and still not
  move the weights, purely because there are few of its members to matter in the
  aggregate.
"""

from __future__ import annotations


def group_welfare_weights(
    group_sizes: dict,
    u_g: dict,
    alpha_fair: float,
    u_floor: float = 1e-3,
    population_weighted: bool = True,
) -> dict:
    """Per-group marginal welfare weight w_g = dW_alpha/du_g (see module docstring).

    group_sizes: group id -> N_g.
    u_g: group id -> current time-averaged realized reach fraction in (0, 1] (may be
        missing/0 for a group with no observed history yet; floored at u_floor).
    alpha_fair: the isoelastic inequality-aversion exponent (roughly (-inf, 1];
        1 = utilitarian/size-only weighting, 0 = proportional fairness, more
        negative = more leximin-like, always favoring the worse-off group more).
    population_weighted: True (default, preserves prior behavior) uses
        Rahmattalabi et al.'s population-weighted form (w_g scales with N_g);
        False uses the population-unweighted/egalitarian form (w_g depends only
        on u_g, not on group size) -- see the module docstring section above for
        when each is the appropriate choice.
    """
    weights: dict = {}
    for g, size in group_sizes.items():
        u = max(u_g.get(g, u_floor), u_floor)
        n_factor = size if population_weighted else 1.0
        if alpha_fair == 0.0:
            weights[g] = n_factor / u
        else:
            weights[g] = n_factor * (u ** (alpha_fair - 1.0))
    return weights
