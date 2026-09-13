"""Tests for the population_weighted toggle in im_lab.fairness.group_welfare_weights
(and its wiring through MFBWIFair/run_mf_bwi_fair).

See im_lab/fairness.py's module docstring ("Population-weighted vs.
population-unweighted (egalitarian) welfare") for the full derivation and
motivation this validates.
"""

from __future__ import annotations

import numpy as np

from im_lab import graphs
from im_lab.fairness import group_welfare_weights
from im_lab.mf_bwi_fair import run_mf_bwi_fair
from im_lab.simulator import count_active


def test_population_weighted_true_scales_with_group_size():
    """Two groups with IDENTICAL u_g but DIFFERENT N_g must get weights in the
    fixed N_g1/N_g2 ratio when population_weighted=True (the default / prior
    behavior) -- for both the alpha_fair==0 branch and the alpha_fair!=0 branch."""
    group_sizes = {"a": 20, "b": 60}
    u_g = {"a": 0.5, "b": 0.5}

    for alpha_fair in (0.0, -2.0, 0.5):
        w = group_welfare_weights(group_sizes, u_g, alpha_fair, population_weighted=True)
        ratio = w["b"] / w["a"]
        assert abs(ratio - (group_sizes["b"] / group_sizes["a"])) < 1e-9, (
            f"alpha_fair={alpha_fair}: expected population-weighted ratio "
            f"{group_sizes['b'] / group_sizes['a']}, got {ratio}"
        )
        # And the two groups' weights must NOT be equal (size actually matters).
        assert w["a"] != w["b"]


def test_population_weighted_false_ignores_group_size():
    """Two groups with IDENTICAL u_g but DIFFERENT N_g must get IDENTICAL weights
    when population_weighted=False, for both the alpha_fair==0 and alpha_fair!=0
    branches -- this is the whole point of the egalitarian/unweighted form."""
    group_sizes = {"a": 20, "b": 60}
    u_g = {"a": 0.5, "b": 0.5}

    for alpha_fair in (0.0, -2.0, 0.5):
        w = group_welfare_weights(group_sizes, u_g, alpha_fair, population_weighted=False)
        assert abs(w["a"] - w["b"]) < 1e-12, (
            f"alpha_fair={alpha_fair}: population_weighted=False weights should be "
            f"size-independent, got a={w['a']}, b={w['b']}"
        )


def test_population_weighted_false_still_depends_on_reach():
    """Sanity: population_weighted=False should not be a no-op -- it must still
    react to differing u_g (a lower-reach group gets a higher weight)."""
    group_sizes = {"a": 20, "b": 60}
    u_g = {"a": 0.3, "b": 0.95}

    w = group_welfare_weights(group_sizes, u_g, 0.0, population_weighted=False)
    assert w["a"] > w["b"]  # group a (lower reach) gets the higher weight

    w_eq = group_welfare_weights(group_sizes, u_g, 0.0, population_weighted=True)
    assert w_eq["a"] > w_eq["b"]


def test_group_welfare_weights_default_matches_population_weighted_true():
    """population_weighted defaults to True -- omitting it must be identical to
    passing it explicitly (backward compatibility / additive-only change)."""
    group_sizes = {0: 20, 1: 40, 2: 60}
    u_g = {0: 0.5, 1: 0.7, 2: 0.9}
    for alpha_fair in (1.0, 0.0, -3.0):
        w_default = group_welfare_weights(group_sizes, u_g, alpha_fair)
        w_explicit = group_welfare_weights(group_sizes, u_g, alpha_fair, population_weighted=True)
        assert w_default == w_explicit


# --- End-to-end empirical validation on the project's actual 3-group SBM graph ---
# Constants replicated from experiments/common.py (that module itself is untouched).
_SIZES = [20, 40, 60]
_P_IN = 0.07
_P_OUT = 0.008
_P_PLUS_RANGE = (0.05, 0.15)
_TOPOLOGY_SEED = 42
_B = 100.0
_T = 30
_DEFAULT_BETA = 0.15
_Q_RANGE = (0.05, 0.05)
_ALPHA_NEUTRAL = 0.0  # proportional fairness -- the neutral/default setting
_N_TRIALS = 10  # kept modest to keep the test suite fast; validate_population_weighting.py uses 15


def _group_reach_fractions(trajectory, group_of, group_sizes):
    groups = sorted(group_sizes)
    sums = {g: 0.0 for g in groups}
    n_rounds = len(trajectory) - 1
    for state in trajectory[1:]:
        counts = {g: 0 for g in groups}
        for v, active in state.items():
            if active:
                counts[group_of[v]] += 1
        for g in groups:
            sums[g] += counts[g] / group_sizes[g]
    return {g: sums[g] / n_rounds for g in groups}


def _run(population_weighted: bool, trial_seed: int):
    G = graphs.stochastic_block_model_graph(_SIZES, p_in=_P_IN, p_out=_P_OUT, seed=_TOPOLOGY_SEED)
    graphs.assign_true_parameters(G, p_plus_range=_P_PLUS_RANGE, q_range=_Q_RANGE, seed=trial_seed)
    group_of = graphs.group_of_map(G)
    group_sizes = graphs.group_sizes(G)

    result = run_mf_bwi_fair(
        G, true_beta=_DEFAULT_BETA, T=_T, budget=_B, alpha_fair=_ALPHA_NEUTRAL,
        seed=trial_seed, population_weighted=population_weighted,
    )
    reach = _group_reach_fractions(result["trajectory"], group_of, group_sizes)
    total_spread = float(np.mean([count_active(s) for s in result["trajectory"][1:]]))
    return reach, total_spread, group_sizes


def test_egalitarian_weighting_improves_smallest_group_reach_at_neutral_alpha():
    """The actual empirical claim: at a NEUTRAL alpha_fair=0.0 (no need to push
    alpha to an extreme), switching from population_weighted=True (the old,
    population-weighted Rahmattalabi et al. form) to population_weighted=False
    (the new egalitarian form) gives the smallest group (group 0, N=20)
    meaningfully higher realized reach on this project's actual 3-group (20/40/60)
    SBM graph -- and does so without a large total-spread cost."""
    reach_true = {0: [], 1: [], 2: []}
    reach_false = {0: [], 1: [], 2: []}
    spread_true = []
    spread_false = []

    for trial in range(_N_TRIALS):
        seed = 2000 + trial
        r_true, s_true, group_sizes = _run(True, seed)
        r_false, s_false, _ = _run(False, seed)
        for g in (0, 1, 2):
            reach_true[g].append(r_true[g])
            reach_false[g].append(r_false[g])
        spread_true.append(s_true)
        spread_false.append(s_false)

    smallest_g = min(group_sizes, key=lambda g: group_sizes[g])
    assert smallest_g == 0

    mean_reach_true_small = float(np.mean(reach_true[smallest_g]))
    mean_reach_false_small = float(np.mean(reach_false[smallest_g]))

    # Core claim: egalitarian weighting meaningfully raises the smallest group's
    # realized reach at this same neutral alpha (not just noise-level).
    assert mean_reach_false_small > mean_reach_true_small + 0.05, (
        f"population_weighted=False did not meaningfully improve group0's reach: "
        f"True={mean_reach_true_small:.4f}, False={mean_reach_false_small:.4f}"
    )

    # ...and it must not come at a large total-spread cost: allow some give but
    # not a collapse (spread should stay within 10% of the population-weighted
    # baseline).
    mean_spread_true = float(np.mean(spread_true))
    mean_spread_false = float(np.mean(spread_false))
    assert mean_spread_false >= 0.9 * mean_spread_true, (
        f"population_weighted=False cost too much total spread: "
        f"True={mean_spread_true:.3f}, False={mean_spread_false:.3f}"
    )
