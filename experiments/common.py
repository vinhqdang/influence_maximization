"""Shared configuration and helpers for the MF-BWI-Fair vs. one-shot-baseline
comparison study (see run_comparison.py).

Design notes on the experimental setup (see RESULTS.md for the full writeup):

- A single SBM graph topology (3 groups, sizes 20/40/60 -- deliberately unequal
  so the fairness floor has something to bite on) is built once and reused for
  every trial; only the *parameter draw* (p_plus per edge, q per node) and the
  simulation noise vary across trials, each keyed off an independent RNG seed.
- Graph density (p_in/p_out) and p_plus range were hand-tuned (see the
  exploration in this task's session) so that the network neither saturates to
  ~100% active nor stays too sparse to cascade at all -- both extremes would
  hide the effects we want to sweep (backfire/recovery erosion, fairness
  tradeoff).
- Budget B=100 and one-shot seed count k=20 are matched so that k*5 == B: the
  one-shot baselines get to spend exactly one round's worth of MF-BWI-Fair's
  budget, no more, so neither side gets a head start.
- graphs.assign_true_parameters draws p_plus (per edge) BEFORE q (per node)
  from the same RNG stream, in a fixed edge/node iteration order. That means
  for a fixed seed, the p_plus draw is byte-identical regardless of q_range.
  We exploit this: the one-shot baselines' seed selection (celf_greedy,
  fair_welfare_greedy) only ever looks at p_plus, so it is computed exactly
  ONCE per trial (not once per sweep, not once per swept parameter value) and
  reused everywhere -- this is the single biggest runtime saving in the whole
  study (fair_welfare_greedy's naive non-lazy greedy is the slowest step).
"""

from __future__ import annotations

import time

import numpy as np

from im_lab import graphs
from im_lab.baselines.fair_greedy import fair_welfare_greedy
from im_lab.baselines.kkt_greedy import celf_greedy
from im_lab.mf_bwi_fair import run_mf_bwi_fair
from im_lab.simulator import (
    count_active,
    initial_state,
    seed_then_none_actions,
    simulate,
    true_params_from_graph,
)

# --- Fixed graph / budget / horizon -----------------------------------------
SIZES = [20, 40, 60]  # 3 groups, unequal on purpose (120 nodes total)
P_IN = 0.07
P_OUT = 0.008
P_PLUS_RANGE = (0.05, 0.15)
TOPOLOGY_SEED = 42

B = 100.0  # per-round budget: 20 CONVERTs (16.7% of nodes) or up to 100 MAINTAINs
T = 30
K = 20  # k * ACTION_COST[CONVERT] (5) == B: one-shot baselines get one round's budget

N_TRIALS = 15
CELF_NUM_SIMS = 40
FAIR_NUM_SIMS = 20

# --- Sweep defaults ----------------------------------------------------------
DEFAULT_BETA = 0.15
DEFAULT_Q = 0.05  # default used while sweeping beta
DEFAULT_ALPHA_FAIR = 0.3

BETA_SWEEP_Q = 0.05
Q_SWEEP_BETA = 0.15
ALPHA_SWEEP_BETA = 0.15
ALPHA_SWEEP_Q = 0.1

BETA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
Q_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4]
ALPHA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]

ALGOS = ["kkt_greedy", "fair_greedy", "mf_bwi_fair"]

# A single master RNG seeds every random draw made anywhere in this study, so
# the whole comparison is reproducible end to end from one seed.
_MASTER_RNG = np.random.default_rng(0)


def next_seed() -> int:
    return int(_MASTER_RNG.integers(0, 2**31 - 1))


def build_graph():
    """The one fixed graph topology reused by every trial in every sweep."""
    return graphs.stochastic_block_model_graph(SIZES, p_in=P_IN, p_out=P_OUT, seed=TOPOLOGY_SEED)


def time_avg_spread(trajectory: list) -> float:
    """sigma(S): mean active-node count over the T post-action rounds."""
    return float(np.mean([count_active(s) for s in trajectory[1:]]))


def group_reach_fractions(trajectory: list, group_of: dict, group_sizes: dict) -> dict:
    """Time-averaged, per-group active fraction (own-group-size-normalized)."""
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


def compute_baseline_seed_sets(G, trial_seed: int) -> dict:
    """Compute the celf-greedy and fair-greedy seed sets ONCE for this trial.

    Uses an arbitrary fixed q_range (irrelevant: p_plus draws never depend on
    q_range -- see module docstring) so the resulting seed sets are valid for
    every sweep/param value that reuses this trial's parameter draw.
    """
    graphs.assign_true_parameters(
        G, p_plus_range=P_PLUS_RANGE, q_range=(BETA_SWEEP_Q, BETA_SWEEP_Q), seed=trial_seed
    )
    p_plus, _q = true_params_from_graph(G)
    group_of = graphs.group_of_map(G)

    seeds_kkt, _ = celf_greedy(
        G, p_plus, k=K, num_sims=CELF_NUM_SIMS, rng=np.random.default_rng(next_seed())
    )
    seeds_fair, _w, _reach = fair_welfare_greedy(
        G, p_plus, k=K, group_of=group_of, num_sims=FAIR_NUM_SIMS,
        rng=np.random.default_rng(next_seed()),
    )
    return {"kkt_greedy": list(seeds_kkt), "fair_greedy": list(seeds_fair)}


def run_baseline_forward(G, seed_set, true_beta, q_range, trial_seed, T=T) -> tuple[list, float]:
    """Re-stamp the graph's true params (q_range for this param value, same
    trial_seed => identical p_plus, different q) and run the seed-then-none
    action sequence through the real sequential simulator."""
    t0 = time.perf_counter()
    graphs.assign_true_parameters(G, p_plus_range=P_PLUS_RANGE, q_range=q_range, seed=trial_seed)
    p_plus, q = true_params_from_graph(G)
    acts = seed_then_none_actions(G, seed_set, T)
    init = initial_state(G, seed_set)
    rng = np.random.default_rng(next_seed())
    traj, _obs = simulate(G, init, p_plus, q, true_beta, acts, rng)
    runtime = time.perf_counter() - t0
    return traj, runtime


def run_mfbwi_forward(G, true_beta, budget, alpha_fair, q_range, trial_seed, T=T) -> tuple[list, float]:
    graphs.assign_true_parameters(G, p_plus_range=P_PLUS_RANGE, q_range=q_range, seed=trial_seed)
    t0 = time.perf_counter()
    result = run_mf_bwi_fair(
        G, true_beta=true_beta, T=T, budget=budget, alpha_fair=alpha_fair, seed=next_seed()
    )
    runtime = time.perf_counter() - t0
    return result["trajectory"], runtime
