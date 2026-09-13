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

- A FOURTH algorithm, repeated_greedy (im_lab.baselines.repeated_greedy), is
  also included: it acts EVERY round under the same per-round budget B as
  mf_bwi_fair (so it is NOT a one-shot baseline), using plain classical
  cost-effective greedy logic with the TRUE p_plus/q/beta handed to it
  directly (no Bayesian estimation, no mean-field, no fairness floors). It
  exists to isolate whether mf_bwi_fair's advantage over the one-shot
  baselines comes from its specific machinery or simply from "gets to act
  every round" -- see im_lab/baselines/repeated_greedy.py's module docstring
  for the full design rationale and compute-budget discussion, and
  RESULTS.md for the four-way comparison this enables. Like mf_bwi_fair (and
  unlike the one-shot baselines), it must be recomputed at every swept
  (beta, q) value since its forward dynamics depend on them; like the
  one-shot baselines (and unlike mf_bwi_fair), it does not use alpha_fair at
  all, so in the alpha sweep it is computed once per trial and replicated
  across alpha_fair values exactly like kkt_greedy/fair_greedy.

- A FIFTH algorithm, imm (im_lab.baselines.imm), is a faithful
  reimplementation of a real, named, published algorithm -- Tang, Shi & Xiao's
  IMM (SIGMOD 2015), the standard near-linear-time (1-1/e-epsilon)-approximate
  classical IM algorithm -- included so the comparison has at least one
  baseline that is not something built from scratch for this project. Like
  kkt_greedy/fair_greedy it is a one-shot seed-selection baseline under plain
  progressive IC (no backfire/recovery/fairness/uncertainty machinery of its
  own), computed once per trial from p_plus alongside the other one-shot
  baselines and run forward through the same sequential simulator. Since it
  targets the same classical objective as kkt_greedy, it is expected to
  degrade similarly under backfire/recovery -- its value here is being a
  citable, published point of comparison, not a qualitatively different
  competitor.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from im_lab import graphs
from im_lab.baselines.fair_greedy import fair_welfare_greedy
from im_lab.baselines.imm import imm_select
from im_lab.baselines.kkt_greedy import celf_greedy
from im_lab.baselines.repeated_greedy import run_repeated_greedy
from im_lab.baselines.robust_kempe import robust_select
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
# NOTE: alpha_fair's MEANING changed with the Lagrangian/welfare redesign (see
# im_lab/fairness.py) -- it used to be a per-group budget-FLOOR fraction in
# [0,1] (0=no floor, 1=fully proportional floor); it is now the isoelastic
# (CES) welfare exponent, roughly in (-inf, 1]: 1=utilitarian/size-only
# weighting, 0=proportional fairness, more negative=more leximin-like. The
# sweep values and the fixed default below are chosen for the NEW semantics,
# not reused from the old floor-fraction sweep.
DEFAULT_BETA = 0.15
DEFAULT_Q = 0.05  # default used while sweeping beta
DEFAULT_ALPHA_FAIR = 0.0  # proportional fairness, a neutral default

BETA_SWEEP_Q = 0.05
Q_SWEEP_BETA = 0.15
ALPHA_SWEEP_BETA = 0.15
ALPHA_SWEEP_Q = 0.1

BETA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
Q_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4]
ALPHA_VALUES = [1.0, 0.5, 0.0, -2.0, -8.0]  # utilitarian -> increasingly leximin-like

ALGOS = ["kkt_greedy", "fair_greedy", "imm", "robust_kempe", "mf_bwi_fair", "repeated_greedy"]
IMM_EPSILON = 0.5  # IMM's approximation-guarantee parameter (see baselines/imm.py)

# He & Kempe's Saturate Greedy is a BICRITERIA algorithm: it may return more
# than K seeds (up to beta*K = (1+ln|Sigma|+ln(3/gamma))*K, see robust_kempe.py)
# in exchange for its worst-case-across-scenarios guarantee. We do not cap
# this -- the actual seed count/cost it uses is recorded and reported as-is,
# not silently truncated to K (that would void its own guarantee). gamma=0.3
# and a modest num_sims keep the binary search's per-iteration Monte-Carlo
# cost tractable at this graph size; both are tuning choices, not from the
# paper (see robust_kempe.py's own docstring for what IS from the paper).
ROBUST_KEMPE_GAMMA = 0.3
ROBUST_KEMPE_NUM_SIMS = 60

# Every random draw in this study is seeded from context_seed(...) below, NOT
# from a shared sequentially-consumed counter. A prior version used a single
# _MASTER_RNG.integers() call site (next_seed()) shared across every algorithm
# and sweep -- which meant the seed any given (sweep, param, algorithm, trial)
# cell received depended on the exact order and count of every next_seed()
# call that happened before it anywhere in the script. That made results
# fragile to unrelated changes (e.g. editing one algorithm silently shifted
# every other algorithm's random draws) and, concretely, caused two separate
# runs of this script to produce different numbers for algorithms whose own
# code had NOT changed, when something upstream in the call sequence had. See
# the trial_seed / context_seed usage below: every seed now depends ONLY on
# its own semantic identity (which sweep, which parameter value, which
# algorithm, which trial), so it is the same regardless of what else runs
# before or after it, and safe to compute independently/out of order.
ROOT_SEED = 0


def context_seed(*parts) -> int:
    """Deterministic seed derived only from `parts` (order-independent w.r.t.
    anything else in the program) -- ints are used as-is; anything else is
    turned into a stable integer via SHA-256 of its repr (NOT Python's
    built-in hash(), which is randomized per-process for str/bytes unless
    PYTHONHASHSEED is fixed, and would silently reintroduce the same
    run-to-run irreproducibility this replaces)."""
    ints = []
    for p in parts:
        if isinstance(p, (int, np.integer, bool)):
            ints.append(int(p))
        else:
            digest = hashlib.sha256(repr(p).encode("utf-8")).digest()[:8]
            ints.append(int.from_bytes(digest, "little"))
    return int(np.random.SeedSequence([ROOT_SEED, *ints]).generate_state(1)[0])


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
        G, p_plus, k=K, num_sims=CELF_NUM_SIMS,
        rng=np.random.default_rng(context_seed("kkt_select", trial_seed)),
    )
    seeds_fair, _w, _reach = fair_welfare_greedy(
        G, p_plus, k=K, group_of=group_of, num_sims=FAIR_NUM_SIMS,
        rng=np.random.default_rng(context_seed("fair_select", trial_seed)),
    )
    seeds_imm = imm_select(
        G, p_plus, k=K, epsilon=IMM_EPSILON,
        rng=np.random.default_rng(context_seed("imm_select", trial_seed)),
    )

    # He & Kempe's scenario set: the two "all-low" / "all-high" corners of the
    # per-edge interval [P_PLUS_RANGE[0], P_PLUS_RANGE[1]] that the true p_plus
    # is actually drawn from -- i.e. robust_kempe is handed exactly the same
    # RANGE information the experiment design already assumes is "publicly
    # known" (it's a fixed constant of this study), never the realized p_plus
    # itself. This is the fair, apples-to-apples counterpart to MF-BWI-Fair,
    # which starts with no information beyond a flat Beta(1,1) prior and only
    # learns the realized values via Bayesian updating over rounds.
    edges = list(G.edges())
    scenario_low = {e: P_PLUS_RANGE[0] for e in edges}
    scenario_high = {e: P_PLUS_RANGE[1] for e in edges}
    seeds_robust = robust_select(
        G, [scenario_low, scenario_high], k=K,
        gamma=ROBUST_KEMPE_GAMMA, num_sims=ROBUST_KEMPE_NUM_SIMS,
        rng=np.random.default_rng(context_seed("robust_select", trial_seed)),
    )

    return {
        "kkt_greedy": list(seeds_kkt),
        "fair_greedy": list(seeds_fair),
        "imm": list(seeds_imm),
        "robust_kempe": list(seeds_robust),
    }


def run_baseline_forward(G, seed_set, true_beta, q_range, trial_seed, algo, sweep, param, T=T) -> tuple[list, float]:
    """Re-stamp the graph's true params (q_range for this param value, same
    trial_seed => identical p_plus, different q) and run the seed-then-none
    action sequence through the real sequential simulator.

    `algo`/`sweep`/`param` identify this specific call for context_seed, so
    forward-simulation noise is deterministic per (algo, sweep, param, trial)
    -- independent draws across swept values as before, but no longer
    dependent on call order elsewhere in the script."""
    t0 = time.perf_counter()
    graphs.assign_true_parameters(G, p_plus_range=P_PLUS_RANGE, q_range=q_range, seed=trial_seed)
    p_plus, q = true_params_from_graph(G)
    acts = seed_then_none_actions(G, seed_set, T)
    init = initial_state(G, seed_set)
    rng = np.random.default_rng(context_seed("forward", algo, sweep, param, trial_seed))
    traj, _obs = simulate(G, init, p_plus, q, true_beta, acts, rng)
    runtime = time.perf_counter() - t0
    return traj, runtime


def run_mfbwi_forward(G, true_beta, budget, alpha_fair, q_range, trial_seed, sweep, param, T=T) -> tuple[list, float]:
    graphs.assign_true_parameters(G, p_plus_range=P_PLUS_RANGE, q_range=q_range, seed=trial_seed)
    t0 = time.perf_counter()
    result = run_mf_bwi_fair(
        G, true_beta=true_beta, T=T, budget=budget, alpha_fair=alpha_fair,
        seed=context_seed("mfbwi_forward", sweep, param, trial_seed),
    )
    runtime = time.perf_counter() - t0
    return result["trajectory"], runtime


def run_repeatedgreedy_forward(G, true_beta, budget, q_range, trial_seed, sweep, param, T=T) -> tuple[list, float]:
    """Full T-round repeated_greedy run (acts every round, no alpha_fair --
    see module docstring above and im_lab/baselines/repeated_greedy.py)."""
    graphs.assign_true_parameters(G, p_plus_range=P_PLUS_RANGE, q_range=q_range, seed=trial_seed)
    t0 = time.perf_counter()
    result = run_repeated_greedy(
        G, true_beta=true_beta, T=T, budget=budget,
        seed=context_seed("repeated_forward", sweep, param, trial_seed),
    )
    runtime = time.perf_counter() - t0
    return result["trajectory"], runtime
