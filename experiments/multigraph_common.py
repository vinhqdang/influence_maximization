"""Parameterized version of experiments/common.py's comparison harness, generalized
to run over ANY graph/config pair instead of the one fixed 120-node SBM graph.

This module is a SEPARATE, ADDITIONAL harness -- it does not import from or modify
experiments/common.py or experiments/run_comparison.py, and running the existing
single-graph study (python experiments/run_comparison.py) is completely unaffected
by anything here. It exists for experiments/run_multigraph_validation.py, which
applies the same six-algorithm comparison structure to a real-world graph and
several additional synthetic graphs (see experiments/MULTIGRAPH_VALIDATION.md for
the write-up of what came out of that).

Design differences from common.py, and why:

- common.py's helper functions (`compute_baseline_seed_sets`,
  `run_baseline_forward`, ...) read several pieces of their configuration
  (P_PLUS_RANGE, K, CELF_NUM_SIMS, ...) directly off MODULE-LEVEL constants --
  fine for a single fixed graph/config, but it means those functions cannot be
  reused as-is for a second graph with different settings without either
  monkey-patching those constants (fragile, and would corrupt any concurrent
  use of common.py) or duplicating the whole file. Here the equivalent pieces
  of configuration are bundled into one `GraphConfig` dataclass instance and
  threaded through explicitly instead.
- A real graph's p_plus is NOT drawn from a random range every trial the way
  the synthetic graphs' is (see im_lab/graphs.py's
  `assign_weighted_cascade_probabilities` docstring) -- it is a fixed,
  deterministic function of the graph's own structure. `GraphConfig.p_plus_mode`
  ("range" vs "fixed") switches between `graphs.assign_true_parameters` (as
  common.py always uses) and `graphs.assign_weighted_cascade_probabilities` +
  `graphs.assign_recovery_rates` (q is still re-drawn per trial either way --
  real datasets don't ship with recovery rates any more than they ship with
  influence probabilities).
- robust_kempe's scenario set (the two "corners" of the uncertainty interval
  it is handed) is built from `p_plus_range`'s two endpoints in "range" mode,
  exactly as common.py does. In "fixed" mode there is no such range to take
  corners of, so the scenario set is instead {p_plus_as_computed, 0.5 *
  p_plus_as_computed} -- i.e. genuine uncertainty about whether the
  Weighted-Cascade point-estimate convention over- or under-states the real
  diffusion strength, rather than reusing the synthetic sweep's range-corner
  trick (there is no natural analogue of it here). This is a deliberate,
  documented modeling choice, not an attempt to reproduce common.py's exact
  robust_kempe semantics on a graph where the underlying assumption (p_plus
  drawn uniformly from a known range) does not hold.
"""

from __future__ import annotations

import csv
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable

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

ALGOS = ["kkt_greedy", "fair_greedy", "imm", "robust_kempe", "mf_bwi_fair", "repeated_greedy"]

ROOT_SEED = 12345  # deliberately different from common.py's ROOT_SEED=0, so that
# even though context_seed's mechanism is identical, this harness's random draws
# never collide bit-for-bit with common.py's (irrelevant for correctness, but
# avoids any appearance of accidentally sharing/reusing a run).


def context_seed(*parts) -> int:
    """Same deterministic-seed-from-semantic-identity mechanism as
    common.py's context_seed -- see that module's docstring for the full
    rationale (order-independence, SHA-256 instead of randomized hash())."""
    ints = []
    for p in parts:
        if isinstance(p, (int, np.integer, bool)):
            ints.append(int(p))
        else:
            digest = hashlib.sha256(repr(p).encode("utf-8")).digest()[:8]
            ints.append(int.from_bytes(digest, "little"))
    return int(np.random.SeedSequence([ROOT_SEED, *ints]).generate_state(1)[0])


@dataclass
class GraphConfig:
    """Everything needed to run the six-algorithm comparison on one graph."""

    name: str
    build_graph: Callable[[], "graphs.nx.DiGraph"]
    p_plus_mode: str  # "range" (assign_true_parameters) or "fixed" (weighted cascade)
    p_plus_range: tuple[float, float] = (0.05, 0.15)  # used when p_plus_mode == "range"

    K: int = 20
    B: float = 100.0
    T: int = 30
    N_TRIALS: int = 6

    CELF_NUM_SIMS: int = 40
    FAIR_NUM_SIMS: int = 20
    IMM_EPSILON: float = 0.5
    ROBUST_KEMPE_GAMMA: float = 0.3
    ROBUST_KEMPE_NUM_SIMS: int = 60

    # repeated_greedy's own rollout cost knobs (see
    # im_lab/baselines/repeated_greedy.py's module docstring): evaluating
    # every candidate node's NUM_SIMS x LOOKAHEAD_ROUNDS rollout every round is
    # the single most expensive part of this whole harness on a graph with
    # many edges (each rollout re-simulates one step of the REAL sequential
    # dynamics, which touches every in-edge of every currently-active node).
    # common.py's single 120-node graph is sparse enough that the library
    # defaults (2, 2) are affordable; several of the graphs here are not (see
    # MULTIGRAPH_VALIDATION.md), so this harness uses 1/1 by default --
    # documented as a scope reduction, not a claim that 1 rollout of 1 round
    # is what the algorithm "should" use.
    REPEATED_GREEDY_LOOKAHEAD: int = 1
    REPEATED_GREEDY_NUM_SIMS: int = 1

    DEFAULT_BETA: float = 0.15
    DEFAULT_ALPHA_FAIR: float = 0.0

    BETA_VALUES: tuple = (0.0, 0.15, 0.3, 0.5)
    Q_VALUES: tuple = (0.0, 0.1, 0.2, 0.4)
    ALPHA_VALUES: tuple = (1.0, 0.0, -4.0)

    BETA_SWEEP_Q: float = 0.05
    Q_SWEEP_BETA: float = 0.15
    ALPHA_SWEEP_BETA: float = 0.15
    ALPHA_SWEEP_Q: float = 0.1

    notes: str = ""


def _assign_params(G, config: GraphConfig, q_range: tuple[float, float], trial_seed: int):
    """Stamp this trial's true p_plus/q onto G, per config.p_plus_mode."""
    if config.p_plus_mode == "range":
        graphs.assign_true_parameters(G, p_plus_range=config.p_plus_range, q_range=q_range, seed=trial_seed)
    elif config.p_plus_mode == "fixed":
        graphs.assign_weighted_cascade_probabilities(G)  # deterministic, cheap to redo
        graphs.assign_recovery_rates(G, q_range=q_range, seed=trial_seed)
    else:
        raise ValueError(f"unknown p_plus_mode {config.p_plus_mode!r}")


def time_avg_spread(trajectory: list) -> float:
    return float(np.mean([count_active(s) for s in trajectory[1:]]))


def group_reach_fractions(trajectory: list, group_of: dict, group_sizes: dict) -> dict:
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


def compute_baseline_seed_sets(G, config: GraphConfig, trial_seed: int) -> dict:
    """One-shot algorithms' seed selection, computed once per trial (see
    common.py's docstring for why this is valid: it only ever looks at p_plus,
    which does not vary across a sweep's swept values for a fixed trial)."""
    _assign_params(G, config, q_range=(config.BETA_SWEEP_Q, config.BETA_SWEEP_Q), trial_seed=trial_seed)
    p_plus, _q = true_params_from_graph(G)
    group_of = graphs.group_of_map(G)

    seeds_kkt, _ = celf_greedy(
        G, p_plus, k=config.K, num_sims=config.CELF_NUM_SIMS,
        rng=np.random.default_rng(context_seed(config.name, "kkt_select", trial_seed)),
    )
    seeds_fair, _w, _reach = fair_welfare_greedy(
        G, p_plus, k=config.K, group_of=group_of, num_sims=config.FAIR_NUM_SIMS,
        rng=np.random.default_rng(context_seed(config.name, "fair_select", trial_seed)),
    )
    seeds_imm = imm_select(
        G, p_plus, k=config.K, epsilon=config.IMM_EPSILON,
        rng=np.random.default_rng(context_seed(config.name, "imm_select", trial_seed)),
    )

    if config.p_plus_mode == "range":
        edges = list(G.edges())
        scenario_low = {e: config.p_plus_range[0] for e in edges}
        scenario_high = {e: config.p_plus_range[1] for e in edges}
    else:
        # No natural "range" on a fixed real-data p_plus -- see module
        # docstring for the reasoning behind this scenario pair.
        scenario_low = dict(p_plus)
        scenario_high = {e: 0.5 * p for e, p in p_plus.items()}
    seeds_robust = robust_select(
        G, [scenario_low, scenario_high], k=config.K,
        gamma=config.ROBUST_KEMPE_GAMMA, num_sims=config.ROBUST_KEMPE_NUM_SIMS,
        rng=np.random.default_rng(context_seed(config.name, "robust_select", trial_seed)),
    )

    return {
        "kkt_greedy": list(seeds_kkt),
        "fair_greedy": list(seeds_fair),
        "imm": list(seeds_imm),
        "robust_kempe": list(seeds_robust),
    }


def run_baseline_forward(G, config: GraphConfig, seed_set, true_beta, q_range, trial_seed, algo, sweep, param):
    t0 = time.perf_counter()
    _assign_params(G, config, q_range=q_range, trial_seed=trial_seed)
    p_plus, q = true_params_from_graph(G)
    acts = seed_then_none_actions(G, seed_set, config.T)
    init = initial_state(G, seed_set)
    rng = np.random.default_rng(context_seed(config.name, "forward", algo, sweep, param, trial_seed))
    traj, _obs = simulate(G, init, p_plus, q, true_beta, acts, rng)
    runtime = time.perf_counter() - t0
    return traj, runtime


def run_mfbwi_forward(G, config: GraphConfig, true_beta, alpha_fair, q_range, trial_seed, sweep, param):
    _assign_params(G, config, q_range=q_range, trial_seed=trial_seed)
    t0 = time.perf_counter()
    result = run_mf_bwi_fair(
        G, true_beta=true_beta, T=config.T, budget=config.B, alpha_fair=alpha_fair,
        seed=context_seed(config.name, "mfbwi_forward", sweep, param, trial_seed),
    )
    runtime = time.perf_counter() - t0
    return result["trajectory"], runtime


def run_repeatedgreedy_forward(G, config: GraphConfig, true_beta, q_range, trial_seed, sweep, param):
    _assign_params(G, config, q_range=q_range, trial_seed=trial_seed)
    t0 = time.perf_counter()
    result = run_repeated_greedy(
        G, true_beta=true_beta, T=config.T, budget=config.B,
        lookahead_rounds=config.REPEATED_GREEDY_LOOKAHEAD, num_sims=config.REPEATED_GREEDY_NUM_SIMS,
        seed=context_seed(config.name, "repeated_forward", sweep, param, trial_seed),
    )
    runtime = time.perf_counter() - t0
    return result["trajectory"], runtime


def metrics_row(sweep, param, algo, trial, trajectory, group_of, group_sizes, runtime_s, n_seeds=None):
    reach = group_reach_fractions(trajectory, group_of, group_sizes)
    groups = sorted(group_sizes)
    row = {
        "sweep": sweep,
        "param": param,
        "algorithm": algo,
        "trial": trial,
        "time_avg_spread": time_avg_spread(trajectory),
        "min_group_reach": min(reach.values()),
        "runtime_s": runtime_s,
        "n_seeds": n_seeds,
        "round0_cost": (n_seeds * 5) if n_seeds is not None else None,
    }
    for g in groups:
        row[f"group{g}_reach"] = reach[g]
    return row


SWEEP_SEED_OFFSET = {"beta": 1, "q": 2, "alpha": 3}


def run_beta_or_q_sweep_value(config: GraphConfig, sweep_name, v, true_beta_of, q_range_of, alpha_fair, G, group_of, group_sizes, trial, trial_seed, seed_sets):
    """One (trial, swept value)'s worth of rows -- the finest-grained unit of
    run_beta_or_q_sweep, factored out so a caller (e.g. a checkpointed
    long-running driver) can persist progress after each value instead of
    only after a whole trial (a trial's full set of values x algorithms can
    itself take longer than this kind of driver can rely on running
    uninterrupted)."""
    rows = []
    true_beta = true_beta_of(v)
    q_range = q_range_of(v)

    for algo in ("kkt_greedy", "fair_greedy", "imm", "robust_kempe"):
        traj, rt = run_baseline_forward(
            G, config, seed_sets[algo], true_beta, q_range, trial_seed, algo, sweep_name, v
        )
        rows.append(metrics_row(
            sweep_name, v, algo, trial, traj, group_of, group_sizes, rt,
            n_seeds=len(seed_sets[algo]),
        ))

    traj, rt = run_mfbwi_forward(G, config, true_beta, alpha_fair, q_range, trial_seed, sweep_name, v)
    rows.append(metrics_row(sweep_name, v, "mf_bwi_fair", trial, traj, group_of, group_sizes, rt))

    traj, rt = run_repeatedgreedy_forward(G, config, true_beta, q_range, trial_seed, sweep_name, v)
    rows.append(metrics_row(sweep_name, v, "repeated_greedy", trial, traj, group_of, group_sizes, rt))
    return rows


def run_beta_or_q_sweep_trial(config: GraphConfig, sweep_name, values, true_beta_of, q_range_of, alpha_fair, G, group_of, group_sizes, trial):
    """One trial's worth of rows for run_beta_or_q_sweep."""
    trial_seed = context_seed(config.name, "trial_seed", sweep_name, trial)
    seed_sets = compute_baseline_seed_sets(G, config, trial_seed)
    rows = []
    for v in values:
        rows.extend(run_beta_or_q_sweep_value(
            config, sweep_name, v, true_beta_of, q_range_of, alpha_fair, G, group_of, group_sizes, trial, trial_seed, seed_sets
        ))
    return rows


def run_beta_or_q_sweep(config: GraphConfig, sweep_name, values, true_beta_of, q_range_of, alpha_fair, G, group_of, group_sizes):
    rows = []
    for trial in range(config.N_TRIALS):
        rows.extend(run_beta_or_q_sweep_trial(
            config, sweep_name, values, true_beta_of, q_range_of, alpha_fair, G, group_of, group_sizes, trial
        ))
    return rows


def compute_alpha_baselines(config: GraphConfig, G, seed_sets, trial_seed):
    """The one-shot-algorithms' forward trajectories shared across all of a
    trial's ALPHA_VALUES (only mf_bwi_fair itself varies with alpha) --
    factored out so a resuming driver can recompute this bounded, once-per-
    (trial-touched) cost without recomputing it once per alpha value."""
    q_range = (config.ALPHA_SWEEP_Q, config.ALPHA_SWEEP_Q)
    baseline_traj = {}
    baseline_rt = {}
    for algo in ("kkt_greedy", "fair_greedy", "imm", "robust_kempe"):
        traj, rt = run_baseline_forward(
            G, config, seed_sets[algo], config.ALPHA_SWEEP_BETA, q_range, trial_seed, algo, "alpha", "fixed"
        )
        baseline_traj[algo] = traj
        baseline_rt[algo] = rt

    traj, rt = run_repeatedgreedy_forward(
        G, config, config.ALPHA_SWEEP_BETA, q_range, trial_seed, "alpha", "fixed"
    )
    baseline_traj["repeated_greedy"] = traj
    baseline_rt["repeated_greedy"] = rt
    return baseline_traj, baseline_rt


def run_alpha_sweep_value(config: GraphConfig, G, group_of, group_sizes, trial, trial_seed, seed_sets, baseline_traj, baseline_rt, v):
    """One (trial, alpha value)'s worth of rows -- the finest-grained unit of
    run_alpha_sweep. See run_beta_or_q_sweep_value's docstring for why."""
    rows = []
    q_range = (config.ALPHA_SWEEP_Q, config.ALPHA_SWEEP_Q)
    for algo in ("kkt_greedy", "fair_greedy", "imm", "robust_kempe", "repeated_greedy"):
        n_seeds = len(seed_sets[algo]) if algo in seed_sets else None
        rows.append(
            metrics_row(
                "alpha", v, algo, trial, baseline_traj[algo], group_of, group_sizes,
                baseline_rt[algo], n_seeds=n_seeds,
            )
        )
    traj, rt = run_mfbwi_forward(G, config, config.ALPHA_SWEEP_BETA, v, q_range, trial_seed, "alpha", v)
    rows.append(metrics_row("alpha", v, "mf_bwi_fair", trial, traj, group_of, group_sizes, rt))
    return rows


def run_alpha_sweep_trial(config: GraphConfig, G, group_of, group_sizes, trial):
    """One trial's worth of rows for run_alpha_sweep."""
    trial_seed = context_seed(config.name, "trial_seed", "alpha", trial)
    seed_sets = compute_baseline_seed_sets(G, config, trial_seed)
    baseline_traj, baseline_rt = compute_alpha_baselines(config, G, seed_sets, trial_seed)
    rows = []
    for v in config.ALPHA_VALUES:
        rows.extend(run_alpha_sweep_value(
            config, G, group_of, group_sizes, trial, trial_seed, seed_sets, baseline_traj, baseline_rt, v
        ))
    return rows


def run_alpha_sweep(config: GraphConfig, G, group_of, group_sizes):
    rows = []
    for trial in range(config.N_TRIALS):
        rows.extend(run_alpha_sweep_trial(config, G, group_of, group_sizes, trial))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, algos, value_key="param"):
    out = {}
    values = sorted({r[value_key] for r in rows})
    for algo in algos:
        out[algo] = {}
        for v in values:
            sub = [r for r in rows if r["algorithm"] == algo and r[value_key] == v]
            if not sub:
                continue
            spreads = np.array([r["time_avg_spread"] for r in sub])
            min_reach = np.array([r["min_group_reach"] for r in sub])
            runtimes = np.array([r["runtime_s"] for r in sub])
            out[algo][v] = {
                "mean_spread": spreads.mean(),
                "std_spread": spreads.std(),
                "mean_min_reach": min_reach.mean(),
                "std_min_reach": min_reach.std(),
                "mean_runtime": runtimes.mean(),
                "n": len(sub),
            }
    return out, values
