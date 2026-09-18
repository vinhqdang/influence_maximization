"""Checkpointed re-run of experiments/common.py's ORIGINAL 120-node main
synthetic-graph study (the paper's headline experiment), using EXACTLY the
same settings as common.py (T=30, B=100, K=20, N_TRIALS=15, full sweep
grids, CELF_NUM_SIMS=40/FAIR_NUM_SIMS=20/ROBUST_KEMPE_NUM_SIMS=60,
repeated_greedy's own library-default 2-lookahead/2-sim rollout) -- but
driven through multigraph_common.py's resumable/checkpointed sweep runner
(reusing run_real_facebook_multi_fullbudget.py's checkpoint helpers) instead
of common.py's own non-resumable run(), since run_comparison.py has no
checkpointing and this execution environment's containers can be reclaimed
with no warning at any time.

This exists SPECIFICALLY to re-run the main study after fixing the
one-round head-start harness bug documented in
REAL_GRAPH_N3_FOLLOWUP.md Section 5.1 and experiments/common.py's
run_mfbwi_forward/run_repeatedgreedy_forward docstrings (commit 19fdbfc):
every existing number from this study used the buggy harness and must be
treated as stale until this re-run completes.

Run with: python experiments/run_main_study_fullbudget.py
Output: experiments/results_multigraph/main_study_fullbudget/
"""

from __future__ import annotations

import os

import numpy as np

import multigraph_common as MC
from im_lab import graphs
from run_multigraph_validation import ALGO_COLORS, ALGO_LABELS, plot_sweep
from run_real_facebook_multi_fullbudget import (
    RESULTS_ROOT,
    _compute_baseline_seed_sets_resumable,
    _load_checkpoint,
    _read_rows,
    _run_sweep_resumable,
    _save_checkpoint,
)

import common as CS  # the original main-study module (build_graph, constants)

CONFIG = MC.GraphConfig(
    name="main_study_fullbudget",
    build_graph=CS.build_graph,
    p_plus_mode="range",
    p_plus_range=CS.P_PLUS_RANGE,
    K=CS.K, B=CS.B, T=CS.T, N_TRIALS=CS.N_TRIALS,
    CELF_NUM_SIMS=CS.CELF_NUM_SIMS, FAIR_NUM_SIMS=CS.FAIR_NUM_SIMS,
    IMM_EPSILON=CS.IMM_EPSILON,
    ROBUST_KEMPE_GAMMA=CS.ROBUST_KEMPE_GAMMA, ROBUST_KEMPE_NUM_SIMS=CS.ROBUST_KEMPE_NUM_SIMS,
    DEFAULT_BETA=CS.DEFAULT_BETA, DEFAULT_ALPHA_FAIR=CS.DEFAULT_ALPHA_FAIR,
    BETA_VALUES=tuple(CS.BETA_VALUES),
    Q_VALUES=tuple(CS.Q_VALUES),
    ALPHA_VALUES=tuple(CS.ALPHA_VALUES),
    BETA_SWEEP_Q=CS.BETA_SWEEP_Q, Q_SWEEP_BETA=CS.Q_SWEEP_BETA,
    ALPHA_SWEEP_BETA=CS.ALPHA_SWEEP_BETA, ALPHA_SWEEP_Q=CS.ALPHA_SWEEP_Q,
    # multigraph_common.GraphConfig's dataclass default (1, 1) is a scope
    # reduction specific to run_multigraph_validation.py's larger/denser
    # graphs -- the main study uses repeated_greedy's own library default
    # (2, 2), same as common.py's run_repeatedgreedy_forward (which passes
    # neither kwarg through, i.e. uses run_repeated_greedy's defaults).
    REPEATED_GREEDY_LOOKAHEAD=2, REPEATED_GREEDY_NUM_SIMS=2,
    notes=(
        "Post-harness-fix re-run of common.py's original 120-node main "
        "study, checkpointed. Same T/B/K/N_TRIALS/num_sims/epsilon/gamma/"
        "sweep grids/repeated_greedy rollout (library default 2/2) as the "
        "original -- no scope reduction, just resumability."
    ),
)


def run():
    import time

    config = CONFIG
    out_dir = os.path.join(RESULTS_ROOT, config.name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.json")

    _G_CACHE: dict = {}

    def build_G():
        if "G" not in _G_CACHE:
            _G_CACHE["G"] = config.build_graph()
        return _G_CACHE["G"]

    G = build_G()
    group_of = graphs.group_of_map(G)
    group_sizes = graphs.group_sizes(G)
    print(f"=== {config.name} === nodes={G.number_of_nodes()} edges={G.number_of_edges()} groups={group_sizes}", flush=True)

    ckpt = _load_checkpoint(ckpt_path)
    t_start = time.time()

    beta_path = os.path.join(out_dir, "beta_sweep.csv")
    beta_rows = _run_sweep_resumable(
        config, build_G, "beta", beta_path, config.BETA_VALUES,
        lambda trial, trial_seed, seed_sets, idx, v: MC.run_beta_or_q_sweep_value(
            config, "beta", v, true_beta_of=lambda v: v,
            q_range_of=lambda v: (config.BETA_SWEEP_Q, config.BETA_SWEEP_Q),
            alpha_fair=config.DEFAULT_ALPHA_FAIR,
            G=G, group_of=group_of, group_sizes=group_sizes,
            trial=trial, trial_seed=trial_seed, seed_sets=seed_sets,
        ),
        ckpt, ckpt_path, t_start,
    )
    beta_summary, beta_values = MC.summarize(beta_rows, MC.ALGOS)
    plot_sweep(beta_summary, beta_values, MC.ALGOS, f"Backfire intensity beta (q={config.BETA_SWEEP_Q})",
               f"{config.name}: spread vs. beta", os.path.join(out_dir, "beta_sweep.png"))
    print(f"  [{config.name}] beta sweep done at {time.time() - t_start:.1f}s", flush=True)

    q_path = os.path.join(out_dir, "q_sweep.csv")
    q_rows = _run_sweep_resumable(
        config, build_G, "q", q_path, config.Q_VALUES,
        lambda trial, trial_seed, seed_sets, idx, v: MC.run_beta_or_q_sweep_value(
            config, "q", v, true_beta_of=lambda v: config.Q_SWEEP_BETA,
            q_range_of=lambda v: (v, v),
            alpha_fair=config.DEFAULT_ALPHA_FAIR,
            G=G, group_of=group_of, group_sizes=group_sizes,
            trial=trial, trial_seed=trial_seed, seed_sets=seed_sets,
        ),
        ckpt, ckpt_path, t_start,
    )
    q_summary, q_values = MC.summarize(q_rows, MC.ALGOS)
    plot_sweep(q_summary, q_values, MC.ALGOS, f"Recovery rate q (beta={config.Q_SWEEP_BETA})",
               f"{config.name}: spread vs. q", os.path.join(out_dir, "q_sweep.png"))
    print(f"  [{config.name}] q sweep done at {time.time() - t_start:.1f}s", flush=True)

    alpha_path = os.path.join(out_dir, "alpha_sweep.csv")
    _baseline_cache: dict = {}

    def alpha_value_fn(trial, trial_seed, seed_sets, idx, v):
        if trial not in _baseline_cache:
            _baseline_cache.clear()
            _baseline_cache[trial] = MC.compute_alpha_baselines(config, G, seed_sets, trial_seed)
        baseline_traj, baseline_rt = _baseline_cache[trial]
        return MC.run_alpha_sweep_value(
            config, G, group_of, group_sizes, trial, trial_seed, seed_sets, baseline_traj, baseline_rt, v
        )

    alpha_rows = _run_sweep_resumable(config, build_G, "alpha", alpha_path, config.ALPHA_VALUES, alpha_value_fn, ckpt, ckpt_path, t_start)
    alpha_summary, alpha_values = MC.summarize(alpha_rows, MC.ALGOS)
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{config.name}: spread vs. alpha_fair", os.path.join(out_dir, "alpha_sweep.png"))
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{config.name}: min-group reach vs. alpha_fair", os.path.join(out_dir, "alpha_sweep_fairness.png"),
               y_key="mean_min_reach", std_key="std_min_reach", ylabel="Min group reach fraction")
    print(f"  [{config.name}] alpha sweep done at {time.time() - t_start:.1f}s total", flush=True)
    print(f"[{config.name}] TOTAL runtime: {time.time() - t_start:.1f}s", flush=True)
    print(f"[{config.name}] DONE", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    run()
