"""Full-budget six-algorithm comparison on the two ADDITIONAL real ego-Facebook
graphs (686, 3437), run at exactly the main benchmark's settings (T=30, B=100,
K=20, N_TRIALS=15, full Monte Carlo sample counts, repeated_greedy's own
2-rollout library default) -- the same no-budget-cuts methodology already
validated on real_facebook_348 in run_real_facebook_fullbudget.py and written
up in experiments/REAL_GRAPH_FULLBUDGET_FOLLOWUP.md.

This exists to address the "N=1 real network" limitation directly (rather
than repeat the earlier two-step reduced-then-full-budget process, which is
now known to risk a misleading interim finding): both graphs here are run
ONLY at full budget, skipping the reduced-budget step entirely.

CHECKPOINTED / RESUMABLE, per (graph, sweep, trial, swept-value) -- same
granularity and same reasoning as run_real_facebook_fullbudget.py's module
docstring (this execution environment's containers can be reclaimed with no
warning at any time). Safe to re-invoke this exact command at any time to
resume from the last completed cell; a resumed run reproduces byte-identical
rows to an uninterrupted one (context_seed() is deterministic per
(config.name, sweep, trial, value)).

Run with: python experiments/run_real_facebook_multi_fullbudget.py
Output per graph: experiments/results_multigraph/<graph_name>_fullbudget/
"""

from __future__ import annotations

import csv
import json
import os
import time

import multigraph_common as MC
from im_lab import graphs
from run_multigraph_validation import (
    ALGO_COLORS,
    ALGO_LABELS,
    build_real_facebook_686,
    build_real_facebook_3437,
    plot_sweep,
)

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results_multigraph")

# Exactly common.py's main-study constants (see run_real_facebook_fullbudget.py's
# CONFIG for the detailed rationale) -- T=30, B=100, K=20, N_TRIALS=15,
# CELF_NUM_SIMS=40, FAIR_NUM_SIMS=20, ROBUST_KEMPE_NUM_SIMS=60, IMM_EPSILON=0.5,
# ROBUST_KEMPE_GAMMA=0.3, the same BETA_VALUES/Q_VALUES/ALPHA_VALUES grids as
# run_comparison.py, and repeated_greedy's own library defaults (2 lookahead
# rounds x 2 sims, i.e. simply not overridden). Only p_plus assignment
# (Weighted Cascade, fixed) and the community-detection group assignment are
# graph-specific -- everything else matches the main benchmark exactly.
# real_facebook_3437_fullbudget is delegated to a separate Colab VM (see
# /tmp/colab_run_3437.py, not checked in -- it is a throwaway copy of this
# same checkpoint logic restricted to just that one graph) to run in
# parallel with this process; it is deliberately NOT in this GRAPHS list so
# this process does not redundantly recompute it once it finishes 686.
GRAPHS = [
    dict(
        name="real_facebook_686_fullbudget",
        build_graph=build_real_facebook_686,
        beta_sweep_q=0.05, q_sweep_beta=0.15, alpha_sweep_beta=0.15, alpha_sweep_q=0.1,
        q_values=(0.0, 0.05, 0.1, 0.2, 0.4),
    ),
]


def make_config(spec):
    return MC.GraphConfig(
        name=spec["name"],
        build_graph=spec["build_graph"],
        p_plus_mode="fixed",
        K=20, B=100.0, T=30, N_TRIALS=15,
        CELF_NUM_SIMS=40, FAIR_NUM_SIMS=20,
        IMM_EPSILON=0.5,
        ROBUST_KEMPE_GAMMA=0.3, ROBUST_KEMPE_NUM_SIMS=60,
        DEFAULT_BETA=0.15, DEFAULT_ALPHA_FAIR=0.0,
        BETA_VALUES=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
        Q_VALUES=spec["q_values"],
        ALPHA_VALUES=(1.0, 0.5, 0.0, -2.0, -8.0),
        BETA_SWEEP_Q=spec["beta_sweep_q"], Q_SWEEP_BETA=spec["q_sweep_beta"],
        ALPHA_SWEEP_BETA=spec["alpha_sweep_beta"], ALPHA_SWEEP_Q=spec["alpha_sweep_q"],
        notes=(
            f"Full-budget real-graph validation ({spec['name']}): same "
            "T/B/K/N_TRIALS/num_sims/epsilon/gamma/sweep grids/"
            "repeated_greedy rollout as common.py's main synthetic-graph "
            "study and as real_facebook_348_fullbudget -- no budget cuts."
        ),
    )


def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"beta": {}, "q": {}, "alpha": {}}


def _save_checkpoint(ckpt, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, path)


def _append_rows(rows, path):
    if not rows:
        return
    file_exists = os.path.exists(path)
    fieldnames = list(rows[0].keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    """De-duplicates by (trial, param, algorithm), keeping the first
    occurrence -- see run_real_facebook_fullbudget.py's _read_rows
    docstring for why this is the one failure mode possible and why it is
    harmless here."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        seen = set()
        rows = []
        for r in reader:
            key = (r["trial"], r["param"], r["algorithm"])
            if key in seen:
                continue
            seen.add(key)
            r["trial"] = int(r["trial"])
            r["time_avg_spread"] = float(r["time_avg_spread"])
            r["min_group_reach"] = float(r["min_group_reach"])
            r["runtime_s"] = float(r["runtime_s"])
            rows.append(r)
        return rows


def _run_sweep_resumable(config, build_G, sweep_name, csv_path, values, value_fn, ckpt, ckpt_path, t_start):
    sweep_ckpt = ckpt[sweep_name]
    for trial in range(config.N_TRIALS):
        key = str(trial)
        entry = sweep_ckpt.setdefault(key, {"seed_sets": None, "done": []})
        trial_seed = MC.context_seed(config.name, "trial_seed", sweep_name, trial)
        if entry["seed_sets"] is None:
            entry["seed_sets"] = MC.compute_baseline_seed_sets(build_G(), config, trial_seed)
            _save_checkpoint(ckpt, ckpt_path)
        seed_sets = entry["seed_sets"]
        done = set(entry["done"])
        for idx, v in enumerate(values):
            if idx in done:
                continue
            rows = value_fn(trial, trial_seed, seed_sets, idx, v)
            _append_rows(rows, csv_path)
            entry["done"].append(idx)
            _save_checkpoint(ckpt, ckpt_path)
            print(f"  [{config.name}] {sweep_name} trial {trial + 1}/{config.N_TRIALS} value "
                  f"{idx + 1}/{len(values)} ({v}) done at {time.time() - t_start:.1f}s", flush=True)
    return _read_rows(csv_path)


def run_one(spec):
    config = make_config(spec)
    out_dir = os.path.join(RESULTS_ROOT, config.name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.json")

    _G_CACHE = {}

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
    _baseline_cache = {}

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


def run():
    for spec in GRAPHS:
        run_one(spec)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    run()
