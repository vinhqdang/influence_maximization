"""Follow-up to experiments/MULTIGRAPH_VALIDATION.md Section 4.2: re-run the
SAME real_facebook_348 graph (data/ego-facebook/348.edges, same loading /
Weighted-Cascade / recovery-rate / community-group assignment as
run_multigraph_validation.py's build_real_facebook_348) but with NO budget
cuts -- i.e. the exact same T/B/K/N_TRIALS/num_sims/epsilon/gamma/sweep-grid
settings as experiments/common.py's main synthetic-graph study
(run_comparison.py), and repeated_greedy's own library-default rollout
(2 lookahead rounds x 2 sims, not the reduced 1x1 used in the multi-graph
validation).

This isolates hypothesis (1) (the reduced budget itself caused
MULTIGRAPH_VALIDATION.md Section 4.2's divergence) from hypothesis (2) (a
genuine structural effect of this graph's density on MF-BWI-Fair's
mean-field approximation) -- see MULTIGRAPH_VALIDATION.md Section 4.2 and
experiments/REAL_GRAPH_FULLBUDGET_FOLLOWUP.md for the write-up of what came
out of this.

This is a SEPARATE, ADDITIONAL script -- it does not modify common.py,
run_comparison.py, multigraph_common.py, or run_multigraph_validation.py; it
only imports from them (graph builder, plotting helper, the parameterized
GraphConfig harness) to avoid duplicating that logic. Output goes to a
clearly separate directory (results_multigraph/real_facebook_348_fullbudget/)
so the original reduced-budget run's results are left untouched for the
before/after comparison.

CHECKPOINTED / RESUMABLE, AT (trial, swept-value) GRANULARITY: this run's
total cost (3 sweeps x 15 trials each, full-budget num_sims, on a 224-node/
6384-edge graph) is estimated at many hours, and this execution
environment's containers are NOT guaranteed to survive that long unattended
-- a container can be reclaimed/restarted with no warning (empirically,
after roughly 30-60 minutes with no active conversation turn, REGARDLESS of
whether a background process is still busy computing), silently killing a
plain background process. Two attempts at this run were lost this way
before this version existed -- the first because nothing was checkpointed
at all, the second because checkpointing only happened once per WHOLE
TRIAL (6 sweep values x 6 algorithms' full-budget forward simulation),
which itself turned out to take longer than this environment's typical
survival window, so it died before completing even one trial.

So this version checkpoints after each single (trial, swept value) --
computing all algorithms' rows for just one value of beta/q/alpha at a
time, appending those rows to the sweep's CSV immediately, and recording
the (trial, value-index) pair as done. It also persists each trial's
one-shot seed sets (CELF/fair-greedy/IMM/robust-Kempe selections) to the
checkpoint the first time that trial is touched, so resuming mid-trial
doesn't redo that selection cost on every restart -- only the forward
simulation of whichever (trial, value) pairs are not yet marked done. All
of this is deterministic per (config.name, sweep, trial, value) via
context_seed, so a resumed run produces byte-identical rows to an
uninterrupted one; re-invoking this exact command after a restart resumes
from the checkpoint automatically.

Run with: python experiments/run_real_facebook_fullbudget.py
(expected runtime: potentially several hours at this graph's density --
run as a background process, e.g.
  nohup python experiments/run_real_facebook_fullbudget.py > /tmp/real_facebook_fullbudget.log 2>&1 &
matching the pattern used elsewhere in this project. Safe to re-invoke the
same command at any time to resume from the last completed (trial, value).)
"""

from __future__ import annotations

import csv
import json
import os
import time

import multigraph_common as MC
from im_lab import graphs
from run_multigraph_validation import ALGO_COLORS, ALGO_LABELS, build_real_facebook_348, plot_sweep

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_multigraph", "real_facebook_348_fullbudget")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")

# Exactly experiments/common.py's main-study constants (see that module) --
# T=30, B=100, K=20, N_TRIALS=15, CELF_NUM_SIMS=40, FAIR_NUM_SIMS=20,
# ROBUST_KEMPE_NUM_SIMS=60, IMM_EPSILON=0.5, ROBUST_KEMPE_GAMMA=0.3, the same
# BETA_VALUES/Q_VALUES/ALPHA_VALUES grids and BETA_SWEEP_Q/Q_SWEEP_BETA/
# ALPHA_SWEEP_BETA/ALPHA_SWEEP_Q sweep points run_comparison.py uses, and
# repeated_greedy's own library defaults (REPEATED_GREEDY_LOOKAHEAD=2,
# REPEATED_GREEDY_NUM_SIMS=2 -- the GraphConfig dataclass defaults, so simply
# not overridden here) instead of the reduced 1x1 used in
# run_multigraph_validation.py's CONFIGS. p_plus_mode/p_plus assignment
# (Weighted Cascade, fixed) and the community-detection group assignment are
# UNCHANGED from the original real_facebook_348 config -- only the compute
# budget is being un-cut here, not the graph itself.
CONFIG = MC.GraphConfig(
    name="real_facebook_348_fullbudget",
    build_graph=build_real_facebook_348,
    p_plus_mode="fixed",
    K=20, B=100.0, T=30, N_TRIALS=15,
    CELF_NUM_SIMS=40, FAIR_NUM_SIMS=20,
    IMM_EPSILON=0.5,
    ROBUST_KEMPE_GAMMA=0.3, ROBUST_KEMPE_NUM_SIMS=60,
    DEFAULT_BETA=0.15, DEFAULT_ALPHA_FAIR=0.0,
    BETA_VALUES=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
    Q_VALUES=(0.0, 0.05, 0.1, 0.2, 0.4),
    ALPHA_VALUES=(1.0, 0.5, 0.0, -2.0, -8.0),
    BETA_SWEEP_Q=0.05, Q_SWEEP_BETA=0.15, ALPHA_SWEEP_BETA=0.15, ALPHA_SWEEP_Q=0.1,
    # REPEATED_GREEDY_LOOKAHEAD/NUM_SIMS left at GraphConfig's own defaults
    # (2, 2) -- i.e. repeated_greedy's library defaults, matching
    # common.py's run_repeatedgreedy_forward (which never overrides them).
    notes=(
        "Full-budget re-run of real_facebook_348 (see "
        "MULTIGRAPH_VALIDATION.md Section 4.2 and "
        "REAL_GRAPH_FULLBUDGET_FOLLOWUP.md): identical graph/groups/p_plus "
        "convention as the original reduced-budget real_facebook_348 config "
        "in run_multigraph_validation.py, but T/B/K/N_TRIALS/num_sims/"
        "epsilon/gamma/sweep grids/repeated_greedy rollout all match "
        "common.py's main synthetic-graph study exactly -- no budget cuts."
    ),
)


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"beta": {}, "q": {}, "alpha": {}}


def _save_checkpoint(ckpt):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CHECKPOINT_PATH)


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
    """Read back a sweep's accumulated CSV, de-duplicating by (trial, param,
    algorithm), keeping the first occurrence of each. A crash between
    _append_rows (which writes a value's rows) and _save_checkpoint (which
    marks that value done) can leave rows written but not marked done, so a
    resumed run recomputes and re-appends that same value -- this is the one
    failure direction possible given the write-then-checkpoint order below
    (never the reverse: a value is never marked done without its rows
    already written), and de-duping here makes it harmless."""
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


def _run_sweep_resumable(sweep_name, csv_path, values, value_fn, ckpt, t_start):
    """value_fn(trial, trial_seed, seed_sets, v_index, v) -> rows for one
    (trial, value). Checkpoints after each value; persists seed_sets the
    first time a trial is touched (see module docstring)."""
    sweep_ckpt = ckpt[sweep_name]
    for trial in range(CONFIG.N_TRIALS):
        key = str(trial)
        entry = sweep_ckpt.setdefault(key, {"seed_sets": None, "done": []})
        trial_seed = MC.context_seed(CONFIG.name, "trial_seed", sweep_name, trial)
        if entry["seed_sets"] is None:
            entry["seed_sets"] = MC.compute_baseline_seed_sets(build_G(), CONFIG, trial_seed)
            _save_checkpoint(ckpt)
        seed_sets = entry["seed_sets"]
        done = set(entry["done"])
        for idx, v in enumerate(values):
            if idx in done:
                continue
            rows = value_fn(trial, trial_seed, seed_sets, idx, v)
            _append_rows(rows, csv_path)
            entry["done"].append(idx)
            _save_checkpoint(ckpt)
            print(f"  {sweep_name} trial {trial + 1}/{CONFIG.N_TRIALS} value {idx + 1}/{len(values)} "
                  f"({v}) done at {time.time() - t_start:.1f}s", flush=True)
    return _read_rows(csv_path)


_G_CACHE = None


def build_G():
    global _G_CACHE
    if _G_CACHE is None:
        _G_CACHE = CONFIG.build_graph()
    return _G_CACHE


def run():
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    G = build_G()
    group_of = graphs.group_of_map(G)
    group_sizes = graphs.group_sizes(G)
    print(f"=== {CONFIG.name} === nodes={G.number_of_nodes()} edges={G.number_of_edges()} groups={group_sizes}", flush=True)

    ckpt = _load_checkpoint()
    t_start = time.time()

    beta_path = os.path.join(out_dir, "beta_sweep.csv")
    beta_rows = _run_sweep_resumable(
        "beta", beta_path, CONFIG.BETA_VALUES,
        lambda trial, trial_seed, seed_sets, idx, v: MC.run_beta_or_q_sweep_value(
            CONFIG, "beta", v, true_beta_of=lambda v: v,
            q_range_of=lambda v: (CONFIG.BETA_SWEEP_Q, CONFIG.BETA_SWEEP_Q),
            alpha_fair=CONFIG.DEFAULT_ALPHA_FAIR,
            G=G, group_of=group_of, group_sizes=group_sizes,
            trial=trial, trial_seed=trial_seed, seed_sets=seed_sets,
        ),
        ckpt, t_start,
    )
    beta_summary, beta_values = MC.summarize(beta_rows, MC.ALGOS)
    plot_sweep(beta_summary, beta_values, MC.ALGOS, f"Backfire intensity beta (q={CONFIG.BETA_SWEEP_Q})",
               f"{CONFIG.name}: spread vs. beta", os.path.join(out_dir, "beta_sweep.png"))
    print(f"  beta sweep done at {time.time() - t_start:.1f}s", flush=True)

    q_path = os.path.join(out_dir, "q_sweep.csv")
    q_rows = _run_sweep_resumable(
        "q", q_path, CONFIG.Q_VALUES,
        lambda trial, trial_seed, seed_sets, idx, v: MC.run_beta_or_q_sweep_value(
            CONFIG, "q", v, true_beta_of=lambda v: CONFIG.Q_SWEEP_BETA,
            q_range_of=lambda v: (v, v),
            alpha_fair=CONFIG.DEFAULT_ALPHA_FAIR,
            G=G, group_of=group_of, group_sizes=group_sizes,
            trial=trial, trial_seed=trial_seed, seed_sets=seed_sets,
        ),
        ckpt, t_start,
    )
    q_summary, q_values = MC.summarize(q_rows, MC.ALGOS)
    plot_sweep(q_summary, q_values, MC.ALGOS, f"Recovery rate q (beta={CONFIG.Q_SWEEP_BETA})",
               f"{CONFIG.name}: spread vs. q", os.path.join(out_dir, "q_sweep.png"))
    print(f"  q sweep done at {time.time() - t_start:.1f}s", flush=True)

    alpha_path = os.path.join(out_dir, "alpha_sweep.csv")

    # Cache the current trial's baseline trajectories across its ALPHA_VALUES
    # calls within this process run (cleared on moving to the next trial) --
    # only mf_bwi_fair varies with alpha, so recomputing this per-value
    # within one uninterrupted run would be pure waste. A restart still just
    # recomputes it once for whichever trial is resumed (see module
    # docstring); nothing here is persisted to the checkpoint.
    _baseline_cache = {}

    def alpha_value_fn(trial, trial_seed, seed_sets, idx, v):
        if trial not in _baseline_cache:
            _baseline_cache.clear()
            _baseline_cache[trial] = MC.compute_alpha_baselines(CONFIG, G, seed_sets, trial_seed)
        baseline_traj, baseline_rt = _baseline_cache[trial]
        return MC.run_alpha_sweep_value(
            CONFIG, G, group_of, group_sizes, trial, trial_seed, seed_sets, baseline_traj, baseline_rt, v
        )

    alpha_rows = _run_sweep_resumable("alpha", alpha_path, CONFIG.ALPHA_VALUES, alpha_value_fn, ckpt, t_start)
    alpha_summary, alpha_values = MC.summarize(alpha_rows, MC.ALGOS)
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{CONFIG.name}: spread vs. alpha_fair", os.path.join(out_dir, "alpha_sweep.png"))
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{CONFIG.name}: min-group reach vs. alpha_fair", os.path.join(out_dir, "alpha_sweep_fairness.png"),
               y_key="mean_min_reach", std_key="std_min_reach", ylabel="Min group reach fraction")
    print(f"  alpha sweep done at {time.time() - t_start:.1f}s total", flush=True)

    print(f"TOTAL runtime: {time.time() - t_start:.1f}s", flush=True)
    print("DONE", flush=True)
    return {
        "graph": {"n": G.number_of_nodes(), "m": G.number_of_edges(), "group_sizes": group_sizes},
        "beta": (beta_summary, beta_values),
        "q": (q_summary, q_values),
        "alpha": (alpha_summary, alpha_values),
    }


if __name__ == "__main__":
    run()
