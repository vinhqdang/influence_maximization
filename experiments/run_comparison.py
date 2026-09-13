"""Comparative study: MF-BWI-Fair vs. four baselines -- three one-shot classical
baselines (celf-greedy, fair-welfare-greedy, and IMM -- Tang, Shi & Xiao 2015,
the real published near-linear-time SOTA classical algorithm, see
im_lab/baselines/imm.py) and one sequential classical baseline that also acts
every round under the same budget (repeated_greedy, see
im_lab/baselines/repeated_greedy.py) -- on a fixed synthetic SBM graph,
sweeping backfire intensity (beta), recovery rate (q), and the fairness knob
(alpha_fair) independently. repeated_greedy isolates whether MF-BWI-Fair's
advantage comes from its specific machinery or merely from "gets to act every
round"; IMM is included so the comparison has a named, citable published
algorithm rather than only in-house baselines (see RESULTS.md).

Run with: python experiments/run_comparison.py
Outputs under experiments/results/: *.csv (raw per-trial rows), *.png (plots).
See experiments/common.py for the shared setup/config and experiments/RESULTS.md
for the write-up of what came out of this.
"""

from __future__ import annotations

import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
from im_lab import graphs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGO_LABELS = {
    "kkt_greedy": "KKT-greedy (one-shot)",
    "fair_greedy": "Fair-greedy (one-shot)",
    "imm": "IMM (Tang-Shi-Xiao 2015, one-shot)",
    "mf_bwi_fair": "MF-BWI-Fair (sequential)",
    "repeated_greedy": "Repeated-greedy (sequential, no fairness/uncertainty)",
}
ALGO_COLORS = {
    "kkt_greedy": "#d95f02",
    "fair_greedy": "#7570b3",
    "imm": "#66a61e",
    "mf_bwi_fair": "#1b9e77",
    "repeated_greedy": "#e6ab02",
}

# Fixed (not hash()-based, which is process-randomized) per-sweep offsets so
# trial seeds never collide across sweeps while staying fully reproducible.
SWEEP_SEED_OFFSET = {"beta": 1, "q": 2, "alpha": 3}


def metrics_row(sweep, param, algo, trial, trajectory, group_of, group_sizes, runtime_s):
    reach = C.group_reach_fractions(trajectory, group_of, group_sizes)
    groups = sorted(group_sizes)
    row = {
        "sweep": sweep,
        "param": param,
        "algorithm": algo,
        "trial": trial,
        "time_avg_spread": C.time_avg_spread(trajectory),
        "min_group_reach": min(reach.values()),
        "runtime_s": runtime_s,
    }
    for g in groups:
        row[f"group{g}_reach"] = reach[g]
    return row


def run_beta_or_q_sweep(sweep_name, values, true_beta_of, q_range_of, alpha_fair, G, group_of, group_sizes):
    """Shared logic for the beta and q sweeps: for both, the baseline forward
    pass must be recomputed at every swept value (true_beta or q_range changes
    the actual simulated dynamics), but the one-shot seed SELECTION is reused
    across the whole sweep for a given trial (see common.py docstring)."""
    rows = []
    for trial in range(C.N_TRIALS):
        trial_seed = 10_000 * SWEEP_SEED_OFFSET[sweep_name] + trial
        seed_sets = C.compute_baseline_seed_sets(G, trial_seed)

        for v in values:
            true_beta = true_beta_of(v)
            q_range = q_range_of(v)

            for algo in ("kkt_greedy", "fair_greedy", "imm"):
                traj, rt = C.run_baseline_forward(
                    G, seed_sets[algo], true_beta, q_range, trial_seed, algo, sweep_name, v
                )
                rows.append(metrics_row(sweep_name, v, algo, trial, traj, group_of, group_sizes, rt))

            traj, rt = C.run_mfbwi_forward(G, true_beta, C.B, alpha_fair, q_range, trial_seed, sweep_name, v)
            rows.append(metrics_row(sweep_name, v, "mf_bwi_fair", trial, traj, group_of, group_sizes, rt))

            traj, rt = C.run_repeatedgreedy_forward(G, true_beta, C.B, q_range, trial_seed, sweep_name, v)
            rows.append(metrics_row(sweep_name, v, "repeated_greedy", trial, traj, group_of, group_sizes, rt))
    return rows


def run_alpha_sweep(G, group_of, group_sizes):
    """alpha_fair only affects MF-BWI-Fair -- the other three algorithms (the
    two one-shot baselines AND repeated_greedy, which has no fairness
    mechanism either) are recomputed only ONCE per trial and their (identical)
    metrics are replicated across every alpha_fair value, since re-running them
    would just repeat the same computation under a different label (see
    common.py docstring)."""
    rows = []
    q_range = (C.ALPHA_SWEEP_Q, C.ALPHA_SWEEP_Q)
    for trial in range(C.N_TRIALS):
        trial_seed = 10_000 * SWEEP_SEED_OFFSET["alpha"] + trial
        seed_sets = C.compute_baseline_seed_sets(G, trial_seed)

        baseline_traj = {}
        baseline_rt = {}
        for algo in ("kkt_greedy", "fair_greedy", "imm"):
            traj, rt = C.run_baseline_forward(
                G, seed_sets[algo], C.ALPHA_SWEEP_BETA, q_range, trial_seed, algo, "alpha", "fixed"
            )
            baseline_traj[algo] = traj
            baseline_rt[algo] = rt

        traj, rt = C.run_repeatedgreedy_forward(
            G, C.ALPHA_SWEEP_BETA, C.B, q_range, trial_seed, "alpha", "fixed"
        )
        baseline_traj["repeated_greedy"] = traj
        baseline_rt["repeated_greedy"] = rt

        for v in C.ALPHA_VALUES:
            for algo in ("kkt_greedy", "fair_greedy", "imm", "repeated_greedy"):
                rows.append(
                    metrics_row(
                        "alpha", v, algo, trial, baseline_traj[algo], group_of, group_sizes,
                        baseline_rt[algo],
                    )
                )
            traj, rt = C.run_mfbwi_forward(G, C.ALPHA_SWEEP_BETA, C.B, v, q_range, trial_seed, "alpha", v)
            rows.append(metrics_row("alpha", v, "mf_bwi_fair", trial, traj, group_of, group_sizes, rt))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, value_key="param"):
    """Return {algo: {value: (mean_spread, std_spread, mean_min_reach,
    std_min_reach, mean_runtime)}}."""
    out = {}
    values = sorted({r[value_key] for r in rows})
    for algo in C.ALGOS:
        out[algo] = {}
        for v in values:
            sub = [r for r in rows if r["algorithm"] == algo and r[value_key] == v]
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


def plot_sweep(summary, values, xlabel, title, out_path, y_key="mean_spread", std_key="std_spread", ylabel="Time-averaged active nodes"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for algo in C.ALGOS:
        means = np.array([summary[algo][v][y_key] for v in values])
        stds = np.array([summary[algo][v][std_key] for v in values])
        ax.errorbar(
            values, means, yerr=stds, marker="o", capsize=3, label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
        )
        ax.fill_between(values, means - stds, means + stds, color=ALGO_COLORS[algo], alpha=0.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    G = C.build_graph()
    group_of = graphs.group_of_map(G)
    group_sizes = graphs.group_sizes(G)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, groups={group_sizes}")

    print("Running beta sweep...")
    beta_rows = run_beta_or_q_sweep(
        "beta", C.BETA_VALUES,
        true_beta_of=lambda v: v,
        q_range_of=lambda v: (C.BETA_SWEEP_Q, C.BETA_SWEEP_Q),
        alpha_fair=C.DEFAULT_ALPHA_FAIR,
        G=G, group_of=group_of, group_sizes=group_sizes,
    )
    write_csv(beta_rows, os.path.join(RESULTS_DIR, "beta_sweep.csv"))
    beta_summary, beta_values = summarize(beta_rows)
    plot_sweep(
        beta_summary, beta_values, "Backfire intensity beta (q=0.05, alpha_fair=0.3)",
        "Spread vs. backfire intensity", os.path.join(RESULTS_DIR, "beta_sweep.png"),
    )
    print(f"  done at {time.time()-t_start:.1f}s")

    print("Running q sweep...")
    q_rows = run_beta_or_q_sweep(
        "q", C.Q_VALUES,
        true_beta_of=lambda v: C.Q_SWEEP_BETA,
        q_range_of=lambda v: (v, v),
        alpha_fair=C.DEFAULT_ALPHA_FAIR,
        G=G, group_of=group_of, group_sizes=group_sizes,
    )
    write_csv(q_rows, os.path.join(RESULTS_DIR, "q_sweep.csv"))
    q_summary, q_values = summarize(q_rows)
    plot_sweep(
        q_summary, q_values, "Recovery rate q (beta=0.15, alpha_fair=0.3)",
        "Spread vs. recovery rate", os.path.join(RESULTS_DIR, "q_sweep.png"),
    )
    print(f"  done at {time.time()-t_start:.1f}s")

    print("Running alpha_fair sweep...")
    alpha_rows = run_alpha_sweep(G, group_of, group_sizes)
    write_csv(alpha_rows, os.path.join(RESULTS_DIR, "alpha_sweep.csv"))
    alpha_summary, alpha_values = summarize(alpha_rows)
    plot_sweep(
        alpha_summary, alpha_values, "Fairness parameter alpha_fair (beta=0.15, q=0.1)",
        "Spread vs. alpha_fair", os.path.join(RESULTS_DIR, "alpha_sweep.png"),
    )
    plot_sweep(
        alpha_summary, alpha_values, "Fairness parameter alpha_fair (beta=0.15, q=0.1)",
        "Worst-off group reach vs. alpha_fair", os.path.join(RESULTS_DIR, "alpha_sweep_fairness.png"),
        y_key="mean_min_reach", std_key="std_min_reach", ylabel="Min group reach fraction",
    )
    print(f"  done at {time.time()-t_start:.1f}s")

    print(f"Total runtime: {time.time()-t_start:.1f}s")

    return {
        "beta": (beta_summary, beta_values),
        "q": (q_summary, q_values),
        "alpha": (alpha_summary, alpha_values),
    }


if __name__ == "__main__":
    main()
