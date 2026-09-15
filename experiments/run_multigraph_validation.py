"""Multi-graph robustness/validation study: does the qualitative story from
experiments/run_comparison.py (MF-BWI-Fair beats one-shot baselines under
backfire/recovery; the fairness fix improves min-group reach at a neutral
alpha) hold beyond the single 120-node hand-tuned synthetic SBM graph that
every result in experiments/RESULTS.md is based on?

Runs the SAME six-algorithm comparison structure (see
experiments/multigraph_common.py, a parameterized generalization of
experiments/common.py) on FOUR graphs:
  1. real_facebook_348  -- a REAL graph: SNAP's ego-Facebook node-348
     ego-network (see data/ego-facebook/README.md for full provenance),
     Weighted-Cascade edge probabilities, community-detection group labels.
  2. small_sbm           -- a smaller/denser synthetic SBM (60 nodes).
  3. large_sbm           -- a larger synthetic SBM (240 nodes, 2x the
     original study's graph).
  4. barabasi_albert     -- a different topology entirely (scale-free,
     preferential attachment, 150 nodes) rather than a block model.

Scope reductions relative to run_comparison.py, and why (this is the
"reduced-scope comparison" the task asked for, given four graphs instead of
one -- see the "Compute-budget note" below and MULTIGRAPH_VALIDATION.md
Section 3 for the full accounting): N_TRIALS is 4-6 instead of 15 (fewer
independent random draws per (algorithm, parameter, graph) cell -- reported
std devs are noisier as a result); BETA_VALUES/Q_VALUES/ALPHA_VALUES are each
a coarser subset of the original sweep's grid (4/4/3 points instead of
6/5/5); CELF_NUM_SIMS/FAIR_NUM_SIMS/ROBUST_KEMPE_NUM_SIMS are reduced
(20/8/15-20 instead of 40/20/60) and repeated_greedy uses 1 rollout of 1
lookahead round instead of the library default 2/2 -- these Monte-Carlo/
rollout reductions are the dominant runtime saving, since several of these
graphs have much higher average degree than the original 120-node graph and
every algorithm's per-candidate cost scales with edges touched, not just
node count. horizon T=30 is kept for the three synthetic graphs but reduced
to T=12 for the real graph specifically (see its GraphConfig's `notes`).

Run with: python experiments/run_multigraph_validation.py
Outputs under experiments/results_multigraph/<graph_name>/: *.csv, *.png.
See experiments/MULTIGRAPH_VALIDATION.md for the write-up.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import multigraph_common as MC
from im_lab import graphs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_multigraph")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ego-facebook", "348.edges")

ALGO_LABELS = {
    "kkt_greedy": "KKT-greedy (one-shot)",
    "fair_greedy": "Fair-greedy (one-shot)",
    "imm": "IMM (Tang-Shi-Xiao 2015, one-shot)",
    "robust_kempe": "Robust-Kempe (He-Kempe 2016, one-shot, bicriteria)",
    "mf_bwi_fair": "MF-BWI-Fair (sequential)",
    "repeated_greedy": "Repeated-greedy (sequential, no fairness/uncertainty)",
}
ALGO_COLORS = {
    "kkt_greedy": "#d95f02",
    "fair_greedy": "#7570b3",
    "imm": "#66a61e",
    "robust_kempe": "#a6761d",
    "mf_bwi_fair": "#1b9e77",
    "repeated_greedy": "#e6ab02",
}


# --- Graph builders ----------------------------------------------------------

def build_real_facebook_348():
    """Real graph: SNAP ego-Facebook node 348's friendship subgraph (224 nodes,
    single connected component -- see data/ego-facebook/README.md). Groups are
    assigned via community detection (Clauset-Newman-Moore greedy modularity)
    as a STRUCTURAL PROXY for demographic fairness groups -- this dataset
    carries no real demographic labels, so this is explicitly NOT a claim that
    the detected communities correspond to any real attribute of these
    (anonymized) Facebook users. Communities smaller than 15 nodes are merged
    into one residual group (see assign_communities_as_groups's min_size
    docstring) so the fairness metric isn't dominated by near-singleton
    groups; the result is 4 groups of sizes 104/82/20/18."""
    G = graphs.load_edge_list_graph(DATA_PATH)
    graphs.assign_communities_as_groups(G, method="greedy_modularity", min_size=15)
    return G


def build_real_facebook_686():
    """Real graph: SNAP ego-Facebook node 686's friendship subgraph (168
    nodes, single connected component -- see data/ego-facebook/README.md).
    Community detection with min_size=40 (higher than 348's 15, since 686's
    detected communities include one small residual that stays below 15
    even after merging) gives 3 groups of sizes 63/56/49."""
    G = graphs.load_edge_list_graph(os.path.join(os.path.dirname(__file__), "..", "data", "ego-facebook", "686.edges"))
    graphs.assign_communities_as_groups(G, method="greedy_modularity", min_size=40)
    return G


def build_real_facebook_3437():
    """Real graph: SNAP ego-Facebook node 3437's friendship subgraph (532
    nodes as loaded, single connected component -- see
    data/ego-facebook/README.md). Community detection with min_size=25
    gives 6 groups of sizes 165/137/79/42/39/70 (the last a merged
    residual of communities below the threshold)."""
    G = graphs.load_edge_list_graph(os.path.join(os.path.dirname(__file__), "..", "data", "ego-facebook", "3437.edges"))
    graphs.assign_communities_as_groups(G, method="greedy_modularity", min_size=25)
    return G


def build_small_sbm():
    """Smaller, denser SBM than the main study's (60 vs 120 nodes; p_in/p_out
    hand-tuned the same way the original graph was -- see common.py's tuning
    notes -- so that a plain degree/greedy seed set neither dies immediately
    nor saturates the network; verified at ~49% time-averaged reach under
    CELF-greedy seeding at the default beta/q before this config was fixed)."""
    return graphs.stochastic_block_model_graph([10, 20, 30], p_in=0.10, p_out=0.01, seed=101)


def build_large_sbm():
    """Larger SBM than the main study's (240 vs 120 nodes, i.e. 2x every
    block size), with p_in/p_out reduced (0.05/0.006 vs 0.07/0.008) to
    compensate for the larger node count -- otherwise average degree would
    roughly double and the network would saturate almost immediately.
    Verified at ~68% time-averaged reach under CELF-greedy seeding at the
    default beta/q."""
    return graphs.stochastic_block_model_graph([40, 80, 120], p_in=0.05, p_out=0.006, seed=102)


def build_barabasi_albert():
    """A genuinely different topology (scale-free / preferential attachment)
    rather than another block model -- BA graphs have no natural community
    structure, so groups are assigned via the existing random-balanced
    partition (graphs.assign_groups, called internally by
    barabasi_albert_graph), exactly the same convention already used for
    ER/BA graphs elsewhere in this package. Verified at ~73% time-averaged
    reach under CELF-greedy seeding at the default beta/q."""
    return graphs.barabasi_albert_graph(150, 3, num_groups=3, seed=103)


# --- Compute-budget note (see MULTIGRAPH_VALIDATION.md for the full story) --
# repeated_greedy's per-round candidate rollout (see multigraph_common.py's
# REPEATED_GREEDY_LOOKAHEAD/NUM_SIMS docstring) and fair_welfare_greedy's
# naive non-lazy greedy are BOTH costs that scale with the number of EDGES a
# single simulated round touches, not just the number of nodes -- so the
# REAL graph here (224 nodes but avg degree ~28.5, vs. ~3-7 for every
# synthetic graph below) is by far the most expensive per trial despite
# being smaller than large_sbm. CELF_NUM_SIMS/FAIR_NUM_SIMS/
# ROBUST_KEMPE_NUM_SIMS are reduced from run_comparison.py's defaults
# (40/20/60) to 20/8/15-20 across the board here, and repeated_greedy uses
# 1 rollout of 1 lookahead round (vs common.py's library defaults of 2/2)
# -- both real reductions in what each algorithm sees, not merely an
# implementation speedup, and reported as such in MULTIGRAPH_VALIDATION.md.
_SEED_SIM_KW = dict(CELF_NUM_SIMS=20, FAIR_NUM_SIMS=8, IMM_EPSILON=0.5)

CONFIGS = [
    MC.GraphConfig(
        name="real_facebook_348",
        build_graph=build_real_facebook_348,
        p_plus_mode="fixed",
        K=10, B=50.0, T=12, N_TRIALS=4,
        ROBUST_KEMPE_GAMMA=0.3, ROBUST_KEMPE_NUM_SIMS=15,
        DEFAULT_BETA=0.15, DEFAULT_ALPHA_FAIR=0.0,
        BETA_VALUES=(0.0, 0.15, 0.3, 0.5),
        Q_VALUES=(0.1, 0.2, 0.3, 0.4),
        ALPHA_VALUES=(1.0, 0.0, -4.0),
        BETA_SWEEP_Q=0.3, Q_SWEEP_BETA=0.15, ALPHA_SWEEP_BETA=0.15, ALPHA_SWEEP_Q=0.3,
        REPEATED_GREEDY_LOOKAHEAD=1, REPEATED_GREEDY_NUM_SIMS=1,
        **_SEED_SIM_KW,
        notes=(
            "Weighted-Cascade p_plus (mean ~0.035, far below the synthetic "
            "graphs' (0.05,0.15) range) on a DENSE real graph (avg degree "
            "~28.5) turned out to cascade very readily regardless of q in "
            "the (0.05,0.15) band the synthetic graphs use -- calibration "
            "(see MULTIGRAPH_VALIDATION.md) showed >=80% reach even at q=0.4 "
            "with as few as 3 seeds. q is therefore swept over a HIGHER "
            "band (0.1-0.4, default 0.3) than the synthetic graphs' "
            "(0.0-0.4, default 0.05) -- a deliberate, documented departure "
            "from matching the synthetic graph's dynamics exactly, per the "
            "task's own guidance that the real graph's dynamics may simply "
            "be different. K/B are also reduced (10/50 vs the synthetic "
            "graphs' proportionally larger budgets) and T is reduced to 12 "
            "(vs 30 elsewhere), purely for compute-budget reasons: this "
            "graph's density (avg degree ~28.5, vs ~3-7 for every synthetic "
            "graph here) makes every per-round candidate evaluation "
            "(fair_greedy's selection step, repeated_greedy's rollout) far "
            "more expensive per call, and cascades on it also visibly "
            "stabilize well before round 12 in practice."
        ),
    ),
    MC.GraphConfig(
        name="small_sbm",
        build_graph=build_small_sbm,
        p_plus_mode="range", p_plus_range=(0.05, 0.15),
        K=10, B=50.0, T=30, N_TRIALS=6,
        ROBUST_KEMPE_GAMMA=0.3, ROBUST_KEMPE_NUM_SIMS=20,
        DEFAULT_BETA=0.15, DEFAULT_ALPHA_FAIR=0.0,
        BETA_VALUES=(0.0, 0.15, 0.3, 0.5),
        Q_VALUES=(0.0, 0.1, 0.2, 0.4),
        ALPHA_VALUES=(1.0, 0.0, -4.0),
        BETA_SWEEP_Q=0.05, Q_SWEEP_BETA=0.15, ALPHA_SWEEP_BETA=0.15, ALPHA_SWEEP_Q=0.1,
        **_SEED_SIM_KW,
    ),
    MC.GraphConfig(
        name="large_sbm",
        build_graph=build_large_sbm,
        p_plus_mode="range", p_plus_range=(0.05, 0.15),
        K=40, B=200.0, T=30, N_TRIALS=4,
        ROBUST_KEMPE_GAMMA=0.3, ROBUST_KEMPE_NUM_SIMS=20,
        DEFAULT_BETA=0.15, DEFAULT_ALPHA_FAIR=0.0,
        BETA_VALUES=(0.0, 0.15, 0.3, 0.5),
        Q_VALUES=(0.0, 0.1, 0.2, 0.4),
        ALPHA_VALUES=(1.0, 0.0, -4.0),
        BETA_SWEEP_Q=0.05, Q_SWEEP_BETA=0.15, ALPHA_SWEEP_BETA=0.15, ALPHA_SWEEP_Q=0.1,
        **_SEED_SIM_KW,
        notes="N_TRIALS=4 (fewest of the four graphs) since this is the largest graph (240 nodes) and the most expensive per trial after the real graph.",
    ),
    MC.GraphConfig(
        name="barabasi_albert",
        build_graph=build_barabasi_albert,
        p_plus_mode="range", p_plus_range=(0.05, 0.15),
        K=25, B=125.0, T=30, N_TRIALS=6,
        ROBUST_KEMPE_GAMMA=0.3, ROBUST_KEMPE_NUM_SIMS=20,
        DEFAULT_BETA=0.15, DEFAULT_ALPHA_FAIR=0.0,
        BETA_VALUES=(0.0, 0.15, 0.3, 0.5),
        Q_VALUES=(0.0, 0.1, 0.2, 0.4),
        ALPHA_VALUES=(1.0, 0.0, -4.0),
        BETA_SWEEP_Q=0.05, Q_SWEEP_BETA=0.15, ALPHA_SWEEP_BETA=0.15, ALPHA_SWEEP_Q=0.1,
        **_SEED_SIM_KW,
    ),
]


def plot_sweep(summary, values, algos, xlabel, title, out_path, y_key="mean_spread", std_key="std_spread", ylabel="Time-averaged active nodes"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for algo in algos:
        if not summary[algo]:
            continue
        vals = [v for v in values if v in summary[algo]]
        means = np.array([summary[algo][v][y_key] for v in vals])
        stds = np.array([summary[algo][v][std_key] for v in vals])
        ax.errorbar(
            vals, means, yerr=stds, marker="o", capsize=3, label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
        )
        ax.fill_between(vals, means - stds, means + stds, color=ALGO_COLORS[algo], alpha=0.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_one_graph(config: MC.GraphConfig):
    out_dir = os.path.join(RESULTS_DIR, config.name)
    os.makedirs(out_dir, exist_ok=True)

    G = config.build_graph()
    group_of = graphs.group_of_map(G)
    group_sizes = graphs.group_sizes(G)
    print(f"\n=== {config.name} === nodes={G.number_of_nodes()} edges={G.number_of_edges()} groups={group_sizes}")

    t0 = time.time()
    beta_rows = MC.run_beta_or_q_sweep(
        config, "beta", config.BETA_VALUES,
        true_beta_of=lambda v: v,
        q_range_of=lambda v: (config.BETA_SWEEP_Q, config.BETA_SWEEP_Q),
        alpha_fair=config.DEFAULT_ALPHA_FAIR,
        G=G, group_of=group_of, group_sizes=group_sizes,
    )
    MC.write_csv(beta_rows, os.path.join(out_dir, "beta_sweep.csv"))
    beta_summary, beta_values = MC.summarize(beta_rows, MC.ALGOS)
    plot_sweep(beta_summary, beta_values, MC.ALGOS, f"Backfire intensity beta (q={config.BETA_SWEEP_Q})",
               f"{config.name}: spread vs. beta", os.path.join(out_dir, "beta_sweep.png"))
    print(f"  beta sweep done at {time.time()-t0:.1f}s")

    q_rows = MC.run_beta_or_q_sweep(
        config, "q", config.Q_VALUES,
        true_beta_of=lambda v: config.Q_SWEEP_BETA,
        q_range_of=lambda v: (v, v),
        alpha_fair=config.DEFAULT_ALPHA_FAIR,
        G=G, group_of=group_of, group_sizes=group_sizes,
    )
    MC.write_csv(q_rows, os.path.join(out_dir, "q_sweep.csv"))
    q_summary, q_values = MC.summarize(q_rows, MC.ALGOS)
    plot_sweep(q_summary, q_values, MC.ALGOS, f"Recovery rate q (beta={config.Q_SWEEP_BETA})",
               f"{config.name}: spread vs. q", os.path.join(out_dir, "q_sweep.png"))
    print(f"  q sweep done at {time.time()-t0:.1f}s")

    alpha_rows = MC.run_alpha_sweep(config, G, group_of, group_sizes)
    MC.write_csv(alpha_rows, os.path.join(out_dir, "alpha_sweep.csv"))
    alpha_summary, alpha_values = MC.summarize(alpha_rows, MC.ALGOS)
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{config.name}: spread vs. alpha_fair", os.path.join(out_dir, "alpha_sweep.png"))
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{config.name}: min-group reach vs. alpha_fair", os.path.join(out_dir, "alpha_sweep_fairness.png"),
               y_key="mean_min_reach", std_key="std_min_reach", ylabel="Min group reach fraction")
    print(f"  alpha sweep done at {time.time()-t0:.1f}s total")

    return {
        "graph": {"n": G.number_of_nodes(), "m": G.number_of_edges(), "group_sizes": group_sizes},
        "beta": (beta_summary, beta_values),
        "q": (q_summary, q_values),
        "alpha": (alpha_summary, alpha_values),
    }


def main():
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for config in CONFIGS:
        results[config.name] = run_one_graph(config)
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
    return results


if __name__ == "__main__":
    main()
