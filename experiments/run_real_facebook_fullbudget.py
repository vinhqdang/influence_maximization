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

Run with: python experiments/run_real_facebook_fullbudget.py
(expected runtime: potentially several hours at this graph's density --
run as a background process, e.g.
  nohup python experiments/run_real_facebook_fullbudget.py > /tmp/real_facebook_fullbudget.log 2>&1 &
matching the pattern used elsewhere in this project.)
"""

from __future__ import annotations

import os
import time

import multigraph_common as MC
from im_lab import graphs
from run_multigraph_validation import ALGO_COLORS, ALGO_LABELS, build_real_facebook_348, plot_sweep

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_multigraph", "real_facebook_348_fullbudget")

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


def run():
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    G = CONFIG.build_graph()
    group_of = graphs.group_of_map(G)
    group_sizes = graphs.group_sizes(G)
    print(f"=== {CONFIG.name} === nodes={G.number_of_nodes()} edges={G.number_of_edges()} groups={group_sizes}", flush=True)

    t_start = time.time()

    beta_rows = MC.run_beta_or_q_sweep(
        CONFIG, "beta", CONFIG.BETA_VALUES,
        true_beta_of=lambda v: v,
        q_range_of=lambda v: (CONFIG.BETA_SWEEP_Q, CONFIG.BETA_SWEEP_Q),
        alpha_fair=CONFIG.DEFAULT_ALPHA_FAIR,
        G=G, group_of=group_of, group_sizes=group_sizes,
    )
    MC.write_csv(beta_rows, os.path.join(out_dir, "beta_sweep.csv"))
    beta_summary, beta_values = MC.summarize(beta_rows, MC.ALGOS)
    plot_sweep(beta_summary, beta_values, MC.ALGOS, f"Backfire intensity beta (q={CONFIG.BETA_SWEEP_Q})",
               f"{CONFIG.name}: spread vs. beta", os.path.join(out_dir, "beta_sweep.png"))
    print(f"  beta sweep done at {time.time()-t_start:.1f}s", flush=True)

    q_rows = MC.run_beta_or_q_sweep(
        CONFIG, "q", CONFIG.Q_VALUES,
        true_beta_of=lambda v: CONFIG.Q_SWEEP_BETA,
        q_range_of=lambda v: (v, v),
        alpha_fair=CONFIG.DEFAULT_ALPHA_FAIR,
        G=G, group_of=group_of, group_sizes=group_sizes,
    )
    MC.write_csv(q_rows, os.path.join(out_dir, "q_sweep.csv"))
    q_summary, q_values = MC.summarize(q_rows, MC.ALGOS)
    plot_sweep(q_summary, q_values, MC.ALGOS, f"Recovery rate q (beta={CONFIG.Q_SWEEP_BETA})",
               f"{CONFIG.name}: spread vs. q", os.path.join(out_dir, "q_sweep.png"))
    print(f"  q sweep done at {time.time()-t_start:.1f}s", flush=True)

    alpha_rows = MC.run_alpha_sweep(CONFIG, G, group_of, group_sizes)
    MC.write_csv(alpha_rows, os.path.join(out_dir, "alpha_sweep.csv"))
    alpha_summary, alpha_values = MC.summarize(alpha_rows, MC.ALGOS)
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{CONFIG.name}: spread vs. alpha_fair", os.path.join(out_dir, "alpha_sweep.png"))
    plot_sweep(alpha_summary, alpha_values, MC.ALGOS, "Fairness parameter alpha_fair",
               f"{CONFIG.name}: min-group reach vs. alpha_fair", os.path.join(out_dir, "alpha_sweep_fairness.png"),
               y_key="mean_min_reach", std_key="std_min_reach", ylabel="Min group reach fraction")
    print(f"  alpha sweep done at {time.time()-t_start:.1f}s total", flush=True)

    print(f"TOTAL runtime: {time.time()-t_start:.1f}s", flush=True)
    return {
        "graph": {"n": G.number_of_nodes(), "m": G.number_of_edges(), "group_sizes": group_sizes},
        "beta": (beta_summary, beta_values),
        "q": (q_summary, q_values),
        "alpha": (alpha_summary, alpha_values),
    }


if __name__ == "__main__":
    run()
