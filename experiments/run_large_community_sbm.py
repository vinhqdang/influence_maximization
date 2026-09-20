"""Does the qualitative pattern from experiments/run_multigraph_validation.py
(MF-BWI-Fair beats every one-shot baseline; trails repeated-greedy by a
small, stable margin on the most community-structured real graph) hold on a
SYNTHETIC graph roughly 2x real_facebook_3437's node count, with the same
6-community, disparate-block-size structure that real_facebook_3437 has?

real_facebook_3437 (532 nodes, 6 detected communities of sizes 39-165) is
the graph on which MF-BWI-Fair trails repeated-greedy; Section 6
(Discussion) attributes this to a genuine control-quality gap specific to
richer, more disparate community structure, not to graph density or degree.
This experiment tests whether that gap holds, grows, or shrinks when the
SAME qualitative structure (6 communities, same size ratios) is scaled up
to roughly double the node count, entirely under our own compute (no
external graph needed, so no waiting on the pending Google Cloud grant).

Graph: stochastic_block_model_graph with 6 blocks in real_facebook_3437's
detected-community size ratio (165:137:79:42:39:70), scaled by 2x
(330:275:158:84:78:140, n=1065), with p_in/p_out calibrated (0.015/0.0015)
to give an average degree (~9.7) in the same modest range as the other
synthetic multi-graph configs (large_sbm, barabasi_albert) rather than
matching real_facebook_3437's own higher density -- isolating the
COMMUNITY-STRUCTURE effect from a density effect, which the true-parameter
ablation (Section 6) already ruled out as the explanation on 3437 itself.

Compute-budget note: REPEATED_GREEDY_LOOKAHEAD=1/NUM_SIMS=1 and reduced
Monte Carlo sample counts, exactly the same reduced-budget convention
already used and disclosed for small_sbm/large_sbm/barabasi_albert in
run_multigraph_validation.py (NOT the real-graph full-budget settings in
run_real_facebook_multi_fullbudget.py, which are far more expensive and
were only affordable on the real graphs' smaller node counts). This
experiment is therefore a like-for-like comparison with those three
existing synthetic checks, not with the real-graph full-budget numbers.
The alpha (fairness) sweep is skipped -- it is already covered on four
other graphs, and this experiment's purpose is specifically the
spread-gap-at-scale question, not another fairness check.

Calibration (see scratch benchmarking, not checked in): one repeated_greedy
round at this n/B takes ~0.9s (T=31 rounds/trajectory ~29s); one-shot seed
selection (dominated by fair_welfare_greedy at ~44s, robust_select at
~16s) is computed once per trial and shared across all 8 sweep cells.
Estimated total runtime: ~6 trials x (~65s seed-set + 8 cells x ~33s) ~= 33
minutes.

CHECKPOINTED / RESUMABLE, same (trial, swept-value) granularity as
run_real_facebook_multi_fullbudget.py, since this environment's containers
can be reclaimed with no warning.

Run with: python3 experiments/run_large_community_sbm.py
Output: experiments/results_multigraph/large_community_sbm/
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

import multigraph_common as MC
from im_lab import graphs
from im_lab.baselines.fair_greedy import fair_welfare_greedy
from im_lab.baselines.imm import imm_select
from im_lab.baselines.kkt_greedy import celf_greedy
from im_lab.baselines.robust_kempe import robust_select
from im_lab.simulator import true_params_from_graph
from multigraph_common import _assign_params
from run_multigraph_validation import ALGO_COLORS, ALGO_LABELS, plot_sweep

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_multigraph", "large_community_sbm")

# real_facebook_3437's 6 detected-community sizes (165, 137, 79, 42, 39, 70),
# scaled 2x, in the same order.
BLOCKS = [330, 275, 158, 84, 78, 140]

CONFIG = MC.GraphConfig(
    name="large_community_sbm",
    build_graph=lambda: graphs.stochastic_block_model_graph(BLOCKS, p_in=0.015, p_out=0.0015, seed=201),
    p_plus_mode="range", p_plus_range=(0.05, 0.15),
    K=40, B=200.0, T=30, N_TRIALS=6,
    CELF_NUM_SIMS=20, FAIR_NUM_SIMS=8, IMM_EPSILON=0.5,
    ROBUST_KEMPE_GAMMA=0.3, ROBUST_KEMPE_NUM_SIMS=20,
    REPEATED_GREEDY_LOOKAHEAD=1, REPEATED_GREEDY_NUM_SIMS=1,
    DEFAULT_BETA=0.15, DEFAULT_ALPHA_FAIR=0.0,
    BETA_VALUES=(0.0, 0.15, 0.3, 0.5),
    Q_VALUES=(0.0, 0.1, 0.2, 0.4),
    BETA_SWEEP_Q=0.05, Q_SWEEP_BETA=0.15,
    notes=(
        "n=1065, 6 blocks in real_facebook_3437's community-size ratio "
        "scaled 2x, same reduced-budget convention as small_sbm/large_sbm/"
        "barabasi_albert (see module docstring)."
    ),
)


def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"beta": {}, "q": {}}


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


def _compute_baseline_seed_sets_resumable(G, config, trial_seed, entry, ckpt, ckpt_path):
    partial = entry.setdefault("seed_sets_partial", {})
    _assign_params(G, config, q_range=(config.BETA_SWEEP_Q, config.BETA_SWEEP_Q), trial_seed=trial_seed)
    p_plus, _q = true_params_from_graph(G)
    group_of = graphs.group_of_map(G)

    if "kkt_greedy" not in partial:
        seeds_kkt, _ = celf_greedy(
            G, p_plus, k=config.K, num_sims=config.CELF_NUM_SIMS,
            rng=np.random.default_rng(MC.context_seed(config.name, "kkt_select", trial_seed)),
        )
        partial["kkt_greedy"] = list(seeds_kkt)
        _save_checkpoint(ckpt, ckpt_path)

    if "fair_greedy" not in partial:
        seeds_fair, _w, _reach = fair_welfare_greedy(
            G, p_plus, k=config.K, group_of=group_of, num_sims=config.FAIR_NUM_SIMS,
            rng=np.random.default_rng(MC.context_seed(config.name, "fair_select", trial_seed)),
        )
        partial["fair_greedy"] = list(seeds_fair)
        _save_checkpoint(ckpt, ckpt_path)

    if "imm" not in partial:
        seeds_imm = imm_select(
            G, p_plus, k=config.K, epsilon=config.IMM_EPSILON,
            rng=np.random.default_rng(MC.context_seed(config.name, "imm_select", trial_seed)),
        )
        partial["imm"] = list(seeds_imm)
        _save_checkpoint(ckpt, ckpt_path)

    if "robust_kempe" not in partial:
        edges = list(G.edges())
        scenario_low = {e: config.p_plus_range[0] for e in edges}
        scenario_high = {e: config.p_plus_range[1] for e in edges}
        seeds_robust = robust_select(
            G, [scenario_low, scenario_high], k=config.K,
            gamma=config.ROBUST_KEMPE_GAMMA, num_sims=config.ROBUST_KEMPE_NUM_SIMS,
            rng=np.random.default_rng(MC.context_seed(config.name, "robust_select", trial_seed)),
        )
        partial["robust_kempe"] = list(seeds_robust)
        _save_checkpoint(ckpt, ckpt_path)

    return dict(partial)


def _run_sweep_resumable(config, G, sweep_name, csv_path, values, value_fn, ckpt, ckpt_path, t_start, group_of, group_sizes):
    sweep_ckpt = ckpt[sweep_name]
    for trial in range(config.N_TRIALS):
        key = str(trial)
        entry = sweep_ckpt.setdefault(key, {"seed_sets": None, "done": []})
        trial_seed = MC.context_seed(config.name, "trial_seed", sweep_name, trial)
        if entry["seed_sets"] is None:
            entry["seed_sets"] = _compute_baseline_seed_sets_resumable(G, config, trial_seed, entry, ckpt, ckpt_path)
            entry.pop("seed_sets_partial", None)
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoint.json")
    ckpt = _load_checkpoint(ckpt_path)

    G = CONFIG.build_graph()
    group_of = graphs.group_of_map(G)
    group_sizes = graphs.group_sizes(G)
    print(f"=== {CONFIG.name} === nodes={G.number_of_nodes()} edges={G.number_of_edges()} groups={group_sizes}", flush=True)

    t_start = time.time()

    beta_path = os.path.join(RESULTS_DIR, "beta_sweep.csv")
    beta_rows = _run_sweep_resumable(
        CONFIG, G, "beta", beta_path, CONFIG.BETA_VALUES,
        lambda trial, trial_seed, seed_sets, idx, v: MC.run_beta_or_q_sweep_value(
            CONFIG, "beta", v, true_beta_of=lambda v: v,
            q_range_of=lambda v: (CONFIG.BETA_SWEEP_Q, CONFIG.BETA_SWEEP_Q),
            alpha_fair=CONFIG.DEFAULT_ALPHA_FAIR,
            G=G, group_of=group_of, group_sizes=group_sizes,
            trial=trial, trial_seed=trial_seed, seed_sets=seed_sets,
        ),
        ckpt, ckpt_path, t_start, group_of, group_sizes,
    )
    beta_summary, beta_values = MC.summarize(beta_rows, MC.ALGOS)
    plot_sweep(beta_summary, beta_values, MC.ALGOS, f"Backfire intensity beta (q={CONFIG.BETA_SWEEP_Q})",
               f"{CONFIG.name}: spread vs. beta", os.path.join(RESULTS_DIR, "beta_sweep.png"))
    print(f"  beta sweep done at {time.time() - t_start:.1f}s", flush=True)

    q_path = os.path.join(RESULTS_DIR, "q_sweep.csv")
    q_rows = _run_sweep_resumable(
        CONFIG, G, "q", q_path, CONFIG.Q_VALUES,
        lambda trial, trial_seed, seed_sets, idx, v: MC.run_beta_or_q_sweep_value(
            CONFIG, "q", v, true_beta_of=lambda v: CONFIG.Q_SWEEP_BETA,
            q_range_of=lambda v: (v, v),
            alpha_fair=CONFIG.DEFAULT_ALPHA_FAIR,
            G=G, group_of=group_of, group_sizes=group_sizes,
            trial=trial, trial_seed=trial_seed, seed_sets=seed_sets,
        ),
        ckpt, ckpt_path, t_start, group_of, group_sizes,
    )
    q_summary, q_values = MC.summarize(q_rows, MC.ALGOS)
    plot_sweep(q_summary, q_values, MC.ALGOS, f"Recovery rate q (beta={CONFIG.Q_SWEEP_BETA})",
               f"{CONFIG.name}: spread vs. q", os.path.join(RESULTS_DIR, "q_sweep.png"))
    print(f"  q sweep done at {time.time() - t_start:.1f}s total", flush=True)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
