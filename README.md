# MF-BWI-Fair: Fair, Non-Progressive Influence Control Under Backfire and Parameter Uncertainty

Research code for a classical (non-ML) influence-maximization control policy —
**MF-BWI-Fair** — designed for the setting where the standard Kempe–Kleinberg–Tardos
assumptions break down: diffusion is non-progressive and can be actively reversed
("backfire" / negative word-of-mouth), edge influence parameters are not known
exactly, and the intervention must not neglect a disadvantaged subgroup.

This repository accompanies a manuscript submitted to *IEEE Transactions on
Computational Social Systems* (see `paper/`).

## What's here

- **`im_lab/`** — core library
  - `mf_bwi_fair.py` — the proposed control policy (mean-field belief propagation +
    Bayesian parameter tracking + closed-form Lagrangian index + fairness reweighting)
  - `closed_form_index.py` — closed-form Lagrangian index for the 2-state/3-action
    per-node control problem (replaces a bisection + value-iteration inner loop)
  - `lagrangian_index.py`, `bayes.py`, `actions.py`, `fairness.py`, `simulator.py`,
    `graphs.py` — supporting theory/simulation machinery
  - `baselines/` — independently reimplemented published baselines: IMM
    (Tang–Shi–Xiao 2015), He–Kempe robust Saturate Greedy (KDD 2016),
    KKT-style one-shot greedy, fairness-aware greedy, repeated greedy
- **`experiments/`** — experiment drivers and result CSVs/checkpoints
  - `run_comparison.py` — synthetic stochastic block model comparison
  - `run_multigraph_validation.py` — multi-graph robustness validation
  - `run_real_facebook_fullbudget.py` / `run_real_facebook_multi_fullbudget.py` —
    full-budget (no compute-cut) real-graph experiments on SNAP ego-Facebook
    subgraphs, checkpointed per (graph, sweep, trial, swept-value) cell so runs
    can resume after interruption
- **`docs/`** — `theory.md` (proofs: sandwich approximation, hardness,
  closed-form index derivation) and `related_work.md`
- **`tests/`** — unit tests for every module above (`pytest`)
- **`paper/`** — the LaTeX manuscript (`main.tex`) and compiled PDF

## Key results (see the paper for full detail)

- A sandwich approximation from two monotone-submodular surrogates gives a
  $\rho(1-1/e)$ guarantee for every backfire intensity $\beta \in [0,1)$,
  degenerating gracefully to the tight $1-1/e$ bound at $\beta = 0$.
- A closed-form Lagrangian index replaces the bisection-plus-value-iteration
  inner loop with an $O(Km + n\log n)$-per-round procedure, ~2 orders of
  magnitude faster in practice.
- $(1-1/e)$-hardness is shown to survive backfire exactly.
- A population-weighting pathology in an existing isoelastic-welfare fairness
  objective (which under-protects small groups) is identified and corrected
  with an egalitarian reweighting.
- On a 120-node synthetic SBM, MF-BWI-Fair sustains spread under
  backfire/recovery where one-shot baselines degrade sharply (>6x the best
  one-shot baseline's spread at the most severe recovery rate tested).
- Validated on further synthetic graphs and on real SNAP ego-Facebook
  subgraphs (348, 686, and 3437 nodes) at full compute budget.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `networkx`, `pytest`, `matplotlib`.

## Running tests

```bash
pytest tests/
```

## Running experiments

```bash
# Synthetic SBM comparison
python3 experiments/run_comparison.py

# Multi-graph robustness validation
python3 experiments/run_multigraph_validation.py

# Full-budget real Facebook-subgraph experiments (checkpointed, resumable)
PYTHONPATH=. python3 experiments/run_real_facebook_multi_fullbudget.py
```

Real-graph experiment scripts write incremental checkpoints under
`experiments/results_multigraph/<graph_name>/checkpoint.json` and can be
re-run to resume from the last completed cell.

## Data

Real-world validation uses subgraphs from the SNAP ego-Facebook dataset
(`data/ego-facebook/`), following McAuley & Leskovec (2012).

## Citation

If you use this code, please cite the accompanying manuscript (see
`paper/fair_backfire_influence_control.tex` / `paper/references.bib`).
