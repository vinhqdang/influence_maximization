# Multi-graph validation: real-world data + additional synthetic graphs

The only experimental validation in `experiments/RESULTS.md` uses ONE hand-tuned
120-node synthetic SBM graph. This document reports on the same six-algorithm
comparison (kkt_greedy, fair_greedy, imm, robust_kempe, mf_bwi_fair,
repeated_greedy) run on a REAL social network (Task 1) and three ADDITIONAL
synthetic graphs at different scales/topologies (Task 2), to check whether
`RESULTS.md`'s qualitative findings generalize. Code:
`experiments/multigraph_common.py` (a parameterized generalization of
`experiments/common.py`) and `experiments/run_multigraph_validation.py`.
Raw per-trial data and plots: `experiments/results_multigraph/<graph_name>/`.
`experiments/RESULTS.md`, `experiments/common.py`, and
`experiments/run_comparison.py` are unmodified by this work.

## 1. The real-world dataset

- **Source**: SNAP, Stanford Network Analysis Project ego-Facebook dataset --
  https://snap.stanford.edu/data/ego-Facebook.html, archive
  https://snap.stanford.edu/data/facebook.tar.gz. Citation: J. McAuley and
  J. Leskovec, "Learning to Discover Social Circles in Ego Networks," NIPS,
  2012.
- **File used**: `348.edges` (this repo's copy: `data/ego-facebook/348.edges`;
  full provenance and the reasoning for picking this specific ego over the
  archive's other 9 in `data/ego-facebook/README.md`) -- the induced
  friendship subgraph among Facebook ego-user "348"'s friends: **224 nodes,
  3,192 undirected edges, a single connected component, average degree
  ~28.5**. Real, publicly available, well under the ~500-node budget this
  task set, and one connected component (unlike several of the archive's
  other ego files, which have small disconnected islands).
- **Why this file and not a larger SNAP graph**: every algorithm compared
  here does per-round or Monte-Carlo work (CELF-greedy's/fair-greedy's/
  robust-kempe's Monte-Carlo spread estimation, IMM's RR-set sampling,
  repeated_greedy's and MF-BWI-Fair's per-round computation) -- the task's
  own budget guidance (a couple hundred nodes, at most ~500) was followed
  deliberately, and even at 224 nodes this graph turned out to be the most
  expensive of the four studied here (see "Compute-budget realities" below),
  which would only have gotten worse on a thousands-of-nodes SNAP graph.

### Influence-probability convention: Weighted Cascade

Real graphs carry no ground-truth influence probabilities. Rather than invent
one, this project uses the **Weighted Cascade (WC)** convention:
`p_plus[(u, v)] = 1 / in-degree(v)`. WC is one of the two standard
conventions the IM literature has used for graphs without measured influence
data since Kempe, Kleinberg & Tardos (2003); it is explicitly named "Weighted
Cascade" in Chen, Wang & Yang, "Efficient Influence Maximization in Social
Networks" (KDD 2009), Section 4, and remains the standard non-uniform default
since (e.g. it is one of the two probability models IMM -- already
reimplemented in `im_lab/baselines/imm.py` -- is evaluated under in Tang, Shi
& Xiao 2015's own experiments). Implemented as
`im_lab.graphs.assign_weighted_cascade_probabilities` (tested in
`tests/test_graphs.py`). On the 348-ego graph, WC gives p_plus with mean
~0.035, min ~0.010, max 1.0 (nodes with in-degree 1) -- much smaller than the
synthetic graphs' hand-tuned `Uniform(0.05, 0.15)` range.

### Recovery rate q and backfire range

`im_lab.graphs.assign_recovery_rates(G, q_range, seed)` samples q per node
uniformly within `q_range`, independent of p_plus (new function, factored out
of `assign_true_parameters` rather than reusing it, since real p_plus is not
itself a random draw from a range -- see that function's docstring for why
`assign_true_parameters` itself is left untouched). Calibration (naive
top-out-degree-seed and actual CELF-greedy-seed forward simulations, see
`multigraph_common.py`/scratch exploration) showed this graph's high average
degree (~28.5, vs ~3-7 for every synthetic graph in this study) makes it
cascade readily even at q values that would keep the synthetic graphs from
saturating: with only 3 seeds and q=0.4 (the top of the synthetic graphs'
usual q range), average reach was still >50%, and with the study's intended
K=10-20 seeds it exceeded 75-85% across the whole (0.02, 0.4) q range tried.
**q is therefore swept over a higher band on this graph (0.1-0.4, default
0.3) than the synthetic graphs (0.0-0.4, default 0.05)** -- a deliberate,
documented departure rather than forcing this graph's dynamics to match the
synthetic ones, per the task's own guidance that a real graph's dynamics may
simply differ.

### Fairness groups: community detection as a structural proxy

SNAP's ego-Facebook files carry no demographic attributes usable here (see
`data/ego-facebook/README.md` for why the archive's own `.circles`/`.feat`
files are deliberately NOT used for this). Groups are instead assigned via
**community detection** -- `im_lab.graphs.assign_communities_as_groups`,
using networkx's `greedy_modularity_communities` (Clauset, Newman & Moore,
2004) on the graph's undirected collapse. **This is explicitly a structural
proxy, not real demographic data**: it merely gives the fairness machinery
some principled, reproducible partition of nodes into groups, grounded in the
graph's own connectivity, and nothing more. Raw community detection on the
348-ego graph found 6 communities of sizes 104/82/18/10/8/2; the three
smallest (which would otherwise make "min group reach" degenerate --a
2-node group's reach can only ever read 0, 0.5 or 1) were merged into one
residual group via the new `min_size` parameter, giving **4 groups: 104 / 82
/ 20 / 18**.

## 2. Additional synthetic graphs

All three reuse `im_lab.graphs`' existing generators (no new generator code
needed) with `assign_true_parameters`/`p_plus_range=(0.05, 0.15)` exactly as
`common.py`'s main study uses, and were calibrated the same way the original
120-node graph was (checked that CELF-greedy-seeded forward spread neither
died out nor saturated near 100% before finalizing p_in/p_out):

| graph | generator | sizes | p_in | p_out | nodes | groups | verified reach* |
|---|---|---|---|---|---|---|---|
| `small_sbm` | `stochastic_block_model_graph` | [10,20,30] | 0.10 | 0.01 | 60 | 3 (SBM blocks) | ~49% |
| `large_sbm` | `stochastic_block_model_graph` | [40,80,120] | 0.05 | 0.006 | 240 | 3 (SBM blocks) | ~68% |
| `barabasi_albert` | `barabasi_albert_graph(n=150, m=3)` | -- | -- | -- | 150 | 3 (random-balanced, via `assign_groups`) | ~73% |

\* time-averaged active fraction under CELF-greedy seeding at beta=0.15,
q=0.05, before committing to the config -- a sanity check, not one of this
document's reported findings.

`small_sbm` is roughly half the original graph's node count at higher
density (denser, since a 60-node graph with the original 0.07/0.008
densities would be too sparse to cascade at all); `large_sbm` is exactly 2x
every block size, with p_in/p_out reduced to compensate for the larger node
count (unchanged densities would have driven average degree up and
saturated the network almost immediately); `barabasi_albert` is a genuinely
different topology (scale-free / preferential attachment) rather than
another block model, using this package's existing random-balanced group
assignment (`assign_groups`, already used for ER/BA graphs elsewhere) since
BA graphs have no natural community structure of their own.

## 3. Reduced-scope comparison: what was reduced, and why

Same six-algorithm comparison structure as `run_comparison.py`, generalized
by `experiments/multigraph_common.py`'s `GraphConfig` (see that module's
docstring for the full design rationale: why this is a NEW parameterized
module rather than either modifying `common.py`'s functions in place or
copy-pasting them). Reductions relative to `run_comparison.py`'s defaults,
applied because four graphs (one of them much denser than anything in the
original study) multiply total runtime:

- **N_TRIALS**: 4-6 (vs. 15) -- 4 for `real_facebook_348` and `large_sbm`
  (the two most expensive graphs), 6 for `small_sbm`/`barabasi_albert`.
- **Sweep grids**: `BETA_VALUES` (4 points vs. 6), `Q_VALUES` (4 vs. 5),
  `ALPHA_VALUES` (3 vs. 5) -- coarser but still spanning the same qualitative
  range (utilitarian -> proportional -> leximin-like for alpha; no-backfire
  -> strong-backfire for beta).
- **Monte-Carlo sample counts**: `CELF_NUM_SIMS` 20 (vs. 40),
  `FAIR_NUM_SIMS` 8 (vs. 20), `ROBUST_KEMPE_NUM_SIMS` 15-20 (vs. 60) across
  all four graphs -- these are the dominant per-trial cost
  (`fair_welfare_greedy`'s naive non-lazy greedy re-simulates every
  remaining candidate's marginal value every round; on a graph with many
  edges, one such re-simulation costs much more per call than on a sparse
  graph of the same node count).
- **`repeated_greedy`'s own rollout cost knobs**: 1 rollout of 1 lookahead
  round (vs. the library default of 2/2) on all four graphs -- this is a
  further, real reduction in what the algorithm itself evaluates per round,
  not just an unrelated speedup, and is reported as such here rather than
  silently.
- **`real_facebook_348` specifically**: K=10, B=50 (vs. 20/100 for the other
  graphs, proportionally similar to their node counts) and T=12 (vs. 30) --
  this graph's average degree (~28.5) is 4-9x every synthetic graph's here,
  which makes every per-round candidate evaluation markedly more expensive
  per call (each one re-simulates a step that touches every in-edge of every
  active node); T=12 was chosen because this graph's cascades visibly
  stabilize well before round 12 given its density.

Everything else (T=30 for the three synthetic graphs, IMM's epsilon=0.5,
`ROBUST_KEMPE_GAMMA`=0.3, the layered backfire/recovery simulator, the
context-seed determinism mechanism) matches `run_comparison.py`/`common.py`
unchanged.

## 4. Results

### 4.1 Synthetic graphs: the core finding reproduces at two scales and a different topology

`small_sbm` (60 nodes, denser SBM), `large_sbm` (240 nodes, 2x every block
size), and `barabasi_albert` (150 nodes, scale-free) all reproduce
`RESULTS.md`'s qualitative story under this validation's reduced trial/grid
budget (see Section 3):

| graph | beta=0 (mf_bwi_fair vs repeated_greedy) | beta=0.5 | q=0 | q=0.4 |
|---|---|---|---|---|
| small_sbm | 51.8 vs 51.7 (tied) | **51.1 vs 47.6** | 52.1 vs 53.2 (tied) | **47.8 vs 35.1** |
| large_sbm | 213.7 vs 212.7 (tied) | **207.9 vs 188.7** | 214.6 vs 211.5 (tied) | **202.8 vs 164.2** |
| barabasi_albert | 137.2 vs 134.1 (mf ahead) | **134.1 vs 123.9** | 136.2 vs 133.6 (mf ahead) | **133.0 vs 108.3** |

Same pattern as the main study every time: roughly tied (or mf_bwi_fair
slightly ahead) with no/low backfire-recovery, a clear and growing
MF-BWI-Fair lead as beta or q increases -- on `large_sbm` at q=0.4 the gap
is nearly identical in relative terms to the main 120-node study (+23%
here vs. the main study's own large gaps at high q). Min-group reach shows
the same pattern: MF-BWI-Fair matches or leads repeated_greedy at every
point checked on all three synthetic graphs (e.g. small_sbm q=0.4: 0.772
vs. 0.553; barabasi_albert q=0.4: 0.879 vs. 0.703), including graphs at
double the node count and a structurally different (preferential-attachment)
topology. **The core finding is not an artifact of the one hand-tuned
120-node SBM graph** -- it reproduces across scale and topology changes
within the synthetic-graph family.

### 4.2 The real Facebook ego-network: the finding does NOT clearly hold

This is the result that matters most and must be reported plainly rather
than smoothed over. On `real_facebook_348` (224 real nodes, avg. degree
~28.5), MF-BWI-Fair does **not** show the dominant advantage seen
everywhere else:

| param | kkt | fair | imm | robust_kempe | **mf_bwi_fair** | **repeated_greedy** |
|---|---|---|---|---|---|---|
| beta=0.0 | 114.3 | 119.7 | 131.2 | 104.0 | **118.4** | **130.5** |
| beta=0.5 | 85.2 | 102.0 | 111.4 | 91.8 | **113.2** | **119.3** |
| q=0.1 | 135.5 | 128.3 | 148.0 | 127.7 | **133.4** | **139.3** |
| q=0.4 | 91.6 | 93.3 | 104.8 | 76.5 | **114.3** | **117.0** |

MF-BWI-Fair is *not* clearly ahead of repeated_greedy at any of these four
points (it trails by 2-11%, though the gap narrows at the highest q), and
at beta=0/q=0.1 it is also behind IMM in absolute spread -- something that
never happened on any synthetic graph, including at zero backfire/recovery
where the two were expected to be roughly tied, not for a one-shot
classical baseline to win outright. Min-group reach is similarly mixed, not
a clean win: mf_bwi_fair leads at q=0.4 (0.451 vs. 0.396) but trails at
beta=0.5 (0.351 vs. 0.474) and q=0.1 (0.417 vs. 0.474). The alpha sweep
(fixed beta/q operating point) is closer to parity (spread 122.2 vs. 122.6
at alpha=0, min-group reach 0.454 vs. 0.410 in MF-BWI-Fair's favor there)
but does not show the large, unambiguous advantage seen on every synthetic
graph.

**Plausible explanations, not yet disentangled -- reported as open
questions, not resolved claims:**
1. **Confounded settings, not just confounded structure.** This graph's run
   used a shorter horizon (T=12 vs. 30), a smaller budget (B=50 vs. 100),
   fewer Monte Carlo samples across every baseline, and a cheaper
   repeated_greedy rollout (1x1 vs. the library default 2x2) -- all
   necessary to keep runtime tractable at this graph's ~28.5 average degree
   (Section 3), but any one of these could independently narrow
   MF-BWI-Fair's margin (less time for Bayesian learning to pay off, a
   weaker repeated_greedy rollout budget could *also* be why repeated_greedy
   itself is doing unusually well here rather than MF-BWI-Fair doing
   unusually poorly, etc.). This has NOT been isolated by an ablation and
   should be before drawing a structural conclusion.
2. **Genuine structural effect.** Average degree ~28.5 is 4-9x every
   synthetic graph tested; the mean-field decoupling MF-BWI-Fair's
   coupling-handling relies on (docs/theory.md Section 0/3) is an
   approximation whose error is not separately quantified anywhere in this
   project (also flagged as an open item in the manuscript's Discussion
   section) -- a much denser, more clustered real network is exactly where
   that approximation is most likely to degrade, and could plausibly explain
   a real, not merely budget-related, narrowing of the advantage.
3. Both could be true simultaneously and to an unknown degree.

**This is not a small caveat.** The manuscript's current framing ("MF-BWI-Fair
wins everywhere except two well-understood ties") is not yet supported on
real-world data by this validation. The honest, current claim is: **the
theoretical guarantees and the empirical advantage are demonstrated on
synthetic graphs across two scales and two topologies; on the one real
social network tested, under a necessarily-reduced compute budget, the
advantage is inconsistent and sometimes reverses.** A clean follow-up (same
T/B/num_sims as the main study, on a real graph small enough to afford it,
or a principled ablation isolating budget/horizon from graph structure)
is needed before claiming this generalizes to real networks, and the paper's
Results/Discussion sections should say exactly this, not soften it.

## 5. Bugs / invariant violations

None across all four graphs and all three sweeps -- budget and fairness
invariants held on every trial.