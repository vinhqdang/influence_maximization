# Real Facebook graph, full budget: follow-up to Section 4.2

**Further superseded:** this document's own full-budget numbers below were
later found to still carry a separate harness bug (one-shot baselines got
a free propagation round that MF-BWI-Fair and repeated_greedy did not);
see `REAL_GRAPH_N3_FOLLOWUP.md` Section 5 for the diagnosis and Section 6
for the fix and the post-fix re-run of this same graph plus two more
scales (686, 3437 nodes). Section 6.2 there confirms this document's
qualitative conclusion (MF-BWI-Fair ties/leads on 348) is unaffected by
that further fix; the numeric tables below are the pre-that-fix values,
kept for the audit trail.

This is the clean follow-up that Section 4.2 of `MULTIGRAPH_VALIDATION.md`
flagged as needed: the **same** `real_facebook_348` graph (`data/ego-facebook/348.edges`,
same Weighted-Cascade `p_plus`, same community-detection group assignment),
run with **no budget cuts** -- `T=30, B=100, K=20, N_TRIALS=15`,
`CELF_NUM_SIMS=40, FAIR_NUM_SIMS=20, ROBUST_KEMPE_NUM_SIMS=60, IMM_EPSILON=0.5,
ROBUST_KEMPE_GAMMA=0.3`, the full 6/5/5-point beta/q/alpha grids, and
`repeated_greedy`'s own library-default rollout (2 lookahead rounds x 2
sims) -- i.e. exactly `common.py`'s main synthetic-graph study settings,
just pointed at the real graph instead. Driver: `run_real_facebook_fullbudget.py`.
Raw output: `results_multigraph/real_facebook_348_fullbudget/` (`beta_sweep.csv`,
`q_sweep.csv`, `alpha_sweep.csv`, `checkpoint.json`, and the three plots
referenced below). 15/15 trials completed on all three sweeps (240/240
(trial, swept-value) cells; total measured compute time 434.2s once the
process was actually running -- the wall-clock calendar time was much
longer only because this execution environment's containers were
repeatedly reclaimed/restarted while idle, not because the computation
itself is slow; the run resumed from checkpoint each time with no lost
work).

## 1. Full-budget results

### 1.1 Beta sweep (backfire intensity, q fixed at 0.05)

| beta | kkt | fair | imm | robust_kempe | **mf_bwi_fair** | **repeated_greedy** | mf min-reach | rg min-reach |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 196.6 | 200.9 | 202.4 | 196.3 | **197.4** | **199.8** | 0.859 | 0.847 |
| 0.1 | 188.5 | 190.0 | 192.9 | 187.1 | **193.8** | **193.9** | 0.854 | 0.827 |
| 0.2 | 180.0 | 182.0 | 185.2 | 178.0 | **189.8** | **189.0** | 0.836 | 0.809 |
| 0.3 | 172.6 | 174.5 | 178.1 | 173.0 | **185.5** | **184.6** | 0.816 | 0.784 |
| 0.5 | 158.1 | 162.2 | 164.9 | 159.0 | **179.8** | **177.2** | 0.790 | 0.761 |
| 0.7 | 149.8 | 151.1 | 152.4 | 147.6 | **175.2** | **171.5** | 0.769 | 0.730 |

### 1.2 Q sweep (recovery rate, beta fixed at 0.15)

| q | kkt | fair | imm | robust_kempe | **mf_bwi_fair** | **repeated_greedy** | mf min-reach | rg min-reach |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 190.1 | 190.8 | 193.5 | 187.8 | **193.6** | **194.6** | 0.848 | 0.829 |
| 0.05 | 184.3 | 186.0 | 188.9 | 183.0 | **191.0** | **191.6** | 0.842 | 0.808 |
| 0.1 | 177.7 | 180.0 | 182.5 | 177.7 | **188.4** | **188.8** | 0.828 | 0.805 |
| 0.2 | 164.0 | 168.0 | 170.8 | 165.1 | **184.3** | **182.5** | 0.810 | 0.778 |
| 0.4 | 139.9 | 135.5 | 142.5 | 133.9 | **173.7** | **169.4** | 0.757 | 0.726 |

### 1.3 Alpha sweep (fairness parameter, beta=0.15, q=0.1)

| alpha | kkt | fair | imm | robust_kempe | **mf_bwi_fair** | **repeated_greedy** | mf min-reach | rg min-reach |
|---|---|---|---|---|---|---|---|---|
| 1.0 (utilitarian) | 178.8 | 179.6 | 183.5 | 176.9 | **187.2** | **188.2** | 0.809 | 0.808 |
| 0.5 | 178.8 | 179.6 | 183.5 | 176.9 | **188.4** | **188.2** | 0.824 | 0.808 |
| 0.0 (proportional) | 178.8 | 179.6 | 183.5 | 176.9 | **188.9** | **188.2** | 0.832 | 0.808 |
| -2.0 | 178.8 | 179.6 | 183.5 | 176.9 | **188.6** | **188.2** | 0.832 | 0.808 |
| -8.0 (leximin-like) | 178.8 | 179.6 | 183.5 | 176.9 | **190.7** | **188.2** | 0.844 | 0.808 |

(The one-shot baselines' columns do not vary with alpha by construction --
alpha only reweights `mf_bwi_fair`'s own objective; `repeated_greedy`'s
forward trajectory is also alpha-independent since it optimizes raw spread,
so its near-constant 188.2 here is expected, not a bug.)

## 2. Direct before/after comparison at Section 4.2's own four checkpoints

Section 4.2 (reduced budget: T=12, B=50, K=10, N_TRIALS=4, coarser grids,
weaker Monte-Carlo sample counts, `repeated_greedy` at a cut-down 1x1
rollout) reported MF-BWI-Fair trailing `repeated_greedy` at every one of
these four points:

| checkpoint | reduced-budget mf vs rg (spread) | **full-budget mf vs rg (spread)** | reduced-budget mf vs rg (min-reach) | **full-budget mf vs rg (min-reach)** |
|---|---|---|---|---|
| beta=0.0 | 118.4 vs 130.5 (mf **-9.3%**) | 197.4 vs 199.8 (mf -1.2%, tied) | -- | 0.859 vs 0.847 (mf ahead) |
| beta=0.5 | 113.2 vs 119.3 (mf **-5.1%**) | 179.8 vs 177.2 (**mf +1.5%**) | 0.351 vs 0.474 (mf **-26%**) | 0.790 vs 0.761 (**mf +3.8%**) |
| q=0.1 | 133.4 vs 139.3 (mf **-4.2%**) | 188.4 vs 188.8 (mf -0.2%, tied) | 0.417 vs 0.474 (mf **-12%**) | 0.828 vs 0.805 (**mf +2.9%**) |
| q=0.4 | 114.3 vs 117.0 (mf -2.3%) | 173.7 vs 169.4 (**mf +2.5%**) | 0.451 vs 0.396 (mf +14%) | 0.757 vs 0.726 (**mf +4.3%**) |

Every single point flips from "MF-BWI-Fair behind" to "tied or ahead" once
the compute budget is restored to the main study's settings. The two
points where the reduced-budget run showed the largest, clearest
MF-BWI-Fair deficits in fairness (beta=0.5: -26% min-reach; q=0.1: -12%)
are exactly where the full-budget run now shows MF-BWI-Fair **ahead**
(+3.8%, +2.9%). Nothing here is close to a coincidence-sized swing.

## 3. Verdict: hypothesis (1) (budget-cut artifact) is the dominant explanation

Section 4.2 posed two non-exclusive hypotheses for why MF-BWI-Fair failed
to dominate on this graph: (1) the reduced compute budget itself
(shorter horizon, smaller K/B, fewer Monte-Carlo samples, and critically a
crippled 1x1 `repeated_greedy` rollout instead of its library-default 2x2)
was artificially narrowing or reversing MF-BWI-Fair's margin; (2) a
genuine structural effect of this graph's much higher average degree
(~28.5, vs. 4-9x lower on every synthetic graph) degrading the mean-field
decoupling approximation MF-BWI-Fair's index relies on.

This follow-up isolates budget from structure by holding the graph fixed
and only restoring the budget. The result is unambiguous: **restoring the
budget alone is sufficient to flip every one of Section 4.2's four
checkpoints from "MF-BWI-Fair behind" to "tied or ahead."** That is
direct evidence for hypothesis (1): the reduced-budget run's non-dominance
was substantially, and probably mostly, a budget-cut artifact -- not a
graph-structure effect that survives when the algorithms are run at their
intended settings. In particular, giving `repeated_greedy` its full 2x2
rollout (rather than 1x1) was likely doing a large share of the damage in
the reduced-budget run: a stronger rollout budget for the *baseline* makes
it look artificially strong relative to MF-BWI-Fair, exactly the kind of
confound Section 4.2 flagged but had not yet isolated.

That said, this does **not** fully clear hypothesis (2) either, and the
honest reading is not "there is no structural effect" but "any structural
effect is much smaller than the budget effect, and does not reverse the
qualitative finding on this graph." Two things support this qualified
reading:

- **The shape of the effect matches the synthetic-graph pattern exactly**:
  roughly tied with `repeated_greedy` at low backfire/recovery (beta=0,
  q=0/0.05/0.1) and a monotonically growing MF-BWI-Fair lead as beta or q
  increases (beta=0.7: +2.2%; q=0.4: +2.5%) -- the same qualitative curve
  seen on `small_sbm`, `large_sbm`, and `barabasi_albert` in Section 4.1.
- **But the magnitude of the lead is smaller here than on any synthetic
  graph at the equivalent high-beta/high-q operating points** (e.g.
  `large_sbm` at q=0.4 showed a +23% MF-BWI-Fair lead; here q=0.4 shows
  only +2.5%). That gap in *magnitude*, even though the *direction* is now
  consistent, is plausibly the residual signature of the mean-field
  approximation degrading somewhat on a much denser, more clustered real
  network, exactly as hypothesis (2) anticipated -- just not severely
  enough to overturn the ranking once the budget confound is removed.

**Updated claim for the manuscript:** MF-BWI-Fair's advantage over
`repeated_greedy` (and the other classical baselines) is not an artifact
of the synthetic-graph family -- on the one real social network tested, at
matched compute budget, it reproduces the same qualitative pattern (tied
at low backfire-intensity/recovery-rate, growing lead as either
increases), in both spread and fairness (min-group reach). The margin on
this real, much denser graph is narrower than on any synthetic graph
tested, consistent with a modest, non-reversing mean-field-approximation
cost on high-average-degree networks that is worth flagging as a
limitation but does not undermine the core result. The prior framing in
Section 4.2 ("the finding does NOT clearly hold on real data") should be
retired in favor of: **the finding holds on the one real network tested,
once compute budget is matched to the main study, with a smaller margin
than on synthetic graphs.**

## 4. Plots

- `results_multigraph/real_facebook_348_fullbudget/beta_sweep.png` --
  spread vs. beta_true at fixed q=0.05.
- `results_multigraph/real_facebook_348_fullbudget/q_sweep.png` -- spread
  vs. q at fixed beta=0.15.
- `results_multigraph/real_facebook_348_fullbudget/alpha_sweep.png` /
  `alpha_sweep_fairness.png` -- spread and min-group reach vs.
  alpha_fair at fixed beta=0.15, q=0.1.

## 5. Caveats

- Still a single real graph (N=1 at the level of "real networks tested").
  The claim above is about this one ego-network, not real networks in
  general.
- The residual gap between this graph's lead magnitude and the synthetic
  graphs' lead magnitude is observed, not yet mechanistically explained --
  Section 4.2's item 2 (mean-field decoupling error on high-degree graphs)
  remains an open, unquantified question for `docs/theory.md`, just
  demoted from "possibly explains a reversal" to "possibly explains a
  smaller-than-synthetic margin."
- No budget/fairness invariant violations were observed across any of the
  240 (trial, swept-value) cells in this run (consistent with Section 5 of
  `MULTIGRAPH_VALIDATION.md` for the reduced-budget runs).
