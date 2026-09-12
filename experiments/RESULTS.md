# MF-BWI-Fair vs. one-shot baselines: comparative study

This is the write-up for `experiments/run_comparison.py` (shared setup/helpers in
`experiments/common.py`). Raw per-trial data: `experiments/results/*.csv`. Plots:
`experiments/results/*.png`.

## Setup

- **Graph**: one fixed Stochastic Block Model, 3 groups of unequal size
  20/40/60 (120 nodes total), `p_in=0.07`, `p_out=0.008`,
  `p_plus ~ Uniform(0.05, 0.15)` per edge. Density was hand-tuned (see
  session exploration) so the network neither saturates near 100% active nor
  fails to cascade at all -- both extremes hide the effects being swept.
- **Budget/horizon**: per-round budget `B = 100`, horizon `T = 30` rounds.
- **One-shot baselines' k**: `k = 20`, chosen so `k * CONVERT_cost (5) = 100 =
  B` -- the one-shot baselines get to spend exactly one round's worth of
  MF-BWI-Fair's budget, once, at round 0 (via
  `simulator.seed_then_none_actions`), then nothing for the remaining 29
  rounds. `k` is held fixed across all three sweeps.
- **Trials**: 15 independent random trials per (algorithm, parameter setting).
  Within a sweep, a given trial index reuses the same random parameter draw
  (`p_plus`, and `q` where relevant) across the swept parameter's values, only
  varying the swept parameter itself and drawing fresh simulation noise --
  this is a paired design (common random numbers across the x-axis for a
  trial) that reduces noise in the sweep trend without narrowing the
  between-trial variance reported in the tables below.
- **Metrics**: (a) time-averaged active-node count over the full horizon
  (`sigma(S)`, averaged over all T post-action rounds, not just the final
  round); (b) per-group reach fraction (own-group-size-normalized, also
  time-averaged) and the min across groups; (c) wall-clock runtime.
- **Efficiency note**: `celf_greedy`'s and `fair_welfare_greedy`'s seed
  *selection* only ever looks at `p_plus` (never at `q`, `beta`, or
  `alpha_fair`), so it is computed once per trial and reused across every
  swept value in that sweep, instead of being recomputed from scratch at each
  point -- this is the dominant cost (`fair_welfare_greedy`'s naive,
  non-lazy greedy is far slower than `celf_greedy` or MF-BWI-Fair per call)
  and the reason the whole script runs in ~2.2 minutes rather than tens of
  minutes. See `experiments/common.py` for the exact mechanism. No trial-count
  or graph-size reduction was needed to hit "a few minutes" -- 15 trials x 3
  sweeps completed in 131.8s end to end.

## Beta sweep (backfire intensity; q=0.05, alpha_fair=0.3 fixed)

Time-averaged active nodes (mean ± std over 15 trials), out of 120 total:

| beta | KKT-greedy | Fair-greedy | MF-BWI-Fair |
|---|---|---|---|
| 0.0 | 71.9 ± 5.6 | 73.3 ± 3.3 | 100.4 ± 0.6 |
| 0.1 | 66.6 ± 2.2 | 66.6 ± 3.7 | 100.0 ± 1.3 |
| 0.2 | 61.9 ± 2.5 | 62.1 ± 3.6 | 99.8 ± 1.4 |
| 0.3 | 57.4 ± 3.3 | 57.7 ± 2.3 | 99.6 ± 1.0 |
| 0.5 | 51.9 ± 3.7 | 51.2 ± 2.8 | 99.2 ± 0.7 |
| 0.7 | 45.0 ± 3.5 | 46.1 ± 3.2 | 98.6 ± 1.0 |

Relative drop from beta=0.0 to beta=0.7: KKT-greedy -37.4%, Fair-greedy
-37.1%, **MF-BWI-Fair only -1.7%**.

## Q sweep (recovery rate; beta=0.15, alpha_fair=0.3 fixed)

| q | KKT-greedy | Fair-greedy | MF-BWI-Fair |
|---|---|---|---|
| 0.00 | 76.9 ± 3.4 | 79.7 ± 3.0 | 101.6 ± 0.9 |
| 0.05 | 65.2 ± 3.9 | 65.6 ± 3.3 | 100.2 ± 0.8 |
| 0.10 | 51.2 ± 4.8 | 52.6 ± 3.4 | 98.8 ± 0.6 |
| 0.20 | 30.7 ± 4.2 | 32.0 ± 5.0 | 97.0 ± 0.8 |
| 0.40 | 10.5 ± 2.6 | 7.8 ± 3.5 | 95.6 ± 0.5 |

Relative drop from q=0.0 to q=0.4: KKT-greedy -86.3%, Fair-greedy -90.2%,
**MF-BWI-Fair only -5.9%**.

## Alpha_fair sweep (beta=0.15, q=0.1 fixed)

Total spread (baselines don't use alpha_fair at all, so their numbers are
identical -- and correctly flat -- at every alpha_fair value; only
MF-BWI-Fair's line can move):

| alpha_fair | KKT-greedy | Fair-greedy | MF-BWI-Fair |
|---|---|---|---|
| 0.0 | 51.2 ± 3.2 | 53.4 ± 2.8 | 98.3 ± 0.7 |
| 0.2 | 51.2 ± 3.2 | 53.4 ± 2.8 | 98.8 ± 0.9 |
| 0.4 | 51.2 ± 3.2 | 53.4 ± 2.8 | 98.5 ± 0.9 |
| 0.6 | 51.2 ± 3.2 | 53.4 ± 2.8 | 98.4 ± 0.9 |
| 0.8 | 51.2 ± 3.2 | 53.4 ± 2.8 | 99.0 ± 0.8 |

Min group reach fraction (worst-off group's own-size-normalized reach):

| alpha_fair | KKT-greedy | Fair-greedy | MF-BWI-Fair |
|---|---|---|---|
| 0.0 | 0.274 ± 0.056 | 0.343 ± 0.046 | 0.668 ± 0.025 |
| 0.2 | 0.274 ± 0.056 | 0.343 ± 0.046 | 0.664 ± 0.028 |
| 0.4 | 0.274 ± 0.056 | 0.343 ± 0.046 | 0.664 ± 0.050 |
| 0.6 | 0.274 ± 0.056 | 0.343 ± 0.046 | 0.666 ± 0.039 |
| 0.8 | 0.274 ± 0.056 | 0.343 ± 0.046 | **0.719 ± 0.023** |

The worst-off group is the smallest block (20 nodes) at every alpha_fair
value tested. Per-group reach for MF-BWI-Fair (mean over 15 trials):

| alpha_fair | group0 (n=20) | group1 (n=40) | group2 (n=60) |
|---|---|---|---|
| 0.0 | 0.668 | 0.782 | 0.894 |
| 0.2 | 0.664 | 0.788 | 0.899 |
| 0.4 | 0.664 | 0.779 | 0.901 |
| 0.6 | 0.666 | 0.775 | 0.901 |
| 0.8 | 0.719 | 0.778 | 0.891 |

## Runtime

Mean wall-clock per forward simulate/run call (excludes the amortized,
once-per-trial baseline seed-selection cost, reported separately below):

| algorithm | mean runtime per run |
|---|---|
| KKT-greedy (forward pass through the sequential simulator) | ~0.009-0.012 s |
| Fair-greedy (forward pass through the sequential simulator) | ~0.005-0.011 s |
| MF-BWI-Fair (full T=30-round run: belief fixed point + allocation + Bayes updates every round) | ~0.19-0.20 s |

One-time cost of the one-shot seed *selection* itself (amortized once per
15-trial sweep, not once per parameter value -- see Setup): `celf_greedy`
(k=20, 40 MC sims/candidate) ≈ 0.21 s/trial; `fair_welfare_greedy` (k=20, 20
MC sims/candidate, naive non-lazy greedy) ≈ 1.63 s/trial -- this is the
single largest cost in the whole script (≈ 15 × (0.21+1.63) ≈ 27.6 s per
sweep, ×3 sweeps ≈ 83 s of the total 131.8 s).

## Interpretation

**Does the data support "MF-BWI-Fair degrades gracefully as beta/q grow while
the one-shot baselines' advantage erodes"?** Partially, and it's worth being
precise about which half of that claim holds. MF-BWI-Fair clearly degrades
*gracefully*: essentially flat (-1.7% over the whole beta range, -5.9% over
the whole q range) against the one-shot baselines' steep decay (-37% and up
to -90%, respectively -- q=0.4 nearly kills fair-greedy's spread entirely,
7.8 out of 120 nodes). That part of the theory's prediction holds cleanly and
strongly in this setup.

The "erodes an *existing advantage*" framing is the part that does **not**
hold as stated, because there never is a baseline advantage to erode: even at
beta=0 and q=0.0 (the most favorable setting for the one-shot baselines),
MF-BWI-Fair's spread (100.4, 101.6) already dominates both baselines (71.9-79.7)
by a wide margin. The gap is present from the very first data point and
mostly just widens in *relative* terms as beta/q grow -- there is no
crossover, and the baselines are never ahead of or even close to MF-BWI-Fair.
The most defensible reading of the data: MF-BWI-Fair's advantage comes from
two compounding sources that this experiment does not separate --
(1) genuine robustness to backfire/recovery via repeated re-intervention
(MAINTAIN actions fighting decay, fresh CONVERTs replacing lost nodes), which
is the mechanism the theory is actually about, and (2) simply getting to
spend the full budget B on *every one* of the 30 rounds, vs. the baselines'
one-time B at round 0 -- an advantage baked into the one-shot-vs-sequential
comparison by construction (as instructed: match k so one CONVERT-batch ≈ one
round of MF-BWI-Fair's budget, not so the *cumulative* spend is equal over
T). Readers should not take the absolute gap size as a pure measure of
"handling backfire better" -- the *relative degradation slopes* (the -37%/-90%
vs. -1.7%/-5.9% comparison) are the part of this experiment that isolates the
graceful-vs-fragile-degradation claim, and that comparison does support the
theory.

**Does alpha_fair trade off total spread against the worst-off group's
share?** Only weakly, and mostly not as a *tradeoff*. Total spread is
essentially flat across the whole alpha_fair range (98.3 to 99.0, no
significant trend, well within the ±0.7-0.9 std). Min-group reach is flat
from alpha_fair=0.0 to 0.6 (0.664-0.668, differences smaller than the ±0.03-0.05
std) and only rises clearly at alpha_fair=0.8 (0.719, a jump of about +0.05
over the 0.664-0.668 plateau, and outside the overlapping error bars of the
lower values). So the knob does move the worst-off group's share in the
right direction, but (a) the effect only shows up at the highest alpha_fair
tested here, not as a smooth monotone curve across all five values, and (b)
there is essentially no total-spread cost to pay for it in this setup --
which is a more favorable finding than a real tradeoff, but also weaker
evidence for "controllable, continuous tradeoff" than the theory write-up
frames it. It's also worth flagging a confound: MF-BWI-Fair's min-group reach
(0.664-0.719) is far above both one-shot baselines' (0.274, 0.343) at
*every* alpha_fair value, including alpha_fair=0.0 where its fairness floor
is switched off entirely -- most of that gap is again the same
budget-every-round effect described above, not the fairness mechanism per se.
A cleaner test of the fairness knob in isolation would need either a harder
graph (so groups don't all saturate under generous budget) or a wider/finer
alpha_fair grid concentrated above 0.6.

## Bugs / concerns found in `im_lab` (reported, not fixed)

None. All budget (`spend <= B`) and fairness-floor invariants held on every
one of the 720 total (sweep x algorithm x trial x parameter-value) runs
across this study without any assertion failures, consistent with the
existing test suite. No unexpected exceptions, NaNs, or out-of-range values
were observed in any of the three sweeps.

## Update: repeated-greedy baseline (isolating the real contribution)

The comparison above leaves open whether MF-BWI-Fair's advantage is just
"gets to act every round" rather than anything specific to its own
mechanism. `repeated_greedy` (im_lab/baselines/repeated_greedy.py) closes
that gap: a classical cost-effective greedy that re-selects actions **every
round** under the same per-round budget B, using **true** parameters
directly (no Bayesian uncertainty to overcome, no fairness mechanism) --
the strongest plausible classical competitor that also gets to act
repeatedly. Same graph/budget/horizon/trial count as above; all four
algorithms run together, still 0 invariant violations across the run.

**Time-averaged spread, out of 120 (mean over 15 trials):**

| beta | kkt_greedy | fair_greedy | mf_bwi_fair | repeated_greedy |
|---|---|---|---|---|
| 0.0 | 71.9 | 73.8 | 100.8 | **103.5** |
| 0.1 | 67.0 | 67.7 | 100.2 | **101.4** |
| 0.2 | 61.5 | 61.9 | **99.6** | 98.6 |
| 0.3 | 58.2 | 58.2 | **100.2** | 96.5 |
| 0.5 | 52.1 | 51.8 | **98.8** | 92.1 |
| 0.7 | 44.4 | 45.0 | **98.7** | 89.2 |

| q | kkt_greedy | fair_greedy | mf_bwi_fair | repeated_greedy |
|---|---|---|---|---|
| 0.00 | 77.4 | 81.1 | 101.4 | **105.3** |
| 0.05 | 64.7 | 63.8 | 99.7 | **100.9** |
| 0.10 | 51.9 | 53.7 | **98.2** | 95.7 |
| 0.20 | 34.1 | 33.3 | **97.1** | 85.0 |
| 0.40 | 8.5 | 9.9 | **95.8** | 69.6 |

**This is a real crossover, not a construction artifact.** Both algorithms
get the same per-round budget and act every round -- the only difference is
*how* they decide. Below roughly beta=0.15 / q=0.075, repeated_greedy
(myopic, true-parameter, uncertainty-free marginal-gain greedy, re-run
fresh each round) is at least as good as MF-BWI-Fair, and slightly better
at beta=q=0 (103.5/105.3 vs 100.8/101.4) -- meaning MF-BWI-Fair's mean-field
+ Bayesian machinery carries real overhead when backfire/recovery are mild
enough that naive repeated greedy already handles them fine. Above that
threshold the picture flips sharply: at beta=0.7, MF-BWI-Fair is +11% over
repeated_greedy; at q=0.4, +38% (95.8 vs 69.6, with repeated_greedy having
lost most of its edge over the one-shot baselines entirely). **This is the
honest, specific claim the paper can make: MF-BWI-Fair is not "the best"
unconditionally -- it is the better algorithm specifically once
backfire/recovery are strong enough that myopic repeated intervention stops
being sufficient, which is exactly the regime this paper is about.**

**Fairness (min-group reach fraction) tells a related but distinct story:**
repeated_greedy's fairness (using true parameters, greedily) actually starts
*higher* than MF-BWI-Fair's at low beta/q (e.g. beta=0: 0.843 vs 0.699;
q=0: 0.866 vs 0.721) and degrades faster, but only clearly crosses over in
the q sweep (q=0.2: 0.677 vs 0.640, still ahead; q=0.4: 0.538 vs **0.621**,
MF-BWI-Fair now ahead) -- it does not clearly cross over within the tested
beta range (repeated_greedy stays fairer even at beta=0.7: 0.725 vs 0.672),
though its degradation slope is far steeper (-14% vs -4% from beta=0 to
0.7). Read plainly: MF-BWI-Fair's fairness mechanism does not yet win
outright on this metric within the ranges tested here -- its real,
demonstrated advantage is in graceful degradation under recovery, not in
uniformly better fairness. This should be stated as-is in the paper, not
reframed as a clean win.
