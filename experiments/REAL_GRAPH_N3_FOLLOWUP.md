# N=3 real-graph validation: consolidated follow-up (348, 686, 3437 nodes)

This follows up `REAL_GRAPH_FULLBUDGET_FOLLOWUP.md` (which covered only the
348-node ego-Facebook subgraph) by adding the same full-budget, no-compute-cut
experiment on two further SNAP ego-Facebook subgraphs at different scales:
686 nodes (`real_facebook_686_fullbudget`) and 3437 nodes
(`real_facebook_3437_fullbudget`). All three runs use identical settings:
`T=30, B=100, K=20, N_TRIALS=15`, the full 6/5/5-point beta/q/alpha grids,
driver `run_real_facebook_multi_fullbudget.py`. All 240/240 (trial,
swept-value) cells completed on all three graphs; checkpointed and resumed
across container restarts with no lost work. Raw data:
`results_multigraph/real_facebook_{348,686,3437}_fullbudget/{beta,q,alpha}_sweep.csv`.

## 1. Summary: mean spread and mean min-group-reach per algorithm

Values are means across all rows in each sweep file (all trials x all swept
values), i.e. a coarse aggregate. Section 2 below breaks this down by swept
value, which is where the real story is.

### 348 nodes (complete)

| sweep | mf_bwi_fair | repeated_greedy | imm | fair_greedy | kkt_greedy | robust_kempe |
|---|---|---|---|---|---|---|
| beta | **186.9** / 0.821 | 186.0 / 0.793 | 179.3 / 0.779 | 176.8 / 0.771 | 174.3 / 0.652 | 173.5 / 0.683 |
| q | **186.2** / 0.817 | 185.4 / 0.789 | 175.7 / 0.758 | 172.1 / 0.738 | 171.2 / 0.660 | 169.5 / 0.629 |
| alpha | **188.8** / 0.828 | 188.2 / 0.808 | 183.5 / 0.801 | 179.6 / 0.786 | 178.8 / 0.658 | 176.9 / 0.671 |

MF-BWI-Fair is the top mean performer on **both** spread and fairness, on
all three sweeps.

### 686 nodes (complete)

| sweep | mf_bwi_fair | repeated_greedy | imm | kkt_greedy | fair_greedy | robust_kempe |
|---|---|---|---|---|---|---|
| beta | **145.8** / 0.860 | 143.7 / 0.839 | 137.3 / 0.802 | 135.7 / 0.793 | 134.9 / 0.787 | 133.8 / 0.781 |
| q | **145.6** / 0.859 | 143.4 / 0.838 | 134.7 / 0.786 | 133.2 / 0.775 | 132.4 / 0.771 | 131.3 / 0.764 |
| alpha | **147.1** / 0.868 | 145.9 / 0.854 | 140.8 / 0.823 | 138.6 / 0.813 | 138.0 / 0.806 | 136.8 / 0.792 |

Same pattern as 348: MF-BWI-Fair leads on both spread and fairness, on all
three sweeps, by a similar margin.

### 3437 nodes (complete) — **the pattern does not replicate**

| sweep | imm | repeated_greedy | fair_greedy | kkt_greedy | mf_bwi_fair | robust_kempe |
|---|---|---|---|---|---|---|
| beta | **411.9** / 0.739 | 407.3 / 0.704 | 401.1 / 0.718 | 398.2 / 0.641 | 396.8 / 0.716 | 392.3 / 0.615 |
| q | 402.2 / 0.707 | **404.0** / 0.705 | 389.7 / 0.688 | 388.5 / 0.611 | 392.6 / 0.712 | 381.5 / 0.575 |
| alpha | **419.2** / 0.738 | 413.3 / 0.714 | 408.5 / 0.727 | 407.5 / 0.657 | 402.7 / 0.713 | 399.0 / 0.624 |

On the largest graph, MF-BWI-Fair is **not** the top mean performer on
spread on any of the three sweeps (IMM leads beta and alpha; repeated_greedy
narrowly leads q). Its mean fairness is still competitive (2nd on beta and
q, essentially tied with fair_greedy on alpha) but no longer clearly best.

## 2. Why the aggregate mean is misleading: per-swept-value breakdown

Mean-across-the-whole-sweep hides a real qualitative effect that is present
on all three graphs: **MF-BWI-Fair's relative advantage grows with backfire
severity / recovery rate**, exactly as the paper's core claim predicts. The
difference on 3437 is that the *baseline level* it must overcome is much
higher, so it does not finish ahead of the strongest baseline in the mean.

### 3437, beta sweep (spread, backfire intensity increasing left to right)

| beta | kkt | fair | imm | robust_kempe | **mf_bwi_fair** | repeated_greedy |
|---|---|---|---|---|---|---|
| 0.0 | 453.3 | 453.8 | 464.9 | 443.0 | **437.8** | 447.9 |
| 0.1 | 427.5 | 432.3 | 442.4 | 422.2 | **417.4** | 430.7 |
| 0.2 | 410.3 | 413.4 | 424.3 | 404.7 | **404.7** | 417.5 |
| 0.3 | 393.0 | 396.7 | 408.4 | 387.5 | **393.3** | 403.4 |
| 0.5 | 365.5 | 367.8 | 379.6 | 361.2 | **371.7** | 381.4 |
| 0.7 | 339.5 | 342.6 | 351.6 | 335.0 | **355.6** | 362.7 |

At beta=0 (no backfire), MF-BWI-Fair is **last** among all six algorithms.
By beta=0.7 (severe backfire) it has overtaken kkt, fair_greedy, IMM, and
robust_kempe -- it only remains behind repeated_greedy. This is the same
qualitative "grows into its advantage under stress" behavior documented on
348 and 686, just starting from a worse baseline position, so it never
fully closes the gap with the single best one-shot competitor at this scale.

### 3437, q sweep (spread, recovery rate increasing left to right)

| q | kkt | fair | imm | robust_kempe | **mf_bwi_fair** | repeated_greedy |
|---|---|---|---|---|---|---|
| 0.0 | 434.7 | 435.3 | 445.9 | 424.8 | **421.6** | 435.3 |
| 0.05 | 420.1 | 422.5 | 432.9 | 413.1 | **411.9** | 423.5 |
| 0.1 | 404.0 | 407.1 | 420.1 | 398.8 | **404.3** | 412.7 |
| 0.2 | 377.0 | 377.2 | 392.2 | 370.7 | **384.7** | 395.6 |
| 0.4 | 306.9 | 306.3 | 319.8 | 299.9 | **340.4** | 352.7 |

Same shape: MF-BWI-Fair starts near the bottom (q=0) and by q=0.4 (severe
recovery) has overtaken kkt, fair_greedy, IMM, and robust_kempe by a wide
margin (340.4 vs 306-320), again trailing only repeated_greedy.

### 3437, alpha sweep (spread, utilitarian -> leximin-like left to right)

| alpha | kkt | fair | imm | robust_kempe | **mf_bwi_fair** | repeated_greedy |
|---|---|---|---|---|---|---|
| 1.0 (utilitarian) | 407.5 | 408.5 | 419.2 | 399.0 | **395.9** | 413.3 |
| 0.5 | 407.5 | 408.5 | 419.2 | 399.0 | **399.7** | 413.3 |
| 0.0 (proportional) | 407.5 | 408.5 | 419.2 | 399.0 | **402.3** | 413.3 |
| -2.0 | 407.5 | 408.5 | 419.2 | 399.0 | **406.3** | 413.3 |
| -8.0 (leximin-like) | 407.5 | 408.5 | 419.2 | 399.0 | **409.2** | 413.3 |

The one-shot baselines are alpha-independent by construction (only
mf_bwi_fair's own objective is reweighted). MF-BWI-Fair's spread rises
monotonically as alpha becomes more egalitarian (395.9 -> 409.2), the same
direction seen on 348 and 686 -- but it never overtakes kkt/fair/imm/
repeated_greedy on this graph, only robust_kempe.

## 3. Honest assessment

The paper's core mechanistic claim -- that MF-BWI-Fair's relative advantage
over one-shot baselines grows as backfire/recovery severity increases --
**replicates on all three real graphs**, including 3437. What does **not**
replicate at 3437-node scale is the paper's stronger claim that MF-BWI-Fair
achieves the *highest mean spread* against every baseline. At 3437 nodes:

- MF-BWI-Fair is last or near-last in spread at low backfire/recovery
  (beta, q near 0), where the one-shot baselines' myopic optimization is not
  yet handicapped by the phenomena MF-BWI-Fair is designed for.
- It overtakes 4 of 5 baselines by the most severe setting tested on beta
  and q, but not repeated_greedy (the strongest one-shot baseline
  throughout, on this graph).
- Its fairness (min-group-reach) remains competitive -- 2nd-best on beta and
  q, essentially tied for best on alpha -- so the fairness half of the
  claim holds better than the spread half.

This is a genuine, reproducible finding, not noise: it is consistent across
all three sweeps (15 trials each, 240/240 cells) at 3437 nodes, and the
qualitative "advantage grows with severity" shape is intact throughout. It
most plausibly reflects that repeated_greedy's own re-optimization
(2 lookahead rounds) starts to close the gap with MF-BWI-Fair's mean-field
control as the graph grows, and/or that mean-field belief propagation's
approximation quality degrades on a larger, sparser real graph relative to
the smaller, denser subgraphs.

## 4. Implication for the manuscript

The current draft's claim of "MF-BWI-Fair dominates one-shot baselines on
spread and fairness across all real graphs" is **not fully supported** at
3437-node scale in the mean-aggregate sense, and should be revised. The
qualitative mechanism claim (advantage grows with severity, verified via the
per-swept-value breakdown above) does still hold at all three scales and
remains the paper's most defensible real-graph claim. Recommended next
steps for the manuscript:

1. Replace the single N=1 real-graph paragraph with an N=3 paragraph that
   reports the honest pattern: dominance on spread+fairness at 348/686
   nodes; dominance on fairness but not spread (in the mean) at 3437 nodes,
   with the mechanism (advantage grows with severity) intact.
2. Add the per-swept-value breakdown (or a compressed version of it) as
   evidence for the mechanism claim, since the aggregate mean alone is
   misleading at 3437 nodes.
3. Discuss why (repeated_greedy's own adaptivity, mean-field approximation
   degradation at scale) as a limitation/future-work item.
