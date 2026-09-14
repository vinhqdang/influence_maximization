# Related work: verified literature map

This is a consolidated record of the literature search conducted for this
project (six independent search agents, each instructed to verify every
paper via actual fetch/search rather than recollection, flag anything
unverified explicitly, and never fabricate a citation). It exists so the
manuscript draft can cite real, checked sources rather than re-deriving or
guessing citations from scratch. Every entry below was independently
verified with a real URL/DOI at the time of the search (2026-09); recheck
before final submission in case anything has moved.

## 1. The classical baseline

- Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of
  influence through a social network. *KDD 2003*. Defines Independent
  Cascade / Linear Threshold, proves the spread function is monotone
  submodular, and gives the (1-1/e) greedy result via Nemhauser, Wolsey &
  Fisher (1978), *Math. Programming* 14(1). Tight per Feige (1998), *JACM*
  45(4) (max-k-cover hardness).
- Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., &
  Glance, N. (2007). Cost-effective outbreak detection in networks. *KDD
  2007*. CELF: lazy-forward greedy exploiting submodularity.
- Tang, Y., Shi, Y., & Xiao, X. (2015). Influence maximization in
  near-linear time: a martingale approach. *SIGMOD 2015*. IMM: RR-set
  sampling (building on Borgs, Brautbar, Chayes & Lucier, *FOCS 2014*,
  "Maximizing Social Influence in Nearly Optimal Time") plus a
  martingale/Chernoff-based adaptive sample schedule,
  O((k+l)(n+m)log n/eps^2), (1-1/e-eps) guarantee. **Official code exists**:
  SourceForge project "im-imm" (https://sourceforge.net/projects/im-imm/),
  C++, contributor handles matching the authors (Youze Tang, Xiaokui Xiao);
  last updated 2017. This project's own im_lab/baselines/imm.py is an
  independent Python reimplementation, verified against the paper's
  Algorithm 2/3 pseudocode (cross-checked via Chen, W. (2018), "An Issue in
  the Martingale Analysis of the Influence Maximization Algorithm IMM,"
  arXiv:1808.09363, which reprints the Sampling pseudocode and flags a minor
  proof-combination subtlety in the original paper, not fixed here since the
  goal was reproducing IMM as published).

## 2. Fair / group-fairness-constrained IM

- Tsang, A., Wilder, B., Rice, E., Tambe, M., & Zick, Y. (2019).
  Group-fairness in influence maximization. *IJCAI 2019*, arXiv:1903.00967.
  Maximin + diversity-constraint fairness notions; multi-objective
  submodular maximization framework.
- Fish, B., Bashardoust, A., boyd, d., Friedler, S., Scheidegger, C., &
  Venkatasubramanian, S. (2019). Gaps in information access in social
  networks. *WWW 2019*, arXiv:1903.02047. Maximin welfare objective; proves
  it is not monotone submodular in general; hardness + empirical greedy
  heuristic.
- Farnadi, G., Babaki, B., & Gendreau, M. (2020). A unifying framework for
  fairness-aware influence maximization. *WWW 2020 Companion*. Unifies
  maximin/"equality"/"equity" fairness notions as MILP formulations; reports
  "price of fairness"; no submodular approximation guarantee (exact solving
  at small scale only).
- Becker, R., D'Angelo, G., Ghobadi, S., & Gilbert, H. (2020/2021). Fairness
  in influence maximization through randomization. *JAIR* / *AAAI 2021*,
  arXiv:2010.03438. Shows deterministic maximin-fair seeding is hard to
  approximate, but randomized/fractional seeding recovers a
  (1-1/e-eps)-type guarantee.
- **Rahmattalabi, A., Vayanos, P., Fulginiti, A., Rice, E., Wilder, B.,
  Yadav, A., & Tambe, M. (2021). Fair influence maximization: a welfare
  optimization approach. *AAAI 2021*, arXiv:2006.07906.** The paper this
  project's fairness mechanism (im_lab/fairness.py) is built on. Isoelastic
  (CES) welfare $W_\alpha(u) = \sum_c N_c u_c^\alpha/\alpha$ over per-group
  reach; proves monotone submodularity via the Lin & Bilmes (2011, ACL)
  composition lemma (concave-nondecreasing $\circ$ monotone-submodular =
  submodular), giving an exact (1-1/e) under cardinality constraints for the
  classical progressive IC model. Their proof does NOT extend to
  non-progressive/backfire models (verified: the paper does not address
  deactivation or negative influence at all) -- this project adopts their
  *objective form* only, not their guarantee, and additionally identified
  and corrected a population-weighting side effect (their $N_c$ factor
  dilutes protection for small groups; this project's default now drops it,
  see docs/theory.md Section 7). **No official code found** for this paper
  specifically (a same-community-adjacent but distinct repo,
  bwilder0/fair_influmax_code_release, implements Tsang et al. 2019
  instead -- do not conflate the two).
- Rui, X., et al. (2023). Scalable fair influence maximization.
  arXiv:2306.06820. "Welfare fairness" via exponentially-weighted per-group
  influenced fractions + RIS; retains (1-1/e-eps) while scaling.
- Feng, ... (2023). Influence maximization with fairness at scale. *KDD
  2023*, arXiv:2306.01587. Industry-scale, data-driven (learns from
  cascades, no explicit diffusion model) -- explicitly a learning-based
  approach, out of scope for this project's classical-algorithms mandate.

## 3. Non-progressive / recovery-capable IM

- Chan, H., & Ning, ... (2015). Influence maximization under the
  non-progressive linear threshold model. arXiv:1504.00427. Proves
  submodularity is lost on general graphs once nodes can deactivate under
  LT; recovered only on DAGs (1/2 deterministic, 1-1/e randomized). Directly
  informed this project's Lemma 1.2 (recovery alone, via a layered-DAG
  reachability argument, stays monotone submodular on ANY graph -- a
  stronger/different result than Chan & Ning's DAG-restricted LT result,
  since this project's backfire-free recovery process doesn't need the
  threshold-rule structure that breaks their submodularity).
- Golnari, G., Asiaee, T., Banerjee, A., & Zhang, Z. (2014). Revisiting
  non-progressive influence models: scalable influence maximization. *UAI
  2014*, arXiv:1412.5718. "Heat Conduction" model; proves submodularity
  under a specially engineered non-progressive rule; scalable greedy
  (C2GREEDY).
- Lou, Bhagat, Lakshmanan & Vaswani (2014). Modeling non-progressive
  phenomena for influence propagation. arXiv:1408.6466. Continuous-time
  two-state Markov model; an estimation-speed paper, not a
  submodularity/guarantee paper.
- Zahoor, Gillani & Bashir (2024/2025). Influence maximization in temporal
  networks with persistent and reactive behaviors. arXiv:2412.20936. SIS/SIR
  -flavored non-progressive model with reactivation; claims monotone
  submodularity for their specific engineered rule.
- Hui, Wang, Chekol, Rudinac & Zwetsloot (2024). Non-progressive influence
  maximization in dynamic social networks. arXiv:2412.07402. Deep
  reinforcement learning + dynamic graph embedding -- learning-based, out of
  scope for this project.
- Tanınmış, Aras & Altınel (2019). Influence maximization with deactivation
  in social networks. *EJOR* 278, DOI:10.1016/j.ejor.2019.04.010. Bilevel
  leader-follower game (an adversarial second actor deactivates nodes),
  different framing from spontaneous/backfire-driven deactivation.

## 4. Non-monotone / non-submodular / negative-influence IM, and the general toolkit

- Bharathi, S., Kempe, D., & Salek, M. (2007). Competitive influence
  maximization in social networks. *WINE 2007*. First formal competitive-
  cascade model; extends the greedy argument when the opponent's seed set
  is fixed.
- Chen, W., Collins, A., Cummings, R., et al. (2011). Influence maximization
  in social networks when negative opinions may emerge and propagate. *SIAM
  SDM 2011*. IC extended with a "quality factor" so activated nodes can flip
  negative and propagate negativity -- the closest classical model to a real
  backfire mechanism, but engineered to remain monotone submodular (does not
  confront genuine non-submodularity).
- He, X., Song, G., Chen, W., & Jiang, Q. (2012). Influence blocking
  maximization in social networks under the competitive linear threshold
  model. *SIAM SDM 2012*, arXiv:1110.4723. Two competing cascades; stays
  submodular; "sandwich approximation" introduced here as a scalability
  device for the two-cascade setting (this project's own sandwich theorem,
  Section 2 of docs/theory.md, borrows the *technique name/pattern* but
  applies it to a materially different bracket -- a recovery-only process
  vs. this paper's two-competing-cascade bracket -- and cites the general
  method via Lu, Chen & Lakshmanan 2015 below, the more directly applicable
  source for the exact form used).
- Lu, Z., Chen, W., & Lakshmanan, L. V. S. (2015). From competition to
  complementarity: comparative influence diffusion and maximization. *VLDB
  2015*. General sandwich-approximation theorem for a monotone-submodular
  bracket $L \le f \le U$ -- the theorem this project's Theorem 2.3 directly
  instantiates.
- Li, Y., Chen, W., Wang, Y., & Zhang, Z.-L. (2014). Influence diffusion
  dynamics and influence maximization in social networks with friend and
  foe relationships (Polarity Related IM / IC-P model). *WSDM 2013* /
  extended version cited as PLOS ONE (2014), DOI:10.1371/journal.pone.0102199.
  Signed-network IC-P model; proves monotone submodularity is preserved
  under this specific construction.
- "Algorithmic Design for Competitive Influence Maximization Problems."
  arXiv:1410.8664. Survey/derivation of approximation results across
  several competitive-cascade variants.
- Buchbinder, N., Feldman, M., Naor, J., & Schwartz, R. (2012/2015). A tight
  linear time (1/2)-approximation for unconstrained submodular
  maximization. *FOCS 2012* / *SIAM J. Computing* 44(5), 2015. Double-greedy;
  tight 1/2 for unconstrained non-monotone submodular maximization.
- Tukan, M., Mualem, L., & Feldman, M. (2024). Practical 0.385-approximation
  for submodular maximization subject to a cardinality constraint.
  arXiv:2405.13994. Verified directly: 0.385-approx, O(n+k^2) query
  complexity, positioned as the practical alternative to an
  impractical-but-higher (~0.401, Buchbinder-Feldman, cited via secondary
  sources only, not independently re-verified) bound for non-monotone
  submodular maximization under cardinality constraints.
- Bian, A., Buhmann, J., Krause, A., & Tschiatschek, S. (2017). Guarantees
  for greedy maximization of non-submodular functions with applications.
  *ICML 2017*, arXiv:1703.02100. Submodularity ratio $\gamma$ / generalized
  curvature $\alpha$; greedy achieves $(1/\alpha)(1-e^{-\gamma\alpha})$.
  Superseded in this project by the sandwich approach (Theorem 2.3), which
  gives a bound valid for *all* $\beta$ rather than a ratio that must be
  estimated/assumed.
- Horel, T., & Singer, Y. (2016). Maximization of approximately submodular
  functions. *NeurIPS 2016*. Query-complexity bounds for functions
  $\varepsilon$-close to submodular; exponential lower bound once
  $\varepsilon = \omega(n^{-1/2})$ for an *adversarial* perturbation. This
  project's original (superseded) approximation bound used this framework,
  valid only for tiny $\beta$; the current Theorem 2.3 (sandwich) escapes
  this limitation because the backfire perturbation is *structured*
  (a bounded extra kill rate), not adversarial -- confirmed explicitly by
  the hardness investigation (docs/theory.md Section 5.4).

## 5. Bayesian / robust / online IM under parameter uncertainty

- **He, X., & Kempe, D. (2016). Robust influence maximization. *KDD 2016*,
  arXiv:1602.05240.** The paper this project's im_lab/baselines/robust_kempe.py
  reimplements (Saturate Greedy). Worst-case-across-scenarios objective
  $\rho(S)=\min_\sigma \sigma(S)/\sigma(S^\star_\sigma)$; Theorem 2 proves a
  ln|\Sigma|-factor budget blowup is unavoidable (hardness); Algorithm 1/2
  (Saturate Greedy / Greedy Mintss) gives a bicriteria
  $(1-1/e)\rho(S^\star)-\gamma$ guarantee with $\le(1+\ln|\Sigma|+\ln(3/\gamma))k$
  seeds -- verified directly against the paper's own theorem statement and
  pseudocode. **No official code found** for this specific paper (a
  same-author repo, xinranhe/uncertain-infmax, implements a *different*
  paper by the same authors, "Stability of Influence Maximization" /
  "Stability and Robustness in Influence Maximization" -- do not conflate).
- Vaswani, S., Lakshmanan, L. V. S., & Schmidt, M. (2015). Influence
  maximization with bandits. arXiv:1503.00024. First combinatorial-bandit
  treatment; edge/node-level feedback; regret bounds, not approximation
  ratios.
- Wen, Z., Kveton, B., Valko, M., & Vaswani, S. (2017). Online influence
  maximization under independent cascade model with semi-bandit feedback.
  *NeurIPS 2017*, arXiv:1605.06593. IMLinUCB; regret bounds via linear
  generalization across edges.
- Lei, S., Maniu, S., Mo, L., Cheng, R., & Senellart, P. (2015). Online
  influence maximization. *KDD 2015*, arXiv:1506.01188. Explore-exploit
  framework (OIM) for unknown influence probabilities.
- Vaswani, S., Kveton, B., Wen, Z., Ghavamzadeh, M., Lakshmanan, L. V. S., &
  Schmidt, M. (2017). Model-independent online learning for influence
  maximization. *ICML 2017*. Model-agnostic bandit generalization.
- Chen, X., Padmanabhan, A., Lim, G., & Natarajan, K. (2020). Correlation
  robust influence maximization. *NeurIPS 2020*, arXiv:2010.14620.
  Distributionally-robust reformulation over adversarial *correlations*
  between edge activations, not just marginal probabilities; (1-1/e)
  guarantee for the worst-case objective.
- Staib, M., Wilder, B., & Jegelka, S. (2019). Distributionally robust
  submodular maximization. *AISTATS 2019*, arXiv:1802.05249. General
  continuous-optimization-based framework for worst-case-over-ambiguity-set
  submodular maximization.
- Chen, W., Wang, Y., & Yuan, Y. (2013/2016). Combinatorial multi-armed
  bandit: general framework, results and applications / and its extension
  to probabilistically triggered arms. *ICML 2013* / *JMLR* 17, 2016. The
  general CMAB-T framework instantiated by the bandit-IM papers above;
  O(log T) distribution-dependent regret via CUCB.

**Synthesis (from the original search):** the dominant paradigm for
uncertainty in IM is sequential/adaptive online learning via combinatorial
bandits (regret-bounded, requires repeated real interaction) or one-shot
worst-case/distributionally-robust optimization (approximation-ratio-bounded,
conservative, no repeated trials needed). A genuinely single-shot Bayesian
formulation (integrate expected spread over a prior, no adaptivity) exists
only embedded inside the bandit papers' posterior-update machinery (Beta
priors + Thompson Sampling/UCB), not as its own standalone anchor paper --
this remains a real gap this project's Bayesian tracker (im_lab/bayes.py)
sits in.

## 6. Restless-bandit / sequential-control literature (motivates MF-BWI-Fair's control layer)

- Mate, A., Killian, J., Xu, H., Perrault, A., & Tambe, M. (2020).
  Collapsing bandits and their application to public health interventions.
  *NeurIPS 2020*, arXiv:2007.04432. Binary-action (pull/no-pull) restless
  bandits; Whittle index via threshold-policy indexability proofs (Thms 2-3);
  explicitly binary-action only (multi-action listed as future work) --
  this is why this project's control layer is NOT based on this paper's
  index machinery directly.
- **Killian, J. A., Perrault, A., & Tambe, M. (2021). Beyond "to act or not
  to act": fast Lagrangian approaches to general multi-action restless
  bandits. *AAMAS 2021*.** The paper this project's closed-form Lagrangian
  index (im_lab/lagrangian_index.py, im_lab/closed_form_index.py) is built
  on: per-arm Lagrangian-relaxed value function $V(s,\lambda)$, proven
  convex/decreasing in $\lambda$ (their Prop 4.1; this project gives an
  independent, simpler proof for its specific 2-state/3-action case via
  finite enumeration of 4 deterministic policies), avoiding the
  "notoriously difficult" general multi-action indexability problem via
  bisection on $\lambda$ (BLam/SampleLam) instead of a derived index.
- Ou, H.-C., Siebenbrunner, C., Killian, J., Brooks, M., Kempe, D.,
  Vorobeychik, Y., & Tambe, M. (2022). Networked restless multi-armed
  bandits for mobile interventions. *AAMAS 2022*, arXiv:2201.12408.
  Investigated as a possible replacement for this project's mean-field
  network-coupling approximation; found NOT applicable -- their coupling is
  through a deterministic, known action-vector/commuting-matrix (which
  arm was pulled with whom), never through another arm's actual random
  state, letting them avoid a mean-field approximation entirely by
  construction. This project's coupling is genuinely bilinear
  neighbor-*state*-dependent (an active neighbor's actual random state
  changes another node's activation/backfire probability), which their
  method does not cover -- confirming the mean-field approach used here is
  the appropriate tool, not a shortcut to replace.
- Killian, Perrault & Tambe et al., "Beyond Predictors..." /
  "Restless Multi-armed Bandits under Frequency and Window Constraints for
  Public Service Inspections," arXiv:2502.00045 -- adjacent deployment
  line, independent-arms, not directly used.
- Li, D., & Varakantham, P. (2022). Efficient resource allocation with
  fairness constraints in restless multi-armed bandits. arXiv:2206.03883.
  Fairness constraints in the RMAB budget-allocation setting; no Whittle-
  index guarantee given for the fair variant; independent-arms throughout
  (not directly applicable to this project's networked setting, but
  confirms fairness-in-RMAB is an active, separate line of work).

## 7. Very recent (2026) cross-axis combinations found

- "Robust Fair Influence Maximization under Multiple Community Partitions."
  *PACMMOD* (2026). Fairness + parameter/partition uncertainty; bicriteria
  algorithm (Saturate + a "Hit-and-Stop" sampler). The closest published
  2-axis combination to this project's fairness+uncertainty overlap.
- Fang, Q., Shi, J., Rui, X., Zhang, J., & Wang, Z. "Scalable Fair Influence
  Blocking Maximization via Approximately Monotonic Submodular
  Optimization." arXiv:2601.22584 (2026). Fairness + non-monotone
  (blocking-style negative influence); CELF-R algorithm with a
  (1-1/e-$\psi$) guarantee. The closest published 2-axis combination to
  this project's fairness+backfire overlap.
- No paper found combining 3 or 4 axes, and none found combining
  non-progressive/recovery dynamics with any other axis at all -- this
  remains, as of the original search (2026-09), the project's central novel
  contribution claim.

## Positioning statement for the manuscript

No existing paper combines group fairness, non-progressive/recovery
dynamics, non-monotone backfire influence, and Bayesian parameter
uncertainty. The two closest works each combine exactly two of these four
axes (fairness+uncertainty: the 2026 PACMMOD paper above; fairness+backfire:
the 2026 arXiv paper above), both very recent, indicating the field is
actively moving toward combination but has not reached this project's
specific combination. Separately, no paper connects submodular/RIS-style
influence-maximization theory to restless-bandit sequential control theory
at all -- Killian et al. (2021) and Mate et al. (2020) are pure RMAB with no
submodularity argument; this project's Section 3 (docs/theory.md) is, to the
depth of the literature search conducted, the first to do so for a
node-level activation/maintenance control problem.
