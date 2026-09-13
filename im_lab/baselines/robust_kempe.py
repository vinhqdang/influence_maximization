"""Robust Influence Maximization, He, X., & Kempe, D. (2016). "Robust Influence
Maximization." Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD '16), pp. 885-894. Also arXiv:1602.05240
(the arXiv v2, dated June 10, 2016, is the version this file was written against
-- fetched and read directly, section by section including exact algorithm
pseudocode and theorem statements, not worked from memory/recollection or from
the abstract alone).

This addresses a different axis than imm.py: IMM (and kkt_greedy) assume the
edge-activation probabilities p_plus are known exactly and maximize plain
expected spread. He & Kempe instead assume p_plus is *not* known exactly --
the algorithm is given a finite set Sigma of candidate influence functions
(here: candidate p_plus dictionaries on the same graph) and must pick a seed
set that is simultaneously good under all of them, without knowing which one
is "true".

WHAT IS TAKEN DIRECTLY FROM THE PAPER (verified against the actual PDF text)
-----------------------------------------------------------------------------
1. The uncertainty model and objective (paper's Definition 2, Section 3.2):
   given a finite (or, for the "Perturbation Interval" model, effectively
   finite after Lemma 1 -- see below) set Sigma of monotone submodular
   influence functions sigma, maximize

       rho(S) = min_{sigma in Sigma}  sigma(S) / sigma(S*_sigma)

   subject to |S| <= k, where S*_sigma is sigma's own optimal size-k seed
   set. I.e. the objective is the worst-case *fraction of each scenario's
   own optimum* achieved simultaneously, exactly the normalized "don't let
   one hard scenario dominate" form the task description anticipated.

   For the Perturbation Interval special case (per-edge probability
   intervals I_e = [l_e, r_e], the shape of uncertainty this project's
   assign_true_parameters(p_plus_range=...) produces), the paper's Lemma 1
   proves that the worst case over the whole (uncountably infinite) interval
   box is *always* attained by pinning every edge independently to l_e or
   r_e -- i.e. a corner of the box -- so only (up to) 2^|E| "scenarios"
   ever matter in principle. This file does not enumerate all 2^|E| corners
   (intractable); see the "reconstructed" section below for how the
   scenario set is actually built in the tests/wiring around this file.

2. Why plain greedy on rho fails (paper's Section 4.2, "Simple Heuristics"
   and Theorem 2): the paper exhibits a family of instances (a directed
   complete bipartite graph K_{k,m} with m >> k, plus k separate pendant
   pairs (u_i, v_i)) on which both of their two natural heuristics --
   "Single Greedy" (greedy directly on rho, one node at a time) and "All
   Greedy" (solve each scenario's own greedy problem separately and keep
   the single best resulting set) -- have worst-case rho arbitrarily close
   to 0, while the true robust optimum has rho close to 1. Concretely (their
   own reported numbers, k=2, m=100): Saturate Greedy achieves rho=0.985,
   Single Greedy 0.038, All Greedy 0.029. test_robust_kempe.py's
   test_saturate_greedy_beats_single_scenario_greedy_on_two_incompatible_scenarios
   is a small, hand-computable instance built on exactly this template
   (k=2, one bipartite "fan" per scenario plus one deterministic pendant
   pair per seed slot) and demonstrates the same failure mode numerically.

3. Theorem 2 (hardness): unless P=NP, no polynomial algorithm can find a set
   S of size <= (1-delta)*ln|Sigma|*k with rho(S) >= rho(S*)*Omega(1/n^(1-eps)),
   for any constants delta, eps > 0 -- i.e. any nontrivial approximation
   *requires* exceeding the seed budget k by (close to) a ln|Sigma| factor.
   This is why the algorithm below is unavoidably a *bicriteria*
   approximation (it may return more than k seeds).

4. The algorithm (paper's Section 4.1, Algorithm 1 "Saturate Greedy" and
   Algorithm 2 "Greedy Mintss"), a modification of the SATURATE algorithm
   for robust submodular optimization (Krause, McMahan, Guestrin & Gupta,
   "Robust Submodular Observation Selection," JMLR 2008 [ref. 27 in the
   paper]), combined with the multiplicative-error greedy-mincover analysis
   of Goyal, Bonchi & Lakshmanan / Goyal et al. ("GreedyMinCostCover", [ref.
   22 in the paper] -- Note: this is a *different* paper by some of the same
   authors as the "random noise" [21] cited elsewhere in He & Kempe; it is
   not itself about robust influence maximization). The user's task
   description's "SATURATE-style with per-scenario clipping, sum into one
   surrogate, run greedy on the sum" sketch is, per the actual paper text,
   exactly right; reproduced here verbatim from the paper (variable names
   kept identical to the PDF: c, c_min, c_max, gamma/precision, eta,
   epsilon/error, beta):

   Since exact OPT_sigma = sigma(S*_sigma) is itself NP-hard to compute, the
   paper first replaces it with sigma(S^g_sigma), the (1-1/e)-approximate
   value found by ordinary greedy on sigma alone, defining
       rho^g(S) = min_sigma sigma(S) / sigma(S^g_sigma)
   which sandwiches the true objective as (1 - 1/e) * rho^g(S) <= rho(S) <=
   rho^g(S) (paper's Eq. (1)) -- so optimizing rho^g costs only an extra
   factor of (1-1/e), folded into the final guarantee below.

   Algorithm 1 SaturateGreedy(Sigma, k, precision gamma):
       c_min <- 0, c_max <- 1
       while (c_max - c_min) >= gamma:
           c <- (c_max + c_min) / 2
           H^(c)(S) <- sum_{sigma in Sigma} min(c, sigma(S) / sigma(S^g_sigma))
           S <- GreedyMintss(H^(c), threshold = c*|Sigma|, error = c*gamma/3)
           if |S| > beta*k:
               c_max <- c
           else:
               c_min <- c * (1 - gamma/3);  S* <- S
       return S*

   Algorithm 2 GreedyMintss(f, threshold eta, error epsilon):
       S <- {}
       while f(S) < eta - epsilon:
           u <- argmax_{v not in S} f(S union {v})
           S <- S union {u}
       return S
   (f = H^(c) is a nonnegative sum of monotone-submodular, c-truncated
   per-scenario coverage functions, hence itself monotone submodular, so
   plain forward greedy on it is meaningful; the paper's Implementation
   subsection explicitly says "we also apply the CELF optimization ... in
   all cases" to accelerate this greedy loop, which is exactly what
   greedy_mintss() below does via a lazy-forward heap, mirroring the same
   pattern already used in kkt_greedy.celf_greedy and imm.node_selection.)

5. Theorem 3 (the exact bicriteria guarantee, quoted from the paper):
   "Let beta = 1 + ln|Sigma| + ln(3/gamma). Saturate Greedy finds a seed set
   S_hat of size |S_hat| <= beta*k with
       rho(S_hat) >= (1 - 1/e) * rho(S*) - gamma,
   where S* is an optimal robust seed set of size k."
   I.e. alpha = (1 - 1/e), the budget blow-up is beta = 1 + ln|Sigma| +
   ln(3/gamma) (NOT a bare O(log|Sigma|) -- the paper gives this exact
   closed form, including the ln(3/gamma) term from the binary-search
   precision), and there is an additive -gamma slack from the same
   precision parameter. Increasing gamma loosens the guarantee (larger
   slack) but shrinks beta (smaller budget overshoot, and fewer binary-
   search iterations, since the loop runs O(log(1/gamma)) times);
   decreasing gamma tightens the guarantee at the cost of a larger budget.

WHAT IS RECONSTRUCTED / PRAGMATICALLY CHOSEN HERE (flagged, not silently
guessed)
-----------------------------------------------------------------------------
- Scenario-set construction from an interval: the paper's Lemma 1 says the
  true worst case over a Perturbation-Interval uncertainty set lives among
  the 2^|E| corner points of the interval box (all-independent per-edge
  choices of l_e vs r_e), which is exactly enumerable only for tiny graphs.
  This module itself is agnostic to how Sigma is built -- robust_select()
  just takes a list of p_plus-shaped dicts -- but where this file's own
  tests derive Sigma from an interval, they use the two "all-low" /
  "all-high" corners (every edge at l_e, respectively every edge at r_e) as
  a small, illustrative, *non-exhaustive* stand-in for the full corner set,
  not a claim that these two corners are provably the worst case in
  general (Lemma 1's proof is per-edge and pins each edge independently; an
  adversarial instance could in principle need a "mixed" corner). This
  matches the task instructions' own suggestion of using the interval's low
  and high endpoints as "illustrative," and keeps the algorithm's actual
  interface general-purpose.
- sigma(S) and sigma(S^g_sigma) are estimated by direct Monte Carlo (reusing
  kkt_greedy.expected_spread / ic_cascade_once) rather than the paper's
  ConTinEst-based fast estimator for the continuous-time model (irrelevant
  here since this project only has the plain discrete-time progressive-IC
  model) or an exact computation (intractable in general). The paper's own
  footnote 7 explicitly sanctions this: "For influence coverage functions,
  arbitrarily close approximations to f can be obtained by Monte Carlo
  simulations... otherwise the approximations carry through in a
  straightforward way, leading to multiplicative factors (1+delta'')" --
  i.e. the paper itself treats "greedy evaluated via Monte Carlo spread
  estimates" as within the scope of its own analysis, modulo an unstated
  small multiplicative slack this file does not separately track.
- S^g_sigma (the per-scenario greedy solution used as the normalizing
  denominator sigma(S^g_sigma) for rho^g) is computed via
  kkt_greedy.celf_greedy, exactly the "ordinary greedy" the paper's Eq. (1)
  calls for.
- A floor of 1e-9 on each sigma(S^g_sigma) denominator (guards against
  division by zero on a degenerate all-isolated-nodes scenario) and a hard
  stop of GreedyMintss once all n nodes are selected (guards against
  non-termination from Monte Carlo noise pushing H^(c) fractionally below
  its analytically-guaranteed-reachable threshold at S=V) are ordinary
  defensive implementation details, not from the paper.
"""

from __future__ import annotations

import heapq
import math

import networkx as nx
import numpy as np

from im_lab.baselines.kkt_greedy import celf_greedy, expected_spread


def greedy_mintss(
    h_funcs: list,
    nodes: list,
    threshold: float,
    error: float,
) -> list:
    """Algorithm 2 (Greedy Mintss) from the paper: greedily grow S, always
    adding the node with the largest marginal gain in f(S) = sum(h_funcs)(S),
    until f(S) >= threshold - error (or all nodes are exhausted, a defensive
    stop not in the paper -- see module docstring).

    h_funcs is a list of per-scenario callables h_sigma(S) -> float (each
    already clipped at c, i.e. each is H^(c)'s sigma-th summand); f(S) is
    their sum. Uses a lazy-forward (CELF) heap exactly as kkt_greedy.
    celf_greedy and imm.node_selection do, since f is monotone submodular
    (a nonnegative sum of monotone submodular functions) -- the paper's own
    Implementation subsection says they do this too, for the same reason
    (speed).
    """

    def f(S: list) -> float:
        return sum(h(S) for h in h_funcs)

    S: list = []
    cur_val = 0.0
    n = len(nodes)

    # Initial marginal gains: f({v}) - f({}) = f({v}) since f({}) = 0.
    heap = []
    for v in nodes:
        heap.append((-f([v]), v, 0))
    heapq.heapify(heap)

    while heap and len(S) < n and cur_val < threshold - error:
        neg_gain, v, stamp = heapq.heappop(heap)
        if stamp == len(S):
            S.append(v)
            cur_val += -neg_gain
        else:
            new_val = f(S + [v])
            new_gain = new_val - cur_val
            heapq.heappush(heap, (-new_gain, v, len(S)))
    return S


def saturate_greedy(
    G: nx.DiGraph,
    scenarios: list,
    k: int,
    gamma: float = 0.2,
    num_sims: int = 100,
    rng: np.random.Generator = None,
) -> list:
    """Algorithm 1 (Saturate Greedy) from the paper. See module docstring
    for the exact pseudocode this mirrors and for Theorem 3's guarantee:
    returns a seed set of size <= beta*k, beta = 1 + ln|Sigma| + ln(3/gamma),
    with rho(returned set) >= (1 - 1/e) * rho(optimal size-k robust set) -
    gamma.

    scenarios: list of p_plus-shaped dicts on G (the finite set Sigma).
    """
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G.nodes())
    m = len(scenarios)
    if m == 0:
        return []

    beta = 1.0 + math.log(m) + math.log(3.0 / gamma) if m > 1 else 1.0 + math.log(3.0 / gamma)

    # Per-scenario (1-1/e)-approximate greedy solutions S^g_sigma and their
    # spreads sigma(S^g_sigma), used as the normalizing denominators for
    # rho^g (paper's Eq. (1)). Reused across the whole binary search below.
    norms = []
    for p_plus_sigma in scenarios:
        _S_g, spread_g = celf_greedy(G, p_plus_sigma, k, num_sims=num_sims, rng=rng)
        norms.append(max(spread_g, 1e-9))

    def make_h_funcs(c: float) -> list:
        """h_sigma(S) = min(c, sigma(S) / sigma(S^g_sigma)) for each scenario."""
        h_funcs = []
        for p_plus_sigma, norm in zip(scenarios, norms):
            def h(S, p_plus_sigma=p_plus_sigma, norm=norm, c=c):
                if not S:
                    return 0.0
                spread = expected_spread(G, S, p_plus_sigma, num_sims, rng)
                return min(c, spread / norm)

            h_funcs.append(h)
        return h_funcs

    c_min, c_max = 0.0, 1.0
    best_S: list = []
    max_iters = 200  # generous safety cap on the binary search
    it = 0
    while (c_max - c_min) >= gamma and it < max_iters:
        it += 1
        c = (c_max + c_min) / 2.0
        h_funcs = make_h_funcs(c)
        threshold = c * m
        error = c * gamma / 3.0
        S = greedy_mintss(h_funcs, nodes, threshold, error)
        if len(S) > beta * k:
            c_max = c
        else:
            c_min = c * (1.0 - gamma / 3.0)
            best_S = S

    return best_S


def robust_select(
    G: nx.DiGraph,
    p_plus_scenarios: list,
    k: int,
    gamma: float = 0.2,
    num_sims: int = 100,
    rng: np.random.Generator = None,
) -> list:
    """Robust Influence Maximization seed selection (He & Kempe 2016,
    "Saturate Greedy" -- see module docstring for the exact algorithm and
    guarantee, and for which parts are taken directly from the paper vs.
    reasonably reconstructed here).

    p_plus_scenarios: the finite candidate set Sigma -- a list of
    p_plus-shaped dicts on G, each a distinct candidate edge-activation-
    probability assignment (e.g. the low/high corners of a per-edge
    interval, as this project's assign_true_parameters(p_plus_range=...)
    would motivate, or any other finite scenario set).

    Unlike celf_greedy/imm_select, this is a genuinely *bicriteria*
    algorithm (Theorem 2 shows no polynomial algorithm can avoid this): the
    returned seed set may contain MORE than k nodes -- up to
    beta*k = (1 + ln|Sigma| + ln(3/gamma)) * k -- in exchange for a
    (1 - 1/e) approximation (minus an additive gamma slack) to the best
    possible worst-case-across-scenarios spread ratio achievable with
    exactly k seeds. Smaller gamma tightens the guarantee at the cost of a
    larger returned seed set (and more binary-search iterations); this is
    an inherent trade-off of the algorithm, not a tuning knob to "fix" away.

    Returns the selected seed set (list of nodes).
    """
    if rng is None:
        rng = np.random.default_rng()
    return saturate_greedy(G, p_plus_scenarios, k, gamma=gamma, num_sims=num_sims, rng=rng)
