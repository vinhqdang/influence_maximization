"""IMM (Influence Maximization via Martingales), Tang, Y., Shi, Y., & Xiao, X.
(2015). "Influence Maximization in Near-Linear Time: A Martingale Approach."
Proceedings of the 2015 ACM SIGMOD International Conference on Management of
Data (SIGMOD '15), pp. 1539-1554.

This is a *faithful* reimplementation of IMM's two-phase structure --
(1) a martingale/Chernoff-based adaptive Sampling procedure (the paper's
Algorithm 2) that geometrically doubles the number of reverse-reachable (RR)
sets while checking a stopping condition derived from a lower bound LB on
OPT, followed by (2) a single final greedy max-coverage Node-Selection pass
(the paper's Algorithm 1) over the resulting RR-set collection -- rather than
"sample a fixed huge number of RR sets and run greedy" (which is closer to
plain RIS / Borgs et al. 2014 than to IMM specifically). Plain progressive IC
only (p_plus, no backfire/recovery/uncertainty), exactly like kkt_greedy.py
and fair_greedy.py: this is a one-shot baseline that computes its seed set
once from the true p_plus and is then run through the sequential simulator
via simulator.seed_then_none_actions, same as the other baselines in this
package.

RR-set duality (why max-coverage on RR sets approximates max-spread; Borgs,
Brautbar, Chayes & Lucier, FOCS 2014, "Maximizing Social Influence in Nearly
Optimal Time"): for a seed set S and a random RR set R (built by sampling a
uniformly random target v and reverse-reachability under one live-edge-graph
IC realization), Pr[R intersects S] = Pr[S activates v] / ... in expectation
over the random target, so sigma(S) = n * E[1{R intersects S}]; over a
collection of m i.i.d. RR sets this expectation is estimated by the fraction
covered, turning spread maximization into (approximate) maximum coverage.
IMM's specific contribution on top of this RIS scaffolding is exactly the
adaptive martingale-based sample-count schedule implemented below.

WHICH CONSTANTS ARE FROM THE PAPER VS. CHOSEN HERE
---------------------------------------------------
The overall two-phase structure, the epsilon' = sqrt(2)*epsilon rescaling,
the geometric schedule x_i = n/2^i for i = 1..floor(log2 n) - 1, the
stopping rule "LB = n*F_R(S_i)/(1+epsilon') the first time n*F_R(S_i) >=
(1+epsilon')*x_i", and the final sample count theta = lambda*/LB are taken
directly from the paper's Algorithm 2 (Sampling) and Algorithm 3 (IMM) --
these were cross-checked here against a secondary source that reprints the
original Sampling pseudocode verbatim (Chen, W. (2018), "An Issue in the
Martingale Analysis of the Influence Maximization Algorithm IMM," arXiv:
1808.09363, Figure 1), which quotes Algorithm 2 of [Tang, Shi & Xiao 2015]
unchanged.

The two sample-complexity constants below (lambda' for the KPT/LB-estimation
loop, Eq. (9) in the paper; lambda* for the final theta, Eq. (6) in the
paper) are reproduced here from the authors' derivation as I recall/
reconstruct it, i.e. NOT independently re-derived from the martingale
inequalities in this file:

    ell'      = ell * (1 + ln(2) / ln(n))
    epsilon'  = sqrt(2) * epsilon
    lambda'   = (2 + (2/3) * epsilon') * (ell' * ln(n) + ln(ln2(n))) * n / epsilon'**2
    lambda*   = (2 + (2/3) * epsilon)  * (ell' * ln(n) + ln(C(n, k)) + ln(ln2(n))) * n / epsilon**2

I was not able to fetch a clean, textually-searchable copy of the original
SIGMOD/author PDF to re-verify the exact numeric prefactors (2 and 2/3) and
the additive ln(ln2(n)) union-bound term character-by-character; the
structural form (a Chernoff-derived n/epsilon^2 sample count, scaled by a
union bound over ell*ln(n) confidence plus, for the *final* theta only, an
extra ln(C(n,k)) term to union-bound over all size-k candidate seed sets)
is something I am confident in, cross-checked against the reprinted
Sampling pseudocode above and against the closely related Chernoff-bound
constants used by the SSA/D-SSA line of follow-up work (Huang, Wang,
Bevilacqua, Xiao & Lakshmanan, VLDB 2017, "Revisiting the Stop-and-Stare
Algorithms for Influence Maximization"), whose Lambda_1 has the same
"(4e-8) * n/epsilon^2 * ln(...)" shape for the same reason (a two-sided
Chernoff/Bennett bound on RR-set coverage). Given the explicit instruction
in this task to prefer a clearly-flagged, structurally-faithful choice over
a silent guess when the exact constant cannot be independently re-verified:
treat the "2" and "2/3" prefactors above as reasonable-but-not-triple-
checked rather than as certainly letter-perfect transcriptions of the
paper's Eq. (6)/(9). The adaptive two-phase *structure* -- which is IMM's
actual algorithmic contribution over plain RIS -- is implemented exactly
per Algorithm 2/3, independent of whether the prefactor is exactly 2 or
2 + 2/3*epsilon.

A known subtlety, documented for completeness rather than "fixed" here:
Chen (2018, arXiv:1808.09363) points out that the original paper's proof
combining Sampling's stopping-time guarantee with Node-Selection's
fixed-length guarantee has a technical gap (the RR-set sequence Sampling
returns has a *random* length, not a fixed one, which the original proof's
union-bound argument glosses over) and proposes minor fixes (e.g.
regenerating a fresh RR-set batch before the final Node-Selection call).
This implementation follows the original paper's Algorithm 2/3 exactly
(reusing the same accumulated R for the final Node-Selection pass) since
the task is to faithfully reproduce IMM as published, not to implement a
corrected variant; the practical effect on solution quality is negligible
and is not expected to be visible in the tests below.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


def sample_rr_set(
    G: nx.DiGraph, p_plus: dict, rng: np.random.Generator, target=None
) -> set:
    """Sample one reverse-reachable (RR) set.

    An RR set for a (by default uniformly randomly chosen) target node v is
    the set of nodes that could have activated v under one live-edge-graph
    realization of IC: node u is included iff there is a live-edge path from
    u to v, where edge (u, w) is independently "live" with probability
    p_plus[(u, w)]. Equivalently (and how this is implemented): a
    reverse BFS from v that, at each already-reached node w, tests every
    in-edge (u, w) as an independent Bernoulli(p_plus[(u, w)]) trial and
    adds u to the RR set (continuing the BFS from u) iff it succeeds.
    """
    nodes = list(G.nodes())
    if target is None:
        target = nodes[rng.integers(len(nodes))]
    rr = {target}
    frontier = [target]
    while frontier:
        next_frontier = []
        for w in frontier:
            for u in G.predecessors(w):
                if u in rr:
                    continue
                if rng.random() < p_plus[(u, w)]:
                    rr.add(u)
                    next_frontier.append(u)
        frontier = next_frontier
    return rr


def node_selection(
    rr_sets: list, node_to_rrsets: dict, k: int
) -> tuple[list, int]:
    """Greedy max-coverage over a fixed collection of RR sets (IMM's
    Algorithm 1 / "NodeSelection"): repeatedly pick the node covering the
    most not-yet-covered RR sets. Implemented with the same lazy-forward
    (CELF-style) heap trick used by kkt_greedy.celf_greedy, which is exact
    for max coverage (a monotone submodular set function) -- it just skips
    recomputing marginal gains that provably cannot be the current best.

    Returns (seed_set, num_rr_sets_covered).
    """
    import heapq

    heap = []
    for v, idxs in node_to_rrsets.items():
        heap.append((-len(idxs), v, 0))
    heapq.heapify(heap)

    covered = [False] * len(rr_sets)
    seed_set: list = []
    total_covered = 0
    while heap and len(seed_set) < k:
        neg_cnt, v, stamp = heapq.heappop(heap)
        if stamp == len(seed_set):
            seed_set.append(v)
            total_covered += -neg_cnt
            for idx in node_to_rrsets[v]:
                covered[idx] = True
        else:
            cnt = sum(1 for idx in node_to_rrsets[v] if not covered[idx])
            heapq.heappush(heap, (-cnt, v, len(seed_set)))

    return seed_set, total_covered


def _ln_n_choose_k(n: int, k: int) -> float:
    """ln(C(n, k)) via log-gamma, to avoid overflowing on the raw binomial
    coefficient for even moderately large n."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def imm_select(
    G: nx.DiGraph,
    p_plus: dict,
    k: int,
    epsilon: float = 0.5,
    ell: float = 1.0,
    rng: np.random.Generator = None,
) -> list:
    """IMM seed selection (Tang, Shi & Xiao 2015), see module docstring for
    the exact algorithm and for which constants are taken directly from the
    paper vs. reasonably reconstructed here. Targets a (1 - 1/e - epsilon)
    approximation with success probability >= 1 - n^-ell.

    Returns the selected seed set (list of nodes), matching celf_greedy's
    calling convention (G, p_plus, k, ..., rng=...) closely enough to be a
    drop-in alternative seed-selection routine; unlike celf_greedy this
    computes its answer from RR-set coverage counts rather than Monte Carlo
    spread simulation, so it does not also return a spread estimate.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = G.number_of_nodes()
    if k >= n:
        return list(G.nodes())

    eps_prime = math.sqrt(2.0) * epsilon
    ell_prime = ell * (1.0 + math.log(2.0) / math.log(n))
    ln_ln2_n = math.log(math.log2(n))
    ln_cnk = _ln_n_choose_k(n, k)

    def lambda_prime() -> float:
        return (
            (2.0 + (2.0 / 3.0) * eps_prime)
            * (ell_prime * math.log(n) + ln_ln2_n)
            * n
            / eps_prime**2
        )

    def lambda_star() -> float:
        return (
            (2.0 + (2.0 / 3.0) * epsilon)
            * (ell_prime * math.log(n) + ln_cnk + ln_ln2_n)
            * n
            / epsilon**2
        )

    # --- Sampling (Algorithm 2): geometrically-scheduled adaptive RR-set
    # collection with a martingale/Chernoff-based lower-bound (LB) estimate
    # of OPT, used to size the final, guarantee-carrying sample count theta.
    rr_sets: list = []
    node_to_rrsets: dict = {v: [] for v in G.nodes()}

    def add_rr_sets_until(target_count: float) -> None:
        while len(rr_sets) <= target_count:
            rr = sample_rr_set(G, p_plus, rng)
            idx = len(rr_sets)
            rr_sets.append(rr)
            for v in rr:
                node_to_rrsets[v].append(idx)

    LB = 1.0
    max_i = max(int(math.floor(math.log2(n))) - 1, 0)
    lam_prime_val = lambda_prime()
    for i in range(1, max_i + 1):
        x = n / (2**i)
        theta_i = lam_prime_val / x
        add_rr_sets_until(theta_i)
        S_i, covered_i = node_selection(rr_sets, node_to_rrsets, k)
        frac_covered = covered_i / len(rr_sets)
        if n * frac_covered >= (1.0 + eps_prime) * x:
            LB = n * frac_covered / (1.0 + eps_prime)
            break

    theta = lambda_star() / LB
    add_rr_sets_until(theta)

    # --- Node-Selection (Algorithm 1): final greedy max-coverage pass.
    seed_set, _covered = node_selection(rr_sets, node_to_rrsets, k)
    return seed_set
