# Theory: guarantees for fair, non-progressive, backfire-tolerant influence control

This document is the theoretical core of the project. Every statement is
labelled as one of:

- **Theorem / Proposition / Lemma** -- proven here (proof or complete proof sketch given).
- **Cited** -- an external result we rely on, with the exact citation.
- **Conjecture** -- stated precisely, with the honest reason it is not yet proven.

Nothing labelled Conjecture is used as a premise for anything labelled Theorem.

---

## 0. The model (formal version used in all proofs)

Graph $G=(V,E)$, $n=|V|$, $m=|E|$. Groups $G_1,\dots,G_M$ partition $V$.
Rounds $t=0,1,\dots,T$. Active set $A_t\subseteq V$, with $A_0=S$ (the seed set).

Per round, independent randomness:
- for each edge $(u,v)$: a positive trial $X^t_{uv}\sim\mathrm{Bern}(p_{uv})$ and a backfire trial $Y^t_{uv}\sim\mathrm{Bern}(\beta p_{uv})$, for a global backfire intensity $\beta\in[0,1)$;
- for each node $v$: a recovery trial $R^t_v\sim\mathrm{Bern}(q_v)$.

**Transition (layered semantics).** $v\in A_{t+1}$ iff
$$\big[\,v\in A_t \ \wedge\ R^t_v=0 \ \wedge\ \forall u\in A_t\cap N_{\mathrm{in}}(v):\ Y^t_{uv}=0\,\big]
\ \ \vee\ \ \big[\,\exists u\in A_t\cap N_{\mathrm{in}}(v):\ X^t_{uv}=1\,\big].$$

That is: an active node stays active unless it recovers or is backfired by an
active in-neighbour, and *any* node (active or not) is (re)activated by a
successful positive trial from an active in-neighbour.

**Monotonicity of the one-step map -- only at $\beta=0$.** For $\beta=0$ the
map $A_t\mapsto A_{t+1}$ is monotone in $A_t$ for fixed randomness: both
clauses can only gain from more active nodes. This is what Lemma 1.2 rests on.
For $\beta>0$ it is **not** monotone: the backfire clause is anti-monotone (an
additional active in-neighbour can kill $v$), and `tests/test_simulator.py`
pins this with an exact shared-randomness witness at $\beta>0$ alongside the
monotone check at $\beta=0$. The Section 2 proofs never assume monotonicity
of the $\beta$-process itself; the couplings there only use (i) monotonicity
of the two $\beta=0$ surrogates and (ii) the uniform bound $\delta_v$ on the
backfire kill probability, which holds whatever subset of in-neighbours is
active.

> **Modelling note (must be aligned with the code).** The simulator in
> `im_lab/simulator.py` currently draws positive trials only into *inactive*
> targets. That differs from the layered semantics above in exactly one corner
> case: a node that recovers at round $t$ cannot be re-activated in the *same*
> round by a live positive trial. Under the simulator's rule the one-step map is
> not monotone in $A_t$ (having $v$ active and then recovering can leave $v$
> inactive where having $v$ inactive would have let a neighbour activate it), so
> the submodularity lemma below does not apply to it verbatim. The layered
> semantics is the more natural model ("an active neighbour's influence acts
> every round, whether or not you currently agree") and is the one all theorems
> use. **Action item:** align `simulate_step` to the layered semantics and re-run
> the experiments so theory and empirics describe the same process.

Objective (cumulative spread; time-averaged is the same up to a factor $1/T$):
$$f_\beta(S)\;=\;\mathbb E\Big[\sum_{t=1}^{T}|A_t|\Big],\qquad A_0=S.$$

---

## 1. Submodularity of the recovery-only process

**Lemma 1.1 (layered live-edge representation).** Fix $\beta=0$. Build the
layered graph $\mathcal L$ on $V\times\{0,\dots,T\}$ with, for each round $t$:
a *self-edge* $(v,t)\to(v,t{+}1)$ that is live iff $R^t_v=0$, and a
*cross-edge* $(u,t)\to(v,t{+}1)$ for each $(u,v)\in E$, live iff $X^t_{uv}=1$.
Then for every realisation of the randomness,
$$A_t \;=\; \{\,v:\ (v,t)\ \text{is reachable from } S\times\{0\}\ \text{along live edges of }\mathcal L\,\}.$$

*Proof.* Induction on $t$. At $t=0$ both sides equal $S$. If the claim holds at
$t$, then $v\in A_{t+1}$ iff ($v\in A_t$ and self-edge live) or (some
$u\in A_t$ has a live cross-edge into $v$) -- which is exactly "$(v,t{+}1)$ is
reachable via a live edge from a reachable node in layer $t$". $\square$

**Lemma 1.2 (recovery alone preserves monotone submodularity).** For $\beta=0$
and any recovery rates $q_v\in[0,1]$, $f_0(S)$ is monotone and submodular in $S$.

*Proof.* For each fixed realisation, $|A_t|$ is the number of layer-$t$ nodes
reachable from $S\times\{0\}$ in a fixed DAG -- a coverage function of $S$,
hence monotone submodular. Sums (over $t$) and expectations (over realisations)
of monotone submodular functions are monotone submodular. $\square$

*Remark.* This isolates exactly which relaxation is "free" and which is not.
Spontaneous, **state-independent** recovery ($q_v$) is harmless: it is a random
self-edge and reachability stays a coverage function. **Neighbour-dependent**
deactivation (backfire, $\beta>0$) is what destroys the coverage structure:
an active in-neighbour can now *remove* $v$, so the reached set is no longer
monotone in the seeds. This matches, and sharpens, the picture from the
literature (Chan & Ning 2015 lose submodularity under non-progressive
*threshold* rules, where deactivation depends on neighbours' states).

---

## 2. Sandwich approximation for the backfire process

**Notation.** $U(S):=f_0(S)$ is the $\beta=0$ process with the true recovery
rates $q_v$. For a node $v$ let
$$\delta_v \;:=\; 1-\prod_{u\in N_{\mathrm{in}}(v)}(1-\beta p_{uv})\ \le\ \beta\sum_{u\in N_{\mathrm{in}}(v)}p_{uv}\ \le\ \beta\,\Delta_{\mathrm{in}},
\qquad \Delta_{\mathrm{in}}:=\max_v\sum_{u\in N_{\mathrm{in}}(v)}p_{uv},$$
and let $L(S)$ be the $\beta=0$ process with **inflated** recovery
$q'_v := 1-(1-q_v)(1-\delta_v)$ (i.e. an extra independent "kill" w.p.
$\delta_v$ per active round).

**Lemma 2.1 (upper coupling).** Under shared randomness $(X,Y,R)$,
$A^{\beta}_t\subseteq A^{0}_t$ for all $t$. Hence $f_\beta(S)\le U(S)$.

*Proof.* Induction. If $A^\beta_t\subseteq A^0_t$: a node in $A^\beta_{t+1}$
either stayed (so it is in $A^0_t$, did not recover, and the $\beta=0$ process
has no backfire clause to fail) or was activated by a live cross-edge from
$A^\beta_t\subseteq A^0_t$. Either way it is in $A^0_{t+1}$. $\square$

**Lemma 2.2 (lower coupling).** There is a coupling under which
$A^{L}_t\subseteq A^{\beta}_t$ for all $t$. Hence $L(S)\le f_\beta(S)$.

*Proof.* Replace the individual backfire trials on $v$ at round $t$ by a single
uniform $W^t_v$: $v$ is backfire-killed iff
$W^t_v< 1-\prod_{u\in A^\beta_t\cap N_{\mathrm{in}}(v)}(1-\beta p_{uv})$. Conditional
on $A^\beta_t$ this has exactly the right probability, so the $\beta$-process is
unchanged in law. In $L$, let $v$ extra-recover iff $W^t_v<\delta_v$ (same
uniform). Since the $\beta$-kill threshold is at most $\delta_v$ (it is maximised
when *every* in-neighbour is active), **$\beta$-kill $\Rightarrow$ $L$-extra-recover**.
Now induct: a node in $A^L_{t+1}$ either stayed (in $A^L_t\subseteq A^\beta_t$,
$R=0$, and $W\ge\delta_v\ge$ kill threshold, so it is not backfired and stays in
$A^\beta_{t+1}$) or was activated by a live cross-edge from
$A^L_t\subseteq A^\beta_t$. $\square$

**Theorem 2.3 (Sandwich approximation, all $\beta$).** $L$ and $U$ are monotone
submodular (Lemma 1.2), and $L\le f_\beta\le U$ pointwise (Lemmas 2.1-2.2). Let
$S_L,S_U$ be the greedy solutions for $L,U$ under $|S|\le k$, $S_\beta$ the greedy
run directly on $f_\beta$, and $S_{\mathrm{sand}}=\arg\max_{S\in\{S_L,S_U,S_\beta\}}f_\beta(S)$.
Then, with $S^\star$ optimal for $f_\beta$,
$$f_\beta(S_{\mathrm{sand}})\ \ge\ \max\Big\{\frac{f_\beta(S_U)}{U(S_U)},\ \frac{L(S^\star)}{f_\beta(S^\star)}\Big\}\Big(1-\frac1e\Big)f_\beta(S^\star)
\ \ge\ \rho\Big(1-\frac1e\Big)\mathrm{OPT}_\beta,\qquad \rho:=\min_{|S|\le k}\frac{L(S)}{U(S)}.$$

*Proof.* Cited: this is the sandwich approximation theorem of Lu, Chen &
Lakshmanan, "From Competition to Complementarity: Comparative Influence
Diffusion and Maximization", VLDB 2015, applied to the bracket
$L\le f_\beta\le U$; the $(1-1/e)$ factors are Nemhauser-Wolsey-Fisher (1978)
for greedy on each submodular surrogate. The final inequality uses
$L(S^\star)/f_\beta(S^\star)\ge L(S^\star)/U(S^\star)\ge\rho$. $\square$

**Why this beats the previous bound.** The earlier Theorem 4 (Horel & Singer
2016 route) treats backfire as an *arbitrary* perturbation from submodularity
and therefore only holds for $\beta=O(n^{-1/2})$ -- about $0.09$ at $n=120$,
i.e. *not at all* in the $\beta\in[0.1,0.7]$ regime where MF-BWI-Fair wins
empirically. Theorem 2.3 exploits the *structure* of the perturbation (backfire
is a bounded extra kill rate), holds for **every** $\beta$, and recovers the
tight $(1-1/e)$ exactly at $\beta=0$ (where $L=U$, $\rho=1$).

**Instance certification (practical corollary).** Both $L(S)$ and $U(S)$ are
ordinary $\beta=0$ processes we can simulate. So for any instance and any
returned set $S$, the quantity $f_\beta(S)/U(S)$ is a *directly computable*
certificate: $f_\beta(S)\ge\frac{f_\beta(S)}{U(S)}(1-\frac1e)\,\mathrm{OPT}_\beta$.
The algorithm ships its own per-instance approximation guarantee.

### 2.4 Explicit bounds on $\rho$

**Proposition 2.4 (finite-horizon bound, proven).**
$\rho\ \ge\ (1-\beta\Delta_{\mathrm{in}})^{T}$.

*Proof.* Under the coupling of Lemma 2.2, $L$'s live self-edges are exactly
$U$'s live self-edges that also pass the extra-uniform test $W\ge\delta_v$.
Condition on $U$'s realisation and suppose $(v,t)$ is reachable via some path
$\pi$; $\pi$ has $t$ edges of which at most $t$ are self-edges, so $\pi$ stays
live in $L$ with probability at least $(1-\max_v\delta_v)^t\ge(1-\beta\Delta_{\mathrm{in}})^t$.
Hence $\mathbb E|A^L_t|\ge(1-\beta\Delta_{\mathrm{in}})^t\,\mathbb E|A^0_t|$ for each
$t\le T$; sum over $t$. $\square$

This is valid but conservative (it charges every self-edge of a *single*
witness path and ignores re-activation through other paths).

**Conjecture 2.5 (steady-state bound).** In the stationary regime,
$\rho\ \gtrsim\ \dfrac{q}{q+\beta\Delta_{\mathrm{in}}}$ (with $q=\min_v q_v$).

*Heuristic.* A node's active sojourn ends by natural recovery at rate $q$ under
$U$ and at rate $\approx q+\delta$ under $L$, so a single sojourn is shortened
by the factor $q/(q+\delta)$ in expectation
($\mathbb E\min(\mathrm{Geom}(q),\mathrm{Geom}(\delta))=1/(q+\delta-q\delta)$).
*Why it is not yet a proof:* cutting $v$'s sojourn short also removes $v$'s
*downstream* activations in $L$; the per-sojourn ratio bounds $v$'s own
contribution but not the cascade it seeds. The coupling shows the loss is
$\sum_t|A^0_t\setminus A^L_t|$; a rigorous version needs to bound the downstream
loss per extra-kill, e.g. by charging each extra-kill at most the expected
$U$-cascade it would have seeded. This is the target for tightening Prop. 2.4.
*Empirically checkable:* $\rho$ is measurable on any instance by simulating
$L$ and $U$.

---

## 3. Closed-form index for the 2-state / 3-action control problem

Per node, per round (given the mean-field field), the control problem is a
2-state MDP: states $s\in\{0,1\}$, discount $\gamma$, reward $w$ at $s=1$.
Natural transitions $a=p_{01}$ (activate), $b=p_{10}$ (deactivate). Actions:
NONE (free) at either state; CONVERT (cost $c_C$, forces $0\to1$) at $s=0$;
MAINTAIN (cost $c_M$, forces stay at 1) at $s=1$. Lagrange price $\lambda$ per
unit cost. There are exactly **four** deterministic stationary policies,
$(\text{action at }0,\text{action at }1)\in\{N,C\}\times\{N,M\}$.

**Proposition 3.1 (closed forms).** With $D:=(1-\gamma)\big(1-\gamma+\gamma(a+b)\big)$:

| policy | $V_1(\lambda)$ | $V_0(\lambda)$ |
|---|---|---|
| NN | $w(1-\gamma+\gamma a)/D$ | $\gamma a w/D$ |
| NM | $(w-\lambda c_M)/(1-\gamma)$ | $\gamma a(w-\lambda c_M)/\big((1-\gamma)(1-\gamma+\gamma a)\big)$ |
| CN | $(w-\gamma b\lambda c_C)/\big((1-\gamma)(1+\gamma b)\big)$ | $-\lambda c_C+\gamma V_1$ |
| CM | $(w-\lambda c_M)/(1-\gamma)$ | $-\lambda c_C+\gamma(w-\lambda c_M)/(1-\gamma)$ |

*Proof.* Each policy fixes both actions, giving a $2\times2$ linear system.
E.g. CN: $V_0=-\lambda c_C+\gamma V_1$, $V_1=w+\gamma[(1-b)V_1+bV_0]$; substituting,
$V_1[1-\gamma(1-b)-\gamma^2 b]=w-\gamma b\lambda c_C$ and
$1-\gamma(1-b)-\gamma^2b=(1-\gamma)(1+\gamma b)$. NN: eliminating $V_0$ gives
denominator $(1-\gamma+\gamma a)(1-\gamma+\gamma b)-\gamma^2ab=D$. The others are
immediate. $\square$

**Proposition 3.2 (convexity in $\lambda$ -- direct proof).** In a discounted
finite MDP some stationary deterministic policy is optimal at every state
simultaneously, so $V^*(s,\lambda)=\max_{\pi}V_\pi(s,\lambda)$ over the four
policies. Each $V_\pi(s,\cdot)$ is affine in $\lambda$ (Prop. 3.1), so
$V^*(s,\cdot)$ is a maximum of four affine functions: **convex, piecewise
linear, with at most three breakpoints**, all computable in $O(1)$.
(Killian, Perrault & Tambe, AAMAS 2021, Prop. 4.1, prove convexity for general
multi-action restless bandits; here it is a one-line consequence of the
finite enumeration.) $\square$

**Proposition 3.3 (index and indexability, this case).** Define the node's
index $\lambda^*_v$ as the largest $\lambda\ge0$ at which the optimal policy
still takes the paid action at $v$'s current state $s_v$. The set of $\lambda$
at which the paid action is optimal is a down-set $[0,\lambda^*_v]$.

*Proof sketch.* The paid action's advantage over NONE at state $s$ is
$Q_{\text{paid}}-Q_{\text{none}}$; at $s=1$ this equals
$-\lambda c_M+\gamma b\,(V^*_1-V^*_0)$, at $s=0$ it equals
$-\lambda c_C+\gamma(1-a)(V^*_1-V^*_0)$. The cost term has slope $-c<0$ in
$\lambda$. The gap $V^*_1-V^*_0$ is the difference of two convex piecewise-linear
functions with the same breakpoint set; checking the four policies of Prop. 3.1
shows it is non-increasing in $\lambda$ on each piece (the paid actions raise
$V_1$ and $V_0$ by amounts that shrink as $\lambda$ grows). Hence the advantage
is non-increasing in $\lambda$, so its positivity set is a down-set.
*Status:* the sign check on each piece is a finite case analysis; it is carried
out numerically in `tests/` (monotonicity of per-node cost in $\lambda$ on a
fine grid) and should be written out in full for the paper. Because there are
only four policies and $\le3$ breakpoints, it is a mechanical, finite
verification -- which is precisely why indexability is tractable *here* while
Killian et al. call it "notoriously difficult" for general multi-action
restless bandits and avoid it. $\square$

**Theorem 3.4 (per-round complexity).** With Props. 3.1-3.3, one round of the
control policy costs
$$O\big(K\,m\;+\;n\log n\big),$$
where $K$ is the number of mean-field fixed-point iterations (geometric
convergence under the contraction condition $\Delta\max(p^+,p^-)<1$; see
Lemma 2 of the earlier notes). The $n\log n$ term is sorting nodes by
$\lambda^*_v$ and filling the budget greedily; it is *exactly* equivalent to
"smallest $\lambda$ with aggregate cost $\le B$", the target of the bisection it
replaces (which cost $O(n\cdot I_{\text{bisect}}\cdot I_{\text{VI}})$, about
$8\,000\,n$ scalar operations per round in our configuration).

For comparison: naive KKT greedy with Monte Carlo is $O(k\,n\,R\,m)$ per seed
set; IMM (Tang, Shi & Xiao 2015) is $O\big((k+\ell)(n+m)\log n/\varepsilon^2\big)$
for one *static* selection. Our per-round cost is near-linear like a single IMM
run but is paid $T$ times, because the problem is a sequential control problem
-- that is the stated price of sustaining influence under decay rather than
seeding once.

---

## 4. What cannot be improved (so the paper does not claim it)

**Cited 4.1.** For $\beta=0$ (and $q=0$, large $T$) the problem contains
max-$k$-cover, so no polynomial-time algorithm achieves better than $1-1/e$
unless P=NP (Feige 1998); in the value-oracle model this is information-
theoretic (Nemhauser & Wolsey 1978). Adding backfire does not remove these
hard instances (take $\beta$ irrelevant on them), so **$1-1/e$ is an unbeatable
ceiling for every $\beta\ge0$**. This is why MF-BWI-Fair *ties* the strongest
repeated-greedy baseline at $\beta=q=0$ in every experiment: both sit at the
ceiling. Any claim of "winning at zero backfire" would be false.

**Cited 4.2.** For arbitrary non-monotone, non-submodular set functions there
is no constant-factor approximation in general; Horel & Singer (NeurIPS 2016)
give an exponential query lower bound once a function is $\omega(n^{-1/2})$-far
from submodular *adversarially*. Theorem 2.3 escapes this only because our
perturbation is structured (a bounded extra kill rate), which is exactly what
the sandwich exploits.

---

## 5. Lower bound for $\beta>0$

**Setup for this section.** Rounds update synchronously from the *start-of-round*
state (an inactive $v$'s activation and an active $v$'s survival are both
evaluated against $A_t$, not against a partially-updated $A_{t+1}$); this
convention is load-bearing for Lemma 5.2 below and is the one used throughout.
Seeds are active at $t=0$; $q=0$ in this section unless stated.

### 5.1 Inherited hardness, rigorously, including a subtlety

**Theorem 5.1.** For every fixed $\beta\in[0,1)$, no polynomial-time algorithm
approximates $f_\beta$ within $(1-1/e+\varepsilon)$ for any $\varepsilon>0$
unless P=NP.

*Proof.* Reduction from max-$k$-cover, $(1-1/e+\varepsilon)$-inapproximable
unless P=NP (**Cited:** Feige, *JACM* 45(4), 1998, §5; the oracle-model version
of the same bound is Nemhauser & Wolsey, *Math. Oper. Res.* 1978 -- both
citations re-checked only for their headline statement, not re-verified
hypothesis-by-hypothesis). Build a bipartite digraph (set-nodes $\to$
element-nodes, $p=1$), each element replicated $M$ times to swamp the additive
$kT$ seed-count term. The one subtlety: backfire *is* realized on this
instance (a seed keeps attacking every element it covers), so it must be shown
harmless, not merely absent. $\square$

**Lemma 5.2 (exact value on layered/bipartite instances).** If element $v$ has
$c_v(S)\ge1$ active covering seeds throughout, its activity is a two-state
chain with up-rate $1$ and down-rate $1-(1-\beta)^{c_v(S)}$, giving
$$f_\beta(S)\;=\;kT+T\sum_v \varphi_\beta\big(c_v(S)\big)+O(n),\qquad
\varphi_\beta(c):=\frac{1}{2-(1-\beta)^{c}},\quad \varphi_\beta(0):=0.$$
$\varphi_\beta$ is strictly decreasing in $c$ for $c\ge1$ with
$\varphi_\beta(1)=1/(1+\beta)$ and $\varphi_\beta(c)\ge1/2$ for all $c\ge1$
(verified against exact Markov-chain computation, $n\le$ a few hundred states,
$T\sim300$-$400$: predicted vs. computed values agreed, e.g. $5.200$ vs.
$5.202$ on a test instance). A YES (partition) instance thus scores
$nT/(1+\beta)$; any $S$ scores at most
$\mathrm{cover}(S)\cdot T/(1+\beta)+O(n)$ -- the max-coverage gap survives
backfire intact, proving Theorem 5.1. $\blacksquare$

This also shows $f_\beta\ge f_0/2$ **for every $S$** on this instance class,
at $q=0$ -- where the sandwich bound of Theorem 2.3 gives $\rho=0$ (vacuous).
Section 5.3 returns to this gap.

### 5.2 Strict worsening: a conditional constant, and why nothing worse

**Structural fact (why gadgets keep failing).** On every edge, in every round,
$P(\text{$u$ activates }v\mid v\text{ inactive})=p(u,v)\ge\beta\,p(u,v)=
P(\text{$u$ kills }v\mid v\text{ active})$. So a node with a persistently
active in-neighbour is revived at least as fast as that neighbour kills it;
Lemma 5.2 makes this rigorous on layered instances ($\varphi_\beta\ge1/2$).
Backfire only becomes destructive through *synchronous mutual extinction* of a
densely, symmetrically self-attacking set with no external reviver (a
bidirected $K_6$, $p=1,\beta=0.5,q=0$: $f_\beta=0.019$ vs. $f_0=6$, a
$300\times$ collapse). But this mechanism is **indiscriminate**: every seed
choice on that substructure collapses equally, so it encodes no hidden
combinatorial constraint an algorithm could get right or wrong. Concretely,
three natural attempts to turn "who collapses" into a hard-to-decide property
of $S$ all fail for the same underlying reason (the revive-$\ge$-kill
asymmetry above, or the fact that $q$ is a global rather than a per-node
lever): (i) conflict edges between candidate seeds -- the pair dies slowly but
its covered elements mostly survive the death round; (ii) mortal elements via
$q$ -- kills isolated "good" seeds too, indiscriminately; (iii) seeds revivable
by their own elements -- yields an oscillator whose collapse rate is
$\beta^{|\text{set}|}$ regardless of which conflicts are present. No gadget
built this way forces a loss that depends on which poly-time algorithm is run.

**Proposition 5.3 (conditional).** Suppose Feige's NO-instances additionally
satisfy: for every candidate $k$-subset $S$, the number of elements covered
*exactly once* is at most $(1/e+\varepsilon)n$. Then no poly-time algorithm
achieves ratio $c(\beta)(1-1/e)+\varepsilon'$, where
$$c(\beta)(1-1/e)\;=\;\frac1e+\Big(1-\frac2e\Big)\frac{\varphi_\beta(2)}{\varphi_\beta(1)}
\;=\;(1-1/e)\;-\;\Big(1-\frac2e\Big)\frac{\beta(1-\beta)}{1+2\beta-\beta^2}.$$
Numerically $c(\beta)\approx0.968,\,0.946,\,0.939,\,0.940,\,0.960,\,0.981$ at
$\beta=0.1,0.25,0.4,0.5,0.75,0.9$ -- worst near $\beta\approx0.45$, vanishing
at both ends. *Proof sketch:* bound the NO value by
$\varphi_\beta(1)\cdot U+\varphi_\beta(2)\cdot(C-U)$ with $U$ the
once-covered count and $C\le(1-1/e+\varepsilon)n$ the total cover, maximized at
the stated boundary; compare to the YES value $n\varphi_\beta(1)$.

**Status: not a theorem.** The extra property is *not* part of Feige's stated
result, though it is plausible from his construction (a probabilistic-method
partition-system argument: Chernoff plus union bound gives, for any fixed
additive statistic over a random partition, concentration around its
expectation under $\mathrm{Bin}(d,1/k)$ overlaps -- in particular for the
"covered exactly once" statistic, which concentrates near $n/e$ at $d=k$).
Two gaps remain before this is a proof: (i) Feige's block-averaging step uses
concavity of $d\mapsto1-(1-1/k)^d$; the analogous step here needs concavity of
$d\mapsto\mathbb E\,\varphi_\beta(\mathrm{Bin}(d,1/k))$, unchecked (a concave
envelope would still work but weaken the constant); (ii) cross-terms from two
sets in the same partition class need the reduction's pairwise-soundness
property re-verified for this statistic, not just for coverage. Best current
description: **very likely true, not yet proven.**

### 5.3 Is the sandwich bound $\rho$ tight? No -- it is loose, and sometimes vacuous

Exact computation on small instances (2-8 nodes, $T\sim300$):

| instance | $q$ | $\rho$ | $f_\beta(S^\star)/f_0(S^\star)$ | greedy-on-$f_0$ / $\mathrm{OPT}_\beta$ |
|---|---|---|---|---|
| directed star, 6 leaves, $\beta=0.5$ | 0 | 0 (vacuous) | 0.715 | 1.000 |
| directed star | 0.05 | 0.091 | 0.723 | 1.000 |
| directed star | 0.2 | 0.286 | 0.801 | 1.000 |
| two $K_4$ + bridge, $p=0.3,k=2$ | 0.05 | 0.077 | 0.441 | 1.000 |
| bidirected $K_6$, $p=1,\beta=0.5$ | 0.05 | 0.020 | 0.004 | 1.000 |
| 8 random digraphs, $k=2$, $\beta=0.5$ | 0/0.1 | 0-0.14 | 0.24-0.73 | 0.991-1.000 |

Three conclusions, none of them a theorem, all informative for where to spend
future effort:

1. **The sandwich is loose by an order of magnitude** where it applies at all
   (e.g. the lower surrogate $L$ underestimates the star instance's value by
   $13\times$, because it inflates every node's recovery uniformly by $\delta_v$
   even on nodes with no active in-neighbour left) **and vacuous whenever**
   $q+\beta\Delta_{\mathrm{in}}\ge1$ (common) **or $q=0$** (always, since
   $\rho=q/(q+\beta\Delta_{\mathrm{in}})=0$).
2. **$f_\beta/f_0$ is not bounded below by any instance-independent constant**
   (the $K_6$ row: $0.004$) -- so no *multiplicative* comparison against $f_0$,
   sandwich or otherwise, can be the source of a strong general guarantee;
   any tighter bound must compare against $\mathrm{OPT}_\beta$ directly, e.g.
   via the exact layered formula (Lemma 5.2) on instances where it applies, or
   a per-instance certificate as in Theorem 2.3's corollary.
3. **Greedy run on the plain $f_0$ surrogate landed within 1% of $\mathrm{OPT}_\beta$
   on every tested instance**, including ones where the sandwich bound is
   vacuous or far off. This is six small instances, not a proof, but it is
   suggestive evidence that the *practical* quality of surrogate-greedy is far
   better than $\rho(1-1/e)$ certifies, and that a sharper analysis (not
   necessarily a sharper algorithm) is the more promising next step before
   trying to design something new.

**Conjecture 2.5 revisited.** Given 5.3's findings, the steady-state guess
$\rho\gtrsim q/(q+\beta\Delta_{\mathrm{in}})$ should be read as a provable but
*weak* lower bound on the sandwich ratio, not as a target for a matching
hardness result -- Proposition 5.3's constant $c(\beta)\ge0.94$ is far above
$\rho$, and nothing found here suggests $\rho$ itself is the right quantity to
try to match with a lower bound.

### 5.4 Alternative framings

**Why Horel & Singer's oracle bound does not bite (confirms Section 4.2's
claim, does not follow from it).** Our $f_\beta$ is a *known* deterministic
function of a *known* submodular surrogate $f_0$ -- an algorithm may simply
query $f_0$. Horel & Singer's (NeurIPS 2016) exponential-query lower bound
applies to an *adversarial*, planted perturbation designed to be
indistinguishable from submodular under query access; ours is structured and
transparent, so their bound does not apply here (this is a structural
distinction, not a re-derivation of their theorem, which was not re-verified
line-by-line here).

**A more promising unconditional route (open).** The class
$\{S\mapsto\sum_v\varphi(c_v(S))\}$ is closed under the same symmetrization
used in Vondrák's symmetry-gap framework (**Cited**, *SIAM J. Comput.* 2013);
the symmetric-instance gap for our $\varphi_\beta$ works out to
$\mathbb E[\varphi_\beta(\mathrm{Poisson}(1))]/\varphi_\beta(1)$, numerically
$0.589$ at $\beta=0.5$ -- close to, and slightly below, Proposition 5.3's
$0.594$. *If* the symmetry-gap theorem's proof (which uses submodularity in
its smoothing/interpolation step) extends to this specific non-submodular but
structured class, this would give an **unconditional** oracle-model lower
bound matching Proposition 5.3 without needing the unverified property (P).
Whether the smoothing step survives is open and was not checked here; it is
the single most promising next step for turning Section 5.2 into a proof.

### 5.5 Bottom line for this section

$(1-1/e)$-hardness is inherited **rigorously** (Theorem 5.1). A strict,
$\beta$-dependent worsening by a modest constant ($c(\beta)\ge0.94$ across all
tested $\beta$) is reduced to one plausible, precisely-stated, unverified
property of Feige's construction (Proposition 5.3) -- this is the concrete
target for turning "very likely true" into a theorem. Structural evidence
(the per-edge revive-$\ge$-kill asymmetry, and every gadget attempt failing
for the same reason) suggests **no super-constant worsening is achievable for
fixed $\beta$**, and separately, the sandwich ratio $\rho$ from Section 2 is
**not** a natural target for a matching lower bound -- it is empirically loose
by up to an order of magnitude and often vacuous, while greedy-on-$f_0$
performed near-optimally on every small instance tested. The practical
takeaway: effort is better spent sharpening the *upper* bound (a per-instance
or instance-class analysis beating Theorem 2.3, in the spirit of Lemma 5.2)
than searching for a stronger lower bound, which the evidence here says
likely does not exist for fixed $\beta$.

---

## 6. Summary of guarantees (as of this document)

| result | status | statement |
|---|---|---|
| Lemma 1.2 | proven | recovery-only spread is monotone submodular (layered reachability) |
| Lemmas 2.1-2.2 | proven | $L\le f_\beta\le U$ via monotone couplings |
| Theorem 2.3 | proven (cites Lu et al. 2015, NWF 1978) | $f_\beta(S_{\text{sand}})\ge\rho(1-1/e)\,\mathrm{OPT}_\beta$ for **all** $\beta$; instance-certifiable |
| Prop. 2.4 | proven | $\rho\ge(1-\beta\Delta_{\text{in}})^T$ |
| Conj. 2.5 | conjecture | $\rho\gtrsim q/(q+\beta\Delta_{\text{in}})$ (downstream-loss accounting open) |
| Prop. 3.1-3.2 | proven | closed-form affine values; convexity in $\lambda$ by finite enumeration |
| Prop. 3.3 | proof sketch + numerical check | indexability for the 2-state/3-action case |
| Theorem 3.4 | proven | $O(Km+n\log n)$ per round |
| 4.1-4.2 | cited | $1-1/e$ ceiling for every $\beta$; no general constant factor without structure |
| Thm 5.1, Lem 5.2 | proven | $(1-1/e)$-hardness survives backfire exactly; layered instances have $f_\beta\ge f_0/2$ |
| Prop 5.3 | conditional (very likely true, not proven) | strict worsening by $c(\beta)\ge0.94$, given one unverified property of Feige's construction |
| Sec 5.3 finding | empirical (6 instances) | sandwich $\rho$ loose by up to $13\times$, vacuous at $q=0$; greedy-on-$f_0$ within 1% of $\mathrm{OPT}_\beta$ throughout |
| Sec 5.4 | open | symmetry-gap route to an unconditional match of Prop 5.3, contingent on extending Vondrák's smoothing step |
| Sec 7 finding | empirical (this project's 20/40/60 SBM graph, 10-15 trials) | population-unweighted (egalitarian) welfare raises the smallest group's realized reach at neutral $\alpha_{\text{fair}}=0$ by ~0.16 (0.72 -> 0.87 in one 15-trial run) at ~0.5% total-spread cost (occasionally a small gain) |

---

## 7. Fairness weighting: population-weighted vs. population-unweighted welfare

`im_lab/fairness.py` computes the per-group marginal welfare weight $w_g$ that
reweights each node's reward inside the Lagrangian index (Section 3). It
implements Rahmattalabi et al.'s isoelastic (CES) welfare exactly as published:

$$w_g = N_g\, u_g^{\alpha-1}\quad(\alpha\ne 0), \qquad w_g = N_g / u_g\quad(\alpha=0),$$

where $N_g$ is group size and $u_g$ is the group's current time-averaged
realized reach fraction. This form is **population-weighted**: it is the
correct marginal utility for maximizing an aggregate social-welfare objective
$\sum_g N_g u_g^\alpha/\alpha$, in which a group's contribution to total
welfare scales with its population -- appropriate when the goal genuinely is
aggregate welfare (more people benefiting counts for more).

The population weighting has a concrete, measured downside for the stated
fairness goal of *not neglecting a small group*: for group $g$'s weight to
exceed group $g'$'s at $\alpha=0$ requires $u_{g'}/u_g > N_{g'}/N_g$, i.e. the
smaller group must be under-served by a factor proportional to the group-size
*ratio* before the mechanism reacts at all. On this project's own 3-group
(20/40/60) SBM graph, at a neutral $\alpha_{\text{fair}}=0$ this "N-dilution"
kept the smallest group (N=20) at a visibly lower realized reach than the
largest group (N=60) purely because 3x the reach gap is needed before the
weight formula starts favoring it.

`group_welfare_weights` (and `MFBWIFair`/`run_mf_bwi_fair`) now also accept
`population_weighted=False`, which drops the $N_g$ factor and implements the
standard **egalitarian/Rawlsian-style** isoelastic welfare instead -- counting
each group once regardless of size:

$$w_g = u_g^{\alpha-1}\quad(\alpha\ne 0), \qquad w_g = 1/u_g\quad(\alpha=0).$$

This is a well-established alternative welfare form in the social-welfare/
fair-division literature (egalitarian welfare vs. utilitarian/population-
weighted welfare), not a new mechanism. It is the appropriate choice when
fairness means each *group* -- not each *person* -- is treated fairly
regardless of its size.

**Empirical validation** (this project's SIZES=[20,40,60], P_IN=0.07,
P_OUT=0.008, P_PLUS_RANGE=(0.05,0.15) graph; $\beta=0.15$, $q=0.05$, $T=30$,
$B=100$, neutral $\alpha_{\text{fair}}=0$, 15 seeds): switching
`population_weighted` True -> False raised the smallest group's mean realized
reach from 0.72 to 0.87 (+0.16) while total time-averaged spread moved from
103.5 to 104.1 (+0.5%, i.e. no meaningful cost). This substantiates the
intended fix: the egalitarian form protects the small group at a *neutral*
default setting, without needing to push $\alpha_{\text{fair}}$ to an extreme
negative value the way the population-weighted form does.

**Recommendation:** `population_weighted=False` (egalitarian) should be the
default going forward for MF-BWI-Fair's fairness objective. The stated design
goal throughout this project is protecting the worst-off *group*, and the
population-weighted form measurably fights that goal by construction (a small
group can be badly under-served and still not move the weights, simply
because there are few of its members to matter in the aggregate), while the
egalitarian form achieves the intended protection at a neutral $\alpha$
setting and at negligible spread cost. The parameter remains available (and
still defaults to `True` in the current code, for backward compatibility with
existing callers/experiments) so this is a recommendation for the next
default change, not a silent behavior change made here.
