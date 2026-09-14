# IMM Validation: Original Authors' C++ Code vs. This Project's Python Reimplementation

This document records a direct, apples-to-apples comparison between the
**original authors' own C++ implementation of IMM** (Tang, Shi & Xiao,
SIGMOD 2015) and this project's from-scratch Python reimplementation
(`im_lab/baselines/imm.py`), plus `celf_greedy` as a second, independent
reference point, all evaluated on the same fixed graph via independent
Monte Carlo spread simulation.

## 1. What was downloaded

- Source: SourceForge project `im-imm`,
  `https://sourceforge.net/projects/im-imm/files/imm_release_v1.1.zip/download`
- File: `imm_release_v1.1.zip` (1,764,815 bytes, downloaded 2026-09-14)
- Contents: a top-level `IMM/` directory containing `src/imm.cpp` and headers
  (`graph.h`, `infgraph.h`, `imm.h`, `continuous_rrset.h`, `discrete_rrset.h`,
  `weibull.h`, `head.h`, `iheap.h`), a bundled SFMT PRNG library
  (`src/sfmt/`), a `Makefile`, a `readme.md`, a GPLv3 `LICENSE`, a sample
  dataset (`nethept/`, ~15k nodes / ~63k edges), and two **prebuilt** Linux
  binaries (`imm_discrete`, `imm_continuous`) which were not used — a fresh
  build from source was done instead so the comparison is against code
  actually compiled here, not an unverifiable precompiled artifact.
- Contributor/author attribution in the release matches the paper's authors
  (Youze Tang, Xiaokui Xiao); last updated 2017, consistent with the prior
  literature search that identified this release.
- This release only implements IMM itself (no separate TIM/TIM+ sibling
  code was bundled in this zip).

## 2. Build process and fixes needed

Build environment: Ubuntu 24.04.4 LTS, g++ 13.3.0 (already installed on the
target machine; no toolchain installation was needed).

The provided `Makefile` target is a direct g++ invocation:

```
g++ -DDISCRETE src/imm.cpp -Wall -std=c++0x -O3 src/sfmt/SFMT.c -o imm_discrete
```

Running `make imm_discrete` as-is failed with one hard compile error (the
rest of the output was harmless deprecation/parenthesization warnings from
2013-era code, e.g. `std::ptr_fun` deprecation, which do not block
compilation under g++13):

```
src/graph.h:76:20: error: ISO C++ forbids comparison between pointer and integer [-fpermissive]
   76 |         ASSERT(fin != false);
```

This is `FILE *fin = fopen(...); ASSERT(fin != false);` in `Graph::readNM()`
(actually the real read is in `readGraph()`, same pattern at line 76) —
old compilers implicitly converted the pointer to `bool`/`int` for this
comparison; g++13 under C++11 mode now rejects it outright rather than
silently doing the old implicit conversion. **Fix applied** (one line, in
the extracted third-party source only, not in this repo's `im_lab/`):

```diff
- ASSERT(fin != false);
+ ASSERT(fin != nullptr);
```

After this single change, `make imm_discrete` compiled cleanly (all
remaining output was pre-existing deprecation warnings, no errors) and
produced a working `imm_discrete` binary. `imm_continuous` was not needed
for this comparison (this project's model is discrete-time progressive IC)
and was not built. The binary was smoke-tested against the bundled
`nethept/` sample dataset (`./imm_discrete -dataset nethept/ -k 10 -model IC
-epsilon 0.5`), which ran in ~0.05s internal time and returned a plausible
10-node seed set, confirming the build is functionally correct before using
it on this project's own graph.

No other source changes, dependency installs, or Makefile edits were
required. This was the **only** modification made to get the original
authors' code running.

## 3. Graph, parameters, and export

Graph: this project's fixed 120-node, 3-group SBM graph, built exactly as
in `experiments/common.py`:

- `SIZES = [20, 40, 60]`, `P_IN = 0.07`, `P_OUT = 0.008`, `TOPOLOGY_SEED = 42`
  (`graphs.stochastic_block_model_graph(SIZES, p_in=P_IN, p_out=P_OUT,
  seed=TOPOLOGY_SEED)`), giving n=120, m=422 directed edges.

p_plus realization (fixed, so all three algorithms see the identical
weighted graph): `graphs.assign_true_parameters(G, p_plus_range=(0.05,
0.15), q_range=(0.05, 0.25), seed=42)`. **P_PLUS_SEED = 42** is fixed and
documented here explicitly (q is irrelevant to all three algorithms
compared, since all three target plain progressive IC / classical spread —
no backfire/recovery — but a q_range was still required as an argument to
`assign_true_parameters` and an arbitrary in-range value was used).

k = 20 (matches this project's `K` constant in `experiments/common.py`).
epsilon = 0.5 (matches this project's `IMM_EPSILON` constant).

Export: node ids in this project's graph are already dense integers
`0..119`, matching IMM's expected 0-indexed node numbering directly, so no
relabeling was needed. A throwaway script
(`export_imm_dataset.py`, scratchpad only, not added to the repo) wrote:

- `attribute.txt`: `n=120` / `m=422`
- `graph_ic.inf`: one `u v p_plus[u,v]` line per directed edge, using the
  realized p_plus values above.

verified p_plus values in the exported file range [0.0507, 0.1499] as
expected from the (0.05, 0.15) sampling range.

C++ IMM was run as:

```
./imm_discrete -dataset <exported_dir>/ -k 20 -model IC -epsilon 0.5
```

The Python side (`imm_select` and `celf_greedy`) was run via a second
throwaway script (`run_comparison.py`, scratchpad only) on the identical
in-memory graph/`p_plus` (same construction, no export round-trip needed
since Python code reads the networkx graph directly).

## 4. Three-way comparison

Seed sets selected (k=20 each), sorted:

| Algorithm | Seed set |
|---|---|
| **C++ IMM** (original authors' code) | 0, 4, 5, 24, 25, 32, 41, 52, 56, 61, 62, 66, 69, 73, 78, 79, 98, 101, 112, 113 |
| **Python IMM** (`im_lab/baselines/imm.py`) | 0, 5, 7, 8, 21, 28, 39, 45, 57, 58, 60, 64, 67, 71, 75, 79, 84, 92, 109, 117 |
| **CELF greedy** (`im_lab/baselines/kkt_greedy.py`) | 0, 8, 31, 41, 56, 59, 65, 66, 67, 71, 77, 79, 80, 82, 92, 94, 97, 98, 116, 117 |

Node-level overlap (out of 20):

- C++ IMM vs. Python IMM: **3/20** shared nodes ({0, 5, 79})
- C++ IMM vs. CELF greedy: **6/20** shared nodes ({0, 41, 56, 66, 79, 98})
- Python IMM vs. CELF greedy: **7/20** shared nodes ({0, 8, 67, 71, 79, 92, 117})

Independent Monte Carlo spread evaluation — **all three** seed sets were
scored with the SAME evaluator, `im_lab.baselines.kkt_greedy.expected_spread`,
2000 simulations each, plain progressive IC with the same `p_plus` (i.e. none
of the numbers below come from any algorithm's own internal spread
estimate):

| Algorithm | Independent MC spread (2000 sims) |
|---|---|
| C++ IMM (original authors' code) | **32.270** |
| Python IMM (this project) | **32.082** |
| CELF greedy | **32.661** |

Spread spread across all three is within ~1.8% of the middle value —
effectively tied given Monte Carlo noise at this graph size. For reference,
CELF's own internal estimate at selection time (200 sims) was 32.955, close
to its independently re-measured 32.661, and the original paper's IMM
guarantee is a (1-1/e-epsilon)-approximation in expectation, not an exact
match to greedy, so C++ IMM and CELF landing within ~1% of each other and
Python IMM within ~2% of both is consistent with all three legitimately
approximating the same submodular objective well.

Runtimes:

- **C++ IMM** (original authors' code): 0.000825s internal
  (`step1=0.000386s, step2=0.000419s`, reported by the binary's own timer),
  ~0.009s wall including process/file I/O — on a 120-node/422-edge graph
  this is negligible; IMM's near-linear-time advantage is designed to show
  up at much larger scale (its own bundled `nethept` example is
  15k nodes / 63k edges).
- **Python IMM** (`imm_select`): 0.0192s (selection only).
- **CELF greedy**: 1.8049s (selection only, num_sims=200) — orders of
  magnitude slower than either IMM variant, as expected: CELF pays for
  Monte Carlo spread estimation on every marginal-gain evaluation, exactly
  the cost IMM's RR-set machinery is built to avoid.

## 5. Assessment

The low **node-level** overlap between the three seed sets (3/20 to 7/20
pairwise) is expected, not a red flag: on a small (120-node), community
structured (3-block SBM) graph with substantial within-block homophily,
many different sets of ~20 well-placed nodes achieve near-identical spread
because of within-block redundancy (multiple nodes in the same dense block
are close substitutes for each other's marginal coverage) and because IMM's
RR-set sampling and node-selection procedure is inherently randomized (the
Python side additionally uses a different RNG draw than the C++ side, which
has a hardcoded internal PRNG seed `95082`). **Spread**, not node identity,
is the correct thing to compare for influence maximization, and on spread
all three land within under 2% of each other.

**Overall conclusion: this project's Python IMM reimplementation is
trustworthy relative to the original authors' own C++ code.** It selects a
different-but-comparable-quality seed set (32.08 vs. 32.27 expected spread,
a 0.6% gap), matches CELF greedy's spread just as closely as the original
C++ IMM does, and — as expected for IMM vs. classical greedy — is
dramatically faster than CELF (0.019s vs. 1.8s) while the C++ version is
faster still at this small scale (sub-millisecond), consistent with IMM's
near-linear-time design advantage over Monte-Carlo-based greedy. No bug was
found in `im_lab/baselines/imm.py` during this exercise; no files under
`im_lab/` were modified.

## 6. Reproducing this comparison

The two throwaway scripts used are not part of this repository (per the
task's instructions they were kept in the working scratchpad); their full
content — `export_imm_dataset.py` (graph -> `attribute.txt` +
`graph_ic.inf`) and `run_comparison.py` (runs Python `imm_select` /
`celf_greedy`, invokes the compiled `imm_discrete` binary on the exported
data, and does the independent Monte Carlo evaluation) — is reproducible
from the fixed constants documented above (`TOPOLOGY_SEED=42`,
`P_PLUS_SEED=42`, `P_PLUS_RANGE=(0.05,0.15)`, `K=20`, `IMM_EPSILON=0.5`,
`EVAL_NUM_SIMS=2000`) and the exact C++ build command
(`g++ -DDISCRETE src/imm.cpp -Wall -std=c++0x -O3 src/sfmt/SFMT.c -o
imm_discrete`, after the one-line `fin != false` -> `fin != nullptr` fix in
`src/graph.h`).
