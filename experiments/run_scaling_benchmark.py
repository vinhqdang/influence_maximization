"""Runtime-scaling benchmark: MF-BWI-Fair per-round wall-clock time vs. graph size n,
at fixed average degree, to empirically substantiate the O(Km+n log n) claim
(Theorem: Per-round complexity). Uses Barabasi-Albert graphs (m=3, avg degree ~6)
so average degree stays roughly constant as n grows -- isolating the n-dependence.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from im_lab.graphs import barabasi_albert_graph
from im_lab.mf_bwi_fair import MFBWIFair

SIZES = [100, 300, 1000, 3000, 10000, 30000, 100000]
M_PARAM = 3  # BA attachment parameter -> avg degree ~ 2*M_PARAM, roughly constant
REPEATS = 5

rng_seed = 42
results = []
for n in SIZES:
    G = barabasi_albert_graph(n, M_PARAM, num_groups=3, seed=rng_seed)
    n_edges = G.number_of_edges()
    budget = max(20, n // 20)
    policy = MFBWIFair(G, budget=budget, alpha_fair=0.0, solver="closed_form")

    rng = np.random.default_rng(rng_seed)
    nodes = list(G.nodes())
    # A nontrivial initial state: activate ~5% of nodes at random so mean-field
    # beliefs and the index computation are not degenerate all-zero.
    state = {v: False for v in nodes}
    n_init_active = max(1, n // 20)
    for v in rng.choice(nodes, size=n_init_active, replace=False):
        state[v] = True

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        policy.choose_actions(state, rng=rng)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean_t = sum(times) / len(times)
    results.append((n, n_edges, budget, mean_t))
    print(f"n={n:7d} m={n_edges:8d} B={budget:6d}  mean_round_time={mean_t:.4f}s  (over {REPEATS} reps)")

print("\nCSV:")
print("n,m,budget,mean_round_time_s")
for n, m, b, t in results:
    print(f"{n},{m},{b},{t:.6f}")
