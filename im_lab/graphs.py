"""Graph generators (Erdos-Renyi, Barabasi-Albert, Stochastic Block Model) plus
true-parameter assignment (p_plus, q) for the diffusion model.

All graphs are returned as directed networkx graphs (nx.DiGraph) since p_plus[u,v]
and p_plus[v,u] are independent trial probabilities in this model. Every node carries
a 'group' attribute (int, 0-indexed) used by the fairness machinery; every directed
edge carries a 'p_plus' attribute (float in (0,1)); every node also carries a 'q'
attribute (spontaneous recovery/decay probability).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _to_digraph_both_directions(G_undirected: nx.Graph) -> nx.DiGraph:
    """Turn an undirected graph into a DiGraph with both (u,v) and (v,u) edges.

    Each direction gets its own independent p_plus later on -- influence need not be
    symmetric even though the underlying structural graph (who is connected to whom)
    is generated as undirected for ER/BA.
    """
    D = nx.DiGraph()
    D.add_nodes_from(G_undirected.nodes())
    for u, v in G_undirected.edges():
        if u == v:
            continue
        D.add_edge(u, v)
        D.add_edge(v, u)
    return D


def assign_groups(G: nx.DiGraph, num_groups: int, seed=None) -> nx.DiGraph:
    """Assign an explicit, roughly-balanced random group partition to G's nodes.

    Used for ER/BA graphs, which have no inherent community structure of their own.
    Groups are assigned via a random round-robin-then-shuffle partition so that group
    sizes differ by at most 1 (this is the "explicit partition" option mentioned in
    the spec, as opposed to SBM block labels which are structural).
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    for i, v in enumerate(nodes):
        G.nodes[v]["group"] = i % num_groups
    return G


def erdos_renyi_graph(n: int, p: float, num_groups: int = 2, seed=None) -> nx.DiGraph:
    """Erdos-Renyi G(n, p) structural graph, directed both ways, with group labels."""
    G_und = nx.gnp_random_graph(n, p, seed=seed)
    G = _to_digraph_both_directions(G_und)
    assign_groups(G, num_groups, seed=seed)
    return G


def barabasi_albert_graph(n: int, m: int, num_groups: int = 2, seed=None) -> nx.DiGraph:
    """Barabasi-Albert preferential-attachment graph, directed both ways, with groups."""
    G_und = nx.barabasi_albert_graph(n, m, seed=seed)
    G = _to_digraph_both_directions(G_und)
    assign_groups(G, num_groups, seed=seed)
    return G


def stochastic_block_model_graph(
    sizes: list[int], p_in: float, p_out: float, seed=None
) -> nx.DiGraph:
    """SBM graph: block i has `sizes[i]` nodes; blocks ARE the fairness groups.

    Intra-block edge probability p_in, inter-block edge probability p_out. Group
    labels come directly from SBM block membership (the "SBM community labels"
    option in the spec), so no separate assign_groups call is needed here.
    """
    num_blocks = len(sizes)
    probs = [
        [p_in if i == j else p_out for j in range(num_blocks)] for i in range(num_blocks)
    ]
    G_und = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = _to_digraph_both_directions(G_und)
    # networkx stochastic_block_model stores the block id on each node as 'block'.
    for v in G.nodes():
        G.nodes[v]["group"] = int(G_und.nodes[v]["block"])
    return G


def assign_true_parameters(
    G: nx.DiGraph,
    p_plus_range: tuple[float, float] = (0.05, 0.3),
    q_range: tuple[float, float] = (0.05, 0.25),
    seed=None,
) -> nx.DiGraph:
    """Sample ground-truth p_plus per edge and q per node, uniformly within ranges.

    p_minus is NOT stored on the graph: it is derived at simulation time as
    beta * p_plus[u, v] for a scalar beta (see simulator.py). The Bayesian estimator
    (bayes.py) nonetheless tracks p_plus and p_minus as independent Beta posteriors,
    since the algorithm is not assumed to know beta and only ever sees realized
    activation/deactivation trial outcomes.
    """
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G.edges[u, v]["p_plus"] = float(rng.uniform(*p_plus_range))
    for v in G.nodes():
        G.nodes[v]["q"] = float(rng.uniform(*q_range))
    return G


def group_sizes(G: nx.DiGraph) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for v in G.nodes():
        g = G.nodes[v]["group"]
        sizes[g] = sizes.get(g, 0) + 1
    return sizes


def group_of_map(G: nx.DiGraph) -> dict:
    return {v: G.nodes[v]["group"] for v in G.nodes()}
