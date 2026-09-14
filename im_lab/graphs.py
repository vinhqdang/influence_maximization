"""Graph generators (Erdos-Renyi, Barabasi-Albert, Stochastic Block Model) plus
true-parameter assignment (p_plus, q) for the diffusion model.

All graphs are returned as directed networkx graphs (nx.DiGraph) since p_plus[u,v]
and p_plus[v,u] are independent trial probabilities in this model. Every node carries
a 'group' attribute (int, 0-indexed) used by the fairness machinery; every directed
edge carries a 'p_plus' attribute (float in (0,1)); every node also carries a 'q'
attribute (spontaneous recovery/decay probability).

Also included, for bringing a REAL (non-synthetic) graph into this same pipeline
(see experiments/run_multigraph_validation.py and experiments/MULTIGRAPH_VALIDATION.md):
  - `load_edge_list_graph`: load a structural graph from a plain SNAP-style
    edge-list file.
  - `assign_weighted_cascade_probabilities`: the standard "Weighted Cascade"
    p_plus convention (Kempe-Kleinberg-Tardos-era; Chen, Wang & Yang 2009) for
    graphs without measured influence probabilities, as an alternative to
    `assign_true_parameters`'s uniform-random synthetic convention.
  - `assign_recovery_rates`: q-only sampling, for pairing with the above (q
    still needs *some* distribution even on a real graph, since real datasets
    don't ship with recovery rates either).
  - `assign_communities_as_groups`: community detection as a structural PROXY
    for demographic fairness-group labels, which real graphs don't carry.
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


def load_edge_list_graph(path: str) -> nx.DiGraph:
    """Load an undirected structural graph from a plain whitespace-separated
    edge-list text file -- the format SNAP's per-ego-network `.edges` files use
    (one 'u v' pair per line; '#'-prefixed / blank lines are ignored). Used to
    bring a REAL social-network graph (as opposed to every generator above,
    which is synthetic) into this package.

    Node ids are relabeled to a contiguous 0..n-1 range, in ascending order of
    the original id (numeric order when the ids parse as ints, which SNAP's
    ids always do; lexicographic order otherwise) so the relabeling is
    deterministic and reproducible from the raw file alone -- the rest of this
    package assumes small contiguous integer node ids, but SNAP ids are
    arbitrary large integers.

    If the raw file is disconnected, only the LARGEST connected component is
    kept: an isolated island can never be reached by any seed set placed
    elsewhere, so it would just be dead weight in every Monte Carlo simulation
    (and would silently shrink every group's *reachable* size below what
    `group_sizes` reports). This is a no-op when the file is already one
    connected component.

    Deliberately does NOT assign 'group', 'p_plus', or 'q' attributes -- unlike
    the synthetic generators above (which bundle group assignment in because
    ER/BA graphs have no community structure of their own to fall back on),
    a real graph's group labels and edge probabilities are each a separate,
    independently-documented modeling choice here: see
    `assign_communities_as_groups` and `assign_weighted_cascade_probabilities`
    / `assign_recovery_rates`.
    """
    edges: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            edges.append((parts[0], parts[1]))
    if not edges:
        raise ValueError(f"no edges found in {path!r}")

    H = nx.Graph()
    H.add_edges_from(edges)
    H.remove_edges_from(list(nx.selfloop_edges(H)))

    largest_cc = max(nx.connected_components(H), key=len)
    H = H.subgraph(largest_cc).copy()

    try:
        old_ids = sorted(H.nodes(), key=int)
    except ValueError:
        old_ids = sorted(H.nodes())
    mapping = {old: i for i, old in enumerate(old_ids)}
    H = nx.relabel_nodes(H, mapping)

    return _to_digraph_both_directions(H)


def assign_weighted_cascade_probabilities(G: nx.DiGraph) -> nx.DiGraph:
    """Assign per-edge activation probabilities via the "Weighted Cascade" (WC)
    convention: p_plus[(u, v)] = 1 / in-degree(v).

    WC is one of the two standard conventions the IM literature has used ever
    since Kempe, Kleinberg & Tardos (2003) for graphs that come with real
    structure but no measured influence probabilities (the other being
    uniform/trivalency assignment) -- it is named and used explicitly as
    "Weighted Cascade" in Chen, Wang & Yang, "Efficient Influence Maximization
    in Social Networks" (KDD 2009), Section 4, and has remained the standard
    non-uniform default since (e.g. it is one of the two probability models
    IMM -- Tang, Shi & Xiao, SIGMOD 2015, already reimplemented in
    `baselines/imm.py` -- is evaluated under in that paper's own experiments).
    It is used here in place of `assign_true_parameters` for the REAL
    (non-synthetic) graph, since a real graph has no ground-truth influence
    probabilities to draw from a hand-tuned range -- WC instead derives a
    probability directly and deterministically from the graph's own observed
    structure (a more-connected target is proportionally *harder* for any one
    neighbour to activate).

    Deterministic given G (no `seed` parameter, unlike `assign_true_parameters`):
    every edge's probability is a fixed function of the graph's in-degrees, not
    a random draw. A node with in-degree 0 has no incoming edges, so there is
    nothing to set for it.
    """
    in_deg = dict(G.in_degree())
    for u, v in G.edges():
        G.edges[u, v]["p_plus"] = 1.0 / in_deg[v]
    return G


def assign_recovery_rates(
    G: nx.DiGraph, q_range: tuple[float, float] = (0.01, 0.05), seed=None
) -> nx.DiGraph:
    """Sample q (spontaneous recovery/decay probability) per node uniformly
    within q_range -- the q half of `assign_true_parameters`, factored out on
    its own so it can be combined with a p_plus assignment that is NOT itself
    a random draw from a range (namely
    `assign_weighted_cascade_probabilities` on a real graph, whose p_plus
    values come from fixed graph structure and should never be re-drawn per
    trial the way the synthetic graphs' p_plus is).

    `assign_true_parameters` (used unchanged by every synthetic-graph
    experiment) is NOT reimplemented in terms of this function: it draws
    p_plus and q from one shared, continuing RNG stream in a specific order
    that existing experiments' reproducibility depends on (see
    experiments/common.py's module docstring) -- changing that internal
    plumbing is out of scope here and this function's own RNG is intentionally
    independent of it.
    """
    rng = np.random.default_rng(seed)
    for v in G.nodes():
        G.nodes[v]["q"] = float(rng.uniform(*q_range))
    return G


def assign_communities_as_groups(
    G: nx.DiGraph, method: str = "greedy_modularity", seed=None, min_size: int | None = None
) -> nx.DiGraph:
    """Partition G into fairness groups via community detection, as a
    STRUCTURAL PROXY for demographic groups on a real graph that (unlike the
    synthetic SBM graphs, whose blocks ARE the groups by construction) carries
    no demographic labels at all.

    IMPORTANT: this is a stand-in, not a claim that detected communities
    correspond to any real demographic attribute (age, gender, ethnicity,
    ...) of the underlying ego-network's users -- SNAP's ego-network files
    carry no such attributes. It merely gives the fairness machinery, which
    needs *some* partition of nodes into groups, a principled and
    reproducible one grounded in the graph's own connectivity structure
    (densely-connected social clusters), rather than an arbitrary or uniformly
    random one.

    method="greedy_modularity" (default): networkx's
    `greedy_modularity_communities` (Clauset, Newman & Moore, 2004) --
    deterministic, so `seed` is accepted but unused for this method.
    method="louvain": networkx's `louvain_communities` (Blondel et al., 2008)
    -- stochastic tie-breaking internally, so pass `seed` for reproducibility.

    Community detection is run on the UNDIRECTED collapse of G (this
    package's graphs always carry both (u, v) and (v, u) for every structural
    edge, so this collapse is lossless).

    `min_size`, if given, merges every community SMALLER than it into one
    single extra group (placed last), instead of leaving a long tail of
    near-singleton groups whose reach-fraction is far too coarse (e.g. a
    2-node group can only ever show a min-group-reach of 0, 0.5 or 1) to
    support a meaningful fairness comparison. The remaining (large-enough)
    communities are otherwise left exactly as detected, sorted largest-first
    purely so group ids are stable/interpretable. Leave as None to keep every
    detected community as its own group, however small.
    """
    H = G.to_undirected()
    if method == "greedy_modularity":
        communities = [set(c) for c in nx.algorithms.community.greedy_modularity_communities(H)]
    elif method == "louvain":
        communities = [set(c) for c in nx.algorithms.community.louvain_communities(H, seed=seed)]
    else:
        raise ValueError(f"unknown method {method!r}")

    if min_size is not None:
        big = sorted((c for c in communities if len(c) >= min_size), key=len, reverse=True)
        leftover = {v for c in communities if len(c) < min_size for v in c}
        communities = big + ([leftover] if leftover else [])

    for gid, community in enumerate(communities):
        for v in community:
            G.nodes[v]["group"] = gid
    return G


def group_sizes(G: nx.DiGraph) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for v in G.nodes():
        g = G.nodes[v]["group"]
        sizes[g] = sizes.get(g, 0) + 1
    return sizes


def group_of_map(G: nx.DiGraph) -> dict:
    return {v: G.nodes[v]["group"] for v in G.nodes()}
