import os

import networkx as nx
import pytest

from im_lab import graphs


def test_erdos_renyi_graph_has_groups_and_directed_edges():
    G = graphs.erdos_renyi_graph(30, 0.15, num_groups=3, seed=1)
    graphs.assign_true_parameters(G, seed=1)

    assert G.number_of_nodes() == 30
    groups = {G.nodes[v]["group"] for v in G.nodes()}
    assert groups <= {0, 1, 2}
    for u, v in G.edges():
        assert 0.0 < G.edges[u, v]["p_plus"] < 1.0
    for v in G.nodes():
        assert 0.0 < G.nodes[v]["q"] < 1.0
    # Every undirected connection should appear in both directions.
    for u, v in list(G.edges())[:20]:
        assert G.has_edge(v, u)


def test_barabasi_albert_graph_basic_properties():
    G = graphs.barabasi_albert_graph(25, 2, num_groups=2, seed=2)
    assert G.number_of_nodes() == 25
    sizes = graphs.group_sizes(G)
    assert sum(sizes.values()) == 25
    assert set(sizes.keys()) <= {0, 1}


def test_sbm_graph_group_labels_match_blocks():
    sizes_in = [10, 15]
    G = graphs.stochastic_block_model_graph(sizes_in, p_in=0.4, p_out=0.02, seed=3)
    assert G.number_of_nodes() == 25
    sizes = graphs.group_sizes(G)
    assert sizes[0] == 10
    assert sizes[1] == 15


def test_group_of_map_covers_all_nodes():
    G = graphs.erdos_renyi_graph(10, 0.3, num_groups=2, seed=4)
    gmap = graphs.group_of_map(G)
    assert set(gmap.keys()) == set(G.nodes())


# --- Real-data support: load_edge_list_graph, assign_weighted_cascade_probabilities,
# assign_recovery_rates, assign_communities_as_groups --------------------------------


def test_assign_weighted_cascade_probabilities_hand_computed():
    """Hand-built directed graph: 0->2, 1->2, 3->2, 0->1. in-degree(2)=3,
    in-degree(1)=1; p_plus[(u,2)] should be 1/3 for every u, p_plus[(0,1)] = 1."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 2), (1, 2), (3, 2), (0, 1)])
    graphs.assign_weighted_cascade_probabilities(G)

    assert G.edges[0, 2]["p_plus"] == pytest.approx(1 / 3)
    assert G.edges[1, 2]["p_plus"] == pytest.approx(1 / 3)
    assert G.edges[3, 2]["p_plus"] == pytest.approx(1 / 3)
    assert G.edges[0, 1]["p_plus"] == pytest.approx(1.0)


def test_assign_weighted_cascade_probabilities_on_both_directions_graph():
    """On a symmetric both-directions graph (the shape every generator in this
    module produces), WC reduces to 1/degree(v) since in-degree == out-degree
    == undirected degree for every node."""
    G_und = nx.cycle_graph(5)  # every node has degree 2
    G = graphs._to_digraph_both_directions(G_und)
    graphs.assign_weighted_cascade_probabilities(G)
    for u, v in G.edges():
        assert G.edges[u, v]["p_plus"] == pytest.approx(0.5)


def test_assign_recovery_rates_independent_of_p_plus():
    G = graphs.erdos_renyi_graph(20, 0.2, num_groups=2, seed=7)
    graphs.assign_weighted_cascade_probabilities(G)
    graphs.assign_recovery_rates(G, q_range=(0.1, 0.3), seed=9)
    for v in G.nodes():
        assert 0.1 <= G.nodes[v]["q"] <= 0.3
    # p_plus attributes must still be the WC values, untouched by the q call.
    in_deg = dict(G.in_degree())
    for u, v in G.edges():
        assert G.edges[u, v]["p_plus"] == pytest.approx(1.0 / in_deg[v])


def test_assign_communities_as_groups_two_cliques_one_bridge():
    """Two disjoint 5-cliques joined by a single bridge edge: any reasonable
    community-detection algorithm should recover exactly the two cliques."""
    G_und = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    G_und.add_edge(0, 5)  # the only inter-clique edge
    G = graphs._to_digraph_both_directions(G_und)

    graphs.assign_communities_as_groups(G, method="greedy_modularity")
    groups = graphs.group_of_map(G)
    clique_a = {groups[v] for v in range(5)}
    clique_b = {groups[v] for v in range(5, 10)}
    assert len(clique_a) == 1
    assert len(clique_b) == 1
    assert clique_a != clique_b


def test_assign_communities_as_groups_min_size_merges_small_communities():
    """A big 8-clique plus three separate small dangling triangles: with
    min_size=5 the triangles (size 3 each) should all merge into one residual
    group, leaving 2 groups total instead of 4."""
    G_und = nx.complete_graph(8)
    for base in (100, 200, 300):
        G_und.add_edges_from([(base, base + 1), (base + 1, base + 2), (base, base + 2)])
        G_und.add_edge(0, base)  # keep the whole thing connected
    G = graphs._to_digraph_both_directions(G_und)

    graphs.assign_communities_as_groups(G, method="greedy_modularity", min_size=5)
    sizes = graphs.group_sizes(G)
    assert len(sizes) == 2
    assert sorted(sizes.values()) == [8, 9]


def test_load_edge_list_graph_relabels_and_keeps_largest_component(tmp_path):
    path = tmp_path / "toy.edges"
    # Main connected component: 10-11-12-13 (a path); disconnected island: 900-901.
    path.write_text("10 11\n11 12\n12 13\n900 901\n")

    G = graphs.load_edge_list_graph(str(path))

    assert G.number_of_nodes() == 4  # the 2-node island was dropped
    assert set(G.nodes()) == {0, 1, 2, 3}
    # Both-directions: every structural edge appears in each direction.
    assert G.number_of_edges() == 6  # 3 undirected edges * 2 directions
    for u, v in list(G.edges()):
        assert G.has_edge(v, u)


def test_load_edge_list_graph_ignores_comments_and_self_loops(tmp_path):
    path = tmp_path / "toy2.edges"
    path.write_text("# a comment\n0 1\n1 1\n1 2\n\n2 0\n")

    G = graphs.load_edge_list_graph(str(path))

    assert G.number_of_nodes() == 3
    assert not any(u == v for u, v in G.edges())


def test_load_facebook_348_sample_if_present():
    """If the project's real-graph data file is present (it is checked into
    data/ego-facebook/348.edges for experiments/run_multigraph_validation.py),
    sanity-check its basic shape end to end through the loader + both
    probability/group assignment functions."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "ego-facebook", "348.edges"
    )
    if not os.path.exists(path):
        pytest.skip("data/ego-facebook/348.edges not present")

    G = graphs.load_edge_list_graph(path)
    assert G.number_of_nodes() == 224
    assert nx.is_weakly_connected(G)

    graphs.assign_weighted_cascade_probabilities(G)
    for u, v in G.edges():
        assert 0.0 < G.edges[u, v]["p_plus"] <= 1.0

    graphs.assign_recovery_rates(G, q_range=(0.1, 0.3), seed=1)
    for v in G.nodes():
        assert 0.1 <= G.nodes[v]["q"] <= 0.3

    graphs.assign_communities_as_groups(G, method="greedy_modularity", min_size=15)
    sizes = graphs.group_sizes(G)
    assert sum(sizes.values()) == 224
    assert min(sizes.values()) >= 15
