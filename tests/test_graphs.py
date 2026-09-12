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
