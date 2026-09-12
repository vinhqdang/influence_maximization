import numpy as np

from im_lab import graphs
from im_lab.baselines.fair_greedy import expected_reach_by_group, fair_welfare_greedy
from im_lab.simulator import true_params_from_graph


def test_fair_welfare_greedy_selects_k_seeds_and_reaches_multiple_groups():
    sizes = [15, 15]
    G = graphs.stochastic_block_model_graph(sizes, p_in=0.3, p_out=0.02, seed=13)
    graphs.assign_true_parameters(G, p_plus_range=(0.15, 0.35), seed=13)
    p_plus, _q = true_params_from_graph(G)
    group_of = graphs.group_of_map(G)

    rng = np.random.default_rng(13)
    k = 4
    seed_set, w, reach = fair_welfare_greedy(
        G, p_plus, k, group_of, num_sims=100, rng=rng
    )

    assert len(seed_set) == k
    assert w >= 0.0
    assert set(reach.keys()) == {0, 1}


def test_fair_welfare_greedy_balances_across_a_weak_bridge_sbm():
    """With two nearly-disconnected blocks, a plain spread-greedy would tend to seed
    only the denser/larger block; the welfare-weighted greedy's concave utility
    should push it to reach into both groups given enough seed budget."""
    sizes = [10, 30]
    G = graphs.stochastic_block_model_graph(sizes, p_in=0.5, p_out=0.0, seed=17)
    graphs.assign_true_parameters(G, p_plus_range=(0.3, 0.5), seed=17)
    p_plus, _q = true_params_from_graph(G)
    group_of = graphs.group_of_map(G)

    rng = np.random.default_rng(17)
    seed_set, _w, reach = fair_welfare_greedy(
        G, p_plus, k=4, group_of=group_of, num_sims=80, rng=rng
    )

    # Every group should get at least one seed since p_out=0 fully isolates them
    # and each group's own marginal welfare gain is strictly positive from log(1+x).
    seeded_groups = {group_of[v] for v in seed_set}
    assert seeded_groups == {0, 1}
    assert reach[0] > 0.0
    assert reach[1] > 0.0
