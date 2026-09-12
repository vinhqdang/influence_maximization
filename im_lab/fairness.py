"""Fairness-constrained budget allocation shared by MF-BWI-Fair.

Per-round group floors: B_g >= alpha * (|G_g| / n) * B. Since the floors are
proportional to group size and sum to exactly alpha * B (the |G_g|/n fractions sum
to 1), sum_g B_g = alpha * B <= B for any alpha in [0,1] -- so the floors are always
"jointly feasible" from a pure budget-arithmetic standpoint. The only way a floor can
fail to be met in practice is if a group runs out of nodes with a meaningful action
available (every node always has one: CONVERT if inactive, MAINTAIN if active, so in
practice this cannot happen for any non-empty group either) or if per-round leftover
budget is expressed in indivisible cost units (1 or 5) and forces a very slight
overshoot -- which is harmless since a floor is a "spend at least this much" bound.
"""

from __future__ import annotations


def group_floors(group_sizes: dict, n: int, budget: float, alpha: float) -> dict:
    """B_g = alpha * (|G_g| / n) * budget for every group g."""
    return {g: alpha * (size / n) * budget for g, size in group_sizes.items()}


def allocate_with_fairness(
    candidates: list,
    group_of: dict,
    group_sizes: dict,
    n: int,
    budget: float,
    alpha: float,
) -> tuple[dict, dict]:
    """Select actions under a per-round budget with per-group fairness floors.

    candidates: list of (node, action, cost, index) tuples, at most one candidate
        action per node (the caller picks the single best action -- CONVERT or
        MAINTAIN -- for each node before calling this).
    group_of: node -> group id.
    group_sizes: group id -> |G_g|.
    n: total number of nodes.
    budget: per-round total budget B.
    alpha: fairness parameter in [0,1].

    Returns (selected: dict node->(action,cost), spend_by_group: dict group->float).

    Two-phase greedy: (1) within each group, fill candidates by descending index
    until that group's floor B_g is met (or the group/budget runs out); (2) fill the
    residual budget across ALL remaining candidates, still by descending index,
    regardless of group. This is the natural greedy relaxation of a knapsack with
    per-group minimums: optimal knapsack-with-floors is itself NP-hard in general,
    so we use this straightforward greedy-by-index heuristic (consistent with the
    rest of MF-BWI-Fair being an index/myopic policy, not an exact optimizer).
    """
    floors = group_floors(group_sizes, n, budget, alpha)

    by_group: dict = {g: [] for g in group_sizes}
    for node, action, cost, index in candidates:
        by_group[group_of[node]].append((node, action, cost, index))
    for g in by_group:
        by_group[g].sort(key=lambda c: c[3], reverse=True)

    selected: dict = {}
    spend_by_group: dict = {g: 0.0 for g in group_sizes}
    remaining_budget = float(budget)
    chosen_nodes: set = set()

    # Phase 1: meet each group's floor, in isolation, by index-descending order.
    for g, group_candidates in by_group.items():
        for node, action, cost, index in group_candidates:
            if spend_by_group[g] >= floors[g]:
                break
            if cost > remaining_budget:
                continue
            selected[node] = (action, cost)
            chosen_nodes.add(node)
            spend_by_group[g] += cost
            remaining_budget -= cost

    # Phase 2: fill the residual budget across all remaining candidates by index.
    residual_pool = [c for c in candidates if c[0] not in chosen_nodes]
    residual_pool.sort(key=lambda c: c[3], reverse=True)
    for node, action, cost, index in residual_pool:
        if cost <= remaining_budget:
            selected[node] = (action, cost)
            chosen_nodes.add(node)
            spend_by_group[group_of[node]] += cost
            remaining_budget -= cost

    total_spend = sum(spend_by_group.values())
    assert total_spend <= budget + 1e-9, (
        f"budget invariant violated: spent {total_spend} > budget {budget}"
    )

    return selected, spend_by_group
