"""Non-GA baselines: nearest neighbour, 2-opt and exact brute force.

All functions work on a cost matrix (see :func:`route_ga.route.cost_matrix`)
and return an individual: a permutation of the nodes 1..n-1 with the depot
left implicit. 2-opt assumes a symmetric matrix, which both course maps are.
"""

from __future__ import annotations

import itertools

import numpy as np

from route_ga.data import DEPOT, is_symmetric
from route_ga.route import population_costs

BRUTE_FORCE_MAX_NODES = 9


def nearest_neighbour(cost: np.ndarray, start: int = DEPOT) -> np.ndarray:
    """Greedy tour: from the depot, always go to the cheapest unvisited node.

    Ties are broken by the lowest node number, so the result is deterministic.
    """
    n = cost.shape[0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False
    current = start
    order = []
    for _ in range(n - 1):
        candidates = np.where(unvisited, cost[current], np.inf)
        nxt = int(np.argmin(candidates))
        order.append(nxt)
        unvisited[nxt] = False
        current = nxt
    return np.asarray(order, dtype=np.int64)


def two_opt(perm: np.ndarray, cost: np.ndarray, max_passes: int = 1000) -> np.ndarray:
    """Improve a tour with 2-opt moves until no move lowers the cost.

    For each position ``i`` the best reversal ``tour[i..j]`` is applied if it
    helps. The depot stays at both ends of the tour, which still allows every
    2-opt move of the closed tour. The cost never increases.
    """
    if not is_symmetric(cost):
        raise ValueError("2-opt here assumes a symmetric cost matrix")
    tour = np.concatenate(([DEPOT], np.asarray(perm, dtype=np.int64), [DEPOT]))
    m = tour.size  # n + 1
    for _ in range(max_passes):
        improved = False
        for i in range(1, m - 2):
            a, b = tour[i - 1], tour[i]
            js = np.arange(i + 1, m - 1)
            c, d = tour[js], tour[js + 1]
            delta = cost[a, c] + cost[b, d] - cost[a, b] - cost[c, d]
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = int(js[k])
                tour[i : j + 1] = tour[i : j + 1][::-1]
                improved = True
        if not improved:
            break
    return tour[1:-1].copy()


def brute_force(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact optimum by checking every tour. Only for n <= 9 nodes."""
    n = cost.shape[0]
    if n > BRUTE_FORCE_MAX_NODES:
        raise ValueError(f"brute force is limited to {BRUTE_FORCE_MAX_NODES} nodes, got {n}")
    if n < 2:
        return np.zeros(0, dtype=np.int64), 0.0
    perms = np.array(list(itertools.permutations(range(1, n))), dtype=np.int64)
    costs = population_costs(perms, cost)
    best = int(np.argmin(costs))
    return perms[best].copy(), float(costs[best])
