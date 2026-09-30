import itertools

import numpy as np

from route_ga.baselines import brute_force
from route_ga.reference import solve_ortools
from route_ga.route import cost_matrix, is_valid_perm, route_cost


def test_ortools_matches_brute_force_on_small_map():
    rng = np.random.default_rng(4)
    d = rng.integers(1, 100, size=(7, 7)).astype(float)
    d = np.triu(d, 1) + np.triu(d, 1).T
    perm, info = solve_ortools(d, time_limit_s=1)
    cost = cost_matrix(d)
    assert is_valid_perm(perm, 7)
    assert info["objective"] == route_cost(perm, cost)
    assert route_cost(perm, cost) == brute_force(cost)[1]
    assert min(route_cost(p, cost) for p in itertools.permutations(range(1, 7))) == \
        route_cost(perm, cost)


def test_ortools_avoids_missing_edge(square4):
    d = square4.copy()
    d[0, 1] = d[1, 0] = 100000
    perm, _ = solve_ortools(d, time_limit_s=1)
    tour = [0, *perm.tolist(), 0]
    assert all({a, b} != {0, 1} for a, b in zip(tour, tour[1:], strict=False))
