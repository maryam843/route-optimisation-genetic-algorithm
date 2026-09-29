import itertools

import numpy as np
import pytest

from route_ga.baselines import brute_force, nearest_neighbour, two_opt
from route_ga.route import cost_matrix, route_cost


def test_nearest_neighbour_hand_made(square4):
    # From 0 the nearest is 1 (10); from 1: 2 is 35, 3 is 25 -> 3; then 2.
    assert nearest_neighbour(cost_matrix(square4)).tolist() == [1, 3, 2]


def test_nearest_neighbour_avoids_missing_edge():
    d = np.array([[0, 1, 5, 9], [1, 0, 100000, 2], [5, 100000, 0, 3], [9, 2, 3, 0]], float)
    assert nearest_neighbour(cost_matrix(d)).tolist() == [1, 3, 2]


def test_brute_force_five_nodes():
    rng = np.random.default_rng(3)
    d = rng.integers(1, 50, size=(5, 5)).astype(float)
    d = np.triu(d, 1) + np.triu(d, 1).T
    cost = cost_matrix(d)
    perm, best = brute_force(cost)
    manual = min(route_cost(p, cost) for p in itertools.permutations(range(1, 5)))
    assert best == manual
    assert route_cost(perm, cost) == best


def test_brute_force_limits():
    with pytest.raises(ValueError):
        brute_force(np.zeros((10, 10)))
    perm, c = brute_force(np.zeros((1, 1)))
    assert perm.size == 0 and c == 0.0


def test_two_opt_improves_crossing_tour():
    # Four corners of a square; the crossing tour 0-2-1-3-0 is longer.
    pts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], float)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    out = two_opt(np.array([2, 1, 3]), d)
    assert route_cost(out, d) == pytest.approx(4.0)


def test_two_opt_needs_symmetric():
    d = np.array([[0, 1, 2], [3, 0, 1], [2, 1, 0]], float)
    with pytest.raises(ValueError):
        two_opt(np.array([1, 2]), d)
