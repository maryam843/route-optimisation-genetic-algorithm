import numpy as np
import pytest

from route_ga.route import (
    cost_matrix,
    from_tour,
    is_valid_perm,
    is_valid_tour,
    population_costs,
    route_cost,
    summarise,
    to_tour,
)


def test_closed_tour_includes_return_edge(square4):
    cost = cost_matrix(square4)
    assert route_cost([1, 3, 2], cost) == 80.0
    assert to_tour([1, 3, 2]) == [0, 1, 3, 2, 0]


def test_from_tour_roundtrip():
    assert from_tour([0, 2, 1, 0]).tolist() == [2, 1]
    with pytest.raises(ValueError):
        from_tour([1, 2, 0])


def test_single_and_empty_perm(square4):
    cost = cost_matrix(square4)
    assert route_cost([2], cost) == 30.0
    assert route_cost([], cost) == 0.0


def test_validity():
    assert is_valid_tour([0, 1, 2, 3, 0], 4)
    assert not is_valid_tour([0, 1, 2, 3], 4)  # not closed
    assert not is_valid_tour([1, 0, 2, 3, 1], 4)  # does not start at depot
    assert not is_valid_tour([0, 1, 1, 3, 0], 4)  # repeats a node
    assert is_valid_perm([3, 1, 2], 4)


def test_population_costs_matches_single(small_random):
    cost = cost_matrix(small_random)
    rng = np.random.default_rng(0)
    pop = np.stack([rng.permutation(np.arange(1, 12)) for _ in range(20)])
    assert np.allclose(population_costs(pop, cost), [route_cost(p, cost) for p in pop])


def test_summarise_reports_missing_edges(square4):
    d = square4.copy()
    d[1, 3] = d[3, 1] = 100000
    s = summarise([1, 3, 2], d)
    assert s["missing_edges"] == 1
    assert not s["feasible"]
    assert s["length"] == 10 + 30 + 15
    assert s["cost"] == 10 + 100000 + 30 + 15
