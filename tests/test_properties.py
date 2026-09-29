"""T3: property-based tests with hypothesis."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from route_ga.baselines import two_opt
from route_ga.ga import order_crossover, swap_mutation
from route_ga.route import cost_matrix, route_cost

seeds = st.integers(min_value=0, max_value=2**32 - 1)
sizes = st.integers(min_value=1, max_value=60)


def perm(size, seed):
    return np.random.default_rng(seed).permutation(np.arange(1, size + 1))


def sym_matrix(n, seed, missing=0.0):
    rng = np.random.default_rng(seed)
    d = rng.integers(1, 100, size=(n, n)).astype(float)
    d[rng.random((n, n)) < missing] = 100000
    d = np.triu(d, 1)
    d = d + d.T
    np.fill_diagonal(d, 0)
    return d


@given(sizes, seeds, seeds, seeds)
def test_crossover_returns_valid_permutation(size, s1, s2, s3):
    p1, p2 = perm(size, s1), perm(size, s2)
    child = order_crossover(p1, p2, np.random.default_rng(s3))
    assert sorted(child.tolist()) == list(range(1, size + 1))


@given(sizes, seeds, st.floats(min_value=0.0, max_value=1.0), seeds)
def test_mutation_returns_valid_permutation(size, s1, rate, s2):
    p = perm(size, s1)
    out = swap_mutation(p, rate, np.random.default_rng(s2))
    assert sorted(out.tolist()) == list(range(1, size + 1))


@given(st.integers(min_value=2, max_value=40), seeds, seeds)
def test_length_is_symmetric_in_direction(n, s1, s2):
    cost = cost_matrix(sym_matrix(n, s1, missing=0.2))
    p = perm(n - 1, s2)
    assert route_cost(p, cost) == route_cost(p[::-1], cost)


@settings(max_examples=60, deadline=None)
@given(st.integers(min_value=3, max_value=30), seeds, seeds)
def test_two_opt_never_increases_length(n, s1, s2):
    cost = cost_matrix(sym_matrix(n, s1, missing=0.2))
    p = perm(n - 1, s2)
    out = two_opt(p, cost)
    assert sorted(out.tolist()) == list(range(1, n))
    assert route_cost(out, cost) <= route_cost(p, cost) + 1e-9
