import numpy as np
import pytest

from route_ga.data import load_map
from route_ga.ga import (
    GAConfig,
    GeneticAlgorithm,
    order_crossover,
    random_population,
    run_ga,
    swap_mutation,
    tournament_select,
)
from route_ga.route import cost_matrix, is_valid_perm, population_costs


def test_tournament_picks_fittest_when_k_is_population():
    costs = np.array([50.0, 40.0, 30.0, 45.0])
    winners = tournament_select(costs, 4, 10, np.random.default_rng(0))
    assert winners.tolist() == [2] * 10


def test_tournament_winner_is_best_of_contestants():
    rng = np.random.default_rng(1)
    costs = rng.random(30)
    winners = tournament_select(costs, 3, 200, np.random.default_rng(2))
    # A size-3 tournament winner can never be one of the two worst.
    worst_two = set(np.argsort(costs)[-2:].tolist())
    assert not worst_two & set(winners.tolist())


def test_order_crossover_keeps_parent1_segment():
    rng = np.random.default_rng(5)
    p1 = np.arange(1, 11)
    p2 = p1[::-1].copy()
    child = order_crossover(p1, p2, rng)
    assert sorted(child.tolist()) == list(range(1, 11))
    # Some slice of p1 is copied in place.
    assert any(child[i] == p1[i] for i in range(10))


def test_order_crossover_tiny():
    rng = np.random.default_rng(0)
    assert order_crossover(np.array([1]), np.array([1]), rng).tolist() == [1]


def test_swap_mutation_returns_copy():
    route = np.array([1, 2, 3, 4])
    out = swap_mutation(route, 1.0, np.random.default_rng(0))
    assert route.tolist() == [1, 2, 3, 4]
    assert out.tolist() != route.tolist()
    assert sorted(out.tolist()) == [1, 2, 3, 4]
    assert swap_mutation(route, 0.0, np.random.default_rng(0)).tolist() == [1, 2, 3, 4]


def test_random_population_shape_and_validity():
    pop = random_population(20, 9, np.random.default_rng(0))
    assert pop.shape == (20, 8)
    assert all(is_valid_perm(p, 9) for p in pop)


@pytest.mark.parametrize("bad", [
    {"population_size": 1},
    {"elite_size": 200},
    {"tournament_size": 0},
    {"mutation_rate": 1.5},
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        GAConfig(**bad)


def test_ga_improves_on_32_node_map():
    cost = cost_matrix(load_map(32))
    cfg = GAConfig(population_size=60, generations=150)
    res = run_ga(cost, cfg, seed=0)
    assert res.history[-1] < res.history[0]
    assert res.best_cost == pytest.approx(min(res.history))
    assert is_valid_perm(res.best_perm, 32)
    assert len(res.history) == cfg.generations + 1


def test_stagnation_restart_happens_and_keeps_best():
    cost = cost_matrix(load_map(32))
    cfg = GAConfig(population_size=20, generations=60, stagnation_limit=3,
                   crossover_rate=0.0, mutation_rate=0.0)
    res = run_ga(cost, cfg, seed=1)
    assert res.restarts > 0
    assert all(b <= a + 1e-9 for a, b in zip(res.history, res.history[1:], strict=False))


def test_receive_replaces_worst_and_updates_best():
    cost = cost_matrix(load_map(32))
    ga = GeneticAlgorithm(cost, GAConfig(population_size=10), np.random.default_rng(0))
    worst = int(np.argmax(ga.costs))
    migrant = ga.population[int(np.argmin(ga.costs))].copy()
    migrant_cost = population_costs(migrant[None, :], cost) - 1  # pretend it is better
    ga.receive(migrant[None, :], migrant_cost)
    assert ga.costs[worst] == migrant_cost[0]
    assert ga.best_cost == migrant_cost[0]
    ga.receive(np.empty((0, 31), dtype=np.int64), np.empty(0))
    ga.receive(ga.population[:1].copy(), np.array([1e12]))  # worse migrant: best unchanged
    assert ga.best_cost == migrant_cost[0]


def test_best_returns_sorted():
    cost = cost_matrix(load_map(32))
    ga = GeneticAlgorithm(cost, GAConfig(population_size=10), np.random.default_rng(0))
    _, c = ga.best(3)
    assert c.tolist() == sorted(ga.costs.tolist())[:3]
