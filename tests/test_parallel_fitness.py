"""Process-pool fitness under the spawn start method (PicklingError fix)."""

import multiprocessing as mp
import pickle

import numpy as np
import pytest

from route_ga.data import load_map
from route_ga.ga import GAConfig, random_population, run_ga
from route_ga.parallel_fitness import (
    PoolEvaluator,
    _chunk_costs,
    costs_python,
    default_workers,
    route_cost_python,
)
from route_ga.route import cost_matrix, population_costs


def test_start_method_is_spawn():
    assert mp.get_start_method() == "spawn"


def test_python_kernel_matches_numpy():
    cost = cost_matrix(load_map(32))
    pop = random_population(50, 32, np.random.default_rng(0))
    assert np.allclose(costs_python(pop, cost), population_costs(pop, cost))
    assert route_cost_python(pop[0].tolist(), cost.tolist()) == population_costs(pop[:1], cost)[0]


def test_worker_function_is_picklable():
    # The 2025 code sent lambdas; module-level functions pickle fine.
    assert pickle.loads(pickle.dumps(_chunk_costs)) is _chunk_costs


def test_uninitialised_worker_raises(monkeypatch):
    import route_ga.parallel_fitness as pf

    monkeypatch.setattr(pf, "_WORKER_COST", None)
    with pytest.raises(RuntimeError, match="not initialised"):
        _chunk_costs((np.zeros((1, 3), dtype=np.int64), "numpy"))


@pytest.mark.parametrize("kernel", ["python", "numpy"])
def test_pool_matches_sequential(kernel):
    cost = cost_matrix(load_map(32))
    pop = random_population(101, 32, np.random.default_rng(1))
    with PoolEvaluator(cost, workers=2, kernel=kernel, chunks_per_worker=3) as pool:
        pool.warm_up()
        out = pool(pop)
    assert np.allclose(out, population_costs(pop, cost))


def test_ga_with_pool_equals_ga_without_pool():
    cost = cost_matrix(load_map(32))
    cfg = GAConfig(population_size=30, generations=15)
    with PoolEvaluator(cost, workers=2) as pool:
        with_pool = run_ga(cost, cfg, seed=4, evaluator=pool)
    without = run_ga(cost, cfg, seed=4)
    assert with_pool.best_perm.tolist() == without.best_perm.tolist()
    assert with_pool.history == without.history


def test_bad_kernel_and_default_workers():
    with pytest.raises(ValueError):
        PoolEvaluator(np.zeros((3, 3)), kernel="gpu")
    assert default_workers() >= 1
