"""Fitness evaluation in a process pool.

The 2025 script sent lambdas to a ``ProcessPoolExecutor`` and crashed with a
PicklingError. Everything sent to a worker here is a module-level function
or plain data, so it pickles, and the pool is created with the ``spawn``
start method so it behaves the same on Linux, macOS and Windows.

Two kernels are available so the benchmark can compare like with like:
``"python"`` loops over the edges of each route (the 2025 style) and
``"numpy"`` uses the vectorised :func:`route_ga.route.population_costs`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from route_ga.data import DEPOT
from route_ga.route import population_costs

KERNELS = ("python", "numpy")

_WORKER_COST: np.ndarray | None = None


def route_cost_python(perm, cost) -> float:
    """Plain-Python cost of one closed tour (depot -> perm -> depot)."""
    total = cost[DEPOT][perm[0]]
    for a, b in zip(perm[:-1], perm[1:], strict=True):
        total += cost[a][b]
    total += cost[perm[-1]][DEPOT]
    return float(total)


def costs_python(population: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Sequential plain-Python evaluation of a whole population."""
    rows = population.tolist()
    table = cost.tolist()
    return np.array([route_cost_python(r, table) for r in rows], dtype=float)


def _init_worker(cost: np.ndarray) -> None:
    global _WORKER_COST
    _WORKER_COST = cost


def _chunk_costs(args: tuple[np.ndarray, str]) -> np.ndarray:
    chunk, kernel = args
    assert _WORKER_COST is not None, "worker not initialised"
    if kernel == "numpy":
        return population_costs(chunk, _WORKER_COST)
    return costs_python(chunk, _WORKER_COST)


def _warm(delay: float) -> int:
    time.sleep(delay)
    return os.getpid()


def default_workers() -> int:
    return max(1, os.cpu_count() or 1)


class PoolEvaluator:
    """Callable that evaluates a population in a spawn-based process pool.

    The distance matrix is sent once per worker (pool initializer), not with
    every task. Use it as a context manager so the workers are shut down.
    """

    def __init__(
        self,
        cost: np.ndarray,
        workers: int | None = None,
        kernel: str = "python",
        chunks_per_worker: int = 1,
    ) -> None:
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        self.cost = np.asarray(cost, dtype=float)
        self.workers = workers or default_workers()
        self.kernel = kernel
        self.chunks_per_worker = max(1, chunks_per_worker)
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self.cost,),
        )

    def warm_up(self) -> None:
        """Start every worker so start-up time is not charged to the first call.

        Workers are started on demand, so each gets a short sleeping task.
        """
        list(self._pool.map(_warm, [0.2] * self.workers))

    def __call__(self, population: np.ndarray) -> np.ndarray:
        pop = np.asarray(population, dtype=np.int64)
        n_chunks = min(len(pop), self.workers * self.chunks_per_worker)
        chunks = np.array_split(pop, n_chunks)
        parts = self._pool.map(_chunk_costs, [(c, self.kernel) for c in chunks])
        return np.concatenate(list(parts))

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> PoolEvaluator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
