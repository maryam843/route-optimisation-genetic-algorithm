"""Genetic algorithm for a single closed delivery tour.

Partly based on course-provided code: the order crossover and swap mutation
follow ``order_crossover`` and ``mutate`` provided by DSAI3202 course staff
(UDST, Spring 2025). Modified by me: yes (rewritten on NumPy arrays with an
explicit random generator; mutation now returns a copy instead of changing
its input). Tournament selection is my 2025 function, fixed and rewritten in
2026. Everything else here (elitism, generational loop, stagnation restart,
migration hooks) is 2026 code.

The GA minimises the tour cost (a positive number). Smaller is better
everywhere: selection takes the minimum, elites are the lowest costs, and
the worst (highest cost) individuals are the ones replaced.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from route_ga.route import population_costs

Evaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 200
    generations: int = 500
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    elite_size: int = 2
    stagnation_limit: int = 150  # 0 disables restarts

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be in [0, population_size)")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be in [1, population_size]")
        for name in ("crossover_rate", "mutation_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def random_population(pop_size: int, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """``pop_size`` random permutations of the customer nodes 1..n-1."""
    base = np.arange(1, n_nodes, dtype=np.int64)
    return np.stack([rng.permutation(base) for _ in range(pop_size)])


def tournament_select(
    costs: np.ndarray, tournament_size: int, n_select: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of the winners of ``n_select`` tournaments (lowest cost wins).

    Each tournament draws ``tournament_size`` distinct individuals. With
    ``tournament_size == len(costs)`` every tournament returns the best one.
    """
    pop_size = len(costs)
    winners = np.empty(n_select, dtype=np.int64)
    for t in range(n_select):
        contestants = rng.choice(pop_size, size=tournament_size, replace=False)
        winners[t] = contestants[int(np.argmin(costs[contestants]))]
    return winners


def order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order crossover (OX), same variant as the course code.

    Copy ``p1[start:end+1]`` into the child, then fill the other positions
    left to right with the remaining nodes in the order they appear in ``p2``.
    """
    size = len(p1)
    if size < 2:
        return np.array(p1, copy=True)
    start, end = sorted(rng.choice(size, size=2, replace=False))
    segment = p1[start : end + 1]
    fill = p2[~np.isin(p2, segment)]
    child = np.empty(size, dtype=np.int64)
    child[start : end + 1] = segment
    child[:start] = fill[:start]
    child[end + 1 :] = fill[start:]
    return child


def swap_mutation(perm: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """With probability ``rate`` swap two random positions. Returns a copy."""
    out = np.array(perm, copy=True)
    if len(out) >= 2 and rng.random() < rate:
        i, j = rng.choice(len(out), size=2, replace=False)
        out[i], out[j] = out[j], out[i]
    return out


class VectorEvaluator:
    """Default evaluator: NumPy-vectorised cost of the whole population."""

    def __init__(self, cost: np.ndarray) -> None:
        self.cost = cost

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return population_costs(population, self.cost)


@dataclass
class GAResult:
    best_perm: np.ndarray
    best_cost: float
    history: list[float] = field(default_factory=list)
    generations_run: int = 0
    last_improvement: int = 0
    restarts: int = 0
    elapsed_s: float = 0.0


class GeneticAlgorithm:
    """One population. ``step()`` runs one generation.

    Kept as a class so the island model can run it generation by generation
    and exchange migrants in between.
    """

    def __init__(
        self,
        cost: np.ndarray,
        config: GAConfig,
        rng: np.random.Generator,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.cost = cost
        self.config = config
        self.rng = rng
        self.evaluate = evaluator if evaluator is not None else VectorEvaluator(cost)
        self.n_nodes = cost.shape[0]
        self.population = random_population(config.population_size, self.n_nodes, rng)
        self.costs = np.asarray(self.evaluate(self.population), dtype=float)
        self.generation = 0
        self.stagnation = 0
        self.restarts = 0
        self.last_improvement = 0
        i = int(np.argmin(self.costs))
        self.best_perm = self.population[i].copy()
        self.best_cost = float(self.costs[i])
        # history[g] = lowest cost in the population after generation g
        self.history: list[float] = [self.best_cost]

    def _breed(self, n_children: int) -> np.ndarray:
        cfg = self.config
        parents = tournament_select(self.costs, cfg.tournament_size, 2 * n_children, self.rng)
        children = np.empty((n_children, self.n_nodes - 1), dtype=np.int64)
        for c in range(n_children):
            p1 = self.population[parents[2 * c]]
            p2 = self.population[parents[2 * c + 1]]
            if self.rng.random() < cfg.crossover_rate:
                child = order_crossover(p1, p2, self.rng)
            else:
                child = p1.copy()
            children[c] = swap_mutation(child, cfg.mutation_rate, self.rng)
        return children

    def _update_best(self) -> None:
        i = int(np.argmin(self.costs))
        if self.costs[i] < self.best_cost - 1e-9:
            self.best_cost = float(self.costs[i])
            self.best_perm = self.population[i].copy()
            self.stagnation = 0
            self.last_improvement = self.generation
        else:
            self.stagnation += 1

    def _restart(self) -> None:
        """Keep the elites, replace everyone else with random tours."""
        cfg = self.config
        keep = max(cfg.elite_size, 1)
        order = np.argsort(self.costs, kind="stable")[:keep]
        fresh = random_population(cfg.population_size - keep, self.n_nodes, self.rng)
        self.population = np.vstack([self.population[order], fresh])
        self.costs = np.asarray(self.evaluate(self.population), dtype=float)
        self.restarts += 1
        self.stagnation = 0

    def step(self) -> None:
        cfg = self.config
        self.generation += 1
        order = np.argsort(self.costs, kind="stable")
        elites = self.population[order[: cfg.elite_size]]
        children = self._breed(cfg.population_size - cfg.elite_size)
        self.population = np.vstack([elites, children])
        self.costs = np.asarray(self.evaluate(self.population), dtype=float)
        self._update_best()
        if cfg.stagnation_limit and self.stagnation >= cfg.stagnation_limit:
            self._restart()
        self.history.append(float(self.costs.min()))

    def best(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.costs, kind="stable")[:k]
        return self.population[order].copy(), self.costs[order].copy()

    def receive(self, migrants: np.ndarray, migrant_costs: np.ndarray) -> None:
        """Replace the worst individuals with incoming migrants."""
        k = len(migrants)
        if k == 0:
            return
        worst = np.argsort(self.costs, kind="stable")[::-1][:k]
        self.population[worst] = migrants
        self.costs[worst] = migrant_costs
        self._update_best_after_migration()

    def _update_best_after_migration(self) -> None:
        i = int(np.argmin(self.costs))
        if self.costs[i] < self.best_cost - 1e-9:
            self.best_cost = float(self.costs[i])
            self.best_perm = self.population[i].copy()
            self.last_improvement = self.generation
            self.stagnation = 0
        self.history[-1] = float(self.costs.min())

    def result(self, elapsed_s: float = 0.0) -> GAResult:
        return GAResult(
            best_perm=self.best_perm.copy(),
            best_cost=self.best_cost,
            history=list(self.history),
            generations_run=self.generation,
            last_improvement=self.last_improvement,
            restarts=self.restarts,
            elapsed_s=elapsed_s,
        )


def run_ga(
    cost: np.ndarray,
    config: GAConfig,
    seed: int,
    evaluator: Evaluator | None = None,
) -> GAResult:
    """Run a single-population GA for ``config.generations`` generations."""
    t0 = time.perf_counter()
    ga = GeneticAlgorithm(cost, config, np.random.default_rng(seed), evaluator)
    for _ in range(config.generations):
        ga.step()
    return ga.result(time.perf_counter() - t0)
