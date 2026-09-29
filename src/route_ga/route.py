"""Routes, tours and their cost.

Representation: an individual is a permutation of the customer nodes
1..n-1 (a 1-D integer array). The depot 0 is fixed and not stored. The
closed tour is ``[0, *perm, 0]``, so every length includes the leg from the
depot to the first stop and the return leg from the last stop to the depot.

Cost: the sum of edge weights where every missing edge (>= 100000) costs
``penalty``. The default penalty equals 100000, which is larger than any
feasible tour on these maps (at most 99 per edge), so a tour with fewer
missing edges always beats one with more.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from route_ga.data import DEPOT, MISSING_EDGE, missing_mask

PENALTY = MISSING_EDGE


def cost_matrix(distances: np.ndarray, penalty: float = PENALTY) -> np.ndarray:
    """Copy of ``distances`` where every missing edge costs ``penalty``."""
    cost = np.array(distances, dtype=float, copy=True)
    cost[missing_mask(cost)] = penalty
    return cost


def to_tour(perm: Sequence[int]) -> list[int]:
    """Closed tour that starts and ends at the depot."""
    return [DEPOT, *(int(x) for x in perm), DEPOT]


def from_tour(tour: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`to_tour`."""
    if len(tour) < 2 or tour[0] != DEPOT or tour[-1] != DEPOT:
        raise ValueError("a tour must start and end at the depot")
    return np.asarray(tour[1:-1], dtype=np.int64)


def is_valid_tour(tour: Sequence[int], n_nodes: int) -> bool:
    """True if the tour starts and ends at 0 and visits every other node once."""
    if len(tour) != n_nodes + 1 or tour[0] != DEPOT or tour[-1] != DEPOT:
        return False
    return sorted(int(x) for x in tour[1:-1]) == list(range(1, n_nodes))


def is_valid_perm(perm: Sequence[int], n_nodes: int) -> bool:
    return is_valid_tour(to_tour(perm), n_nodes)


def route_cost(perm: Sequence[int], cost: np.ndarray) -> float:
    """Cost of the closed tour for one individual (return edge included)."""
    p = np.asarray(perm, dtype=np.int64)
    if p.size == 0:
        return 0.0
    inner = cost[p[:-1], p[1:]].sum() if p.size > 1 else 0.0
    return float(cost[DEPOT, p[0]] + inner + cost[p[-1], DEPOT])


def population_costs(population: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Vectorised cost of every row of a (pop_size, n-1) population array."""
    pop = np.asarray(population, dtype=np.int64)
    first = cost[DEPOT, pop[:, 0]]
    last = cost[pop[:, -1], DEPOT]
    inner = cost[pop[:, :-1], pop[:, 1:]].sum(axis=1)
    return first + inner + last


def tour_edges(tour: Sequence[int]) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in zip(tour[:-1], tour[1:], strict=True)]


def summarise(perm: Sequence[int], distances: np.ndarray) -> dict:
    """Length, missing-edge count and feasibility of one individual."""
    tour = to_tour(perm)
    missing = missing_mask(distances)
    edges = tour_edges(tour)
    n_missing = sum(bool(missing[a, b]) for a, b in edges)
    length = float(sum(distances[a, b] for a, b in edges if not missing[a, b]))
    return {
        "tour": tour,
        "length": length,
        "missing_edges": n_missing,
        "feasible": n_missing == 0,
        "cost": route_cost(perm, cost_matrix(distances)),
    }
