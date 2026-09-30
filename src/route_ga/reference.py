"""Reference solver: Google OR-Tools routing with guided local search.

Used only to judge the other methods. Its tour is labelled "reference
solver, not proven optimal": guided local search stops at a time limit and
gives no optimality proof. Missing edges get the same 100,000 penalty as
everywhere else in this package.
"""

from __future__ import annotations

import time

import numpy as np

from route_ga.data import DEPOT
from route_ga.route import cost_matrix


def solve_ortools(distances: np.ndarray, time_limit_s: int = 30) -> tuple[np.ndarray, dict]:
    """Return (perm, info) for a single vehicle starting and ending at the depot."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    cost = np.rint(cost_matrix(distances)).astype(np.int64)
    n = cost.shape[0]
    manager = pywrapcp.RoutingIndexManager(n, 1, DEPOT)
    routing = pywrapcp.RoutingModel(manager)
    table = cost.tolist()

    def arc_cost(i: int, j: int) -> int:
        return table[manager.IndexToNode(i)][manager.IndexToNode(j)]

    callback = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(callback)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = int(time_limit_s)
    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - t0
    if solution is None:
        raise RuntimeError("OR-Tools found no solution")
    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != DEPOT:
            order.append(node)
        index = solution.Value(routing.NextVar(index))
    info = {"objective": int(solution.ObjectiveValue()), "time_limit_s": int(time_limit_s),
            "elapsed_s": elapsed, "strategy": "PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH"}
    return np.asarray(order, dtype=np.int64), info
