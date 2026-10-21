"""Command line entry point: ``python -m route_ga --help``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from route_ga import __version__
from route_ga.baselines import nearest_neighbour, two_opt
from route_ga.data import MAP_FILES, load_map, load_matrix
from route_ga.ga import GAConfig, run_ga
from route_ga.islands_mpi import (
    IslandConfig,
    mpi_world_size_from_env,
    run_islands_mpi,
    run_islands_serial,
)
from route_ga.parallel_fitness import PoolEvaluator
from route_ga.provenance import environment_info
from route_ga.route import cost_matrix, summarise

METHODS = ("nn", "2opt", "ga", "ga-islands")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="route_ga",
        description="Single-vehicle delivery tour (depot 0 -> every node -> depot 0) "
        "with a genetic algorithm and classic baselines.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--map", type=int, choices=sorted(MAP_FILES), default=32,
                     help="course map to use (default: 32)")
    src.add_argument("--matrix", type=Path, help="path to another distance-matrix CSV")
    p.add_argument("--method", choices=METHODS, default="ga", help="default: ga")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--generations", type=int, default=GAConfig.generations)
    p.add_argument("--population", type=int, default=GAConfig.population_size,
                   help="population size (per island for ga-islands)")
    p.add_argument("--polish", action="store_true", help="apply 2-opt to the GA result")
    p.add_argument("--workers", type=int, default=0,
                   help="ga only: evaluate fitness in a process pool with this many workers")
    p.add_argument("--islands", type=int, default=IslandConfig.n_islands,
                   help="ga-islands serial backend: number of islands")
    p.add_argument("--migration-interval", type=int, default=IslandConfig.migration_interval)
    p.add_argument("--migrants", type=int, default=IslandConfig.migrants)
    p.add_argument("--backend", choices=("auto", "serial", "mpi"), default="auto",
                   help="ga-islands: auto uses MPI when started under mpiexec with >1 process")
    p.add_argument("--out", type=Path, help="write the result as JSON to this file")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def solve(args: argparse.Namespace, distances: np.ndarray) -> dict:
    cost = cost_matrix(distances)
    ga_cfg = GAConfig(population_size=args.population, generations=args.generations)
    extra: dict = {}
    t0 = time.perf_counter()
    if args.method == "nn":
        perm = nearest_neighbour(cost)
    elif args.method == "2opt":
        perm = two_opt(nearest_neighbour(cost), cost)
    elif args.method == "ga":
        if args.workers > 0:
            with PoolEvaluator(cost, workers=args.workers) as pool:
                res = run_ga(cost, ga_cfg, args.seed, evaluator=pool)
        else:
            res = run_ga(cost, ga_cfg, args.seed)
        perm = res.best_perm
        extra = {"ga_config": ga_cfg.to_dict(), "history": res.history,
                 "restarts": res.restarts, "workers": args.workers}
    else:
        backend = args.backend
        if backend == "auto":
            backend = "mpi" if mpi_world_size_from_env() > 1 else "serial"
        isl_cfg = IslandConfig(args.islands, args.migration_interval, args.migrants)
        if backend == "mpi":
            from mpi4py import MPI

            res = run_islands_mpi(cost, ga_cfg, isl_cfg, args.seed)
            rank = MPI.COMM_WORLD.Get_rank()
            if rank != 0:
                # Every rank gets the combined result; only rank 0 polishes and reports it.
                return {"mpi_rank": rank}
        else:
            res = run_islands_serial(cost, ga_cfg, isl_cfg, args.seed)
        perm = res.best_perm
        extra = {"ga_config": ga_cfg.to_dict(), "island_config": isl_cfg.to_dict(),
                 "backend": res.backend, "island_best_costs": res.island_best_costs,
                 "islands_elapsed_s": res.elapsed_s,
                 "history": res.history, "restarts": res.restarts}
    if args.polish and args.method in ("ga", "ga-islands"):
        perm = two_opt(perm, cost)
    elapsed = time.perf_counter() - t0
    summary = summarise(perm, distances)
    return {
        "method": args.method + ("+2opt" if args.polish and args.method.startswith("ga") else ""),
        "map": None if args.matrix else args.map,
        "matrix": str(args.matrix) if args.matrix else None,
        "n_nodes": int(distances.shape[0]),
        "seed": args.seed,
        "elapsed_s": elapsed,
        **summary,
        **extra,
        "environment": environment_info(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    distances = load_matrix(args.matrix) if args.matrix else load_map(args.map)
    result = solve(args, distances)
    if result.get("mpi_rank", 0) != 0:
        return 0
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        status = "feasible" if result["feasible"] else f"{result['missing_edges']} missing edges"
        print(f"method={result['method']} nodes={result['n_nodes']} seed={args.seed}")
        print(f"tour length={result['length']:.0f} ({status}), time={result['elapsed_s']:.2f}s")
        print("tour:", " ".join(str(x) for x in result["tour"]))
        if args.out:
            print(f"saved {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
