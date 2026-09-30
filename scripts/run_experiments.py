"""Run every experiment reported in the README and save raw results.

Usage:
    python scripts/run_experiments.py                # everything (about 40 min on 4 cores)
    python scripts/run_experiments.py --machine-note "shared cloud VM"
    python scripts/run_experiments.py --only anchor  # one experiment
    python scripts/run_experiments.py --quick --out-dir /tmp/r   # tiny smoke run

Each experiment writes one JSON file with its seeds, settings, every single
run, the machine/library details and a UTC timestamp. Nothing is summarised
by hand: results/tables.md is rendered from these files by
scripts/results_tables.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from route_ga.baselines import brute_force, nearest_neighbour, two_opt
from route_ga.data import MISSING_EDGE, load_map
from route_ga.ga import GAConfig, GeneticAlgorithm, random_population, run_ga
from route_ga.islands_mpi import (
    IslandConfig,
    clean_mpi_env,
    mpiexec_command,
    run_islands_serial,
)
from route_ga.parallel_fitness import PoolEvaluator, costs_python
from route_ga.provenance import environment_info
from route_ga.route import cost_matrix, population_costs, summarise

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from results_tables import write_tables  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TUNING_SEEDS = [100, 101, 102]
REPEATS = 5  # timing repeats per population and seed
REFERENCE_REPEATS = 5
REFERENCE_TIME_LIMIT_S = 30
MACHINE_NOTE: str | None = None
MISSING_SHARE = 0.19  # share of missing edges in both course maps (192/992, 1900/9900)


def generations_to_converge(history: list[float]) -> int:
    """First generation at which the run reached its final best cost."""
    final = min(history)
    return next(g for g, c in enumerate(history) if c <= final + 1e-9)


def mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def median_iqr(values: list[float]) -> dict:
    """Median and interquartile range (25th and 75th percentiles)."""
    q1, med, q3 = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "n": len(values)}


def save(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": name, "environment": environment_info(MACHINE_NOTE), **payload}
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"saved {path}", flush=True)
    return path


# ---------------------------------------------------------------- tuning
def exp_tuning(out_dir: Path, quick: bool) -> None:
    """Small grid on the 32-node map with seeds that are NOT used later."""
    cost = cost_matrix(load_map(32))
    gens = 50 if quick else 500
    grid = [(t, m) for t in (2, 3, 5) for m in (0.1, 0.3, 0.6, 0.9)]
    if quick:
        grid = grid[:2]
    rows = []
    for t, m in grid:
        cfg = replace(GAConfig(), tournament_size=t, mutation_rate=m, generations=gens)
        lengths = [run_ga(cost, cfg, s).best_cost for s in TUNING_SEEDS]
        mu, sd = mean_sd(lengths)
        rows.append({"tournament_size": t, "mutation_rate": m, "costs": lengths,
                     "mean": mu, "sd": sd})
        print(f"tuning t={t} m={m}: {mu:.1f} +/- {sd:.1f}", flush=True)
    best = min(rows, key=lambda r: r["mean"])
    save(out_dir, "tuning", {"map": 32, "seeds": TUNING_SEEDS, "generations": gens,
                             "base_config": GAConfig().to_dict(), "grid": rows, "best": best})


# ---------------------------------------------------------------- anchor
def random_instance(n: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric integer distances 1..99 with about 19 % missing edges."""
    d = rng.integers(1, 100, size=(n, n)).astype(float)
    missing = rng.random((n, n)) < MISSING_SHARE
    d[missing] = MISSING_EDGE
    d = np.triu(d, 1)
    d = d + d.T
    np.fill_diagonal(d, 0.0)
    return d


def exp_anchor(out_dir: Path, quick: bool) -> None:
    n_nodes, per_seed = 8, (3 if quick else 20)
    ga_cfg = GAConfig(population_size=50, generations=30 if quick else 100, elite_size=2)
    runs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for i in range(per_seed):
            d = random_instance(n_nodes, rng)
            cost = cost_matrix(d)
            _, opt = brute_force(cost)
            nn = nearest_neighbour(cost)
            res = {
                "nn": summarise(nn, d)["cost"],
                "nn+2opt": summarise(two_opt(nn, cost), d)["cost"],
                "ga": run_ga(cost, ga_cfg, seed * 1000 + i).best_cost,
            }
            runs.append({"seed": seed, "instance": i, "optimum": opt,
                         "optimum_feasible": opt < MISSING_EDGE, **res})
    methods = ("nn", "nn+2opt", "ga")
    summary = {}
    for m in methods:
        per_seed_rate = []
        for seed in SEEDS:
            rs = [r for r in runs if r["seed"] == seed]
            per_seed_rate.append(100.0 * sum(abs(r[m] - r["optimum"]) < 1e-9 for r in rs) / len(rs))
        hits = sum(abs(r[m] - r["optimum"]) < 1e-9 for r in runs)
        mu, sd = mean_sd(per_seed_rate)
        feas = [r for r in runs if r["optimum_feasible"]]
        gaps = [100.0 * (r[m] - r["optimum"]) / r["optimum"] for r in feas]
        summary[m] = {"hits": hits, "instances": len(runs), "hit_rate_mean": mu,
                      "hit_rate_sd": sd, "mean_gap_pct_feasible": float(np.mean(gaps))}
    save(out_dir, "anchor_8node", {
        "n_nodes": n_nodes, "seeds": SEEDS, "instances_per_seed": per_seed,
        "missing_share": MISSING_SHARE, "ga_config": ga_cfg.to_dict(),
        "infeasible_optima": sum(not r["optimum_feasible"] for r in runs),
        "summary": summary, "runs": runs})


# ---------------------------------------------------------------- maps
def run_mpi(n_map: int, n_procs: int, seed: int, cfg: GAConfig, isl: IslandConfig) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        cmd = mpiexec_command(
            n_procs, sys.executable, "-m", "route_ga", "--map", str(n_map),
            "--method", "ga-islands", "--backend", "mpi", "--seed", str(seed),
            "--generations", str(cfg.generations), "--population", str(cfg.population_size),
            "--migration-interval", str(isl.migration_interval), "--migrants", str(isl.migrants),
            "--quiet", "--out", str(out))
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, env=clean_mpi_env())
        launch = time.perf_counter() - t0
        data = json.loads(out.read_text())
    data["launch_wall_s"] = launch
    return data


def exp_map(out_dir: Path, n_map: int, quick: bool, name: str | None = None) -> None:
    d = load_map(n_map)
    cost = cost_matrix(d)
    gens = {32: 1000, 100: 2000}[n_map] if not quick else 40
    cfg = replace(GAConfig(), generations=gens)
    isl = IslandConfig(migration_interval=25 if not quick else 10, migrants=2)
    islands = (1, 2, 4)
    runs = []

    def add(method, seed, perm, seconds, history=None, **extra):
        s = summarise(perm, d)
        row = {"method": method, "seed": seed, "length": s["length"], "cost": s["cost"],
               "feasible": s["feasible"], "missing_edges": s["missing_edges"],
               "time_s": seconds, "tour": s["tour"], **extra}
        if history is not None:
            row["history"] = history
            row["generations_to_converge"] = generations_to_converge(history)
        runs.append(row)
        print(f"map{n_map} seed={seed} {method}: {s['length']:.0f} "
              f"feasible={s['feasible']} {seconds:.2f}s", flush=True)

    for seed in SEEDS:
        t0 = time.perf_counter()
        nn = nearest_neighbour(cost)
        add("nn", seed, nn, time.perf_counter() - t0)
        t0 = time.perf_counter()
        nn2 = two_opt(nearest_neighbour(cost), cost)
        add("nn+2opt", seed, nn2, time.perf_counter() - t0)

        res = run_ga(cost, cfg, seed)
        add("ga", seed, res.best_perm, res.elapsed_s, res.history, restarts=res.restarts)
        t0 = time.perf_counter()
        pol = two_opt(res.best_perm, cost)
        add("ga+2opt", seed, pol, res.elapsed_s + time.perf_counter() - t0)

        for k in islands:
            m = run_mpi(n_map, k, seed, cfg, isl)
            extra = {"launch_wall_s": m["launch_wall_s"], "backend": m["backend"],
                     "island_best_costs": m["island_best_costs"], "restarts": m["restarts"]}
            if k > 1:
                ser = run_islands_serial(cost, cfg, replace(isl, n_islands=k), seed)
                extra["serial_time_s"] = ser.elapsed_s
                extra["matches_serial"] = bool(
                    abs(ser.best_cost - m["cost"]) < 1e-9
                    and ser.best_perm.tolist() == m["tour"][1:-1])
            add(f"ga-islands{k}", seed, m["tour"][1:-1], m["islands_elapsed_s"],
                m["history"], **extra)
            if k == 4:
                t0 = time.perf_counter()
                pol = two_opt(np.asarray(m["tour"][1:-1]), cost)
                add("ga-islands4+2opt", seed, pol,
                    m["islands_elapsed_s"] + time.perf_counter() - t0)

    order = ["nn", "nn+2opt", "ga", "ga+2opt", "ga-islands1", "ga-islands2", "ga-islands4",
             "ga-islands4+2opt"]
    summary = {}
    for m in order:
        rs = [r for r in runs if r["method"] == m]
        lengths = [r["cost"] for r in rs]
        mu, sd = mean_sd(lengths)
        entry = {"mean": mu, "sd": sd, "min": min(lengths), "max": max(lengths),
                 "all_feasible": all(r["feasible"] for r in rs),
                 "time": median_iqr([r["time_s"] for r in rs])}
        if "generations_to_converge" in rs[0]:
            gmu, gsd = mean_sd([r["generations_to_converge"] for r in rs])
            entry.update(gens_to_converge_mean=gmu, gens_to_converge_sd=gsd)
        if "serial_time_s" in rs[0]:
            serial = median_iqr([r["serial_time_s"] for r in rs])
            entry.update(serial_time=serial,
                         all_match_serial=all(r["matches_serial"] for r in rs),
                         matches_serial_count=sum(r["matches_serial"] for r in rs),
                         speedup_median=serial["median"] / entry["time"]["median"])
        if m.startswith("ga-islands") and "launch_wall_s" in rs[0]:
            entry["launch_wall"] = median_iqr([r["launch_wall_s"] for r in rs])
        summary[m] = entry
    best = min(runs, key=lambda r: r["cost"])
    save(out_dir, name or f"map{n_map}", {
        "map": n_map, "seeds": SEEDS, "quick": quick, "ga_config": cfg.to_dict(),
        "island_config": isl.to_dict(), "islands": list(islands),
        "island_population": "each island has population_size individuals",
        "timing": "time per run; median and IQR over the 5 seeds (one run per seed)",
        "best_cost": best["cost"], "best_method": best["method"], "best_seed": best["seed"],
        "best_tour": best["tour"], "summary": summary, "runs": runs})


# ---------------------------------------------------------------- fitness
def exp_fitness(out_dir: Path, quick: bool) -> None:
    cost = cost_matrix(load_map(100))
    sizes = (500, 2000) if quick else (500, 2000, 10000)
    seeds = SEEDS[:2] if quick else SEEDS
    workers = 4
    t0 = time.perf_counter()
    pools = {k: PoolEvaluator(cost, workers=workers, kernel=k) for k in ("python", "numpy")}
    for p in pools.values():
        p.warm_up()
    startup = (time.perf_counter() - t0) / 2
    rows = []
    try:
        for size in sizes:
            for seed in seeds:
                pop = random_population(size, cost.shape[0], np.random.default_rng(seed))
                ref = population_costs(pop, cost)
                timings = {}
                for name, fn in (
                    ("python_seq", lambda: costs_python(pop, cost)),  # noqa: B023
                    ("python_pool", lambda: pools["python"](pop)),  # noqa: B023
                    ("numpy_seq", lambda: population_costs(pop, cost)),  # noqa: B023
                    ("numpy_pool", lambda: pools["numpy"](pop)),  # noqa: B023
                ):
                    samples = []
                    for _ in range(REPEATS):
                        t0 = time.perf_counter()
                        out = fn()
                        samples.append(time.perf_counter() - t0)
                        assert np.allclose(out, ref), name
                    timings[name] = samples
                step_samples = []
                for rep_i in range(REPEATS):
                    ga = GeneticAlgorithm(cost, GAConfig(population_size=size),
                                          np.random.default_rng([seed, rep_i]))
                    t0 = time.perf_counter()
                    ga.step()
                    step_samples.append(time.perf_counter() - t0)
                timings["ga_step_numpy"] = step_samples
                rows.append({"population": size, "seed": seed, **timings})
                print(f"fitness P={size} seed={seed}: "
                      + " ".join(f"{k}={statistics.median(v) * 1000:.1f}ms"
                                 for k, v in timings.items()), flush=True)
    finally:
        for p in pools.values():
            p.close()
    summary = {}
    for size in sizes:
        rs = [r for r in rows if r["population"] == size]
        summary[str(size)] = {k: median_iqr([x for r in rs for x in r[k]])
                              for k in rs[0] if k not in ("population", "seed")}
    save(out_dir, "parallel_fitness", {
        "map": 100, "workers": workers, "seeds": seeds, "start_method": "spawn",
        "timing": f"median and IQR over {REPEATS} repeats x {len(seeds)} seeds; "
                  "pool start-up excluded (pool warmed first)",
        "pool_startup_s_each": startup, "note": "all pool outputs checked equal to sequential",
        "summary": summary, "runs": rows})


# ---------------------------------------------------------------- legacy
def exp_legacy(out_dir: Path, quick: bool) -> None:
    """Re-run the 2025 loop (transcribed, lambdas removed) to document the bugs."""
    sys.path.insert(0, str(ROOT / "tests"))
    from legacy_2025 import legacy_ga_loop

    d = load_map(32)
    pop, gens = (50, 5) if quick else (300, 60)
    recs = legacy_ga_loop(d, population_size=pop, num_generations=gens, seed=42)
    chosen = [-r["chosen_fitness"] for r in recs]
    shortest = [r["shortest_in_population"] for r in recs]
    save(out_dir, "legacy_2025", {
        "map": 32, "population_size": pop, "generations": gens, "seed": 42,
        "note": "tests/legacy_2025.py: 2025 loop with the process pools replaced by map(). "
                "Random numbers are drawn in one process, so per-generation values differ "
                "from a pool run; the first and last values match the Phase 0 pool run.",
        "chosen_best_first": chosen[0], "chosen_best_last": chosen[-1],
        "shortest_in_population_first": shortest[0], "shortest_in_population_last": shortest[-1],
        "generations_where_even_the_shortest_route_uses_a_missing_edge":
            sum(x >= MISSING_EDGE for x in shortest),
        "chosen_best": chosen, "shortest_in_population": shortest})


# ---------------------------------------------------------------- reference
def exp_reference(out_dir: Path, quick: bool) -> None:
    """OR-Tools guided local search: reference solver, not proven optimal."""
    from route_ga.reference import solve_ortools

    limit = 2 if quick else REFERENCE_TIME_LIMIT_S
    repeats = 1 if quick else REFERENCE_REPEATS
    out = {}
    for n in (32, 100):
        d = load_map(n)
        rows = []
        for r in range(repeats):
            perm, info = solve_ortools(d, time_limit_s=limit)
            s = summarise(perm, d)
            rows.append({"repeat": r, "length": s["length"], "cost": s["cost"],
                         "feasible": s["feasible"], "tour": s["tour"], **info})
            print(f"reference map{n} repeat={r}: {s['length']:.0f} feasible={s['feasible']}",
                  flush=True)
        best = min(rows, key=lambda x: x["cost"])
        out[str(n)] = {"runs": rows, "best_cost": best["cost"], "best_tour": best["tour"],
                       "all_feasible": all(x["feasible"] for x in rows),
                       "costs": [x["cost"] for x in rows]}
    save(out_dir, "reference_ortools", {
        "label": "reference solver, not proven optimal",
        "solver": "OR-Tools routing, PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH",
        "time_limit_s": limit, "repeats": repeats, "maps": out})


EXPERIMENTS = ("phase0", "legacy", "tuning", "anchor", "map32", "map100", "map32_quick40",
               "reference",
               "fitness")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=EXPERIMENTS, action="append")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, for smoke tests")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--machine-note", help="stored in every result, e.g. 'shared cloud VM'")
    args = ap.parse_args(argv)
    global MACHINE_NOTE
    MACHINE_NOTE = args.machine_note
    todo = args.only or list(EXPERIMENTS)
    for name in todo:
        t0 = time.perf_counter()
        if name == "phase0":
            from phase0_check import run as phase0_run

            phase0_run(args.out_dir)
        elif name == "legacy":
            exp_legacy(args.out_dir, args.quick)
        elif name == "tuning":
            exp_tuning(args.out_dir, args.quick)
        elif name == "anchor":
            exp_anchor(args.out_dir, args.quick)
        elif name == "map32":
            exp_map(args.out_dir, 32, args.quick)
        elif name == "map100":
            exp_map(args.out_dir, 100, args.quick)
        elif name == "map32_quick40":
            # Re-runs the 40-generation smoke configuration that found a 334 tour
            # during development, so that number is backed by a results file.
            exp_map(args.out_dir, 32, True, name="map32_quick40")
        elif name == "reference":
            exp_reference(args.out_dir, args.quick)
        else:
            exp_fitness(args.out_dir, args.quick)
        print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
    write_tables(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
