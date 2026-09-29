"""Island model: serial twin, in-process MPI, and mpiexec integration (T4)."""

import json
import subprocess
import sys

import numpy as np
import pytest
from conftest import mpi_available

from route_ga.data import load_map
from route_ga.ga import GAConfig
from route_ga.islands_mpi import (
    IslandConfig,
    clean_mpi_env,
    island_rngs,
    mpi_world_size_from_env,
    mpiexec_command,
    run_islands_mpi,
    run_islands_serial,
)
from route_ga.route import cost_matrix, is_valid_tour, to_tour

MPI_OK, MPI_REASON = mpi_available()
needs_mpi = pytest.mark.skipif(not MPI_OK, reason=MPI_REASON)

SMALL_CFG = GAConfig(population_size=20, generations=30)


def small_matrix(n=12, seed=11):
    rng = np.random.default_rng(seed)
    d = rng.integers(1, 100, size=(n, n)).astype(float)
    d[rng.random((n, n)) < 0.19] = 100000
    d = np.triu(d, 1)
    d = d + d.T
    np.fill_diagonal(d, 0)
    return d


def write_csv(path, d):
    header = ",".join(str(i) for i in range(d.shape[0]))
    np.savetxt(path, d, delimiter=",", header=header, comments="", fmt="%.1f")


def test_island_rngs_are_independent_and_reproducible():
    a = [r.random() for r in island_rngs(5, 3)]
    b = [r.random() for r in island_rngs(5, 3)]
    assert a == b and len(set(a)) == 3


@pytest.mark.parametrize("bad", [
    {"n_islands": 0}, {"migration_interval": 0}, {"migrants": -1}])
def test_island_config_validation(bad):
    with pytest.raises(ValueError):
        IslandConfig(**bad)


def test_serial_islands_valid_and_monotone():
    cost = cost_matrix(load_map(32))
    res = run_islands_serial(cost, SMALL_CFG, IslandConfig(3, 5, 2), seed=0)
    assert is_valid_tour(to_tour(res.best_perm), 32)
    assert len(res.island_best_costs) == 3
    assert res.best_cost == min(res.island_best_costs)
    assert all(b <= a + 1e-9 for a, b in zip(res.history, res.history[1:], strict=False))


def test_migration_spreads_the_best_tour():
    # With migration every generation and two islands, the global best after
    # the last migration must be present on both islands, so both island
    # bests are equal once the run ends on a migration step.
    cost = cost_matrix(load_map(32))
    cfg = GAConfig(population_size=20, generations=10, stagnation_limit=0)
    res = run_islands_serial(cost, cfg, IslandConfig(2, 1, 1), seed=1)
    assert res.island_best_costs[0] == res.island_best_costs[1]
    none = run_islands_serial(cost, cfg, IslandConfig(2, 1, 0), seed=1)
    assert none.island_best_costs[0] != none.island_best_costs[1]


def test_world_size_env(monkeypatch):
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS"):
        monkeypatch.delenv(var, raising=False)
    assert mpi_world_size_from_env() == 1
    monkeypatch.setenv("PMI_SIZE", "3")
    assert mpi_world_size_from_env() == 3


@needs_mpi
def test_mpi_in_process_single_rank_matches_serial():
    from mpi4py import MPI

    cost = cost_matrix(small_matrix())
    mpi = run_islands_mpi(cost, SMALL_CFG, IslandConfig(1, 5, 2), seed=2, comm=MPI.COMM_SELF)
    ser = run_islands_serial(cost, SMALL_CFG, IslandConfig(1, 5, 2), seed=2)
    assert mpi.best_perm.tolist() == ser.best_perm.tolist()
    assert mpi.history == ser.history
    assert mpi.backend == "mpi(1)"


def _run_mpiexec(tmp_path, tag, csv, n=2, seed=3):
    out = tmp_path / f"{tag}.json"
    cmd = mpiexec_command(
        n, sys.executable, "-m", "route_ga", "--matrix", str(csv), "--method", "ga-islands",
        "--seed", str(seed), "--generations", "30", "--population", "20",
        "--migration-interval", "5", "--quiet", "--out", str(out))
    subprocess.run(cmd, check=True, timeout=300, env=clean_mpi_env())
    return json.loads(out.read_text())


@needs_mpi
@pytest.mark.slow
def test_mpiexec_two_islands_valid_reproducible_and_equal_to_serial(tmp_path):
    d = small_matrix()
    csv = tmp_path / "small.csv"
    write_csv(csv, d)
    a = _run_mpiexec(tmp_path, "a", csv)
    b = _run_mpiexec(tmp_path, "b", csv)
    assert a["backend"] == "mpi(2)"
    assert is_valid_tour(a["tour"], 12)
    assert a["tour"] == b["tour"] and a["cost"] == b["cost"]
    ser = run_islands_serial(cost_matrix(d), SMALL_CFG, IslandConfig(2, 5, 2), seed=3)
    assert to_tour(ser.best_perm) == a["tour"]
