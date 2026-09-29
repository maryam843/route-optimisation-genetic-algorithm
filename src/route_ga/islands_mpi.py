"""Island-model GA, distributed with MPI (mpi4py), plus a serial twin.

Each island is an independent :class:`route_ga.ga.GeneticAlgorithm` with its
own random stream. Every ``migration_interval`` generations each island sends
copies of its ``migrants`` best tours to the next island in a ring
(rank r -> r+1 mod size) and the receiver replaces its worst tours with them.

``run_islands_serial`` runs exactly the same islands one after another in a
single process. Given the same seed, both functions return the same best
tour, which the tests check. The serial version is also what runs on
machines without MPI (for example the Windows CI job).

Only tested on one machine (several MPI processes on one host).
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from route_ga.ga import GAConfig, GeneticAlgorithm


@dataclass(frozen=True)
class IslandConfig:
    n_islands: int = 4
    migration_interval: int = 25
    migrants: int = 2

    def __post_init__(self) -> None:
        if self.n_islands < 1:
            raise ValueError("n_islands must be >= 1")
        if self.migration_interval < 1:
            raise ValueError("migration_interval must be >= 1")
        if self.migrants < 0:
            raise ValueError("migrants must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IslandResult:
    best_perm: np.ndarray
    best_cost: float
    best_island: int
    history: list[float] = field(default_factory=list)
    island_best_costs: list[float] = field(default_factory=list)
    generations_run: int = 0
    restarts: int = 0
    elapsed_s: float = 0.0
    backend: str = "serial"


def island_rngs(seed: int, n_islands: int) -> list[np.random.Generator]:
    """Independent random streams, one per island, all derived from ``seed``."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_islands)]


def _should_migrate(generation: int, n_islands: int, cfg: IslandConfig) -> bool:
    return n_islands > 1 and cfg.migrants > 0 and generation % cfg.migration_interval == 0


def _combine(
    best: list[tuple[float, np.ndarray]],
    histories: list[list[float]],
    restarts: int,
    generations: int,
    elapsed: float,
    backend: str,
) -> IslandResult:
    costs = [c for c, _ in best]
    winner = int(np.argmin(costs))
    history = np.min(np.array(histories, dtype=float), axis=0).tolist()
    return IslandResult(
        best_perm=np.asarray(best[winner][1], dtype=np.int64).copy(),
        best_cost=float(costs[winner]),
        best_island=winner,
        history=history,
        island_best_costs=[float(c) for c in costs],
        generations_run=generations,
        restarts=restarts,
        elapsed_s=elapsed,
        backend=backend,
    )


def run_islands_serial(
    cost: np.ndarray, ga_config: GAConfig, island_config: IslandConfig, seed: int
) -> IslandResult:
    t0 = time.perf_counter()
    n = island_config.n_islands
    islands = [GeneticAlgorithm(cost, ga_config, rng) for rng in island_rngs(seed, n)]
    for g in range(1, ga_config.generations + 1):
        for isl in islands:
            isl.step()
        if _should_migrate(g, n, island_config):
            outgoing = [isl.best(island_config.migrants) for isl in islands]
            for i, isl in enumerate(islands):
                isl.receive(*outgoing[(i - 1) % n])
    elapsed = time.perf_counter() - t0
    return _combine(
        [(isl.best_cost, isl.best_perm) for isl in islands],
        [isl.history for isl in islands],
        sum(isl.restarts for isl in islands),
        ga_config.generations,
        elapsed,
        "serial",
    )


def run_islands_mpi(
    cost: np.ndarray, ga_config: GAConfig, island_config: IslandConfig, seed: int, comm=None
) -> IslandResult:
    """Run one island per MPI rank. Every rank returns the combined result.

    ``island_config.n_islands`` is ignored: the number of islands is the MPI
    world size.
    """
    from mpi4py import MPI

    comm = comm if comm is not None else MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rng = island_rngs(seed, size)[rank]
    comm.Barrier()
    t0 = MPI.Wtime()
    isl = GeneticAlgorithm(cost, ga_config, rng)
    right, left = (rank + 1) % size, (rank - 1) % size
    for g in range(1, ga_config.generations + 1):
        isl.step()
        if _should_migrate(g, size, island_config):
            outgoing = isl.best(island_config.migrants)
            incoming = comm.sendrecv(outgoing, dest=right, source=left)
            isl.receive(*incoming)
    comm.Barrier()
    elapsed = MPI.Wtime() - t0
    gathered = comm.gather((isl.best_cost, isl.best_perm, isl.history, isl.restarts), root=0)
    result = None
    if rank == 0:
        result = _combine(
            [(c, p) for c, p, _, _ in gathered],
            [h for _, _, h, _ in gathered],
            sum(r for *_, r in gathered),
            ga_config.generations,
            elapsed,
            f"mpi({size})",
        )
    return comm.bcast(result, root=0)


def mpi_world_size_from_env() -> int:
    """World size advertised by the MPI launcher, without importing mpi4py."""
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS"):
        value = os.environ.get(var)
        if value and value.isdigit():
            return int(value)
    return 1


def mpiexec_command(n_procs: int, *args: str) -> list[str]:
    """``mpiexec -n N`` plus the flags Open MPI needs in containers and on CI."""
    import shutil
    import subprocess

    exe = shutil.which("mpiexec")
    if exe is None:
        raise FileNotFoundError("mpiexec not found on PATH")
    cmd = [exe, "-n", str(n_procs)]
    banner = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    if "Open MPI" in banner or "OpenRTE" in banner:
        cmd.append("--oversubscribe")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd.append("--allow-run-as-root")
    return [*cmd, *args]


def clean_mpi_env() -> dict[str, str]:
    """Copy of os.environ without the variables a running MPI process sets.

    A process that already called MPI_Init (for example a test using
    ``MPI.COMM_SELF``) exports OMPI_*/PMIX_* variables; a child ``mpiexec``
    that inherits them fails to start.
    """
    prefixes = ("OMPI_", "PMIX_", "PMI_", "OPAL_", "PRTE_")
    return {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
