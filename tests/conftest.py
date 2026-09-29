import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    # T7: force the Windows/macOS start method everywhere, so a pickling or
    # bootstrapping bug shows up on Linux too.
    mp.set_start_method("spawn", force=True)


def mpi_available() -> tuple[bool, str]:
    if sys.platform == "win32":
        return False, "MPI tests run on Linux only (no MS-MPI on the Windows runner)"
    if shutil.which("mpiexec") is None:
        return False, "mpiexec not on PATH"
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False, "mpi4py not installed"
    return True, ""


@pytest.fixture
def square4():
    # Hand-made symmetric matrix; tour 0-1-3-2-0 = 10 + 25 + 30 + 15 = 80.
    return np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ], dtype=float)


@pytest.fixture
def small_random():
    rng = np.random.default_rng(7)
    d = rng.integers(1, 100, size=(12, 12)).astype(float)
    d = np.triu(d, 1)
    d = d + d.T
    np.fill_diagonal(d, 0)
    return d
