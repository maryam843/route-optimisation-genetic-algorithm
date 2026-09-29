"""Load the course distance matrices.

The two CSV files in ``data/`` were provided by DSAI3202 course staff (UDST,
Spring 2025). Modified by me: no. This loader is my 2026 code.

A cell >= ``MISSING_EDGE`` (100000) means there is no road between two nodes.
The 2025 code compared against 10000, so the penalty never fired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MISSING_EDGE = 100_000.0
DEPOT = 0

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MAP_FILES = {
    32: "city_distances.csv",
    100: "city_distances_extended.csv",
}


def load_matrix(path: str | Path) -> np.ndarray:
    """Read a square distance matrix from a CSV with one header row."""
    matrix = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float, ndmin=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"distance matrix must be square, got {matrix.shape}")
    return matrix


def load_map(n_nodes: int) -> np.ndarray:
    """Load the 32-node or the 100-node course map."""
    if n_nodes not in MAP_FILES:
        raise ValueError(f"unknown map {n_nodes}; choose from {sorted(MAP_FILES)}")
    return load_matrix(DATA_DIR / MAP_FILES[n_nodes])


def missing_mask(matrix: np.ndarray) -> np.ndarray:
    """Boolean mask of missing edges (off-diagonal cells >= MISSING_EDGE)."""
    return np.asarray(matrix) >= MISSING_EDGE


def is_symmetric(matrix: np.ndarray) -> bool:
    return bool(np.allclose(matrix, np.asarray(matrix).T))
