"""Load the two distance matrices.

The two distance tables were supplied with the original exercise and are kept
byte for byte. Their real-world origin is unknown.

A cell >= ``MISSING_EDGE`` (100000) means there is no road between two nodes.
The earlier code compared against 10000, so the penalty never fired.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

MISSING_EDGE = 100_000.0
DEPOT = 0

MAP_FILES = {
    32: "city_distances.csv",
    100: "city_distances_extended.csv",
}


def _find_data_dir() -> Path:
    """Folder with the map CSVs.

    An installed package ships them as ``route_ga.maps`` (package data). A
    source checkout used without installing falls back to ``data/``.
    """
    packaged = Path(str(resources.files("route_ga"))) / "maps"
    if (packaged / MAP_FILES[32]).is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "data"


DATA_DIR = _find_data_dir()


def load_matrix(path: str | Path) -> np.ndarray:
    """Read a square distance matrix from a CSV with one header row."""
    matrix = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float, ndmin=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"distance matrix must be square, got {matrix.shape}")
    return matrix


def load_map(n_nodes: int) -> np.ndarray:
    """Load the 32-node or the 100-node map."""
    if n_nodes not in MAP_FILES:
        raise ValueError(f"unknown map {n_nodes}; choose from {sorted(MAP_FILES)}")
    return load_matrix(DATA_DIR / MAP_FILES[n_nodes])


def missing_mask(matrix: np.ndarray) -> np.ndarray:
    """Boolean mask of missing edges (off-diagonal cells >= MISSING_EDGE)."""
    return np.asarray(matrix) >= MISSING_EDGE


def is_symmetric(matrix: np.ndarray) -> bool:
    return bool(np.allclose(matrix, np.asarray(matrix).T))
