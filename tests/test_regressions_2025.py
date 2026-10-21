"""One regression test per bug fixed since 2025.

Each test pins the fixed behaviour of the current code for one bug that the
2025 coursework had. The 2025 code itself is no longer in the repo.
"""

import json
import subprocess
import sys

import pytest

from route_ga.cli import main as cli_main
from route_ga.data import load_map
from route_ga.ga import GAConfig, run_ga
from route_ga.parallel_fitness import PoolEvaluator
from route_ga.route import cost_matrix, route_cost, summarise


def non_increasing(xs):
    return all(b <= a + 1e-9 for a, b in zip(xs, xs[1:], strict=False))


def test_best_route_length_never_increases():
    """Direction bug: the 2025 loop kept the longest routes."""
    d = load_map(32)
    res = run_ga(cost_matrix(d), GAConfig(population_size=100, generations=60), seed=42)
    assert non_increasing(res.history)
    assert res.history[-1] < res.history[0]


def test_selection_prefers_shorter_route():
    """The route reported as best must be the shortest in the population."""
    d = load_map(32)
    res = run_ga(cost_matrix(d), GAConfig(population_size=50, generations=1), seed=0)
    assert res.best_cost == min(res.history)


@pytest.mark.slow
def test_process_pool_path_runs():
    """PicklingError: the 2025 script sent lambdas to a process pool."""
    cost = cost_matrix(load_map(32))
    with PoolEvaluator(cost, workers=2) as pool:
        res = run_ga(cost, GAConfig(population_size=40, generations=5), 0, evaluator=pool)
    assert res.generations_run == 5


def test_missing_edge_is_penalised():
    """Missing edges are 100000 in the data; 2025 checked for 10000."""
    d = load_map(32)
    assert d[0, 5] == 100000
    s = summarise([5, 1] + [n for n in range(2, 32) if n != 5], d)
    assert not s["feasible"] and s["missing_edges"] >= 1
    assert s["cost"] >= 100000


def test_tour_returns_to_depot(square4):
    """Closed tour: 0-1-3-2-0 is 10 + 25 + 30 + 15 = 80, return edge included."""
    assert route_cost([1, 3, 2], cost_matrix(square4)) == 80


def test_result_is_saved_to_disk(tmp_path):
    """2025 only printed (once, outside the loop); nothing was saved."""
    out = tmp_path / "r.json"
    assert cli_main(["--method", "ga", "--generations", "5", "--population", "10",
                     "--out", str(out), "--quiet"]) == 0
    data = json.loads(out.read_text())
    assert data["tour"][0] == data["tour"][-1] == 0
    assert len(data["history"]) == 6


def test_runs_from_any_working_directory(tmp_path):
    """2025 read '../data/...' relative to the current directory."""
    proc = subprocess.run([sys.executable, "-m", "route_ga", "--method", "nn"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "feasible" in proc.stdout
