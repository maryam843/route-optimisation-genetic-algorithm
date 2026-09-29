"""T2: one regression test per bug fixed since 2025.

Each test is parametrised over ``impl``. The ``"2025"`` case runs the code in
original_2025/ and is marked ``xfail(strict=True)``: pytest reports it as
XFAIL when the old bug shows up, and the suite FAILS if the old code ever
passes. The ``"2026"`` case runs the new package and must pass.
"""

import json
import shutil
import subprocess
import sys

import pytest
from legacy_2025 import ORIGINAL, legacy_ga_loop, load_original_functions

from route_ga.cli import main as cli_main
from route_ga.data import load_map
from route_ga.ga import GAConfig, run_ga
from route_ga.parallel_fitness import PoolEvaluator
from route_ga.route import cost_matrix, route_cost, summarise

ROOT = ORIGINAL.parent
OLD = pytest.param("2025", marks=pytest.mark.xfail(strict=True, reason="bug in the 2025 code"))
NEW = "2026"
IMPLS = [OLD, NEW]


def non_increasing(xs):
    return all(b <= a + 1e-9 for a, b in zip(xs, xs[1:], strict=False))


@pytest.mark.parametrize("impl", IMPLS)
def test_best_route_length_never_increases(impl):
    """Direction bug: the 2025 loop kept the longest routes."""
    d = load_map(32)
    if impl == "2025":
        records = legacy_ga_loop(d, population_size=300, num_generations=60)
        chosen_lengths = [-r["chosen_fitness"] for r in records]
        assert non_increasing(chosen_lengths), (chosen_lengths[0], chosen_lengths[-1])
    else:
        res = run_ga(cost_matrix(d), GAConfig(population_size=100, generations=60), seed=42)
        assert non_increasing(res.history)
        assert res.history[-1] < res.history[0]


@pytest.mark.parametrize("impl", IMPLS)
def test_selection_prefers_shorter_route(impl):
    """The route reported as best must be the shortest in the population."""
    d = load_map(32)
    if impl == "2025":
        rec = legacy_ga_loop(d, population_size=50, num_generations=1)[0]
        assert -rec["chosen_fitness"] == rec["shortest_in_population"]
    else:
        res = run_ga(cost_matrix(d), GAConfig(population_size=50, generations=1), seed=0)
        assert res.best_cost == min(res.history)


def _copy_2025_layout(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "data").mkdir()
    for name in ("genetic_algorithm_trial.py", "genetic_algorithms_functions.py"):
        shutil.copy(ORIGINAL / "src" / name, tmp_path / "src" / name)
    shutil.copy(ROOT / "data" / "city_distances.csv", tmp_path / "data" / "city_distances.csv")


@pytest.mark.slow
@pytest.mark.parametrize("impl", IMPLS)
def test_process_pool_path_runs(impl, tmp_path):
    """PicklingError: the 2025 script sent lambdas to a process pool."""
    if impl == "2025":
        pytest.importorskip("pandas", reason="the 2025 script needs pandas")
        _copy_2025_layout(tmp_path)
        proc = subprocess.run(
            [sys.executable, "genetic_algorithm_trial.py"], cwd=tmp_path / "src",
            capture_output=True, text=True, timeout=600)
        print(proc.stderr[-300:])  # shown with -rx: "PicklingError: Can't pickle <lambda>"
        assert "PicklingError" not in proc.stderr
        assert proc.returncode == 0
    else:
        cost = cost_matrix(load_map(32))
        with PoolEvaluator(cost, workers=2) as pool:
            res = run_ga(cost, GAConfig(population_size=40, generations=5), 0, evaluator=pool)
        assert res.generations_run == 5


@pytest.mark.parametrize("impl", IMPLS)
def test_missing_edge_is_penalised(impl):
    """Missing edges are 100000 in the data; 2025 checked for 10000."""
    d = load_map(32)
    assert d[0, 5] == 100000
    if impl == "2025":
        f = load_original_functions()
        assert f.calculate_fitness([0, 5, 1], d) <= -1e6
    else:
        s = summarise([5, 1] + [n for n in range(2, 32) if n != 5], d)
        assert not s["feasible"] and s["missing_edges"] >= 1
        assert s["cost"] >= 100000


@pytest.mark.parametrize("impl", IMPLS)
def test_tour_returns_to_depot(impl, square4):
    """Closed tour: 0-1-3-2-0 is 10 + 25 + 30 + 15 = 80, return edge included."""
    if impl == "2025":
        f = load_original_functions()
        assert f.calculate_fitness([0, 1, 3, 2], square4) == -80
    else:
        assert route_cost([1, 3, 2], cost_matrix(square4)) == 80


@pytest.mark.parametrize("impl", IMPLS)
def test_result_is_saved_to_disk(impl, tmp_path):
    """2025 only printed (once, outside the loop); nothing was saved."""
    if impl == "2025":
        source = (ORIGINAL / "src" / "genetic_algorithm_trial.py").read_text()
        assert any(k in source for k in ("json.dump", "to_csv", "np.save", "open("))
    else:
        out = tmp_path / "r.json"
        assert cli_main(["--method", "ga", "--generations", "5", "--population", "10",
                         "--out", str(out), "--quiet"]) == 0
        data = json.loads(out.read_text())
        assert data["tour"][0] == data["tour"][-1] == 0
        assert len(data["history"]) == 6


@pytest.mark.parametrize("impl", IMPLS)
def test_runs_from_any_working_directory(impl, tmp_path):
    """2025 read '../data/...' relative to the current directory."""
    if impl == "2025":
        pytest.importorskip("pandas", reason="the 2025 script needs pandas")
        _copy_2025_layout(tmp_path)
        proc = subprocess.run(
            [sys.executable, "src/genetic_algorithm_trial.py"], cwd=tmp_path,
            capture_output=True, text=True, timeout=600)
        print(proc.stderr[-300:])
        assert "FileNotFoundError" not in proc.stderr
    else:
        proc = subprocess.run([sys.executable, "-m", "route_ga", "--method", "nn"],
                              cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert "feasible" in proc.stdout


def test_legacy_transcription_matches_phase0_numbers():
    """Same first and last values as running the real 2025 script with the lambdas
    replaced by module-level functions (pop 300, 60 generations, seed 42)."""
    records = legacy_ga_loop(load_map(32), population_size=300, num_generations=60)
    assert -records[0]["chosen_fitness"] == 1_300_840
    assert -records[-1]["chosen_fitness"] == 1_400_983
    assert records[0]["shortest_in_population"] == 101_498
    assert records[-1]["shortest_in_population"] == 201_175
