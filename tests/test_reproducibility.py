"""The same seed gives the same best route and length."""

from route_ga.cli import main
from route_ga.data import load_map
from route_ga.ga import GAConfig, run_ga
from route_ga.islands_mpi import IslandConfig, run_islands_serial
from route_ga.route import cost_matrix

CFG = GAConfig(population_size=40, generations=40)


def test_ga_same_seed_same_result():
    cost = cost_matrix(load_map(100))
    a, b = run_ga(cost, CFG, seed=9), run_ga(cost, CFG, seed=9)
    assert a.best_perm.tolist() == b.best_perm.tolist()
    assert a.best_cost == b.best_cost and a.history == b.history


def test_ga_different_seed_differs():
    cost = cost_matrix(load_map(100))
    assert run_ga(cost, CFG, seed=1).history != run_ga(cost, CFG, seed=2).history


def test_islands_same_seed_same_result():
    cost = cost_matrix(load_map(32))
    isl = IslandConfig(3, 10, 2)
    a = run_islands_serial(cost, CFG, isl, seed=5)
    b = run_islands_serial(cost, CFG, isl, seed=5)
    assert a.best_perm.tolist() == b.best_perm.tolist() and a.best_cost == b.best_cost


def test_cli_same_seed_same_json(tmp_path):
    import json

    paths = []
    for tag in ("a", "b"):
        p = tmp_path / f"{tag}.json"
        main(["--map", "32", "--method", "ga", "--seed", "11", "--generations", "20",
              "--population", "20", "--quiet", "--out", str(p)])
        paths.append(json.loads(p.read_text()))
    assert paths[0]["tour"] == paths[1]["tour"]
    assert paths[0]["length"] == paths[1]["length"]
