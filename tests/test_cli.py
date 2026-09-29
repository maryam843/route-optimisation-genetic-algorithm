import json
import subprocess
import sys

import pytest

from route_ga.cli import main
from route_ga.route import is_valid_tour


def test_help_via_python_m():
    proc = subprocess.run([sys.executable, "-m", "route_ga", "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    for flag in ("--map", "--method", "--seed", "ga-islands"):
        assert flag in proc.stdout


@pytest.mark.parametrize("n_map", [32, 100])
@pytest.mark.parametrize("method", ["nn", "2opt", "ga", "ga-islands"])
def test_every_method_on_both_maps(n_map, method, tmp_path, capsys):
    out = tmp_path / "r.json"
    args = ["--map", str(n_map), "--method", method, "--seed", "3",
            "--generations", "5", "--population", "12", "--islands", "2",
            "--migration-interval", "2", "--out", str(out)]
    assert main(args) == 0
    data = json.loads(out.read_text())
    assert is_valid_tour(data["tour"], n_map)
    assert data["n_nodes"] == n_map
    assert data["environment"]["python"]
    assert "tour length=" in capsys.readouterr().out


def test_polish_and_pool_workers(tmp_path):
    out = tmp_path / "r.json"
    assert main(["--method", "ga", "--polish", "--workers", "2", "--generations", "3",
                 "--population", "10", "--quiet", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["method"] == "ga+2opt"
    assert data["workers"] == 2


def test_custom_matrix(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("0,1,2,3\n0,10,15,20\n10,0,35,25\n15,35,0,30\n20,25,30,0\n")
    out = tmp_path / "r.json"
    assert main(["--matrix", str(csv), "--method", "2opt", "--quiet", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["length"] == 80 and data["map"] is None


def test_bad_arguments():
    with pytest.raises(SystemExit):
        main(["--map", "20"])
    with pytest.raises(SystemExit):
        main(["--method", "annealing"])
