"""A non-editable install must find the two maps (package data)."""

import subprocess
import sys

import pytest
from conftest import ROOT


@pytest.mark.slow
def test_non_editable_install_runs_from_another_folder(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
    subprocess.run([str(python), "-m", "pip", "install", "-q", str(ROOT)], check=True,
                   timeout=600)
    work = tmp_path / "elsewhere"
    work.mkdir()
    proc = subprocess.run([str(python), "-m", "route_ga", "--map", "32", "--method", "nn"],
                          cwd=work, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "tour length=445 (feasible)" in proc.stdout
    where = subprocess.run([str(python), "-c", "import route_ga.data as d; print(d.DATA_DIR)"],
                           cwd=work, capture_output=True, text=True, check=True).stdout
    assert str(ROOT) not in where and "maps" in where
