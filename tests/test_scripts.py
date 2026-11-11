"""Scripts: experiments (quick mode), figures, README claims check, PII scan."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ROOT, mpi_available

from route_ga.provenance import mpiexec_version

sys.path.insert(0, str(ROOT / "scripts"))
import check_readme_claims  # noqa: E402
import make_figures  # noqa: E402
import pii_scan  # noqa: E402
import results_tables  # noqa: E402
import run_experiments  # noqa: E402

MPI_OK, MPI_REASON = mpi_available()


@pytest.fixture(scope="module")
def quick_results(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    only = ["tuning", "anchor", "fitness", "reference"]
    if MPI_OK:
        only.append("map32")
    args = ["--quick", "--out-dir", str(out)]
    for name in only:
        args += ["--only", name]
    assert run_experiments.main(args) == 0
    return out


def test_quick_results_have_provenance(quick_results):
    for p in quick_results.glob("*.json"):
        data = json.loads(p.read_text())
        env = data["environment"]
        assert env["timestamp_utc"] and env["python"] and env["nproc"] and env["os"]
        assert "cpu_model" in env and "ram_bytes" in env
        assert "mpiexec" in env
        if MPI_OK:
            # Recorded exactly as the launcher reports it. Open MPI 5 on Ubuntu
            # 26.04 gives no version line on stdout, so the value can be None.
            assert env["mpiexec"] == mpiexec_version()
        assert env["code"]["src_commit"] is None or len(env["code"]["src_commit"]) == 40
    assert (quick_results / "tables.md").exists()


def test_anchor_brute_force_is_a_lower_bound(quick_results):
    data = json.loads((quick_results / "anchor_8node.json").read_text())
    for r in data["runs"]:
        for m in ("nn", "nn+2opt", "ga"):
            assert r[m] >= r["optimum"] - 1e-9


@pytest.mark.skipif(not MPI_OK, reason=MPI_REASON)
def test_quick_map_and_figures(quick_results, tmp_path):
    data = json.loads((quick_results / "map32.json").read_text())
    assert data["summary"]["ga-islands2"]["all_match_serial"]
    assert data["summary"]["ga-islands4"]["all_match_serial"]
    assert make_figures.main(["--results", str(quick_results), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "convergence_32.png").stat().st_size > 10_000
    assert (tmp_path / "best_route_32.png").exists()


def test_figures_without_results(tmp_path):
    assert make_figures.main(["--results", str(tmp_path), "--out", str(tmp_path / "f")]) == 1


def test_generations_to_converge():
    assert run_experiments.generations_to_converge([9, 7, 7, 5, 5]) == 3
    assert run_experiments.mean_sd([4.0]) == (4.0, 0.0)
    m = run_experiments.median_iqr([1, 2, 3, 4, 5])
    assert m["median"] == 3 and m["q1"] == 2 and m["q3"] == 4 and m["n"] == 5


def test_quick_reference(quick_results):
    data = json.loads((quick_results / "reference_ortools.json").read_text())
    assert data["label"] == "reference solver, not proven optimal"
    for n in ("32", "100"):
        assert data["maps"][n]["all_feasible"]


def test_repository_readme_matches_results():
    """The committed README against the committed results."""
    errors = check_readme_claims.check(ROOT / "README.md", ROOT / "results")
    assert errors == []


def test_provenance_check_skips_without_full_history(tmp_path):
    """A ZIP download or a shallow clone skips the commit check with a reason."""
    assert "ZIP" in check_readme_claims.provenance_skip_reason(tmp_path)
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", ROOT.as_uri(), str(shallow)],
                   check=True)
    assert "shallow" in check_readme_claims.provenance_skip_reason(shallow)


def test_claims_checker_catches_problems(tmp_path):
    readme = tmp_path / "README.md"
    blocks = results_tables.render_all(ROOT / "results")
    name = next(iter(blocks))
    readme.write_text(
        f"<!-- BEGIN:{name} -->\n| wrong |\n<!-- END:{name} -->\n"
        "It found a tour of 12,347 in 3.14159 s.\n", encoding="utf-8")
    errors = check_readme_claims.check(readme, ROOT / "results")
    joined = "\n".join(errors)
    assert "README differs" in joined
    assert "number 12,347 not found" in joined and "number 3.14159 not found" in joined
    assert check_readme_claims.main(["--readme", str(readme)]) == 1


def test_pii_scan_clean():
    """No personal data or secrets anywhere in the git history."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    hits, n_commits, _ = pii_scan.scan(ROOT)
    assert n_commits >= 1
    assert hits == []


# Built at run time so this file does not itself contain an email or an ID.
FAKE_EMAIL = "someone" + "@" + "example.org"
FAKE_ID = "603" + "12345"
NOREPLY = "185722545+maryam843@users.noreply.github.com"


def test_pii_scan_detects(tmp_path):
    assert pii_scan.check_text(f"mail me at {FAKE_EMAIL}", "x")
    assert pii_scan.check_text("saved in " + "/home/" + "someone/run.json", "x")
    assert not pii_scan.check_text("185722545+maryam843@users.noreply.github.com", "x")
    assert not pii_scan.check_text(f"id {FAKE_ID}", "x")


def test_pii_scan_extra_patterns_stay_out_of_the_repo(tmp_path, monkeypatch):
    """Personal patterns come from an untracked file or an environment variable."""
    monkeypatch.delenv(pii_scan.ENV_VAR, raising=False)
    assert pii_scan.extra_patterns(tmp_path) == {}
    monkeypatch.setenv(pii_scan.ENV_VAR, "# comment\n" + FAKE_ID[:3] + r"\d{5}")
    extra = pii_scan.extra_patterns(tmp_path)
    hits = pii_scan.check_text(f"id {FAKE_ID}", "x", extra)
    assert hits == ["x: extra_1: [match hidden]"]
    monkeypatch.delenv(pii_scan.ENV_VAR)
    (tmp_path / pii_scan.LOCAL_FILE).write_text("some" + "one\n", encoding="utf-8")
    hits = pii_scan.check_text(f"mail me at {FAKE_EMAIL}", "x", pii_scan.extra_patterns(tmp_path))
    assert hits[-1] == "x: extra_1: [match hidden]"


def test_pii_scan_on_new_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text(f"contact: {FAKE_EMAIL}\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": NOREPLY,
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": NOREPLY}

    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "x"], check=True,
                   env={**os.environ, **env})
    assert pii_scan.main([str(tmp_path)]) == 1


def test_tables_script_entry(tmp_path):
    assert results_tables.write_tables(Path(tmp_path)).exists()


def test_fitness_evaluation_formula_matches_a_counting_evaluator():
    """The README's evaluation counts come from this formula; check it on real runs."""
    import numpy as np

    from route_ga.data import load_map
    from route_ga.ga import GAConfig, GeneticAlgorithm, VectorEvaluator
    from route_ga.route import cost_matrix

    class Counting(VectorEvaluator):
        def __init__(self, cost):
            super().__init__(cost)
            self.rows = 0

        def __call__(self, population):
            self.rows += len(population)
            return super().__call__(population)

    cost = cost_matrix(load_map(32))
    cfg = GAConfig(population_size=20, generations=40, stagnation_limit=3)
    ev = Counting(cost)
    ga = GeneticAlgorithm(cost, cfg, np.random.default_rng(0), ev)
    for _ in range(cfg.generations):
        ga.step()
    assert ga.restarts > 0
    assert ev.rows == results_tables.fitness_evaluations(20, 40, 1, ga.restarts)
    assert results_tables.fitness_evaluations(200, 1000, 4, 16) == 200 * (4 * 1001 + 16)
