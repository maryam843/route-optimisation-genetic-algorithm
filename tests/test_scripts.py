"""Scripts: experiments (quick mode), figures, README claims check, PII scan."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ROOT, mpi_available

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
    only = ["legacy", "tuning", "anchor", "fitness", "reference"]
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
    """T11: the committed README against the committed results."""
    errors = check_readme_claims.check(ROOT / "README.md", ROOT / "results")
    assert errors == []


def test_claims_checker_catches_problems(tmp_path):
    readme = tmp_path / "README.md"
    blocks = results_tables.render_all(ROOT / "results")
    name = next(iter(blocks))
    readme.write_text(
        f"<!-- BEGIN:{name} -->\n| wrong |\n<!-- END:{name} -->\n"
        "The GA finds the optimal route \u2014 on a fleet of vans.\n"
        "It found a tour of 12,347 in 3.14159 s, tested on Windows.\n", encoding="utf-8")
    errors = check_readme_claims.check(readme, ROOT / "results")
    joined = "\n".join(errors)
    assert "README differs" in joined
    assert "optimal" in joined and "multi-vehicle" in joined and "em dash" in joined
    assert "number 12,347 not found" in joined and "number 3.14159 not found" in joined
    assert "tested on windows" in joined.lower()
    assert check_readme_claims.main(["--readme", str(readme)]) == 1


def test_pii_scan_clean():
    """T12: no personal data or secrets anywhere in the git history."""
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
    assert pii_scan.check_text(f"id {FAKE_ID}", "x")
    assert not pii_scan.check_text("185722545+maryam843@users.noreply.github.com", "x")


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
