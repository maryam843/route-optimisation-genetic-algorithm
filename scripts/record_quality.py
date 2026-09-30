"""Record test counts, coverage and code size in results/quality.json.

Runs the test suite twice: once normally and once with ROUTE_GA_NO_MPI=1
(the MPI tests skip), each with coverage of src/route_ga. The README and the
report quote these numbers, and scripts/check_readme_claims.py checks them.

Usage: python scripts/record_quality.py [--out results/quality.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_suite(no_mpi: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        junit, cov = Path(tmp) / "junit.xml", Path(tmp) / "cov.json"
        env = {**os.environ}
        if no_mpi:
            env["ROUTE_GA_NO_MPI"] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--cov=route_ga",
             f"--cov-report=json:{cov}", f"--junitxml={junit}"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        suite = ET.parse(junit).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        skipped = xfailed = 0
        skip_reasons = []
        for case in suite.iter("testcase"):
            node = case.find("skipped")
            if node is not None:
                if node.get("type") == "pytest.xfail":
                    xfailed += 1
                else:
                    skipped += 1
                    skip_reasons.append(node.get("message", ""))
        total = int(suite.get("tests"))
        failed = int(suite.get("failures")) + int(suite.get("errors"))
        cov_pct = json.loads(cov.read_text())["totals"]["percent_covered"]
    return {"returncode": proc.returncode, "collected": total,
            "passed": total - skipped - xfailed - failed, "failed": failed,
            "xfailed": xfailed, "skipped": skipped, "skip_reasons": sorted(set(skip_reasons)),
            "coverage_pct": round(cov_pct, 1)}


def code_stats() -> dict:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()

    def lines(prefix: str) -> int:
        return sum(len((ROOT / f).read_text(encoding="utf-8").splitlines())
                   for f in files if f.startswith(prefix) and f.endswith(".py"))

    sizes = {f: (ROOT / f).stat().st_size for f in files if (ROOT / f).exists()}
    biggest = max(sizes, key=sizes.get)
    return {"python_lines": {"src": lines("src/"), "tests": lines("tests/"),
                             "scripts": lines("scripts/")},
            "tracked_files": len(files), "tracked_bytes": sum(sizes.values()),
            "largest_file": biggest, "largest_file_bytes": sizes[biggest]}


def main(argv: list[str] | None = None) -> int:
    from route_ga.provenance import environment_info

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "quality.json")
    args = ap.parse_args(argv)
    payload = {"experiment": "quality", "environment": environment_info(),
               "tests": run_suite(no_mpi=False), "tests_without_mpi": run_suite(no_mpi=True),
               "code": code_stats()}
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("tests", "tests_without_mpi", "code")}, indent=1))
    ok = payload["tests"]["returncode"] == 0 and payload["tests_without_mpi"]["returncode"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
