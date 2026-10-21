"""CI gate on the pytest JUnit XML: 0 failures, no xfails, and no skipped tests
except MPI tests on Windows.

Usage: python scripts/check_test_summary.py junit.xml [--allow-mpi-skips]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("junit")
    ap.add_argument("--allow-mpi-skips", action="store_true")
    args = ap.parse_args(argv)
    root = ET.parse(args.junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    failed = int(suite.get("failures")) + int(suite.get("errors"))
    xfailed, skipped, bad = [], [], []
    for case in suite.iter("testcase"):
        node = case.find("skipped")
        if node is None:
            continue
        name = f"{case.get('classname')}::{case.get('name')}"
        if node.get("type") == "pytest.xfail":
            xfailed.append(name)
        else:
            skipped.append(name)
            if not (args.allow_mpi_skips and "MPI" in node.get("message", "")):
                bad.append(f"{name}: {node.get('message')}")
    print(f"tests={suite.get('tests')} failed={failed} xfailed={len(xfailed)} "
          f"skipped={len(skipped)}")
    problems = []
    if failed:
        problems.append(f"{failed} failed")
    problems += [f"unexpected xfail: {x}" for x in xfailed]
    problems += [f"unexpected skip: {b}" for b in bad]
    for p in problems:
        print("FAIL", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
