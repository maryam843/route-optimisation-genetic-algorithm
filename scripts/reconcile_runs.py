"""V5 number reconciliation: re-run every ad-hoc configuration whose tour length
was seen during development (smoke runs, README command checks, CI smoke step),
so each of those numbers is backed by results/reconciliation.json.

None of these runs is lower than the "best found" tours in the README; the
check_readme_claims script and the report compare them.

Usage: python scripts/reconcile_runs.py [--out-dir results]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from route_ga.cli import build_parser, solve  # noqa: E402
from route_ga.data import load_map  # noqa: E402
from route_ga.islands_mpi import clean_mpi_env, mpiexec_command  # noqa: E402
from route_ga.provenance import environment_info  # noqa: E402
from route_ga.reference import solve_ortools  # noqa: E402
from route_ga.route import summarise  # noqa: E402

# (where it was seen, CLI arguments)
CLI_RUNS = [
    ("development smoke run", "--map 32 --method ga --seed 1 --generations 300"),
    ("development smoke run", "--map 32 --method ga-islands --seed 1 --generations 300"),
    ("development smoke run (serial twin of the MPI run)",
     "--map 32 --method ga-islands --islands 2 --seed 1 --generations 300"),
    ("development smoke run", "--map 100 --method ga --seed 1 --generations 2000"),
    ("development smoke run", "--map 100 --method ga --seed 1 --generations 2000 --polish"),
    ("README command check", "--map 32 --method nn"),
    ("README command check", "--map 100 --method 2opt"),
    ("README command check", "--map 32 --method ga --seed 0"),
    ("README command check", "--map 32 --method ga --seed 0 --polish"),
    ("README command check", "--map 32 --method ga-islands --islands 4 --seed 0"),
    ("CI smoke step", "--map 32 --method ga --seed 0 --generations 50"),
]
# Seen once during development but not saved at the time, and not reproduced
# exactly: OR-Tools with a time limit is not deterministic between runs.
OBSERVED_NOT_REPRODUCED = [
    {"seen_in": "first OR-Tools trial (5-second limit)", "map": 100, "length": 382.0,
     "note": "the re-run below gave a different length; both are longer than the "
             "30-second OR-Tools tours in reference_ortools.json"},
]

MPI_RUNS = [
    ("development smoke run", 2, "--map 32 --method ga-islands --seed 1 --generations 300"),
    ("README command check", 4, "--map 100 --method ga-islands --seed 0"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    rows = []
    for where, cli in CLI_RUNS:
        ns = build_parser().parse_args(cli.split())
        r = solve(ns, load_map(ns.map))
        rows.append({"seen_in": where, "command": f"python -m route_ga {cli}",
                     "length": r["length"], "feasible": r["feasible"]})
        print(rows[-1], flush=True)
    for where, n, cli in MPI_RUNS:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.json"
            cmd = mpiexec_command(n, sys.executable, "-m", "route_ga", *cli.split(),
                                  "--quiet", "--out", str(out))
            subprocess.run(cmd, check=True, env=clean_mpi_env())
            r = json.loads(out.read_text())
        rows.append({"seen_in": where, "command": f"mpiexec -n {n} python -m route_ga {cli}",
                     "length": r["length"], "feasible": r["feasible"]})
        print(rows[-1], flush=True)
    for n in (32, 100):
        perm, info = solve_ortools(load_map(n), time_limit_s=5)
        s = summarise(perm, load_map(n))
        rows.append({"seen_in": "first OR-Tools trial (5-second limit)",
                     "command": f"solve_ortools(map {n}, time_limit_s=5)",
                     "length": s["length"], "feasible": s["feasible"]})
        print(rows[-1], flush=True)
    from run_experiments import exp_map

    exp_map(args.out_dir, 100, True, name="map100_quick40")
    payload = {"experiment": "reconciliation", "environment": environment_info(),
               "note": "ad-hoc runs seen during development, re-run with the final code; "
                       "the 40-generation smoke runs are in map32_quick40.json and "
                       "map100_quick40.json",
               "runs": rows, "observed_not_reproduced": OBSERVED_NOT_REPRODUCED}
    path = args.out_dir / "reconciliation.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
