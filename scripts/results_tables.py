"""Render the Markdown tables used in the README from the JSON in results/.

The README copies these tables between ``<!-- BEGIN:<name> -->`` and
``<!-- END:<name> -->`` markers; scripts/check_readme_claims.py re-renders
them and fails if the README differs.
"""

from __future__ import annotations

import json
from pathlib import Path

MISSING = 100_000.0

LABELS = {
    "nn": "Nearest neighbour",
    "nn+2opt": "Nearest neighbour + 2-opt",
    "ga": "GA, single population",
    "ga+2opt": "GA + 2-opt polish",
    "ga-islands1": "GA, MPI islands x1",
    "ga-islands2": "GA, MPI islands x2",
    "ga-islands4": "GA, MPI islands x4",
    "ga-islands4+2opt": "GA, MPI islands x4 + 2-opt polish",
}


def load(results_dir: Path, name: str) -> dict | None:
    p = Path(results_dir) / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f1(x: float) -> str:
    return f"{x:,.1f}"


def ms(mu: float, sd: float, fmt=f1) -> str:
    return f"{fmt(mu)} +/- {fmt(sd)}"


def secs(x: float) -> str:
    return f"{x:.2f}"


def table_map(res: dict) -> str:
    lines = [
        "| Method | Tour length, mean +/- sd | Best of 5 | Gap to best known | "
        "Time (s), mean +/- sd | Generations to converge |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, s in res["summary"].items():
        gens = (ms(s["gens_to_converge_mean"], s["gens_to_converge_sd"], lambda v: f"{v:,.0f}")
                if "gens_to_converge_mean" in s else "n/a")
        feas = "" if s["all_feasible"] else " (not all feasible)"
        lines.append(
            f"| {LABELS[key]}{feas} | {ms(s['mean'], s['sd'])} | {s['min']:,.0f} | "
            f"{s['gap_pct_mean']:.1f} % | {ms(s['time_mean'], s['time_sd'], secs)} | {gens} |")
    return "\n".join(lines)


def table_islands(res: dict) -> str:
    lines = [
        "| Islands (MPI processes) | Tour length, mean +/- sd | MPI time (s) | "
        "Same islands run serially (s) | Speed-up | Same result as serial? |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for k in res["islands"]:
        s = res["summary"][f"ga-islands{k}"]
        if "serial_time_mean" in s:
            serial = ms(s["serial_time_mean"], s["serial_time_sd"], secs)
            speed = f"{s['speedup_mean']:.2f}x"
            same = "yes, 5 of 5 seeds" if s["all_match_serial"] else "NO"
        else:
            serial, speed, same = "n/a", "1.00x (reference)", "n/a"
        lines.append(f"| {k} | {ms(s['mean'], s['sd'])} | "
                     f"{ms(s['time_mean'], s['time_sd'], secs)} | {serial} | {speed} | {same} |")
    return "\n".join(lines)


def table_anchor(res: dict) -> str:
    lines = [
        "| Method | Found the exact optimum | Hit rate per seed, mean +/- sd | "
        "Tours using a missing edge | Mean gap to optimum, feasible tours only |",
        "|---|---:|---:|---:|---:|",
    ]
    runs = res["runs"]
    for key in ("nn", "nn+2opt", "ga"):
        s = res["summary"][key]
        bad = sum(r[key] >= MISSING for r in runs)
        gaps = [100.0 * (r[key] - r["optimum"]) / r["optimum"] for r in runs
                if r[key] < MISSING and r["optimum_feasible"]]
        gap = f"{sum(gaps) / len(gaps):.1f} %" if gaps else "n/a"
        lines.append(f"| {LABELS[key]} | {s['hits']} of {s['instances']} | "
                     f"{s['hit_rate_mean']:.0f} % +/- {s['hit_rate_sd']:.0f} % | "
                     f"{bad} of {len(runs)} | {gap} |")
    return "\n".join(lines)


def table_fitness(res: dict) -> str:
    lines = [
        "| Population | Python loop, 1 process (ms) | Python loop, 4-process pool (ms) | "
        "NumPy, 1 process (ms) | NumPy, 4-process pool (ms) | Whole GA generation (ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(s, k):
        return ms(s[k]["mean"] * 1000, s[k]["sd"] * 1000)

    for size, s in res["summary"].items():
        lines.append(f"| {int(size):,} | {cell(s, 'python_seq')} | {cell(s, 'python_pool')} | "
                     f"{cell(s, 'numpy_seq')} | {cell(s, 'numpy_pool')} | "
                     f"{cell(s, 'ga_step_numpy')} |")
    return "\n".join(lines)


def table_tuning(res: dict) -> str:
    lines = ["| Tournament size | Mutation rate | Cost, mean +/- sd (3 tuning seeds) |",
             "|---:|---:|---:|"]
    for r in res["grid"]:
        lines.append(
            f"| {r['tournament_size']} | {r['mutation_rate']} | {ms(r['mean'], r['sd'])} |")
    return "\n".join(lines)


def render_all(results_dir: Path) -> dict[str, str]:
    out = {}
    for name, fn, key in (
        ("map32", table_map, "map32"),
        ("map100", table_map, "map100"),
        ("islands32", table_islands, "map32"),
        ("islands100", table_islands, "map100"),
        ("anchor", table_anchor, "anchor_8node"),
        ("fitness", table_fitness, "parallel_fitness"),
        ("tuning", table_tuning, "tuning"),
    ):
        res = load(results_dir, key)
        if res is not None:
            out[name] = fn(res)
    return out


def write_tables(results_dir: Path) -> Path:
    blocks = render_all(results_dir)
    text = "# Result tables (generated by scripts/results_tables.py, do not edit)\n\n"
    for name, table in blocks.items():
        text += f"<!-- BEGIN:{name} -->\n{table}\n<!-- END:{name} -->\n\n"
    path = Path(results_dir) / "tables.md"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    print(write_tables(Path(sys.argv[1] if len(sys.argv) > 1 else "results")))
