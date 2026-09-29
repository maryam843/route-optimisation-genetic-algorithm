"""Check that the README only states numbers that are in results/.

1. Every ``<!-- BEGIN:name -->`` ... ``<!-- END:name -->`` table in README.md
   must equal the table rendered from results/*.json.
2. Every inline number listed in ``claims()`` must appear in README.md with
   exactly the value computed from results/*.json.
3. Forbidden wording: "optimal", multi-vehicle or multi-machine claims
   (allowed only in a sentence that says it was not done), em dashes, emojis.

Usage: python scripts/check_readme_claims.py [--readme README.md] [--results results]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from results_tables import load, render_all  # noqa: E402

BLOCK = re.compile(r"<!-- BEGIN:(\w+) -->\n(.*?)\n<!-- END:\1 -->", re.DOTALL)
NEGATION = re.compile(r"\b(not|no|never|without|only one|single|does not|did not)\b", re.I)
FORBIDDEN = {
    "optimal": re.compile(r"\boptimal(ly)?\b", re.I),
    "multi-vehicle": re.compile(r"multi[- ]vehicle|multiple vehicles|several vans|fleet", re.I),
    "multi-machine": re.compile(r"multi[- ]machine|several machines|multiple machines|"
                                r"across machines|cluster of", re.I),
}
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


def fmt0(x: float) -> str:
    return f"{x:,.0f}"


def claims(results: Path) -> list[tuple[str, str]]:
    """(description, exact text that must appear in the README)."""
    out: list[tuple[str, str]] = []
    leg = load(results, "legacy_2025")
    if leg:
        out += [
            ("2025 chosen 'best' at generation 0", fmt0(leg["chosen_best_first"])),
            ("2025 chosen 'best' at the end", fmt0(leg["chosen_best_last"])),
            ("2025 shortest route at generation 0", fmt0(leg["shortest_in_population_first"])),
            ("2025 shortest route at the end", fmt0(leg["shortest_in_population_last"])),
        ]
    for n in (32, 100):
        res = load(results, f"map{n}")
        if not res:
            continue
        s = res["summary"]
        out += [
            (f"map{n} best known length", f"{fmt0(res['best_known_cost'])}"),
            (f"map{n} NN+2opt length", fmt0(s["nn+2opt"]["mean"])),
            (f"map{n} GA mean", fmt0(s["ga"]["mean"])),
            (f"map{n} GA+2opt mean", fmt0(s["ga+2opt"]["mean"])),
            (f"map{n} islands4+2opt mean", fmt0(s["ga-islands4+2opt"]["mean"])),
            (f"map{n} 4-island speed-up", f"{s['ga-islands4']['speedup_mean']:.2f}x"),
        ]
    anc = load(results, "anchor_8node")
    if anc:
        for m in ("nn", "nn+2opt", "ga"):
            a = anc["summary"][m]
            out.append((f"anchor {m} hits", f"{a['hits']} of {a['instances']}"))
    fit = load(results, "parallel_fitness")
    if fit:
        big = fit["summary"][max(fit["summary"], key=int)]
        out.append(("pool vs numpy at largest population",
                    f"{big['python_pool']['mean'] / big['numpy_seq']['mean']:.0f} times"))
    return out


def check(readme: Path, results: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    errors: list[str] = []
    rendered = render_all(results)
    found = dict(BLOCK.findall(text))
    for name, table in rendered.items():
        if name not in found:
            errors.append(f"table {name}: missing from README")
        elif found[name].strip() != table.strip():
            errors.append(f"table {name}: README differs from results (re-copy results/tables.md)")
    for name in found:
        if name not in rendered:
            errors.append(f"table {name}: in README but no results file backs it")
    for desc, value in claims(results):
        if value not in text:
            errors.append(f"claim '{desc}': expected '{value}' in README")
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, pat in FORBIDDEN.items():
            if pat.search(line) and not NEGATION.search(line):
                errors.append(f"line {lineno}: forbidden claim ({label}): {line.strip()[:80]}")
        if "—" in line:
            errors.append(f"line {lineno}: em dash")
        if EMOJI.search(line):
            errors.append(f"line {lineno}: emoji")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    errors = check(args.readme, args.results)
    for e in errors:
        print("FAIL", e)
    n_claims = len(claims(args.results))
    n_tables = len(render_all(args.results))
    print(f"checked {n_tables} tables and {n_claims} inline claims: {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
