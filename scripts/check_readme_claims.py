"""Check that the README (and optionally the report and skills JSON) only state
numbers that come from results/, and that results/ matches the current code.

Checks:
1. Every ``<!-- BEGIN:name -->`` ... ``<!-- END:name -->`` table in README.md
   equals the table rendered from results/*.json.
2. Key inline claims (``claims()``) appear with exactly the computed value.
3. Every number in the README text (and in --report / --skills files) is found
   in results/*.json, in the rendered tables, in the data files, or in the short
   ``CITED`` list below (each with its source). Exempt: code spans and code
   blocks, URLs, hashes, dates, versions, identifiers like route_ga,
   file:line references, "N/10" scores, whole numbers 0 to 10, and
   four-digit years from 2020 to 2030.
4. Every results/*.json was produced from the last commit that changed src/,
   with no uncommitted changes in src/. Skipped, with the reason printed, when
   there is no full git history (a ZIP download or a shallow clone).

Usage:
    python scripts/check_readme_claims.py
    python scripts/check_readme_claims.py --report notes.md --skills skills.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from results_tables import load, render_all  # noqa: E402

BLOCK = re.compile(r"<!-- BEGIN:(\w+) -->\n(.*?)\n<!-- END:\1 -->", re.DOTALL)

# Numbers that do not come from results/, each with its source.
CITED: dict[str, str] = {}

NUMBER = re.compile(r"(?<![\w.:/#@$\\-])-?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?=x\b|(?![\w.]\w))")
EXEMPT_SPANS = [
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"https?://\S+"),
    re.compile(r"\b[0-9a-f]{7,64}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(T[\d:+.Z-]+)?\b"),
    re.compile(r"\b\d+\.\d+\.\d+(\.\d+)?\b"),
    re.compile(r"\b\d+(\.\d+)?/10\b"),
    re.compile(r"\b[\w./-]+\.(py|json|md|csv|yml|txt|png|toml):\d+(-\d+)?(,\d+)*"),
    re.compile(r"<!--.*?-->", re.DOTALL),
]


def fmt0(x: float) -> str:
    return f"{x:,.0f}"


def _variants(v: float) -> set[str]:
    out = set()
    if isinstance(v, bool) or not math.isfinite(v):
        return out
    for x in (v, abs(v)):
        for d in (0, 1, 2):
            out.add(f"{x:.{d}f}")
            out.add(f"{x:,.{d}f}")
    return out


def _walk(obj, acc: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _walk(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, acc)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        acc |= _variants(float(obj))
    elif isinstance(obj, str) and re.fullmatch(r"\d+\.\d+(\.\d+)?", obj):
        parts = obj.split(".")
        acc.add(obj)
        acc.add(".".join(parts[:2]))


def derived(results: Path) -> dict[str, float]:
    """Numbers computed from results (not stored verbatim), with their meaning."""
    out: dict[str, float] = {}
    anc = load(results, "anchor_8node")
    if anc:
        cfg = anc["ga_config"]
        out["anchor GA evaluations (population x generations)"] = (
            cfg["population_size"] * cfg["generations"])
        out["anchor tours for brute force ((n-1)!)"] = math.factorial(anc["n_nodes"] - 1)
        out["anchor instances missed by GA"] = anc["summary"]["ga"]["instances"] - anc[
            "summary"]["ga"]["hits"]
        out["anchor missing-edge share %"] = 100 * anc["missing_share"]
    fit = load(results, "parallel_fitness")
    if fit:
        big = fit["summary"][max(fit["summary"], key=int)]
        out["pool/numpy ratio"] = round(big["python_pool"]["median"] / big["numpy_seq"]["median"])
        out["numpy share of a generation %"] = (
            100 * big["numpy_seq"]["median"] / big["ga_step_numpy"]["median"])
    for n in (32, 100):
        res = load(results, f"map{n}")
        if res:
            s = res["summary"]
            for k in ("ga-islands2", "ga-islands4"):
                if "speedup_median" in s.get(k, {}):
                    out[f"map{n} {k} speed-up"] = s[k]["speedup_median"]
    q = load(results, "quality")
    if q:
        t = q["tests"]
        out["test cases that must pass or xfail"] = t["passed"] + t["xfailed"]
    return out


def allowed_numbers(results: Path) -> set[str]:
    acc: set[str] = set()
    for p in sorted(Path(results).glob("*.json")):
        _walk(json.loads(p.read_text(encoding="utf-8")), acc)
    for table in render_all(results).values():
        acc |= {m.group(0).lstrip("-") for m in NUMBER.finditer(table)}
    for v in derived(results).values():
        acc |= _variants(float(v))
    from route_ga.data import DATA_DIR, MAP_FILES, MISSING_EDGE, load_map, missing_mask

    acc |= _variants(MISSING_EDGE)
    for n, f in MAP_FILES.items():
        d = load_map(n)
        real = d[~missing_mask(d) & (d > 0)]
        acc |= _variants(n) | _variants(int(missing_mask(d).sum()))
        acc |= _variants(real.min()) | _variants(real.max())
        acc |= _variants((DATA_DIR / f).stat().st_size)
    acc |= set(CITED)
    return acc


def claims(results: Path) -> list[tuple[str, str]]:
    """(description, exact text that must appear in the README)."""
    out: list[tuple[str, str]] = []
    from results_tables import best_found

    for n in (32, 100):
        b = best_found(results, n)
        if b:
            out.append((f"map{n} best found", fmt0(b[0])))
        res = load(results, f"map{n}")
        if not res:
            continue
        s = res["summary"]
        out += [
            (f"map{n} NN+2opt length", fmt0(s["nn+2opt"]["mean"])),
            (f"map{n} GA mean", fmt0(s["ga"]["mean"])),
            (f"map{n} GA+2opt mean", fmt0(s["ga+2opt"]["mean"])),
            (f"map{n} islands4+2opt mean", fmt0(s["ga-islands4+2opt"]["mean"])),
            (f"map{n} 4-island speed-up", f"{s['ga-islands4']['speedup_median']:.2f} times faster"),
        ]
    anc = load(results, "anchor_8node")
    if anc:
        for m in ("nn", "nn+2opt", "ga"):
            a = anc["summary"][m]
            out.append((f"anchor {m} hits", f"{a['hits']} of {a['instances']}"))
    return out


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def src_commit(root: Path = ROOT) -> tuple[str | None, bool | None]:
    commit = _git(root, "log", "-1", "--format=%H", "--", "src")
    status = _git(root, "status", "--porcelain", "--", "src")
    return commit or None, None if status is None else bool(status)


def provenance_skip_reason(root: Path = ROOT) -> str | None:
    """Why the provenance check cannot run in ``root``, or None if it can."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != Path(root).resolve():
        return ("no git history here (for example a ZIP download); "
                "use a full git clone to check it")
    if _git(root, "rev-parse", "--is-shallow-repository") == "true":
        return ("shallow clone, so the last commit that changed src/ may be missing; "
                "clone without --depth to check it")
    return None


def check_provenance(results: Path) -> list[str]:
    errors = []
    commit, dirty = src_commit()
    if commit is None:
        return ["git not available: cannot check results provenance"]
    if dirty:
        errors.append("src/ has uncommitted changes")
    for p in sorted(Path(results).glob("*.json")):
        code = json.loads(p.read_text(encoding="utf-8")).get("environment", {}).get("code", {})
        if code.get("src_commit") != commit:
            errors.append(f"{p.name}: made from src commit {str(code.get('src_commit'))[:8]}, "
                          f"current src commit is {commit[:8]} (re-run it)")
        elif code.get("src_dirty"):
            errors.append(f"{p.name}: made with uncommitted changes in src/")
    return errors


def strip_exempt(text: str) -> str:
    for pat in EXEMPT_SPANS:
        text = pat.sub(" ", text)
    return text


def check_numbers(text: str, allowed: set[str], label: str) -> list[str]:
    errors = []
    for lineno, line in enumerate(strip_exempt(text).splitlines(), 1):
        for m in NUMBER.finditer(line):
            tok = m.group(0).lstrip("-")
            plain = tok.replace(",", "")
            is_year = "," not in tok and "." not in plain and 2020 <= int(plain) <= 2030
            if "." not in plain and int(plain) <= 10 or is_year:
                continue
            if tok in allowed or plain in allowed:
                continue
            errors.append(f"{label} line {lineno}: number {tok} not found in results/: "
                          f"{line.strip()[:90]}")
    return errors


def check(readme: Path, results: Path, extra: list[Path] | None = None,
          provenance: bool = True) -> list[str]:
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
    allowed = allowed_numbers(results)
    errors += check_numbers(text, allowed, "README")
    for path in extra or []:
        errors += check_numbers(path.read_text(encoding="utf-8"), allowed, path.name)
    if provenance:
        reason = provenance_skip_reason()
        if reason:
            print(f"SKIP results provenance check: {reason}")
        else:
            errors += check_provenance(results)
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--report", type=Path, action="append", default=[])
    ap.add_argument("--skills", type=Path, action="append", default=[])
    ap.add_argument("--no-provenance", action="store_true", help="skip the commit check")
    args = ap.parse_args(argv)
    extra = args.report + args.skills
    errors = check(args.readme, args.results, extra, provenance=not args.no_provenance)
    for e in errors:
        print("FAIL", e)
    prov = "results provenance"
    if args.no_provenance or provenance_skip_reason():
        prov += " (skipped)"
    print(f"checked {len(render_all(args.results))} tables, {len(claims(args.results))} inline "
          f"claims, every number in {1 + len(extra)} file(s) and {prov}: "
          f"{len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
