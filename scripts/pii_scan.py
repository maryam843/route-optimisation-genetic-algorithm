"""Scan every blob and every commit in the git history for personal data and secrets.

Usage: python scripts/pii_scan.py [repo_path]
Exit code 1 if anything is found. Also runs in CI (with full history).

Only generic patterns live here (e-mail, phone, keys, tokens, passwords, user
paths). Personal patterns are kept out of the repo: put one regex per line in an
untracked ``.pii_local.txt`` at the repo root, and/or in the environment
variable PII_EXTRA_PATTERNS (one regex per line). Lines that are empty or start
with ``#`` are ignored. Text matched by these extra patterns is never printed.
The scan works with neither set.

This file is skipped by the blob scan because it contains the patterns
themselves; its content is reviewed by hand.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SELF = "scripts/pii_scan.py"
ALLOWED_EMAILS = {"185722545+maryam843@users.noreply.github.com"}

PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "qatar_phone": re.compile(r"(?:\+|00)974[\s-]?\d{4}[\s-]?\d{4}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b|github_pat_[A-Za-z0-9_]{20,}"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b|\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "password_assignment": re.compile(
        r"(?i)\b(pass(word)?|secret|token)\s*[:=]\s*['\"][^'\"]{4,}['\"]"),
    "azure_conn": re.compile(r"AccountKey=[A-Za-z0-9+/=]{20,}"),
    "user_path": re.compile(
        r"[A-Za-z]:\\{1,2}Users\\{1,2}[^\\\s\"']+|/(?:home|Users)/[A-Za-z][\w.-]*/"),
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True, encoding="utf-8", errors="replace").stdout


LOCAL_FILE = ".pii_local.txt"
ENV_VAR = "PII_EXTRA_PATTERNS"


def _pattern_lines(text: str) -> list[str]:
    lines = (ln.strip() for ln in text.splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def extra_patterns(repo: Path | None = None) -> dict[str, re.Pattern]:
    """Personal patterns from the untracked local file and the environment."""
    lines = _pattern_lines(os.environ.get(ENV_VAR, ""))
    local = (repo or Path(__file__).resolve().parents[1]) / LOCAL_FILE
    if local.is_file():
        lines += _pattern_lines(local.read_text(encoding="utf-8"))
    return {f"extra_{i}": re.compile(p, re.IGNORECASE) for i, p in enumerate(lines, 1)}


def check_text(text: str, where: str, extra: dict[str, re.Pattern] | None = None) -> list[str]:
    hits = []
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            if name == "email" and m.group(0) in ALLOWED_EMAILS:
                continue
            hits.append(f"{where}: {name}: {m.group(0)[:60]!r}")
    for name, pat in (extra or {}).items():
        hits += [f"{where}: {name}: [match hidden]" for _ in pat.finditer(text)]
    return hits


def scan(repo: Path) -> tuple[list[str], int, int]:
    hits: list[str] = []
    extra = extra_patterns(repo)
    commits = git(repo, "rev-list", "--all").split()
    for c in commits:
        meta = git(repo, "show", "-s", "--format=%an%n%ae%n%cn%n%ce%n%B", c)
        hits += check_text(meta, f"commit {c[:8]}", extra)
    seen: set[str] = set()
    blobs = 0
    for line in git(repo, "rev-list", "--all", "--objects").splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2 or parts[0] in seen:
            continue
        sha, path = parts
        if git(repo, "cat-file", "-t", sha).strip() != "blob":
            continue
        seen.add(sha)
        if path == SELF:
            continue
        blobs += 1
        data = subprocess.run(["git", "-C", str(repo), "cat-file", "-p", sha],
                              capture_output=True, check=True).stdout
        if b"\0" in data[:8000]:  # binary (PNG): scan printable metadata only
            text = data.decode("latin-1")
            text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
        else:
            text = data.decode("utf-8", errors="replace")
        hits += check_text(text, f"{path} ({sha[:8]})", extra)
    return hits, len(commits), blobs


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    repo = Path(argv[0]) if argv else Path(__file__).resolve().parents[1]
    hits, n_commits, n_blobs = scan(repo)
    for h in hits:
        print(h)
    print(f"scanned {n_commits} commits and {n_blobs} blobs: {len(hits)} hits")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
