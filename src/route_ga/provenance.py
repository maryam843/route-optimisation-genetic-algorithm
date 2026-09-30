"""Machine, library and code-version details stored next to every result."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def cpu_model() -> str | None:
    """CPU model name (Linux: /proc/cpuinfo; elsewhere: platform.processor())."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                             text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def code_version() -> dict:
    """Last commit that touched src/ and whether src/ or scripts/ have local changes."""
    src_commit = _git("log", "-1", "--format=%H", "--", "src")
    head = _git("rev-parse", "HEAD")
    src_status = _git("status", "--porcelain", "--", "src")
    scripts_status = _git("status", "--porcelain", "--", "scripts")
    return {
        "src_commit": src_commit or None,
        "src_dirty": None if src_status is None else bool(src_status),
        "head_commit": head or None,
        "scripts_dirty": None if scripts_status is None else bool(scripts_status),
    }


def environment_info(machine_note: str | None = None) -> dict:
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "nproc": os.cpu_count(),
        "ram_bytes": ram_bytes(),
        "machine_note": machine_note or os.environ.get("ROUTE_GA_MACHINE_NOTE") or None,
        "packages": {p: _version(p) for p in ("numpy", "matplotlib", "mpi4py", "ortools",
                                              "route-ga")},
        "code": code_version(),
    }
