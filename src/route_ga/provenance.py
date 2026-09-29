"""Machine and library details stored next to every result."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import os
import platform
import sys


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def environment_info() -> dict:
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "packages": {p: _version(p) for p in ("numpy", "matplotlib", "mpi4py", "route-ga")},
    }
