"""Draw the README figures from the JSON files in results/.

Usage: python scripts/make_figures.py [--results results] [--out figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from route_ga.data import load_map, missing_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from results_tables import LABELS  # noqa: E402

# Fixed categorical order (reference palette, light mode); colour follows the method.
SERIES = {
    "ga": "#2a78d6",
    "ga-islands1": "#eb6834",
    "ga-islands2": "#1baf7a",
    "ga-islands4": "#4a3aa7",
}
BAR = "#2a78d6"
MUTED = "#52514e"
INK = "#0b0b0b"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False,
})


def load(results: Path, name: str) -> dict | None:
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def convergence(res: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for method, color in SERIES.items():
        hs = np.array([r["history"] for r in res["runs"] if r["method"] == method], dtype=float)
        if hs.size == 0:
            continue
        mu, sd = hs.mean(axis=0), hs.std(axis=0, ddof=1) if len(hs) > 1 else 0 * hs[0]
        g = np.arange(len(mu))
        ax.fill_between(g, np.maximum(mu - sd, 1), mu + sd, color=color, alpha=0.15, lw=0)
        ax.plot(g, mu, color=color, lw=2, label=LABELS[method])
        ax.annotate(f"{mu[-1]:,.0f}", (g[-1], mu[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    nn2 = res["summary"]["nn+2opt"]["mean"]
    ax.axhline(nn2, color=MUTED, lw=1, ls="--")
    ax.annotate(f"nearest neighbour + 2-opt: {nn2:,.0f}", (0, nn2), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("best tour cost (log scale)")
    n = len(res["seeds"])
    ax.set_title(f"{res['map']}-node map: best cost per generation, mean +/- sd over {n} seeds\n"
                 "cost = length + 100,000 per missing edge; early values are infeasible tours",
                 loc="left")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / f"convergence_{res['map']}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def methods(results: list[dict], out: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.2))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results, strict=True):
        keys = list(res["summary"])[::-1]
        mu = [res["summary"][k]["mean"] for k in keys]
        sd = [res["summary"][k]["sd"] for k in keys]
        y = np.arange(len(keys))
        ax.barh(y, mu, xerr=sd, color=BAR, height=0.6, error_kw={"ecolor": MUTED, "lw": 1})
        ax.set_axisbelow(True)
        for yi, m, e in zip(y, mu, sd, strict=True):
            ax.annotate(f"{m:,.0f}", (m + e, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [LABELS[k] for k in keys], fontsize=8)
        ax.set_xlabel("tour length (mean +/- sd, 5 seeds)")
        ax.set_title(f"{res['map']}-node map", loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(mu) * 1.25)
    feasible = all(s["all_feasible"] for r in results for s in r["summary"].values())
    note = "All reported tours are feasible (no missing edges)." if feasible else (
        "Some tours use missing edges; see results/*.json.")
    fig.suptitle(f"Shorter is better. {note}",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    path = out / "methods.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def shortest_paths(d: np.ndarray) -> np.ndarray:
    """Floyd-Warshall; missing edges are removed so the layout uses real roads."""
    sp = np.where(missing_mask(d), np.inf, d)
    for k in range(len(sp)):
        sp = np.minimum(sp, sp[:, [k]] + sp[[k], :])
    return sp


def mds_layout(d: np.ndarray) -> np.ndarray:
    """Classical MDS on shortest-path distances. For display only."""
    sp = shortest_paths(d)
    n = len(sp)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (sp**2) @ j
    vals, vecs = np.linalg.eigh(b)
    idx = np.argsort(vals)[::-1][:2]
    return vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))


def best_route(res: dict, out: Path) -> Path:
    d = load_map(res["map"])
    xy = mds_layout(d)
    tour = res["best_known_tour"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[tour, 0], xy[tour, 1], color=BAR, lw=1.5, zorder=1)
    ax.scatter(xy[1:, 0], xy[1:, 1], s=36, color="#fcfcfb", edgecolor=INK, lw=1, zorder=2)
    ax.scatter(xy[0, 0], xy[0, 1], s=110, marker="s", color="#eb6834", edgecolor=INK, zorder=3)
    ax.annotate("depot 0", xy[0], xytext=(8, 8), textcoords="offset points", fontsize=9)
    if res["map"] <= 32:
        for i, (x, yv) in enumerate(xy):
            if i:
                ax.annotate(str(i), (x, yv), xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    label = LABELS[res["best_known_method"]]
    ax.set_title(f"{res['map']}-node map: shortest tour found in this study, length "
                 f"{res['best_known_cost']:,.0f}\n({label}). Node positions come from MDS "
                 "on road distances\nand are for display only; they are not real coordinates.",
                 loc="left", fontsize=9)
    fig.tight_layout()
    path = out / f"best_route_{res['map']}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def island_scaling(results: list[dict], out: Path) -> Path:
    fig, (ax_t, ax_q) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {32: "#2a78d6", 100: "#eb6834"}
    for res in results:
        ks = res["islands"]
        s = [res["summary"][f"ga-islands{k}"] for k in ks]
        c = colors.get(res["map"], BAR)
        mpi = [x["time_mean"] for x in s]
        serial = [x.get("serial_time_mean", x["time_mean"]) for x in s]
        ax_t.plot(ks, mpi, color=c, lw=2, marker="o", ms=6, label=f"{res['map']} nodes, MPI")
        ax_t.plot(ks, serial, color=c, lw=2, ls="--", marker="o", ms=6, mfc="#fcfcfb",
                  label=f"{res['map']} nodes, same islands in one process")
        ax_q.errorbar(ks, [x["mean"] for x in s], yerr=[x["sd"] for x in s], color=c, lw=2,
                      marker="o", ms=6, capsize=3, label=f"{res['map']} nodes")
    ax_t.set_xticks([1, 2, 4])
    ax_q.set_xticks([1, 2, 4])
    ax_t.set_xlabel("islands (one MPI process each, one machine)")
    ax_q.set_xlabel("islands (one MPI process each, one machine)")
    ax_t.set_ylabel("wall time (s), mean of 5 seeds")
    ax_q.set_ylabel("tour length, mean +/- sd")
    ax_t.set_title("Time: each island has the full population", loc="left")
    ax_q.set_title("Quality (shorter is better)", loc="left")
    ax_t.legend(fontsize=8)
    ax_q.legend(fontsize=8)
    ax_t.set_ylim(bottom=0)
    fig.tight_layout()
    path = out / "island_scaling.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    maps = [r for r in (load(args.results, "map32"), load(args.results, "map100")) if r]
    if not maps:
        print("no map results found; run scripts/run_experiments.py first")
        return 1
    made = [convergence(r, args.out) for r in maps]
    made += [methods(maps, args.out), island_scaling(maps, args.out)]
    made += [best_route(r, args.out) for r in maps]
    for p in made:
        print(f"saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
