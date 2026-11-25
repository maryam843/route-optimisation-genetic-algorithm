# Route optimisation with a genetic algorithm

[![CI](https://github.com/maryam843/route-optimisation-genetic-algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/maryam843/route-optimisation-genetic-algorithm/actions/workflows/ci.yml)

Plans the shortest round trip for one delivery van through 32 or 100 stops, where some pairs of stops have no road, and compares a genetic algorithm (GA), its MPI island version, simple baselines and the Google OR-Tools reference solver.

## Key results

- **The GA does not beat the baselines.** On the 100-stop map the GA averaged 1,061 over 5 seeds (553 after a 2-opt polish, 528 for 4 MPI islands + 2-opt). Nearest neighbour + 2-opt found 434 in well under a second, and the OR-Tools reference found 338 (best of 5 repeats with a 30 s limit, not proven optimal).
- **Same order on the 32-stop map:** GA 520, GA + 2-opt polish 398, nearest neighbour + 2-opt 364, OR-Tools reference 308.
- **The MPI island model is correct and faster:** identical routes to the serial run on 5 of 5 seeds, and 3.45 times faster with 4 MPI processes on one 4-core machine (100 stops; 3.55 times faster on 32 stops).
- **94 tests pass** with 92.4 % coverage, and CI checks every table and number in this README against `results/`.

![Best tour length per generation on the 100-stop map for the single GA and 1, 2 and 4 MPI islands, levelling off far above the nearest neighbour + 2-opt result of 434](figures/convergence_100.png)

**Tools:** Python, NumPy, multiprocessing, mpi4py, Open MPI, Google OR-Tools, Matplotlib, pytest, Hypothesis, Ruff, GitHub Actions

## Data

Two symmetric distance tables, supplied with the original exercise and credited in [`data/README.md`](data/README.md):

| File | Stops | Size | Missing roads (cells equal to 100000) |
|---|---:|---:|---:|
| `data/city_distances.csv` | 32 | 5,888 B | 192 |
| `data/city_distances_extended.csv` | 100 | 57,084 B | 1,900 |

- Stop 0 is the depot. Real distances are whole numbers from 1 to 99; 100000 means there is no road.
- The real-world origin is unknown and no licence is stated. The numbers have no units and are not coordinates, so tour lengths are plain distance units.
- Both files are small, so they are kept in the repo byte for byte and also ship inside the package as `route_ga.maps`. Nothing needs to be downloaded.

## Approach

1. **Problem:** the van leaves the depot, visits every other stop once and returns (a travelling salesman problem). A tour costs its length plus 100,000 for each missing road used, so a tour with fewer missing roads always wins.
2. **Baselines:** nearest neighbour, nearest neighbour + 2-opt, and brute force on 100 random 8-stop maps where the exact optimum is known, to check every method against a true answer.
3. **Reference:** OR-Tools routing (guided local search) with a 30 s time limit and 5 repeats per map. It is the strongest result here but not a proven optimum.
4. **GA:** 200 random tours, tournament selection of 3, order crossover, swap mutation (rate 0.3), the 2 best tours kept each generation, and a restart after 150 generations without improvement; 1,000 generations on 32 stops and 2,000 on 100. An optional 2-opt polish runs on the final tour.
5. **Fixes to the starting code:** the order crossover and swap mutation follow versions supplied with the original exercise. The earlier version kept the longest routes, crashed with a PicklingError in its process pool, never penalised missing roads (it checked for 10000, the data uses 100000) and left the tour open. Each fix has a regression test in `tests/test_regressions.py`.
6. **MPI islands:** each MPI process runs its own population. Every 25 generations it sends its 2 best tours to the next island in a ring, replacing that island's 2 worst. A serial twin runs the same islands in one process, and with the same seed both must give the same tour, which is how the MPI code is tested.
7. **Parallel scoring check:** a 4-process pool versus vectorised NumPy for scoring a population. NumPy in one process was faster at every size, and scoring was 5.9 ms of a 394.6 ms generation at 10,000 tours, so the parallel effort went into whole islands instead.
8. **Reproducibility:** seeds 0 to 4 for every result, separate seeds (100, 101, 102) for tuning, and each `results/*.json` stores its settings, machine, package versions and the git commit of `src/`.

## Results

100-stop map, 5 seeds per method. Times are medians [IQR] on a shared 4-vCPU cloud VM (Intel Xeon @ 2.10GHz, Python 3.12.3, Open MPI 4.1.6) and vary between runs. "Best found" is the shortest tour any method found here, not a proven optimum.

<!-- BEGIN:map100 -->
| Method | Tour length, mean +/- sd | Best of 5 | Gap to best found | Gap to OR-Tools | Time (s), median [IQR] | Generations to converge |
|---|---:|---:|---:|---:|---:|---:|
| Nearest neighbour | 576.0 +/- 0.0 | 576 | 70.4 % | 70.4 % | 0.00 [0.00-0.00] | n/a |
| Nearest neighbour + 2-opt | 434.0 +/- 0.0 | 434 | 28.4 % | 28.4 % | 0.00 [0.00-0.00] | n/a |
| GA, single population | 1,061.2 +/- 56.0 | 999 | 214.0 % | 214.0 % | 14.75 [14.46-15.08] | 1,735 +/- 193 |
| GA + 2-opt polish | 552.6 +/- 33.8 | 512 | 63.5 % | 63.5 % | 14.76 [14.47-15.09] | n/a |
| GA, MPI islands x1 | 1,070.0 +/- 100.6 | 925 | 216.6 % | 216.6 % | 15.30 [15.25-15.48] | 1,605 +/- 342 |
| GA, MPI islands x2 | 1,005.6 +/- 51.7 | 924 | 197.5 % | 197.5 % | 16.86 [16.73-16.87] | 1,578 +/- 380 |
| GA, MPI islands x4 | 967.6 +/- 47.5 | 910 | 186.3 % | 186.3 % | 17.59 [17.39-18.05] | 1,843 +/- 83 |
| GA, MPI islands x4 + 2-opt polish | 527.6 +/- 24.4 | 498 | 56.1 % | 56.1 % | 17.60 [17.39-18.05] | n/a |
| OR-Tools reference (not proven optimal) | best of 5 repeats: 338 | 338 | 0.0 % | 0.0 % | 30 (time limit) | n/a |
<!-- END:map100 -->

The tables are generated by `scripts/results_tables.py` from `results/*.json` (also in [`results/tables.md`](results/tables.md)), and `scripts/check_readme_claims.py` fails CI if any of them differs.

<details>
<summary>All other result tables (32-stop map, 8-stop check, MPI scaling, work per run, parallel scoring, short runs, tuning)</summary>

Shortest tour found in this study:

<!-- BEGIN:bestfound -->
| Map | Shortest tour found in this study | Found by | Results file |
|---:|---:|---|---|
| 32 nodes | 308 | OR-Tools reference, best of 5 repeats with a 30 s time limit (not proven optimal) | `results/reference_ortools.json` |
| 100 nodes | 338 | OR-Tools reference, best of 5 repeats with a 30 s time limit (not proven optimal) | `results/reference_ortools.json` |
<!-- END:bestfound -->

32-stop map:

<!-- BEGIN:map32 -->
| Method | Tour length, mean +/- sd | Best of 5 | Gap to best found | Gap to OR-Tools | Time (s), median [IQR] | Generations to converge |
|---|---:|---:|---:|---:|---:|---:|
| Nearest neighbour | 445.0 +/- 0.0 | 445 | 44.5 % | 44.5 % | 0.00 [0.00-0.00] | n/a |
| Nearest neighbour + 2-opt | 364.0 +/- 0.0 | 364 | 18.2 % | 18.2 % | 0.00 [0.00-0.00] | n/a |
| GA, single population | 519.6 +/- 29.6 | 493 | 68.7 % | 68.7 % | 6.99 [6.74-7.36] | 668 +/- 280 |
| GA + 2-opt polish | 397.6 +/- 24.1 | 370 | 29.1 % | 29.1 % | 6.99 [6.74-7.36] | n/a |
| GA, MPI islands x1 | 519.2 +/- 17.9 | 503 | 68.6 % | 68.6 % | 7.38 [7.11-7.49] | 578 +/- 245 |
| GA, MPI islands x2 | 454.8 +/- 40.4 | 390 | 47.7 % | 47.7 % | 8.00 [7.92-8.02] | 555 +/- 276 |
| GA, MPI islands x4 | 424.2 +/- 33.6 | 367 | 37.7 % | 37.7 % | 8.13 [8.09-8.39] | 562 +/- 227 |
| GA, MPI islands x4 + 2-opt polish | 383.4 +/- 22.5 | 355 | 24.5 % | 24.5 % | 8.13 [8.09-8.39] | n/a |
| OR-Tools reference (not proven optimal) | best of 5 repeats: 308 | 308 | 0.0 % | 0.0 % | 30 (time limit) | n/a |
<!-- END:map32 -->

8-stop maps with a known optimum (100 maps, brute force):

<!-- BEGIN:anchor -->
| Method | Found the exact optimum | Hit rate per seed, mean +/- sd | Tours using a missing edge | Mean gap to optimum, feasible tours only |
|---|---:|---:|---:|---:|
| Nearest neighbour | 12 of 100 | 12 % +/- 3 % | 38 of 100 | 15.1 % |
| Nearest neighbour + 2-opt | 68 of 100 | 68 % +/- 9 % | 0 of 100 | 2.8 % |
| GA, single population | 62 of 100 | 62 % +/- 14 % | 0 of 100 | 4.2 % |
<!-- END:anchor -->

MPI islands vs the same islands run one after another, 32 and 100 stops:

<!-- BEGIN:islands32 -->
| Islands (MPI processes) | Tour length, mean +/- sd | MPI time (s), median [IQR] | Same islands one after another (s), median [IQR] | Speed-up | Same result? |
|---:|---:|---:|---:|---:|---|
| 1 | 519.2 +/- 17.9 | 7.38 [7.11-7.49] | n/a | n/a (one island) | n/a |
| 2 | 454.8 +/- 40.4 | 8.00 [7.92-8.02] | 14.84 [13.99-15.10] | 1.85 times faster | yes, 5 of 5 seeds |
| 4 | 424.2 +/- 33.6 | 8.13 [8.09-8.39] | 28.84 [28.67-28.87] | 3.55 times faster | yes, 5 of 5 seeds |
<!-- END:islands32 -->

<!-- BEGIN:islands100 -->
| Islands (MPI processes) | Tour length, mean +/- sd | MPI time (s), median [IQR] | Same islands one after another (s), median [IQR] | Speed-up | Same result? |
|---:|---:|---:|---:|---:|---|
| 1 | 1,070.0 +/- 100.6 | 15.30 [15.25-15.48] | n/a | n/a (one island) | n/a |
| 2 | 1,005.6 +/- 51.7 | 16.86 [16.73-16.87] | 30.50 [30.34-30.67] | 1.81 times faster | yes, 5 of 5 seeds |
| 4 | 967.6 +/- 47.5 | 17.59 [17.39-18.05] | 60.69 [60.12-61.25] | 3.45 times faster | yes, 5 of 5 seeds |
<!-- END:islands100 -->

Fitness evaluations per run (N islands do about N times the work of one population):

<!-- BEGIN:evaluations -->
| Map | Method | Fitness evaluations per run, mean | Times the single GA | Tour length, mean |
|---:|---|---:|---:|---:|
| 32 | GA, single population | 201,000 | 1.00 | 519.6 |
| 32 | GA, MPI islands x1 | 200,920 | 1.00 | 519.2 |
| 32 | GA, MPI islands x2 | 402,320 | 2.00 | 454.8 |
| 32 | GA, MPI islands x4 | 803,880 | 4.00 | 424.2 |
| 100 | GA, single population | 401,400 | 1.00 | 1,061.2 |
| 100 | GA, MPI islands x1 | 401,440 | 1.00 | 1,070.0 |
| 100 | GA, MPI islands x2 | 802,960 | 2.00 | 1,005.6 |
| 100 | GA, MPI islands x4 | 1,603,960 | 4.00 | 967.6 |
<!-- END:evaluations -->

Scoring one population on the 100-stop map, ms, median [IQR]:

<!-- BEGIN:fitness -->
| Population | Python loop, 1 process (ms) | Python loop, 4-process pool (ms) | NumPy, 1 process (ms) | NumPy, 4-process pool (ms) | Whole GA generation (ms) |
|---:|---:|---:|---:|---:|---:|
| 500 | 2.4 [2.3-2.6] | 3.3 [2.9-3.8] | 0.3 [0.2-0.4] | 1.8 [1.6-2.4] | 20.8 [19.6-28.3] |
| 2,000 | 10.6 [10.1-18.0] | 9.5 [8.4-10.0] | 1.3 [1.1-1.5] | 4.0 [3.6-4.2] | 87.1 [74.7-117.5] |
| 10,000 | 49.1 [47.0-49.9] | 31.7 [28.3-36.8] | 5.9 [5.3-6.8] | 14.9 [14.2-16.5] | 394.6 [375.4-425.6] |
<!-- END:fitness -->

32-stop map, 40 generations with migration every 10:

<!-- BEGIN:quick40 -->
| Method (40 generations, migration every 10) | Tour length, mean +/- sd | Best of 5 |
|---|---:|---:|
| GA + 2-opt polish | 417.8 +/- 30.4 | 368 |
| GA, MPI islands x4 + 2-opt polish | 373.8 +/- 31.1 | 334 |
<!-- END:quick40 -->

Tuning grid on the 32-stop map (3 tuning seeds, 500 generations):

<!-- BEGIN:tuning -->
| Tournament size | Mutation rate | Cost, mean +/- sd (3 tuning seeds) |
|---:|---:|---:|
| 2 | 0.1 | 575.0 +/- 54.0 |
| 2 | 0.3 | 719.3 +/- 18.6 |
| 2 | 0.6 | 808.7 +/- 60.9 |
| 2 | 0.9 | 770.7 +/- 80.3 |
| 3 | 0.1 | 591.3 +/- 35.9 |
| 3 | 0.3 | 517.0 +/- 29.1 |
| 3 | 0.6 | 592.0 +/- 41.5 |
| 3 | 0.9 | 691.0 +/- 62.6 |
| 5 | 0.1 | 563.0 +/- 12.2 |
| 5 | 0.3 | 572.0 +/- 70.5 |
| 5 | 0.6 | 531.0 +/- 33.9 |
| 5 | 0.9 | 562.0 +/- 30.5 |
<!-- END:tuning -->

</details>

## Limitations

- The GA uses order crossover and swap mutation, which move stops without looking at distances, so it loses to 2-opt and OR-Tools. It would need distance-aware operators or local search inside the loop to compete.
- One van with fixed distances: no capacities, time windows or traffic. The tables have no units and an unknown origin, so the lengths cannot be turned into real distances or cost.
- MPI ran on one 4-core machine only, so nothing shows multi-node scaling. Each island has a full population, so better tours with more islands come mostly from more total work, not from migration.
- OR-Tools stops at a time limit and gives no optimality proof, so its tour can depend on machine speed. The GA settings were tuned on the 32-stop map (other seeds), so the 32-stop GA numbers may be slightly optimistic.

## How to run

Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

Solve a map:

```bash
python -m route_ga --help
python -m route_ga --map 32 --method nn
python -m route_ga --map 100 --method 2opt
python -m route_ga --map 32 --method ga --seed 0 --out runs/ga32.json
python -m route_ga --map 32 --method ga --seed 0 --polish
python -m route_ga --map 32 --method ga-islands --islands 4 --seed 0
```

MPI islands need Open MPI (Linux, for example `sudo apt install openmpi-bin libopenmpi-dev`):

```bash
pip install -r requirements-mpi.txt
mpiexec -n 4 python -m route_ga --map 100 --method ga-islands --seed 0
```

Tests, lint and the checks CI runs:

```bash
pytest -rsx --cov=route_ga
ruff check .
python scripts/check_readme_claims.py
python scripts/pii_scan.py
```

Reproduce every result and figure (needs MPI; under an hour on 4 cores):

```bash
python scripts/run_experiments.py
python scripts/reconcile_runs.py
python scripts/make_figures.py
python scripts/record_quality.py
```

## Repo structure

```text
data/        the two distance tables and their notes
docs/        DECISIONS.md: design choices and the reason for each
figures/     PNG figures made by scripts/make_figures.py
results/     result JSON files and tables.md
scripts/     experiments, tables, figures, README claims check, PII scan
src/route_ga/  package: data, route cost, baselines, GA, MPI islands, OR-Tools reference, CLI
tests/       pytest suite, including regression tests for the fixed bugs
.github/workflows/ci.yml
```

## Licence and credits

- Code: MIT, see [`LICENSE`](LICENSE).
- Distance tables: supplied with the original exercise, real-world origin unknown, no licence stated; cite as "distance matrices as credited in `data/README.md`".
- The order crossover and swap mutation follow reference versions supplied with the original exercise, rewritten on NumPy arrays (see the header of `src/route_ga/ga.py`).
- Libraries: NumPy, mpi4py, Open MPI, Google OR-Tools, Matplotlib, pytest, Hypothesis, Ruff.
