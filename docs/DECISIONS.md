# Design decisions

One "why" per choice. Numbers quoted here come from `results/`.

## Representation
An individual is a permutation of the customer nodes 1..n-1; the depot 0 is
not stored. The closed tour is always built as `[0, *perm, 0]`.
**Why:** the depot can never move, be duplicated or be lost, so crossover and
mutation only ever need to keep a plain permutation valid, and the return leg
to the depot is added in exactly one place (`route.py`). The 2025 code stored
the depot inside the route and forgot the return leg.

## Fitness sign and missing edges
The GA minimises a positive cost: the tour length, where every missing edge
(cell >= 100000) costs 100000.
**Why:** one direction everywhere (lower is better) removes the 2025 bug where
negative fitness was combined with `min`/`argmin`. The penalty is larger than
any feasible tour on these maps (at most 99 per edge, so under 10,000 for 100
nodes), so any tour with fewer missing edges always ranks first. Using `>=`
instead of `==` also covers the 10000 vs 100000 mismatch from 2025.

## Selection
Tournament selection, size 3, lowest cost wins, contestants drawn without
replacement.
**Why:** it only compares costs, so it does not care about the scale of the
penalty (roulette-wheel selection would be dominated by it). Size 3 came out
best in a small grid (`results/tuning.json`, 3 tuning seeds that are not used
in any reported result). Drawing without replacement makes a tournament of
the whole population return the best individual, which a unit test checks.

## Crossover
Order crossover (OX), the same variant as the course code, rate 0.9.
**Why:** it keeps a block of one parent's order and the relative order of the
other, and always produces a valid permutation (a property test checks this).
I kept the course variant so the 2025 and 2026 operators can be compared.

## Mutation
Swap two random positions with probability 0.3 per child. Returns a copy.
**Why:** swap mutation was the course operator and the brief asked for it.
0.3 was best of {0.1, 0.3, 0.6, 0.9} in the tuning grid. The 2025 `mutate`
changed its input in place, which is easy to get wrong with shared parents, so
the new one copies. I know an inversion (2-opt style) mutation would fit TSP
better; I did not add it so the GA stays the one described in the brief, and
2-opt is used as a separate polish step instead.

## Elitism
The 2 best individuals are copied unchanged into the next generation.
**Why:** it guarantees that the best cost in the population never gets worse,
which is exactly the property the 2025 code broke. The regression test checks
it on every generation.

## Stagnation restart
If the best cost does not improve for 150 generations, keep the elites and
replace everyone else with random tours.
**Why:** the course skeleton had a restart idea (limit 5, which restarted
almost every few generations). A longer limit gives the population time to
converge first; keeping the elites keeps the monotone guarantee.

## Population and budget
Population 200; 1,000 generations on the 32-node map and 2,000 on the
100-node map; 5 seeds (0 to 4) for every reported number.
**Why:** a single-population GA run takes seconds to about half a minute here,
so 5 seeds of every method, including the serial and MPI island runs, fit in
well under an hour on a 4-core machine. The 2025 script used population
10,000 with only 2 children per generation, so it barely evolved.

## Island topology
Ring: island r sends to r+1 (mod size) and receives from r-1, using
`comm.sendrecv`. Each island has its own random stream derived from the seed
with `numpy.random.SeedSequence.spawn`.
**Why:** a ring is the simplest topology with one message per island per
migration, so communication stays small and deadlock free. Deriving the
streams from one seed makes the MPI run give exactly the same result as the
serial twin (`run_islands_serial`), which the tests and the experiments check.
That is how I know the MPI version is correct and not just fast.

## Migration interval and size
Every 25 generations, copies of the 2 best tours move to the next island and
replace its 2 worst.
**Why:** short enough that good tours spread several times per run, long
enough that islands develop different tours in between. Copies (not moves)
mean an island never loses its own best. These values were chosen, not tuned.

## Island scaling design
Each island keeps the full population of 200 (more islands = more total
search), and each island is one MPI process on the same machine.
**Why:** this shows the useful case for parallel islands: 4 times the search
in about the time of one. To get a fair speed-up number I also run the same
islands one after another in one process and compare the times; the results
are identical, so the speed-up is for the same computation.

## Baselines
Nearest neighbour from the depot, the same tour improved by 2-opt, and exact
brute force for 8-node instances.
**Why:** a GA result means little without a simple reference. Nearest
neighbour is the obvious greedy answer; 2-opt is the standard local search
for symmetric TSP; brute force gives a true optimum on small instances, so it
anchors how often each method actually finds the best tour. "Best known" on
the 32- and 100-node maps means the shortest tour any method found in this
study, not a proven optimum.

## Fitness evaluation in parallel
Kept as an option (`--workers`), with a spawn-based pool and module-level
workers, but not used in the main experiments.
**Why:** the benchmark (`results/parallel_fitness.json`) shows that a NumPy
vectorised evaluation of the whole population in one process is faster than a
4-process pool at every population size tested, because one generation of
fitness is only milliseconds of work and the pool pays for sending data to
other processes. The pool only beats the plain Python loop it replaces.

## Randomness
`numpy.random.Generator` objects are passed explicitly; there is no global
`np.random.seed`.
**Why:** results must be reproducible from the seed in the JSON file, and the
same code must work inside worker processes and MPI ranks.
