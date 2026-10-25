# Canonical BPT benchmarks

Run from the Higra root. These runners compare the current stable-sort Kruskal
implementation, comparator-sort Kruskal, Waterfall–Boruvka exact output, basin
numbering with identical ties, and basin numbering with reverse edge-ID ties.
The Python runner calls the public APIs and offers exact and basin numbering;
reverse ties and comparator-sort Kruskal are native experiments only.

## Build

Provide Google Benchmark (tested with 1.9.4), the usual Higra dependencies, and
oneTBB for the on build. Use independent Release build directories:

```bash
cmake -S . -B build/bpt-off -DCMAKE_BUILD_TYPE=Release \
  -DDO_BENCHMARK=ON -DHG_USE_TBB=OFF -DHG_BPT_BENCH_DIAGNOSTICS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/bpt-off --target benchmark_higra \
  benchmark_higra_diagnostics benchmark_higra_cas higram --parallel 2
```

Repeat with `-DHG_USE_TBB=ON` in `build/bpt-on`; use `TBB_DIR` and `benchmark_DIR`
when dependencies are installed outside standard search paths. Offline builds
can provide `FETCHCONTENT_SOURCE_DIR_CATCH2`. Inspect `compile_commands.json`:
all methods share the root C++17/Release/LTO/SIMD flags, with fast math disabled
on benchmark targets. Do not inject fast math into binding builds. The two
additional executables compile optional diagnostic hooks; the primary executable
and Python extension contain no hooks. No scratch-reuse experiment is included.

## Smoke and campaign profiles

```bash
python benchmark/benchmark_bpt_canonical.py --build build/bpt-off \
  --out /tmp/bpt-off.json --repetitions 2 --min-time .005 --warmup .002
python benchmark/benchmark_bpt_canonical.py --build build/bpt-on --threads 4 \
  --backend python --out /tmp/bpt-python.json \
  --repetitions 2 --min-time .005 --warmup .002
```

The default `smoke` profile uses 64 vertices (64 for square grids and a cube
near that size for volumes). It contains 17 deliberately paired workloads,
rather than a Cartesian product:

| Family | Weights / dtype | Input |
| --- | --- | --- |
| 4-neighbor grid | Unique f64, equal i64, smooth gradient f64 | Full graph |
| 8-neighbor grid | Eight levels f32 | Full graph |
| 3-D 6-neighbor grid | High-cardinality f32 | Full graph |
| Path | Increasing u64 above 2^63 | Tree |
| Star | Eight levels u64 above 2^63 | Tree |
| Balanced tree | Unique i64 above 2^54 | Tree |
| Irregular tree | High-cardinality f64 | Tree |
| Connected random graph, average degree 4/8/32 | Unique f64 / eight levels i64 / high-cardinality f32 | Full graph |
| Local ring graph | Gradient f64 | Full graph |
| Dense complete graph | Unique i64 | Full graph |
| Hierarchically weighted path | Nested levels i64 | Tree, many rounds |
| 4-neighbor grid | Eight levels i64 | Explicit authoritative shuffled order |
| Connected random graph, degree 8 | Eight levels i64 | Already supplied MST |

`geometric` is the case label for the local ring graph: edges connect nearby
indices on a circle, with a spanning path. It is not a sampled Euclidean graph.
Grid gradients use a smooth sinusoidal signal. Random graphs start with a
spanning path and can contain parallel edges. Non-ordered cases shuffle edge
storage. Each case includes family, regime, dtype, input kind, actual vertex
count and seed in its name. Float and integer arrays retain their native dtype.
Python additionally measures a two-feature vector case on the 8-neighbor grid.
No downloaded images are required.

`--profile campaign` selects 65,536 vertices by default. Use separate invocations
with `--size 1024`, `10000`, `100000`, `1000000`, or image square sizes; grid
sizes round down to a square/cube, and dense graphs cap at 512 vertices. Select
families with `--filter REGEX`. `--seed` defaults to 1729. `--max-mib` defaults
to 1024 and gates a conservative input/workspace payload estimate before
allocation. It is a preflight guard, not an OS RSS limit. Increase it explicitly
for larger cases after checking available RAM. Cases over budget fail visibly.

For evaluation, use the defaults of five repetitions, 0.5 seconds per repetition
and 0.2 seconds warmup; run separate processes at 1, 2, 4, etc. up to the physical
core count. Run invocations sequentially on an otherwise idle host, under the
same externally selected `taskset`/`numactl` placement. Optional SMT sweeps should
be labeled separately. Smoke durations demonstrate operability only.

The native runner reads `TBB_NUM_THREADS` and owns a oneTBB `global_control`.
The driver supplies this variable and the Python worker separately calls
`hg.set_num_threads(P)`, which owns the corresponding limit outside the library
algorithm. Merely setting `TBB_NUM_THREADS` does not limit a plain Python process.
Limits are maxima (including the calling thread), not measurements of active
workers. Off builds require `--threads 1`. The driver sets OMP, MKL and OpenBLAS
limits to one, records affinity/NUMA information and does not change placement.
Native JSON records the active TBB cap and compiled TBB version. Observed worker
utilization is not sampled by this harness.

## Timed boundaries and correctness

Native cases use Google Benchmark `UseRealTime()`. Each timed call includes its
frontend validation, fresh allocations, all construction phases and required
sorts, numbering, altitude arrays, tree construction and result destruction.
The existing baseline retains its existing validation behavior. The comparator
baseline uses `hg::sort` on unique `(weight, edge ID)` keys and the existing
sorted-edge Kruskal core. Compare it with `kruskal` to isolate the sorting
implementation from contraction. Compare `exact` with `basin` to isolate node
numbering; compare `basin` with `reverse` separately for tie sensitivity.

Supplied-order Kruskal consumes the provided permutation directly. Boruvka
validates it and constructs inverse ranks inside its timed call. Its order is
authoritative and deliberately shuffled independently of weights. Both methods
receive the same permutation. MST cases supply the same tree topology and
input-local edge IDs to every method. MST preparation and order generation
are outside all timings. These cases cannot establish graph-to-BPT speedups.

The Python worker verifies package and extension paths under `--build`, then
includes the entire public call, normalization/validation, inverse ranks, possible
copies, altitudes, concepts, optional MST and result destruction. It compares
identical settings across methods: graph defaults, graph without MST, tuple
without MST, and tuple with tree-only output. Calls retain the existing GIL
behavior. Timing uses `perf_counter_ns`, records each batch duration per call
and iteration count, and consumes the root before deleting the result. Imports,
input generation/loading, graph creation, references and warmup are excluded.

Every primary timing validates its result first, outside timing. Exact and
comparator-sort outputs match current Kruskal parents, ordered maps and
altitudes. Basin results match every edge-labeled parent under the map bijection.
Reverse-tie results match a sorted-edge Kruskal reference using that same total
order. Validation also checks topological IDs, the self-parented last root and
binary degree. No arbitrary tied minima are implemented. Reference validation
on an explicit order checks that order, not numeric-weight monotonicity.

Native repetitions use Benchmark's random interleaving. Benchmark 1.9.4 seeds
that shuffle from `std::random_device` and exposes no replay seed; the input
seed remains reproducible, but the native repetition schedule is not replayed.
Python shuffles jobs
with the logged seed; repetitions of each job are consecutive. Input generators
are shared: the native exporter writes typed arrays read by NumPy, so each
method/frontend/thread limit can reproduce identical input hashes. Binary
exports use native endianness and are temporary; preserve manifests and regenerate
on the same platform. No cache/scratch reuse occurs between public calls.

## Diagnostics and memory

```bash
python benchmark/benchmark_bpt_canonical.py --build build/bpt-on --threads 4 \
  --mode diagnostics --filter 'hierarchical|grid8' --out /tmp/bpt-phases.json
python benchmark/benchmark_bpt_canonical.py --build build/bpt-on --threads 4 \
  --mode cas --filter 'star' --out /tmp/bpt-cas.json
python benchmark/benchmark_bpt_canonical.py --build build/bpt-on --threads 4 \
  --mode memory --filter 'grid8' --out /tmp/bpt-rss.json
python benchmark/benchmark_bpt_canonical.py --build build/bpt-on --threads 4 \
  --backend python --mode memory --filter 'grid8' --out /tmp/bpt-python-rss.json \
  --repetitions 1 --min-time .001 --warmup .001
```

Diagnostic calls run once in separate processes with the phase executable;
CAS runs use another executable with atomic attempt/failure counters, which can
change contention. Neither output is a primary timing. The optional engine sink
is private to these executables and supports one synchronous construction at a
time; do not enable it in a shared library or concurrent API clients.

Round records contain input components, crossing active edges, newly accepted
edges, remaining components and successor jump passes (including the final
no-change pass). Phases include minima with internal-edge compaction, selection
compaction, labeling, vertex relabeling, history assignment/conversion, expansion,
basin sort, rewiring, exact numbering sort, export mapping and native frontend
finalization. `waterfall_total` nests its subphases: do not sum it together with
those subphases. Frontend/unattributed time is the whole call minus disjoint
waterfall/export/finalization phases and includes validation and destruction.
Expansion records the number of ancestor climbs per selected slot.

`tracked_scratch_high_water_bytes` measures the high-water payload of explicitly
owned engine buffers at their overlapping lifetimes, using actual array sizes
and vector capacities. It includes history, labels, active arrays, inverse edge
map, flags, compaction counts, atomic saddles, saddles, labeler arrays and export
scratch. It excludes input/output arrays, frontend rank arrays, instrumentation
storage, allocator overhead, library sort scratch and TBB internals. It is not
RSS or total peak memory. CAS attempts/failures are absent from non-CAS runs;
in a serial build they are zero.

RSS uses `getrusage(RUSAGE_SELF).ru_maxrss`, normalized to bytes on Linux/macOS.
Each method/case is measured in a fresh process with one construction, no warmup
and no reference call. Inputs are exported in another process, so generating a
supplied MST/reference cannot pollute the measured high water. A separate child
validates each native case first; Python validates all jobs in a separate worker
before starting one memory worker per job. RSS includes executable/runtime or
Python imports, prebuilt inputs and loading/graph-creation temporaries. It is an
absolute process high water, not a baseline-subtracted algorithm allocation.
Memory calls use the ordinary uninstrumented executable/extension. On Linux,
the driver requires `/usr/bin/time` as a small external forking launcher, so a
native child's high water is not floored by the Python controller's pre-exec
address space. The launcher records the final child high water via `wait4`; the child's earlier
`getrusage` value is retained as `in_call_peak_rss_bytes`. Other platforms do
not use this GNU-specific launcher and need their RSS accounting checked before
interpreting small native cases.

## Output and manifests

Every run writes the requested JSON and a sibling `.manifest.json`. Native
primary JSON is Google Benchmark's `context`/`benchmarks` schema; `real_time`
uses each row's `time_unit`, and `items_per_second` measures input edges/s.
Use context `thread_limit` for the TBB cap; Benchmark's row `threads=1` counts
benchmark controller threads, not TBB workers.
Preserve individual `run_type=iteration` samples for ratios and variability;
aggregate rows are additional summaries. A sibling `.console.log` retains the
native table. Python timing rows record case/method/flags, validation,
`seconds_per_call` samples and `iterations`. Memory rows record
`peak_rss_bytes`; phase rows record `rounds`, `phases` and `expansion_climbs`.
All custom JSON uses `schema: 1`.

Manifests contain commands and overrides, seeds/profile/selection, Git revisions
and dirty states, tracked diff and source hashes (including new benchmark files),
binary hashes, CMake cache and actual compile commands, package/extension paths
and hashes for Python runs, Python/NumPy/SciPy versions, native Benchmark/TBB
context, linkage, CPU/topology/RAM, affinity, NUMA availability and limits.
Missing tools are recorded as unavailable. Compare output hashes with manifests
before combining runs. The runner rejects empty results and native error rows.

The command-line options used here are documented by the
[Google Benchmark guide](https://google.github.io/benchmark/user_guide.html)
and checked against the installed executable's `--help`. Speedup versus current,
self-scaling, exact/basin numbering cost and comparator-sort/tie sensitivity are
separate comparisons; report distributions and workload-specific outcomes.

## Uniform-grid size sweep

A focused native runner compares five curves on row-major square 4-adjacency
grids with seeded uniform float64 weights in `[0, 1)`:

```bash
cmake --build build/bpt-off --target benchmark_bpt_grid --parallel 2
cmake --build build/bpt-on --target benchmark_bpt_grid --parallel 2
ctest --test-dir build/bpt-off -R '^Test_bpt_grid_runner$' --output-on-failure
ctest --test-dir build/bpt-on -R '^Test_bpt_grid_runner$' --output-on-failure
python benchmark/benchmark_bpt_grid.py \
  --off-build build/bpt-off --on-build build/bpt-on --threads max \
  --sides 128 256 512 1024 1536 2048 --outdir /tmp/higra-grid-sweep
```

Reconfigure existing builds after registering this target. The sequential
Kruskal curve uses the TBB-disabled build. Kruskal with its existing parallel
stable sort and all three Boruvka variants use the TBB-enabled build at the same
caller-owned limit. `--threads max` selects the number of logical CPUs allowed
by process affinity, including SMT; the runner records requested and active
TBB caps. No thread-count sweep is implied.

The three Boruvka cases share one scalar frontend. Numbering and tie policies
are template parameters; the comparator contains no method-string or
supplied-rank dispatch. Thus the earlier general native adapter's dispatch
confound is absent from this sweep. No engine implementation changes occur.
Uniform double weights usually have few or no equal keys, so reverse tie
refinement is primarily a control, not a substantial tied-weight workload.

Each method/size is a fresh process with input generation and full reference
validation before warmup. Each primary call includes contract validation,
allocations, construction, all required sorts, altitudes/tree finalization,
consumption and destruction. Reference arrays are released before warmup.
Default measurement is five repetitions of batches lasting at least 0.5 s,
plus 0.2 s warmup; a call exceeding a batch floor is measured once. The driver
shuffles the five jobs per size using the logged seed and runs jobs sequentially.
Repetitions within a job are consecutive. Graphs append each vertex's right
edge then down edge, with no edge shuffle. The RNG and input fingerprint are
shared across both builds and all methods.

Outputs include one JSON/log per curve/size, combined `results.json`, and a
`manifest.json` with exact commands, flags, source/binary hashes, build caches,
versions, hardware, affinity and CPU quota. Results retain wall/CPU seconds per
call and batch iteration counts. CPU seconds / wall seconds is only an average
process CPU-use estimate, not a worker-count trace. This is a native construction
comparison; it does not include Python metadata or an explicit output MST graph.
