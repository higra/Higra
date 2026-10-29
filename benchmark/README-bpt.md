# BPT anchor benchmark

`benchmark_higra` compares stable Kruskal with Borůvka exact/basin numbering,
with and without active-edge compaction. It uses the native scalar constructors,
including input validation, all output allocation, altitude/tree finalization
and result destruction. Graph generation, oracle checks and one explicit warmup
are outside timing. Exact outputs match Kruskal; basin outputs are checked through
original building-edge IDs. Untimed checks also verify binary structure, parent
ordering, altitude monotonicity and leaf/root properties.

The fixed input has 1,048,576 vertices and 4,188,162 edges: a 1024 × 1024 grid,
row-major right/down/down-right/down-left edges, xorshift seed 1729, float32
weights on multiples of `2^-24`. Its fingerprint is `7594927944782184811`.

From the repository root, with Google Benchmark installed:

```bash
cmake -S . -B build/bpt-on -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DDO_BENCHMARK=ON -DHG_USE_TBB=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/bpt-on --target benchmark_higra --parallel 2
TBB_NUM_THREADS=8 build/bpt-on/benchmark/benchmark_higra \
  --benchmark_filter='^BPT/' --benchmark_out=anchor.json --benchmark_out_format=json
```

Supply `TBB_DIR` or `benchmark_DIR` if their CMake packages are outside the
normal search path. Offline configurations also need an existing Catch2 source
via `FETCHCONTENT_SOURCE_DIR_CATCH2`; see the repository build guide. For serial
measurements use another build directory and `-DHG_USE_TBB=OFF`. All cases inherit
the project's compiler/LTO/SIMD settings, with ordinary IEEE floating comparisons.
TBB thread control belongs to the executable; `TBB_NUM_THREADS` is a maximum,
not a promise of observed worker utilization. The documented anchor cap is 8.

Google Benchmark uses real wall time and may calibrate/batch many calls, exposing
a different allocator warm state from a single-call sample. For randomized
matched blocks with a fresh process per case, use the same native implementation:

```bash
python benchmark/benchmark_bpt_canonical.py \
  --binary build/bpt-on/benchmark/benchmark_higra --threads 8 --blocks 10 \
  --window 1 --seed 1729 --output build/bpt-results-window1
python benchmark/benchmark_bpt_canonical.py \
  --binary build/bpt-on/benchmark/benchmark_higra --threads 8 --blocks 10 \
  --window 2 --seed 1730 --output build/bpt-results-window2
```

Each native process validates its output, destroys the reference/check output,
performs one warmup and times one allocating call through destruction using a
steady wall clock. JSONL rows record case, wall seconds, fingerprint, validation,
TBB mode/cap, window/block/order and command. The manifest records binary hash,
platform, affinity and protocol; adjacent build cache/compile commands are copied
when available. Keep machine state controlled and avoid overlapping builds/tests
or other benchmark jobs. Analyze within-block ratios with uncertainty; do not pool
these samples with Google Benchmark's batched measurements.

For a single sample or the small runner fixture:

```bash
TBB_NUM_THREADS=8 build/bpt-on/benchmark/benchmark_higra --bpt-case boruvka
build/bpt-on/benchmark/benchmark_higra --bpt-self-test
```

The five case names are `kruskal`, `boruvka`, `boruvka_compact`, `boruvka_basin`,
and `boruvka_basin_compact`. `boruvka` uses exact numbering and full-edge scans;
`compact` enables active lists. Both numberings use ascending original edge IDs
to break weight ties. The fixture uses 16 vertices and checks adapters/argument
validation; run it explicitly when checking the benchmark runner. Results
apply to this synthetic workload and measured host, without a universal algorithm
or traversal recommendation.
