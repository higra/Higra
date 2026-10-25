#!/usr/bin/env python3
# Copyright ESIEE Paris (2026)
# Distributed under the CECILL-B License; see LICENSE.
"""Run five native BPT curves on square 4-adjacency grids with uniform doubles."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(args, cwd=ROOT):
    run = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return run.stdout.strip() if run.returncode == 0 else dict(returncode=run.returncode, output=run.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off-build', type=Path, default=Path('build/agents-boruvka-bench-off'))
    parser.add_argument('--on-build', type=Path, default=Path('build/agents-boruvka-bench-on'))
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--sides', type=int, nargs='+', default=[128, 256, 512, 1024, 1536, 2048])
    parser.add_argument('--threads', default='max', help="'max' uses all CPUs allowed by process affinity")
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--min-time', type=float, default=.5)
    parser.add_argument('--warmup', type=float, default=.2)
    parser.add_argument('--max-mib', type=int, default=8192)
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    threads = len(affinity) if args.threads == 'max' and affinity else os.cpu_count() if args.threads == 'max' else int(args.threads)
    if not threads or threads < 1 or any(s < 2 for s in args.sides):
        parser.error('Invalid threads or sizes')
    builds = {'off': args.off_build.resolve(), 'on': args.on_build.resolve()}
    binaries = {k: path / 'benchmark/benchmark_bpt_grid' for k, path in builds.items()}
    manifest = dict(schema=1, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        invocation=sys.argv, cwd=str(ROOT), python=sys.version, platform=platform.platform(),
        public_revision=capture(['git', 'rev-parse', 'HEAD']),
        public_status=capture(['git', 'status', '--short', '--untracked-files=all']),
        private_revision=capture(['git', 'rev-parse', 'HEAD'], ROOT / 'higra-workspace'),
        private_status=capture(['git', 'status', '--short'], ROOT / 'higra-workspace'),
        affinity=affinity, max_thread_limit=threads,
        hardware=capture(['lscpu']), memory=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for folder in (ROOT / 'benchmark', ROOT / 'include/higra')
                       for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ('.py', '.cpp', '.hpp', '.txt')},
        builds={}, commands=[],
        protocol=dict(sides=args.sides, seed=args.seed, repetitions=args.repetitions,
                      minimum_batch_seconds=args.min_time, warmup_seconds=args.warmup,
                      max_mib=args.max_mib, weights='float64 uniform [0,1), xorshift64 top 53 bits',
                      edge_order='row-major, right edge then down edge, no shuffle',
                      timed='complete construction, allocations, required sorts, validation, finalization, consumption and destruction',
                      excluded='input generation, reference validation and warmup',
                      method_order='seeded shuffle per size; jobs run sequentially; repetitions consecutive within job'))
    for key, build in builds.items():
        manifest['builds'][key] = dict(path=str(build), binary_sha256=sha(binaries[key]),
            cache=(build / 'CMakeCache.txt').read_text(), linkage=capture(['ldd', str(binaries[key])]),
            compile_commands=[r for r in json.loads((build / 'compile_commands.json').read_text())
                              if r['file'].endswith('/benchmark/benchmark_bpt_grid.cpp')])
    methods = [('kruskal_sequential', 'off', 'kruskal'), ('kruskal_parallel_sort', 'on', 'kruskal'),
               ('boruvka_exact', 'on', 'exact'), ('boruvka_basin', 'on', 'basin'), ('boruvka_reverse', 'on', 'reverse')]
    rows = []
    result_path, manifest_path = args.outdir / 'results.json', args.outdir / 'manifest.json'
    # Reset controls inherited from earlier benchmark commands. The executable
    # owns its own TBB limit; OMP/BLAS runtimes get one thread consistently.
    overrides = {key: '1' for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}
    for key in ['BPT_PROFILE', 'BPT_SIZE', 'BPT_CASE', 'BPT_LOAD_PREFIX', 'BPT_RUN_MODE', 'TBB_NUM_THREADS']:
        overrides[key] = ''
    rng = random.Random(args.seed)
    for side in args.sides:
        order = list(methods)
        rng.shuffle(order)
        for curve, mode, method in order:
            target = args.outdir / f'{side}-{curve}.json'
            cmd = [str(binaries[mode]), '--side', str(side), '--method', method, '--seed', str(args.seed),
                   '--threads', str(1 if mode == 'off' else threads), '--repetitions', str(args.repetitions),
                   '--min-time', str(args.min_time), '--warmup', str(args.warmup), '--max-mib', str(args.max_mib),
                   '--out', str(target)]
            print(f'RUN {side} x {side} ({side*side:,} vertices) · {curve} · threads={1 if mode == "off" else threads}', flush=True)
            started = time.perf_counter()
            with target.with_suffix('.log').open('w') as log:
                run = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **overrides}, stdout=log, stderr=subprocess.STDOUT)
            manifest['commands'].append(dict(argv=cmd, cwd=str(ROOT), environment_overrides=overrides,
                                             returncode=run.returncode, wall_seconds=time.perf_counter()-started))
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            if run.returncode:
                raise RuntimeError(f'{curve}, side {side} failed; see {target.with_suffix(".log")}')
            record = json.loads(target.read_text())
            if not record['validated'] or len(record['seconds_per_call']) != args.repetitions:
                raise RuntimeError(f'Invalid output: {target}')
            record.update(curve=curve, build=mode, source_file=target.name, source_sha256=sha(target),
                          median_seconds=statistics.median(record['seconds_per_call']),
                          min_seconds=min(record['seconds_per_call']), max_seconds=max(record['seconds_per_call']))
            rows.append(record)
            result_path.write_text(json.dumps(dict(schema=1, rows=rows), indent=2) + '\n')
            print(f'  median {record["median_seconds"]:.6f} s; validation passed', flush=True)
        group = [r for r in rows if r['side'] == side]
        if len({r['input_fnv1a64'] for r in group}) != 1:
            raise RuntimeError(f'Mismatched inputs at side {side}')
        strict = [r for r in group if r['method'] in ('kruskal', 'exact')]
        if len({r['output_fnv1a64'] for r in strict}) != 1:
            raise RuntimeError(f'Exact outputs differ across builds at side {side}')
    manifest['result_sha256'] = sha(result_path)
    manifest['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Completed {len(rows)} validated method/size cells at {threads} max parallel threads.', flush=True)
    print(result_path, flush=True)


if __name__ == '__main__':
    main()
