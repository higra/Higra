#!/usr/bin/env python3
# Copyright ESIEE Paris (2026)
# Distributed under the CECILL-B License; see LICENSE.
"""Run the five native BPT anchor cases in randomized fresh-process blocks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

CASES = ('kruskal', 'boruvka', 'boruvka_compact', 'boruvka_basin', 'boruvka_basin_compact')


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--threads', type=positive, default=8, help='caller TBB cap; serial builds report one')
    parser.add_argument('--blocks', type=positive, default=10)
    parser.add_argument('--seed', type=int, default=1729, help='case-order seed; graph seed stays 1729')
    parser.add_argument('--window', default='1', help='label for this execution window')
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    build = binary.parent.parent
    metadata = dict(schema=1, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    requested_thread_limit=args.threads, blocks=args.blocks, seed=args.seed,
                    window=args.window, platform=platform.platform(),
                    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    protocol='fresh process, oracle+structural check, one warmup, one timed native call through destruction')
    for name in ('CMakeCache.txt', 'compile_commands.json'):
        if (build / name).exists():
            (args.output / name).write_bytes((build / name).read_bytes())
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rng = random.Random(args.seed)
    with (args.output / 'samples.jsonl').open('w') as out, (args.output / 'stderr.log').open('w') as log:
        for block in range(args.blocks):
            order = list(CASES)
            rng.shuffle(order)
            for position, case in enumerate(order):
                command = [str(binary), '--bpt-case', case]
                result = subprocess.run(command, env={**os.environ, 'TBB_NUM_THREADS': str(args.threads)},
                                        capture_output=True, text=True, check=True)
                log.write(result.stderr)
                log.flush()
                row = json.loads(result.stdout)
                if (row['case'] != case or row['fingerprint'] != '7594927944782184811'
                        or not row['validated'] or row['n'] != 1048576 or row['m'] != 4188162
                        or row['seconds'] <= 0 or row['timed_calls'] != 1 or row['warmups'] != 1
                        or row['thread_limit'] != (args.threads if row['tbb'] else 1)):
                    raise RuntimeError(f'Invalid native result: {row}')
                row.update(block=block, position=position, window=args.window, command=command)
                out.write(json.dumps(row) + '\n')
                out.flush()
            print(f'window {args.window}: block {block + 1}/{args.blocks}', flush=True)


if __name__ == '__main__':
    main()
