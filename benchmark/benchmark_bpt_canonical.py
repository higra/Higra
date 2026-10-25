#!/usr/bin/env python3
# Copyright ESIEE Paris (2026)
# Distributed under the CECILL-B License; see LICENSE.
"""Reproducible native/public-Python BPT timings and isolated measurements.

Run from the Higra root. See benchmark/README-bpt.md for boundaries and schemas.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
COMMANDS = []
LAST_CHILD_RSS_BYTES = None


def command(argv, *, env=None, cwd=ROOT, capture=True, fresh_child=False):
    # Linux can charge the pre-exec Python address-space high water to a
    # directly spawned native child. GNU time forks from its small own image.
    global LAST_CHILD_RSS_BYTES
    LAST_CHILD_RSS_BYTES = None
    if fresh_child and sys.platform.startswith("linux"):
        if not Path("/usr/bin/time").exists():
            raise RuntimeError("Isolated Linux RSS runs require /usr/bin/time")
        argv = ["/usr/bin/time", "-f", "HIGRA_RSS_KIB=%M", *argv]
    COMMANDS.append({"argv": list(map(str, argv)), "cwd": str(cwd),
                     "environment_overrides": env or {}})
    result = subprocess.run(list(map(str, argv)), cwd=cwd,
                            env={**os.environ, **(env or {})}, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {argv}\n"
                           f"{result.stdout}\n{result.stderr}")
    if fresh_child and sys.platform.startswith("linux"):
        matched = re.search(r"HIGRA_RSS_KIB=(\d+)", result.stderr)
        if not matched:
            raise RuntimeError("Missing GNU time RSS measurement")
        LAST_CHILD_RSS_BYTES = int(matched[1]) * 1024
    return result.stdout if capture else None


def optional(argv, cwd=ROOT):
    try:
        return command(argv, cwd=cwd).strip()
    except (OSError, RuntimeError) as error:
        return {"unavailable": str(error)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(args):
    cache = args.build / "CMakeCache.txt"
    compile_file = args.build / "compile_commands.json"
    compile_commands = json.loads(compile_file.read_text())
    relevant = [entry for entry in compile_commands
                if "/benchmark/" in entry["file"] or "/higra/" in entry["file"]]
    return {
        "schema": 1, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "invocation": sys.argv, "cwd": str(ROOT), "python_executable": sys.executable,
        "python_version": sys.version, "platform": platform.platform(),
        "public_revision": optional(["git", "rev-parse", "HEAD"]),
        "public_status": optional(["git", "status", "--porcelain=v1", "--untracked-files=all"]),
        "public_diff_sha256": hashlib.sha256(command(["git", "diff", "HEAD"]).encode()).hexdigest(),
        "private_revision": optional(["git", "rev-parse", "HEAD"], ROOT / "higra-workspace"),
        "private_status": optional(["git", "status", "--porcelain=v1"], ROOT / "higra-workspace"),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for directory in
                          (ROOT / "benchmark", ROOT / "include/higra", ROOT / "higra")
                          for p in sorted(directory.rglob("*")) if p.is_file()
                          and p.suffix in (".hpp", ".cpp", ".py", ".txt")},
        "cmake_cache": cache.read_text(), "compile_commands": relevant,
        "benchmark_binary": str(args.binary), "binary_sha256": digest(args.binary),
        "benchmark_help": optional([str(args.binary), "--help"]),
        "linkage": optional(["ldd", str(args.binary)]),
        "hardware": optional(["lscpu"]), "numa": optional(["numactl", "--show"]),
        "memory": Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith(("BPT_", "TBB_", "OMP_", "MKL_", "OPENBLAS_")) or
                        k in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH")},
        "thread_limit": args.threads, "thread_limit_is_maximum": True,
        "observed_worker_utilization": None,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "timing_boundary": "whole call plus result destruction; inputs/reference validation excluded; fresh allocations",
    }


def select_cases(args, env):
    listing = command([args.binary, "--benchmark_list_tests=true"], env=env)
    names = [line.strip().removesuffix("/real_time") for line in listing.splitlines()
             if line.startswith("BM_bpt/")]
    cases = sorted({name.removeprefix("BM_bpt/").rsplit("/", 1)[0] for name in names})
    return [case for case in cases if re.search(args.filter, case)], names


def native_single(args, env, case, method, mode, binary=None, prefix=None):
    extra = {"BPT_CASE": case, "BPT_METHOD": method, "BPT_RUN_MODE": mode}
    if prefix is not None:
        extra["BPT_INPUT_PREFIX" if mode == "inputs" else "BPT_LOAD_PREFIX"] = str(prefix)
    result = json.loads(command([binary or args.binary], env={**env, **extra}, fresh_child=mode == "memory"))
    if mode == "memory" and LAST_CHILD_RSS_BYTES is not None:
        result["in_call_peak_rss_bytes"] = result["peak_rss_bytes"]
        result["peak_rss_bytes"] = LAST_CHILD_RSS_BYTES
        result["rss_source"] = "GNU time wait4 final process high water"
    return result


def export_input(args, env, case, directory):
    prefix = Path(directory) / "input"
    meta = native_single(args, env, case, "exact", "inputs", prefix=prefix)
    meta["prefix"] = str(prefix)
    meta["sha256"] = {suffix: digest(str(prefix) + "." + suffix)
                      for suffix in ("sources", "targets", "weights", "order")}
    return meta


def load_higra(args):
    # Running a script by path puts benchmark/ first. Select and verify the
    # configured build explicitly, even if invoked from the repository root.
    sys.path.insert(0, str(args.build))
    import higra as hg
    import numpy as np
    for path in (hg.__file__, hg.cpp.__file__):
        if not Path(path).resolve().is_relative_to(args.build):
            raise RuntimeError(f"Wrong build import: {path}")
    cache = (args.build / "CMakeCache.txt").read_text()
    tbb = bool(re.search(r"^HG_USE_TBB:BOOL=ON$", cache, re.M))
    if not tbb and args.threads != 1:
        raise ValueError("A TBB-off build requires --threads=1.")
    if tbb:
        hg.set_num_threads(args.threads)  # caller-owned oneTBB global_control
    return hg, np, {"package": hg.__file__, "extension": hg.cpp.__file__,
                    "extension_sha256": digest(hg.cpp.__file__), "numpy": np.__version__,
                    "scipy": __import__("scipy").__version__, "tbb": tbb,
                    "limiter": "hg.set_num_threads" if tbb else "serial build",
                    "extension_linkage": optional(["ldd", hg.cpp.__file__])}


def python_data(hg, np, meta, case):
    prefix = meta["prefix"]
    src = np.fromfile(prefix + ".sources", dtype=np.int64)
    dst = np.fromfile(prefix + ".targets", dtype=np.int64)
    dtype = {"f32": np.float32, "f64": np.float64, "i64": np.int64, "u64": np.uint64}[meta["dtype"]]
    weights = np.fromfile(prefix + ".weights", dtype=dtype)
    order = np.fromfile(prefix + ".order", dtype=np.int64) if "/order/" in case else None
    graph = hg.UndirectedGraph(meta["n"])
    graph.add_edges(src, dst)
    return graph, (src, dst, meta["n"]), weights, order


def python_call(hg, data, method, flags):
    graph, triplet, weights, order = data
    fn = hg.bpt_canonical if method == "kruskal" else hg.bpt_canonical_boruvka
    kwargs = dict(sorted_edge_indices=order, compute_mst=flags["compute_mst"],
                  return_altitudes=flags["return_altitudes"])
    if method != "kruskal":
        kwargs["stable"] = method == "exact"
    return fn(triplet if flags["tuple"] else graph, weights, **kwargs)


def python_validate(hg, np, data, method, flags):
    got = python_call(hg, data, method, flags)
    ref = python_call(hg, data, "kruskal", flags)
    a, b = (got[0], ref[0]) if flags["return_altitudes"] else (got, ref)
    n = data[0].num_vertices()
    p, q = a.parents(), b.parents()
    if not (len(p) == 2 * n - 1 and p[-1] == len(p) - 1 and
            np.all(p[:-1] > np.arange(len(p) - 1)) and
            np.all(np.bincount(p[:-1], minlength=len(p))[n:] == 2)):
        raise AssertionError("Invalid binary/topological tree")
    edge_to_node = {int(e): n + i for i, e in enumerate(b.mst_edge_map)}
    remap = np.concatenate((np.arange(n), [edge_to_node[int(e)] for e in a.mst_edge_map]))
    np.testing.assert_array_equal(remap[p], q[remap])
    if method != "basin":
        np.testing.assert_array_equal(remap, np.arange(len(p)))
    if flags["return_altitudes"]:
        np.testing.assert_array_equal(got[1], ref[1][remap])
    if not flags["tuple"]:
        if not (hg.CptHierarchy.validate(a) and hg.CptBinaryHierarchy.validate(a) and
                (a.mst is not None) == flags["compute_mst"]):
            raise AssertionError("Wrong hierarchy/MST concepts")


def python_worker(args, payload):
    hg, np, imports = load_higra(args)
    rows = []
    jobs = payload["jobs"]
    random.Random(args.seed).shuffle(jobs)
    for job in jobs:
        meta, case, method, flags = (job[k] for k in ("meta", "case", "method", "flags"))
        data = python_data(hg, np, meta, case)
        if flags.get("vector"):
            data = (*data[:2], np.stack((data[2], data[2] % 3), axis=1), data[3])
        if args.mode == "memory":
            # References and warmups run in a DIFFERENT process before this.
            result = python_call(hg, data, method, flags)
            del result
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rows.append({**job, "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024})
            continue
        python_validate(hg, np, data, method, flags)
        def batch(seconds):
            start = time.perf_counter_ns()
            count = 0
            while True:
                result = python_call(hg, data, method, flags)
                # Consume output, including tree-only calls; destruction timed.
                tree = result[0] if flags["return_altitudes"] else result
                root = tree.root()
                del tree, result
                count += 1
                elapsed = (time.perf_counter_ns() - start) * 1e-9
                if elapsed >= seconds:
                    return elapsed / count, count, root
        batch(args.warmup)
        samples = [batch(args.min_time) for _ in range(args.repetitions)]
        rows.append({**job, "validated": True, "seconds_per_call": [s[0] for s in samples],
                     "iterations": [s[1] for s in samples], "consumed_root": samples[-1][2]})
    return {"schema": 1, "imports": imports, "rows": rows,
            "job_order": [f'{j["case"]}/{j["method"]}/{j["flags"]["name"]}' for j in jobs]}


def jobs_for(cases, metas):
    flag_sets = [
        dict(name="graph_defaults", tuple=False, compute_mst=True, return_altitudes=True),
        dict(name="graph_no_mst", tuple=False, compute_mst=False, return_altitudes=True),
        dict(name="tuple_no_mst", tuple=True, compute_mst=False, return_altitudes=True),
        dict(name="tuple_tree", tuple=True, compute_mst=False, return_altitudes=False),
    ]
    jobs = []
    for case in cases:
        flags = list(flag_sets)
        if case.startswith("grid8/"):
            flags.append(dict(name="tuple_vector", tuple=True, compute_mst=False, return_altitudes=True, vector=True))
        for method in ("kruskal", "exact", "basin"):
            for option in flags:
                jobs.append(dict(case=case, meta=metas[case], method=method, flags=option))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    parser.add_argument("--mode", choices=("timing", "memory", "diagnostics", "cas"), default="timing")
    parser.add_argument("--profile", choices=("smoke", "campaign"), default="smoke")
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-mib", type=int, default=1024)
    parser.add_argument("--filter", default=".*", help="Regex on input case names")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-time", type=float, default=.5)
    parser.add_argument("--warmup", type=float, default=.2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-payload", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.build = args.build.resolve()
    args.binary = args.build / "benchmark/benchmark_higra"
    args.out = args.out.resolve()
    if args.threads < 1 or args.repetitions < 1 or args.min_time <= 0 or args.warmup < 0:
        parser.error("Thread/repetition counts and timing duration must be positive.")
    if args.worker_payload:
        result = python_worker(args, json.loads(args.worker_payload.read_text()))
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = {"BPT_PROFILE": args.profile, "BPT_SEED": str(args.seed),
           "BPT_MAX_MIB": str(args.max_mib), "TBB_NUM_THREADS": str(args.threads),
           "BPT_RUN_MODE": "", "BPT_LOAD_PREFIX": "", "BPT_SIZE": "64" if args.profile == "smoke" else "65536",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if args.size:
        env["BPT_SIZE"] = str(args.size)
    cache = (args.build / "CMakeCache.txt").read_text()
    if args.threads != 1 and not re.search(r"^HG_USE_TBB:BOOL=ON$", cache, re.M):
        parser.error("TBB-off builds require --threads=1.")
    info = manifest(args)
    cases, names = select_cases(args, env)
    if not cases:
        parser.error("No matching cases.")
    info["selected_cases"] = cases
    with tempfile.TemporaryDirectory(prefix="higra-bpt-") as temporary:
        metas = {}
        if args.backend == "python" or args.mode != "timing":
            for i, case in enumerate(cases):
                directory = Path(temporary) / str(i)
                directory.mkdir()
                metas[case] = export_input(args, env, case, directory)
            info["inputs"] = {case: {k: v for k, v in meta.items() if k != "prefix"}
                              for case, meta in metas.items()}
        if args.backend == "native":
            if args.mode == "timing":
                selected = "|".join(re.escape("BM_bpt/" + c + "/") for c in cases)
                console = command([args.binary, "--benchmark_filter=" + selected,
                                   "--benchmark_repetitions=" + str(args.repetitions),
                                   "--benchmark_min_time=" + str(args.min_time) + "s",
                                   "--benchmark_min_warmup_time=" + str(args.warmup),
                                   "--benchmark_enable_random_interleaving=true",
                                   "--benchmark_out=" + str(args.out), "--benchmark_out_format=json"], env=env)
                args.out.with_suffix(".console.log").write_text(console)
                result = json.loads(args.out.read_text())
                if not result.get("benchmarks") or any(r.get("error_occurred") for r in result["benchmarks"]):
                    raise RuntimeError("Empty/failed native benchmark result")
            else:
                rows = []
                binary = args.binary if args.mode == "memory" else args.binary.with_name(
                    "benchmark_higra_cas" if args.mode == "cas" else "benchmark_higra_diagnostics")
                for case in cases:
                    for method in ("kruskal", "kruskal_comparator", "exact", "basin", "reverse"):
                        if "/order/" in case and method in ("reverse", "kruskal_comparator"):
                            continue
                        prefix = metas[case]["prefix"]
                        native_single(args, env, case, method, "validate", binary=binary, prefix=prefix)
                        row = native_single(args, env, case, method, "memory" if args.mode == "memory" else "diagnostics",
                                            binary=binary, prefix=prefix)
                        rows.append({"case": case, "method": method, "validated_in_separate_process": True, **row})
                result = {"schema": 1, "mode": args.mode, "rows": rows}
                args.out.write_text(json.dumps(result, indent=2) + "\n")
        else:
            if args.mode not in ("timing", "memory"):
                parser.error("Phase/CAS diagnostics use the native runner.")
            jobs = jobs_for(cases, metas)
            def worker(worker_jobs, suffix, mode):
                payload = Path(temporary) / (suffix + ".json")
                payload.write_text(json.dumps({"jobs": worker_jobs}))
                target = Path(temporary) / (suffix + "-result.json")
                command([sys.executable, SCRIPT, "--build", args.build, "--backend", "python",
                         "--mode", mode, "--threads", args.threads, "--seed", args.seed,
                         "--worker-payload", payload, "--out", target,
                         "--repetitions", args.repetitions, "--min-time", args.min_time, "--warmup", args.warmup],
                        env=env, cwd=args.build, fresh_child=mode == "memory")
                measured = json.loads(target.read_text())
                if mode == "memory" and LAST_CHILD_RSS_BYTES is not None:
                    for row in measured["rows"]:
                        row["in_call_peak_rss_bytes"] = row["peak_rss_bytes"]
                        row["peak_rss_bytes"] = LAST_CHILD_RSS_BYTES
                        row["rss_source"] = "GNU time wait4 final process high water"
                return measured
            if args.mode == "timing":
                result = worker(jobs, "timing", "timing")
            else:
                worker(jobs, "validation", "timing")
                rows = []
                for i, job in enumerate(jobs):
                    measured = worker([job], str(i), "memory")
                    rows.extend(measured["rows"])
                result = {"schema": 1, "imports": measured["imports"], "rows": rows,
                          "validation_in_separate_process": True}
            info["imports"] = result["imports"]
            args.out.write_text(json.dumps(result, indent=2) + "\n")
    info["commands"] = COMMANDS
    info["result_sha256"] = digest(args.out)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"result": str(args.out), "manifest": str(args.out.with_suffix('.manifest.json')),
                      "cases": len(cases), "rows": len(result.get("rows", result.get("benchmarks", [])))}))


if __name__ == "__main__":
    main()
