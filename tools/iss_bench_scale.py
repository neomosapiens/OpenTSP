#!/usr/bin/env python3
"""Linux CPU scaling of the eight-layer P8_4_128 ISS benchmark prepared by iss_bench_prep.py."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def physical_cpus():
    """Pick one allowed logical CPU per physical core, filling each socket in order."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = tuple(int((topology / field).read_text()) for field in ("physical_package_id", "core_id"))
        cores.setdefault(key, cpu)
    return [cores[key] for key in sorted(cores)]


def batch(binary, bench, groups, threads):
    with tempfile.TemporaryDirectory() as tmp:
        children = []
        start = time.perf_counter()
        try:
            for idx, group in enumerate(groups):
                stats = Path(tmp) / f"{idx}.json"
                cmd = ["taskset", "-c", ",".join(map(str, group)), str(binary), "run", "--profile", "ref",
                       "--weights", str(bench / "P8_4_128.safetensors"), f"--stats={stats}", str(bench / "P8_4_128.elf")]
                children.append((stats, subprocess.Popen(cmd, env=dict(os.environ, TSP_ISS_THREADS=str(threads)),
                                                         stdout=subprocess.DEVNULL)))
            for _, child in children:
                if child.wait():
                    raise RuntimeError(f"ISS exited with status {child.returncode}")
            wall = time.perf_counter() - start
        finally:
            for _, child in children:
                if child.poll() is None:
                    child.kill()
                child.wait()
        times = []
        for stats, _ in children:
            row = json.loads(stats.read_bytes())
            counts = tuple(row[k] for k in ("last_cycle", "ticks", "dispatches", "diagnostics"))
            if counts != (221648, 168186, 496698, 0):
                raise ValueError(f"unexpected eight-layer P8 counts {counts}")
            times.append(row["wall_seconds"])
        return {"batch_seconds": wall, "median_simulation_seconds": statistics.median(times),
                "aggregate_cycles_per_second": len(groups) * 221648 / wall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("bench_dir", type=Path)
    parser.add_argument("--workers", nargs="+", type=int, help="thread counts for one process")
    parser.add_argument("--processes", nargs="+", type=int, help="numbers of concurrent single-core processes")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux CPU affinity and sysfs CPU topology are required")
    cpus = physical_cpus()
    workers = args.workers or [n for n in (1, 2, 4, 8) if n <= len(cpus)]
    processes = args.processes or [n for n in (1, 2, 4, 8, 16, 32, 48) if n <= len(cpus)]
    if args.repeats < 1 or min(workers + processes) < 1 or max(workers + processes) > len(cpus):
        parser.error(f"choose positive repeats and counts within the {len(cpus)} available physical cores")
    if max(workers) > 16:
        parser.error("the single-chip matrix pool supports at most 16 workers including the calling thread")
    cases = [("workers", n) for n in workers] + [("processes", n) for n in processes]
    for r in range(args.repeats + 1):
        for mode, count in (cases if r % 2 == 0 else reversed(cases)):
            groups = [cpus[:count]] if mode == "workers" else [[cpu] for cpu in cpus[:count]]
            row = batch(args.binary.resolve(), args.bench_dir, groups, count if mode == "workers" else 1)
            print(json.dumps(dict(round=r, mode=mode, cpus=cpus[:count], count=count, **row)), flush=True)


if __name__ == "__main__":
    main()
