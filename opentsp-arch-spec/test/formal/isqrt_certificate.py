#!/usr/bin/env python3
"""Complete bounded production-isqrt root-floor postcondition certificate.

This establishes s*s <= n < (s+1)*(s+1) for n=m2*2^(2p),
all m2 in [0,2^25) and p=64,128,256,512,1024,2048,4096, under native trust.
It does not establish interval enclosure, correct rounding, or a kernel proof.
Completed immutable shards are retained across a pilot and the full invocation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time

WIDTHS = (25, 25, 25, 25, 25, 25, 25)
CHUNK = 1 << 16
EXPECTED = sum(1 << width for width in WIDTHS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs(pilot=False, formats=range(7), operations=(23,)):
    for fmt, width in enumerate(WIDTHS):
        if fmt not in formats:
            continue
        limit = 1 << width
        for op in operations:
            if pilot and width == 32:
                # Uniform range strata plus nontrivial near-one/range-reduction
                # clusters. These exact full-size shards count toward full work.
                starts = sorted(set(i * (1 << 27) for i in range(32)) |
                                {0x3E000000, 0x3F000000, 0x3F800000, 0x3F700000,
                                 0x41000000, 0xC1000000, 0x7F000000, 0xFF000000})
            else:
                starts = list(range(0, limit, CHUNK))
                # Full coverage with a deterministic order spread over value
                # strata, so early wall-time checkpoints see the whole range.
                random.Random(20261123).shuffle(starts)
            for start in starts:
                yield fmt, op, start, min(start + CHUNK, limit)


def key(job):
    return "%d-%d-%010d-%010d" % job


def valid_result(record, job, identity):
    fmt, op, start, end = job
    expected = dict(format=fmt, op=op, start=start, end=end,
                    checked=end - start, status="COMPLETE")
    if record.get("identity") != identity or record.get("exit") != 0 or record.get("stderr", "").strip():
        raise ValueError("identity/exit mismatch: " + key(job))
    actual = record.get("result", {})
    if json.loads(record.get("stdout", "")) != actual:
        raise ValueError("native stdout/result mismatch: " + key(job))
    if any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("incomplete or mismatched shard: " + key(job))
    if actual.get("seconds", -1) < 0 or not isinstance(actual.get("digest"), str):
        raise ValueError("missing native completion data: " + key(job))


def validate(records, identity, complete, formats=range(7), operations=(23,)):
    seen = set()
    ranges = {}
    count = 0
    for record in records:
        result = record.get("result", {})
        job = tuple(result.get(k) for k in ("format", "op", "start", "end"))
        if any(type(n) is not int for n in job):
            raise ValueError("non-integer shard identity")
        fmt, op, start, end = job
        if fmt not in formats or op not in operations:
            raise ValueError("unexpected format/op")
        if not 0 <= start < end <= (1 << WIDTHS[fmt]):
            raise ValueError("unexpected input interval")
        if job in seen:
            raise ValueError("duplicate shard: " + key(job))
        seen.add(job)
        valid_result(record, job, identity)
        ranges.setdefault((fmt, op), []).append((start, end))
        count += end - start
    for fmt in formats:
        for op in operations:
            intervals = sorted(ranges.get((fmt, op), []))
            frontier = 0
            for start, end in intervals:
                if start < frontier or (complete and start != frontier):
                    raise ValueError("overlap or missing interval: %d/%d/%d" % (fmt, op, frontier))
                frontier = end
            if complete and frontier != 1 << WIDTHS[fmt]:
                raise ValueError("missing input suffix/domain: %d/%d" % (fmt, op))
    expected = len(operations) * sum(1 << WIDTHS[fmt] for fmt in formats)
    if complete and count != expected:
        raise ValueError("wrong full cardinality")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--deadline-utc", help="absolute hard stop, ISO8601 UTC")
    parser.add_argument("--format", type=int, choices=range(7), action="append", dest="formats")
    parser.add_argument("--op", type=int, choices=(23,), action="append", dest="operations")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    formats = sorted(set(args.formats)) if args.formats else list(range(7))
    operations = sorted(set(args.operations)) if args.operations else [23]
    is_full_domain = formats == list(range(7)) and operations == [23]
    expected = len(operations) * sum(1 << WIDTHS[fmt] for fmt in formats)
    if not 1 <= args.workers <= (os.cpu_count() or 1) or args.seconds <= 0:
        parser.error("invalid worker/time allocation")
    args.out.mkdir(parents=True, exist_ok=True)
    build_receipt = json.loads((args.binary.parent / "build-receipt.json").read_text())
    if (not build_receipt.get("input_identity_equal") or
            build_receipt.get("inputs_before") != build_receipt.get("inputs_after") or
            build_receipt.get("binary_sha256") != digest(args.binary) or
            build_receipt.get("generated_c_sha256") != digest(args.binary.parent / "model.c") or
            build_receipt.get("generated_header_sha256") != digest(args.binary.parent / "model.h")):
        raise ValueError("native binary/build/source association is not established")
    current_model = {"model/" + str(path.relative_to(args.model)): digest(path)
                     for path in sorted(args.model.rglob("*")) if path.is_file()}
    pinned_model = {name: value for name, value in build_receipt["inputs_after"].items()
                    if name.startswith("model/")}
    if current_model != pinned_model:
        raise ValueError("current production model path/hash set differs from pinned build input")
    for name, value in build_receipt["inputs_after"].items():
        if name.startswith("driver/") and digest(Path(__file__).parent / name.removeprefix("driver/")) != value:
            raise ValueError("current native driver differs from pinned build input")
    manifest = {
        "scope": "production isqrt floor-root postcondition on complete bounded caller argument family; not all-N Newton/RNE/kernel proof",
        "precision_indices": [64,128,256,512,1024,2048,4096],
        "widths": list(WIDTHS), "selected_formats": formats, "ops": operations,
        "expected_evaluations": expected, "complete_release_domain": is_full_domain,
        "binary_sha256": digest(args.binary), "config_sha256": digest(args.config),
        "generated_c_sha256": digest(args.binary.parent / "model.c"),
        "compiler": (args.binary.parent / "compiler.txt").read_text(),
        "generator": (args.binary.parent / "generator.txt").read_text(),
        "model": {str(p.relative_to(args.model)): digest(p)
                  for p in sorted(args.model.rglob("*.sail"))},
        "driver": {p.name: digest(p) for p in
                   (Path(__file__).with_name("isqrt_driver.c"),
                    Path(__file__).with_name("isqrt_build.py"))},
    }
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["identity"] = identity
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("immutable source/build manifest differs; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.verify_only:
        records = [json.loads(p.read_text()) for p in sorted(args.out.glob("shard-*.json"))]
        count = validate(records, identity, complete=not args.pilot, formats=formats, operations=operations)
        status = "PILOT_ONLY" if args.pilot else "FULL_DOMAIN_COMPLETE" if is_full_domain else "SELECTED_DOMAINS_COMPLETE"
        print(json.dumps({"status": status,
                          "evaluations": count, "identity": identity,
                          "checker_sha256": digest(__file__)}), flush=True)
        return 0
    start_time = time.monotonic()
    deadline = start_time + args.seconds
    if args.deadline_utc:
        hard_end = datetime.fromisoformat(args.deadline_utc)
        if hard_end.tzinfo is None:
            raise ValueError("absolute deadline requires a timezone")
        left = (hard_end - datetime.now(timezone.utc)).total_seconds()
        deadline = min(deadline, start_time + left)
        if left <= 0:
            raise TimeoutError("absolute deadline has expired")
    chosen = list(jobs(args.pilot, formats, operations))
    existing = []
    pending = []
    stop = threading.Event()
    process_lock = threading.Lock()
    processes = set()
    for job in chosen:
        path = args.out / ("shard-" + key(job) + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            valid_result(record, job, identity)
            existing.append(record)
        else:
            pending.append(job)

    def execute(job):
        left = deadline - time.monotonic()
        if stop.is_set() or left <= 0:
            raise TimeoutError("allocation expired before shard launch")
        command = [str(args.binary.resolve()), str(args.config.resolve()), *map(str, job)]
        elapsed = time.monotonic()
        with process_lock:
            if stop.is_set():
                raise TimeoutError("campaign stopped before shard launch")
            process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.add(process)
        try:
            stdout, stderr = process.communicate(timeout=left)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        finally:
            with process_lock:
                processes.discard(process)
        record = {"identity": identity, "command": command, "exit": process.returncode,
                  "wall_seconds": time.monotonic() - elapsed,
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "stdout": stdout, "stderr": stderr}
        try:
            record["result"] = json.loads(stdout)
        except ValueError:
            record["result"] = {}
        path = args.out / ("shard-" + key(job) + ".json")
        # A failed result is retained with a separate name and cannot be resumed
        # as a completed shard or counted toward a certificate.
        if process.returncode or stderr.strip():
            path = path.with_name("failed-" + key(job) + ".json")
            path.write_text(json.dumps(record, indent=2) + "\n")
            raise ValueError("native shard failed: " + str(path))
        valid_result(record, job, identity)
        temp = path.with_suffix(".partial")
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.rename(path)
        return record

    records = list(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        try:
            for job in pending:
                futures.append(pool.submit(execute, job))
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(json.dumps({"completed": len(records), "chosen": len(chosen),
                                  "elapsed": time.monotonic() - start_time,
                                  "result": record["result"]}), flush=True)
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            with process_lock:
                for process in processes:
                    process.kill()
            raise
    count = validate(records, identity, complete=not args.pilot, formats=formats, operations=operations)
    status = "PILOT_ONLY" if args.pilot else "FULL_DOMAIN_COMPLETE" if is_full_domain else "SELECTED_DOMAINS_COMPLETE"
    summary = {"status": status,
               "identity": identity, "evaluations": count, "shards": len(records),
               "formats": formats, "ops": operations, "complete_release_domain": is_full_domain,
               "workers": args.workers, "wall_seconds": time.monotonic() - start_time,
               "checker_sha256": digest(__file__),
               "native_worker_seconds": sum(r["result"]["seconds"] for r in records),
               "remaining_requirement": "interval enclosure and correct rounding are separate obligations"}
    (args.out / ("pilot-summary.json" if args.pilot else "full-summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("numeric_cap_certificate: interrupted; owned workers stopped", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, subprocess.SubprocessError, TimeoutError) as error:
        print("numeric_cap_certificate: " + str(error), file=sys.stderr)
        sys.exit(1)
