#!/usr/bin/env python3
"""Interleaved A/B of `tsp-iss run` binaries on the benchmark inputs of `tools/iss_bench_prep.py`.

    python3 tools/iss_ab_overhead.py [-n RUNS] [-d DIR] [-p PROGRAM ...] LABEL=BIN[:EXTRA ARGS] ...

For each run and program, every binary runs once in the given order (base, yours, base, yours, ...), with
`--profile ref --weights DIR/P.safetensors --stats=FILE DIR/P.elf` plus its extra arguments (`{tmp}` in an
extra argument is replaced by a scratch directory). Per reading it records the whole-process wall time (the
spawn to the exit, loading included), the ISS's own `wall_seconds`, cycles/s over it, us/tick, and the one-minute
load average; it prints every reading, then the best of the runs per binary and program, and each binary's best
over the first one's.

With `--bench FILE` (repeated) the programs are text files run by `tsp-iss bench --program FILE --repeat 1` (self-contained: no
bucket inputs), the metric is the `seconds` of that line, and the buckets named by `-p` (with `-d DIR`) are timed beside them by the
ISS's own `wall_seconds`. The summary is, per program, the median of the pair ratios (binary over the first binary, above 1 is slower).
`--gate RATIO` then exits 1 when ANY program's median ratio for the last binary is above RATIO (`--gate-small RATIO`, default 1.30, for
a program the first binary runs in under 5 ms), and the last line names the programs: one program's regression is not averaged away by
the others (the iss-perf-gate recipe: 7 pairs, 1.15). The overall median is printed for information only. Every reading and the summary
carry the one-minute load average.

Each side runs its own objects: `--dir LABEL=DIR` names the fixture directory of one binary (the base's objects are assembled by the
base commit's own assembler and encoding, so they run on any pinned base); `-d DIR` is the default for a label without one. The
buckets are timed as whole `run` processes without `--stats` (so the counters stay off, as in serving); the work is compared from an
untimed `--stats` run per side and bucket (`last_cycle` and `ticks`, before either leg): a program whose two sides did different work
is "not comparable", not fast or slow.

`--insn-gate RATIO` adds an instruction leg: the least of three runs per program and side of the instructions retired (`/usr/bin/time -l`;
not exact: the count moves by a few tenths of a percent with the host load, so one run per side is not enough near the limit), the verdict
being the last binary over the first. `--small NAME` (repeated) lists the programs that get the looser
threshold (a program name, not a timing, so the class does not move with the host load).

Exit codes: 0 pass; 1 a regression (a program above its threshold); 2 usage or environment; 3 an error of the base side (the first
binary failed on a program: a stale fixture or a base that cannot run it, not a regression); 4 an error of the tree side; 5 not
comparable (the two sides ran different work).
"""
import argparse
import math
import platform
import json
import statistics
import os
import shlex
import subprocess
import sys
import tempfile
import time

EXIT_REGRESSION, EXIT_SETUP, EXIT_BASE, EXIT_TREE, EXIT_NOT_COMPARABLE = 1, 2, 3, 4, 5
# the programs that get the looser threshold (the ones whose run is a few milliseconds, where timer and process noise are a large share)
SMALL_PROGRAMS = ("b1", "syncnotify")


class RunFailed(Exception):
    """A binary exited with an error on a program."""


class NotComparable(Exception):
    """The two sides did different work on a program (different simulated cycles)."""


# the simulated cycles each side reported on its last run of a program: {(program, binary): cycles}
WORK = {}


def load():
    return os.getloadavg()[0]


def run_command(argv, **kwargs):
    try:
        return subprocess.run(argv, **kwargs)
    except OSError as error:
        raise RunFailed(f'cannot execute {argv[0]}: {error}') from error


def positive_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise RunFailed('invalid benchmark seconds') from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise RunFailed('benchmark seconds must be finite and positive')
    return seconds


def one(bin_, extra, d, prog, tmp):
    stats = os.path.join(tmp, "stats.json")
    if os.path.exists(stats):
        os.unlink(stats)
    argv = [bin_, "run", "--profile", "ref", "--weights", f"{d}/{prog}.safetensors", f"--stats={stats}"]
    argv += [a.replace("{tmp}", tmp) for a in extra]
    argv.append(f"{d}/{prog}.elf")
    t0 = time.perf_counter()
    p = run_command(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    whole = time.perf_counter() - t0
    if p.returncode != 0:
        raise RunFailed(f"{argv}: rc {p.returncode}: {p.stderr.decode()[:300]}")
    try:
        with open(stats) as f:
            s = json.load(f)
    except (OSError, ValueError) as error:
        raise RunFailed('binary did not write fresh valid benchmark statistics') from error
    w = positive_seconds(s['wall_seconds'])
    if not isinstance(s.get('last_cycle'), int) or s['last_cycle'] < 0 or not isinstance(s.get('ticks'), int) or s['ticks'] <= 0:
        raise RunFailed('invalid cycle/tick counters in benchmark statistics')
    return {
        "whole": whole, "iss": w, "cps": s["last_cycle"] / w, "ust": w * 1e6 / s["ticks"], "load": load(),
        "work": (s["last_cycle"], s["ticks"]),
    }


def one_bench(bin_, prog):
    argv = [bin_, "bench", "--program", prog, "--repeat", "1"]
    p = run_command(argv, capture_output=True, text=True)
    if p.returncode != 0:
        raise RunFailed(f"{argv}: rc {p.returncode}: {p.stderr[:300]}")
    WORK[(prog, bin_)] = int(p.stdout.split("cycles=")[1].split()[0])
    return positive_seconds(p.stdout.split("seconds=")[1].split()[0])


def run_argv(bin_, extra, d, prog, tmp):
    argv = [bin_, "run", "--profile", "ref", "--weights", f"{d}/{prog}.safetensors"]
    argv += [a.replace("{tmp}", tmp) for a in extra]
    argv.append(f"{d}/{prog}.elf")
    return argv


def bucket_work(bin_, extra, d, prog, tmp):
    """The work of one untimed `--stats` run of a bucket program: (`last_cycle`, `ticks`) of the ISS's own counters. (The `E` line
    that `run` prints is `E <first-code> <first-cycle> <count>`: `E - -1 0` on every clean run, so it says nothing about the work.)"""
    return one(bin_, extra, d, prog, tmp)["work"]


def one_bucket(bin_, extra, d, prog, tmp):
    """The whole-process seconds of one `run` of a bucket program, without `--stats`: the counters are off as in serving, and any
    base binary takes the same arguments. The work of the two sides is compared by [`bucket_work`], once per program."""
    argv = run_argv(bin_, extra, d, prog, tmp)
    t0 = time.perf_counter()
    p = run_command(argv, capture_output=True, text=True)
    whole = time.perf_counter() - t0
    if p.returncode != 0:
        raise RunFailed(f"{argv}: rc {p.returncode}: {p.stderr[:300]}")
    return whole


def one_insn(bin_, extra, d, prog, bucket, tmp):
    """The instructions retired by one run of a program (a `bench` text file, or a bucket when `bucket`), from `/usr/bin/time -l`."""
    argv = [bin_, "bench", "--program", prog, "--repeat", "1"] if not bucket else run_argv(bin_, extra, d, prog, tmp)
    try:
        from opentsp_tools.host_counters import count
    except ImportError as error:
        raise RunFailed('install the checkout Python packages with just python-setup') from error
    system = platform.system()
    if system == 'Darwin':
        command = ['/usr/bin/time', '-l', *argv]
    elif system == 'Linux':
        command = [os.environ.get('OPENTSP_PERF', 'perf'), 'stat', '--no-big-num', '-x,', '-e', 'instructions', '--', *argv]
    else:
        raise RunFailed('unsupported instruction-counter platform: ' + system)
    result = run_command(command, capture_output=True, text=True, env=dict(os.environ, LC_ALL='C'))
    if result.returncode:
        raise RunFailed(f'instruction counter failed (exit {result.returncode}); verify counter tool and host permissions: {result.stderr[-500:]}')
    try:
        return count(system, result.stderr)
    except ValueError as error:
        raise RunFailed(str(error)) from error



def verdict(rows, limit, limit_small, small=SMALL_PROGRAMS):
    """The programs whose median ratio is above their threshold: `rows` maps a program to (median ratio, base median seconds).

    A program named in `small` gets the looser `limit_small`: its timer and process noise is a larger share. The class comes from the
    program list, not from a measured time, so that it does not move with the load of the host. One program over its threshold fails
    the gate, whatever the others do."""
    bad = []
    for name, (ratio, _base_s) in rows.items():
        lim = limit_small if name in small else limit
        if ratio > lim:
            bad.append((name, ratio, lim))
    return bad


SMALL_S = 0.005


def side_error(e, bins, label_of):
    """Leave with the exit code of the side that failed: a base error is not a regression."""
    which = label_of
    code = EXIT_BASE if which == bins[0][0] else EXIT_TREE
    print(f"iss-perf-gate: ERROR on the {'base' if code == EXIT_BASE else 'tree'} side ({which}): {e}", file=sys.stderr)
    if code == EXIT_BASE:
        print(
            "iss-perf-gate: the baseline could not run a program of this tree's fixture; this is not a regression. Set ISS_PERF_BASE_DIR to "
            "a source directory whose binary runs the benchmark programs, or remove its stale fixture and rerun.",
            file=sys.stderr,
        )
    sys.exit(code)


def dir_of(a, label):
    return (getattr(a, "dirs", None) or {}).get(label, a.d)


def check_work(name, bins):
    """Equal simulated cycles on both sides, when both reported them: the sides must have done the same work."""
    done = [(label, WORK.get((name, bin_))) for label, bin_, _ in bins]
    done = [(l, c) for l, c in done if c is not None]
    if len({c for _, c in done}) > 1:
        raise NotComparable(f"{name}: " + ", ".join(f"{l} ran {c} cycles" for l, c in done))


def check_buckets_work(a, bins, buckets, tmp):
    """The two sides must do the same work on each bucket (equal `last_cycle` and `ticks` of an untimed `--stats` run): it is the
    check of both legs, made before either. Leaves with exit 5 when they differ."""
    for b in buckets:
        done = []
        for label, bin_, extra in bins:
            try:
                done.append((label, bucket_work(bin_, extra, dir_of(a, label), b, tmp)))
            except RunFailed as e:
                side_error(e, bins, label)
        if len({w for _, w in done}) > 1:
            what = ", ".join(f"{l} ran {w[0]} cycles in {w[1]} ticks" for l, w in done)
            print(f"iss-perf-gate: NOT COMPARABLE: {b}: {what}: the two sides did different work, so their times and instruction counts say nothing", file=sys.stderr)
            sys.exit(EXIT_NOT_COMPARABLE)


INSN_RUNS = 3


def least_insn(run):
    """The least of three runs, for every program, short or long. The count is not exact: it moves with the host load (the pool's spinning, the
    start-up) by a few tenths of a percent on a program of a billion instructions or more (A/A of the base against itself, figures recorded
    for this gate: P1_1_64 single pairs 0.9740 to 1.0021, P8_4_128 0.9972 to 1.0004; tree against base on P8 single pairs
    1.0066 to 1.0124), which is of the order of the 1.01 limit; the least of three keeps the work and drops most of the noise."""
    return min(run() for _ in range(INSN_RUNS))


def insn_leg(a, bins, names, cases):
    """The least of three runs per program and side of the instructions retired; the tree over the base. Returns the programs above the limit."""
    limit = a.insn_gate
    bad, rows = [], {}
    with tempfile.TemporaryDirectory(prefix="iss-insn-") as tmp:
        for name, prog, bucket in cases:
            counts = {}
            for label, bin_, extra in bins:
                try:
                    counts[label] = least_insn(lambda: one_insn(bin_, extra, dir_of(a, label), prog, bucket, tmp))
                except RunFailed as e:
                    side_error(e, bins, label)
            rows[name] = counts
    last = bins[-1][0]
    print(f"\ninstructions retired, the least of three runs per program and side (ratio {last} / {bins[0][0]}; above 1 is more work; limit {limit}):")
    for name, counts in rows.items():
        r = counts[last] / counts[bins[0][0]]
        print("  " + f"{name:18s} {r:.4f}x  (" + ", ".join(f"{l} {c / 1e6:.1f} M" for l, c in counts.items()) + ")")
        if r > limit:
            bad.append((name, r))
    return bad


def bench_mode(a, bins):
    """Interleaved A/B of `tsp-iss bench` runs (the programs of `--bench`) and of `run` on the buckets of `-p`; the verdict of the gate."""
    progs = list(a.bench or [])
    buckets = list(a.progs or []) if a.bench else []
    names = [os.path.basename(p).removesuffix(".txt") for p in progs] + buckets
    small = tuple(a.small) if getattr(a, "small", None) else SMALL_PROGRAMS
    ratios = {n: [] for n in names}
    times = {(n, label): [] for n in names for label, _, _ in bins}
    loads = []
    cases = [(os.path.basename(p).removesuffix(".txt"), p, None) for p in progs] + [(b, b, b) for b in buckets]
    insn_bad = []
    with tempfile.TemporaryDirectory(prefix="iss-work-") as wtmp:
        check_buckets_work(a, bins, buckets, wtmp)
    if getattr(a, "insn_gate", None) is not None:
        insn_bad = insn_leg(a, bins, names, cases)
        if getattr(a, "insn_only", False):
            last = bins[-1][0]
            if insn_bad:
                what = ", ".join(f"{n} {r:.4f}x" for n, r in insn_bad)
                print(f"iss-perf-gate: FAIL: {last} retires more instructions than {bins[0][0]} on {what} (limit {a.insn_gate})")
                sys.exit(EXIT_REGRESSION)
            print(f"iss-perf-gate: ok: no program of {len(names)} retires more than {a.insn_gate}x the instructions of {bins[0][0]}")
            return
    with tempfile.TemporaryDirectory(prefix="iss-ab-") as tmp:
        for r in range(1, a.n + 1):
            for name, prog, bucket in cases:
                row = {}
                for label, bin_, extra in bins:
                    try:
                        row[label] = (
                            one_bench(bin_, prog) if bucket is None else one_bucket(bin_, extra, dir_of(a, label), bucket, tmp)
                        )
                    except RunFailed as e:
                        side_error(e, bins, label)
                    loads.append(load())
                    times[(name, label)].append(row[label])
                try:
                    check_work(prog if bucket is None else bucket, bins)
                except NotComparable as e:
                    print(f"iss-perf-gate: NOT COMPARABLE: {e}: the two sides did different work, so their times say nothing", file=sys.stderr)
                    sys.exit(EXIT_NOT_COMPARABLE)
                base = row[bins[0][0]]
                for label, _, _ in bins[1:]:
                    ratios[name].append(row[label] / base)
                print(
                    f"pair {r} {name}: " + ", ".join(f"{label} {row[label] * 1e3:.2f} ms" for label, _, _ in bins) + f", load {loads[-1]:.2f}",
                    flush=True,
                )
    last = bins[-1][0]
    print(f"\nmedian of {a.n} pairs per program (ratio {last} / {bins[0][0]}; above 1 is slower):")
    rows = {}
    for n in names:
        t0, t1 = times[(n, bins[0][0])], times[(n, last)]
        # the ratios of the last binary: with more than two binaries the pairs are per binary, take the ones of the last
        rn = ratios[n][len(bins) - 2 :: len(bins) - 1] if len(bins) > 2 else ratios[n]
        rows[n] = (statistics.median(rn), statistics.median(t0))
        print(f"  {n:18s} {rows[n][0]:.3f}x  ({bins[0][0]} {statistics.median(t0) * 1e3:.2f} ms, {last} {statistics.median(t1) * 1e3:.2f} ms)")
    allr = [x for n in names for x in ratios[n]]
    print(f"overall median ratio {statistics.median(allr):.3f}x over {len(allr)} pairs (information only); load average {min(loads):.1f}-{max(loads):.1f} (median {statistics.median(loads):.1f})")
    if a.gate is not None or insn_bad:
        bad = verdict(rows, a.gate, getattr(a, "gate_small", None) or a.gate, small) if a.gate is not None else []
        if insn_bad:
            what = ", ".join(f"{n} {r:.4f}x" for n, r in insn_bad)
            print(f"iss-perf-gate: FAIL: {last} retires more instructions than {bins[0][0]} on {what} (limit {a.insn_gate})")
        if bad:
            what = ", ".join(f"{n} {r:.3f}x (limit {lim})" for n, r, lim in bad)
            print(f"iss-perf-gate: FAIL: {last} is slower than {bins[0][0]} on {what}")
        if bad or insn_bad:
            sys.exit(EXIT_REGRESSION)
        print(f"iss-perf-gate: ok: no program of {len(names)} is above its threshold ({a.gate}; {a.gate_small} for {', '.join(small)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("-d", default="/tmp/iss-bench")
    ap.add_argument("-p", action="append", dest="progs")
    ap.add_argument("--bench", action="append", metavar="FILE", help="a benchmark program file for `tsp-iss bench` instead of buckets (repeat)")
    ap.add_argument("--gate", type=float, help="with --bench: fail when the median ratio of any one program of the last binary is above this")
    ap.add_argument("--gate-small", type=float, default=1.30, help="the threshold of the programs the base runs in under 5 ms (default 1.30)")
    ap.add_argument("--dir", action="append", default=[], metavar="LABEL=DIR", help="the fixture directory of one binary (default -d)")
    ap.add_argument("--insn-gate", type=float, help="also compare the instructions retired (one run per program and side); fail above this ratio")
    ap.add_argument("--insn-only", action="store_true", help="with --insn-gate: run the instruction leg alone (no timed pairs)")
    ap.add_argument("--small", action="append", metavar="NAME", help="a program that gets the looser threshold (repeat; default %s)" % ", ".join(SMALL_PROGRAMS))
    ap.add_argument("bins", nargs="+")
    a = ap.parse_args()
    if a.n < 1:
        ap.error('-n must be positive')
    for name in ('gate', 'gate_small', 'insn_gate'):
        value = getattr(a, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            ap.error('--' + name.replace('_', '-') + ' must be finite and positive')
    try:
        a.dirs = dict(d.split('=', 1) for d in a.dir)
    except ValueError:
        ap.error('--dir requires LABEL=DIR')
    progs = a.progs or ["P1_1_64", "P8_4_128"]
    if a.insn_only and a.insn_gate is None:
        ap.error("--insn-only needs --insn-gate")
    if a.gate is not None and not a.bench:
        ap.error("--gate needs --bench programs (the buckets of -p are timed beside them)")
    bins = []
    try:
        for spec in a.bins:
            label, rest = spec.split('=', 1)
            parts = shlex.split(rest)
            if not label or not parts:
                raise ValueError('empty binary label or command')
            bins.append((label, parts[0], parts[1:]))
    except ValueError:
        ap.error('each binary requires a nonempty LABEL=COMMAND')
    labels = [label for label, _, _ in bins]
    if len(set(labels)) != len(labels):
        ap.error('binary labels must be unique')
    if set(a.dirs) - set(labels):
        ap.error('--dir names an unknown binary label')
    if (a.gate is not None or a.insn_gate is not None) and len(bins) < 2:
        ap.error('comparison gates need at least two binaries')
    if a.insn_gate is not None and not a.bench:
        ap.error('--insn-gate needs --bench programs')
    try:
        return run_modes(a, bins, progs)
    except RunFailed as e:
        print(f"iss_ab_overhead: {e}", file=sys.stderr)
        sys.exit(EXIT_SETUP)


def run_modes(a, bins, progs):
    if a.bench:
        return bench_mode(a, bins)
    best = {}
    with tempfile.TemporaryDirectory(prefix="iss-ab-") as tmp:
        check_buckets_work(a, bins, progs, tmp)
        for r in range(1, a.n + 1):
            for prog in progs:
                for label, bin_, extra in bins:
                    m = one(bin_, extra, dir_of(a, label), prog, tmp)
                    print(
                        f"run {r} {prog} {label}: whole {m['whole'] * 1e3:.1f} ms, iss {m['iss'] * 1e3:.1f} ms, "
                        f"{m['cps']:,.0f} cycles/s, {m['ust']:.3f} us/tick, load {m['load']:.2f}",
                        flush=True,
                    )
                    k = (prog, label)
                    b = best.setdefault(k, {"whole": 9e9, "iss": 9e9, "cps": 0.0, "ust": 9e9, "loads": []})
                    for f, fn in (("whole", min), ("iss", min), ("cps", max), ("ust", min)):
                        b[f] = fn(b[f], m[f])
                    b["loads"].append(m["load"])
    print("\nbest of", a.n)
    for prog in progs:
        base = best[(prog, bins[0][0])]
        for label, _, _ in bins:
            b = best[(prog, label)]
            print(
                f"{prog} {label}: whole {b['whole'] * 1e3:.1f} ms ({base['whole'] / b['whole']:.3f}x), "
                f"iss {b['iss'] * 1e3:.1f} ms, {b['cps']:,.0f} cycles/s ({b['cps'] / base['cps']:.3f}x), "
                f"{b['ust']:.3f} us/tick, load {min(b['loads']):.1f}-{max(b['loads']):.1f}"
            )


if __name__ == "__main__":
    main()
