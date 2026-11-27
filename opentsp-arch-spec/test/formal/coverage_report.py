#!/usr/bin/env python3
"""Merge the coverage files of many runs against the static file of the instrumented model .

`sail -c --c-coverage STATIC` lists every function (F), branch (B) and branch target (T) of the model with its source
span; test/formal/coverage_shim.c writes the ones a process executed, in the same line format. A function is covered when
some run entered it, a branch target when some run took it. Entries match by source span, so runs of differently
optimised builds of the same model (the unit tests, the simulator) merge. Prelude and library files (absolute paths),
the props module and the unit tests are not counted.

Usage: coverage_report.py STATIC RUN_FILE... [--missed FILE]
"""
from pathlib import Path
import collections
import re
import sys



def parse(text):
    """Yield (kind, name, file, span) for each line; F carries a name, T carries the target id in the span key."""
    for line in text.splitlines():
        if not line.strip():
            continue
        kind = line[0]
        if kind == "F":
            m = re.match(r'F (\d+), "([^"]*)", "([^"]*)", (\d+), (\d+), (\d+), (\d+)$', line)
            if m:
                yield "F", m.group(2), m.group(3), tuple(int(x) for x in m.groups()[3:])
        elif kind == "B":
            m = re.match(r'B (\d+), "([^"]*)", (\d+), (\d+), (\d+), (\d+)$', line)
            if m:
                yield "B", "", m.group(2), tuple(int(x) for x in m.groups()[2:])
        elif kind == "T":
            m = re.match(r'T (\d+), (\d+), "([^"]*)", (\d+), (\d+), (\d+), (\d+)$', line)
            if m:
                yield "T", "", m.group(3), tuple(int(x) for x in m.groups()[3:])
        else:
            raise ValueError("unrecognised coverage line: " + line)


def counted(file):
    return bool(file) and not file.startswith("/") and file.split("/")[0] not in ("props", "unit_tests")


def area(file):
    head = file.split("/")
    return "/".join(head[:2]) if head[0] == "slices" and len(head) > 2 else head[0]


def merge(static_text, run_texts):
    """(totals, covered, missed functions): per area counts of F and T entries."""
    # Several static files may be concatenated (one per build); an entry counts once however many list it.
    universe = sorted({e for e in parse(static_text) if counted(e[2])})
    seen = set()
    for text in run_texts:
        for kind, name, file, span in parse(text):
            seen.add((kind, name, file, span))
    totals, covered = collections.Counter(), collections.Counter()
    missed = []
    for kind, name, file, span in universe:
        if kind == "B":
            continue
        key = (area(file), kind)
        totals[key] += 1
        if (kind, name, file, span) in seen:
            covered[key] += 1
        elif kind == "F":
            missed.append("%s\t%s:%d" % (name, file, span[0]))
    return totals, covered, sorted(missed)


def report(totals, covered):
    lines = []
    for kind, label in (("F", "functions entered"), ("T", "branch targets taken")):
        total = sum(v for (a, k), v in totals.items() if k == kind)
        hit = sum(v for (a, k), v in covered.items() if k == kind)
        lines.append("coverage: %s %d of %d (%.1f%%)" % (label, hit, total, 100.0 * hit / total if total else 0.0))
    for a in sorted({a for a, _ in totals}):
        f, t = totals[(a, "F")], totals[(a, "T")]
        lines.append("  %-14s functions %4d of %4d   branch targets %5d of %5d" %
                     (a, covered[(a, "F")], f, covered[(a, "T")], t))
    return "\n".join(lines)


def main(argv):
    missed_file = None
    if "--missed" in argv:
        index = argv.index("--missed")
        missed_file = argv[index + 1]
        argv = argv[:index] + argv[index + 2:]
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    static = Path(argv[0]).read_text()
    runs = [Path(p).read_text() for p in argv[1:]]
    totals, covered, missed = merge(static, runs)
    print(report(totals, covered))
    print("coverage: %d run file(s) merged" % len(runs))
    if missed_file:
        Path(missed_file).write_text("".join(m + "\n" for m in missed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
