#!/usr/bin/env python3
"""Exact structural checks of the instruction encodings .

For every entry of the catalogue (isa-db/generated/encoding.json) and not on a sample:

  1. the match pattern is as long as the entry's width;
  2. every variable field is a contiguous bit range (`msb-lsb`, msb >= lsb), inside the word; a split
     field has a second contiguous range (`hi_location`);
  3. fields do not overlap, and the wildcard bits of the match pattern are exactly the field bits
     (no unassigned wildcard, no field on a fixed bit);
  4. any two entries of the same width have match patterns that are disjoint on their fixed bits (some
     bit is fixed in both, to different values); a pair whose patterns do overlap is listed, and must
     be separated by field legality (the other entry's fixed bits over a whole field lie outside that
     field's `values` list), which is the only remaining way the decode rule leaves one entry.

With `--smt` it also runs one per-clause query: that the 64-bit match patterns of two entries are
disjoint, encoded as a Z3 bit-vector problem, as a record of whether per-clause SMT queries finish.

Usage: encoding_exact.py [encoding.json] [--smt]
"""
from pathlib import Path
import itertools
import json
import subprocess
import sys

SPEC = Path(__file__).resolve().parents[2]
DEFAULT = SPEC / "isa-db/generated/encoding.json"


def span(text):
    """`msb-lsb` or `n` as (msb, lsb); a malformed range raises ValueError."""
    high, _, low = str(text).partition("-")
    msb, lsb = int(high), int(low or high)
    return msb, lsb


def field_ranges(variable):
    ranges = [("location", span(variable["location"]))]
    for key in ("hi_location", "top_location"):
        if key in variable:
            ranges.append((key, span(variable[key])))
    return ranges


def check_entry(entry):
    """Problems of one entry (empty when it is exact)."""
    name, width, match = entry["name"], entry["width"], entry["match"]
    problems = []
    if len(match) != width or set(match) - set("01-"):
        return ["%s: match pattern has length %d (width %d) or a character outside 0, 1, -" % (name, len(match), width)]
    wildcards = {width - 1 - index for index, char in enumerate(match) if char == "-"}
    owner = {}
    for variable in entry["variables"]:
        for key, (msb, lsb) in field_ranges(variable):
            if msb < lsb or lsb < 0 or msb >= width:
                problems.append("%s.%s: %s %d-%d is not a contiguous range inside %d bits" %
                                (name, variable["name"], key, msb, lsb, width))
                continue
            for bit in range(lsb, msb + 1):
                if bit in owner:
                    problems.append("%s: bit %d belongs to both %s and %s" % (name, bit, owner[bit], variable["name"]))
                owner[bit] = variable["name"]
    unassigned = sorted(wildcards - set(owner))
    on_fixed = sorted(set(owner) - wildcards)
    if unassigned:
        problems.append("%s: wildcard bits %s belong to no field" % (name, unassigned))
    if on_fixed:
        problems.append("%s: field bits %s are fixed in the match pattern" % (name, on_fixed))
    return problems


def overlap(a, b):
    """True when some word matches both patterns."""
    return all(x == "-" or y == "-" or x == y for x, y in zip(a["match"], b["match"]))


def fixed_value(entry, msb, lsb):
    """The fixed value of bits msb..lsb, or None when any of them is a wildcard."""
    chars = entry["match"][entry["width"] - 1 - msb: entry["width"] - lsb]
    return None if "-" in chars else int(chars, 2)


def separated_by_field(a, b):
    """The (variable, value) that keeps the overlapping entries `a` and `b` apart, else None."""
    for x, y in ((a, b), (b, a)):
        for variable in x["variables"]:
            if "values" not in variable:
                continue
            value, shift = 0, 0
            for _, (msb, lsb) in field_ranges(variable):
                part = fixed_value(y, msb, lsb)
                if part is None:
                    value = None
                    break
                value |= part << shift
                shift += msb - lsb + 1
            if value is not None and value not in variable["values"]:
                return "%s.%s lists %s, %s fixes %d" % (x["name"], variable["name"], variable["values"], y["name"], value)
    return None


def check(catalogue):
    """(problems, resolved) over the whole catalogue; `resolved` lists the overlaps that field legality separates."""
    entries = catalogue["instructions"]
    problems = [p for entry in entries for p in check_entry(entry)]
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        problems.append("duplicate entry names")
    resolved = []
    for a, b in itertools.combinations(entries, 2):
        if a["width"] == b["width"] and "match" in a and len(a["match"]) == a["width"] and len(b["match"]) == b["width"] \
                and overlap(a, b):
            how = separated_by_field(a, b)
            if how is None:
                problems.append("%s and %s: match patterns overlap and no field separates them" % (a["name"], b["name"]))
            else:
                resolved.append("%s / %s: %s" % (a["name"], b["name"], how))
    return problems, resolved


def smt_pair_query(a, b):
    """Z3 query that some 64-bit word matches both patterns; `unsat` means the patterns are disjoint."""
    def mask(entry):
        m = entry["match"]
        return int("".join("0" if c == "-" else "1" for c in m), 2), int("".join("1" if c == "1" else "0" for c in m), 2)
    (ma, va), (mb, vb) = mask(a), mask(b)
    n = a["width"]
    return ("(declare-const w (_ BitVec %d))\n(assert (= (bvand w (_ bv%d %d)) (_ bv%d %d)))\n"
            "(assert (= (bvand w (_ bv%d %d)) (_ bv%d %d)))\n(check-sat)\n" % (n, ma, n, va, n, mb, n, vb, n))


def main(argv):
    flags = [a for a in argv if a.startswith("--")]
    paths = [a for a in argv if not a.startswith("--")]
    source = Path(paths[0] if paths else DEFAULT)
    try:
        catalogue = json.loads(source.read_text())
        problems, resolved = check(catalogue)
        entries = catalogue["instructions"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("encoding_exact: cannot check %s: %s: %s" % (source, type(error).__name__, error), file=sys.stderr)
        return 2
    pairs = sum(1 for a, b in itertools.combinations(entries, 2) if a["width"] == b["width"])
    print("encoding_exact: %d entries, %d same-width pairs checked exactly, %d overlapping pair(s) separated by field legality" %
          (len(entries), pairs, len(resolved)))
    for line in resolved:
        print("  resolved: " + line)
    if "--smt" in flags:
        wide = [e for e in entries if e["width"] == 64][:2]
        query = smt_pair_query(*wide)
        result = subprocess.run(["z3", "-in", "-T:60"], input=query, text=True, capture_output=True)
        print("encoding_exact: SMT per-clause query (%s vs %s, 64-bit): %s" % (wide[0]["name"], wide[1]["name"],
                                                                               result.stdout.strip() or result.stderr.strip()))
    for line in problems:
        print("encoding_exact: " + line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
