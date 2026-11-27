#!/usr/bin/env python3
"""Cross-check the binary16 rounding of the model against the solver's floating-point theory.

usage: fround_fp_theory.py PROPS.sail QUERY_DIR INVENTORY

QUERY_DIR holds the query that `sail -smt` generated for `p_fround_f16_vs_fp` (props/fptheory.sail): a property over a 15-bit significand
`m`, a 7-bit exponent `e` and a 16-bit pattern `z`. For each exponent of EXPONENTS (-40 to 20) the query is extended with
  e == that exponent   and   z == the binary16 bit pattern of m * 2^e in the solver's theory of floating point
and must be unsatisfiable: for every significand, the model's fround returns the pattern that IEEE round-to-nearest-even gives. The range
covers the subnormal numbers, the normal ones, the tie points and the overflow to infinity. Controls keep the check honest: the same query
with `z` forced to differ from the solver's pattern must be satisfiable (the property can fail), and the query without the property's
negation must be satisfiable (the extension is consistent).
"""
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import check_smt_obligations as c

NAME = "p_fround_f16_vs_fp"
# One query takes about 15 s: the exponents are the ones where the rounding changes character (the underflow to zero, the tie with the smallest
# subnormal, the subnormal range, the boundary to the normal range, the normal range, the overflow to infinity).
EXPONENTS = (-40, -30, -26, -25, -24, -23, -20, -16, -14, -10, -1, 0, 2, 20)


def power_of_two(k):
    """The real literal 2^k as SMT-LIB text."""
    return "%d.0" % (2 ** k) if k >= 0 else "(/ 1.0 %d.0)" % (2 ** -k)


def expected_pattern(m, k):
    """The binary16 bit pattern of m * 2^k in the solver's theory: exact widening, exact scaling, one rounding."""
    wide = "((_ to_fp 15 24) RNE (concat #b0 %s))" % m
    scaled = "(fp.mul RNE %s ((_ to_fp 15 24) RNE %s))" % (wide, power_of_two(k))
    return "(fp.to_ieee_bv ((_ to_fp 5 11) RNE %s))" % scaled


def parameters(query):
    """The declared constants of the property's three parameters, in order: significand, exponent, pattern."""
    found = re.findall(r"\(declare-const (zz\d+/1) \(_ BitVec (\d+)\)\)", query)
    if [width for _, width in found] != ["15", "7", "16"]:
        raise ValueError("the query of %s does not declare its parameters as bits(15), bits(7), bits(16): %s" % (NAME, found))
    return [name for name, _ in found]


def solve(text, timeout=240):
    result = subprocess.run(["z3", "-in", "-T:%d" % timeout], input=text, text=True, capture_output=True, timeout=timeout + 10)
    if result.returncode != 0 or result.stderr.strip():
        raise ValueError("solver error: " + (result.stderr.strip() or result.stdout.strip()))
    actual = result.stdout.strip()
    if actual not in {"sat", "unsat", "unknown", "timeout"}:
        raise ValueError("malformed solver output: " + repr(result.stdout))
    return actual


def split_query(query):
    """The query without its check-sat and get-model commands, and without the final assertion (the negated property)."""
    body = re.sub(r"(?m)^\(get-model\)\s*$", "", query)
    body, count = re.subn(r"(?m)^\(check-sat\)\s*$", "", body)
    if count != 1:
        raise ValueError("the query of %s has %d check-sat commands, expected 1" % (NAME, count))
    return body


def last_assertion_removed(body):
    lines = body.rstrip().split("\n")
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("(assert "):
            return "\n".join(lines[:index] + lines[index + 1:]) + "\n"
    raise ValueError("the query of %s has no assertion" % NAME)


class TheoryUnproved(ValueError):
    """A clean mismatch or undecided result, distinct from a solver/protocol failure."""


def run(directory):
    query = split_query((Path(directory) / (NAME + ".smt2")).read_text())
    m, e, z = parameters(query)
    failures = []
    for k in EXPONENTS:
        pin = "(assert (= %s (_ bv%d 7)))\n" % (e, k % 128)
        match = "(assert (= %s %s))\n" % (z, expected_pattern(m, k))
        verdict = solve(query + pin + match + "(check-sat)\n")
        print("%s e=%d: expected unsat, got %s (exit 0)" % ("PASS" if verdict == "unsat" else "FAIL", k, verdict))
        if verdict != "unsat":
            failures.append(k)
    # Controls, at one exponent in the normal range and one in the subnormal range.
    for k in (-3, -30):
        pin = "(assert (= %s (_ bv%d 7)))\n" % (e, k % 128)
        differ = "(assert (not (= %s %s)))\n" % (z, expected_pattern(m, k))
        verdict = solve(query + pin + differ + "(check-sat)\n")
        print("%s control e=%d (z differs from the solver's pattern): expected sat, got %s (exit 0)" % ("PASS" if verdict == "sat" else "FAIL", k, verdict))
        if verdict != "sat":
            failures.append("control %d" % k)
    pin = "(assert (= %s (_ bv5 7)))\n" % e
    consistent = solve(last_assertion_removed(query) + pin + "(assert (= %s %s))\n(check-sat)\n" % (z, expected_pattern(m, 5)))
    print("%s control (no negated property): expected sat, got %s (exit 0)" % ("PASS" if consistent == "sat" else "FAIL", consistent))
    if consistent != "sat":
        failures.append("consistency")
    if failures:
        raise TheoryUnproved("fround obligations or controls remain unproved: %s" % failures)


def main(argv):
    source = Path(argv[0]).read_text()
    try:
        c.check_vacuity(source)
        c.check_inventory(source, Path(argv[2]).read_text())
        run(argv[1])
    except TheoryUnproved as error:
        print("fround_fp_theory: " + str(error), file=sys.stderr)
        return 1
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("ERROR SMT fround_fp_theory: " + str(error), file=sys.stderr)
        return 1
    print("fround_fp_theory: binary16 rounding agrees with the solver's floating-point theory for %d exponents" % len(EXPONENTS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
