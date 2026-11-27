#!/usr/bin/env python3
"""Check the simulator-lint registry.

A lint is registered by ONE line in model/core/lints.sail (`let LINT_<NAME> : string = "OTS-W-<AREA>-<nnn>"`);
this check keeps the rest in step:
  - every code has the form OTS-W-<AREA>-<nnn> and is defined once;
  - every defined lint is used by a `lint(q, c, LINT_<NAME>)` call in the model (outside lints.sail);
  - no model file calls lint() with a string literal or an undefined name;
  - every code is a row of the lint table of validation.md ("Simulator lints"), and every row of that table is defined;
  - no lint code is in the diagnostics of encoding.json (a lint is not a diagnostic) and the oracle
    (test/oracle/machine.py LINT) defines exactly the same codes.

usage: check_lints.py
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.normpath(os.path.join(SPEC, ".."))
MODEL = os.path.join(SPEC, "model")
VALIDATION = os.path.join(ROOT, "opentsp-arch-spec", "docs", "validation.md")
CODE = re.compile(r"OTS-W-[A-Z0-9]+-[0-9]{3}")
errs = []


def err(m):
    errs.append(m)


def main():
    src = open(os.path.join(MODEL, "core", "lints.sail")).read()
    defs = re.findall(r'^let\s+(LINT_[A-Z0-9_]+)\s*:\s*string\s*=\s*"([^"]*)"', src, re.M)
    if not defs:
        err("no lint defined in model/core/lints.sail")
    names, codes = {}, {}
    for n, c in defs:
        if not re.fullmatch(CODE.pattern, c):
            err(f"{n}: code {c!r} is not OTS-W-<AREA>-<nnn>")
        if n in names:
            err(f"{n} is defined twice")
        if c in codes:
            err(f"code {c} is defined by {codes[c]} and {n}")
        names[n], codes[c] = c, n
    used, bad = set(), []
    for dp, _, fs in os.walk(MODEL):
        for f in fs:
            if not f.endswith(".sail") or f == "lints.sail":
                continue
            for call in re.findall(r"\blint\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", open(os.path.join(dp, f)).read()):
                last = call.split(",")[-1].strip()
                if last in names:
                    used.add(last)
                else:
                    bad.append(f"{f}: lint(..., {last}) is not a lint of lints.sail")
    for b in bad:
        err(b)
    for n in names:
        if n not in used:
            err(f"{n} is defined but no lint(q, c, {n}) call uses it")
    # validation.md: the lint table
    doc = open(VALIDATION).read()
    sec = doc.split("Simulator lints", 1)
    rows = set()
    if len(sec) < 2:
        err("validation.md has no 'Simulator lints' section")
    else:
        body = sec[-1]
        rows = set(re.findall(r"^\|\s*(OTS-W-[A-Z0-9]+-[0-9]{3})\s*\|", body, re.M))
    for c in codes:
        if c not in rows:
            err(f"{c} is not a row of the lint table of validation.md")
    for c in rows:
        if c not in codes:
            err(f"validation.md lists {c}, which lints.sail does not define")
    # a lint is not a diagnostic
    enc = json.load(open(os.path.join(SPEC, "isa-db", "generated", "encoding.json")))
    for c in enc["diagnostics"]:
        if c.startswith("OTS-W-"):
            err(f"encoding.json diagnostics lists the lint {c}")
    # the oracle
    oracle = open(os.path.join(SPEC, "test", "oracle", "machine.py")).read()
    mm = re.search(r"^LINT = dict\(\n(?:[ \t]+.*\n)+", oracle, re.M)     # the statement: indented lines
    ocodes = set(CODE.findall(mm.group(0))) if mm else set()
    if ocodes != set(codes):
        err(f"the oracle LINT table {sorted(ocodes)} differs from lints.sail {sorted(codes)}")
    for e in errs:
        print("FAIL", e)
    if errs:
        return 1
    print(f"check_lints: {len(codes)} lints ({', '.join(sorted(codes))}) registered, used, documented and in the oracle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
