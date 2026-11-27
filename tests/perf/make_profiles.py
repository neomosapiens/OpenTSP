#!/usr/bin/env python3
"""Writes the fixture profiles of tests/perf/profiles/ from the `ref` profile of this tree, each with the keys it changes.

A fixture written by hand drifts from `ref` whenever the schema gains a required key; this script derives it from the current `ref`,
so the fixtures follow the schema.

usage: make_profiles.py            rewrite the fixtures
       make_profiles.py --check    exit 1 when a fixture differs from what this script writes"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(ROOT, "opentsp-arch-spec", "config", "ref.json")
OUT = os.path.join(ROOT, "tests", "perf", "profiles")

# fixture name -> {dotted key: value} changed against `ref`
FIXTURES = {
    "iw3": {"timing.mxm_iw.d_func": 3},
    "iw3mem3": {"timing.mxm_iw.d_func": 3, "timing.mem.d_func": 3},
}


def build(ref, changes):
    cfg = copy.deepcopy(ref)
    for dotted, value in changes.items():
        node = cfg
        *path, last = dotted.split(".")
        for part in path:
            node = node[part]
        if last not in node:
            raise KeyError(f"{dotted}: not a key of ref")
        node[last] = value
    return cfg


def text_of(cfg):
    return json.dumps(cfg, indent=2) + "\n"


def main(argv):
    check = argv == ["--check"]
    if argv and not check:
        print(__doc__, file=sys.stderr)
        return 2
    with open(REF) as fh:
        ref = json.load(fh)
    bad = 0
    for name, changes in FIXTURES.items():
        path = os.path.join(OUT, name + ".json")
        want = text_of(build(ref, changes))
        if check:
            have = open(path).read() if os.path.exists(path) else None
            if have != want:
                print(f"make_profiles: {path} differs from ref with {changes}", file=sys.stderr)
                bad = 1
        else:
            with open(path, "w") as fh:
                fh.write(want)
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
