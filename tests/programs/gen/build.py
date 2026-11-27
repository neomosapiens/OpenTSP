#!/usr/bin/env python3
"""Regenerate the checked-in program cases of tests/programs/ .

usage: build.py [--check] [--out DIR] [--profile P ...] [--seed S] [NAME ...]

  (no option)   write every case's .ots and .case.json under tests/programs/<category>/
  --check       compare instead of writing; exit 1 on any difference or on a stale checked-in file
  --out DIR     write under DIR (on-demand points: with --profile and --seed any point of the matrix)
  NAME          restrict to these case names (basic/vec_add, ...)
Same arguments, byte-identical output.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from opentsp_tools import profile as tprofile
import basic  # noqa: E402
import negative  # noqa: E402
import control  # noqa: E402
import mem  # noqa: E402
import sxm  # noqa: E402
import vxm  # noqa: E402
import mxm  # noqa: E402
import model  # noqa: E402
import kv  # noqa: E402
import model_serve  # noqa: E402
import pager  # noqa: E402
from opentsp_vllm.models.gpt2 import wide as ship_wide
from opentsp_vllm.models.gpt2 import attention as ship_attn
from opentsp_vllm.models.gpt2 import layer as ship_layer
from opentsp_vllm.models.gpt2 import head as ship_head
import division  # noqa: E402

MODULES = [basic, negative, control, mem, vxm, sxm, mxm, model, kv, model_serve, pager, ship_wide, ship_attn, ship_layer, ship_head, division]
PROGRAMS = os.path.dirname(HERE)


def registry(names=()):
    """The checked-in cases; the modules' `ON_DEMAND` cases (too large to check in) only when named."""
    for m in MODULES:
        for name, fn, profiles, kw in m.CASES + [c for c in getattr(m, "ON_DEMAND", []) if c[0] in names]:
            yield name, fn, profiles, kw


def render(case):
    return json.dumps(case, indent=1) + "\n"


def generate(names, profiles, seed):
    out = {}
    for name, fn, default_profiles, kw in registry(names):
        if names and name not in names:
            continue
        for pname in profiles or default_profiles:
            p = tprofile.load(pname)
            args = dict(kw)
            if seed is not None:
                args["seed"] = seed
            text, case = fn(p, **args)
            base = os.path.join(*name.split("/")) + f".{p.tag}"
            out[base + ".ots"] = text
            out[base + ".case.json"] = render(case)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--profile", action="append")
    ap.add_argument("--seed", type=int)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    root = a.out or PROGRAMS
    files = generate(a.names, a.profile, a.seed)
    bad = 0
    for rel, text in sorted(files.items()):
        path = os.path.join(root, rel)
        if a.check:
            old = open(path).read() if os.path.exists(path) else None
            if old != text:
                print(f"build --check: {rel} {'differs' if old is not None else 'missing'}")
                bad += 1
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
    if a.check and not a.names and not a.profile and a.seed is None and not a.out:
        cats = {rel.split(os.sep)[0] for rel in files}
        for cat in sorted(cats):
            d = os.path.join(root, cat)
            for fn in sorted(os.listdir(d) if os.path.isdir(d) else []):
                rel = os.path.join(cat, fn)
                if (fn.endswith(".ots") or fn.endswith(".case.json")) and rel not in files:
                    print(f"build --check: {rel} is stale (no generator produces it)")
                    bad += 1
    print(f"build: {len(files)} files {'checked' if a.check else 'written'}, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
