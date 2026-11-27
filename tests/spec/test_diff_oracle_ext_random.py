#!/usr/bin/env python3
"""The randomized differential test emits extension instructions on extension profiles.

A random mix (no directed templates) on `ext_bf16`, `ref_ext` and `mini_ext` must include at least one of
{bf16, x128, bfp8} in the generated instruction stream. A mutant that drops extension opcodes from the random
pool leaves the set empty and fails the assertion.

usage: python3 tests/spec/test_diff_oracle_ext_random.py"""
import json
import pathlib
import random
import sys

root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "opentsp-arch-spec" / "test"))
sys.path.insert(0, str(root / "opentsp-arch-spec" / "tools"))
from oracle import diff_oracle as d  # noqa: E402
from oracle import machine as mc  # noqa: E402

CFG = root / "opentsp-arch-spec" / "config" / "test"
failures = []


def collect_random(cfg_path, n=80, seed=11):
    cfg = json.loads(pathlib.Path(cfg_path).read_text())
    m = mc.Machine(cfg)
    exts = cfg.get("extensions", [])
    rnd = random.Random(seed)
    found = set()
    for i in range(n):
        kinds = d.kinds_for(rnd, i, n, exts)
        p = d.build(m, seed * 1000003 + i, kinds)
        found |= d.extension_opcodes_in(m.isa, p.recs)
        if found:
            break
    return found, exts


def expect(label, cfg_name, want_any):
    found, exts = collect_random(CFG / cfg_name)
    if not (found & want_any):
        failures.append("%s: random mixes on extensions=%s emitted %s, want one of %s" %
                        (label, exts, sorted(found) or "(none)", sorted(want_any)))
    else:
        print("ok: %s random stream has %s" % (label, ",".join(sorted(found))))


expect("ext_bf16", "ext_bf16.json", {"bf16"})
expect("ref_ext", "ref_ext.json", {"bfp8", "x128"})
expect("mini_ext", "mini_ext.json", {"bfp8", "x128"})

# Mutant: drop extension codes from the type pool and keep mxm_ext out of the random weights.
saved_ext_codes, saved_kinds = d.ext_codes, d.kinds_for


def base_only(isa, ex):
    return [c for c in ex["codes"]
            if all(c not in codes for _e, codes in ex.get("extension_codes", {}).items())]


def kinds_no_ext(rnd, i, n, exts=()):
    names = list(d.TEMPLATES)
    weights = [d.TEMPLATES[k][1] for k in names]
    return [rnd.choices(names, weights)[0] for _ in range(rnd.choice([1, 1, 2, 3, 4]))]


d.ext_codes = base_only
d.kinds_for = kinds_no_ext
mut_found, _ = collect_random(CFG / "ref_ext.json", n=120, seed=11)
d.ext_codes, d.kinds_for = saved_ext_codes, saved_kinds
if mut_found:
    failures.append("mutant that drops extension opcodes still emitted %s" % sorted(mut_found))
else:
    print("ok: mutant that drops extension opcodes from the random pool emits none (assertion would fail)")

# Production still emits after the mutant was restored.
found, _ = collect_random(CFG / "ref_ext.json", n=80, seed=11)
if not (found & {"bfp8", "x128"}):
    failures.append("after restoring the pool, ref_ext random mixes emitted %s" % (sorted(found) or "(none)",))
else:
    print("ok: restored pool again emits %s" % ",".join(sorted(found)))

if failures:
    print("test_diff_oracle_ext_random: %d failures" % len(failures))
    for f in failures:
        print("FAIL", f)
    sys.exit(1)
print("test_diff_oracle_ext_random: ok")
