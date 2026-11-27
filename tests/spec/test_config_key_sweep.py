#!/usr/bin/env python3
"""a profile that lacks one key never crashes an engine.

For each of the profiles ref, mini, test/acc8 (accumulator depth 8) and test/c2c_wide, and for every leaf key of the file, the profile
without that key is run through tsp_sim and tsp-iss. Each must end by loading (exit 0: the key has a default) or by a refusal naming a key
(exit 2, "config"), never by a crash of the model (`Pattern match failure in set_abstract`, exit 1) or a panic (exit 101). The generated schema
lists every key the model reads (`just model-check`, `model_check.sh --check`); this is the end-to-end proof that the harness's check holds.

the same file read by the two assemblers with `--no-validate` while it lacks `mxm.acc_depth` (the key of a profile variable) is refused
by both with OTS-E-ASM-003 naming the key: no traceback from tsp_as.py, and no fallback to the lane count for the bound of an ACC index in ots-asm.

usage: TSP_SIM=<tsp_sim> [TSP_ISS=<tsp-iss>] [OTS_ASM=<ots-asm>] python3 tests/spec/test_config_key_sweep.py
"""
import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[2]
cfgdir = root / "opentsp-arch-spec/config"
binaries = [os.environ["TSP_SIM"], os.environ.get("TSP_ISS", str(root / "target/release/tsp-iss"))]
profiles = ["ref", "mini", "test/acc8", "test/c2c_wide"]
failures, runs = [], 0


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (k,))
    else:
        yield path


with tempfile.TemporaryDirectory(prefix="key_sweep_") as tmp:
    f = pathlib.Path(tmp) / "p.json"
    for name in profiles:
        base = json.loads((cfgdir / f"{name}.json").read_text())
        for path in leaves(base):
            cfg = copy.deepcopy(base)
            node = cfg
            for k in path[:-1]:
                node = node[k]
            del node[path[-1]]
            f.write_text(json.dumps(cfg))
            for binary in binaries:
                p = subprocess.run([binary, str(f)], input=b"", capture_output=True, timeout=60)
                runs += 1
                err = p.stderr.decode(errors="replace")
                crashed = p.returncode not in (0, 2) or "match failure" in err or "panicked" in err
                if p.returncode == 2 and "config" not in err:
                    crashed = True
                if crashed:
                    failures.append(f"{pathlib.Path(binary).name} on {name} without {'.'.join(path)}: exit {p.returncode} {err[:120]!r}")

ots_asm = os.environ.get("OTS_ASM", str(root / "target/release/ots-asm"))
tsp_as = str(root / "opentsp-arch-spec/tools/tsp_as.py")
with tempfile.TemporaryDirectory(prefix="key_novar_") as tmp:
    cfg = json.loads((cfgdir / "ref.json").read_text())
    del cfg["mxm"]["acc_depth"]
    f = pathlib.Path(tmp) / "noacc.json"
    f.write_text(json.dumps(cfg))
    for name, argv in (("tsp_as.py", [sys.executable, tsp_as]), ("ots-asm", [ots_asm])):
        p = subprocess.run([*argv, "insn", "--no-validate", "--config", str(f), "acc.i32 W:0, replace, 31"], capture_output=True, text=True, timeout=60)
        runs += 1
        if p.returncode == 0 or "OTS-E-ASM-003" not in p.stderr or "acc_depth" not in p.stderr or "Traceback" in p.stderr or p.stdout:
            failures.append(f"{name} --no-validate on a profile without mxm.acc_depth: exit {p.returncode} {p.stdout!r} {p.stderr[:200]!r}")

for line in failures[:20]:
    print("FAIL:", line)
if failures:
    sys.exit(1)
print(f"config key sweep: all pass ({runs} runs: every leaf key of {len(profiles)} profiles removed in turn, on {len(binaries)} engines; the assemblers without validation)")
