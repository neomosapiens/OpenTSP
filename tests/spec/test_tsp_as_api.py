#!/usr/bin/env python3
"""The reference assembler's Python API refuses a profile beyond the caps as its command line does.

`tsp_as.ISA(name, config)` used to return an ISA for a profile file that `tsp_as.py insn --config` refuses (the geometry caps of
`config/schema.json`: 1 to 32 superlanes, the words per slice with their bank bit, the MEM slices, the MXM rows), so a script that
built the ISA directly sized the grammar and the memory from a profile that no harness loads. It raises `ConfigError` with the message
of the command line now; the command line with `--no-validate` still skips the check.

usage: python3 tests/spec/test_tsp_as_api.py"""
import json
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "opentsp-arch-spec/tools"))
import tsp_as  # noqa: E402

ref = json.loads((root / "opentsp-arch-spec/config/ref.json").read_text())
failures = []


def profile(tmp, name, edit):
    cfg = json.loads(json.dumps(ref))
    edit(cfg)
    path = pathlib.Path(tmp) / f"{name}.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def _prog(name):
    prog = tsp_as.Program()
    prog.profile = name
    return prog


def expect_refused(label, path, key):
    try:
        tsp_as.ISA("ref", path)
    except tsp_as.ConfigError as e:
        if key not in str(e):
            failures.append(f"{label}: refused without naming {key}: {e}")
        return str(e)
    failures.append(f"{label}: ISA() accepted it")
    return ""


with tempfile.TemporaryDirectory(prefix="tsp_as_api_") as tmp:
    big = profile(tmp, "words", lambda c: c["sram"].update(words_per_slice=16384))
    msg = expect_refused("16384 words per slice", big, "sram.")
    cli = subprocess.run([sys.executable, str(root / "opentsp-arch-spec/tools/tsp_as.py"), "insn", "--config", big, "step 0x20"],
                         capture_output=True, text=True)
    if cli.returncode != 2 or msg.split(": ", 1)[-1] not in cli.stderr:
        failures.append(f"the command line and the API disagree on 16384 words: exit {cli.returncode} {cli.stderr!r} vs {msg!r}")
    expect_refused("33 superlanes", profile(tmp, "sl33", lambda c: c["geometry"].update(superlanes=33)), "geometry.superlanes")
    expect_refused("640 MXM rows", profile(tmp, "rows", lambda c: c["mxm"].update(rows=640)), "mxm.rows")
    # the whole validation, not only the caps, on every entry of the import surface (ISA, assemble, write_object)
    bad = {
        "unknown extension": (profile(tmp, "ext", lambda c: c.update(extensions=["bf16", "nonesuch"])), "extensions"),
        "repeated extension": (profile(tmp, "ext2", lambda c: c.update(extensions=["bf16", "bf16"])), "extensions"),
        "wide_acc without acc_bits 128": (profile(tmp, "wide", lambda c: c.update(extensions=c["extensions"] + ["wide_acc"])), "wide_acc"),
        "acc_bits 128 without wide_acc": (profile(tmp, "acc128", lambda c: c["mxm"].update(acc_bits=128)), "acc_bits"),
        "acc_bits 64": (profile(tmp, "acc64", lambda c: c["mxm"].update(acc_bits=64)), "acc_bits"),
        "d_skew not below d_func": (profile(tmp, "skew", lambda c: c["timing"]["vxm"].update(d_skew=99)), "d_skew"),
    }
    for label, (path, key) in bad.items():
        if path is None:
            continue
        cli = subprocess.run([sys.executable, str(root / "opentsp-arch-spec/tools/tsp_as.py"), "insn", "--config", path, "step 0x20"],
                             capture_output=True, text=True)
        if cli.returncode != 2:
            failures.append(f"{label}: the command line accepted it (exit {cli.returncode}); the test profile is not refused")
            continue
        for what, call in (("ISA", lambda: tsp_as.ISA("ref", path)),
                           ("assemble", lambda: tsp_as.assemble(".arch opentsp-tsa\n.profile ref\n", "x.tsp", path)),
                           ("write_object", lambda: tsp_as.write_object(_prog("ref"), path))):
            try:
                call()
            except tsp_as.ConfigError as e:
                if key and key not in str(e):
                    failures.append(f"{label}: {what} refused without naming {key}: {e}")
            except Exception as e:  # noqa: BLE001
                failures.append(f"{label}: {what} raised {type(e).__name__}: {e}, not ConfigError")
            else:
                failures.append(f"{label}: {what} accepted it")
    # the shipped profiles inside the caps are accepted: one, eight and thirty-two superlanes
    for name, n in (("mini", 1), ("test/sl8", 8), ("test/sl32", 32)):
        isa = tsp_as.ISA(pathlib.Path(name).name, str(root / f"opentsp-arch-spec/config/{name}.json"))
        if isa.vlanes != 16 * n:
            failures.append(f"{name}: vlanes {isa.vlanes}")
    # --no-validate skips the check, as it skips the full validation
    skip = subprocess.run([sys.executable, str(root / "opentsp-arch-spec/tools/tsp_as.py"), "insn", "--no-validate", "--config", big, "step 0x20"],
                          capture_output=True, text=True)
    if skip.returncode != 0:
        failures.append(f"--no-validate did not skip the check: exit {skip.returncode} {skip.stderr!r}")

for f in failures:
    print("FAIL", f)
print(f"tsp_as api: {len(failures)} failures")
sys.exit(1 if failures else 0)
