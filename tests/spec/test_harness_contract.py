#!/usr/bin/env python3
"""The C harness (TSP_SIM) and the Rust ISS (TSP_ISS) agree on three harness edges.

1. Every configuration in config_corpus/ is rejected (exit 2) by both.
2. An unreadable standard input (a directory) exits 2 with "cannot read standard input".
3. A profile without `ras.diag_depth` loads on both engines and on `ots-asm insn --config` and behaves as one with the key at its
   default 16 (the same output); `diag_depth` 0 and 65 are still refused by all three, naming the key.
4. A same-cycle C2C conflict under a legal timing override (Transmit d_skew 2, Send d_skew 0):
   the delayed Transmit of queue 127 and the immediate Send of queue 126 execute in the same
   cycle; the delayed one reaches the link first, so queue 126 is the one rejected
   (microarchitecture.md, "Same-cycle link order").

4. A timing row keyed by an element type (`<entry>.<type>`, microarchitecture.md section 1) is accepted when its entry takes
   the type and refused with the same message by the three implementations of the timing rules (tsp_sim, tsp-iss, tsp_as.py).

5. A profile whose `extensions` list or `mxm.acc_bits` is refused (an unknown or repeated extension, a width other than 32 or 128, wide_acc
   without 128 bits, 128 bits without wide_acc) is refused by tsp_sim, tsp-iss and tsp_as.py with the model's message (a width other than 32 or 128 is named by the key, `mxm.acc_bits:`).

usage: TSP_SIM=<tsp_sim> [TSP_ISS=<tsp-iss>] [OTS_ASM=<ots-asm>] python3 tests/spec/test_harness_contract.py
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

here = pathlib.Path(__file__).resolve().parent
root = here.parents[1]
mini = root / "opentsp-arch-spec/config/mini.json"
binaries = [os.environ["TSP_SIM"], os.environ.get("TSP_ISS", str(root / "target/release/tsp-iss"))]
failures = []


def run(binary, args, stdin=b""):
    kw = {"stdin": stdin} if isinstance(stdin, int) else {"input": stdin}
    return subprocess.run([binary, *map(str, args)], capture_output=True, timeout=30, **kw)


corpus = sorted((here / "config_corpus").glob("*.json"))
assert corpus, "empty corpus"
for binary in binaries:
    for f in corpus:
        p = run(binary, [f])
        if p.returncode != 2 or b"config" not in p.stderr:
            failures.append(f"{pathlib.Path(binary).name} accepts {f.name}: exit {p.returncode} {p.stdout!r} {p.stderr!r}")
    p = run(binary, [mini])
    if p.returncode != 0:
        failures.append(f"{pathlib.Path(binary).name} rejects the shipped mini profile: {p.stderr!r}")

for binary in binaries:
    fd = os.open(here, os.O_RDONLY)
    try:
        p = run(binary, [mini], stdin=fd)
    finally:
        os.close(fd)
    if p.returncode != 2 or b"cannot read standard input" not in p.stderr:
        failures.append(f"{pathlib.Path(binary).name} with a directory as stdin: exit {p.returncode} {p.stdout!r} {p.stderr!r}")

ots_asm = os.environ.get("OTS_ASM", str(root / "target/release/ots-asm"))
mini_cfg = json.loads(mini.read_text())
assert mini_cfg["ras"]["diag_depth"] == 16, "the shipped mini profile carries the default depth"
with tempfile.TemporaryDirectory(prefix="diag_depth_") as tmp:
    def profile(name, edit):
        cfg = json.loads(json.dumps(mini_cfg))
        edit(cfg["ras"])
        path = pathlib.Path(tmp) / f"{name}.json"
        path.write_text(json.dumps(cfg))
        return path

    absent = profile("absent", lambda r: r.pop("diag_depth"))
    for binary in binaries:
        name = pathlib.Path(binary).name
        a, b = run(binary, [absent, "--trace"]), run(binary, [mini, "--trace"])
        if a.returncode != 0 or a.stdout != b.stdout:
            failures.append(f"{name} on a profile without ras.diag_depth: exit {a.returncode} {a.stdout!r} {a.stderr!r}")
        for depth in (0, 65):
            p = run(binary, [profile(f"d{depth}", lambda r, d=depth: r.update(diag_depth=d)), "--trace"])
            if p.returncode != 2 or b"ras.diag_depth" not in p.stderr:
                failures.append(f"{name} accepts ras.diag_depth {depth}: exit {p.returncode} {p.stderr!r}")
    a = run(ots_asm, ["insn", "--config", absent, "step 0x20"])
    b = run(ots_asm, ["insn", "--config", mini, "step 0x20"])
    if a.returncode != 0 or a.stdout != b.stdout or not a.stdout:
        failures.append(f"ots-asm insn --config on a profile without ras.diag_depth: exit {a.returncode} {a.stdout!r} {a.stderr!r}")
    for depth in (0, 65):
        p = run(ots_asm, ["insn", "--config", profile(f"a{depth}", lambda r, d=depth: r.update(diag_depth=d)), "step 0x20"])
        if p.returncode != 2 or b"ras.diag_depth" not in p.stderr:
            failures.append(f"ots-asm accepts ras.diag_depth {depth}: exit {p.returncode} {p.stderr!r}")

c2c = here / "c2c"
for binary in binaries:
    p = run(binary, [root / "opentsp-arch-spec/config/test/c2c_tx_delayed.json", "--trace"], (c2c / "c2c-high-queue-delayed.iqt").read_bytes())
    out = p.stdout.decode()
    if p.returncode != 0 or "T diag 100 queue 126 OTS-E-LNK-001" not in out \
            or "01000000000000000000000000000000" not in out:
        failures.append(f"{pathlib.Path(binary).name} same-cycle link order: exit {p.returncode} {out!r} {p.stderr!r}")

# 4. timing rows keyed by an element type
sys.path.insert(0, str(root / "opentsp-arch-spec/tools"))
import tsp_as  # noqa: E402

base = json.loads(mini.read_text())


def typed_rows(*rows):
    cfg = json.loads(json.dumps(base))
    cfg["timing"]["overrides"] += [{"entry": e, "d_func": f, "d_skew": k} for e, f, k in rows]
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as out:
        json.dump(cfg, out)
    return path


TYPED = [
    ([("Max.i8", 3, 2)], None),
    ([("Max", 4, 2), ("Max.i8", 3, 2), ("Max.f32", 9, 2)], None),
    ([("Exp.f16", 6, 2), ("Convert.i8", 6, 2), ("Cmp.b8", 6, 2), ("Conv_F16_I8.f16", 6, 2)], None),
    ([("Read.i8", 5, 0)], "timing.overrides: Read.i8: Read does not take element type i8"),
    ([("Max.b8", 5, 2)], "timing.overrides: Max.b8: Max does not take element type b8"),
    ([("Exp.i32", 5, 2)], "timing.overrides: Exp.i32: Exp does not take element type i32"),
    ([("Conv_F16_I8.i8", 5, 2)], "timing.overrides: Conv_F16_I8.i8: Conv_F16_I8 does not take element type i8"),
    ([("Max.q9", 5, 2)], "timing.overrides: Max.q9 is not a catalogue entry"),
    ([("Max.", 5, 2)], "timing.overrides: Max. is not a catalogue entry"),
    ([("Max.i8", 3, 2), ("Max.i8", 4, 2)], "timing.overrides: Max.i8 is listed more than once"),
    ([("Max.i8", 2, 2)], "timing.overrides.Max.i8: Max.i8: d_skew 2 must be below d_func 2"),
]
for rows, want in TYPED:
    path = typed_rows(*rows)
    names = ", ".join(r[0] for r in rows)
    for binary in binaries:
        p = run(binary, [path])
        err = p.stderr.decode()
        if (p.returncode, want is None or want in err) != ((0, True) if want is None else (2, True)):
            failures.append(f"{pathlib.Path(binary).name} typed rows [{names}]: exit {p.returncode} {err!r}, want {want!r}")
    try:
        tsp_as.validate_config(path)
        got = None
    except tsp_as.ConfigError as e:
        got = str(e)
    if (got is None) != (want is None) or (want is not None and want not in got):
        failures.append(f"tsp_as typed rows [{names}]: {got!r}, want {want!r}")
    os.unlink(path)

# 5. the `extensions` list and the accumulator width: the three implementations of the configuration rules refuse the same profiles
WIDE = "extensions: wide_acc is declared exactly when mxm.acc_bits is 128"
EXTENSION_CASES = [
    ("wide_acc with 128 bits", ["wide_acc"], 128, None),
    ("bf16 with 32 bits", ["bf16"], 32, None),
    ("acc_bits 64", [], 64, "mxm.acc_bits:"),
    ("wide_acc with 32 bits", ["wide_acc"], 32, WIDE),
    ("128 bits without wide_acc", [], 128, WIDE),
    ("unknown extension", ["block_fp9"], 32, "extensions: block_fp9 is not an extension of this version"),
    ("duplicate extension", ["bf16", "bf16"], 32, "extensions: bf16 is listed more than once"),
]
for label, names, bits, want in EXTENSION_CASES:
    cfg = json.loads(json.dumps(base))
    cfg["extensions"], cfg["mxm"]["acc_bits"] = names, bits
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as out:
        json.dump(cfg, out)
    for binary in binaries:
        p = run(binary, [path])
        err = p.stderr.decode()
        if (p.returncode, want is None or want in err) != ((0, True) if want is None else (2, True)):
            failures.append(f"{pathlib.Path(binary).name} {label}: exit {p.returncode} {err!r}, want {want!r}")
    try:
        tsp_as.validate_config(path)
        got = None
    except tsp_as.ConfigError as e:
        got = str(e)
    if (got is None) != (want is None) or (want is not None and want not in got):
        failures.append(f"tsp_as {label}: {got!r}, want {want!r}")
    os.unlink(path)

for f in failures:
    print("FAIL", f)
print(f"harness errata: {len(corpus)} corpus files, {len(binaries)} engines, {len(failures)} failures")
sys.exit(1 if failures else 0)
