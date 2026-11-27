#!/usr/bin/env python3
"""Decimal fields longer than Python's integer-string limit (4,300 digits): `tsp_as.py` and `ots-asm` give the same
result and exit code, a value with leading zeros is accepted, an out-of-range value is a line diagnostic OTS-E-ASM-001.

    python3 opentsp-sw-toolchain/tests/test_long_decimals.py [path/to/ots-asm]
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSP_AS = os.path.join(ROOT, "opentsp-arch-spec", "tools", "tsp_as.py")
EXE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "target")), "release", "ots-asm")

PROGRAM = ".arch opentsp-tsa\n.profile mini\n.section text\nbundle @cycle={c} {{\nMEM[0]: read 0x0, E:0, 0x0\n}}\n"
CASES = [
    ("cycle 1 after 5,000 zeros", "0" * 5000 + "1", 0, "Q 0 00000001 10000000 00000000"),
    ("cycle 2^32 after 5,000 zeros", "0" * 5000 + "4294967296", 1, "OTS-E-ASM-001"),
    ("5,000 nines", "9" * 5000, 1, "OTS-E-ASM-001"),
    ("4,301 nines", "9" * 4301, 1, "OTS-E-ASM-001"),
    ("5,001 zeros", "0" * 5001, 0, None),
]

fails = 0
with tempfile.TemporaryDirectory() as tmp:
    for name, cycle, rc, want in CASES:
        path = os.path.join(tmp, "p.ots")
        with open(path, "w") as f:
            f.write(PROGRAM.format(c=cycle))
        ref = subprocess.run([sys.executable, TSP_AS, "asm", path, "--input"], capture_output=True, text=True)
        new = subprocess.run([EXE, "asm", path, "--input"], capture_output=True, text=True)
        ok = ref.returncode == rc and new.returncode == rc and ref.stdout == new.stdout
        if rc:
            ok = ok and ref.stderr == new.stderr and ref.stderr.startswith(f"{path}:4: {want}: bundle @cycle={cycle}:")
        elif want:
            ok = ok and want in ref.stdout.splitlines()
        if not ok:
            fails += 1
            print(f"FAIL {name}\n  tsp_as : {ref.returncode} {ref.stdout[:200]!r} {ref.stderr[-300:]!r}\n"
                  f"  ots-asm: {new.returncode} {new.stdout[:200]!r} {new.stderr[-300:]!r}")
print(f"test_long_decimals: {len(CASES)} cases, {fails} failures")
sys.exit(1 if fails else 0)
