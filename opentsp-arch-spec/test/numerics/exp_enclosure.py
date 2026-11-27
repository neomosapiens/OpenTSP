#!/usr/bin/env python3
"""Check the actual Sail exp_small against exact rational Taylor enclosures."""
from fractions import Fraction
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def bounds(p, a):
    scale = 1 << p
    term = Fraction(1)
    total = term
    value = Fraction(abs(a), 8)
    for k in range(10000):
        next_term = term * value / (k + 1)
        # Subsequent ratios <= 1/2: the entire positive tail is < 2*next_term.
        if next_term * scale * 32 < 1:
            lower, upper = total, total + 2 * next_term
            if a < 0:
                lower, upper = 1 / upper, 1 / lower
            return lower * scale, upper * scale
        term = next_term
        total += term
    raise AssertionError("exact oracle failed to converge")


def check():
    model = Path(os.environ.get("OPENTSP_MODEL_DIR", ROOT / "model")).resolve()
    with tempfile.TemporaryDirectory(prefix="exp-enclosure-") as tmp:
        work = Path(tmp)
        (work / "prelude").symlink_to(model / "prelude")
        (work / "numerics").symlink_to(model / "core/numerics")
        (work / "core").mkdir()
        for name in ("geometry", "types"):
            (work / "core" / (name + ".sail")).symlink_to(model / "core" / (name + ".sail"))
        project = """prelude { files prelude/prelude.sail, prelude/errors.sail, }
core { requires prelude files core/geometry.sail, core/types.sail, }
numerics { requires prelude, core files numerics/fp.sail, numerics/lanes.sail, numerics/transcendental.sail, }
probe { requires prelude, core, numerics files probe.sail, }
"""
        (work / "probe.sail_project").write_text(project)
        (work / "probe.sail").write_text("""function loops_equiv_run() -> unit = {
  foreach (j from 0 to 6) {
    let p = 64 * pow2i(j);
    foreach (a from -4 to 4) {
      let y = tdiv_int(pow2i(p) * a, 8);
      let (lo, hi) = exp_small(y, p);
      print_endline(dec_str(p) ^ "," ^ dec_str(a) ^ "," ^ dec_str(lo) ^ "," ^ dec_str(hi))
    }
  }
}
function main() -> unit = ()
""")
        lib = Path(shutil.which("sail")).parent.parent / "share/sail/lib"
        subprocess.run(["sail", "-c", "--strict-var", "--strict-bitvector", "--strict-exponentials",
                        "--all-modules", "--c-no-main", "--c-preserve", "loops_equiv_run",
                        "-o", "probe", "probe.sail_project"], cwd=work, check=True)
        includes = ["-I" + str(lib), "-I" + str(work)]
        if os.environ.get("GMP_PREFIX"):
            includes += ["-I" + os.environ["GMP_PREFIX"] + "/include"]
        # The existing module-only C entry point loads a real profile and calls the probe.
        shutil.copyfile(ROOT / "test/export/loops_equiv.c", work / "main.c")
        subprocess.run(["cc", "-O1", "-w", *includes, "probe.c", "main.c",
                        *[str(lib / f) for f in ("sail.c", "rts.c", "elf.c", "sail_failure.c",
                                                 "sail_config.c", "cJSON.c")],
                        "-o", "probe_bin", os.environ.get("GMP_LIBS", "-lgmp"),
                        os.environ.get("Z_LIBS", "-lz")], cwd=work, check=True)
        result = subprocess.run([str(work / "probe_bin"), str(ROOT / "config/mini.json")],
                                text=True, capture_output=True, check=True)
        cases = 0
        failures = []
        for line in result.stdout.splitlines():
            p, a, lo, hi = map(int, line.split(","))
            lower, upper = bounds(p, a)
            cases += 1
            if not (lo <= lower <= upper <= hi):
                failures.append((p, a))
        assert cases == 63, cases
        assert not failures, f"false exp_small enclosures (precision, eighths): {failures}"
        print(f"PASS: {cases} actual Sail enclosures against exact rational bounds")


if __name__ == "__main__":
    check()
