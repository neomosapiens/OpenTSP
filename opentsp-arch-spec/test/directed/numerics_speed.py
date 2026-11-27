#!/usr/bin/env python3
"""Speed guard of the model's integer log: the scan time of the binary16 add and multiply checkers on the model must stay within
1.5 times of the same scan on a copy of the model whose ilog2 is the one-bit-at-a-time loop (the loop is slow to prove, the
loop-free cascade must not be slow to run). The copy is made from the model under test, so only ilog2 differs.
Runs test/numerics/f16_exhaustive.sh with F16_X_MAX=16, JOBS=1, F16_X_INSN=1, old and new interleaved, and compares the least instructions
retired by the scan of each (`/usr/bin/time -l` of the scan process: a gate on a shared host sees the load in seconds, whole seconds at that,
and not in the instructions retired; ).
usage: numerics_speed.py [--ratio R] [--runs N]    (OPENTSP_MODEL_DIR selects the model directory under test)
Cost: two compilations and about 2 x 2 x 25 s of scanning on an idle core."""
import argparse, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = os.environ.get("OPENTSP_MODEL_DIR", os.path.join(ROOT, "model"))
SCRIPT = os.path.join(ROOT, "test", "numerics", "f16_exhaustive.sh")
LOOP = """function ilog2(n : int) -> int = {
  var k : int = 0;
  var m : int = n;
  while m >= 2 do { m = tdiv_int(m, 2); k = k + 1 };
  k
}

"""


def loop_copy(dst):
    shutil.copytree(MODEL, dst)
    p = os.path.join(dst, "core", "numerics", "fp.sail")
    s = open(p).read()
    a, b = s.index("$ifdef OPENTSP_PROOF_BUILD\nfunction ilog2(n : int) -> int = {"), s.index("function pow2i")
    open(p, "w").write(s[:a] + LOOP + s[b:])


def scan_seconds(model):
    env = dict(os.environ, OPENTSP_MODEL_DIR=model, F16_X_MAX="16", JOBS="1", F16_X_INSN="1")
    r = subprocess.run(["bash", SCRIPT, "add", "mul"], env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.exit("numerics_speed: the scan failed:\n" + r.stdout + r.stderr)
    t = parse_instructions(r.stdout)
    if len(t) != 2:
        sys.exit("numerics_speed: unexpected scan output:\n" + r.stdout)
    return sum(t)


def parse_instructions(out):
    """The instructions retired by each operation's scan, from the `, N instructions retired` of the script's summary lines."""
    return [int(m.group(1)) for m in re.finditer(r", (\d+) instructions retired$", out, re.M)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--selftest", action="store_true", help="check the parser of the script's output and exit")
    a = ap.parse_args()
    if a.selftest:
        sample = ("f16_exhaustive: add: 1048576 pairs, 3 NaN results compared by class, 0 mismatches, 4 s, 5012345678 instructions retired\n"
                  "f16_exhaustive: mul: 1048576 pairs, 2 NaN results compared by class, 0 mismatches, 4 s, 4012345678 instructions retired\n")
        assert parse_instructions(sample) == [5012345678, 4012345678], parse_instructions(sample)
        assert parse_instructions("f16_exhaustive: add: 1 pairs, 0 NaN results compared by class, 0 mismatches, 4 s\n") == []
        print("numerics_speed: selftest passed")
        return
    tmp = tempfile.mkdtemp(prefix="numerics_speed_")
    old = os.path.join(tmp, "model_loop")
    loop_copy(old)
    new_t, old_t = [], []
    for _ in range(a.runs):
        old_t.append(scan_seconds(old))
        new_t.append(scan_seconds(MODEL))
    o, n = min(old_t), min(new_t)
    ratio = n / max(o, 1)
    ok = ratio <= a.ratio
    print("numerics_speed: loop %s instructions, cascade %s instructions, ratio %.2f (limit %.2f): %s" % (old_t, new_t, ratio, a.ratio, "PASS" if ok else "FAIL"))
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


main()
