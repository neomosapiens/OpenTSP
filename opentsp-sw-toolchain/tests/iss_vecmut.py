"""Golden-vector mutation differential (`just iss-vecmut`): take each shipped vector, apply random mutations (SRAM and SRF
faults at random cycles, bits and superlanes, t/p marks, shifted host-line cycles, flipped text-unit bits, dropped or
duplicated units, an `H start`, link-fault edits) and run `tsp_sim` and `tsp-iss` with --trace --observe --lint and an
--on-irq policy. The exit code, stdout and stderr (program name aside) must be equal. A run that exceeds the timeout is
counted and printed; more than TIMEOUT_SHARE of the runs timing out fails the job, so a slow reference cannot hide
divergences. A differing input is written to --out with its arguments.

`--starts` gives start cycles (hex, comma separated, default 0 and 2^62, 2^63 and 2^64 - 2^20): a run takes one of them in turn and
has `H start` set to it and every `H`, `F` and `P` host line moved by it, with `--max-cycles` 2^64 - 1, so the fault paths of the
cycle range above 2^63 are compared too. A profile with a patrol scrub (its name has `patrol`) runs from cycle 0 only: the scrub is
time-driven from reset, so a huge start is that many scrub events on both sides.

usage: iss_vecmut.py [--seed N] [--count N] [--jobs N] [--starts HEX,HEX,...] [--sim PATH] [--iss PATH] [--out DIR] [profile ...]
(default 400 runs on `ref` and `mini`, about 12 s with 3 jobs)."""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "opentsp-arch-spec"
SIM = ROOT / "target/spec/tsp_sim"
ISS = ROOT / "target/release/tsp-iss"
MUTS = 4
TIMEOUT_SECONDS = 200
TIMEOUT_SHARE = 0.02
TIMEOUT = "TIMEOUT"
M64 = (1 << 64) - 1
DEFAULT_STARTS = "0,4000000000000000,8000000000000000,fffffffffff00000"


def cfg(name):
    p = SPEC / "config" / f"{name}.json"
    return str(p if p.exists() else SPEC / "config/test" / f"{name}.json")


def run(binary, args, stdin):
    """(returncode, stdout, stderr with the program name replaced) or (TIMEOUT, b'', '')."""
    try:
        r = subprocess.run([str(binary), *args], input=stdin.encode(), capture_output=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return (TIMEOUT, b"", "")
    err = r.stderr.decode(errors="replace").replace(str(binary), "BIN")
    err = re.sub(r"(?m)^(tsp_sim|tsp-iss):", "PROG:", err)
    return (r.returncode, r.stdout, err)


def shift(stdin, offset):
    """The harness input with every `H`, `F` and `P` host line's cycle moved by `offset` (capped below 2^64) and `H start` set."""
    lines, started = [], False
    for line in stdin.splitlines():
        p = line.split()
        if len(p) > 2 and p[0] in ("H", "F", "P"):
            try:
                p[2] = f"{min(int(p[2], 16) + offset, M64):x}"
                line = " ".join(p)
            except ValueError:
                pass
            started = started or p[:2] == ["H", "start"]
        lines.append(line)
    if not started:
        lines.append(f"H start {offset:x}")
    return "\n".join(lines) + "\n"


def show(a, b, n=4000):
    return "\n".join(f"--- {tag} rc={x[0]}\nstdout:\n{x[1].decode(errors='replace')[:n]}\nstderr:\n{x[2][:n]}"
                     for tag, x in (("SAIL", a), ("ISS", b)))


def address(key):
    s, a = map(int, key.split(":"))
    return s, a


def base_lines(v):
    lines = [f"S {s:x} {a:x} {bytes.fromhex(w).hex(' ')}" for (s, a), w in ((address(k), w) for k, w in v["sram"].items())]
    text = {int(q): [int(u, 16) for u in units.split()] for q, units in v["text"].items()}
    dumps = [f"D {s:x} {a:x}" for s, a in (address(k) for k in v["dump"])]
    return lines, text, dumps, list(v.get("host", []))


def mutate(rnd, v, sl, prof, srf_seen):
    lines, text, dumps, host = base_lines(v)
    pre = [address(k) for k in v["sram"]]
    for _ in range(rnd.randint(1, MUTS)):
        k = rnd.random()
        if k < 0.3 and pre:
            s, a = rnd.choice(pre)
            host.append(f"F sram {rnd.randrange(0, 80):x} {s:x} {a:x} {rnd.randrange(sl):x} "
                        f"{rnd.choice([0, 1, 5, 0x7f, 0x80, 0x83, 0x88, 0x89]):x} {rnd.choice('tp')}")
        elif k < 0.45 and srf_seen:
            c, row, st, gap = rnd.choice(srf_seen)
            host.append(f"F srf {c:x} {row.lower()} {st:x} {gap:x} {rnd.randrange(sl):x} "
                        f"{rnd.choice([0, 3, 0x7f, 0x80, 0x88, 0x89]):x} {rnd.choice('tp')}")
        elif k < 0.6 and host:
            i = rnd.randrange(len(host))
            p = host[i].split()
            if len(p) > 2 and p[0] in ("H", "F", "P"):
                f = 2 if p[0] == "H" else 2
                try:
                    p[f] = f"{max(0, int(p[f], 16) + rnd.choice([-3, -1, 1, 2, 7])):x}"
                    host[i] = " ".join(p)
                except ValueError:
                    pass
        elif k < 0.75 and text:
            q = rnd.choice(list(text))
            us = text[q]
            j = rnd.randrange(len(us))
            us[j] ^= 1 << rnd.choice([31, 28, 27, 26, 20, 16, 15, 3, 0, rnd.randrange(32)])
        elif k < 0.85 and text:
            q = rnd.choice(list(text))
            us = text[q]
            j = rnd.randrange(len(us))
            if rnd.random() < 0.5 and len(us) > 1:
                del us[j]
            else:
                us.insert(j, us[j])
        elif k < 0.88 and any(h.startswith("F ") for h in host):
            fl = [j for j, h in enumerate(host) if h.startswith("F ")]
            j = rnd.choice(fl)
            p = host[j].split()
            if p[1] == "link":
                if rnd.random() < 0.5:
                    p[5] = "uncorr" if p[5] == "corr" else "corr"
                else:
                    p[4] = f"{max(0, int(p[4], 16) + rnd.choice([-1, 1, 2])):x}"
            elif p[1] in ("sram", "srf"):
                q = list(p)
                bi = 6 if p[1] == "sram" else 7
                q[bi] = f"{(int(p[bi], 16) + rnd.choice([1, 2, 9])) % 0x8a:x}"
                host.append(" ".join(q[:bi + 1]))
            if rnd.random() < 0.5:
                if p[-1] in ("t", "p"):
                    p[-1] = "p" if p[-1] == "t" else "t"
                else:
                    p.append(rnd.choice("tp"))
            host[j] = " ".join(p)
        elif k < 0.92:
            host.append(f"H start {rnd.randrange(1, 9):x}")
        else:
            host.append(f"H db {rnd.randrange(0, 60):x} {rnd.randrange(4):x}")
    q_lines = [f"Q {q:x} " + " ".join(f"{u:08x}" for u in us) for q, us in text.items() if us]
    return "\n".join(lines + q_lines + dumps + host) + "\n"


def srf_events(stdin, prof):
    r = subprocess.run([SIM, cfg(prof), "--trace"], input=stdin.encode(), capture_output=True, timeout=120)
    out = []
    for l in r.stdout.decode().splitlines():
        if l.startswith("T srf "):
            p = l.split()
            row, st = p[3].split(":")
            out.append((int(p[2]), row, int(st), int(p[5])))
    return out


def one(args):
    seed, i, prof, v, sl, out, start = args
    rnd = random.Random(seed * 1000033 + i)
    lines, text, dumps, host = base_lines(v)
    base = "\n".join(lines + [f"Q {q:x} " + " ".join(f"{u:08x}" for u in us) for q, us in text.items()] + dumps + host) + "\n"
    try:
        seen = srf_events(base, prof)[:200]
    except Exception:
        seen = []
    stdin = mutate(rnd, v, sl, prof, seen)
    opts = ["--trace", "--observe", "--lint"]
    if start and "patrol" not in prof:
        stdin = shift(stdin, start)
        opts += ["--max-cycles", str(M64)]
    k = rnd.random()
    if k < 0.35:
        opts += ["--on-irq", "replay"]
    elif k < 0.55:
        opts += ["--on-irq", "stop"]
    a = run(SIM, [cfg(prof), *opts], stdin)
    b = run(ISS, [cfg(prof), *opts], stdin)
    if a[0] == TIMEOUT or b[0] == TIMEOUT:
        return ("TIMEOUT", prof, v["name"], i)
    if a != b:
        p = out / f"vecmut-{prof}-{v['name']}-{seed}-{i}"
        p.with_suffix(".txt").write_text(stdin)
        p.with_suffix(".args").write_text(" ".join([prof, *opts]) + "\n")
        p.with_suffix(".diff").write_text(show(a, b))
        return ("DIFF", prof, v["name"], i, a[0], b[0], str(p))
    return ("ok",)


def main():
    global SIM, ISS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--starts", default=DEFAULT_STARTS, help="start cycles in hex, used in turn (default: %(default)s)")
    ap.add_argument("--sim", type=Path, default=SIM)
    ap.add_argument("--iss", type=Path, default=ISS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("profiles", nargs="*", default=["ref", "mini"])
    a = ap.parse_args()
    SIM, ISS = a.sim, a.iss
    out = a.out or Path(tempfile.mkdtemp(prefix="iss-vecmut-"))
    out.mkdir(parents=True, exist_ok=True)
    try:
        starts = [int(x, 16) for x in a.starts.split(",") if x]
    except ValueError:
        sys.exit(f"iss-vecmut: --starts takes hexadecimal cycles separated by commas, not {a.starts!r}")
    if not starts or any(x > M64 for x in starts):
        sys.exit(f"iss-vecmut: --starts needs at least one cycle below 2^64, not {a.starts!r}")
    rnd = random.Random(a.seed)
    jobs = []
    for prof in a.profiles:
        vs = json.load(open(SPEC / "conformance/model" / f"{prof}.json"))["vectors"]
        sl = json.load(open(cfg(prof)))["geometry"]["superlanes"]
        pick = [v for v in vs if re.match(r"(ecc|ras|host|scrub|link|c2c|mxm)", v["name"])] or vs
        for _ in range(a.count // len(a.profiles)):
            v = rnd.choice(pick if rnd.random() < 0.6 else vs)
            jobs.append((a.seed, len(jobs), prof, v, sl, out, starts[len(jobs) % len(starts)]))
    bad = timeouts = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, jobs):
            if r[0] == "DIFF":
                bad += 1
                print(*r, flush=True)
            elif r[0] == "TIMEOUT":
                timeouts += 1
    print(f"iss-vecmut seed {a.seed}: {len(jobs)} runs, {bad} differing, {timeouts} timeouts (failures in {out})")
    if bad:
        sys.exit(1)
    if timeouts > TIMEOUT_SHARE * len(jobs):
        sys.exit(f"iss-vecmut: {timeouts} of {len(jobs)} runs timed out; the comparison does not cover them")


if __name__ == "__main__":
    main()
