"""Independent chip-level rate, causality, fetch-port and packet-layout regressions.

Run with a built simulator and the ref profile. Expectations come from conservation
of resources and explicit source byte ranges, rather than mirrored model helpers.
"""
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "opentsp-arch-spec/tools"))
from tsp_as import lower_records, q_lines, word_units

SIM = str(Path(sys.argv[1]).resolve())
CFG = Path(sys.argv[2]).resolve()
BASE = json.loads(CFG.read_text())
assert BASE["geometry"]["superlanes"] == 20, "this full-chip contract uses ref geometry"
OK = True


def check(name, condition, detail=""):
    global OK
    print(("PASS " if condition else "FAIL ") + name, detail if not condition else "")
    OK &= condition


def run(text="", cfg=None):
    with tempfile.TemporaryDirectory() as td:
        path = CFG
        if cfg is not None:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(cfg))
        return subprocess.run([SIM, str(path), "--trace"], input=text,
                              capture_output=True, text=True, timeout=120)


def mem(op, addr, stream):
    return (1 << 60) | (op << 54) | (addr << 32) | (stream << 26)


def mxm(op, typ, stream=0, mode=0):
    return (3 << 28) | (op << 22) | (typ << 20) | (stream << 14) | (mode << 13)


def program(recs, ram):
    return "\n".join(["S %x %x " % key + " ".join("%x" % b for b in value)
                      for key, value in ram.items()] + q_lines(lower_records(recs))) + "\n"


def chip_rate(typ):
    recs, ram, attempts = [], {}, []

    def feed(sl, addr, value, stream, at):
        row, quad = stream >> 5, (sl % 44) // 4
        gap = 18 - quad if (row == 0) == (sl < 44) else 24 + quad
        ram[sl, addr] = value
        recs.append((at - 5 - (43 - gap), sl, 1, mem(0, addr, stream), 0))

    for plane in range(4):
        row, start = (1, 44) if plane < 2 else (0, 0)
        first = 100 + plane * 40
        for batch in range(20):
            for stream in range(16):
                value = [0] * 320
                value[0] = [0, 0x3c][stream % 2] if typ == 1 else 1
                sl = start + 4 * (stream % 11) + stream // 11
                feed(sl, 100 + plane * 320 + batch * 16 + stream,
                     value, (row << 5) | stream, first + batch)
            recs.append((first + batch, 104 + 3 * plane, 1, mxm(1, typ, row << 5, 1), 0))
        for k in range(4):
            at = 300 + k
            for byte in range(2 if typ == 1 else 1):
                value = [0] * 320
                value[0] = [0, 0x3c][byte] if typ == 1 else 1
                if plane % 2 == 0:
                    feed(start + 16 + byte, 1500 + k * 2 + byte,
                         value, (row << 5) | 24 + byte, at)
            q = 105 + 3 * plane
            recs.append((at, q, 1, mxm(2, typ, (row << 5) | 24), 0))
            attempts.append((at, q))
    result = run(program(recs, ram))
    diags = [line for line in result.stdout.splitlines() if line.startswith("T diag")]
    rejected = {(int(line.split()[2]), int(line.split()[4])) for line in diags}
    accepted = [x for x in attempts if x not in rejected]
    counts = [sum(q == 105 + 3 * p for _, q in accepted) for p in range(4)]
    check("rate: only paired-plane resource refusals", all("RES-001" in l for l in diags), diags[:5])
    check("rate: whole-chip accepted issues type%d" % typ,
          counts == ([4, 0, 4, 0] if typ == 1 else [4, 4, 4, 4]), counts)
    return len(accepted) * (160 if typ == 1 else 320) * 320 / 4


i8, f16 = chip_rate(0), chip_rate(1)
check("FP16 whole-chip arithmetic capacity is one quarter", f16 / i8 == 0.25, (i8, f16))

for entry, cls in [("Shift", "sxm_cross"), ("Permute", "sxm_cross"),
                   ("ABC", "mxm_abc"), ("ABC8", "mxm_abc")]:
    for valid in (False, True):
        cfg = copy.deepcopy(BASE)
        cfg["timing"]["overrides"] = [r for r in cfg["timing"]["overrides"] if r["entry"] != entry]
        cfg["timing"]["overrides"].append(dict(entry=entry, d_skew=15, d_func=35 if valid else 34))
        result = run(cfg=cfg)
        check("physical-time boundary %s valid=%s" % (entry, valid), result.returncode == (0 if valid else 2), result.stderr)
cfg = copy.deepcopy(BASE)
cfg["timing"]["sxm_map_ready"] = 19
check("map dissemination cannot precede last superlane", run(cfg=cfg).returncode == 2)

# At 64 B/cycle the first unit from tile 19 needs 19 completed cycles.
# A delivery at 19 would already have passed the IQ enqueue phase.
for latency, expected in ((1, 2), (14, 2), (15, 0)):
    cfg = copy.deepcopy(BASE)
    cfg["icu"].update(fetch_bytes_per_cycle=64, fetch_latency=latency)
    out = run(cfg=cfg)
    check(f"fetch latency {latency} at 64 bytes/cycle", out.returncode == expected, out.stderr)

# A scheduled producer/consumer. Fetch a Read from tile 19, then consume its
# eastbound stream at slice 44. The Write must move with the IQ service rate.
for rate, latency, issue in ((16, 4, 24), (64, 15, 20)):
    cfg = copy.deepcopy(BASE)
    cfg["icu"].update(fetch_bytes_per_cycle=rate, fetch_latency=latency)
    first = bytearray([255] * 320)
    first[304:312] = b"".join(u.to_bytes(4, "little") for u in word_units(mem(0, 2, 0)))
    at = issue + 5 + 6
    text = program([(0, 0, 1, 0x01c00000, 0), (at, 44, 1, mem(1, 9, 0), 0)],
                   {(0, 0): first, (0, 1): [255] * 320, (0, 2): [37] * 320})
    r = run(text + "D 2c 9\n", cfg)
    dump = next((l.split()[3:] for l in r.stdout.splitlines() if l.startswith("W ")), [])
    check(f"{rate} bytes/cycle: fetched Read issues at {issue}, consuming Write at {at}",
          r.returncode == 0 and "T diag" not in r.stdout and len(dump) == 320
          and all(int(b, 16) == 37 for b in dump), (r.stderr, dump[:3]))

# The queue waits for an occupied path; it must not invent a second R port.
fetch = program([(0, 0, 1, 0x01c00010, 0), (1, 0, 1, mem(0, 0x12, 7), 0)],
                {(0, 0x10): [255] * 320, (0, 0x11): [255] * 320})
r = run(fetch)
read_cycles = [int(l.split()[2]) - BASE["timing"]["mem"]["d_func"]
               for l in r.stdout.splitlines() if l.startswith("T srf") and l.split()[3] == "E:7"]
check("local fetch reserves two R-port cycles", read_cycles == [2], read_cycles)

# At a bank boundary, the fetch uses bank0 then bank1, not both at once.
bank_boundary = 1 << BASE["sram"]["bank_bit"]
for write_at, write_addr, expected in ((100, bank_boundary, 0), (100, 0, 2), (101, bank_boundary, 1), (101, 0, 0)):
    text = program([(100, 0, 1, 0x01c00000 | (bank_boundary - 1), 0),
                    (write_at, 0, 1, mem(1, write_addr, 7), 0)],
                   {(0, bank_boundary - 1): [255] * 320, (0, bank_boundary): [255] * 320})
    r = run(text)
    got = sum("OTS-E-MEM-001" in l for l in r.stdout.splitlines() if l.startswith("T diag"))
    check("local fetch bank accounting at%d address%d" % (write_at, write_addr), got == expected, (got, expected))

# Use an invalid text unit as an arrival marker. On ref, admission of a fetch
# at 10 permits the first four units at 15 and the last four at 54.
for index, arrival in [(0, 15), (159, 54)]:
    payload = bytearray([255] * 640)
    payload[4 * index:4 * index + 4] = (0x70000000).to_bytes(4, "little")
    image = program([(10, 0, 1, 0x01c00000 | (bank_boundary - 1), 0)],
                    {(0, bank_boundary - 1): payload[:320], (0, bank_boundary): payload[320:]})
    diagnostics = [line for line in run(image).stdout.splitlines() if line.startswith("T diag")]
    check(f"ref local fetch unit {index} reaches decode at {arrival}",
          diagnostics == [f"T diag {arrival} queue 0 OTS-E-DEC-003"], diagnostics)

# A temporal fault distinguishes two real reads from one read plus delayed delivery.
text_bytes = list(b"".join(u.to_bytes(4, "little") for u in word_units(mem(0, 0x12, 7))))
for which in (0, 1):
    ram = {(0, 0x10): [255] * 320, (0, 0x11): [255] * 320}
    ram[0, 0x10 + which] = text_bytes + [255] * (320 - len(text_bytes))
    text = program([(0, 0, 1, 0x01c00010, 0)], ram)
    for at in (0, 1, 2):
        r = run(text + "F sram %x 0 %x 0 0\n" % (at, 0x10 + which))
        corr = any(l.startswith("T ecc") and "corr text" in l for l in r.stdout.splitlines())
        check("local fetch word%d samples ECC at its own read: fault at%d" % (which, at),
              corr == (at <= which), r.stdout[-100:])

recs = [(75, 44, 1, mem(0, 1, 32), 0), (100, 126, 1, (5 << 28) | (32 << 10), 0),
        (316, 142, 1, (5 << 28) | (1 << 22) | (33 << 10), 0),
        (325, 45, 1, mem(0, 2, 34), 0),
        (331, 88, 1, (2 << 60) | (36 << 46) | (33 << 40) | (34 << 34), 0)]
text = program(recs, {(44, 1): [3] * 320, (45, 2): [0] * 320})
r = run(text + "F link 64 0 2 uncorr\n")
ecc = [line for line in r.stdout.splitlines() if line.startswith("T ecc") and "queue 88" in line]
check("body flit2 bytes13..20 crosses superlanes0 and1",
      any(" sl 0 " in l for l in ecc) and any(" sl 1 " in l for l in ecc), ecc)
sys.exit(0 if OK else 1)
