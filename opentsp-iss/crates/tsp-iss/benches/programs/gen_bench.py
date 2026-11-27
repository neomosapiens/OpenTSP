#!/usr/bin/env python3
"""Generate the ISS benchmark programs as Q-line text.

B0 a `D` line and no text (start-up); B1 64 MEM queues, `read` + `iterations 200`; B2 16 VXM queues,
`add.i8` + `repeat 190, 0` (d = 0: consecutive cycles, a spacing of d + 1 = 1); B3 = B1 + B2; B4 = B3 scaled to about 10^6 cycles (16 MEM loops of 65535
iterations, 245 chained `repeat 4095, 0`); `idle` is a read after NOPs that reach cycle 10^8 (a queue holds
320 units, 65535 cycles each, so the NOPs come in IfetchM-chained groups held in SRAM). The text is lowered with `tsp_as.lower` (the rules of `gen_model_vectors.py`) for profile `ref`.

The coverage programs, all with zero diagnostics on `tsp_sim` (`--check` does not run it; the
`bench_coverage` test of the ISS does): `dense_f16` (sustained f16 ABCs on planes 0 and 2, the two the chip can run at once:
an f16 matrix reserves an even and an odd plane), `dense_i8` (i8 ABCs on all four planes, a weight set loaded into each pair),
`vxm_valid` (four elementwise i8 adds with their operands streamed from MEM, no timing hazard), `transc` (f16 TanH of streamed
operands) and `syncnotify` (Notify on the control channel every 40 cycles, 100 Syncs on each of 143 queues). The MXM programs take the
LW/IW prologue of `tests/programs/mxm/matmul_{f16,i8}.ref.ots` (cycles below 50) and its SRAM layout, with deterministic data
(seed 1); `--check` fails when those programs change, and the files are regenerated then.

usage: gen_bench.py [--check | --small DIR]   (--check fails when a checked-in file differs from the generated one; --small writes the
short variants of the coverage programs for `tsp_sim` into DIR)
"""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "..", "opentsp-arch-spec", "tools"))
import tsp_as as A  # noqa: E402

ISA = A.ISA("ref")
MEM_QUEUES = 64
VXM_QUEUES = 16
IDLE_CYCLE = 100_000_000
HEADER = ""   # `tsp-iss run` takes a text program only when its first byte is Q


def word(text):
    return ISA.parse(text)[1]


def b1_entries(iterations):
    """Queue q reads its own SRAM word onto its own stream (E for the first hemisphere, W after)."""
    out = []
    for q in range(MEM_QUEUES):
        row = "E" if q < 44 else "W"
        out.append((0, q, word(f"read 0x10, {row}:{q % 32}, 0x0")))
        out.append((1, q, word(f"iterations {iterations:#x}")))
    return out


def b1b_entries(loops):
    """B1 with the loop repeated: after a loop of 0xffff iterations the next read opens the next."""
    out = []
    for q in range(MEM_QUEUES):
        row = "E" if q < 44 else "W"
        t = 0
        for _ in range(loops):
            out.append((t, q, word(f"read 0x10, {row}:{q % 32}, 0x0")))
            out.append((t + 1, q, word("iterations 0xffff")))
            t += 0xFFFF + 1
    return out


def b2_entries(repeats, chain):
    """Each VXM queue adds streams that nothing wrote (a TIM-001 at every execution)."""
    out = []
    for i in range(VXM_QUEUES):
        q = 88 + i
        out.append((0, q, word(f"add.i8 E:{8 + i}, E:{i}, E:{16 + i}")))
        t = 1
        for _ in range(chain):
            out.append((t, q, word(f"repeat {repeats:#x}, 0x0")))
            t += 1  # the next Repeat waits (hold Repeat) for this one to finish
    return out


def lines(entries, extra=()):
    return HEADER + "\n".join(A.q_lines(A.lower(entries)) + list(extra)) + "\n"


def idle_lines():
    """NOP units for IDLE_CYCLE - 1 cycles, in groups of 150 that end with an `ifetchm` of the next group
    (two SRAM words of 80 units); the last group ends with the read. Group 0 is the initial image of
    queue 4, group k >= 1 is stored at word BASE + 2 * (k - 1) of slice 4."""
    base = 16
    nops = [0xFFFF] * ((IDLE_CYCLE - 1) // 0xFFFF) + [(IDLE_CYCLE - 1) % 0xFFFF]
    groups = [nops[i : i + 150] for i in range(0, len(nops), 150)]
    s_lines = []
    for k, g in enumerate(groups):
        if k == len(groups) - 1:
            tail = A.word_units(word("read 0x10, E:0, 0x0"))
        else:
            tail = A.word_units(word(f"ifetchm {base + 2 * k:#x}"))
        units = g + tail
        if k == 0:
            image = {4: units}
            continue
        units += [A.PADDING] * (160 - len(units))
        for h in (0, 1):
            raw = b"".join(u.to_bytes(4, "little") for u in units[80 * h : 80 * h + 80])
            s_lines.append(f"S 4 {base + 2 * (k - 1) + h:x} " + " ".join(f"{b:x}" for b in raw))
    return A.q_lines(image) + s_lines


REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
MXM_TESTS = os.path.join(REPO, "tests", "programs", "mxm")
PROLOGUE_END = 50   # the weight stream and LW/IW prologue of the matmul tests ends before this cycle
BUNDLE = re.compile(r"bundle @cycle=(\d+) \{\n(.*?)\n\}", re.S)


def prologue(name):
    """The bundles of `tests/programs/mxm/<name>.ref.ots` below cycle 50 as {cycle: [lines]}, and the SRAM words of its case."""
    src = open(os.path.join(MXM_TESTS, name + ".ref.ots")).read()
    b = {int(c): body.split("\n") for c, body in BUNDLE.findall(src) if int(c) < PROLOGUE_END}
    case = json.load(open(os.path.join(MXM_TESTS, name + ".ref.case.json")))
    return b, sorted(tuple(map(int, k.split(":"))) for k in case["preload"])


def mirror(lines):
    """The east-hemisphere twin of plane-0 text: MEM[k] -> MEM[k + 44], W <-> E streams, MXM[0] -> MXM[2], MXM[1] -> MXM[3]."""
    out = []
    for l in lines:
        l = re.sub(r"MEM\[(\d+)\]", lambda m: f"MEM[{int(m.group(1)) + 44}]", l)
        l = re.sub(r"MXM\[(\d)\]", lambda m: f"MXM[{int(m.group(1)) + 2}]", l)
        l = re.sub(r"\b([EW]):", lambda m: ("W:" if m.group(1) == "E" else "E:"), l)
        out.append(l)
    return out


def repeat_chain(total, first):
    """`repeat` counts that re-issue an instruction `total` more times: the field is 12 bits, so a chain of
    `repeat 4095, 0` (each waits for the one before) and a last one for the rest; one per cycle from `first`."""
    out = []
    while total > 0:
        k = min(total, 4095)
        out.append((first + len(out), k))
        total -= k
    return out


def ots(bundles):
    out = [".arch opentsp-tsa", ".profile ref", ".section text"]
    for c in sorted(bundles):
        out.append(f"bundle @cycle={c} {{")
        out += bundles[c]
        out.append("}")
    return "\n".join(out) + "\n"


def assemble_text(bundles, words):
    """The harness text of a program given as bundles: its Q lines (`tsp_as.assemble`) and an `S` line per SRAM word."""
    isa, prog = A.assemble(ots(bundles), "<bench>")
    s_lines = [f"S {s:x} {a:x} " + " ".join(f"{b:x}" for b in w) for (s, a), w in sorted(words.items())]
    return A.input_text(prog.images) + "\n".join(s_lines) + "\n"


def f16_word(rng):
    """A random finite f16 as two bytes (low, high)."""
    v = (rng.getrandbits(1) << 15) | (rng.randrange(1, 16) << 10) | rng.getrandbits(10)
    return v & 0xFF, v >> 8


def dense_data(keys, f16, rng):
    """The SRAM words of a dense MXM program: finite f16 activations and weights (a low-byte word and a high-byte word per
    pair of slices 16/17 of an address), or random int8 bytes."""
    words = {}
    for s, a in keys:
        if f16 and s in (16, 17):
            if (16, a) not in words:
                v = [f16_word(rng) for _ in range(320)]
                words[(16, a)] = bytes(x[0] for x in v)
                words[(17, a)] = bytes(x[1] for x in v)
        elif (s, a) not in words:
            words[(s, a)] = bytes(rng.getrandbits(8) & 0xBF for _ in range(320))
    return words


def dense(f16, n=None):
    """Sustained dense MXM: `n` ABCs per cycle on the planes the chip runs at once (f16: planes 0 and 2; i8: planes 0 to 3,
    the weights of each hemisphere loaded into both of its planes), accumulated on planes 0 and 2."""
    n = n or (1500 if f16 else 20000)
    t = "f16" if f16 else "i8"
    base, keys = prologue("matmul_" + t)
    b = {c: list(v) for c, v in base.items()}
    side = {0: (0,), 1: (2,)} if f16 else {0: (0, 1), 1: (2, 3)}   # the planes of the west and the east hemisphere
    if not f16:   # a second weight set per hemisphere: plane 1 (3) loads the same streams in the same cycles
        for c, v in base.items():
            b[c] += [l.replace("MXM[0]", "MXM[1]") for l in v if "MXM[0]" in l]
    east = {}
    for c, v in b.items():
        east.setdefault(c, []).extend(mirror(v))
    b = {c: v + (east[c] if f16 else []) for c, v in b.items()}
    if not f16:
        for c, v in list(b.items()):
            b[c] = v
        for c, v in east.items():
            plane1 = [l for l in v if "MXM[2]" in l]
            b[c] = b[c] + [x for x in east[c] if "MXM[" not in x or "MXM[2]" in x] + [l.replace("MXM[2]", "MXM[3]") for l in plane1]
    stream = 16
    for hem, (w_in, w_out, mem) in enumerate((("W", "E", 16), ("E", "W", 60))):
        regs = [f"MEM[{mem}]: read 0x30, {w_in}:{stream}, 0x0"]
        loops = [f"MEM[{mem}]: iterations {n:#x}"]
        if f16:
            regs.append(f"MEM[{mem + 1}]: read 0x30, {w_in}:{stream + 1}, 0x0")
            loops.append(f"MEM[{mem + 1}]: iterations {n:#x}")
        b.setdefault(50, []).extend(regs)
        b.setdefault(51, []).extend(loops)
        for p in side[hem]:
            b.setdefault(69, []).append(f"MXM[{p}].LW: iw.{t}")
            b.setdefault(70, []).append(f"MXM[{p}].ABC: abc.{t} {w_in}:{stream}")
            for c, k in repeat_chain(n - 1, 71):
                b.setdefault(c, []).append(f"MXM[{p}].ABC: repeat {k:#x}, 0x0")
        acc = "acc.f32" if f16 else "acc.i32"
        p = side[hem][0]
        b.setdefault(90, []).append(f"MXM[{p}].ACC: {acc} {w_out}:8, accumulate, 0")
        for c, k in repeat_chain(n - 1, 91):
            b.setdefault(c, []).append(f"MXM[{p}].ACC: repeat {k:#x}, 0x0")
    return assemble_text(b, dense_data(keys + [(s + 44, a) for s, a in keys], f16, random.Random(1)))


def vxm_valid(n=20000):
    """Four elementwise i8 adds, every cycle for `n` cycles: x and y streamed from MEM (quads 1 and 2, y one cycle earlier),
    the sum written back by MEM[44..47]; no hazard (the operands are on the streams when the add reads them)."""
    e = []
    for k in range(4):
        e.append((1, 8 + k, word(f"read 0x10, E:{4 + k}, 0x0")))
        e.append((2, 8 + k, word(f"iterations {n:#x}")))
        e.append((2, 4 + k, word(f"read 0x10, E:{k}, 0x0")))
        e.append((3, 4 + k, word(f"iterations {n:#x}")))
        e.append((9, 88 + k, word(f"add.i8 E:{8 + k}, E:{k}, E:{4 + k}")))
        for c, r in repeat_chain(n - 1, 10):
            e.append((c, 88 + k, word(f"repeat {r:#x}, 0x0")))
        e.append((17, 44 + k, word(f"write 0x30, E:{8 + k}, 0x0")))
        e.append((18, 44 + k, word(f"iterations {n:#x}")))
    return lines(e)


def transc(n=3000):
    """f16 TanH of streamed operands, every cycle for `n` cycles (the operand is the stream pair E:16, E:17)."""
    e = [
        (2, 4, word("read 0x10, E:16, 0x0")),
        (2, 5, word("read 0x10, E:17, 0x0")),
        (3, 4, word(f"iterations {n:#x}")),
        (3, 5, word(f"iterations {n:#x}")),
        (9, 88, word("tanh.f16 E:12, E:16")),
    ]
    for c, r in repeat_chain(n - 1, 10):
        e.append((c, 88, word(f"repeat {r:#x}, 0x0")))
    return lines(e)


def syncnotify(k=100, queues=144):
    """Barriers: the data queues 1..143 each issue `k` Syncs, the control channel a Notify every 40 cycles: every Notify releases
    the whole chip, which parks again at its next Sync."""
    e = []
    for i in range(k):
        e.append((40 * i + 2, 0xFFFF, word("notify")))
    for q in range(1, queues):
        for i in range(k):
            e.append((i, q, word("sync")))
    return lines(e)


def coverage(small=False):
    """The coverage programs; `small` shortens each to about 100 cycles so that `tsp_sim` can run them."""
    return {
        "dense_f16.txt": dense(True, 40 if small else None),
        "dense_i8.txt": dense(False, 40 if small else None),
        "vxm_valid.txt": vxm_valid(200 if small else 20000),
        "transc.txt": transc(40 if small else 3000),
        "syncnotify.txt": syncnotify(6 if small else 100),
    }


def programs():
    b1 = b1_entries(200)
    b2 = b2_entries(190, 1)
    b4 = b1b_entries(16) + b2_entries(4095, 245)
    return {
        "b0.txt": HEADER + "D 4 0\n",
        "b1.txt": lines(b1),
        "b2.txt": lines(b2),
        "b3.txt": lines(b1 + b2),
        "b4.txt": lines(b4),
        "idle.txt": HEADER + "\n".join(idle_lines()) + "\n",
        **coverage(),
    }


def main(argv):
    if "--small" in argv:   # --small DIR: the short variants of the coverage programs, for `tsp_sim`
        out = argv[argv.index("--small") + 1]
        os.makedirs(out, exist_ok=True)
        for name, text in coverage(True).items():
            with open(os.path.join(out, name), "w") as f:
                f.write(text)
        return 0
    check = "--check" in argv
    bad = 0
    for name, text in programs().items():
        path = os.path.join(HERE, name)
        if check:
            old = open(path).read() if os.path.exists(path) else None
            if old != text:
                print(f"gen_bench: {name} is out of date", file=sys.stderr)
                bad = 1
        else:
            with open(path, "w") as f:
                f.write(text)
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
