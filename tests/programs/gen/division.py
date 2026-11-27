"""Reciprocal idioms on f32 (the ISA has RSqrt but no Recip; , 'Division and square
root'). Three ways to form 1/x, each a short chain of VXM operations:

  rsqrt_sq    y = rsqrt(x * x)         = 1/|x|            two chained ALUs
  rsqrt_pair  y = rsqrt(x) * rsqrt(x)  = 1/x for x > 0    two chained ALUs (the product reads one result twice)
  newton      y1 = y0 * (2 - x * y0)   one Newton step of a seed y0 for 1/x, three chained ALUs; `newton_sq` and
                                       `newton_pair` start from the result of the first and of the second idiom

The expectations are the numpy twin of opentsp_tools.programs.numerics (checked against the Python oracle by test_numerics.py), so a
program built from these groups pins what the simulators compute. `division_domains.py` measures, by running the same
groups on the ISS, where each idiom is usable, and `test_division.py` checks that the cookbook quotes those numbers.
Groups are `GAP` cycles apart; the vectors of a program are whatever the caller passes."""
import json
import os

import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import numerics as nx

ADDR = 0x10
GAP = 24                 # cycles between two groups: chained results and their stores never overlap
IDIOMS = ("rsqrt_sq", "rsqrt_pair", "newton_sq", "newton_pair")
SEED = {"newton_sq": "rsqrt_sq", "newton_pair": "rsqrt_pair"}      # the idiom whose result seeds a Newton step


def f32(a):
    return np.asarray(a, dtype=np.float32)


def twin(idiom, x, y0=None):
    """The result the numpy twin gives for one group (x and the seed y0 are f32 lane vectors)."""
    x = f32(x)
    with np.errstate(all="ignore"):
        if idiom == "rsqrt_sq":
            return nx.transcendental("rsqrt", "f32", nx.binary("mul", "f32", x, x))
        if idiom == "rsqrt_pair":
            r = nx.transcendental("rsqrt", "f32", x)
            return nx.binary("mul", "f32", r, r)
        y0 = f32(twin(SEED[idiom], x) if y0 is None else y0)
        two = np.full_like(x, 2.0)
        return nx.binary("mul", "f32", y0, nx.binary("sub", "f32", two, nx.binary("mul", "f32", x, y0)))


def group(k, j, idiom, x, y0, two):
    """Emit group j of `idiom` on x (and the seed y0 of newton; `two` is the placed constant 2); the result goes to
    east quad 0 at word ADDR + j. Returns the expected vector."""
    t0 = k.lead("mul.f32" if idiom != "rsqrt_pair" else "rsqrt.f32") + GAP * j
    base = ADDR + j
    xs = k.place(f"x{j}", x, 1, base)
    alu = 3 * j % 15
    if idiom == "rsqrt_sq":
        vx = k.feed(xs, base, "E", 16, t0, "mul.f32")
        v0 = k.b.vxm(alu, "mul.f32", "E", 8, [vx, vx], width=4)
        v1 = k.b.vxm(alu + 1, "rsqrt.f32", "E", 12, [v0], chain=(0,), width=4)
    elif idiom == "rsqrt_pair":
        vx = k.feed(xs, base, "E", 16, t0, "rsqrt.f32")
        v0 = k.b.vxm(alu, "rsqrt.f32", "E", 8, [vx], width=4)
        v1 = k.b.vxm(alu + 1, "mul.f32", "E", 12, [v0, v0], chain=(0, 1), width=4)
    else:
        ys = k.place(f"y{j}", y0, 2, base)
        vx = k.feed(xs, base, "E", 16, t0, "mul.f32")
        vy = k.feed(ys, base, "E", 20, t0, "mul.f32")
        v0 = k.b.vxm(alu, "mul.f32", "E", 8, [vx, vy], width=4)
        t1 = v0.cycle + 1 - k.p.timing("sub.f32")[1]
        v2 = k.feed(two, ADDR - 1, "E", 24, t1, "sub.f32")
        v1 = k.b.vxm(alu + 1, "sub.f32", "E", 12, [v2, v0], chain=(1,), width=4)
        t2 = v1.cycle + 1 - k.p.timing("mul.f32")[1]
        vy2 = k.feed(ys, base, "E", 28, t2, "mul.f32")
        v1 = k.b.vxm(alu + 2, "mul.f32", "E", 8, [vy2, v1], chain=(1,), width=4)
    want = twin(idiom, x, y0)
    k.store(v1, 0, base, want)
    return want


def program(p, name, groups, doc, seed=1, tags=("sim",)):
    """(Kit, expected vectors): one group per (idiom, x) or (idiom, x, y0); the constant 2 is placed once and read
    by every newton group."""
    k = Kit(p, name, doc, seed, module="division", source="numpy-twin", tags=tags)
    two = k.place("two", np.full(p.vlanes, 2.0, dtype=np.float32), 3, ADDR - 1)
    wants = []
    for j, g in enumerate(groups):
        idiom, x = g[0], f32(g[1])
        y0 = f32(g[2]) if len(g) > 2 else (twin(SEED[idiom], x) if idiom in SEED else None)
        wants.append(group(k, j, idiom, x, y0, two))
    return k, wants


# -- the golden program: the measured edges of every idiom, and the special values ------------------------------------

DOMAINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "division_domains.json")

SPECIAL_BITS = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x3F800000, 0xBF800000, 0x40000000,
                0x40400000, 0x3F000000, 0x40E00000, 0xC0E00000, 0x0DA24260, 0x7149F2CA, 0x00000001, 0x7F7FFFFF]


def edge_bits(idiom, domains):
    """Bit patterns at the measured edges of `idiom`: one below, at and above the first and the last value with a finite
    non-zero result, and around both ends of the accurate binades."""
    d = domains["idioms"][idiom]
    lo_e, hi_e = d["accurate_binades"]

    def bits(e):
        return (e + 127) << 23 if e >= -126 else 1 << (e + 149)
    pts = [int(d["first_usable_bits"], 16), int(d["last_usable_bits"], 16), bits(lo_e), bits(hi_e + 1)]
    out = []
    for b in pts:
        out += [b - 1, b, b + 1]
    return [b for b in out if 0 < b < 0x7F800000]


def fill(bits, n):
    """The patterns `bits`, cycled to n lanes."""
    return np.array([bits[i % len(bits)] for i in range(n)], dtype=np.uint32).view(np.float32)


def edges(p, seed=1):
    """For each idiom two groups: the edge values of the measured domains (division_domains.json), and the special
    values (zeros, infinities, a NaN, negative numbers). The expected vectors are the numpy twin's; the ISS runs of
    division_domains.py show that the twin and the ISS agree on every binade."""
    with open(DOMAINS) as f:
        domains = json.load(f)
    groups = []
    for idiom in IDIOMS:
        groups.append((idiom, fill(edge_bits(idiom, domains), p.vlanes)))
        groups.append((idiom, fill(SPECIAL_BITS, p.vlanes)))
    k, _ = program(p, "vxm/division_edges", groups,
                   "reciprocal idioms at the edges of their measured domains and on special values", seed)
    return k.done("edges")


CASES = [("vxm/division_edges", edges, ["ref", "test/ref_slow"], {})]
