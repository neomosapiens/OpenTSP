"""Category P of the design's sec. 2.2 and 4.6, the instruction pager: programs whose per-queue images exceed
`icu.iq_depth`, so `opentsp_tools.pager` turns them into fetch-driven ones (Ifetch and IfetchM blocks of text bound as
dtype-255 tensors; the `.ots` is written from the paged cycle view). Every cycle follows from the profile through
the toolchain and every expectation is the byte arithmetic of the program, stated in each docstring. The fetch plan of a case
has neither the lint OTS-W-IQ-001 nor the diagnostic OTS-E-IQ-002 (`expect.lints` is empty): a late block is an error of the generator."""
import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import data
from opentsp_tools.programs import numerics as nx
from opentsp_tools import pager
from opentsp_tools import stations
from opentsp_tools.programs.kernels import qname
from opentsp_tools.programs.kernels.install import install
from opentsp_tools.stations import ScheduleError

PROFILES = ["ref", "mini", "test/mid_min", "test/fetch_alt"]
MXM_PROFILES = ["ref", "mini"]       # the matrix install needs sixteen west slices (see mxm.py)

# Operations of the long variants by profile tag (ISS only); the sim variants stay below the tsp_sim cycle budget.
# A MEM queue's text lives in its own SRAM, two words a block: `mini` has 128 words of text per slice.
LONG = {"sxm": {"ref": 400}, "vxm": {"ref": 2000, "mini": 200, "mid_min": 1000, "fetch_alt": 1000},
        "mem": {"ref": 1600, "mini": 160, "mid_min": 800, "fetch_alt": 800},
        "mxm": {"ref": 600, "mini": 300}}
SIM_BUDGET = 2400
ADDR = 0x08          # data words below 0x40 (bank 0 on every profile); text is allocated from the top of the SRAM


def cap_units(p):
    return p.icu["iq_depth"] // 4


class Case(Kit):
    """A Kit whose program is paged on `done`: every queue over the IQ becomes fetch-driven, the blocks are weights."""

    def __init__(self, p, name, doc, seed, long):
        super().__init__(p, name, doc, seed, module="pager")
        self.long = long

    def done(self, fn, expect_paged=()):
        view = pager.CycleView.of(self.b.prog)
        view2, blocks, externs = pager.page(view, self.p, self.b.t)
        self.b.t.check()
        paged = sorted({blk.queue for blk in blocks})
        for q in expect_paged:
            if q not in paged:
                raise ScheduleError(f"{self.name}: queue {q} is not paged (paged: {paged})")
        self.b.prog = view2.program(self.p)
        for name, words in pager.tensor_words(blocks).items():
            self.weights[name] = ("raw", b"".join(bytes(w) for w in words))
        text, case = super().done(fn)
        case["expect"]["lints"] = []
        case["golden"].update(paged_queues=paged, blocks=len(blocks))
        case["budget"]["cycles"] = self.last + 48        # the last values still cross the rows after the last write
        if not self.long and case["budget"]["cycles"] > SIM_BUDGET:
            raise ScheduleError(f"{self.name}: {case['budget']['cycles']} cycles exceed the sim budget {SIM_BUDGET}")
        case["tags"] = ["iss"] if self.long else ["sim"]
        return text, case


def attempt(builders):
    """The first of the parameter sets whose program the tracker accepts."""
    last = None
    for build in builders:
        try:
            return build()
        except ScheduleError as e:
            last = e
    raise ScheduleError(f"no parameters schedule: {last}")


def case_name(base, long):
    return f"pager/{base}" + ("" if long else "_sim")


def ops(p, kind, long, factor):
    """Operations of a case: `factor` x the IQ capacity in units for the sim variant, else the long count."""
    return LONG[kind][p.tag] if long else max(8, int(factor * cap_units(p)))


# -- VXM: K words circulating through one ALU queue ------------------------------------------------------------------
def _ring(p, long, seed, step, k_words):
    name = case_name("long_vxm", long)
    k = Case(p, name, "a ring of i8 adds on one VXM queue, paged", seed, long)
    quad = 1 if p.slices_per_hemisphere > 4 else 0
    ends = (4 * quad, 4 * quad + 1)
    sc = 4 * quad + 2
    b, n = k.b, p.vlanes
    rounds = max(2, -(-ops(p, "vxm", long, 0.55) // k_words))
    x = [data.rand_int(k.rng, "i8", n) for _ in range(k_words)]
    y = [data.rand_int(k.rng, "i8", n) for _ in range(k_words)]
    period = step * k_words
    g_w = stations.need_gap(p, ends[0], "E", "write")
    g_r = stations.need_gap(p, 88, "E", "read")
    d_func, d_skew = p.timing("read")[0], p.timing("add.i8")[1]
    lead = (g_r - g_w) + d_func - d_skew
    t0 = max(0, lead + 8)
    ready = [0] * k_words                       # first cycle a read of the ring's word may be dispatched
    for r in range(k_words):
        b.prog.extern(f"x{r}", ends[0], ADDR + r, 1)
        k.weights[f"x{r}"] = ("i8", bytes(x[r].view(np.uint8)))
        b.prog.extern(f"y{r}", sc, ADDR + r, 1)
        k.weights[f"y{r}"] = ("i8", bytes(y[r].view(np.uint8)))
    for j in range(rounds):
        src, dst = ends[j % 2], ends[(j + 1) % 2]
        for r in range(k_words):
            t_v = t0 + step * r + period * j
            if t_v - lead < ready[r]:
                raise ScheduleError(f"ring period {period} is shorter than the loop of word {r}")
            names = (lambda s: [s]) if j == 0 else (lambda s: None)
            vx = b.read_for(t_v, "add.i8", 88, [src], [ADDR + r], "E", 0, addr_text=names(f"x{r}"))
            vy = b.read_for(t_v, "add.i8", 88, [sc], [ADDR + r], "E", 4, addr_text=names(f"y{r}"))
            v = b.vxm(0, "add.i8", "W", 8, [vx, vy])
            ready[r] = b.write(v, [dst], [ADDR + r]) + 2
            k.last = max(k.last, ready[r])
    final = ends[rounds % 2]
    for r in range(k_words):
        k.expect(final, ADDR + r, bytes((x[r].astype(np.int64) + rounds * y[r].astype(np.int64)).astype(np.int8)
                                        .view(np.uint8)))
    return k


def long_vxm(p, long=False, seed=1):
    """K words circulate between two west slices of one quad, one `add.i8` on VXM[0] per word per round:
    x_{j+1} = x_j + y (wrapping i8), x read from slice A (rounds alternate A and B), y from slice C, the sum going
    west on row W back to the other slice. The N = K x R operations are one queue's every `step` cycles, so the VXM
    queue and the three slices' queues outrun icu.iq_depth and are paged. The dumped words are x_0 + R y: a block
    that arrives late, early or garbled changes one."""
    return attempt([lambda s=s, w=w: _ring(p, long, seed, s, w).done("long_vxm", expect_paged=(88,))
                    for s in (2, 3) for w in range(34, 3, -1)])


# -- MEM: K words bouncing between a west and an east slice ------------------------------------------------------------
def _relay(p, long, seed, period, stagger, k_words):
    name = case_name("long_mem", long)
    k = Case(p, name, "words relayed between a west and an east slice, paged", seed, long)
    quad = 1 if p.slices_per_hemisphere > 4 else 0
    ends = (4 * quad, 44 + 4 * quad)             # west slice, east slice
    b, n = k.b, p.vlanes
    rounds = max(2, -(-ops(p, "mem", long, 0.55) // k_words))
    x = [data.rand_int(k.rng, "i8", n) for _ in range(k_words)]
    for r in range(k_words):
        b.prog.extern(f"x{r}", ends[0], ADDR + r, 1)
        k.weights[f"x{r}"] = ("i8", bytes(x[r].view(np.uint8)))
    ready = [0] * k_words
    for j in range(rounds):
        src, dst = ends[j % 2], ends[(j + 1) % 2]
        row = "E" if j % 2 == 0 else "W"
        for r in range(k_words):
            t = stagger * r + period * j
            if t < ready[r]:
                raise ScheduleError(f"relay period {period} is shorter than the hop of word {r}")
            v = b.read(t, [src], [ADDR + r], row, 0, addr_text=[f"x{r}"] if j == 0 else None)
            ready[r] = b.write(v, [dst], [ADDR + r]) + 2
            k.last = max(k.last, ready[r])
    for r in range(k_words):
        k.expect(ends[rounds % 2], ADDR + r, bytes(x[r].view(np.uint8)))
    return k


def long_mem(p, long=False, seed=1):
    """K words hop between a west slice A and an east slice B (a Read onto row E, a Write on B; then back on row W),
    one hop per word per round, R rounds: both slices' queues hold N = K x R reads and writes and outrun
    icu.iq_depth, so IfetchM brings their text from their own SRAM. The dumped words are the originals, on A after an
    even and on B after an odd number of rounds: a lost, early or garbled hop leaves a zero word there."""
    west = 4 * (1 if p.slices_per_hemisphere > 4 else 0)
    return attempt([lambda h=h, d=d, w=w: _relay(p, long, seed, h, d, w).done("long_mem", expect_paged=(west,))
                    for h in range(36, 90, 2) for d in (2, 3) for w in range(h // d, 3, -1)])


# -- MXM: M activations through one installed matrix ----------------------------------------------------------------
ADDR_W, ADDR_X, ADDR_Y = 0x10, 0x30, 0x30
SAMPLES = 12         # results stored: every stride-th and the last (a result is four i32 planes on four slices)


def _mxm(p, long, seed, step):
    name = case_name("long_mxm", long)
    k = Case(p, name, "M activations through one installed i8 matrix, paged", seed, long)
    rows, n = p.mxm_rows, p.vlanes
    nlw = rows // 16
    count = ops(p, "mxm", long, 0.8)
    b = k.b
    W = data.rand_int(k.rng, "i8", rows * n).reshape(rows, n)
    for j in range(16):
        b.prog.extern(f"w{j}", j, ADDR_W, nlw)
        k.weights[f"w{j}"] = ("i8", b"".join(bytes(W[16 * i + j].view(np.uint8)) for i in range(nlw)))
    t_abc = install(k, "i8", nlw, True, ADDR_W)
    d_abc = p.timing("abc")[0]
    stride = max(1, count // SAMPLES)
    stored = 0
    for j in range(count):
        x = data.rand_int(k.rng, "i8", n)
        sl, addr = 16 + j % 4, ADDR_X + j // 4
        b.prog.extern(f"x{j}", sl, addr, 1)
        k.weights[f"x{j}"] = ("i8", bytes(x.view(np.uint8)))
        vx = b.read_for(t_abc + step * j, "abc", 105, [sl], [addr], "W", 16, addr_text=[f"x{j}"])
        b.unit(105, "abc", "abc.i8 W:16", [vx])
        _, v = b.unit(106, "acc", "acc.i32 E:8, replace, 0", cycle=t_abc + d_abc + step * j, dst=("E", 8, 4))
        if j % stride == 0 or j == count - 1:
            k.store(v, 1, ADDR_Y + stored, (W.astype(np.int64) @ x.astype(np.int64)).astype(np.int32), hemi="W")
            stored += 1
    return k


def long_mxm(p, long=False, seed=1):
    """M activations x_j (i8, from four west slices) go through one installed i8 matrix: ABC at one cycle in `step`,
    the ACC of its result `d_func(abc)` later on the next queue, so both MXM queues outrun icu.iq_depth and are paged.
    A dozen results (every stride-th and the last) are stored as four i32 planes and are the exact integer products
    W x_j: the last one is the product after M ABCs and M ACCs have each run on schedule."""
    return attempt([lambda s=s: _mxm(p, long, seed, s).done("long_mxm", expect_paged=(105, 106)) for s in (2, 3)])


# -- SXM: words permuted and rotated round a ring of two slices ----------------------------------------------------
SXM_PROFILES = ["ref"]
MAP_WORD = 0x60      # word of the permuter's map on west slices 2 (low bytes) and 3 (high bytes)


def _hops(k, queue, mnemonic, ends, ring, words, rounds, stagger, period, first_stream, t0):
    """`words` words hop `rounds` times between ends[0] (west) and ends[1] (east), one `mnemonic` on SXM queue `queue`
    per hop: the permuter's result is the one stream, the rotator's output 1 of nine. Words are read onto row W (both
    hemispheres' slices reach the SXM station there) and written from row E. A word read in an even round comes from
    bank 0 and goes to bank 1, and the other way round in an odd round, so a slice's Read and Write of one cycle never
    share a bank."""
    b, bank1 = k.b, 1 << k.p.bank_bit
    ready = [0] * words
    for j in range(rounds):
        src, dst = ends[j % 2], ends[(j + 1) % 2]
        for r in range(words):
            t = t0 + stagger * r + period * j
            if t < ready[r]:
                raise ScheduleError(f"{qname(queue)}: period {period} is shorter than the hop of word {r}")
            s = first_stream + r % 4
            d = 4 + 4 * (r % 4) if queue == 124 else 16 * (r % 2)
            vp = b.read_for(t, mnemonic, queue, [src], [ADDR + r + bank1 * (j % 2)], "W", s,
                            addr_text=[f"x{ring}w{r}"] if j == 0 else None)
            if queue == 124:
                _, v = b.unit(queue, mnemonic, f"permute W:{s}, 0x1, E:{d}", [vp], dst=("E", d, 1))
            else:
                _, v = b.unit(queue, mnemonic, f"rotate W:{s}, 0x3, E:{d}", [vp], dst=("E", d, 9))
                v = v.planes(1, 1)
            ready[r] = b.write(v, [dst], [ADDR + r + bank1 * ((j + 1) % 2)]) + 2
            k.last = max(k.last, ready[r])


def _sxm(p, long, seed, period, stagger, words):
    k = Case(p, case_name("long_sxm", long), "words permuted and rotated round a ring, SXM queues paged", seed, long)
    b, n, bank1 = k.b, p.vlanes, 1 << p.bank_bit
    rounds = max(2, -(-ops(p, "sxm", long, 0.9) // words))
    perm = nx.rotation_map(n, 1)
    wide = n > 256
    maps = [(perm & 255).astype(np.uint8)] + ([(perm >> 8).astype(np.uint8)] if wide else [])
    for j, m in enumerate(maps):
        k.extern_word(f"m{j}", 2 + j, MAP_WORD, bytes(m))
    t_load = 60
    vs = [b.read_for(t_load, "loadmap", 124, [2 + j], [MAP_WORD], "W", 24 + j, addr_text=[f"m{j}"])
          for j in range(len(maps))]
    mnemonic = "loadmapwide" if wide else "loadmap"
    b.unit(124, mnemonic, f"{mnemonic} 0x1, " + ", ".join(f"W:{24 + j}" for j in range(len(maps))), vs)
    rings = ((124, "permute", (4, 48), 8), (120, "rotate", (5, 49), 16))
    xs = {}
    for queue, _, ends, _ in rings:
        for r in range(words):
            xs[queue, r] = data.rand_int(k.rng, "i8", n)
            k.extern_word(f"x{queue}w{r}", ends[0], ADDR + r, bytes(xs[queue, r].view(np.uint8)))
    for queue, mnemonic, ends, first in rings:
        _hops(k, queue, mnemonic, ends, queue, words, rounds, stagger, period, first, t_load + p.superlanes + 62)
    for queue, _, ends, _ in rings:
        for r in range(words):
            x = xs[queue, r]
            for _ in range(rounds):
                x = x[perm] if queue == 124 else rotate_once(x)
            k.expect(ends[rounds % 2], ADDR + r + bank1 * (rounds % 2), bytes(x.view(np.uint8)))
    return k


def rotate_once(x):
    """Output 1 of Rotate with n = 3 (sxm.rotate_want): per superlane, the 3 x 3 tile of lanes 0..8 shifted one
    column; the other lanes are zero."""
    sl = x.reshape(-1, 16)
    o = np.zeros_like(sl)
    for r in range(3):
        for c in range(3):
            o[:, r * 3 + c] = sl[:, r * 3 + (c + 1) % 3]
    return o.reshape(-1)


def long_sxm(p, long=False, seed=1):
    """K words hop between a west slice and an east slice, once per round through one SXM queue: on SXM.W.PRM (queue
    124) a Permute by the map that moves every superlane of a segment of four to the one before it, on SXM.W.TSP[0]
    (queue 120) output 1 of a Rotate with n = 3. Each hop writes the word the next round reads, so the dumped words
    are the originals permuted, or rotated, `rounds` times: a block that arrives late, early or garbled changes one."""
    return attempt([lambda s=s, d=d, w=w: _sxm(p, long, seed, s, d, w).done("long_sxm", expect_paged=(120, 124))
                    for s in (90, 110, 140) for d in (2, 3) for w in (20, 16, 12, 8)])


CASES = [
    (f"pager/long_vxm{'' if long else '_sim'}", lambda p, long=long, **kw: long_vxm(p, long, **kw), PROFILES, {})
    for long in (True, False)
] + [
    (f"pager/long_mem{'' if long else '_sim'}", lambda p, long=long, **kw: long_mem(p, long, **kw), PROFILES, {})
    for long in (True, False)
]
CASES += [
    (f"pager/long_sxm{'' if long else '_sim'}", lambda p, long=long, **kw: long_sxm(p, long, **kw), SXM_PROFILES, {})
    for long in (True, False)
]
CASES += [
    (f"pager/long_mxm{'' if long else '_sim'}", lambda p, long=long, **kw: long_mxm(p, long, **kw), MXM_PROFILES, {})
    for long in (True, False)
]
