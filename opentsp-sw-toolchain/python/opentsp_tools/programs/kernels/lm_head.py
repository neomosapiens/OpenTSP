"""lm_head: the logits `wte . h` over the packed head table . Install i of 160 packed rows holds, in the
five 64-lane segments of row r, the vocabulary rows 5 (160 i + r) + k; even installs go to the west MXM plane (plane 0),
odd ones to the east plane (plane 2). The hidden vector is rounded to f16 once, kept in MEM, and shifted by 64 k lanes
(zero fill) on the SXM shifter pair into the five segment vectors h_k; one f16 ABC of h_k on install i gives, in lane r,
the logit of entry 5 (160 i + r) + k (the exact f16 dot over 320 lanes rounded once to f32: the lanes of the other
segments multiply +0). Each ACC is written to the logits region at word 315 row + 5 i + k.

The segment vectors are kept in MEM twice, because an ABC reads the stream row that leads to its plane: on east slices
for the west plane and on west slices for the east plane."""
from dataclasses import dataclass
from typing import Optional, Tuple

from opentsp_tools import stations
from opentsp_tools.programs.kernels import install as inst

SHF = (122, 123)              # SXM.W.SHF[0], [1]: one byte plane each
SEGMENTS = 5
SEG_LANES = 64
NLW = 20                      # LW cycles of an install of 160 rows
H_STAGE = (16, 17)            # west slices: the f16 hidden vector (low, high plane), word H_STAGE_ADDR + row
H_STAGE_ADDR = 0x820
HK_WEST, HK_EAST = (18, 19), (70, 71)      # the segment vectors for the east and for the west plane
HK_ADDR = 0x800               # word of row r, segment k: HK_ADDR + 5 r + k
LOGIT_STRIDE = 63 * SEGMENTS  # words of one logit row in the logits region (the layout's 315)


class Pool:
    """Streams lo..hi-1 of a row in rotation for the values of one kind. The ABC reads of one plane and the ACC results
    of the other travel on the same row: kept in disjoint pools, a value never meets, on its diagonal, one of another
    kind (the shared rotation of `Flow.stream` follows the order of generation, not of time)."""

    def __init__(self, lo, hi, width):
        self.lo, self.hi, self.width, self.next = lo, hi, width, lo

    def take(self):
        if self.next + self.width > self.hi:
            self.next = self.lo
        self.next += self.width
        return self.next - self.width


@dataclass(frozen=True)
class Head:
    west_addr: int            # word of install 0's first LW on the west plane's slices 0..15
    east_addr: int            # the same on slices 44..59
    installs: int
    logit_slices: Tuple[int, ...]
    logit_addr: int
    logit_slices_east: Optional[Tuple[int, ...]] = None     # the east plane's results go to this quad (the same words), when given
    logit_addr_east: Optional[int] = None
    segments: int = SEGMENTS      # vocabulary rows per packed row (the profile's `pack`)
    nlw: int = NLW                # LW cycles of an install: its packed rows / 8
    stride: int = LOGIT_STRIDE    # words of one logit row in the logits region: installs * segments

    def results(self, plane):
        """(slices, word of logit row 0 install 0 segment 0) the ACC results of `plane` are written to."""
        if plane == 2 and self.logit_slices_east:
            return self.logit_slices_east, self.logit_addr_east
        return self.logit_slices, self.logit_addr


def hidden_segments(f, h, r, t_min=0, segments=SEGMENTS):
    """Round the f32 Vec h (lanes 0..63, +0 beyond) to f16, keep it, and write the five shifted segment vectors of
    logit row r to both copies. Returns the first cycle at which a Read of them may be dispatched (after their last
    Write)."""
    b, p = f.b, f.p
    hc = f.vxm(0, "conv_f32_f16.f32", [h], to="W", width=2, at=t_min)
    t_w = b.write(hc, list(H_STAGE), [H_STAGE_ADDR + r] * 2)
    d_read, sk_shf = p.timing("read")[0], p.timing("shift")[1]
    g_w = stations.need_gap(p, H_STAGE[0], "W", "write")
    g_s = stations.need_gap(p, SHF[0], "W", "read")
    t_s = t_w + 1 + (g_s - g_w) + d_read - sk_shf
    t_last = 0
    for k in range(segments):
        vs = b.read_for(t_s + k, "shift", SHF[0], list(H_STAGE), [H_STAGE_ADDR + r] * 2, "W", f.stream("W", 2))
        for pl in range(2):
            s_out = f.stream("E", 1)
            _, sv = b.unit(SHF[pl], "shift", f"shift W:{vs.stream + pl}, up, {SEG_LANES * k:#x}, E:{s_out}",
                           [vs.planes(pl, 1)], dst=("E", s_out, 1))
            for slices in (HK_WEST, HK_EAST):
                t_last = max(t_last, b.write(sv, [slices[pl]], [HK_ADDR + segments * r + k]))
    return t_last + 1


def lm_head(f, hs, head, t0=30):
    """The logits of the f32 Vecs `hs` (one logit row each). The installs start so that their first ABC finds the segment
    vectors written; each plane installs every 40 cycles (the next LW waits for the IW, which waits for the last LW).
    Returns the cycle of the last Write."""
    k, b, p = f.k, f.b, f.p
    f.low = {"E": 16, "W": 16}
    ready = 0
    for r, h in enumerate(hs):
        ready = max(ready, hidden_segments(f, h, r, ready, head.segments))
    d_abc = p.timing("abc")[0]
    d_read = p.timing("read")[0]
    sk_abc = p.timing("abc")[1]
    lead = {}
    for plane in (0, 2):
        hk = HK_EAST if plane == 0 else HK_WEST
        row_in = "W" if plane == 0 else "E"
        lead[plane] = (stations.need_gap(p, inst.plane_queues(plane)[2] + 1, row_in, "read")
                       - stations.need_gap(p, hk[0], row_in, "write") + d_read - sk_abc)
    # row E carries the west plane's results and the east plane's reads of h_k; row W the reverse
    reads = {0: Pool(16, 24, 2), 2: Pool(24, 32, 2)}
    outs = {0: Pool(16, 24, 4), 2: Pool(24, 32, 4)}
    free = {0: t0, 2: t0 + 1}
    first_abc = {pl: ready + lead[pl] for pl in (0, 2)}
    d_iw = p.timing("iw")[0]
    nlw_lat = head.nlw - 1 + p.timing("lw")[0]        # the IW follows the first LW by this many cycles
    last_write = 0
    for i in range(head.installs):
        plane = 2 * (i % 2)
        base = 0 if plane == 0 else 44
        off = head.nlw * (i // 2)
        addr = (head.west_addr if plane == 0 else head.east_addr) + off
        hk = HK_EAST if plane == 0 else HK_WEST
        row_in, row_out = ("W", "E") if plane == 0 else ("E", "W")
        lw = inst.plane_queues(plane)[2]
        t_first = inst.install(k, "f16", head.nlw, True, addr, max(free[plane], first_abc[plane] - nlw_lat - d_iw), plane,
                               text=lambda j, w, o=off, bs=base: f"hd{bs + j}+{o + w:#x}")
        # the two planes' ACC results reach the logit slices an odd number of cycles after the ABC, one per 2 cycles: the west
        # plane's ABCs go on even cycles and the east plane's on odd ones, so that their Writes never meet
        t_abc = t_first + (t_first - plane // 2) % 2
        for r in range(len(hs)):
            for seg in range(head.segments):
                v = b.read_for(t_abc, "abc", lw + 1, list(hk), [HK_ADDR + head.segments * r + seg] * 2, row_in,
                               reads[plane].take())
                t, _ = b.unit(lw + 1, "abc", f"abc.f16 {row_in}:{v.stream}", [v])
                s = outs[plane].take()
                _, acc = b.unit(lw + 2, "acc", f"acc.f32 {row_out}:{s}, replace, 0", cycle=t + d_abc, dst=(row_out, s, 4))
                out_slices, out_addr = head.results(plane)
                last_write = max(last_write, b.write(acc, list(out_slices),
                                                     [out_addr + head.stride * r + head.segments * i + seg] * 4))
                t_abc += 2
        # the next install's LWs follow this IW, and its IW follows this install's last ABC
        free[plane] = max(t_first - d_iw + 1, t_abc - 2 - nlw_lat)
    k.last = max(k.last, last_write)
    return last_write
