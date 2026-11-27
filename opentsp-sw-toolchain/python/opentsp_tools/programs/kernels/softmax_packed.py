"""Packed softmax : the score vectors of several heads share one vector, one head per segment of C/16
superlanes (on 320 lanes 5 heads at C = 64, 2 at C = 128; in general `lanes // C`), so that one softmax call does the work of several. Each head is moved into
its segment by two lane shifts with zero fill, one on an east and one on a west shifter (`place`): the first pushes every
lane at or above C out of the vector, so the segment holds the head's C scores and every other lane +0. A pack is the sum
of its placed heads (x + (+0) = x for every score: a visible score is never -0, it is a sum with the mask's +0), so each
segment of the pack is bit for bit the head's score vector, and the softmax of a segment uses that segment's lanes only.
After the softmax two shifts bring head k of a pack back to lanes 0..C-1 (the lanes above hold the next heads'
probabilities: finite, and `expand` multiplies them by +0)."""
from opentsp_tools import stations
from opentsp_tools.stations import ScheduleError
from opentsp_tools.programs.trial import Trial
from opentsp_tools.programs.kernels.flow import Vec

SHF_E = (136, 137)     # SXM.E.SHF[0], [1]: read row E, write row W
SHF_W = (122, 123)     # SXM.W.SHF[0], [1]: read row W, write row E
VLANES = 320           # the lanes of `ref`: the default of the functions below, which take the profile's own count (`lanes`)
TRIALS = 400           # start cycles a pack's sum tries before it reports the failure (each retry moves the sum one cycle later)


MAX_HEADS = 5          # heads of one pack: the sum of a pack is one add of two heads and one chained add on each of the other VXM ALUs


def per_vector(C, lanes=VLANES):
    """Heads per packed vector: as many segments of C lanes as the vector holds, up to what one chain of adds sums."""
    return min(lanes // C, MAX_HEADS)


def placement(C, k, lanes=VLANES):
    """The two shifts [(direction, lanes)] that move lanes 0..C-1 to segment k and clear every other lane (the first pushes
    the lanes at or above C off the end of the vector)."""
    second = lanes - C - C * k
    if second > 0:
        return [("up", lanes - C), ("down", second)]
    return [("up", lanes - C - 16), ("up", 16)]


def retrieval(C, k):
    """The two shifts that bring segment k down to lanes 0..C-1."""
    return [("down", 16), ("down", C * k - 16)]


def _shift_pair(f, src, dst, shifts, t0):
    """Plane by plane, read the Vec `src` from its east quad onto row E, shift it on an east shifter (onto row W) and on a
    west shifter (onto row E), and write it to the east Vec `dst`'s word. Returns the cycle of the last Write."""
    b, p = f.b, f.p
    tr = b.t
    d_read = p.timing("read")[0]
    t_last = 0
    for pl, (s_in, s_out) in enumerate(zip(src.slices, dst.slices)):
        qe, qw = SHF_E[pl % 2], SHF_W[pl % 2]
        g_w = stations.need_gap(p, s_in, "E", "write")
        g_r = stations.need_gap(p, qe, "E", "read")
        t = max(t0, src.ready + (g_r - g_w) + d_read - p.timing("shift")[1])
        while True:
            t_rd = b.read_cycle_for(t, "shift", qe, s_in, "E")
            g1, c1 = stations.commit(p, t, "shift", qe, "W")
            t2 = stations.meet(p, g1, c1, "shift", qw, "W")
            g2, c2 = stations.commit(p, t2, "shift", qw, "E")
            t_wr = stations.meet(p, g2, c2, "write", s_out, "E")
            if (tr.free(qe, t) and tr.free(qw, t2) and tr.free(s_in, t_rd, "R") and tr.free(s_out, t_wr, "W")
                    and tr.bank_ok(s_in, t_rd, "R", src.addr) and tr.bank_ok(s_out, t_wr, "W", dst.addr)):
                break
            t += 1
        g_rd, c_rd = stations.commit(p, t_rd, "read", s_in, "E")
        si = _stream(f, "E", g_rd, c_rd, stations.need_gap(p, qe, "E", "read"))
        v = b.read(t_rd, [s_in], [src.addr], "E", si)
        (d1, a1), (d2, a2) = shifts
        o1 = _stream(f, "W", g1, c1, stations.need_gap(p, qw, "W", "read"))
        _, v1 = b.unit(qe, "shift", f"shift E:{v.stream}, {d1}, {a1:#x}, W:{o1}", [v], dst=("W", o1, 1), cycle=t)
        o2 = _stream(f, "E", g2, c2, stations.need_gap(p, s_out, "E", "write"))
        _, v2 = b.unit(qw, "shift", f"shift W:{o1}, {d2}, {a2:#x}, E:{o2}", [v1], dst=("E", o2, 1), cycle=t2)
        t_last = max(t_last, b.write(v2, [s_out], [dst.addr], cycle=t_wr))
    f.k.last = max(f.k.last, t_last)
    return t_last


def _stream(f, row, g_w, c_w, g_r):
    """A stream of `row` (from the Flow's rotation) on which a value committed at gap g_w in cycle c_w reaches gap g_r."""
    for _ in range(f.p.streams):
        s = f.stream(row, 1)
        if f.b.t.stream_free(row, s, g_w, c_w, g_r):
            return s
    raise ScheduleError(f"no free stream on row {row} from gap {g_w} at cycle {c_w}")


def place(f, sc, C, t0=0):
    """Move the east Vecs `sc` (one score vector per head, lanes 0..C-1) into their segments in place. Returns the Vecs."""
    out = []
    lanes = f.p.vlanes
    for h, v in enumerate(sc):
        k = h % per_vector(C, lanes)
        t = _shift_pair(f, v, v, placement(C, k, lanes), t0)
        out.append(Vec(v.slices, v.addr, t + 1, v.hemi))
    return out


def _sum(f, group, alus, quad, at):
    """The sum of `group` on the ALUs `alus` (the first two heads, then one more per chained add) stored on `quad` from the
    first cycle `at` on. Returns (the Vec, the dispatch cycle of the first add). The chained adds follow the first at fixed
    cycles, so each of their Reads is placed by the first add's cycle: the caller retries from a later `at` when one meets a
    Write of the same bank in its slice (a conflict the latencies of the profile decide)."""
    first = [None]
    try:
        acc = f.vxm(alus[0], "add.f32", [group[0], group[1]], at=at)
        first[0] = acc.cycle - f.p.timing("add.f32")[0]
        for i, v in enumerate(group[2:]):
            acc = f.vxm(alus[i + 1], "add.f32", [acc, v], chain=(0,))
        return f.store(acc, quad, hemi="E"), first[0]
    except ScheduleError as e:
        e.first_add = first[0]
        raise


def packs(f, placed, C, quads, alus=(8, 9, 10, 11)):
    """The packed score Vecs: the sum of each group of placed heads, stored on `quads` in turn (east). A sum that does not
    fit from its first cycle (a Read of the chain on a Write of the same bank, whatever the profile's latencies make of the
    cycles of the shifts that wrote the heads) is tried one cycle later, up to `TRIALS` times, in a `Trial`."""
    n = per_vector(C, f.p.vlanes)
    out = []
    for j in range(0, len(placed), n):
        group = placed[j:j + n]
        if len(group) == 1:
            out.append(group[0])
            continue
        quad, at, last = quads[(j // n) % len(quads)], 0, None
        for _ in range(TRIALS):
            err = None
            with Trial(f) as tri:
                try:
                    v, first = _sum(f, group, alus, quad, at)
                except ScheduleError as e:
                    err = e
                    raise
            if tri.ok:
                out.append(v)
                break
            first = getattr(err, "first_add", None)
            at = (first if first is not None else at) + 1
            last = err
        else:
            raise ScheduleError(f"pack {j // n}: no start for the sum of its heads in {TRIALS} trials ({last})")
    return out


def unpack(f, probs, C, heads, quads):
    """Per head, its probabilities in lanes 0..C-1: head 0 of a pack is the pack itself, head k > 0 a copy shifted down
    by k segments onto `quads` in turn (east). Returns one Vec per head."""
    n = per_vector(C, f.p.vlanes)
    out = []
    for h in range(heads):
        pv, k = probs[h // n], h % n
        if k == 0:
            out.append(pv)
            continue
        quad = quads[h % len(quads)]
        a = f.addr("E", quad, 1)
        base = 44 + 4 * quad
        dst = Vec(list(range(base, base + 4)), a, 0, "E")
        t = _shift_pair(f, pv, dst, retrieval(C, k), 0)
        out.append(Vec(dst.slices, a, t + 1, "E"))
    return out
