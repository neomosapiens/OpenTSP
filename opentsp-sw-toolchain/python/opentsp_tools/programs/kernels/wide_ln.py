"""wide_ln: LayerNorm over a feature vector held as several 256-lane vectors (GPT-2 small: 768 features, three vectors).

The total of a quantity over all features (mean, then sum of squared deviations) is the arithmetic of `ref_wide._vector_total`:
Reduce (per superlane) of each vector, the elementwise f32 sums of the vectors in ascending order, then the four-round butterfly
over the 16 superlanes of the 256 lanes (rotations 8, 4, 2, 1) and Distribute. The new pieces over `reduce_bcast` are the head
(one Reduce per vector, the sums through MEM) and the fourth round; the rounds and the tail are `reduce_bcast.tree_round` and the
Distribute sequence of `reduce_bcast.reduce_bcast`."""
from opentsp_tools import stations
from opentsp_tools.programs.kernels import reduce_bcast as rb
from opentsp_tools.programs.kernels.flow import Vec

ROUNDS = 4                 # 16 superlanes: rotations 8, 4, 2, 1
ALUS = (1, 2, 3, 4)        # the VXM ALUs that combine the rounds (the Reduce is on ALU 0)
PRE = 8                    # offset of the words that hold the per-vector Reduce results


def words(a0, bank1):
    """(source, destination, scratch) words of the four rounds at a0 (a round reads one bank and writes the other)."""
    return [(a0, a0 + bank1, a0 + 2 + bank1), (a0 + bank1, a0 + 1, a0 + 3 + bank1),
            (a0 + 1, a0 + 4 + bank1, a0 + 5 + bank1), (a0 + 4 + bank1, a0 + 2, a0 + 6 + bank1)]


def round_words(a0, bank1, rounds):
    return words(a0, bank1) if rounds == 4 else rb.tree_words(a0, bank1, rounds)


def total_words(f, op, sources, a0, t_ready, chain=False, rounds=ROUNDS):
    """The `op` (add) of f32 over the superlanes of 16 and over every vector of `sources` (Values on a stream for the Reduce
    on VXM[0], chained to their producer when `chain`): per vector a Reduce written to east slices 44..47, word a0 + PRE of the quad of vector i (east quad i);
    the vectors' partial results summed in ascending order on ALUs 1 and 2 and written to word a0; the four rounds.
    Returns (the cycle of the last Write, the word of the totals on east slices 44..)."""
    k, b, bank1 = f.k, f.b, 1 << f.p.bank_bit
    assert 2 <= len(sources) <= 6, len(sources)
    parts = []
    for i, vx in enumerate(sources):
        vr = b.vxm(0, f"reduce.{op}.f32", "E", f.stream("E", 4), [vx], width=4, chain=(0,) if chain else ())
        sl = [44 + 4 * i + j for j in range(4)]       # one quad per vector: the sums read two of them in one cycle
        t = b.write(vr, sl, [a0 + PRE] * 4)
        parts.append(Vec(sl, a0 + PRE, t + 1, "E"))
    at = f.bound(parts[-1], f"{op}.f32", 88 + len(parts) - 1) if len(parts) > 2 else 0     # the chained sums must find their vectors ready
    s = f.vxm(1, f"{op}.f32", [parts[0], parts[1]], to="E", at=at)
    for i in range(2, len(parts)):
        s = f.vxm(i, f"{op}.f32", [s, parts[i]], to="E", chain=(0,))
    t_w = b.write(s, [44, 45, 46, 47], [a0] * 4)
    rounds = round_words(a0, bank1, rounds)
    for rnd, ws in enumerate(rounds):
        t_w = rb.tree_round(k, "f32", op, 4, ws, rnd + 1, ALUS[rnd], t_ready, t_w)
    return t_w, rounds[-1][1]


def broadcast_total(f, op, sources, a0, out_addr, maps, chain=False, rounds=ROUNDS):
    """total_words followed by Distribute (the sequence of `reduce_bcast.reduce_bcast`): a Vec on west quad 0 at `out_addr` in which
    every lane of a 16-superlane segment holds the segment's total. `maps` = (first cycle the rotation maps are ready, first
    cycle the Distribute map (id 5) is ready)."""
    k, p, b = f.k, f.p, f.b
    t_w, a2 = total_words(f, op, sources, a0, maps[0], chain, rounds)
    d_read = p.timing("read")[0]
    g_w, g_r = stations.need_gap(p, 44, "W", "write"), stations.need_gap(p, 116, "W", "read")
    t_d = max(maps[1], t_w + 1 + (g_r - g_w) + d_read)
    t_end = 0
    for u in range(4):
        vp = b.read_for(t_d + u, "distribute", 116 + u, [44 + u], [a2], "W", 8 + u)
        _, vd = b.unit(116 + u, "distribute", f"distribute W:{8 + u}, {rb.dist_map(rounds):#x}, W:{12 + u}", [vp], dst=("W", 12 + u, 1))
        t_sel = b.when(vd, "sxselect", 125)
        vz = b.read_for(t_sel, "sxselect", 125, [7], [rb.ZERO_WORD], "W", 16 + u)
        _, vo = b.unit(125, "sxselect", f"sxselect W:{12 + u}, W:{12 + u}, W:{12 + u}, W:{16 + u}, E:{20 + u}", [vd, vd, vd, vz],
                       dst=("E", 20 + u, 1))
        t_end = max(t_end, b.write(vo, [u], [out_addr]))
    k.last = max(k.last, t_end)
    return Vec(list(range(4)), out_addr, t_end + 1, "W")


def layernorm_wide(f, xs, gammas, betas, inv_n, eps, maps, a0, out_addr, quads=(0, 1, 4), to="E", rounds=ROUNDS, at=0):
    """LayerNorm of the f32 Vecs `xs` (one feature vector in 256-lane tiles); gammas and betas are lists of Vecs, inv_n and eps splat
    Vecs; a0 and a0 + 16 are the words of the two totals, out_addr and out_addr + 1 those of their broadcasts. Returns the list of
    result Vecs (`at`: the first cycle the LayerNorm may start; f32, the mean in east quad quads[0], the centred tiles from quad quads[1], 1 / sigma in quad quads[2])."""
    reads = [f.read(x, max(at, f.bound(x, "reduce.add.f32", 88)) + 4 * i, "reduce.add.f32", 88) for i, x in enumerate(xs)]
    mean = f.vxm(2, "mul.f32", [broadcast_total(f, "add", reads, a0, out_addr, maps, rounds=rounds), inv_n])
    mean_v = f.store(mean, quads[0])
    centred = [f.store(f.vxm(3, "sub.f32", [x, mean_v]), quads[1] + i) for i, x in enumerate(xs)]   # one quad each: the later
    # stores must not share a bank with the earlier reads
    squares = [f.vxm(4, "mul.f32", [c, c]) for c in centred]
    var = f.vxm(5, "mul.f32", [broadcast_total(f, "add", squares, a0 + 16, out_addr + 1, maps, chain=True, rounds=rounds), inv_n])
    r = f.vxm(7, "rsqrt.f32", [f.vxm(6, "add.f32", [var, eps], chain=(0,))], chain=(0,))
    r_v = f.store(r, quads[2])
    out = []
    for c, g, bta in zip(centred, gammas, betas):
        scaled = f.vxm(9, "mul.f32", [f.vxm(8, "mul.f32", [c, r_v]), g], chain=(0,))
        out.append(f.vxm(10, "add.f32", [scaled, bta], chain=(0,), to=to))
    return out
