"""wide_softmax: the softmax of one head over a pool of several chunks of 128 slots (SHIP full-pool attention; `ref_wide.softmax_pool`). A chunk's
scores are an f32 vector whose lanes 0..127 are the 128 slots (8 superlanes). The maximum (and later the sum of the exponentials) is combined over the
chunks elementwise, in ascending chunk order, on a chain of VXM ALUs; the tree of 8 superlanes and Distribute (`layernorm.broadcast_vec` with three
rounds) then give every lane its head's value; each chunk is `exp(s - max)` and the sum's reciprocal is `rsqrt(sum * sum)`."""
from opentsp_tools.programs.kernels.layernorm import broadcast_vec

CHAIN = 8        # the ALUs 8.. combine the chunks


def combine(f, op, vecs, at=0):
    """`op` (max, add) of the f32 Vecs `vecs` in ascending order as a chain of VXM ALUs (the previous result chained, the next Vec read from MEM);
    returns the Value of the last ALU, whose commit the next operation takes chained."""
    acc = f.vxm(CHAIN, f"{op}.f32", [vecs[0], vecs[1]], at=at)
    for i in range(2, len(vecs)):
        acc = f.vxm(CHAIN + (i - 1) % (16 - CHAIN), f"{op}.f32", [acc, vecs[i]], chain=(0,))
    return acc


def softmax_pool(f, scores, maps, a0, out_addr, quads, at=0):
    """Softmax over the chunk Vecs `scores` (f32, 2 or more; adjacent scores can be read together). `maps` as for
    `layernorm.broadcast_vec` (three rounds: (ready, dist ready, 3, base, Distribute id)); a0, a0 + 8 and out_addr, out_addr + 1 are the words of
    the two trees. `quads` are the east quads (one per chunk) that keep the exponentials (bank 1) and then hold the probabilities (bank 0).
    The first combine is at cycle `at` or later (a previous head's trees use the same slices). Returns the list of probability Vecs, one per chunk."""
    m = broadcast_vec(f, "max", combine(f, "max", scores, at), a0, out_addr, maps, chain=True)
    kept = []
    for s, q in zip(scores, quads):
        e = f.vxm(4, "exp.f32", [f.vxm(3, "sub.f32", [s, m])], chain=(0,))
        kept.append(f.store(e, q))
    total = broadcast_vec(f, "add", combine(f, "add", kept), a0 + 8, out_addr + 1, maps, chain=True)
    r = f.store(f.vxm(6, "rsqrt.f32", [f.vxm(5, "mul.f32", [total, total])], chain=(0,)), 9)
    return [f.store(f.vxm(7, "mul.f32", [e, r]), q, bank=0) for e, q in zip(kept, quads)]
