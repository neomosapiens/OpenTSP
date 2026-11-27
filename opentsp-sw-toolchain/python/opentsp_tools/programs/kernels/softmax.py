"""softmax over each segment of four superlanes of an f32 Vec : reduce_bcast of the maximum, subtract,
Exp, reduce_bcast of the sum, the reciprocal as RSqrt of the square, multiply. A masked lane (-inf) gives +0."""
from opentsp_tools.programs.kernels.layernorm import broadcast_vec


def softmax(f, s, maps, a0, out_addr, e_quad=1, to="E", at=0):
    """Softmax of the Vec s; `maps` as for layernorm; a0 and out_addr (+ 1) are the words the two reduce_bcasts use.
    The exponentials are kept in MEM on east quad `e_quad`; the first Reduce is at cycle `at` or later. Returns the
    result Value (row `to`)."""
    vx = f.read(s, max(at, f.bound(s, "reduce.max.f32", 88)), "reduce.max.f32", 88)
    peak = broadcast_vec(f, "max", vx, a0, out_addr, maps)
    e = f.vxm(4, "exp.f32", [f.vxm(3, "sub.f32", [s, peak])], chain=(0,))
    kept = f.store(e, e_quad)
    total = broadcast_vec(f, "add", e, a0 + 8, out_addr + 1, maps, chain=True)
    r = f.vxm(6, "rsqrt.f32", [f.vxm(5, "mul.f32", [total, total])], chain=(0,))
    return f.vxm(7, "mul.f32", [kept, r], chain=(1,), to=to)
