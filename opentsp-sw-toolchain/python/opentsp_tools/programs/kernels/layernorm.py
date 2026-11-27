"""layernorm: two-pass mean and variance in f32 over one segment of four superlanes, then the affine transform
: reduce_bcast, mean, centred values, reduce_bcast of their squares, variance, epsilon, RSqrt and
the scale and shift as chained VXM operations."""
from opentsp_tools.programs.kernels import reduce_bcast as rb
from opentsp_tools.programs.kernels.flow import Vec


def broadcast_vec(f, op, v, a0, out_addr, maps, chain=False):
    """reduce_bcast of f32 (Value v) as a Vec on west quad 0: every lane holds its segment's total. `maps` =
    (ready cycle of the rotation maps, of the Distribute map[, rounds of the tree, base of its maps, id of the
    Distribute map: 2, 0 and the one after the maps if absent]); see reduce_bcast.load_maps."""
    slices, addr, ready = rb.reduce_bcast(f.k, op, v, a0, out_addr, maps[0], maps[1], chain, *maps[2:])
    return Vec(slices, addr, ready, "W")


def layernorm(f, x, gamma, beta, inv_n, eps, maps, a0, out_addr, to="E", at=0):
    """LayerNorm of the Vec x (f32): the Vecs gamma, beta, inv_n (1 / lanes of a segment) and eps are splat or
    affine constants; `maps` = (first cycle the rotation maps are ready, first cycle the Distribute map is ready);
    a0 and out_addr (+ 1) are the words the two reduce_bcasts use (8 words from a0 on east slices, bank 1 words on
    west quad 0); the first Reduce is at cycle `at` or later. Returns the result Value, on row `to`."""
    t = max(at, f.bound(x, "reduce.add.f32", 88))
    while not f.readable(x, t, "reduce.add.f32", 88):          # a Write of the bank in that cycle (another lane's store) moves it on
        t += 1
    vx = f.read(x, t, "reduce.add.f32", 88)
    mean = f.vxm(2, "mul.f32", [broadcast_vec(f, "add", vx, a0, out_addr, maps), inv_n])
    d = f.vxm(3, "sub.f32", [x, mean], chain=(1,))
    centred = f.store(d, 1)
    squares = f.vxm(4, "mul.f32", [centred, centred])
    var = f.vxm(5, "mul.f32", [broadcast_vec(f, "add", squares, a0 + 8, out_addr + 1, maps, chain=True), inv_n])
    r = f.vxm(7, "rsqrt.f32", [f.vxm(6, "add.f32", [var, eps], chain=(0,))], chain=(0,))
    scaled = f.vxm(9, "mul.f32", [f.vxm(8, "mul.f32", [centred, r], chain=(1,)), gamma], chain=(0,))
    return f.vxm(10, "add.f32", [scaled, beta], chain=(0,), to=to)
