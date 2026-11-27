"""scores: attention scores of every head against the K cache : the cache is installed as an f16
matrix in east plane 2 (row s is slot s), the query is narrowed to f16 once, and each head contributes one ABC of the
query masked to its lanes and one ACC; the VXM scales by 0.5 and adds the host's mask (0 or -inf per slot)."""
from opentsp_tools.programs.kernels import install as inst
from opentsp_tools.programs.kernels.head_mask import head_mask

PLANE = 2
ABC_SPACING = 2      # an f16 ABC holds its plane for two cycles


def scores(f, slots, q, masks, half, add_mask, t0=30, addr=inst.ADDR_W, text=None, install=None):
    """Install the `slots`-row cache bound by install.bind_weights(plane=PLANE, prefix 'k') (or, with `install`, a
    function (k, t0) -> the cycle of the first ABC that installs it, as kv_install.install_k), then for every head h
    (the Vecs `masks`) ABC and ACC the masked query and compute acc * half + add_mask on VXM[5 + 2 (h mod 5)] and
    the next ALU. q is the f32 query Vec. Returns the f32 result Values, one per head: lane s is the score of slot s.
    The cache is at word `addr` of the plane's slices (`text(j, i)` names the words of the LWs)."""
    b, p = f.b, f.p
    t_abc = install(f.k, t0) if install else inst.install(f.k, "f16", slots // 8, True, addr, t0, PLANE, "k", text)
    lw = inst.plane_queues(PLANE)[2]
    abc, acc_q = lw + 1, lw + 2
    q16 = f.store(f.vxm(0, "conv_f32_f16.f32", [q], to="W", width=2), 3)
    d_mask = p.timing("mask.f32")[0]
    transit = f.transit(abc, "E", 0)
    out, t_mask = [], t_abc - d_mask - transit + p.timing("abc")[1]
    for h, mask in enumerate(masks):
        m = head_mask(f, q16, mask, 1 + h % 4, at=t_mask)
        t_mask = m.cycle - d_mask + ABC_SPACING
        t, _ = b.unit(abc, "abc", f"abc.f16 E:{m.stream}", [m])
        s = f.stream("W", 4)
        _, acc = b.unit(acc_q, "acc", f"acc.f32 W:{s}, replace, 0", cycle=t + p.timing("abc")[0], dst=("W", s, 4))
        alu = 5 + 2 * (h % 5)
        out.append(f.vxm(alu + 1, "add.f32", [f.vxm(alu, "mul.f32", [acc, half]), add_mask], chain=(0,)))
    return out
