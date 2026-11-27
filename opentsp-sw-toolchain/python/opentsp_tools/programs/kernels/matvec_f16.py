"""matvec_f16: y = W x + b with f16 weights in MXM plane 0 and an f32 activation : the weights are
installed (install), the activation is narrowed to f16 on the VXM, ABC and ACC take the exact dot product rounded
once to f32, and the VXM adds the f32 bias."""
from opentsp_tools.programs.kernels import install as inst


def matvec_f16(f, rows, x, bias, t0=30, alu=0, to="E"):
    """Install (LW from cycle t0) the `rows`-row matrix bound by install.bind_weights, narrow the Vec x (f32) to f16 on
    VXM[alu], ABC and ACC on plane 0, add the Vec `bias` on VXM[alu + 1]. Returns the result Value (f32, row `to`):
    lane l is row l, the lanes of the rows not installed are zero."""
    k, b, p = f.k, f.b, f.p
    t_abc = inst.install(k, "f16", rows // 8, True, inst.ADDR_W, t0)
    d_conv = p.timing("conv_f32_f16.f32")[0]
    abc, acc_q = inst.plane_queues(0)[2] + 1, inst.plane_queues(0)[2] + 2
    transit = f.transit(abc, "W", alu)
    xc = f.vxm(alu, "conv_f32_f16.f32", [x], to="W", width=2, at=t_abc - d_conv - transit + p.timing("abc")[1])
    t, _ = b.unit(abc, "abc", f"abc.f16 W:{xc.stream}", [xc])
    s = f.stream("E", 4)
    _, acc = b.unit(acc_q, "acc", f"acc.f32 E:{s}, replace, 0", cycle=t + p.timing("abc")[0], dst=("E", s, 4))
    return f.vxm(alu + 1, "add.f32", [acc, bias], to=to)



def matvec(f, rows, x, bias, plane, addr, text, t0, alu=0, to="E"):
    """`matvec_f16` for any f16 plane (0 west, 2 east) and any matrix: the `rows`-row matrix at word `addr` of the
    plane's 16 slices (`text(j, i)` names word i of slice j) is installed (LW from cycle t0), the f32 Vec x is
    narrowed on VXM[alu], and the ACC result, an f32 vector whose lane l is row l, is returned as it is (`bias`
    None) or with the f32 Vec `bias` added on VXM[alu + 1] onto row `to`. The ABC reads row W on a west plane and
    row E on an east plane; the ACC writes the other row."""
    k, b, p = f.k, f.b, f.p
    t_abc = inst.install(k, "f16", rows // 8, True, addr, t0, plane, text=text)
    d_conv = p.timing("conv_f32_f16.f32")[0]
    lw = inst.plane_queues(plane)[2]
    abc, acc_q = lw + 1, lw + 2
    row_in, row_out = ("W", "E") if plane < 2 else ("E", "W")
    transit = f.transit(abc, row_in, alu)
    xc = f.vxm(alu, "conv_f32_f16.f32", [x], to=row_in, width=2, at=t_abc - d_conv - transit + p.timing("abc")[1])
    t, _ = b.unit(abc, "abc", f"abc.f16 {row_in}:{xc.stream}", [xc])
    s = f.stream(row_out, 4)
    _, acc = b.unit(acc_q, "acc", f"acc.f32 {row_out}:{s}, replace, 0", cycle=t + p.timing("abc")[0],
                    dst=(row_out, s, 4))
    return acc if bias is None else f.vxm(alu + 1, "add.f32", [acc, bias], to=to)
