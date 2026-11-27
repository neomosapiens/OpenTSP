"""kv_write: the new token's K and V rows into the paged pool . The slice a K row lives in depends on
its position (slice pair o mod 8), which a static program cannot choose, so all eight pairs write the row in the
window: the host patches the pair of the row to the block's word and the other seven to the sink block. A V row's slice
is fixed (the layer's V quad): only its address is patched. Every Write is a pool write window (Tracker.window): no
Read of that slice in the pool's bank shares its cycle, whatever the patched address."""
from opentsp_tools.programs.kernels import install as inst
from opentsp_tools.programs.kernels.flow import READ_ROW

K_PLANE = 2


def site(prefix, pair, plane=None):
    return f"{prefix}.p{pair}" + ("" if plane is None else f".{plane}")


def write_k_paged(f, row, addrs, krange, plane=K_PLANE, alu=0, prefix="kw"):
    """The f32 Vec `row` narrowed to f16 on VXM[alu] and written by each of the eight slice pairs r of the K slices
    of `plane`: low byte plane to slice base + 2 r, high plane to base + 2 r + 1, at word `addrs[r]` (the
    placeholder; patch sites `<prefix>.p<r>.<plane>` with the range `krange` = (lo, hi, stride, span), in the pool's bank). Returns
    the cycle of the last Write."""
    b = f.b
    _, hemi, _ = inst.plane_queues(plane)
    slices = inst.group(f.p, plane)
    v = f.vxm(alu, "conv_f32_f16.f32", [row], to=hemi, width=2)
    return max(b.write(v, [slices[2 * r], slices[2 * r + 1]], [addrs[r]] * 2, window=True,
                       patch=[(site(prefix, r, pl), *krange) for pl in range(2)]) for r in range(8))


def write_v_paged(f, row, slices, addr, vrange, prefix="vw"):
    """The f32 Vec `row` copied to word `addr` (the placeholder: the block's address plus the row) of the 4 `slices` of
    the layer's V quad, which are in the other hemisphere (patch sites `<prefix>.s<slice>`, range `vrange`). Returns
    the cycle of the Write."""
    b = f.b
    src = READ_ROW[row.hemi]
    v = b.read(max(row.ready, 0), row.slices, [row.addr] * row.width, src, f.stream(src, row.width))
    return b.write(v, slices, [addr] * len(slices), window=True,
                   patch=[(f"{prefix}.s{s}", *vrange) for s in slices])
