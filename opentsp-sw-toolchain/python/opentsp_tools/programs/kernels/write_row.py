"""write_row: the current token's K and V rows into their fixed cache slot : the K row is narrowed to
f16 and written into the install layout of the K cache, the V row is copied through the streams to its slot of the V
cache (the f32 planes on the four slices of one quad)."""
from opentsp_tools.programs.kernels import install as inst
from opentsp_tools.programs.kernels.flow import READ_ROW

K_PLANE = 2


def write_k(f, row, slot, plane=K_PLANE, alu=0, addr=inst.ADDR_W):
    """The f32 Vec `row` narrowed to f16 on VXM[alu] and written to the K cache slot `slot` of the matrix installed
    in `plane`: slices j and j + 1 (low and high byte) of the plane's hemisphere, j = 2 (slot mod 8), word
    `addr` + slot div 8. Returns the cycle of the Write."""
    slices = inst.group(f.p, plane)
    j = 2 * (slot % 8)
    hemi = "E" if slices[j] >= 44 else "W"          # the pair of the slot is in the plane's hemisphere, or in the other one
    v = f.vxm(alu, "conv_f32_f16.f32", [row], to=hemi, width=2)
    return f.b.write(v, [slices[j], slices[j + 1]], [addr + slot // 8] * 2)


def write_v(f, row, slot, quad, base_addr):
    """The f32 Vec `row` copied to word base_addr + slot of the four slices of `quad` of the other hemisphere (the V
    cache). Returns the cycle of the Write."""
    b = f.b
    src = READ_ROW[row.hemi]
    hemi = "E" if row.hemi == "W" else "W"
    first = (0 if hemi == "W" else 44) + 4 * f.fold(quad)
    v = b.read(max(row.ready, 0), row.slices, [row.addr] * row.width, src, f.stream(src, row.width))
    return b.write(v, list(range(first, first + row.width)), [base_addr + slot] * row.width)
