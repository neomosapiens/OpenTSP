"""Category D of the design's sec. 2.2, SXM: lane shifts and the shifter select (ISCA20 Fig. 8; the J1 migration of
tests/e2e/shift.sh). Every cycle follows from the profile through the toolchain; expectations are hand rules stated here."""
import numpy as np

from opentsp_tools.programs.case import Kit
from basic import PROFILES
from opentsp_tools.programs import data

SRC = 0x10           # first word of every operand slice
DST = 0x20           # first word of every result slice


def shift_select(p, seed=1):
    """x is read from west slice 4 onto row E and meets both east shifters (SXM.E.SHF[0] `up`, SXM.E.SHF[1] `down`) at
    their read station; their results, committed on row W, go to MEM[52] and MEM[53] and also, with an unshifted copy
    of x (MEM[48]) and a lane selector (MEM[49]) read onto row W, to the west select (SXM.W.SEL): per lane
    sel 0 picks the copy, 1 the north (up) shift and 2 the south (down) shift (Fig. 8); the select writes row E and
    MEM[8] stores it. Shifts fill with zero; the selector holds 0, 1 and 2 only, so there are no diagnostics. Then
    the amounts 0, 1 and vlanes - 1 on the west shifters, up and down (MEM[56..61])."""
    k = Kit(p, "sxm/shift_select", "shift up and down and the shifter select", seed, module="sxm",
            tags=("sim", "e2e"))
    n = p.vlanes
    x = data.rand_int(k.rng, "i8", n)
    sel = k.rng.integers(0, 3, size=n).astype(np.int8)
    up, dn = n // 8 + 1, n // 5
    b = np.zeros(n, np.int8)
    b[up:] = x[:n - up]
    c = np.zeros(n, np.int8)
    c[:n - dn] = x[dn:]
    out = np.choose(sel.astype(np.intp), [x, b, c]).astype(np.int8)
    for nm, s, a, arr in (("x", 4, 0x10, x), ("xa", 48, 0x10, x), ("sel", 49, 0x10, sel)):
        k.extern_word(nm, s, a, bytes(arr.view(np.uint8)))
    bl = k.b
    t_shift = 60
    vx = bl.read_for(t_shift, "shift", 136, [4], [0x10], "E", 0, addr_text=["x"])
    _, vb = bl.unit(136, "shift", f"shift E:0, up, {up:#x}, W:4", [vx], ("W", 4, 1))
    _, vc = bl.unit(137, "shift", f"shift E:0, down, {dn:#x}, W:5", [vx], ("W", 5, 1))
    t_sel = bl.when(vb, "sxselect", 125)
    va = bl.read_for(t_sel, "sxselect", 125, [48], [0x10], "W", 0, addr_text=["xa"])
    vs = bl.read_for(t_sel, "sxselect", 125, [49], [0x10], "W", 8, addr_text=["sel"])
    _, vr = bl.unit(125, "sxselect", "sxselect W:0, W:4, W:5, W:8, E:12", [va, vb, vc, vs], ("E", 12, 1))
    k.last = max(bl.write(vb, [52], [0x20]), bl.write(vc, [53], [0x20]), bl.write(vr, [8], [0x20]))
    k.expect(52, 0x20, bytes(b.view(np.uint8)))
    k.expect(53, 0x20, bytes(c.view(np.uint8)))
    k.expect(8, 0x20, bytes(out.view(np.uint8)))
    # the shift amounts at the ends of the range: 0, 1 and vlanes - 1, up on SXM.W.SHF[0] and down on SXM.W.SHF[1],
    # both reading one stream of row W that slice 4 writes
    for i, amount in enumerate((0, 1, n - 1)):
        t = t_sel + 12 + 3 * i
        vw = bl.read_for(t, "shift", 122, [4], [0x10], "W", 16 + i, addr_text=["x"])
        _, vu = bl.unit(122, "shift", f"shift W:{16 + i}, up, {amount:#x}, E:{2 * i}", [vw], ("E", 2 * i, 1))
        _, vd = bl.unit(123, "shift", f"shift W:{16 + i}, down, {amount:#x}, E:{2 * i + 1}", [vw],
                        ("E", 2 * i + 1, 1))
        wu, wd = np.zeros(n, np.int8), np.zeros(n, np.int8)
        wu[amount:] = x[:n - amount]
        wd[:n - amount] = x[amount:]
        for j, (v, want) in enumerate(((vu, wu), (vd, wd))):
            s = 56 + 2 * i + j
            k.last = max(k.last, bl.write(v, [s], [0x20]))
            k.expect(s, 0x20, bytes(want.view(np.uint8)))
    return k.done("shift_select")


def transpose_group(k, t, queue, hemi, src_base, dst_base, src_slices, dst_slices, x, tag):
    """One Transpose on SXM queue `queue` of hemisphere `hemi` at cycle t: 16 words (one per stream src_base ..
    src_base + 15, from `src_slices`, quad by quad) in, 16 words (streams dst_base .., turned onto the other row
    at the unit's write station) out to `dst_slices`. Per superlane g: out_c[16g + r] = in_r[16g + c]."""
    row, other = ("W", "E") if hemi == "W" else ("E", "W")
    b = k.b
    vs = [b.read_for(t, "transpose", queue, [s], [SRC], row, src_base + r, addr_text=[f"{tag}{r}"])
          for r, s in enumerate(src_slices)]
    _, v = b.unit(queue, "transpose", f"transpose {row}:{src_base}, {other}:{dst_base}", vs,
                  dst=(other, dst_base, 16))
    n = k.p.vlanes
    want = x.reshape(16, n // 16, 16).transpose(2, 1, 0).reshape(16, n)
    for q in range(4):
        sl = dst_slices[4 * q:4 * q + 4]
        k.last = max(k.last, b.write(v.planes(4 * q, 4), sl, [DST] * 4))
        for c, s in zip(range(4 * q, 4 * q + 4), sl):
            k.expect(s, DST, bytes(want[c].view(np.uint8)))


def transpose16(p, seed=1):
    """Transpose of 16 streams, per superlane (ISCA20 Table I), on the three units that can run it: SXM.W.TSP[0] on
    streams 0..15 and SXM.W.TSP[1] on streams 16..31 in the same cycle (west slices 0..15 and 16..31 in, east
    slices 44..59 and 60..75 out), then SXM.E.TSP[0] (east slices 44..59 in, west slices 16..31 out)."""
    k = Kit(p, "sxm/transpose16", "16 x 16 transpose per superlane", seed, module="sxm")
    n = p.vlanes
    t = 45
    x = {}
    for name, hemi, slices in (("a", "W", range(0, 16)), ("b", "W", range(16, 32)), ("c", "E", range(44, 60))):
        x[name] = data.rand_int(k.rng, "i8", 16 * n).reshape(16, n)
        for r, s in enumerate(slices):
            k.extern_word(f"{name}{r}", s, SRC, bytes(x[name][r].view(np.uint8)))
    transpose_group(k, t, 120, "W", 0, 0, range(0, 16), range(44, 60), x["a"], "a")
    transpose_group(k, t, 121, "W", 16, 16, range(16, 32), range(60, 76), x["b"], "b")
    transpose_group(k, t + 40, 134, "E", 0, 0, range(44, 60), range(16, 32), x["c"], "c")
    return k.done("transpose16")


def rotate_want(x, tile):
    """The n*n outputs of `rotate` on x (vlanes lanes): output t = tr*n + tc holds, per superlane, the n x n tile at
    lanes 0..n*n-1 cyclically shifted by tr rows and tc columns; the other lanes are zero."""
    sl = x.reshape(-1, 16)
    out = []
    for tr in range(tile):
        for tc in range(tile):
            o = np.zeros_like(sl)
            for r in range(tile):
                for c in range(tile):
                    o[:, r * tile + c] = sl[:, ((r + tr) % tile) * tile + (c + tc) % tile]
            out.append(o.reshape(-1))
    return out


def rotate(p, seed=1):
    """Rotate with n = 3 on SXM.W.TSP[0] (9 streams out) and n = 4 on SXM.W.TSP[1] (16 streams out) in the same
    cycle, each result stream stored by one east slice; then dst + n^2 > 32 (n = 4, dst 20) on TSP[0]: OTS-E-OPR-002
    at its dispatch cycle, no effect."""
    k = Kit(p, "sxm/rotate", "Rotate n = 3 and n = 4 and the dst + n^2 > 32 error", seed, module="sxm")
    b, t = k.b, 45
    for q, (queue, tile, src_slice, dst_base, dst_slices, tag) in enumerate((
            (120, 3, 4, 0, range(44, 53), "a"), (121, 4, 5, 16, range(53, 69), "b"))):
        x = data.rand_int(k.rng, "i8", p.vlanes)
        k.extern_word(tag, src_slice, SRC, bytes(x.view(np.uint8)))
        vx = b.read_for(t, "rotate", queue, [src_slice], [SRC], "W", q, addr_text=[tag])
        _, v = b.unit(queue, "rotate", f"rotate W:{q}, {tile:#x}, E:{dst_base}", [vx],
                      dst=("E", dst_base, tile * tile))
        for i, (o, s) in enumerate(zip(rotate_want(x, tile), dst_slices)):
            k.last = max(k.last, b.write(v.planes(i, 1), [s], [DST]))
            k.expect(s, DST, bytes(o.view(np.uint8)))
    x = data.rand_int(k.rng, "i8", p.vlanes)
    k.extern_word("z", 6, SRC, bytes(x.view(np.uint8)))
    t_bad = t + 30
    vz = b.read_for(t_bad, "rotate", 120, [6], [SRC], "W", 8, addr_text=["z"])
    cycle, _ = b.unit(120, "rotate", "rotate W:8, 0x4, E:20", [vz])
    k.diags.append((cycle + p.timing("rotate")[1], 120, "OTS-E-OPR-002"))
    return k.done("rotate")


def permute_maps(p, seed=1):
    """LoadMap (or LoadMapWide when a lane index exceeds a byte) and Permute on SXM.W.PRM (queue 124), four maps
    loaded one after the other (S = geometry.superlanes):
      0 a bijection: dst[l] = x[m[l]], used at the first cycle the map is ready, L + S;
      1 a bijection with a quarter of the lanes masked by entries >= vlanes (zero fill), also at L + S;
      2 a bijection, first used at L + S - 1 (OTS-E-RES-001, no effect), then at L + S, which works (the early
        use needs S >= 2: a Permute in the load's own cycle sees the previous map);
      3 two entries below vlanes equal, used at L + S: OTS-E-SXM-001, no effect.
      4 (vlanes > 256 only) the low bytes of map 0 through LoadMap: a byte map cannot be bijective on more than
        256 lanes, so OTS-E-SXM-001.
    The words of the maps are in SRAM; the three working Permutes write east slices 44..46."""
    k = Kit(p, "sxm/permute_maps", "LoadMap, LoadMapWide and Permute: bijection, mask, readiness, bijectivity",
            seed, module="sxm")
    n, s, b = p.vlanes, p.superlanes, k.b
    wide = n > 256
    x_slice, map_slice = (8, 12) if p.slices_per_hemisphere > 4 else (0, 1)     # one quad per hemisphere: quad 0
    x = data.rand_int(k.rng, "i8", n)
    k.extern_word("x", x_slice, SRC, bytes(x.view(np.uint8)))
    maps = [k.rng.permutation(n) for _ in range(4)]
    sel = k.rng.random(n) < 0.25
    maps[1][sel] = k.rng.integers(n, 1024 if wide else 256, size=int(sel.sum()))
    maps[3][9] = maps[3][5]
    # (map entries, LoadMapWide?, code of the Permute at L + S or None): a byte map (LoadMap) cannot be bijective
    # when vlanes > 256, so on `ref` the last block loads the first map's low bytes with LoadMap: SXM-001
    blocks = [(m, wide, "SXM-001" if i == 3 else None) for i, m in enumerate(maps)]
    if wide:
        blocks.append((maps[0] & 255, False, "SXM-001"))
    t_load, out = 45, 0
    for i, (m, wide_load, code) in enumerate(blocks):
        planes = [(m & 255).astype(np.uint8)] + ([(m >> 8).astype(np.uint8)] if wide_load else [])
        for j, pl in enumerate(planes):
            k.extern_word(f"m{i}{j}", map_slice + j, SRC + i, bytes(pl))
        vs = [b.read_for(t_load, "loadmap", 124, [map_slice + j], [SRC + i], "W", 8 + j, addr_text=[f"m{i}{j}"])
              for j in range(len(planes))]
        mnem = "loadmapwide" if wide_load else "loadmap"
        b.unit(124, mnem, f"{mnem} {i:#x}, " + ", ".join(f"W:{8 + j}" for j in range(len(planes))), vs)
        uses = [(t_load + s - 1, "RES-001")] if i == 2 and s >= 2 else []
        uses.append((t_load + s, code))
        for t_use, err in uses:
            vx = b.read_for(t_use, "permute", 124, [x_slice], [SRC], "W", 0, addr_text=["x"])
            text = f"permute W:0, {i:#x}, E:4"
            if err:
                cycle, _ = b.unit(124, "permute", text, [vx])
                k.diags.append((cycle + p.timing("permute")[1], 124, f"OTS-E-{err}"))
                continue
            _, v = b.unit(124, "permute", text, [vx], dst=("E", 4, 1))
            k.last = max(k.last, b.write(v, [44 + out], [DST]))
            k.expect(44 + out, DST, bytes(np.where(m < n, x[np.minimum(m, n - 1)], 0).astype(np.int8).view(np.uint8)))
            out += 1
        t_load += s + 4
    return k.done("permute_maps")


CASES = [
    ("sxm/shift_select", shift_select, PROFILES, {}),
    ("sxm/permute_maps", permute_maps, PROFILES + ["test/mid_min"], {}),
    ("sxm/transpose16", transpose16, PROFILES, {}),
    ("sxm/rotate", rotate, PROFILES, {}),
]
