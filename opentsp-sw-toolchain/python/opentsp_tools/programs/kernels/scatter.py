"""scatter: dynamic writes of the new token's K and V rows into the KV pool by Scatter (the pipeline's S-6 choice: the address comes from a map
vector the host sends each step, not from the program text). A Scatter moves one byte per tile, so one 320-byte plane of a row costs 16
instructions (k = 0..15) on the target slice; the source plane and the map are each read once from MEM and repeated 16 times (`repeat`) so
that they pass every target slice on 16 consecutive cycles and meet there (the streams move east at one gap per cycle, so a stream pair
that meets at one slice meets at all of them)."""

REF = 44         # the slice whose timeline the repeated reads are scheduled for
PLANE_BYTES = 16


def repeated_read(b, t_ref, slice_, addr, stream, text=None):
    """One Read of word `addr` of `slice_` onto row E stream `stream`, repeated 16 times on consecutive cycles, such that iteration i meets a
    consumer on slice REF dispatched at t_ref + i. Returns the 16 Values."""
    vs = [b.read_for(t_ref + i, "scatter", REF, [slice_], [addr], "E", stream, addr_text=[text] if text else None, emit=(i == 0))
          for i in range(PLANE_BYTES)]
    b.raw(vs[0].cycle - b.p.timing("read")[0] + 1, slice_, f"repeat {PLANE_BYTES - 1:#x}, 0x0", slot=None)
    return vs


def scatter_plane(b, targets, src, mp):
    """Scatter k = 0..15 of the repeated source plane `src` by the repeated map `mp` on every slice of `targets`: the byte 16 g + k of the
    word at the map's tile-g address, on each target, takes lane 16 g + k of the source. Returns the last dispatch cycle."""
    t_last = 0
    for T in targets:
        for k in range(PLANE_BYTES):
            t = b.when(src[k], "scatter", T)
            assert b.when(mp[k], "scatter", T) == t, (T, k)
            b._read(src[k], T, "scatter", t)
            b._read(mp[k], T, "scatter", t)
            b.raw(t, T, f"scatter E:{src[k].stream}, E:{mp[k].stream}, {k:#x}, 0x0", slot="W")
            t_last = max(t_last, t)
    return t_last


def scatter_row(b, t_ref, planes, groups, spare=2):
    """One f16 row tile written to several places: `planes` = [(slice, addr) of the low byte plane, (slice, addr) of the high plane] in MEM
    (west quads), `groups` = [(map slice, map addr, map text, (low target slice, high target slice))...]: for each group the two planes are
    scattered by the group's map to its two slices. Streams: 0 and 1 carry the planes, 2 + i the group maps (`spare` is the first stream of
    the maps). Returns the last Scatter's dispatch cycle."""
    src = [repeated_read(b, t_ref, s, a, pl) for pl, (s, a) in enumerate(planes)]
    t_last = 0
    for i, (ms, ma, mtext, targets) in enumerate(groups):
        mp = repeated_read(b, t_ref, ms, ma, spare + i, mtext)
        for pl, T in enumerate(targets):
            t_last = max(t_last, scatter_plane(b, [T], src[pl], mp))
    return t_last
