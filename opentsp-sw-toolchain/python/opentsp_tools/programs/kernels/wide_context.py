"""wide_context: the PV product of one 256-lane tile on the VXM (`context.context` for f16 V rows): per slot the f16 V row (two byte planes)
is widened by `conv_f16_f32`, masked to zero when P_s is zero, multiplied by P_s and added in two interleaved chains (slots s mod 2): a chained
add takes four cycles, so a chain takes a slot every four cycles, and a slot takes two cycles (widening, masking, multiply and add take an ALU each)."""

SPACING = 2
CHAINS = 2
CONV, MASK, MUL, ADD = 0, 4, 8, 12  # four ALUs per stage (slot mod 4)


def context(f, ps, vs, zero, t0, quads, to="E"):
    """`ps`: the f32 Vecs P_s; `vs`: the f16 Vecs of the V rows (two slices), one per slot. Slot s widens at cycle t0 + 2 s; `zero` is the f32
    zero Vec the first additions take; the two chain totals are kept on the east quads `quads`. Returns (the f32 sum, its f16 narrowing)."""
    previous_avoid, f.avoid = f.avoid, True
    acc = []
    for s, (pv, vv) in enumerate(zip(ps, vs)):
        w = f.vxm(CONV + s % 4, "conv_f16_f32.f16", [vv], to="E", width=4, at=t0 + SPACING * s)
        # Zero probability excludes the value, including nonfinite stale data.
        # The probability is read twice, four cycles apart. Callers distribute
        # consecutive slots across three SRAM quads to avoid read collisions.
        safe = f.vxm(MASK + s % 4, "mask.f32", [w, pv], chain=(0,))
        prod = f.vxm(MUL + s % 4, "mul.f32", [pv, safe], chain=(1,))
        if s < CHAINS:
            acc.append(f.vxm(ADD + s % 4, "add.f32", [prod, zero], chain=(0,)))
        else:
            acc[s % CHAINS] = f.vxm(ADD + s % 4, "add.f32", [prod, acc[s % CHAINS]], chain=(0, 1))
    kept = [f.store(a, q) for a, q in zip(acc, quads)]
    both = max(f.bound(v, "add.f32", 88) for v in kept)
    z = f.vxm(0, "add.f32", [kept[0], kept[1]], at=both, to=to)
    narrow = f.vxm(1, "conv_f32_f16.f32", [z], chain=(0,), to=to, width=2)
    f.avoid = previous_avoid
    return z, narrow
