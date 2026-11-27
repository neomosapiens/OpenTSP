"""head_mask: the f16 query kept in the lanes of one head, +0 elsewhere ."""


def head_mask(f, q16, mask, alu, at=0, to="E"):
    """mask.f16 on VXM[alu] of the f16 Vec q16 with the f16 Vec `mask` (non-zero in the head's lanes); the result
    (a stream pair, row `to`) is the ABC operand of the head."""
    return f.vxm(alu, "mask.f16", [q16, mask], to=to, width=2, at=at)
