"""residual: a + b in f32 on one VXM ALU."""


def residual(f, a, b, alu=0, to="E"):
    """add.f32 of two operands (Vecs or Values) on VXM[alu]; the result goes onto row `to`."""
    return f.vxm(alu, "add.f32", [a, b], to=to)
