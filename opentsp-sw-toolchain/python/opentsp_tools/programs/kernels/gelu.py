"""gelu: 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3))) as nine chained VXM operations ."""
import numpy as np

from opentsp_tools.programs import data
from opentsp_tools.programs import numerics as nx

# (operation, other operand, the accumulator is the left operand)
GELU_STEPS = [("mul", "x", True), ("mul", "x", True), ("mul", 0.044715, True), ("add", "x", False),
              ("mul", 0.7978845608028654, True), ("tanh", None, True), ("add", 1.0, True), ("mul", 0.5, True),
              ("mul", "x", True)]


def gelu_chain(k, t, t0, x, slot, quad, addr):
    """GELU of x (type t, west quad `quad`, words from `addr + 40 * slot`) on ALUs 0..8 (chained: each takes the
    previous result from the station in the cycle after its commit), the first multiplying x by itself, starting at
    dispatch cycle t0. Returns (the result Value, the expected vector)."""
    w, n = data.width(t), k.p.vlanes
    base = addr + 40 * slot
    x_at = k.place(f"x{slot}", x, quad, base)
    prev = acc = None
    for i, (op, other, acc_first) in enumerate(GELU_STEPS):
        cycle = t0 if prev is None else prev.cycle + 1 - k.p.timing(f"{op}.{t}")[1]
        stream = 16 + 4 * (i % 2)
        if other is None:
            srcs, arrs = [prev], [acc]
        else:
            if other == "x":
                arr, v = x, k.feed(x_at, base, "E", stream, cycle, f"{op}.{t}")
            else:
                arr = np.full(n, other, dtype=data.DTYPES[t])
                v = k.feed(k.place(f"c{slot}{i}", arr, quad, base + 1 + i), base + 1 + i, "E", stream, cycle, f"{op}.{t}")
            if prev is None:
                srcs, arrs = [v, v], [arr, arr]
            else:
                srcs, arrs = ([prev, v], [acc, arr]) if acc_first else ([v, prev], [arr, acc])
        chain = tuple(i for i, s in enumerate(srcs) if s is prev and prev is not None)
        prev = k.b.vxm(i, f"{op}.{t}", "E", 8 + 4 * (i % 2), srcs, chain=chain, cycle=cycle, width=w)
        acc = nx.transcendental(op, t, arrs[0]) if op == "tanh" else nx.binary(op, t, *arrs)
    return prev, acc


def gelu_vec(f, x, c0, c1, one, half, to="E"):
    """GELU of the f32 Vec x as nine chained VXM operations on ALUs 0..8 (the same operation order as GELU_STEPS); the
    Vecs c0, c1, one and half are the constants 0.044715, sqrt(2/pi), 1 and 0.5 in every lane. The result is on row
    `to`."""
    a = f.vxm(0, "mul.f32", [x, x])
    a = f.vxm(1, "mul.f32", [a, x], chain=(0,))
    a = f.vxm(2, "mul.f32", [a, c0], chain=(0,))
    a = f.vxm(3, "add.f32", [x, a], chain=(1,))
    a = f.vxm(4, "mul.f32", [a, c1], chain=(0,))
    a = f.vxm(5, "tanh.f32", [a], chain=(0,))
    a = f.vxm(6, "add.f32", [a, one], chain=(0,))
    a = f.vxm(7, "mul.f32", [a, half], chain=(0,))
    return f.vxm(8, "mul.f32", [a, x], chain=(0,), to=to)
