"""Differential test of the Sail model's emulator against the
independent Python oracle (test/oracle/), on generated programs.

usage: diff_oracle.py <tsp_sim> <config.json> <random-programs> <seed>
                      [--jobs N] [--only KIND] [--keep DIR] [--impl CMD]

--impl CMD replaces the Python oracle by another implementation (for example a second ISS)
that speaks the tsp_sim line protocol (porting-guide.md §2.3): CMD is split on spaces, run as
`CMD <config.json> --trace` with the program on stdin, and its output is compared with tsp_sim's
after canonical(): the `T srf` lines of one cycle are compared as a set, sorted by (row,
stream, gap); every other line in order (microarchitecture.md §12).

Each program is a set of core instructions (directed templates first, then random mixes of
every slice: CU (SetVL, Config liquid threads), MEM with AGT loops (StreamStep) and Read +
Write pairs, VXM, MXM (staged state, RES-003), SXM, C2C over looped-back links, barriers,
Repeat and malformed words). The program is a cycle-view list of (cycle, queue, word) that the
normative lowering of tools/tsp_as.py turns into the text images of the Q lines. The oracle is first run without effects to learn which elements execute; every
stream operand of every element is then fed by a MEM Read timed to arrive at the reader's
gap in its read cycle, and every result plane is drained into SRAM by a MEM Write (SXM
distributor results through a Shift). The program runs on tsp_sim --trace and on the oracle;
the run passes when every trace event (stream-register commit, SRAM write, diagnostic: cycle,
place and code, in order) and every dumped SRAM word agree. Exit status 1 on any mismatch,
with a reproducer (the tsp_sim input) written under --keep (default: the system temp dir).
"""
import argparse
import json
import multiprocessing
import os
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tools'))
from oracle import machine as mc            # noqa: E402
import tsp_as                               # noqa: E402
from oracle import numerics as nx           # noqa: E402


def enc_rec(isa, name, **f):
    """isa.encode with the VXM chain bits and group bit
    defaulting to 0 (unchained, not grouped)."""
    for (n, hi, lo, kind, ex) in isa.by_name[name].vars:
        if kind in ('chain', 'group4'):
            f.setdefault(n, 0)
    return isa.encode(name, **f)

VXM_Q = list(range(88, 104))
# The 14-queue SXM blocks (topology.md 5): DST 4, TSP 2, SHF 2, PRM, SEL, C2C TX 2, RX 2.
SXM_BASE = {'W': 116, 'E': 130}
SXM_KIND_Q = {(h, k): [SXM_BASE[h] + o for o in offs] for h in 'WE'
              for k, offs in (('DST', range(0, 4)), ('TSP', (4, 5)), ('SHF', (6, 7)), ('PRM', (8,)), ('SEL', (9,)))}
SXM_Q = sorted(q for qs in SXM_KIND_Q.values() for q in qs)
C2C_TX_Q = {h: [SXM_BASE[h] + 10, SXM_BASE[h] + 11] for h in 'WE'}
C2C_RX_Q = {h: [SXM_BASE[h] + 12, SXM_BASE[h] + 13] for h in 'WE'}
C2C_Q = sorted(C2C_TX_Q['W'] + C2C_TX_Q['E'] + C2C_RX_Q['W'] + C2C_RX_Q['E'])
MXM_Q = {p: [104 + 3 * p + k for k in range(3)] for p in range(4)}
VXM_NAMES = (mc.VXM_BIN + list(mc.VXM_UN) + ['Mask', 'Select', 'Cmp', 'Reduce', 'Convert']
             + list(mc.VXM_CONV))


def sref(row, i):
    return (32 if row == 'W' else 0) | (i & 31)


def splus(s, b):
    return (s & 32) | ((s & 31) + b)


# ------------------------------------------------------------------ element values

def f16(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def f32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def fp8(t):
    """The encoder of an FP8 format: the nearest even value (an overflow saturates in e4m3, is infinite in e5m2)."""
    f = nx.fmt(t)
    return lambda x: nx.fround(f, x < 0, nx.Fraction(abs(x)))


def encoder(t):
    return f16 if t == 'f16' else f32 if t == 'f32' else fp8(t)


SPECIAL_F = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, -2.5, 3.5, 0.25, 2.0, 10.0, -7.0, 100.0, 1e-3, 127.5,
             -128.5, 255.5, 300.0, -300.0, 64.0, -64.0, 63.9, 512.0, -512.0, 88.0, 0.1, 3.0]


def rand_elem(rnd, t):
    w = nx.WIDTH[t]
    n = 8 * w
    r = rnd.random()
    if nx.is_bool(t):
        return rnd.choice([0, (1 << n) - 1, rnd.getrandbits(n), 1])
    if nx.is_float(t):
        enc = encoder(t)
        f = nx.fmt(t)
        if r < 0.35:
            return enc(rnd.choice(SPECIAL_F))
        if r < 0.5:
            return rnd.choice([f.inf(False), f.inf(True), f.cnan, f.cnan | 1, f.cnan | f.sign, 1, f.sign | 1,
                               f.expmask - 1, (1 << (f.width - 1)) - (1 << f.mbits) - 1, 1 << f.mbits])
        if r < 0.8:
            return enc(rnd.choice([1, -1]) * rnd.uniform(0, 8) * 2 ** rnd.randint(-6, 6)) if t in ('f32', 'e4m3', 'e5m2') else \
                enc(round(rnd.uniform(-60, 60) * 4) / 4)
        return rnd.getrandbits(n)
    lo, hi = nx.irange(t)
    if r < 0.3:
        return nx.raw(t, rnd.choice([0, 1, -1, lo, hi, lo + 1, hi - 1, 2, -2, 127, -128, 128, 255]))
    if r < 0.6:
        return nx.raw(t, rnd.randint(-20, 20))
    return rnd.getrandbits(n)


def to_planes(vals, w):
    return [bytes((v >> (8 * b)) & 0xff for v in vals) for b in range(w)]


def rand_vec(rnd, t, vl=320):
    w = nx.WIDTH[t]
    return to_planes([rand_elem(rnd, t) for _ in range(vl)], w)


def rand_bytes(rnd, vl=320):
    mode = rnd.random()
    if mode < 0.15:
        return bytes(vl)
    if mode < 0.3:
        return bytes(rnd.choice([0, 0, 1, 255, rnd.getrandbits(8)]) for _ in range(vl))
    return bytes(rnd.getrandbits(8) for _ in range(vl))


# ------------------------------------------------------------------ operands of one element

def operands(m, q, name, f, c0, k, soff=0):
    """(reads, writes) of one element: reads [(stream0, width, cycle, gap, type)],
    writes [(stream0, width, commit cycle, gap)], only for operands that have a gap and
    are aligned (the others are diagnostics without a value)."""
    unit, sl, plane, sx = mc.queue_unit(q)
    lay = m.lay
    R, W = [], []

    def rd(s, w, t, typ):
        g = lay.read_gap(unit, sl, mc.srow(s))
        if g is not None and (m.aligned(s, w) if w in (1, 2, 4) else (s & 15) == 0):
            R.append((s, w, t, g, typ))

    def wr(s, w, cc):
        g = lay.write_gap(unit, sl, mc.srow(s))
        if g is not None and (m.aligned(s, w) if w in (1, 2, 4) else True):
            W.append((s, w, cc, g))
    e = m.isa.by_name[name]
    # The element reads at c0 + k + d_skew and commits at
    # c0 + k + d_func, with its entry's latency
    df, ds = m.lat(name, f.get('type'))
    rc, cc = c0 + k + ds, c0 + k + df
    if name == 'Ifetch':
        rd(f['stream'], 1, rc, 'bytes')          # one stream operand: the vector at the execute cycle ...
        rd(f['stream'], 1, rc + 1, 'bytes')      # ... and the next one on the same stream
    elif e.slice == 'MEM':
        def st(s):
            e = (s & 31) + soff
            return (s & 32) | e if 0 <= e <= 31 else None
        if name == 'Read' and st(f['dst']) is not None:
            wr(st(f['dst']), 1, cc)
        elif name == 'Write' and st(f['src']) is not None:
            rd(st(f['src']), 1, rc, 'bytes')
        elif name == 'Gather':
            rd(f['map'], 1, rc, 'map')
            if st(f['dst']) is not None:
                wr(st(f['dst']), 1, cc)
        elif name == 'Scatter' and st(f['src']) is not None:
            rd(st(f['src']), 1, rc, 'bytes')
            rd(f['map'], 1, rc, 'map')
    elif e.slice == 'VXM':
        t = f['type']
        ws = wd = nx.WIDTH[t]
        st = t
        if name in mc.VXM_CONV:
            st, dt = nx.CONV[mc.VXM_CONV[name]]
            ws, wd = nx.WIDTH[st], nx.WIDTH[dt]
        elif name == 'Convert':
            wd = nx.WIDTH[nx.TYPE_NAME[f['dtype']]]
        if f.get('sg4', 0):
            # Each operand is an SG4 group of 4 / ws typed parts; a group
            # that is not SG4-aligned is OTS-E-OPR-002 and gets no operands.
            parts = range(0, 4, ws)
            for key in ('lhs', 'rhs', 'mask'):
                if key in f and (f[key] & 31) % 4 == 0:
                    for o in parts:
                        rd(splus(f[key], o), ws, rc, st)
            if (f['dst'] & 31) % 4 == 0:
                for o in parts:
                    wr(splus(f['dst'], o), wd, cc)
            return R, W
        for key in ('lhs', 'rhs', 'mask'):
            if key in f:
                rd(f[key], ws, rc, st)
        wr(f['dst'], wd, cc)
    elif e.slice == 'MXM':
        if name in ('LW', 'IWS'):
            rd(f['src'] << 4, 16, rc, 'w')
        elif name == 'ABC8':
            rd(f['src'], 1, rc, 'bytes')
        elif name == 'ABC':
            w = 2 if mc.MXM_WT[f['type']] in mc.PAIR else 1     # f16, bf16, bfp8: an aligned pair of streams
            rd(f['src'], w, rc, 'f16' if w == 2 else 'bytes')
        elif name == 'ACC':
            wr(f['dst'], 4, cc)
    elif e.slice == 'SXM':
        if name == 'Shift':
            rd(f['src'], 1, rc, 'bytes')
            wr(f['dst'], 1, cc)
        elif name in ('Permute', 'Distribute'):
            rd(f['src'], 1, rc, 'bytes')
            wr(f['dst'], 1, cc)
        elif name == 'Rotate':
            rd(f['src'], 1, rc, 'bytes')
            d = f['dst']
            if (d & 31) + f['n'] ** 2 <= 32:
                for t in range(f['n'] ** 2):
                    wr((d & 32) | (((d & 31) + t) & 31), 1, cc)
        elif name == 'Transpose':
            rd(f['src'] << 4, 16, rc, 'bytes')
            d = f['dst'] << 4
            for c in range(16):
                wr(splus(d, c), 1, cc)
        elif name == 'LoadMap':
            rd(f['src'], 1, rc, 'maplo')
        elif name == 'LoadMapWide':
            rd(f['src'], 1, rc, 'maplo')
            rd(f['src2'], 1, rc, 'maphi')
        elif name == 'SelectS':
            for key in ('a', 'b', 'c'):
                rd(f[key], 1, rc, 'bytes')
            rd(f['sel'], 1, rc, 'sel3')
            wr(f['dst'], 1, cc)
    elif e.slice == 'C2C':
        if name == 'Send':
            rd(f['stream'], 1, rc, 'bytes')
        elif name in ('Receive', 'DeskewRecv'):
            wr(f['stream'], 1, cc)
        elif name == 'PciSend':
            rd(f['stream'], 1, rc, 'bytes')
        elif name == 'PciReceive':
            wr(f['stream'], 1, cc)
    return R, W


# ------------------------------------------------------------------ program builder

class Prog:
    def __init__(self, rnd, m):
        self.rnd, self.m, self.isa = rnd, m, m.isa
        self.recs = []
        self.units = {}           # queue -> text units appended after the lowered records (front-end templates)
        self.pre = {}
        self.dump = set()
        self.used = set()
        self.core_q = set()
        self.addr_used = {}
        self.data = {}            # (stream, read cycle) -> planes, preset by templates
        self.maps = {}            # (side, map id) -> map entries (for map-shaped feeds)
        self.nodrop = False       # set by templates whose effect needs every operand present
        self.nofeed = set()       # (stream, read cycle) operands left unfed on purpose
        self.faults = []          # ECC fault injection (--ecc): ('sram', cycle, slice, addr, superlane, bit) / ('srf', cycle, row, stream, gap, superlane, bit)

    # records
    def rec(self, cyc, q, name, **f):
        w = enc_rec(self.isa, name, **f)
        self.raw(cyc, q, 1, w, 0)

    def raw(self, c, q, wc, w0, w1=0):
        if c < 0:
            return
        self.recs.append((c, q, wc, w0, w1))
        self.used.add((q, c))

    def core(self, q):
        self.core_q.add(q)
        return q

    def fresh_addr(self, s, lo=0, hi=None):
        if hi is None:
            hi = WORDS - 1
        u = self.addr_used.setdefault(s, set())
        while True:
            a = self.rnd.randint(lo, hi)
            if a not in u:
                u.add(a)
                return a

    # feeds and drains through MEM slices outside the program's core queues
    def mem_slices(self, row, ok):
        out = []
        for s in range(88):
            if s in self.core_q:
                continue
            g = self.m.lay.mem_gap(s, row)
            if g is not None and ok(g):
                out.append((s, g))
        self.rnd.shuffle(out)
        return out

    def feed(self, s6, t, R, data):
        row = mc.srow(s6)
        for s, g in self.mem_slices(row, lambda g: g < R):
            c = t - self.m.lat('Read')[0] - (R - g)
            if c >= 0 and (s, c) not in self.used:
                a = self.fresh_addr(s)
                self.pre[(s, a)] = data
                self.rec(c, s, 'Read', addr=a, dst=s6, thread=0)
                return True
        return False

    def drain(self, s6, cc, W):
        row = mc.srow(s6)
        for s, g in self.mem_slices(row, lambda g: g > W):
            c = cc + (g - W) - self.m.lat('Write')[1]
            if (s, c) not in self.used:
                a = self.fresh_addr(s)
                self.rec(c, s, 'Write', addr=a, src=s6, thread=0)
                self.dump.add((s, a))
                return True
        return False

    def drain_via_shift(self, s6, cc, W):
        """A value at gap W >= every MEM gap (SXM distributor output): a Shift by 0 on the
        same side's shifter reads it at the turn gap and re-emits it on the other row."""
        side = mc.srow(s6)
        if W >= self.m.lay.tin:
            return False
        for q in SXM_KIND_Q[(side, 'SHF')]:
            c = cc + (self.m.lay.tin - W) - self.m.lat('Shift')[1]
            if (q, c) not in self.used:
                d = sref('E' if side == 'W' else 'W', self.rnd.randrange(32))
                self.rec(c, q, 'Shift', dst=d, src=s6, dir=0, amount=0)
                self.drain(d, c + self.m.lat('Shift')[0], self.m.lay.tout)
                return True
        return False

    def value(self, typ, s6, b, cache):
        rnd = self.rnd
        if typ == 'bytes':
            return rand_bytes(rnd)
        if typ in ('map', 'maplo', 'maphi'):
            return rand_bytes(rnd)
        if typ == 'sel3':
            # mostly legal selectors (0, 1, 2); sometimes one lane is out of range (OPR-001)
            v = bytearray(rnd.choice([0, 1, 2]) for _ in range(320))
            if rnd.random() < 0.15:           # those lanes are zero, the others selected
                for _ in range(rnd.choice([1, 1, 5, 60])):
                    v[rnd.randrange(320)] = rnd.choice([3, 255])
            return bytes(v)
        if typ == 'w':
            return rand_bytes(rnd)
        return rand_vec(rnd, typ)[b] if nx.WIDTH[typ] == 1 else cache(typ)[b]

    def attach(self, elems):
        rnd = self.rnd
        for (q, name, f, c0, k, off, soff) in elems:
            R, W = operands(self.m, q, name, f, c0, k, soff)
            for (s0, w, t, g, typ) in R:
                if (s0, t) in self.nofeed:
                    continue
                n = 16 if w == 16 else w
                preset = self.data.get((s0, t))
                vec = None
                if preset is None and typ not in ('bytes', 'map', 'maplo', 'maphi', 'w', 'sel3'):
                    vec = rand_vec(rnd, typ)
                for b in range(n):
                    if rnd.random() < 0.015 and not self.nodrop:
                        continue                       # a missing operand (timing diagnostic)
                    if preset is not None and b < len(preset):
                        v = preset[b]
                    elif preset is not None:
                        v = rand_bytes(rnd)          # another operand shares the stream and cycle
                    elif vec is not None:
                        v = vec[b]
                    else:
                        v = self.value(typ, s0, b, None)
                    self.feed(splus(s0, b), t, g, v)
            for (s0, w, cc, g) in W:
                for b in range(w):
                    s6 = splus(s0, b)
                    if not self.drain(s6, cc, g):
                        self.drain_via_shift(s6, cc, g)


# ------------------------------------------------------------------ templates

def ext_codes(isa, ex):
    """The codes of an eltype_code or weight-type variable that the profile declares: the base codes (below 16 for an element type) and
    the codes of every extension the profile lists (`extension_codes` of the catalogue)."""
    return [c for c in ex['codes']
            if all(c not in codes or e in isa.declared for e, codes in ex.get('extension_codes', {}).items())]


def extension_opcodes_in(isa, recs):
    """The extension instruction tags present in a program's records: `bf16` (VXM type / Convert dtype 16 or MXM
    weight type 4), `bfp8` (MXM weight type 5), `x128` (ACC type 0). Empty when the program uses only the base type set."""
    found = set()
    for (_c, _q, _wc, w0, _w1) in recs:
        for e in isa.w32 + isa.w64:
            if (w0 & e.mask) != e.value:
                continue
            f = e.fields_of(w0, isa.vlanes, isa.sram_words, isa.acc_depth, isa.declared)
            if f is None:
                break
            t, dt = f.get('type'), f.get('dtype')
            if t == 'bf16' or t == 16 or dt == 16:
                found.add('bf16')
            if e.name in ('LW', 'IW', 'IWS', 'ABC') and t == 4:
                found.add('bf16')
            if e.name in ('LW', 'IW', 'IWS', 'ABC') and t == 5:
                found.add('bfp8')
            if e.name == 'ACC' and t == 0:
                found.add('x128')
            break
    return found


def pick_type(rnd, isa, name):
    e = isa.by_name[name]
    for (n, hi, lo, kind, ex) in e.vars:
        if kind == 'eltype_code':
            return nx.TYPE_NAME[rnd.choice(ext_codes(isa, ex))]     # the base set, and an extension type when the profile declares it
    return None


def stream_for(rnd, w, row=None):
    row = row or rnd.choice('EW')
    if rnd.random() < 0.03:
        return sref(row, rnd.randrange(32))            # possibly misaligned
    return sref(row, rnd.randrange(32 // w) * w)


def vxm_fields(rnd, isa, name, sg4=None):
    t = pick_type(rnd, isa, name)
    f = {'type': t}
    ws = wd = nx.WIDTH[t]
    if name in mc.VXM_CONV:
        st, dt = nx.CONV[mc.VXM_CONV[name]]
        ws, wd = nx.WIDTH[st], nx.WIDTH[dt]
    if name == 'Convert':
        f['dtype'] = rnd.randrange(13) if 'bf16' not in isa.declared else rnd.choice(list(range(13)) + [16])
        wd = nx.WIDTH[nx.TYPE_NAME[f['dtype']]]
    e = isa.by_name[name]
    for (n, hi, lo, kind, ex) in e.vars:
        if kind == 'stream':
            f[n] = stream_for(rnd, wd if n == 'dst' else ws)
        elif kind == 'cmp_mode':
            f[n] = rnd.choice([0, 1] if nx.is_bool(t) else ex['values'])
        elif kind == 'reduce_op':
            f[n] = rnd.choice(ex['values'])
        elif kind == 'chain':
            f[n] = 0            # chained reads: t_vxm_chain
        elif kind == 'group4':
            f[n] = (1 if rnd.random() < 0.3 else 0) if sg4 is None else sg4
    if f.get('sg4'):
        for (n, hi, lo, kind, ex) in e.vars:
            if kind == 'stream' and rnd.random() < 0.95:
                f[n] = stream_for(rnd, 4)              # SG4-aligned; otherwise OTS-E-OPR-002
    return f, ws, wd


def t_vxm(p, c, name=None, q=None, pair=None, sg4=None):
    rnd, isa = p.rnd, p.isa
    name = name or rnd.choice(VXM_NAMES)
    f, ws, wd = vxm_fields(rnd, isa, name, sg4)
    if isa.decode64(enc_rec(isa, name, **f)) is None:
        return
    q = q if q is not None else rnd.choice(VXM_Q)
    p.rec(c, q, name, **f)
    # paired operands: equal lanes, signed zeros, the same value
    t = f['type']
    st = nx.CONV[mc.VXM_CONV[name]][0] if name in mc.VXM_CONV else t
    rc = c + p.m.lat(name, t)[1]
    if 'rhs' in f or 'mask' in f:
        a = [rand_elem(rnd, st) for _ in range(320)]
        b = [rand_elem(rnd, st) for _ in range(320)]
        for l in range(320):
            r = rnd.random()
            if r < 0.25:
                b[l] = a[l]
            elif r < 0.35 and nx.is_float(st):
                fm = nx.fmt(st)
                a[l], b[l] = rnd.choice([(0, fm.sign), (fm.sign, 0)])
        p.data[(f['lhs'], rc)] = to_planes(a, ws)
        other = f.get('rhs', f.get('mask'))
        p.data[(other, rc)] = to_planes(b, ws)
    return q, f


def t_vxm_sg4(p, c):
    """A same-width VXM operation with the group bit set on one ALU, its
    operands SG4 groups (rarely misaligned: OTS-E-OPR-002)."""
    names = [n for n in VXM_NAMES if any(v[3] == 'group4' for v in p.isa.by_name[n].vars)]
    t_vxm(p, c, name=p.rnd.choice(names), sg4=1)


def t_vxm_chain(p, c):
    """VXM ALU chaining: a consumer with the chain bit set reads the VXM result
    written in the previous cycle (consumer dispatch = producer dispatch + d_func - d_skew + 1);
    sometimes one cycle off (TIM-001)."""
    r = t_vxm(p, c, name='Add')
    if r:
        q, f = r
        d = f['dst']
        t = f['type']
        df, ds = p.m.lat('Add', t)
        off = p.rnd.choice([0, 0, 0, -1, 1, p.rnd.randint(2, 9)])
        p.rec(c + df - ds + 1 + off, p.rnd.choice(VXM_Q), 'Neg' if nx.is_float(t) else 'Not',
              type=t, dst=stream_for(p.rnd, nx.WIDTH[t]), lhs=d, lhs_chain=1)


def t_vxm_listing1(p, c):
    """An unchained add whose destination is also a source, issued every cycle (ISCA20 Listing 1):
    each read takes the upstream operand, never the result of the previous issue."""
    d = stream_for(p.rnd, 1)
    q = p.rnd.choice(VXM_Q)
    b = stream_for(p.rnd, 1)
    for k in range(p.rnd.randint(2, 6)):
        p.rec(c + k, q, 'Add', type='i8', dst=d, lhs=d, rhs=b, lhs_chain=0, rhs_chain=0)


def t_vxm_cmp(p, c):
    """Every comparison mode of one element type, several cycles apart (equal lanes included)."""
    t = p.rnd.choice(nx.TYPES)
    for mode in ([0, 1] if nx.is_bool(t) else range(6)):
        f, ws, wd = vxm_fields(p.rnd, p.isa, 'Cmp')
        f['type'], f['mode'] = t, mode
        w = nx.WIDTH[t]
        for k in ('dst', 'lhs', 'rhs'):
            f[k] = sref(p.rnd.choice('EW'), p.rnd.randrange(32 // w) * w)
        p.rec(c, p.rnd.choice(VXM_Q), 'Cmp', **f)
        rc = c + p.m.lat('Cmp', t)[1]
        a = [rand_elem(p.rnd, t) for _ in range(320)]
        b = [a[l] if l % 3 == 0 else rand_elem(p.rnd, t) for l in range(320)]
        p.data[(f['lhs'], rc)] = to_planes(a, w)
        p.data[(f['rhs'], rc)] = to_planes(b, w)
        c += p.rnd.randint(3, 9)


def rnd_int(rnd, a, b):
    return rnd.randint(a, b)


def t_vxm_collide(p, c):
    """Two VXM results to the same stream in the same cycle (SRF-002)."""
    qs = p.rnd.sample(VXM_Q, 2)
    d = stream_for(p.rnd, 1)
    for q in qs:
        p.rec(c, q, 'Add', type='i8', dst=d, lhs=stream_for(p.rnd, 1), rhs=stream_for(p.rnd, 1), lhs_chain=0, rhs_chain=0)


def mem_core_slice(p, row=None, exists=False):
    s = p.rnd.randrange(88)
    if not p.m.lay.present(s) and (exists or p.rnd.random() < 0.9):      # mostly a slice that exists
        s = (s // 44) * 44 + s % 44 % p.m.lay.mem_per_hemi
    p.core(s)
    return s


def agt_config(p, q, c):
    """Countdown/Step/Iterations records in the cycles after c; returns the next free cycle."""
    rnd = p.rnd
    c += 1
    for lv in range(rnd.choice([0, 1, 1, 2, 3, 5])):
        for fld in ('Countdown', 'Step', 'StreamStep', 'Iterations'):
            if rnd.random() < (0.3 if fld == 'StreamStep' else 0.75):
                if fld == 'StreamStep':
                    p.rec(c, q, fld, s=rnd.choice([0, 1, 1, 2, 4, 31]))
                elif fld == 'Countdown':
                    p.rec(c, q, fld, d=rnd.choice([0, 1, 1, 2, 3, 5]))
                elif fld == 'Step':
                    p.rec(c, q, fld, a=rnd.choice([1, -1, 2, -3, 16, -16, 4095, -4096, 8191, -8191,
                                                   rnd.randint(-40, 40)]))
                else:
                    p.rec(c, q, fld, n=rnd.choice([0, 1, 2, 3, 4]))
                c += 1
    return c


def t_mem(p, c):
    rnd = p.rnd
    s = mem_core_slice(p)
    row = rnd.choice('EW')
    st = sref(row, rnd.randrange(32))
    kind = rnd.choice(['Read', 'Write', 'Read', 'Write', 'Gather', 'Scatter', 'Pair', 'SetSegment', 'Gating'])
    th = rnd.choice([0, 0, 0, 1])
    if kind == 'Gating':
        # SetVL and Config (liquid threads) on the queue, then a tagged reference (MA 3, 4)
        if rnd.random() < 0.7:
            p.rec(c, s, 'SetVL', n=rnd.choice([1, 3, 10, 19, 20]))
            c += 1
        if rnd.random() < 0.7:
            p.rec(c, s, 'Config', v=rnd.choice([0, 6, 18, 19, 20, 300]))
            c += 1
        kind = rnd.choice(['Read', 'Write', 'Gather', 'Scatter'])
    segn = None
    if kind == 'SetSegment':
        n = rnd.choice([0, 1, 100, 4096, 8191, 8192])
        p.rec(c, s, 'SetSegment', n=n)
        kind = rnd.choice(['Read', 'Write', 'Gather', 'Scatter'])      # Gather and Scatter see good and bad tiles
        segn = n
        c += 1
    a = rnd.choice([0, 1, 8191, 4095, 4096, rnd.randrange(WORDS)])
    if kind in ('Read', 'Write'):
        p.rec(c, s, kind, addr=a, thread=th, **({'dst': st} if kind == 'Read' else {'src': st}))
        agt_config(p, s, c)
    elif kind in ('Gather', 'Scatter'):
        mp = sref(row, rnd.randrange(32))
        k = rnd.choice([0, 15, rnd.randrange(16)])
        if kind == 'Gather':
            p.rec(c, s, kind, dst=st, map=mp, k=k, thread=th)
        else:
            p.rec(c, s, kind, src=st, map=mp, k=k, thread=th)
        agt_config(p, s, c)
        # map data: tile addresses from a small set (high byte used, duplicates)
        addrs = [rnd.choice([rnd.randrange(WORDS), rnd.randrange(256), 0x1f00 | rnd.randrange(256)]
                            + ([max(0, segn - 1), min(WORDS - 1, segn)] * 3 if segn is not None else []))
                 for _ in range(rnd.choice([1, 2, 3, 20]))]
        p.mapaddrs = getattr(p, 'mapaddrs', {})
        p.mapaddrs[(s, mp)] = addrs
    else:
        # a Read + Write pair on one queue (MA 4): different banks dispatch, one bank is MEM-001
        bb = 1 << p.m.bank_bit
        b = rnd.choice([a ^ bb, a ^ bb, a ^ 1, a, rnd.randrange(WORDS)])
        recs = [('Read', dict(addr=a, dst=st, thread=0)),
                ('Write', dict(addr=b, src=sref(row, rnd.randrange(32)), thread=0))]
        if rnd.random() < 0.1:
            recs[1] = ('Read', dict(addr=b, dst=sref(row, rnd.randrange(32)), thread=0))
        rnd.shuffle(recs)
        for name, f in recs:
            p.rec(c, s, name, **f)
        if rnd.random() < 0.3:
            p.rec(c + 1, s, 'Countdown', d=2)            # a pair is not loopable: stray


def t_mxm(p, c):
    rnd = p.rnd
    plane = rnd.randrange(4)
    qs = MXM_Q[plane]
    row = 'W' if plane < 2 else 'E'
    t = rnd.choice([0, 1, 2, 1])
    q = rnd.choice(qs)

    def grp():
        return (2 if row == 'W' else 0) + rnd.randrange(2) if rnd.random() < 0.95 else rnd.randrange(4)
    K = p.m.rows // 16                    # LWs that fit (20 on ref; the one after is RES-002)
    nlw = rnd.choice([1, 2, 3, 8, max(1, K - 1), K, K + 1])
    if rnd.random() < 0.7:
        for i in range(nlw):
            g = grp()
            p.rec(c, q, 'LW', type=t if rnd.random() < 0.97 else rnd.randrange(3), src=g)
            if t == 1:
                vals = [[f16(rnd.uniform(-2, 2)) if rnd.random() < 0.9 else rand_elem(rnd, 'f16')
                         for _ in range(320)] for _ in range(8)]
                pl = []
                for r in vals:
                    lo, hi = to_planes(r, 2)
                    pl += [lo, hi]
                p.data[(g << 4, c)] = pl
            c += 1
        c += rnd.choice([19, 20, 20, 21])
        p.rec(c, rnd.choice(qs), 'IW', type=t)
    else:
        for i in range(rnd.choice([1, 2, 3, 10])):
            p.rec(c, rnd.choice(qs), 'IWS', type=t, src=grp())
            c += 1
    c += rnd.choice([1, 1, 2, 0])
    for j in range(rnd.choice([1, 2, 2, 3])):
        s = sref(row, rnd.randrange(16) * 2 if t == 1 else rnd.randrange(32))
        p.rec(c, rnd.choice(qs), 'ABC', type=(t if t == 1 or rnd.random() < 0.6 else rnd.choice([0, 2])), src=s)   # mixed sign
        c += rnd.choice([20, 20, 21, 19])
        acc = rnd.choice([2, 3]) if rnd.random() < 0.1 else (3 if t == 1 else 2)
        p.rec(c, rnd.choice(qs), 'ACC', type=acc, dst=sref('E' if row == 'W' else 'W', rnd.randrange(8) * 4),
              mode=rnd.choice([0, 1, 1]), acc=rnd.randrange(p.m.acc_depth))
        c += rnd.choice([1, 2])


def t_mxm_f16(p, c):
    """f16 LW/IW/ABC/ACC with finite weights and operands, so every product reaches the result
    (a NaN or infinity anywhere in the 320-lane operand hides the weight bytes)."""
    rnd = p.rnd
    plane = rnd.randrange(4)
    qs = MXM_Q[plane]
    row = 'W' if plane < 2 else 'E'
    q = rnd.choice(qs)
    g0 = 2 if row == 'W' else 0
    zero_case = rnd.random() < 0.25          # signed-zero sums (MA 6: -0 only if every product is -0)
    wlo = 0 if zero_case else -4
    if plane in (0, 2) and rnd.random() < 0.4:
        # the odd partner plane is busy while the even plane holds f16 rows (MA 6, RES-001)
        odd = MXM_Q[plane + 1]
        p.rec(c + rnd.randint(1, 30), rnd.choice(odd), 'LW', type=0, src=g0 + rnd.randrange(2))
    for i in range(min(rnd.choice([1, 2, 3]), p.m.rows // 16)):
        g = g0 + rnd.randrange(2)
        p.rec(c, q, 'LW', type=1, src=g)
        pl = []
        for _ in range(8):
            lo, hi = to_planes([f16(round(rnd.uniform(wlo, 4) * 64) / 64) for _ in range(320)], 2)
            pl += [lo, hi]
        p.data[(g << 4, c)] = pl
        c += 1
    c += 20
    p.rec(c, q, 'IW', type=1)
    c += 1
    s = sref(row, rnd.randrange(16) * 2)
    p.rec(c, q, 'ABC', type=1, src=s)
    if rnd.random() < 0.3:
        p.rec(c + 1, rnd.choice(qs), 'ABC', type=1, src=s)   # MACC busy: RES-001 (MA 6)
    if zero_case:
        # zero activations: every product is a signed zero; all -0 gives -0 (MA 6)
        sign = rnd.choice([0x8000, 0x8000, 0])
        xs = [sign] * 320
        if rnd.random() < 0.3:
            xs[rnd.randrange(320)] ^= 0x8000
    else:
        xs = [f16(round(rnd.uniform(-4, 4) * 64) / 64) for _ in range(320)]
    p.data[(s, c + p.m.lat('ABC')[1])] = to_planes(xs, 2)
    c += 20
    p.rec(c, q, 'ACC', type=3, dst=sref('E' if row == 'W' else 'W', rnd.randrange(8) * 4), mode=0,
          acc=rnd.randrange(p.m.acc_depth))


def t_mxm_fp8(p, c):
    """fp8 LW/IW/ABC/ACC in both formats: weights and activations are mostly finite values of the format
    (so that the products reach the result), now and then a NaN, an e5m2 infinity or a zero; an ABC of another type
    on fp8 weights, or an fp8 ABC on other weights, is OTS-E-RES-001."""
    rnd = p.rnd
    plane = rnd.randrange(4)
    qs = MXM_Q[plane]
    row = 'W' if plane < 2 else 'E'
    q = rnd.choice(qs)
    g0 = 2 if row == 'W' else 0
    fmt = rnd.randrange(2)
    ff = nx.E4M3 if fmt == 0 else nx.E5M2
    big = 0x7E if fmt == 0 else 0x7B

    def rv():
        r = rnd.random()
        if r < 0.7:
            return ff.zero(rnd.random() < 0.5) | rnd.randrange(0x60 if fmt == 0 else 0x50)    # small magnitudes: exact sums matter
        if r < 0.85:
            return (rnd.choice([0, 0x80])) | rnd.choice([big, big - 1, 1, 2, 3, 0x30, 0x38])
        if r < 0.9:
            return rnd.choice([0x7F, 0xFF]) if fmt == 0 else rnd.choice([0x7C, 0xFC, 0x7D, 0xFE])
        return rnd.getrandbits(8)
    wtype = 3 if rnd.random() < 0.9 else rnd.choice([0, 1, 2])
    for i in range(min(rnd.choice([1, 2]), p.m.rows // 16)):
        g = g0 + rnd.randrange(2)
        p.rec(c, q, 'LW', type=wtype, src=g)
        p.data[(g << 4, c + p.m.lat('LW')[1])] = [bytes(rv() for _ in range(320)) for _ in range(16)]
        c += 1
    c += rnd.choice([19, 20, 21])
    p.rec(c, q, 'IW', type=wtype)
    c += rnd.choice([1, 1, 2])
    for j in range(rnd.choice([1, 2])):
        s = sref(row, rnd.randrange(32))
        if rnd.random() < 0.9:
            p.rec(c, rnd.choice(qs), 'ABC8', fmt=fmt, src=s)
        else:
            p.rec(c, rnd.choice(qs), 'ABC', type=rnd.choice([0, 1, 2]), src=s)
        p.data[(s, c + p.m.lat('ABC8')[1])] = [bytes(rv() for _ in range(320))]
        c += rnd.choice([20, 20, 21])
        p.rec(c, rnd.choice(qs), 'ACC', type=3 if rnd.random() < 0.9 else 2,
              dst=sref('E' if row == 'W' else 'W', rnd.randrange(8) * 4), mode=rnd.choice([0, 1, 1]), acc=rnd.randrange(p.m.acc_depth))
        c += rnd.choice([1, 2])


def bf16(x):
    return struct.unpack('<I', struct.pack('<f', x))[0] >> 16


def t_mxm_ext(p, c):
    """The extension MXM types on a profile that declares them: bf16 (weight type 4) and block-scaled fp8 (weight type 5, a stream pair
    of e4m3 data and block scales per row and per activation), each through LW or IWS, IW, ABC and ACC, the product read as f32 (ACC type 3) or,
    when the profile declares wide_acc and the product is not bf16, as the wide accumulator (ACC type 0, x128); now and then the pair
    that must be refused (an x128 read of a bf16 product, OTS-E-OPR-001). Values are mostly small finite numbers so that the products reach
    the result."""
    rnd = p.rnd
    exts = p.m.exts
    types = (['bf16'] if 'bf16' in exts else []) + (['bfp8'] if 'block_fp8' in exts else [])
    if not types:
        return t_mxm_f16(p, c)
    kind = rnd.choice(types)
    t = 4 if kind == 'bf16' else 5
    plane = rnd.randrange(4)
    qs = MXM_Q[plane]
    row = 'W' if plane < 2 else 'E'
    q = rnd.choice(qs)
    g0 = 2 if row == 'W' else 0
    e4m3 = fp8('e4m3')

    def row_planes():
        """One weight row or one activation: the two stream values of a pair element, (low, high) bytes of a bf16 or (data, scale) of bfp8."""
        if kind == 'bf16':
            return to_planes([bf16(rnd.uniform(-4, 4)) if rnd.random() < 0.95 else rand_elem(rnd, 'bf16') for _ in range(320)], 2)
        data = bytes(e4m3(rnd.uniform(-4, 4)) if rnd.random() < 0.95 else rnd.getrandbits(8) for _ in range(320))
        scale = bytes(rnd.randrange(16) if rnd.random() < 0.95 else rnd.getrandbits(8) for _ in range(320))
        return [data, scale]
    if plane in (0, 2) and rnd.random() < 0.25:     # the odd partner plane is busy while the even plane holds pair rows (RES-001)
        p.rec(c + rnd.randint(1, 30), rnd.choice(MXM_Q[plane + 1]), 'LW', type=0, src=g0 + rnd.randrange(2))
    if rnd.random() < 0.7:
        for i in range(min(rnd.choice([1, 2, 3]), p.m.rows // 16)):
            g = g0 + rnd.randrange(2)
            p.rec(c, q, 'LW', type=t, src=g)
            p.data[(g << 4, c)] = [pl for _ in range(8) for pl in row_planes()]
            c += 1
        c += 20
        p.rec(c, q, 'IW', type=t)
    else:
        for i in range(rnd.choice([1, 2, 3])):
            g = g0 + rnd.randrange(2)
            p.rec(c, q, 'IWS', type=t, src=g)
            p.data[(g << 4, c)] = [pl for _ in range(8) for pl in row_planes()]
            c += 1
        c += 20                                  # the rows installed from streams commit like an IW
    c += 1
    s = sref(row, rnd.randrange(16) * 2)
    p.rec(c, q, 'ABC', type=t if rnd.random() < 0.95 else rnd.choice([0, 1, 2]), src=s)
    p.data[(s, c + p.m.lat('ABC')[1])] = row_planes()
    c += 20
    wide = p.m.wide and kind == 'bfp8' and rnd.random() < 0.6
    acc = 0 if (wide or (p.m.wide and rnd.random() < 0.1)) else 3
    p.rec(c, q, 'ACC', type=acc, dst=sref('E' if row == 'W' else 'W', rnd.randrange(8) * 4), mode=rnd.choice([0, 1, 1]),
          acc=rnd.randrange(p.m.acc_depth))


def t_state_clash(p, c):
    """Two queues change one MXM plane part or one SXM map in one cycle (RES-003, MA 1)."""
    rnd = p.rnd
    if rnd.random() < 0.6:
        plane = rnd.randrange(4)
        q1, q2 = rnd.sample(MXM_Q[plane], 2)
        row = 'W' if plane < 2 else 'E'
        g = (2 if row == 'W' else 0)
        p.rec(c, q1, 'LW', type=0, src=g)
        if rnd.random() < 0.5:
            p.rec(c, q2, 'LW', type=0, src=g + 1)
        else:
            p.rec(c, q2, 'IWS', type=0, src=g + 1)           # a different part: no clash
        p.rec(c + 21, rnd.choice(MXM_Q[plane]), 'IW', type=0)
    else:
        side = rnd.choice('EW')
        qa = rnd.choice(SXM_KIND_Q[(side, 'PRM')])
        qb = rnd.choice(SXM_KIND_Q[(side, 'DST')])
        mid = rnd.randrange(16)
        p.rec(c, qa, 'LoadMap', src=sref(side, rnd.randrange(32)), map=mid)
        p.rec(c, qb, 'LoadMap', src=sref(side, rnd.randrange(32)), map=mid if rnd.random() < 0.7 else (mid + 1) % 16)


def sxm_side_q(p, side, kind):
    if p.rnd.random() < 0.04:                    # a wrong queue for the instruction
        return p.rnd.choice(SXM_Q)
    return p.rnd.choice(SXM_KIND_Q[(side, kind)])


def t_sxm(p, c):
    rnd = p.rnd
    side = rnd.choice('EW')
    other = 'E' if side == 'W' else 'W'
    own = lambda: sref(side, rnd.randrange(32))
    out = lambda: sref(other, rnd.randrange(32)) if rnd.random() < 0.95 else own()
    kind = rnd.choice(['Shift', 'Permute', 'Distribute', 'Rotate', 'Transpose', 'SelectS', 'Permute'])
    if kind == 'Shift':
        p.rec(c, sxm_side_q(p, side, 'SHF'), 'Shift', dst=out(), src=own(), dir=rnd.randrange(2),
              amount=rnd.choice([0, 1, 15, 16, 17, 100, 319, rnd.randrange(max(320, p.m.vl))]))
        if rnd.random() < 0.3:     # Shift on a permuter queue
            p.rec(c + 1, rnd.choice(SXM_KIND_Q[(side, 'PRM')]), 'Shift', dst=out(), src=own(), dir=0, amount=3)
    elif kind in ('Permute', 'Distribute'):
        mid = rnd.randrange(16)
        lq = sxm_side_q(p, side, rnd.choice(['PRM', 'DST']))
        lo_s, hi_s = own(), own()
        if kind == 'Permute':
            perm = list(range(320))
            rnd.shuffle(perm)
            if rnd.random() < 0.2:
                perm[rnd.randrange(320)] = perm[rnd.randrange(320)]
            if rnd.random() < 0.4:      # masked lanes: entries >= vlanes zero-fill
                for _ in range(rnd.randint(1, 40)):
                    perm[rnd.randrange(320)] = rnd.choice([320, 321, 511, 1023])
        else:
            perm = [rnd.choice([0, 1, 5, 14, 15, 15, 16, 17, 255, 300]) if rnd.random() < 0.5 else rnd.randrange(16)
                    for _ in range(320)]
        lo = bytes(v & 255 for v in perm)
        hi = bytes((v >> 8) | (rnd.choice([0, 4, 8, 252]) if rnd.random() < 0.3 else 0) for v in perm)
        # A Permute needs its map `superlanes` cycles after the LoadMap; a Distribute the next
        # cycle. Mostly far enough, sometimes too early (RES-001), around the boundary.
        sl = p.m.sl
        rc = c - (rnd.randint(1, 6) if kind == 'Distribute' or rnd.random() < 0.15
                  else rnd.choice([sl - 1, sl, sl, sl + 1, sl + rnd.randint(0, 8)]) or 1)
        p.data[(lo_s, rc)] = [lo]
        if any(v >> 8 for v in perm) or rnd.random() < 0.5:
            p.data[(hi_s, rc)] = [hi]
            p.rec(rc, lq, 'LoadMapWide', src=lo_s, map=mid, src2=hi_s)
        else:
            p.rec(rc, lq, 'LoadMap', src=lo_s, map=mid)
        if kind == 'Permute':
            p.rec(c, sxm_side_q(p, side, 'PRM'), 'Permute', dst=out(), src=own(), map=mid)
        else:
            p.rec(c, sxm_side_q(p, side, 'DST'), 'Distribute', dst=sref(side, rnd.randrange(32)), src=own(), map=mid)
    elif kind == 'Rotate':
        n = rnd.choice([3, 4])
        d = sref(other, rnd.randrange(32 - n * n + 1) if rnd.random() < 0.9 else rnd.randrange(32))
        p.rec(c, sxm_side_q(p, side, 'TSP'), 'Rotate', dst=d, src=own(), n=n)
    elif kind == 'Transpose':
        sg = (2 if side == 'W' else 0) + rnd.randrange(2)
        dg = (2 if other == 'W' else 0) + rnd.randrange(2)
        p.rec(c, sxm_side_q(p, side, 'TSP'), 'Transpose', dst=dg, src=sg)
    else:
        p.rec(c, sxm_side_q(p, side, 'SEL'), 'SelectS', dst=out(), a=own(), b=own(), c=own(), sel=own())


def t_sxm_wide(p, c):
    """LoadMapWide of a permutation, high parts 0..3 (bit 1 set on one lane in most programs), then a Permute."""
    rnd = p.rnd
    side = rnd.choice('EW')
    other = 'E' if side == 'W' else 'W'
    mid = rnd.randrange(16)
    perm = list(range(320))
    rnd.shuffle(perm)
    hi = [v >> 8 for v in perm]
    if rnd.random() < 0.8:
        # one lane's high part gains bit 1 (entry + 512: a masked lane; its old source lane is simply unused)
        l = rnd.randrange(320)
        hi[l] |= 2
    lo_s, hi_s = sref(side, rnd.randrange(32)), sref(side, rnd.randrange(32))
    rc = c - (p.m.sl + rnd.randint(0, 6))        # ready `superlanes` cycles after the LoadMapWide
    p.data[(lo_s, rc)] = [bytes(v & 255 for v in perm)]
    p.data[(hi_s, rc)] = [bytes(hi)]
    p.rec(rc, rnd.choice(SXM_KIND_Q[(side, 'PRM')]), 'LoadMapWide', src=lo_s, map=mid, src2=hi_s)
    p.rec(c, rnd.choice(SXM_KIND_Q[(side, 'PRM')]), 'Permute', dst=sref(other, rnd.randrange(32)),
          src=sref(side, rnd.randrange(32)), map=mid)


def t_mxm_full(p, c):
    """rows/16 - 1, rows/16 or rows/16 + 1 i8 LWs on one plane: the buffer holds mxm.rows rows (the 21st on ref is RES-002)."""
    rnd = p.rnd
    plane = rnd.randrange(4)
    q = rnd.choice(MXM_Q[plane])
    g0 = 2 if plane < 2 else 0
    p.nodrop = True
    K = p.m.rows // 16
    for i in range(rnd.choice([max(1, K - 1), K, K + 1, K + 1])):
        p.rec(c, q, 'LW', type=0, src=g0 + rnd.randrange(2))
        c += 1


def t_c2c(p, c):
    """MA 10 with one chip: every link loops back to itself; any TX queue (C2C_TX_Q) sends
    and any RX queue (C2C_RX_Q) receives on any link; the peer is chip 0; vectors arrive
    link_latency cycles after the send. A few links are out of range (>= c2c.links)."""
    rnd = p.rnd
    lat = p.m.link_lat
    chip = lambda: 0 if rnd.random() < 0.9 else rnd.randrange(1, 64)
    nl = p.m.n_links
    link = rnd.randrange(nl)
    lk = lambda: link if rnd.random() < 0.9 else rnd.randrange(16)
    k = rnd.random()
    if k < 0.6:
        side = rnd.choice('WE')
        tx = rnd.choice(C2C_TX_Q[side])
        rx_side = side if rnd.random() < 0.2 else {'W': 'E', 'E': 'W'}[side]   # any queue, any link
        rx = rnd.choice(C2C_RX_Q[rx_side])
        row_tx = side                               # a TX reads its hemisphere's outward row
        row_rx = 'W' if rx_side == 'E' else 'E'     # topology.md 4: RX writes the inward row
        n = rnd.choice([1, 1, 2, 3])
        step = rnd.choice([2, 2, p.m.cpv])               # 2 cycles apart is busy (LNK-001), cycles_per_vector apart is not
        for i in range(n):
            if rnd.random() < 0.7:
                p.rec(c + step * i, tx, 'Send', chip_id=chip(), link=lk(), stream=sref(row_tx, rnd.randrange(32)))
            else:
                p.rec(c + step * i, tx, 'Transmit', chip_id=chip(), link=lk())
        for i in range(n + rnd.choice([0, 0, 1])):
            t = c + lat + step * i + rnd.choice([0, 0, 0, 1, -1, 5])
            if (rx, t) not in p.used:
                p.rec(t, rx, 'Receive', chip_id=chip(), link=lk(), stream=sref(row_rx, rnd.randrange(32)))
    elif k < 0.7:
        # : DeskewRecv polls at the epoch boundaries for a vector that arrives before, between or after them; a probe on
        # its queue behind it shows the cycle at which the queue is released (never: the vector did not come)
        side = rnd.choice('WE')
        tx = rnd.choice(C2C_TX_Q[side])
        rx_side = rnd.choice('WE')
        rx = rnd.choice(C2C_RX_Q[rx_side])
        row_rx = 'W' if rx_side == 'E' else 'E'
        if rnd.random() < 0.85:
            if rnd.random() < 0.6:
                p.rec(c, tx, 'Transmit', chip_id=0, link=link)
            else:
                p.rec(c, tx, 'Send', chip_id=0, link=link, stream=sref(side, rnd.randrange(32)))
        t0 = c + rnd.choice([-300, -5, 0, 1, 3, 100, 251, 252, 400])
        if t0 >= 0 and (rx, t0) not in p.used:
            p.rec(t0, rx, 'DeskewRecv', chip_id=chip(), link=lk(), stream=sref(row_rx, rnd.randrange(32)))
            for u in (t0 + 1, t0 + p.m.hac, t0 + p.m.hac + lat + p.m.hac):
                if rnd.random() < 0.5 and (rx, u) not in p.used:
                    probe(p, u, rx)
    elif k < 0.8:
        q = rnd.choice(list(range(88, 104)) + C2C_Q + [rnd.randrange(88)])
        if q < 88:
            p.core(q)
        p.rec(c, q, 'Deskew')
        for t in (c + 1, c + p.m.hac - c % p.m.hac - 1, c + p.m.hac - c % p.m.hac):
            if rnd.random() < 0.5 and (q, t) not in p.used and t > c:
                probe(p, t, q) if q >= 88 else p.rec(t, q, 'NOP', n=1)
    else:
        q = rnd.choice(list(range(88, 104)) + C2C_Q)
        t = rnd.choice([0, 1, 2, 5, 40])
        p.rec(c, q, 'RuntimeDeskew', t=t)
        for u in (c + max(t, 1) - 1, c + max(t, 1)):
            if rnd.random() < 0.6 and (q, u) not in p.used and u > c:
                probe(p, u, q)


def t_host(p, c):
    """The host path of a run without a host script: PciReceive finds the ingress FIFO empty (TIM-001),
    PciSend is accepted once per host.cycles_per_vector (HIO-001 inside the window), HostWait parks its queue for
    good (the doorbell is never rung; a doorbell number out of range is OPR-001), HostNotify has no effect."""
    rnd = p.rnd
    hcpv = p.m.host_cpv
    k = rnd.random()
    if k < 0.3:
        rx = rnd.choice(C2C_RX_Q[rnd.choice('WE')])
        row_rx = 'W' if rx in C2C_RX_Q['E'] else 'E'
        for i in range(rnd.choice([1, 2])):
            if (rx, c + i) not in p.used:
                p.rec(c + i, rx, 'PciReceive', stream=sref(row_rx, 24 + rnd.randrange(8)))
    elif k < 0.6:
        side = rnd.choice('WE')
        tx = rnd.choice(C2C_TX_Q[side])
        t = c
        for i in range(rnd.choice([1, 2, 3])):
            if (tx, t) not in p.used:
                p.rec(t, tx, 'PciSend', stream=sref(side, rnd.randrange(32)))
            t += rnd.choice([1, 2, max(hcpv - 1, 1), hcpv, hcpv + 1])
    elif k < 0.85:
        q = rnd.choice(list(range(88, 104)) + C2C_Q)
        i = rnd.randrange(p.m.n_doorbells) if rnd.random() < 0.8 else p.m.n_doorbells + rnd.randrange(3)
        p.rec(c, q, 'HostWait', i=i)
        for u in (c + 1, c + 3):
            if rnd.random() < 0.6 and (q, u) not in p.used:
                probe(p, u, q)
    else:
        q = rnd.choice(list(range(88, 104)) + C2C_Q)
        p.rec(c, q, 'HostNotify', n=rnd.randrange(1, 16))


def t_c2c_limits(p, c):
    """Bursts of sends on one link spaced around c2c.cycles_per_vector (busy: LNK-001; the
    window runs from the last accepted vector), a second TX queue in the same cycle, and Receives
    around the arrivals, so that the receive FIFO (c2c.rx_depth) overruns (LNK-002) or just fits,
    a Receive one cycle before or in the arrival cycle of an overrunning vector."""
    rnd = p.rnd
    m = p.m
    cpv, depth, lat = m.cpv, m.rx_depth, m.link_lat
    link = rnd.randrange(m.n_links)
    side = rnd.choice('WE')
    rx_side = rnd.choice('WE')
    row_rx = 'W' if rx_side == 'E' else 'E'
    n = rnd.choice([2, 3, depth, depth + 1, depth + 1, depth + 2]) if depth <= 4 else rnd.choice([2, 3, depth + 1])
    t, sent = c, []
    for i in range(n):
        tx = rnd.choice(C2C_TX_Q[side])
        if rnd.random() < 0.2:
            p.rec(t, tx, 'Send', chip_id=0, link=link, stream=sref(side, rnd.randrange(32)))
        else:
            p.rec(t, tx, 'Transmit', chip_id=0, link=link)
        sent.append(t)
        if rnd.random() < 0.15:                      # the other TX queue in the same cycle: busy
            other = [q for q in C2C_TX_Q[side] if q != tx][0]
            if (other, t) not in p.used:
                p.rec(t, other, 'Transmit', chip_id=0, link=link)
        t += max(1, cpv + rnd.choice([-2, -1, 0, 0, 0, 0, 1, 3]))
    rx = rnd.choice(C2C_RX_Q[rx_side])
    for _ in range(rnd.randrange(0, 4)):
        k = rnd.randrange(len(sent))
        u = sent[k] + lat + rnd.choice([-1, 0, 0, 1, 2, 5])
        if (rx, u) not in p.used:
            p.rec(u, rx, 'Receive', chip_id=0, link=link, stream=sref(row_rx, rnd.randrange(32)))
    if rnd.random() < 0.3:                           # a Receive on the idle link before anything arrives
        u = c + rnd.randrange(1, lat)
        if (rx, u) not in p.used:
            p.rec(u, rx, 'Receive', chip_id=0, link=(link + 1) % m.n_links, stream=sref(row_rx, rnd.randrange(32)))


def t_deskewrecv(p, c):
    """, directed: a vector sent on a looped-back link and a DeskewRecv on a RX queue that polls for it; a probe on the
    queue behind it, at a random cycle up to two epochs on, shows when the queue is released. Weight 0: the random mixes and
    the golden vectors keep their seeds; the plan runs it as a directed program."""
    rnd = p.rnd
    side = rnd.choice('WE')
    tx = rnd.choice(C2C_TX_Q[side])
    rx_side = rnd.choice('WE')
    rx = rnd.choice(C2C_RX_Q[rx_side])
    row_rx = 'W' if rx_side == 'E' else 'E'
    link = rnd.randrange(p.m.n_links)
    if rnd.random() < 0.5:
        p.rec(c, tx, 'Transmit', chip_id=0, link=link)
    else:
        p.rec(c, tx, 'Send', chip_id=0, link=link, stream=sref(side, rnd.randrange(32)))
    t0 = c + rnd.choice([1, 2, 5, p.m.link_lat, p.m.hac])
    p.rec(t0, rx, 'DeskewRecv', chip_id=0, link=link, stream=sref(row_rx, rnd.randrange(32)))
    for u in sorted({t0 + rnd.randrange(1, 2 * p.m.hac + p.m.link_lat) for _ in range(2)}):
        if (rx, u) not in p.used:
            probe(p, u, rx)



def probe(p, c, q=None):
    rnd = p.rnd
    q = q if q is not None else rnd.choice(VXM_Q + SXM_Q + C2C_Q)
    if rnd.random() < 0.6:
        p.rec(c, q, 'NOP', n=rnd.choice([1, 1, 2, 7]))
    else:
        p.rec(c, q, 'Config', v=rnd.randrange(65536))


def t_barrier(p, c):
    rnd = p.rnd
    s = c
    sync_q = rnd.choice([144, 144, rnd.choice(VXM_Q), rnd.randrange(88)])
    p.rec(s, sync_q, 'Sync')
    notifies = []
    for _ in range(rnd.choice([0, 1, 1, 1, 2, 2])):
        n = s + rnd.choice([0, 0, 1, 3, 10])
        nq = rnd.choice([144, sync_q if sync_q != 144 else 90, rnd.choice(VXM_Q)])
        if (nq, n) in p.used:
            continue
        p.rec(n, nq, 'Notify')
        notifies.append(n)
    if rnd.random() < 0.2:
        p.rec(s - rnd.randint(1, 5), 144, 'Notify')
    pq = [sync_q if sync_q < 144 else rnd.choice(VXM_Q), rnd.choice(VXM_Q), rnd.choice(SXM_Q)]
    pts = [s - 1, s, s + 1]
    for n in notifies:
        pts += [n + 33, n + 34, n + 35, n + 36]
    for t in pts:
        for q in pq:
            if rnd.random() < 0.6 and (q, t) not in p.used:
                probe(p, t, q)
    # a Notify on a parked queue
    if rnd.random() < 0.4 and notifies:
        q = rnd.choice(VXM_Q)
        t = max(notifies) + rnd.randint(1, 20)
        if (q, t) not in p.used:
            p.rec(t, q, 'Notify')


def t_repeat(p, c):
    rnd = p.rnd
    q = rnd.choice(VXM_Q + [rnd.randrange(88)])
    if q < 88:
        p.core(q)
        kind = rnd.choice(['Read', 'Write'])
        p.rec(c, q, kind, addr=rnd.randrange(WORDS), thread=0,
              **({('dst' if kind == 'Read' else 'src'): sref(rnd.choice('EW'), rnd.randrange(32))}))
    else:
        if not t_vxm(p, c, q=q):
            p.rec(c, q, 'NOP', n=1)
    c2 = c + rnd.choice([1, 1, 2])
    n, d = rnd.choice([1, 2, 3, 5]), rnd.choice([1, 2, 3, 7])
    p.rec(c2, q, 'Repeat', n=n, d=d)
    end = c2 + n * (d + 1)
    for t in (end - 1, end, end + 1, end + 2):
        if t > c2 and rnd.random() < 0.5 and (q, t) not in p.used:
            probe(p, t, q)


def t_setvl(p, c):
    """SetVL on a VXM or SXM queue, then a result on that queue: superlanes at or beyond the
    length are zero in every stream the queue writes (MA 3)."""
    rnd = p.rnd
    if rnd.random() < 0.6:
        q = rnd.choice(VXM_Q)
        p.rec(c, q, 'SetVL', n=rnd.choice([1, 5, 10, 19]))
        t_vxm(p, c + rnd.randint(1, 4), name=rnd.choice(['Add', 'Not', 'Max']), q=q)
    else:
        side = rnd.choice('EW')
        q = rnd.choice(SXM_KIND_Q[(side, 'SHF')])
        p.rec(c, q, 'SetVL', n=rnd.choice([1, 7, 19]))
        other = 'E' if side == 'W' else 'W'
        p.rec(c + rnd.randint(1, 4), q, 'Shift', dst=sref(other, rnd.randrange(32)), src=sref(side, rnd.randrange(32)),
              dir=rnd.randrange(2), amount=rnd.choice([0, 3]))


def t_park_repeat(p, c):
    """A Repeat whose iterations meet a barrier on its queue: they wait while the queue is
    parked and resume, with their spacing, at the first unparked cycle (MA 3)."""
    rnd = p.rnd
    q = rnd.choice(VXM_Q)
    if not t_vxm(p, c, name='Add', q=q):
        p.rec(c, q, 'NOP', n=1)
    n, d = rnd.choice([3, 5]), rnd.choice([2, 3])
    p.rec(c + 1, q, 'Repeat', n=n, d=d)
    s = c + 2 + d + rnd.randint(0, d)                   # park after the first iteration (at c + d + 1)
    sq = rnd.choice([144, 144, q])
    if (sq, s) not in p.used:
        p.rec(s, sq, 'Sync')
        nq = rnd.choice([144, rnd.choice(VXM_Q)])
        if (nq, s + 1) not in p.used:
            p.rec(s + rnd.randint(0, 3), nq, 'Notify')


def t_cu(p, c):
    rnd = p.rnd
    q = rnd.randrange(145)
    k = rnd.random()
    if k < 0.3:
        p.rec(c, q, 'NOP', n=rnd.choice([1, 2, 5, 300]))
        if rnd.random() < 0.5 and q < 144:
            probe(p, c + rnd.randint(1, 3), q)
    elif k < 0.4:
        p.rec(c, q, 'Ifetch', stream=rnd.randrange(64))
    elif k < 0.5:                                     # local-memory form
        p.rec(c, q, 'IfetchM', addr=rnd.choice([0, 5, rnd.randrange(WORDS)]))
    elif k < 0.7:
        p.rec(c, q, 'SetVL', n=rnd.randint(1, 20))
    elif k < 0.8:
        p.rec(c, q, 'Repeat', n=2, d=1)               # nothing to repeat yet
    else:
        q = rnd.randrange(88)
        p.core(q)
        name = rnd.choice(['Countdown', 'Step', 'StreamStep', 'Iterations', 'PowerConfig', 'DeepSleep'])
        p.rec(c, q, name, **{'Countdown': {'d': 3}, 'Step': {'a': 2}, 'StreamStep': {'s': 1}, 'Iterations': {'n': 1},
                             'PowerConfig': {'v': 9}, 'DeepSleep': {}}[name])


def t_junk(p, c):
    """Malformed or misplaced words."""
    rnd = p.rnd
    q = rnd.randrange(145)
    k = rnd.random()
    isa = p.isa
    if k < 0.25:
        e = rnd.choice(isa.entries)
        f = {}
        for (n, hi, lo, kind, ex) in e.vars:
            f[n] = nx.TYPES[rnd.choice([c for c in ex['codes'] if c < 16])] if kind == 'eltype_code' else rnd.getrandbits(hi - lo + 1)
        w = isa.encode(e.name, **f)
        p.raw(c, q, 1, w)
    elif k < 0.4:
        tag = rnd.choice([0, 3, 5, 1, 2, 4])
        p.raw(c, q, 1, (tag << 60) | rnd.getrandbits(60))
    elif k < 0.5:
        p.raw(c, q, 1, rnd.getrandbits(64) if rnd.random() < 0.5 else 0x00400000)
    elif k < 0.6:
        p.raw(c, q, 1, (rnd.choice([6, 7]) << 28) | rnd.getrandbits(28))     # tags 6 and 7 do not decode
    elif k < 0.75:
        s = rnd.randrange(88)
        p.core(s)
        if rnd.random() < 0.5:
            p.raw(c, s, 1, isa.encode('Step', a=rnd.choice([-8192, 8192, -8191, 8191, -1])))
        else:
            p.raw(c, s, 1, isa.encode('StreamStep', s=rnd.choice([31, 32, 0xffff])))
    elif k < 0.9:
        w = rnd.getrandbits(64)
        p.raw(c, q, 1, w)
        p.raw(c, q, 1, w ^ 1)                         # a second record in the same cycle
    else:
        p.raw(c, q, 1, rnd.getrandbits(32))


# ------------------------------------------------------------------ c6 mutation-gap templates
# One template per gap the c6 mutation round found (validation.md sec. 3): type pairs and
# special values the random templates rarely produce, and timing/capacity boundaries.

def f_specials(t):
    """Special encodings of a float type: zeros, infinities, NaNs, subnormals, the values
    around every integer type's range and the rounding ties."""
    f = nx.fmt(t)
    enc = encoder(t)
    out = [0, f.sign, f.inf(False), f.inf(True), f.cnan, f.cnan | f.sign, f.expmask | 1, 1, f.sign | 1,
           f.expmask - 1, f.sign | (f.expmask - 1)]
    for x in (0.5, -0.5, 1.5, 2.5, -2.5, 1.0, -1.0, 127.0, 127.5, 128.0, -128.0, -128.5, -129.0, 200.0, 255.0,
              255.5, 256.0, 32767.0, 32768.0, -32769.0, 65535.0, 65536.0, 2.0 ** 31, -2.0 ** 31 - 4096, 2.0 ** 32, 1e-3):
        try:
            out.append(enc(x))
        except OverflowError:
            out.append(f.inf(x < 0))
    return out


def i_specials(t):
    lo, hi = nx.irange(t)
    return [nx.raw(t, v) for v in sorted({0, 1, -1 if lo < 0 else 2, lo, hi, lo + 1, hi - 1, 127, 128, 255, 256, -128, -129,
                                          32767, 32768, 65535, 65536} ) if lo <= v <= hi]


def specials(t):
    if nx.is_bool(t):
        return [0, 1, nx.ones(t), 1 << (8 * nx.WIDTH[t] - 1)]
    return f_specials(t) if nx.is_float(t) else i_specials(t)


def special_vec(rnd, t):
    sp = specials(t)
    v = [sp[l % len(sp)] for l in range(320)]
    for l in range(len(sp), 320):
        if rnd.random() < 0.5:
            v[l] = rand_elem(rnd, t)
    return v


def vx_special(p, c, q, name, f, st, vals):
    """One VXM record with its operand lhs preset to vals (elements of type st)."""
    if p.isa.decode64(enc_rec(p.isa, name, **f)) is None:
        return False
    p.rec(c, q, name, **f)
    p.data[(f['lhs'], c + p.m.lat(name, f.get('type'))[1])] = to_planes(vals, nx.WIDTH[st])
    return True


def t_convert(p, c):
    """Every (source, destination) type pair of Convert (121), one per cycle, each on special
    values: float -> unsigned saturation, float -> bool on Inf/NaN, ties, signed zeros."""
    rnd = p.rnd
    p.nodrop = True
    i = 0
    for st in nx.TYPES:
        for dt in nx.TYPES:
            ws, wd = nx.WIDTH[st], nx.WIDTH[dt]
            f = {'type': st, 'dtype': nx.TYPES.index(dt), 'dst': stream_for(rnd, wd), 'lhs': stream_for(rnd, ws)}
            vx_special(p, c + 2 * i, VXM_Q[i % 16], 'Convert', f, st, special_vec(rnd, st))
            i += 1


def t_transc(p, c):
    """TanH, Exp, RSqrt and Log on every legal type at the special inputs (all 256 i8 values,
    0, -1, 1, max, min; float zeros, infinities, NaNs, subnormals)."""
    rnd = p.rnd
    p.nodrop = True
    i = 0
    for name in ('TanH', 'Exp', 'RSqrt', 'Log'):
        for t in ('i8', 'f16', 'f32'):
            w = nx.WIDTH[t]
            if t == 'i8':
                vals = [l & 255 for l in range(320)]
            else:
                vals = special_vec(rnd, t)
            f = {'type': t, 'dst': stream_for(rnd, w), 'lhs': stream_for(rnd, w)}
            vx_special(p, c + 2 * i, VXM_Q[i % 16], name, f, t, vals)
            i += 1


def t_reduce_order(p, c):
    """Float Reduce whose result depends on the accumulation order (ascending lanes, rounded
    at each step: microarchitecture.md sec. 5): big/small mixes, cancellation, ties."""
    rnd = p.rnd
    p.nodrop = True
    for i, t in enumerate(rnd.sample(['f32', 'f16', 'f32', 'f16'], rnd.choice([1, 2, 4]))):
        enc = f16 if t == 'f16' else f32
        big = 2.0 ** (24 if t == 'f32' else 11)
        vals = []
        for g in range(20):
            k = rnd.randrange(5)
            if k == 0:
                grp = [1.0, 1.0] + [0.0] * 13 + [big]
            elif k == 1:
                grp = [big] + [1.0] * 15
            elif k == 2:
                grp = [big, -big] + [rnd.choice([1.0, 0.5, 3.0]) for _ in range(14)]
            elif k == 3:
                grp = [rnd.choice([big, -big, 1.0, -1.0, 0.5, 1.5, 2.0 ** -8]) for _ in range(16)]
            else:
                grp = [rnd.uniform(-4, 4) * 2 ** rnd.randint(-8, 10) for _ in range(16)]
            vals += [enc(x) for x in grp]
        mode = 0 if rnd.random() < 0.8 else rnd.choice([1, 2])
        if mode and rnd.random() < 0.5:
            vals[rnd.randrange(320)] = nx.fmt(t).cnan
        f = {'type': t, 'mode': mode, 'dst': stream_for(rnd, nx.WIDTH[t]), 'lhs': stream_for(rnd, nx.WIDTH[t])}
        vx_special(p, c + 2 * i, VXM_Q[i], 'Reduce', f, t, vals)


def t_mxm_f16_special(p, c):
    """f16 matmul whose rows hold +Inf and -Inf products (Inf + -Inf = NaN), Inf x 0, NaN and
    all -0 products (MA 6)."""
    rnd = p.rnd
    p.nodrop = True
    plane = rnd.randrange(4)
    qs = MXM_Q[plane]
    row = 'W' if plane < 2 else 'E'
    g = (2 if row == 'W' else 0) + rnd.randrange(2)
    q = rnd.choice(qs)
    INF, NINF, NAN_, NZ, ONE, MONE = 0x7C00, 0xFC00, 0x7E00, 0x8000, 0x3C00, 0xBC00
    xs = [ONE, MONE, 0, ONE, ONE, MONE, NZ, ONE] + [0] * 312
    kinds = [[INF, INF], [INF, 0, 0, NINF], [0, 0, INF], [0, 0, 0, NAN_], [NZ] * 8, [INF, NINF], [NINF, INF], [ONE, MONE]]
    rows = []
    for r in range(8):
        k = kinds[r] if rnd.random() < 0.7 else rnd.choice(kinds)
        rows.append((k + [0] * 320)[:320])
    pl = []
    for r in rows:
        lo, hi = to_planes(r, 2)
        pl += [lo, hi]
    p.rec(c, q, 'LW', type=1, src=g)
    p.data[(g << 4, c + p.m.lat('LW')[1])] = pl
    c += p.m.lat('LW')[0]
    p.rec(c, q, 'IW', type=1)
    c += 1
    s = sref(row, rnd.randrange(16) * 2)
    p.rec(c, q, 'ABC', type=1, src=s)
    p.data[(s, c + p.m.lat('ABC')[1])] = to_planes(xs, 2)
    c += p.m.lat('ABC')[0]
    p.rec(c, q, 'ACC', type=3, dst=sref('E' if row == 'W' else 'W', rnd.randrange(8) * 4), mode=0, acc=rnd.randrange(p.m.acc_depth))


def t_iws_full(p, c):
    """rows/16 - 1, rows/16 or rows/16 + 1 IW-from-streams installs on one plane: mxm.rows rows fit (the 21st on ref is RES-002)."""
    rnd = p.rnd
    plane = rnd.randrange(4)
    q = rnd.choice(MXM_Q[plane])
    g0 = 2 if plane < 2 else 0
    p.nodrop = True
    K = p.m.rows // 16
    for i in range(rnd.choice([max(1, K - 1), K, K + 1, K + 1])):
        p.rec(c + i, q, 'IWS', type=0, src=g0 + rnd.randrange(2))


def t_iw_ready(p, c):
    """IW from the buffer one cycle before (RES-001) or exactly when the last LW commits."""
    rnd = p.rnd
    plane = rnd.randrange(4)
    qs = MXM_Q[plane]
    g0 = 2 if plane < 2 else 0
    p.nodrop = True
    n = rnd.choice([1, 2])
    for i in range(n):
        p.rec(c + i, rnd.choice(qs), 'LW', type=0, src=g0 + rnd.randrange(2))
    t = c + n - 1 + p.m.lat('LW')[0] + rnd.choice([-1, 0])
    p.rec(t, rnd.choice(qs), 'IW', type=0)
    s = sref('W' if plane < 2 else 'E', rnd.randrange(32))
    p.rec(t + p.m.lat('IW')[0] + rnd.choice([-1, 0, 0]), rnd.choice(qs), 'ABC', type=0, src=s)


def t_rotate_edge(p, c):
    """Rotate n with dst = 32 - n^2 (the last legal stream) or 33 - n^2 (OPR-002)."""
    rnd = p.rnd
    side = rnd.choice('EW')
    other = 'E' if side == 'W' else 'W'
    n = rnd.choice([3, 4])
    p.rec(c, rnd.choice(SXM_KIND_Q[(side, 'TSP')]), 'Rotate', dst=sref(other, 32 - n * n + rnd.choice([0, 0, 1])),
          src=sref(side, rnd.randrange(32)), n=n)


def t_segment(p, c):
    """SetSegment n, then a Read or Write at n - 1 (in segment) or n (OPR-001), n = 1, 16,
    8192 or random."""
    rnd = p.rnd
    s = mem_core_slice(p)
    n = rnd.choice([1, 16, 16, WORDS, WORDS, rnd.randrange(1, WORDS)])
    p.rec(c, s, 'SetSegment', n=n)
    a = n - 1 if n == WORDS or rnd.random() < 0.5 else n
    kind = rnd.choice(['Read', 'Write'])
    st = sref(rnd.choice('EW'), rnd.randrange(32))
    p.rec(c + rnd.choice([1, 2]), s, kind, addr=a, thread=0, **({'dst': st} if kind == 'Read' else {'src': st}))


def t_repeat_after(p, c):
    """Repeat after a Read + Write pair (not repeatable: RES-001) or after a NOP (the Read
    before the NOP is repeated)."""
    rnd = p.rnd
    s = mem_core_slice(p)
    row = rnd.choice('EW')
    p.rec(c, s, 'Read', addr=rnd.randrange(min(4096, WORDS)), dst=sref(row, rnd.randrange(32)), thread=0)
    c += rnd.choice([1, 2, 5])
    if rnd.random() < 0.5:
        a = rnd.randrange(4096)
        p.rec(c, s, 'Read', addr=a, dst=sref(row, rnd.randrange(32)), thread=0)
        p.rec(c, s, 'Write', addr=a | (1 << p.m.bank_bit), src=sref(row, rnd.randrange(32)), thread=0)
    else:
        p.rec(c, s, 'NOP', n=1)
    p.rec(c + rnd.choice([1, 1, 2]), s, 'Repeat', n=rnd.choice([1, 2]), d=rnd.choice([1, 2, 3]))


def t_acc_clash(p, c):
    """Two ACCs on one plane (any of the four, so both MXM hemispheres) in one cycle after a
    committed ABC: RES-003 for both, neither commits (MA 1)."""
    rnd = p.rnd
    p.nodrop = True
    plane = rnd.randrange(4)
    qs = MXM_Q[plane]
    row = 'W' if plane < 2 else 'E'
    p.rec(c, rnd.choice(qs), 'IWS', type=0, src=(2 if row == 'W' else 0) + rnd.randrange(2))
    c += 2
    p.rec(c, rnd.choice(qs), 'ABC', type=0, src=sref(row, rnd.randrange(32)))
    c += p.m.lat('ABC')[0] + rnd.choice([0, 0, 1])
    q1, q2 = rnd.sample(qs, 2)
    d = sref('E' if row == 'W' else 'W', rnd.randrange(8) * 4)
    p.rec(c, q1, 'ACC', type=2, dst=d, mode=0, acc=rnd.randrange(p.m.acc_depth))
    p.rec(c, q2, 'ACC', type=2, dst=sref('E' if row == 'W' else 'W', rnd.randrange(8) * 4), mode=0, acc=rnd.randrange(p.m.acc_depth))


def t_map_hemispheres(p, c):
    """LoadMap of one map index on both SXM hemispheres in one cycle (no clash), sometimes
    with a same-hemisphere clash added."""
    rnd = p.rnd
    mid = rnd.randrange(16)
    for side in 'WE':
        p.rec(c, rnd.choice(SXM_KIND_Q[(side, rnd.choice(['PRM', 'DST']))]), 'LoadMap', src=sref(side, rnd.randrange(32)), map=mid)
    if rnd.random() < 0.3:
        side = rnd.choice('WE')
        q = rnd.choice([x for x in SXM_KIND_Q[(side, 'DST')] if (x, c) not in p.used])
        p.rec(c, q, 'LoadMap', src=sref(side, rnd.randrange(32)), map=mid)


def t_srf_horizon(p, c):
    """A VXM result read by an upstream MEM Write around the cycle it leaves the row (the
    live-write horizon): DIR-001 while it is live downstream, TIM-001 after (MA 2)."""
    rnd = p.rnd
    row = rnd.choice('EW')
    q = rnd.choice(VXM_Q)
    d = sref(row, rnd.randrange(32))
    a = sref(rnd.choice('EW'), rnd.randrange(32))
    p.rec(c, q, 'Add', type='i8', dst=d, lhs=a, rhs=a, lhs_chain=0, rhs_chain=0)
    cc = c + p.m.lat('Add')[0]
    ups = [s for s in range(88) if s not in p.core_q and (p.m.lay.mem_gap(s, row) or 99) < p.m.lay.vxm]
    s = rnd.choice(ups)
    p.core(s)
    t = cc + (43 - p.m.lay.vxm) - p.m.lat('Write')[1] + rnd.choice([0, 1, 2, rnd.randint(-3, 5)])
    p.rec(t, s, 'Write', addr=rnd.randrange(WORDS), src=d, thread=0)
    p.nofeed.add((d, t + p.m.lat('Write')[1]))


def t_agt_overlap(p, c):
    """An R loop and a W loop of one queue, often in one bank, overlapped, with new
    references, whole-queue records, Ifetch and configuration records during them (the
    per-path dispatch, loop-reference and configuration-count rules)."""
    rnd = p.rnd
    s = mem_core_slice(p)
    row = rnd.choice('EW')
    st = lambda: sref(row, rnd.randrange(32))
    bank = 1 << p.m.bank_bit
    a = rnd.randrange(bank)
    b = a + rnd.randint(0, 40) if rnd.random() < 0.7 else (a | bank)
    p.rec(c, s, 'Read', addr=a, dst=st(), thread=0)
    p.rec(c + 1, s, 'Countdown', d=rnd.choice([2, 3, 4]))
    p.rec(c + 2, s, 'Iterations', n=rnd.choice([3, 5, 6]))
    w = c + rnd.randint(3, 8)
    p.rec(w, s, 'Write', addr=b, src=st(), thread=0)
    p.rec(w + 1, s, 'Countdown', d=rnd.choice([2, 3, 4, 7]))
    p.rec(w + 2, s, 'Iterations', n=rnd.choice([2, 3, 4]))
    for t in range(c + 3, c + 32):
        if (t in (w, w + 1, w + 2)) or rnd.random() > 0.3:
            continue
        kind = rnd.choice(['Read', 'Write', 'NOP', 'Ifetch', 'IfetchM', 'Iterations', 'Countdown'])
        if kind == 'Read':
            p.rec(t, s, kind, addr=rnd.choice([a, a | bank, rnd.randrange(WORDS)]), dst=st(), thread=rnd.choice([0, 0, 1]))
        elif kind == 'Write':
            p.rec(t, s, kind, addr=rnd.choice([b, a | bank, rnd.randrange(WORDS)]), src=st(), thread=rnd.choice([0, 0, 1]))
        elif kind == 'NOP':
            p.rec(t, s, kind, n=1)
        elif kind == 'Ifetch':
            p.rec(t, s, kind, stream=rnd.randrange(64))
        elif kind == 'IfetchM':
            p.rec(t, s, kind, addr=rnd.randrange(WORDS))
        elif kind == 'Iterations':
            p.rec(t, s, kind, n=rnd.choice([1, 2, 3]))
        else:
            p.rec(t, s, kind, d=rnd.choice([1, 2, 3]))


def t_agt_whole(p, c):
    """A whole-queue record (SetVL, Repeat, NOP, Config) due in the first cycle after a
    reference or one of its configuration records, when the path's configuration is still open
    (a reference due then, or not): it waits for the loop and its tail (design 7.6); a Read follows.
    Weight 0 and not in DIRECTED, so the golden vectors and the seeds of the random mixes stay
    as they were: model_test.sh runs it with `--only agt_whole --no-directed`."""
    rnd = p.rnd
    s = mem_core_slice(p)
    rd = rnd.random() < 0.6
    st = sref(rnd.choice('EW'), rnd.randrange(32))
    if rd:
        p.rec(c, s, 'Read', addr=rnd.randrange(WORDS), dst=st, thread=0)
    else:
        p.rec(c, s, 'Write', addr=rnd.randrange(WORDS), src=st, thread=0)
    t = c + 1
    for fld in rnd.choice([[], ['Countdown'], ['Countdown'], ['Countdown', 'Iterations'], ['Step', 'Iterations']]):
        if fld == 'Countdown':
            p.rec(t, s, fld, d=rnd.choice([1, 1, 2, 3]))
        elif fld == 'Step':
            p.rec(t, s, fld, a=rnd.choice([1, 2, 16]))
        else:
            p.rec(t, s, fld, n=rnd.choice([2, 3, 4]))
        t += 1
    kind = rnd.choice(['SetVL', 'SetVL', 'SetVL', 'NOP', 'Config'])
    if kind == 'SetVL':
        p.rec(t, s, 'SetVL', n=rnd.choice([1, 5, 19]))
    elif kind == 'NOP':
        p.rec(t, s, 'NOP', n=1)
    else:
        p.rec(t, s, 'Config', v=rnd.randrange(24))
    p.rec(t + 1 + rnd.randint(0, 2), s, 'Read', addr=rnd.randrange(WORDS), dst=sref(rnd.choice('EW'), rnd.randrange(32)), thread=0)


TEMPLATES = dict(vxm=(t_vxm, 6), vxm_chain=(t_vxm_chain, 1), vxm_listing1=(t_vxm_listing1, 1), vxm_collide=(t_vxm_collide, 1), vxm_cmp=(t_vxm_cmp, 1), mem=(t_mem, 5),
                 mxm=(t_mxm, 3), mxm_f16=(t_mxm_f16, 1), mxm_full=(t_mxm_full, 1), sxm_wide=(t_sxm_wide, 1), sxm=(t_sxm, 4), c2c=(t_c2c, 1), c2c_limits=(t_c2c_limits, 0), host=(t_host, 1), barrier=(t_barrier, 2), repeat=(t_repeat, 2),
                 cu=(t_cu, 1), junk=(t_junk, 2), clash=(t_state_clash, 1), setvl=(t_setvl, 1),
                 park_repeat=(t_park_repeat, 1), convert=(t_convert, 1), transc=(t_transc, 1), reduce_order=(t_reduce_order, 1),
                 mxm_f16_special=(t_mxm_f16_special, 1), iws_full=(t_iws_full, 1), iw_ready=(t_iw_ready, 1),
                 rotate_edge=(t_rotate_edge, 1), segment=(t_segment, 1), repeat_after=(t_repeat_after, 1),
                 acc_clash=(t_acc_clash, 1), map_hemispheres=(t_map_hemispheres, 1), srf_horizon=(t_srf_horizon, 1), agt_overlap=(t_agt_overlap, 1), agt_whole=(t_agt_whole, 0))


def build(m, seed, kinds):
    """One program: the core templates, the element dry run, then feeds and drains."""
    rnd = random.Random(seed)
    p = Prog(rnd, m)
    for kind in kinds:
        TEMPLATES[kind][0](p, rnd.randint(60, 220))
    core = list(p.recs)
    m.run(images_of(core, p.units), dict(p.pre), dry=True)
    elems = list(m.elems)
    # SRAM contents and dumps for the core MEM elements
    for (q, name, f, c0, k, off, soff) in elems:
        if q >= 88 or name not in ('Read', 'Write', 'Gather', 'Scatter'):
            continue
        addrs = []
        if name in ('Read', 'Write'):
            addrs = [f['addr'] + off] if 0 <= f['addr'] + off < WORDS else []
        else:
            ma = getattr(p, 'mapaddrs', {}).get((q, f['map']))
            if ma:
                tiles = [rnd.choice(ma) for _ in range(20)]
                addrs = [a + off for a in tiles if 0 <= a + off < WORDS]
                mp = bytearray(rand_bytes(rnd))
                for g, a in enumerate(tiles):
                    mp[16 * g] = a & 255
                    mp[16 * g + 1] = (a >> 8) | (rnd.choice([0, 0, 0x20, 0xe0]))
                p.data[(f['map'], c0 + m.lat(name)[1] + k)] = [bytes(mp)]
        for a in addrs:
            if (q, a) not in p.pre and rnd.random() < 0.95:
                p.pre[(q, a)] = rand_bytes(rnd)
            p.dump.add((q, a))
    p.attach(elems)
    p.pre = {k: bytes(v)[:m.vl].ljust(m.vl, b'\0') for k, v in p.pre.items()}      # the templates build 320-lane words; a narrower profile keeps its first vlanes bytes, a wider one is zero above lane 320
    return p


def kinds_for(rnd, i, n, exts=()):
    names = list(TEMPLATES)
    weights = [TEMPLATES[k][1] for k in names]
    if exts:                                # a profile with an extension also draws the extension MXM template (the base profiles' seeds stay)
        names, weights = names + ['mxm_ext'], weights + [3]
    return [rnd.choices(names, weights)[0] for _ in range(rnd.choice([1, 1, 2, 3, 4]))]


DIRECTED = (['vxm'] * 40 + ['vxm_chain'] * 6 + ['vxm_listing1'] * 4 + ['vxm_collide'] * 4 + ['vxm_cmp'] * 16 + ['mem'] * 16 + ['mxm'] * 14 + ['mxm_f16'] * 6 + ['mxm_full'] * 8 + ['sxm_wide'] * 8 + ['sxm'] * 30
            + ['c2c'] * 16 + ['c2c_limits'] * 12 + ['barrier'] * 16 + ['repeat'] * 12 + ['cu'] * 8 + ['junk'] * 16 + ['clash'] * 10
            + ['setvl'] * 10 + ['park_repeat'] * 10
            # c6 mutation gaps (validation.md sec. 3)
            + ['convert'] * 3 + ['transc'] * 3 + ['reduce_order'] * 8 + ['mxm_f16_special'] * 8 + ['iws_full'] * 6
            + ['iw_ready'] * 10 + ['rotate_edge'] * 8 + ['segment'] * 12 + ['repeat_after'] * 10 + ['acc_clash'] * 10
            + ['map_hemispheres'] * 6 + ['srf_horizon'] * 16
            + ['agt_overlap'] * 24)                 # F4-AGT


# ------------------------------------------------------------------ running and comparing

def images_of(recs, units=None):
    """The text images of cycle-view records (c, queue (144 = control), 1, word, 0): the normative
    lowering (tsp_as.lower), then the units a template gave directly."""
    images = tsp_as.lower([(c, 0xffff if q == 144 else q, w0) for (c, q, wc, w0, w1) in recs])
    for q, us in (units or {}).items():
        images[q] = images.get(q, []) + us
    return dict(sorted(images.items(), key=lambda kv: (kv[0] == 0xffff, kv[0])))


def sim_input(p):
    lines = []
    for (s, a), v in sorted(p.pre.items()):
        lines.append('S %x %x ' % (s, a) + ' '.join('%x' % b for b in v))
    lines += tsp_as.q_lines(images_of(p.recs, p.units))
    for (s, a) in sorted(p.dump):
        lines.append('D %x %x' % (s, a))
    for f in p.faults:
        if f[0] == 'sram':
            lines.append('F sram %x %x %x %x %x' % f[1:])
        elif f[0] == 'link':
            lines.append('F link %x %x %x %s' % (f[1], f[2], f[3], 'uncorr' if f[4] else 'corr'))
        elif f[0] == 'mxm':
            lines.append('F mxm %x %x %x %x' % f[1:])
        elif f[0] == 'iq':
            lines.append('F iq %x %x %x %x' % f[1:])
        else:
            lines.append('F srf %x %s %x %x %x %x' % (f[1], f[2].lower(), f[3], f[4], f[5], f[6]))
    return '\n'.join(lines) + '\n'


def gen_faults(m, p, seed):
    """ECC faults for a built program (--ecc): flips of the words the program preloads (at cycle 0) and of the values
    its instructions commit (a register on the value's diagonal, a few cycles after the commit), a second and a third
    bit of the same target now and then (an uncorrectable superlane, or a miscorrection by three bits)."""
    rnd = random.Random(seed ^ 0x5eed)
    ev = m.run(images_of(p.recs, p.units), dict(p.pre))
    srf = [e for e in ev if e[0] == 'srf']
    sends = list(m.sends)                                    # (cycle, link) of the vectors the program sent
    words = sorted(p.pre)
    out = []
    if m.abcs and rnd.random() < 0.4:                        # matrix-unit faults (sec. 18): flips in the exact sums of ABCs; no other fault in the program, so that every ABC reads its activation
        for c, pl in rnd.sample(m.abcs, min(len(m.abcs), rnd.choice([1, 1, 2]))):
            lanes = rnd.sample(range(m.rows), rnd.choice([1, 1, 1, 2]))
            out += [('mxm', c, pl, ln, rnd.randrange(96)) for ln in lanes]
        return out
    nflits = 2 + (m.vl + 7) // 8
    if sends:                                                # link FEC faults (microarchitecture.md sec. 17): flits of a sent vector
        for _ in range(rnd.choice([0, 1, 1, 2])):
            c, l = rnd.choice(sends)
            for fl in rnd.sample(range(nflits), rnd.choice([1, 1, 2, 3])):
                if not any(x[0] == 'link' and x[1:4] == (c, l, fl) for x in out):
                    out.append(('link', c, l, fl, rnd.random() < 0.5))
    imgs = {q: len([u for u in us if u != mc.PADDING]) for q, us in images_of(p.recs, p.units).items() if q != mc.CTRL_ID}
    texts = sorted(q for q, n in imgs.items() if n)
    for _ in range(rnd.choice([1, 1, 2, 3])):
        if texts and rnd.random() < 0.3:                      # a flip in a text word of an initial image, at cycle 0 (the word is in the IQ then)
            q = rnd.choice(texts)
            w = rnd.randrange((imgs[q] + 3) // 4)
            bits = rnd.sample(range(138), rnd.choice([1, 1, 1, 2, 2, 3]))
            out += [f for f in [('iq', 0, q, w, b) for b in bits] if f not in out]     # a second line for one bit and cycle is an input error
        elif words and (not srf or rnd.random() < 0.5):
            s_, a = rnd.choice(words)
            g = rnd.randrange(m.sl)
            bits = rnd.sample(range(138), rnd.choice([1, 1, 1, 2, 2, 3]))
            out += [('sram', 0, s_, a, g, b) for b in bits]
        elif srf:
            c, row, st, gp = rnd.choice(srf)[1:5]
            d = rnd.randrange(1, 8)
            if gp + d > 43:
                continue
            g = rnd.randrange(m.sl)
            bits = rnd.sample(range(138), rnd.choice([1, 1, 1, 2, 2, 3]))
            out += [('srf', c + d, row, st, gp + d, g, b) for b in bits]
    out = [f for k, f in enumerate(out) if f not in out[:k]]         # one line per bit and cycle (a second is an input error)
    if any(f[0] == 'iq' for f in out):                       # a damaged text word may keep the Transmit from running: a link fault would then be an input error
        out = [f for f in out if f[0] != 'link']
    return out


def oracle_output(m, p):
    ev = m.run(images_of(p.recs, p.units), dict(p.pre), faults=p.faults)
    out = []
    diags = [e for e in ev if e[0] == 'diag']
    for e in ev:
        if e[0] == 'srf':
            out.append('T srf %d %s:%d gap %d %s' % (e[1], e[2], e[3], e[4], e[5].hex()))
        elif e[0] == 'sram':
            out.append('T sram %d slice %d addr %d %s' % (e[1], e[2], e[3], e[4].hex()))
        elif e[0] == 'ecc':
            if e[5] is None:                                 # a text word has no superlane
                out.append('T ecc %d queue %d %s %s pos %d' % (e[1], e[2], 'unc' if e[3] else 'corr', e[4], e[6]))
            else:
                out.append('T ecc %d queue %d %s %s sl %d pos %d' % (e[1], e[2], 'unc' if e[3] else 'corr', e[4], e[5], e[6]))
        elif e[0] == 'ecc_spare':
            out.append('T ecc %d queue %d spare %s sl %d pos %d' % e[1:])
        elif e[0] == 'ecc_mxm':
            out.append('T ecc %d queue %d mxm corr plane %d lane %d' % e[1:])
        elif e[0] == 'ecc_link':
            out.append('T ecc %d link %d %s flit %d' % (e[1], e[2], 'unc' if e[3] else 'corr', e[4]))
        elif e[0] == 'scrub':
            out.append('T ecc %d scrub slice %d addr %d sl %d%s' % (*e[1:5], ' unc' if e[5] else ''))
        elif e[0] == 'lint':
            out.append('T lint %d queue %d %s' % (e[1], e[2], e[3]))
        elif e[0] == 'irq':
            out.append('T irq %d msix %d' % (e[1], e[2]))
        elif e[0] == 'dma_out':
            out.append('T dma %d out' % e[1])
        else:
            out.append('T diag %d queue %d %s' % (e[1], e[2], e[3]))
    out.append('E %s %d %d' % ((diags[0][3], diags[0][1], len(diags)) if diags else ('-', -1, 0)))
    for (s, a) in sorted(p.dump):
        v = m.sram.get((s, a), bytes(m.vl))
        out.append('W %x %x ' % (s, a) + ' '.join('%02x' % b for b in v) + ' 00' * (320 - m.vl))
    for e in ev:
        if e[0] == 'dma_out':                              # the host's egress list: one Y line per PciSend
            out.append('Y %d ' % e[1] + ' '.join('%02x' % b for b in e[2]))
    return out


WORDS = 8192                         # sram.words_per_slice of the profile (set by init_worker)
WORKER = {}


def init_worker(sim, cfg, impl=None, impl_lint=False, ecc=False):
    import json
    with open(cfg) as fh:
        c = json.load(fh)
    global WORDS
    WORDS = c['sram']['words_per_slice']
    WORKER.update(sim=sim, cfgpath=cfg, m=mc.Machine(c), impl=impl, impl_lint=impl_lint, ecc=ecc)


def canonical(lines):
    """The --impl comparison form (microarchitecture.md sec. 12, porting-guide.md sec. 3): within
    each run of consecutive `T srf` lines of one cycle, the lines are sorted by (row, stream,
    gap), so the order an implementation emits a cycle's commits in does not matter. Every other line keeps its position."""
    out, run = [], []
    def key(ln):
        t = ln.split()                  # T srf <cycle> <row>:<stream> gap <gap> <value>
        row, st = t[3].split(':')
        return (row, int(st), int(t[5]), ln)
    def flush():
        out.extend(sorted(run, key=key))
        run.clear()
    for ln in lines:
        t = ln.split()
        if len(t) == 7 and t[:2] == ['T', 'srf'] and (not run or run[0].split()[2] == t[2]):
            run.append(ln)
            continue
        flush()
        if len(t) == 7 and t[:2] == ['T', 'srf']:
            run.append(ln)
        else:
            out.append(ln)
    flush()
    return out


def _check_canonical():
    """canonical() sorts only a cycle's run of T srf lines; checked on every run."""
    a = ['T diag 5 queue 7 OTS-E-IQ-002', 'T srf 5 W:1 gap 3 0a0b', 'T srf 5 E:30 gap 11 0a0b', 'T srf 5 E:4 gap 11 0a0b',
         'T srf 6 E:0 gap 2 0a0b', 'T diag 6 queue 4 OTS-E-TIM-001', 'T srf 7 E:9 gap 1 0a0b', 'E - -1 0']
    b = ['T diag 5 queue 7 OTS-E-IQ-002', 'T srf 5 E:4 gap 11 0a0b', 'T srf 5 E:30 gap 11 0a0b', 'T srf 5 W:1 gap 3 0a0b',
         'T srf 6 E:0 gap 2 0a0b', 'T diag 6 queue 4 OTS-E-TIM-001', 'T srf 7 E:9 gap 1 0a0b', 'E - -1 0']
    assert canonical(a) == b and canonical(b) == b, canonical(a)
    c = ['T srf 6 E:0 gap 2 0a0b', 'T diag 6 queue 4 OTS-E-TIM-001']
    assert canonical(list(reversed(c))) != canonical(c)          # a diag never moves across T srf lines


def over_range_only(lines):
    """True when the diagnostics of a run (the `T diag` lines) are all OTS-E-OPR-001, an operand out of range, and there is at least one. On a
    profile with fewer superlanes than the generator's literal draws, such a program exercises nothing but the range check."""
    codes = {ln.split()[-1] for ln in lines if ln.startswith('T diag ')}
    return bool(codes) and codes == {'OTS-E-OPR-001'}


def one(job):
    idx, seed, kinds = job
    m = WORKER['m']
    p = build(m, seed, kinds)
    if WORKER.get('ecc'):
        p.faults = gen_faults(m, p, seed)
    inp = sim_input(p)
    r = subprocess.run([WORKER['sim'], WORKER['cfgpath'], '--trace', '--lint'], input=inp, capture_output=True, text=True,
                       timeout=900)
    if r.returncode == 2 and 'initial IQ image longer than icu.iq_depth' in r.stderr:
        return (idx, kinds, None, 'skip', len(p.recs), False)     # the generator does not size programs to icu.iq_depth: a narrow IQ rejects the input
    got = [ln.rstrip() for ln in r.stdout.splitlines() if ln.strip()]
    over = over_range_only(got)
    if WORKER.get('impl'):
        # an implementation need not report lints (validation.md): its lint lines are compared only with --impl-lint
        ri = subprocess.run(WORKER['impl'].split() + [WORKER['cfgpath'], '--trace'] + (['--lint'] if WORKER.get('impl_lint') else []), input=inp,
                            capture_output=True, text=True, timeout=900)
        want = canonical([ln.rstrip() for ln in ri.stdout.splitlines() if ln.strip()])
        got = canonical(got if WORKER.get('impl_lint') else [ln for ln in got if not ln.startswith('T lint ')])
    else:
        want = oracle_output(m, p)
    if got == want:
        return (idx, kinds, None, None, len(p.recs), over)
    k = next((i for i in range(min(len(got), len(want))) if got[i] != want[i]), min(len(got), len(want)))
    return (idx, kinds, inp, (k, got[k:k + 3], want[k:k + 3], r.stderr[-300:]), len(p.recs), over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim')
    ap.add_argument('config')
    ap.add_argument('n', type=int)
    ap.add_argument('seed', type=int)
    ap.add_argument('--jobs', type=int, default=int(os.environ.get('ORACLE_JOBS', os.cpu_count() or 1)))
    ap.add_argument('--only')
    ap.add_argument('--no-directed', action='store_true')
    ap.add_argument('--ecc', action='store_true', help='inject ECC faults (F lines) into every program: flips of preloaded words and of committed values (microarchitecture.md sec. 15)')
    ap.add_argument('--keep', default=tempfile.gettempdir())
    ap.add_argument('--impl', help='compare tsp_sim with this implementation instead of the Python oracle')
    ap.add_argument('--impl-lint', action='store_true', help='the implementation takes --lint: compare its T lint lines too (lints are non-normative)')
    ap.add_argument('--max-fail', type=int, default=int(os.environ.get('ORACLE_MAX_FAIL', 0)))
    a = ap.parse_args()
    _check_canonical()
    rnd = random.Random(a.seed)
    jobs = []
    with open(a.config) as fh:
        exts = json.load(fh).get('extensions', [])
    # a profile with an extension also gets directed runs of the extension templates: bf16 on the VXM, bf16 and bfp8 through the MXM, x128 through ACC
    ext_plan = ([['mxm_ext']] * 60 if exts else []) + ([['vxm']] * 80 + [['vxm_cmp']] * 20 + [['vxm_sg4']] * 20 if 'bf16' in exts else [])
    plan = [] if a.no_directed else [[k] for k in DIRECTED] + ext_plan
    plan += [kinds_for(rnd, i, a.n, exts) for i in range(a.n)]
    if a.only:
        plan = [[a.only] for _ in plan]
    for i, kinds in enumerate(plan):
        jobs.append((i, a.seed * 1000003 + i, kinds))
    bad = 0
    skipped = 0
    total = 0
    recs = 0
    narrow = 0
    # : on a profile that declares an extension, the compared stream must include at least one extension instruction
    # (bf16 / x128 / bfp8). Count them while building the plan's programs (cheap: decode the records, no simulator).
    ext_seen = set()
    if exts and not a.only:
        with open(a.config) as fh:
            cfg = json.load(fh)
        m0 = mc.Machine(cfg)
        for i, kinds in enumerate(plan):
            p0 = build(m0, a.seed * 1000003 + i, kinds)
            ext_seen |= extension_opcodes_in(m0.isa, p0.recs)
            if ext_seen:
                break
    with multiprocessing.Pool(a.jobs, init_worker, (a.sim, a.config, a.impl, a.impl_lint, a.ecc)) as pool:
        for (idx, kinds, inp, why, nrec, over) in pool.imap_unordered(one, jobs):
            total += 1
            recs += nrec
            narrow += over
            if why is None:
                continue
            if why == 'skip':
                skipped += 1
                continue
            bad += 1
            if bad <= 5:
                path = os.path.join(a.keep, 'diff_oracle_seed%d_prog%d.in' % (a.seed, idx))
                with open(path, 'w') as fh:
                    fh.write(inp)
                k, got, want, err = why
                print('MISMATCH program %d %s at output line %d (reproducer: %s %s --trace --lint < %s)'
                      % (idx, kinds, k, a.sim, a.config, path))
                print('  tsp_sim: %s' % got)
                print('  %s: %s' % ('impl  ' if a.impl else 'oracle', want))
                if err:
                    print('  stderr : %s' % err)
            if a.max_fail and bad >= a.max_fail:
                pool.terminate()
                break
    print('diff_oracle: %d programs (%d records), %d mismatches' % (total, recs, bad) +
          (', %d skipped (initial IQ image longer than icu.iq_depth)' % skipped if skipped else ''))
    if exts and not a.only:
        print('diff_oracle: extension opcodes in the compared stream: %s' %
              (', '.join(sorted(ext_seen)) if ext_seen else '(none)'))
        if not ext_seen:
            print('diff_oracle: : a profile with extensions=%s produced no bf16/x128/bfp8 instruction' % (exts,))
            bad = max(bad, 1)
    with open(a.config) as fh:
        sl = json.load(fh)['geometry']['superlanes']
    if sl != 20 and total:
        # the generator draws superlane and lane numbers for the 20 superlanes of ref; on another geometry part of its programs only report OTS-E-OPR-001
        print('diff_oracle: %d of %d programs (%.1f %%) report nothing but operands out of range (OTS-E-OPR-001) on %d superlanes: the generator\'s '
              'literal draws are for 20' % (narrow, total, 100.0 * narrow / total, sl))
    sys.exit(1 if bad else 0)


# =======================================================================================
# SEL and TSP writes of one SXM hemisphere to one stream in one cycle (the OR
# merge of HC34 p.31), and the groups that still collide (SHF or PRM with SEL and TSP, two
# TSPs with SEL). The two hemispheres turn out onto different rows, so never meet.
def t_sxm_or(p, c):
    rnd = p.rnd
    side = rnd.choice('EW')
    other = 'E' if side == 'W' else 'W'
    own = lambda: sref(side, rnd.randrange(32))
    dg = (2 if other == 'W' else 0) + rnd.randrange(2)
    tgt = sref(other, 16 * (dg % 2) + rnd.randrange(16))
    sg = (2 if side == 'W' else 0) + rnd.randrange(2)
    p.rec(c, rnd.choice(SXM_KIND_Q[(side, 'TSP')]), 'Transpose', dst=dg, src=sg)
    p.rec(c, SXM_KIND_Q[(side, 'SEL')][0], 'SelectS', dst=tgt, a=own(), b=own(), c=own(), sel=own())
    extra = rnd.random()
    if extra < 0.15:
        p.rec(c, rnd.choice(SXM_KIND_Q[(side, 'SHF')]), 'Shift', dst=tgt, src=own(), dir=rnd.randrange(2), amount=rnd.randrange(max(320, p.m.vl)))
    elif extra < 0.25:
        p.rec(c, SXM_KIND_Q[(side, 'TSP')][1] if rnd.random() < 0.5 else SXM_KIND_Q[(side, 'TSP')][0], 'Transpose', dst=dg, src=sg ^ 1)
    elif extra < 0.35:
        p.rec(c, SXM_KIND_Q[(side, 'PRM')][0], 'Permute', dst=tgt, src=own(), map=rnd.randrange(16))


TEMPLATES['sxm_or'] = (t_sxm_or, 1)
DIRECTED += ['sxm_or'] * 12
# The VXM group bit.
TEMPLATES['vxm_sg4'] = (t_vxm_sg4, 3)
TEMPLATES['mxm_fp8'] = (t_mxm_fp8, 2)
TEMPLATES['mxm_ext'] = (t_mxm_ext, 0)        # weight 0: only a profile with an extension draws it (kinds_for, DIRECTED in main), so the seeds of the others stay
DIRECTED += ['vxm_sg4'] * 16
# =======================================================================================


# =======================================================================================
# One clean program per VXM operation and for Transpose, so that every catalogue entry
# executes without a diagnostic in some program (the oracle and the golden vectors). Each program
# issues the instruction alone; build() feeds its operands and drains its results.
def _t_one_vxm(name):
    def t(p, c):
        for _ in range(50):
            if t_vxm(p, c, name=name):
                return
    return t


def t_transpose_alone(p, c):
    rnd = p.rnd
    side = rnd.choice('EW')
    other = 'E' if side == 'W' else 'W'
    sg = (2 if side == 'W' else 0) + rnd.randrange(2)
    dg = (2 if other == 'W' else 0) + rnd.randrange(2)
    p.rec(c, rnd.choice(SXM_KIND_Q[(side, 'TSP')]), 'Transpose', dst=dg, src=sg)


for _n in VXM_NAMES:
    TEMPLATES['one_' + _n.lower()] = (_t_one_vxm(_n), 0)
    DIRECTED += ['one_' + _n.lower()] * 2
TEMPLATES['one_transpose'] = (t_transpose_alone, 0)
DIRECTED += ['one_transpose'] * 2


def t_distribute_alone(p, c):
    """One LoadMap and one Distribute with an in-superlane map (some entries >= 16
    zero-fill), so that a golden vector runs Distribute without a diagnostic."""
    rnd = p.rnd
    side = rnd.choice('EW')
    own = lambda: sref(side, rnd.randrange(32))
    mid = rnd.randrange(16)
    lq = sxm_side_q(p, side, 'DST')
    lo_s, src = own(), own()
    rc = c - rnd.randint(1, 6)
    p.data[(lo_s, rc)] = [bytes(rnd.choice([rnd.randrange(16), 16]) for _ in range(320))]
    p.rec(rc, lq, 'LoadMap', src=lo_s, map=mid)
    p.rec(c, sxm_side_q(p, side, 'DST'), 'Distribute', dst=sref(side, rnd.randrange(32)), src=src, map=mid)


TEMPLATES['one_distribute'] = (t_distribute_alone, 0)
TEMPLATES['deskewrecv'] = (t_deskewrecv, 0)
DIRECTED += ['deskewrecv'] * 10


def t_partial_tiles(p, c):
    """SetSegment n, then a Gather or Scatter whose tile addresses are partly inside and
    partly outside the segment: the good tiles act, the bad ones drive zero (Gather) or write
    nothing (Scatter), and one OPR-001 is reported."""
    rnd = p.rnd
    s = mem_core_slice(p)
    row = rnd.choice('EW')
    n = rnd.choice([1, 8, 20, 100, 4000])
    p.rec(c, s, 'SetSegment', n=n)
    st, mp = sref(row, rnd.randrange(32)), sref(row, rnd.randrange(32))
    k, th = rnd.randrange(16), rnd.choice([0, 0, 1])
    if rnd.random() < 0.5:
        p.rec(c + 1, s, 'Gather', dst=st, map=mp, k=k, thread=th)
    else:
        p.rec(c + 1, s, 'Scatter', src=st, map=mp, k=k, thread=th)
    pool = [rnd.randrange(n), rnd.randrange(n), min(WORDS - 1, n), min(WORDS - 1, n + rnd.randrange(1, 50)), rnd.randrange(WORDS)]
    p.mapaddrs = getattr(p, 'mapaddrs', {})
    p.mapaddrs[(s, mp)] = pool


TEMPLATES['partial_tiles'] = (t_partial_tiles, 0)
DIRECTED += ['partial_tiles'] * 14      # 'mem' gives up 14 slots so that the plan keeps its length (the seeds of the random mixes)
# =======================================================================================


# =======================================================================================
# The instruction front end (microarchitecture.md sec. 3 "Instruction front end"):
# waiting instead of errors, NOP prefix timing, Ifetch / IfetchM text arriving at the fetch
# latency and rate, padding, the lint IQ-001 and the diagnostics IQ-002 to IQ-004.

def text_unit_pool(p, q):
    """Units of one random text for MEM queue q: instructions (bundle bit random), NOP prefixes,
    padding, a tag-6 unit."""
    rnd, isa = p.rnd, p.isa
    out = []
    for _ in range(rnd.randint(4, 40)):
        k = rnd.random()
        b = rnd.choice([0, 0, 0, 1 << 31])
        if k < 0.30:
            w = enc_rec(isa, 'Read', addr=rnd.randrange(WORDS), dst=sref('EW'[rnd.randrange(2)], rnd.randrange(32)), thread=rnd.choice([0, 0, 1]))
            out += [(w >> 32) | b, w & 0xffffffff]
        elif k < 0.50:
            w = enc_rec(isa, 'Write', addr=rnd.randrange(WORDS), src=sref('EW'[rnd.randrange(2)], rnd.randrange(32)), thread=rnd.choice([0, 0, 1]))
            out += [(w >> 32) | b, w & 0xffffffff]
        elif k < 0.60:
            out.append(enc_rec(isa, 'SetVL', n=rnd.randint(1, 20)) | b)
        elif k < 0.75:
            out.append(rnd.choice([1, 2, 3, 9, 40, 300]))
        elif k < 0.85:
            out.append(0xffffffff)
        elif k < 0.90:
            out.append((rnd.choice([6, 7]) << 28) | rnd.getrandbits(28) | b)
        elif k < 0.95:
            out.append(enc_rec(isa, 'SetVL', n=rnd.randint(1, 20)))
        else:
            out.append(enc_rec(isa, 'Config', v=rnd.randrange(30)) | b)
    return out


def text_word(units, vl):
    """The SRAM word (vl bytes) that holds the units (little endian), padded with 0xff."""
    us = (units + [0xffffffff] * (vl // 4))[:vl // 4]
    return b''.join(u.to_bytes(4, 'little') for u in us)


def t_front_fetch(p, c):
    """IfetchM text arriving later (lint IQ-001 when the queue idled), at the fetch rate, with padding."""
    rnd = p.rnd
    q = mem_core_slice(p, exists=True)           # its SRAM holds the text (an absent slice has none)
    a = rnd.choice([0, 6, rnd.randrange(min(8000, WORDS - 1))])
    vl = p.m.vl
    us = text_unit_pool(p, q)
    cut = (vl // 4) if rnd.random() < 0.7 else rnd.randint(1, vl // 4)
    p.pre[(q, a)] = text_word(us[:cut], vl)
    p.pre[(q, a + 1)] = text_word(us[cut:], vl)
    p.rec(c, q, 'IfetchM', addr=a)
    if rnd.random() < 0.5:                                  # the head parcel wants to run before the text arrives
        p.rec(c + 1, q, 'SetVL', n=rnd.randint(1, 20))
    if rnd.random() < 0.3:
        p.rec(c + rnd.randint(3, 60), q, 'IfetchM', addr=rnd.choice([0, 6, a]))   # a second fetch queues behind the first


def t_front_overflow(p, c):
    """A queue whose initial image nearly fills the IQ, then a fetch: units lost to the full IQ (IQ-002)."""
    rnd = p.rnd
    q = mem_core_slice(p, exists=True)
    cap = p.m.iq_cap
    a = rnd.randrange(min(8000, WORDS - 1))
    p.pre[(q, a)] = text_word(text_unit_pool(p, q), p.m.vl)
    p.pre[(q, a + 1)] = text_word(text_unit_pool(p, q), p.m.vl)
    w = enc_rec(p.isa, 'IfetchM', addr=a)
    fill = [enc_rec(p.isa, 'SetVL', n=rnd.randint(1, 20)) for _ in range(cap - 3 - rnd.choice([0, 0, 20, 100]))]
    p.units[q] = [w] + fill
    if rnd.random() < 0.5:
        p.units[q] = [rnd.choice([1, 5, 30])] + p.units[q]


def t_front_burst(p, c):
    """Instructions given faster than the queue takes them: they wait (no QUE-001, no BAR-001)."""
    rnd = p.rnd
    s = mem_core_slice(p)
    row = rnd.choice('EW')
    bb = 1 << p.m.bank_bit
    cyc = c
    for _ in range(rnd.randint(3, 10)):
        k = rnd.random()
        if k < 0.30:
            p.rec(cyc, s, 'Read', addr=rnd.randrange(64) | rnd.choice([0, bb]), dst=sref(row, rnd.randrange(32)), thread=rnd.choice([0, 0, 1]))
        elif k < 0.55:
            p.rec(cyc, s, 'Write', addr=rnd.randrange(64) | rnd.choice([0, bb]), src=sref(row, rnd.randrange(32)), thread=rnd.choice([0, 0, 1]))
        elif k < 0.70:
            fld = rnd.choice(['Countdown', 'Step', 'Iterations', 'StreamStep'])
            p.rec(cyc, s, fld, **{'Countdown': {'d': rnd.randint(0, 3)}, 'Step': {'a': rnd.randint(-3, 3)},
                                  'Iterations': {'n': rnd.randint(1, 4)}, 'StreamStep': {'s': rnd.randint(0, 2)}}[fld])
        elif k < 0.80:
            p.rec(cyc, s, 'Repeat', n=rnd.randint(1, 3), d=rnd.randint(1, 3))
        elif k < 0.90:
            p.rec(cyc, s, 'SetVL', n=rnd.randint(1, 20))
        else:
            p.rec(cyc, s, 'NOP', n=rnd.randint(1, 4))
        cyc += rnd.choice([0, 0, 1, 1, 1, 2, 5])


def t_front_barrier(p, c):
    """Queues that wait behind a Sync of the control channel, released by a Notify of another queue."""
    rnd = p.rnd
    p.rec(c, 144, 'Sync')
    q = rnd.randrange(104, 130)
    p.rec(c + rnd.randint(1, 12), q, 'Notify')
    for _ in range(rnd.randint(2, 6)):
        s = mem_core_slice(p)
        p.rec(c + rnd.randint(0, 30), s, 'Read', addr=rnd.randrange(64), dst=sref(rnd.choice('EW'), rnd.randrange(32)), thread=0)


def t_front_ifetch(p, c):
    """Ifetch on one stream operand (ISCA20 III-A3): the two text vectors are two consecutive vectors of the
    stream, Read by an upstream MEM slice in consecutive cycles so that they reach the fetching queue at its
    execute cycle t and t + 1; sometimes one is a cycle off or missing (TIM-001, no fetch), any stream (odd too).
    Weight 0 and not in DIRECTED, so the golden vectors and the seeds of the random mixes stay as they were:
    model_test.sh runs it with `--only front_ifetch --no-directed`."""
    rnd, m = p.rnd, p.m
    q = mem_core_slice(p, exists=True)
    unit, sl, _, _ = mc.queue_unit(q)
    row = rnd.choice('EW')
    st = rnd.randrange(32)
    g = m.lay.read_gap(unit, sl, mc.srow(sref(row, st)))
    prods = []
    for s2 in range(88):
        if s2 == q or not m.lay.present(s2):
            continue
        u2 = mc.queue_unit(s2)
        gw = m.lay.write_gap(u2[0], u2[1], mc.srow(sref(row, st)))
        if g is not None and gw is not None and gw < g:
            prods.append((s2, gw))
    if not prods:
        return
    s2, gw = rnd.choice(prods)
    p.core(s2)
    df, _ = m.lat('Read')
    _, ds = m.lat('Ifetch')
    c = max(c, 150)
    t = c + ds
    a = rnd.randrange(min(8000, WORDS - 2))
    us = text_unit_pool(p, q) if rnd.random() < 0.7 else []      # no text: padding only, which runs nothing (the clean golden vector)
    cut = (m.vl // 4) if rnd.random() < 0.7 else rnd.randint(1, m.vl // 4)
    p.pre[(s2, a)] = text_word(us[:cut], m.vl)
    p.pre[(s2, a + 1)] = text_word(us[cut:], m.vl)
    p.rec(c, q, 'Ifetch', stream=sref(row, st))
    for k in range(2):
        if rnd.random() < 0.15:
            continue                                     # a vector that never arrives
        off = rnd.choice([0, 0, 0, 0, 1, -1])            # sometimes a cycle off
        p.rec(t + k + off - df - (g - gw), s2, 'Read', addr=a + k, dst=sref(row, st), thread=0)


TEMPLATES['front_fetch'] = (t_front_fetch, 2)
TEMPLATES['front_ifetch'] = (t_front_ifetch, 0)
TEMPLATES['front_overflow'] = (t_front_overflow, 1)
TEMPLATES['front_burst'] = (t_front_burst, 3)
TEMPLATES['front_barrier'] = (t_front_barrier, 1)
DIRECTED += ['front_fetch'] * 30 + ['front_overflow'] * 8 + ['front_burst'] * 30 + ['front_barrier'] * 10
# =======================================================================================


if __name__ == '__main__':
    main()
