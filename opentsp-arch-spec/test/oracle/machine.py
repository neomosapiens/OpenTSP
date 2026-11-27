"""The cycle machine of the oracle (microarchitecture.md, topology.md, encoding.md).

Machine(cfg).run(images, preload) executes one object (the text images of the tsp_sim Q
lines: queue -> list of 32-bit units; 0xffff or 144 is the control channel) on one chip whose C2C links loop back (microarchitecture.md 10), and returns the
event list the simulator prints with --trace:
  ('srf', commit_cycle, row, stream, gap, value)   a stream register write entering the SRF; value: the
                                                   committed vector (vlanes bytes, after any merge)
  ('sram', cycle, slice, addr, value)              an SRAM word write; value: the word written
  ('diag', cycle, queue, code)              a diagnostic (queue 65535 = control channel)
in emission order, and the final SRAM contents.

Every rule cites the section of microarchitecture.md (MA) it implements. The oracle is
written from those documents and the catalogue, not translated from the Sail model.
"""
from fractions import Fraction

from . import numerics as nx
from . import ecc as ec
from .isa import ISA

CTRL = 144
CTRL_ID = 65535

D = dict(
    DEC_OPCODE='OTS-E-DEC-003',
    QUE_BUSY='OTS-E-QUE-001', QUE_UNIT='OTS-E-QUE-002',
    TIM='OTS-E-TIM-001', DIR='OTS-E-DIR-001', SRF='OTS-E-SRF-002', MEM_BANK='OTS-E-MEM-001',
    OPR_RANGE='OTS-E-OPR-001', OPR_ALIGN='OTS-E-OPR-002', RES='OTS-E-RES-001',
    RES_FULL='OTS-E-RES-002', RES_STATE='OTS-E-RES-003', SXM_PERM='OTS-E-SXM-001',
    HIO_BUSY='OTS-E-HIO-001', ECC_UNC='OTS-E-ECC-001', ECC_TXT='OTS-E-ECC-002', LNK_UNC='OTS-E-LNK-004', ECC_MXM='OTS-E-ECC-003',   # ECC_MXM: the sum check of an ABC found an error it cannot correct (sec. 18);
    # LNK_UNC: a Receive took a vector whose link transfer was uncorrectable
    LNK_BUSY='OTS-E-LNK-001', LNK_OVERRUN='OTS-E-LNK-002',   # link busy (sender), receive overrun (harness)
    LNK_PACKET='OTS-E-LNK-003',                              # a Send or Transmit released into a csr or ctrl packet in flight (no host script in a run of the oracle: never raised)
    IQ_OVERFLOW='OTS-E-IQ-002', IQ_STAGING='OTS-E-IQ-003',   # units lost to a full IQ, a fetch that finds the staging FIFO full
    IQ_STRANDED='OTS-E-IQ-004')                              # text left in an IQ at the end of the run
# States of a link FIFO entry
LV_PENDING, LV_QUEUED, LV_TAKEN, LV_DROPPED = range(4)
# Simulator lints (model/core/lints.sail): non-normative, reported apart
# from the diagnostics. One line per lint here, as in lints.sail.
LINT = dict(
    IQ_UNDERFLOW='OTS-W-IQ-001',      # IQ underflow (MA 3 front end)
    RAS_EVICT='OTS-W-RAS-001')        # a live spare-bit error report was evicted from the 256 the array keeps (MA 15.5)
PADDING = 0xFFFFFFFF                                 # a padding unit is dropped on enqueue
FETCH = ('Ifetch', 'IfetchM')                        # the Ifetch slot of a MEM queue

LOOPABLE = ('Read', 'Write', 'Gather', 'Scatter')
# MA 4 AGT table: field order within a level, and the field holding the value
AGT_FIELD = {'Countdown': 0, 'Step': 1, 'StreamStep': 2, 'Iterations': 3}
AGT_VALUE = {'Countdown': 'd', 'Step': 'a', 'StreamStep': 's', 'Iterations': 'n'}
CONTROL_OK = ('NOP', 'Sync', 'Notify', 'Config')     # MA 3 control channel
ROW_GAPS = 45                                        # registers 0..44 per row (MA 2)

VXM_BIN = ['Add', 'Add_sat', 'Add_mod', 'Sub', 'Sub_sat', 'Sub_mod', 'Mul', 'Mul_sat', 'Mul_mod',
           'Max', 'Min', 'And', 'Or', 'Xor']
VXM_UN = {'Not': 14, 'Neg': 15, 'Abs': 16, 'ReLU': 20, 'TanH': 21, 'Exp': 22, 'RSqrt': 23, 'Log': 24}
VXM_CONV = {'Conv_I8_F32': 25, 'Conv_F32_I8': 26, 'Conv_I8_F16': 27, 'Conv_F16_I8': 28,
            'Conv_I8_I32': 29, 'Conv_I32_I8': 30, 'Conv_F16_F32': 31, 'Conv_F32_F16': 32}
MXM_WT = {0: 'i8', 1: 'f16', 2: 'u8', 3: 'fp8', 4: 'bf16', 5: 'bfp8'}
PAIR = ('f16', 'bf16', 'bfp8')      # weight types of two byte planes (or data and scales) per element and a plane pair


class Diag(Exception):
    def __init__(self, code):
        Exception.__init__(self, code)
        self.code = code


def srow(s):
    return 'W' if s & 32 else 'E'


def sid(s):
    return s & 31


# ------------------------------------------------------------------ topology (topology.md 4)

# mxm.planes_per_hemisphere of the configuration under test (set by Machine): the plane of an east MXM queue is that many
# plus the index of its plane in the hemisphere.
MXM_PLANES = 2


def mxm_queue_present(q):
    """The MXM queue q (104..115) exists when the index of its plane in its hemisphere is below mxm.planes_per_hemisphere."""
    k = (0 if q <= 106 else 1) if q <= 109 else (0 if q <= 112 else 1)
    return k < MXM_PLANES


def queue_unit(q):
    """(unit, mem slice, mxm plane, sxm hemisphere) of data queue q."""
    if q <= 43:
        return ('MEM_W', q, 0, 0)
    if q <= 87:
        return ('MEM_E', q, 0, 0)
    if q <= 103:
        return ('VXM', 0, 0, 0)
    if q <= 109:                                   # MXM: 3 queues per plane (topology.md 5)
        return ('MXM_W', 0, 0 if q <= 106 else 1, 0)
    if q <= 115:
        return ('MXM_E', 0, MXM_PLANES + (0 if q <= 112 else 1), 0)
    east = q >= 130                                # 14-queue SXM blocks, C2C included
    k = q - (130 if east else 116)
    h = 'E' if east else 'W'
    if k >= 10:
        return ('C2C_%s_%s' % (h, 'TX' if k <= 11 else 'RX'), 0, 0, 1 if east else 0)
    kind = 'DST' if k <= 3 else 'TSP' if k <= 5 else 'SHF' if k <= 7 else 'PRM' if k == 8 else 'SEL'
    return ('SXM_%s_%s' % (h, kind), 0, 0, 1 if east else 0)


def sxm_or_pair(qa, qb):
    """SEL and TSP of one SXM hemisphere feed an OR (HC34 p.31)."""
    ua, ub = queue_unit(qa)[0], queue_unit(qb)[0]
    if not (ua.startswith('SXM_') and ub.startswith('SXM_')) or ua[4] != ub[4]:
        return False
    return {ua[6:], ub[6:]} == {'SEL', 'TSP'}


class Layout:
    def __init__(self, cfg):
        L = cfg['layout']
        self.mem_per_hemi = cfg['geometry']['mem_slices_per_hemisphere']
        self.vxm, self.up, self.down = L['vxm_gap'], L['mem_up_gap_base'], L['mem_down_gap_base']
        self.mxm_in, self.mxm_acc = L['mxm_in_gap'], L['mxm_acc_gap']
        self.dst, self.tin, self.tout = L['sxm_dst_gap'], L['sxm_turn_in_gap'], L['sxm_turn_out_gap']
        self.tx, self.rx = L['c2c_tx_gap'], L['c2c_rx_gap']

    def present(self, sl):
        """MEM slice sl exists when (sl mod 44) < geometry.mem_slices_per_hemisphere."""
        return sl % 44 < self.mem_per_hemi

    def mem_gap(self, sl, row):
        if not self.present(sl):                    # an absent slice has no station
            return None
        quad = (sl % 44) // 4
        west = sl < 44
        inward = (row == 'E') if west else (row == 'W')
        g = self.up - quad if inward else self.down + quad
        return g if 0 <= g <= 43 else None

    def read_gap(self, unit, sl, row):
        if unit == 'VXM':
            return self.vxm
        if unit in ('MEM_W', 'MEM_E'):
            return self.mem_gap(sl, row)
        if unit == 'MXM_W':
            return self.mxm_in if row == 'W' else None
        if unit == 'MXM_E':
            return self.mxm_in if row == 'E' else None
        if unit.startswith('SXM_'):
            side, kind = unit[4], unit[6:]
            if row != side:
                return None
            return self.dst if kind == 'DST' else self.tin
        if unit in ('C2C_W_TX', 'C2C_W_RX'):                # RX: Ifetch only
            return self.tx if row == 'W' else None
        if unit in ('C2C_E_TX', 'C2C_E_RX'):
            return self.tx if row == 'E' else None
        return None

    def write_gap(self, unit, sl, row):
        if unit == 'VXM':
            return self.vxm
        if unit in ('MEM_W', 'MEM_E'):
            return self.mem_gap(sl, row)
        if unit == 'MXM_W':
            return self.mxm_acc if row == 'E' else None
        if unit == 'MXM_E':
            return self.mxm_acc if row == 'W' else None
        if unit.startswith('SXM_'):
            side, kind = unit[4], unit[6:]
            if kind == 'DST':
                return self.dst if row == side else None
            return self.tout if row != side else None
        if unit == 'C2C_W_RX':
            return self.rx if row == 'E' else None
        if unit == 'C2C_E_RX':
            return self.rx if row == 'W' else None
        return None


# ------------------------------------------------------------------ the stream register file

class SRF:
    """MA 2: a value committed at gap g in cycle c is in register g + k at cycle c + k and
    falls off after register 44. A reader at gap G in cycle t sees the write with the
    largest gap g < G on diagonal G - t. With none, it is DIR-001 if the stream has live
    writes on the row and all of them are at or downstream of G, else TIM-001."""

    def __init__(self):
        self.w = {}          # (row, s) -> {(gap, diagonal): value}
        self.ck = {}         # (row, s) -> {(gap, diagonal): the explicit check bits (oracle/ecc.py) or None: valid}
        self.reps = []       # spare-bit error reports (sec. 15.5): dicts row, diag, gap, sp (30 masks, one per stream), pr (priority per superlane)
        self.rep_n = 0       # new reports written: the array keeps the 256 latest (sec. 15.5, report retention)

    def write(self, row, s, g, c, v, ck=None):
        self.w.setdefault((row, s), {})[(g, g - c)] = v
        self.ck.setdefault((row, s), {})[(g, g - c)] = ck

    def best(self, row, s, g, t):
        """The write a reader at gap g in cycle t sees: (its gap, value, check bits), or None."""
        ws = self.w.get((row, s), {})
        best = None
        for (wg, d), v in ws.items():
            if d == g - t and wg < g and (best is None or wg > best[0]):
                best = (wg, v, self.ck[(row, s)].get((wg, d)))
        return best

    def read(self, row, s, g, t):
        best = self.best(row, s, g, t)
        if best:
            return ('ok', best[1])
        ws = self.w.get((row, s), {})
        live = [wg for (wg, d) in ws if d + t <= ROW_GAPS - 1]
        if live and min(live) >= g:
            return ('up', None)
        return ('none', None)

    def read_ck(self, row, s, g, t):
        """read, with the check bits as a third result."""
        best = self.best(row, s, g, t)
        if best:
            return ('ok', best[1], best[2])
        k, _ = self.read(row, s, g, t)
        return (k, None, None)

    def report(self, row, g, c, prio, xs, sp, nsl):  # True: a report still in its row was evicted (lint OTS-W-RAS-001)
        """Record a report written in cycle c at gap g of the row for the superlanes in the mask xs. sp[s] has bit x set for
        superlane x of stream s (0..2 the error types, 4..29 the auxiliary streams); S[3] is the even parity of S[0..2],
        recomputed from the combined reports. Where a report of the same register already holds a superlane, the auxiliary streams of
        the higher priority win (equal: OR)."""
        d = g - c
        e = None
        for r in self.reps:
            if r['row'] == row and r['diag'] == d and r['gap'] == g:
                e = r
        evicted = False
        if e is None:
            e = {'row': row, 'diag': d, 'gap': g, 'sp': [0] * 30, 'pr': [0] * 32}
            if len(self.reps) < 256:
                self.reps.append(e)
            else:                                            # the slot of the oldest report
                k = self.rep_n % 256
                evicted = self.reps[k]['diag'] + c <= 44
                self.reps[k] = e
            self.rep_n += 1
        for x in range(nsl):
            if not (xs >> x) & 1:
                continue
            for s in range(30):
                bit = sp[s] & (1 << x)
                if s < 3:
                    e['sp'][s] |= bit
                elif s >= 4:
                    if e['pr'][x] < prio:
                        e['sp'][s] = (e['sp'][s] & ~(1 << x)) | bit
                    elif e['pr'][x] == prio:
                        e['sp'][s] |= bit
            par = ((e['sp'][0] >> x) ^ (e['sp'][1] >> x) ^ (e['sp'][2] >> x)) & 1
            e['sp'][3] = (e['sp'][3] & ~(1 << x)) | (par << x)
            e['pr'][x] = max(e['pr'][x], prio)
        return evicted

    def spare(self, row, s, g, t):
        """The superlane mask of the spare bits a reader at gap g in cycle t sees on stream s: the reports of its diagonal below
        its gap, unless a data write of the diagonal between the report's gap and the reader's replaced the register."""
        m = 0
        if s >= 30:
            return 0
        ws = self.w.get((row, s), {})
        for r in self.reps:
            if r['row'] == row and r['diag'] == g - t and r['gap'] < g:
                if not any(d == r['diag'] and r['gap'] < wg < g for (wg, d) in ws):
                    m |= r['sp'][s]
        return m

    def s03(self, row, g, t, x):
        return sum(((self.spare(row, s, g, t) >> x) & 1) << s for s in range(4))

    def flip(self, row, s, gap, c, sl, bit, nsl):
        """ECC fault injection: flip a bit of the value in register `gap` of the stream at cycle c: the nearest
        upstream write of the diagonal gap - c. A register that holds no value has nothing to flip."""
        ws = self.w.get((row, s), {})
        best = None
        for (wg, d), v in ws.items():
            if d == gap - c and wg < gap and (best is None or wg > best[0]):
                best = wg, d
        if best is None or sl >= nsl:
            return
        v, ck = ec.flip(ws[best], self.ck[(row, s)].get(best), sl, bit, nsl)
        ws[best] = v
        self.ck[(row, s)][best] = ck


class Plane:
    """One MXM plane (MA 6): buffer, installed weights, ABC results, mxm.acc_depth accumulators. The buffer and the weights hold
    `wvals` = 2 * mxm.f16_rows stream values."""

    def __init__(self, vl, wvals=320, acc_depth=2):
        z = bytes(vl)
        self.btype = 'i8'
        self.buf_rows = 0
        self.buf = [z] * wvals
        self.ready_lw = 0
        self.wtype = 'i8'
        self.rows = 0
        self.weights = [z] * wvals
        self.iw_open = False
        self.iws_cycle = 0
        self.ready_iw = 0
        self.abc = []            # (dispatch, commit, type, product, poison, queue, wide), newest first
        self.acc = [[0] * 320 for _ in range(acc_depth)]
        self.acc128 = [[nx.WIDE_ZERO] * 320 for _ in range(acc_depth)]      # the wide accumulators (extension wide_acc)
        self.buf_bad = [False] * wvals    # the buffer / weight rows read with an uncorrectable superlane
        self.w_bad = [False] * wvals
        self.acc_poison = [0] * acc_depth        # the superlanes of each accumulator that hold poison

    def copy(self):
        p = Plane.__new__(Plane)
        p.__dict__ = dict(self.__dict__)
        p.buf = list(self.buf)
        p.weights = list(self.weights)
        p.abc = list(self.abc)
        p.acc = [list(a) for a in self.acc]
        p.acc128 = [list(a) for a in self.acc128]
        p.buf_bad = list(self.buf_bad)
        p.w_bad = list(self.w_bad)
        p.acc_poison = list(self.acc_poison)
        return p


PART_FIELDS = {
    'BUF': ('btype', 'buf_rows', 'buf', 'buf_bad', 'ready_lw'),
    'W': ('wtype', 'rows', 'weights', 'w_bad', 'iw_open', 'iws_cycle', 'ready_iw'),
    'ABC': ('abc',),
}


# The bank use of an address outside the slice: it accesses no bank.
NO_BANK = 'none'

class Machine:
    def __init__(self, cfg, isa=None):
        self.cfg = cfg
        self.sl = cfg['geometry']['superlanes']
        self.vl = 16 * self.sl
        self.words = cfg['sram']['words_per_slice']
        self.bank_bit = cfg['sram']['bank_bit']
        self.addr_mask = (1 << cfg['sram']['addr_bits']) - 1   # a map entry addresses with its low addr_bits bits
        global MXM_PLANES
        MXM_PLANES = cfg['mxm']['planes_per_hemisphere']
        self.planes = MXM_PLANES
        self.rows = cfg['mxm']['rows']                       # stream values in an i8 weight buffer or matrix
        self.f16_rows = cfg['mxm']['f16_rows']               # the rows of an f16 matrix (two stream values each)
        self.wvals = 2 * self.f16_rows                       # stream values a weight buffer or matrix holds
        self.acc_depth = cfg['mxm']['acc_depth']
        self.fp8_rate = cfg['mxm']['fp8_rate']
        self.lay = Layout(cfg)
        tm = cfg['timing']
        self.tm = {k: (v['d_func'], v['d_skew']) for k, v in tm.items() if isinstance(v, dict)}
        # timing keys that are not part of a class row
        self.map_delay = tm['sxm_map_ready']                  # LoadMap/LoadMapWide execute -> Permute may use the map
        self.occ_f16 = tm['mxm_abc']['occupancy_f16']         # an f16 ABC holds its plane's MACCs this many cycles
        self.rtd_floor = tm['c2c']['rtdeskew_floor']          # the shortest hold of a RuntimeDeskew
        # An entry's latency is its timing.overrides row, else its class row
        self.overrides = {o['entry']: (o['d_func'], o['d_skew']) for o in tm.get('overrides', [])}
        self.bar_lat = cfg['barrier']['release_latency']
        self.link_lat = cfg['c2c']['link_latency']
        self.n_links = cfg['c2c']['links']
        self.hac = cfg['c2c']['hac_period']
        self.cpv = cfg['c2c']['cycles_per_vector']          # one vector per cpv cycles per link
        self.rx_depth = cfg['c2c']['rx_depth']              # the receive FIFO of a link
        self.host_cpv = cfg['host']['cycles_per_vector']    # the PCIe interface accepts one vector per host_cpv cycles
        self.n_doorbells = cfg['host']['doorbells']
        # The front end (MA 3 "Instruction front end"): IQ capacity and fetch path, in 32-bit units.
        icu = cfg['icu']
        self.iq_cap = icu['iq_depth'] // 4
        self.fetch_lat = icu['fetch_latency']
        self.fetch_rate = icu['fetch_bytes_per_cycle'] // 4
        self.staging = icu['fetch_staging_depth']            # payloads a queue's fetch staging FIFO holds
        self.exts = tuple(cfg.get('extensions', []))
        self.wide = 'wide_acc' in self.exts
        self.isa = isa or ISA(vlanes=self.vl, addr_bits=cfg['sram']['addr_bits'], acc_depth=cfg['mxm']['acc_depth'], extensions=self.exts)
        self.zero = bytes(self.vl)

    def lat(self, name, ty=None):
        """(d_func, d_skew) of a catalogue entry at element type `ty` (the type name, 'i8'; core/timing.sail entry_latency_typed):
        the override row '<entry>.<ty>', else the entry's override row, else its class row."""
        if isinstance(ty, str) and (name + '.' + ty) in self.overrides:
            return self.overrides[name + '.' + ty]
        return self.overrides.get(name) or self.tm[self.isa.timing(name).lower()]

    # Reference k of an instruction dispatched at issue executes (reads) at
    # issue + k + d_skew and commits at issue + k + d_func.
    def rc(self):
        return self.issue + self.elem + self.ds

    def cc(self):
        return self.issue + self.elem + self.df

    # ------------------------------------------------------------ events
    def report(self, q, c, code):
        self.events.append(('diag', c, CTRL_ID if q == CTRL else q, code))

    def lint(self, q, c, code):
        """A simulator lint: an event of its own kind, never a diagnostic (not counted in E)."""
        self.events.append(('lint', c, CTRL_ID if q == CTRL else q, code))

    # ------------------------------------------------------------ decode (MA 8, encoding.md)
    def decode_text(self, us):
        """(True, (name, fields)) or (False, code) for the instruction that starts the units us:
        bits 30..28 of the first unit select the layout (tags 0, 3, 5: one unit; 1, 2, 4: two,
        the first unit holding bits 63..32); bit 31 is the bundle bit and is not decoded; tags 6
        and 7 do not decode."""
        t = (us[0] >> 28) & 7
        hi = us[0] & 0x7FFFFFFF
        if t in (0, 3, 5):
            i = self.isa.decode32(hi)
        elif t in (1, 2, 4):
            i = self.isa.decode64((hi << 32) | us[1])
        else:
            i = None
        return (True, i) if i else (False, D['DEC_OPCODE'])

    # ------------------------------------------------------------ barrier (MA 3)
    def parked(self, q, c):
        def one(k):
            s, r = self.parked_since.get(k), self.release_at.get(k)
            return s is not None and s <= c and (r is None or c < r)
        return one(q) or (q != CTRL and one(CTRL))

    # HostWait (core/csr.sail): parks the queue until the host rings the doorbell. The oracle has no
    # host script: nothing is ever pushed or rung, so a PciReceive finds the ingress FIFO empty and a
    # HostWait parks its queue for good (unless a doorbell is already set, which only the host can do).
    def host_parked(self, q, c):
        s, r = self.hw_since.get(q), self.hw_release.get(q)
        return s is not None and s <= c and (r is None or c < r)

    def barrier_sync(self, k, c):
        self.parked_since[k] = c
        self.release_at[k] = c + self.bar_lat if self.last_notify == c else None

    def barrier_notify(self, n):
        self.last_notify = n
        for k, s in self.parked_since.items():
            if self.release_at.get(k) is None and s <= n:
                self.release_at[k] = n + self.bar_lat

    def barrier_retire(self, c):
        for k in list(self.parked_since):
            r = self.release_at.get(k)
            if r is not None and c >= r:
                del self.parked_since[k]
                del self.release_at[k]

    # ------------------------------------------------------------ tile gating (MA 3, 4)
    def tile_on(self, g, mem):
        q = self.q
        if g >= self.qvl.get(q, self.sl):
            return False
        if not mem:
            return True
        split = self.split.get(q, self.sl)
        return g < split if self.thread == 0 else g >= split

    def gate(self, v):
        """Zero the superlanes of inactive tiles in a value this queue writes."""
        q = self.q
        mem = q <= 87
        if all(self.tile_on(g, mem) for g in range(self.sl)):
            return v
        out = bytearray(v)
        for g in range(self.sl):
            if not self.tile_on(g, mem):
                out[16 * g:16 * g + 16] = bytes(16)
        return bytes(out)

    # ------------------------------------------------------------ operands (MA 2)
    def aligned(self, s, w):
        i = sid(s)
        return i % w == 0 and i + w <= 32

    def read_planes(self, s, w, t, chk=True):
        """chk: check the ECC of each stream value (microarchitecture.md sec. 15); not chk gives (value, check bits)
        pairs as they flow (the MEM Write's data)."""
        row = srow(s)
        g = self.lay.read_gap(self.unit, self.slice, row)
        if g is None:
            raise Diag(D['DIR'])
        if not self.aligned(s, w):
            raise Diag(D['OPR_ALIGN'])
        return self._read_n(row, sid(s), w, g, t, chk)

    def read_plane_raw(self, s, t):
        return self.read_planes(s, 1, t, False)[0]

    def ecc_event(self, c, unc, place, sl, pos, q=None):
        q = self.q if q is None else q
        self.events.append(('ecc', c, CTRL_ID if q == CTRL else q, unc, place, sl, pos))
        if unc:                                           # the error interrupt (sec. 16): the first change of ERR_STATUS bit 1 (data) or 3 (text)
            bit = 3 if place.startswith('text') else 1
            if not self.estat & (1 << bit):
                self.estat |= 1 << bit
                self.events.append(('irq', c, 0))

    def ecc_spare(self, c, row, stream, sl, s03):
        self.events.append(('ecc_spare', c, CTRL_ID if self.q == CTRL else self.q, '%s:%d' % (row, stream), sl, s03))
        self.estat |= 1 << 8                              # ERR_STATUS bit 8: no interrupt (ERR_MASK resets to 0xAA)

    def ras_irq(self, c, bit):
        """The error interrupt of a matrix or link event (sec. 16): the first change of ERR_STATUS bit `bit` (4 matrix corrected, 5
        matrix uncorrectable, 6 link corrected, 7 link uncorrectable) raises MSI-X vector 0 when ERR_MASK (reset 0x0AA: the uncorrectable
        bits; the oracle has no host script to change it) enables it."""
        if not self.estat & (1 << bit):
            self.estat |= 1 << bit
            if (0xAA >> bit) & 1:
                self.events.append(('irq', c, 0))

    def ecc_check_value(self, t, row, stream, v, ck, spare=0, g_read=None):
        """The consumer's check of one stream value, superlane by superlane: the value after the single-bit
        corrections; the uncorrectable superlanes join self.eb (and the value's own set is self.last_bad)."""
        self.last_bad = 0
        if ck is None and not spare:
            return v
        out = bytearray(v)
        for g in range(self.sl):
            if (spare >> g) & 1 or (ck is not None and ck[g][1]):          # an error report (sec. 15.5): logged by its consumer
                self.ecc_spare(t, row, stream, g, self.srf.s03(row, g_read, t, g) if g_read is not None else 0)
            if ck is None:
                continue
            kind, d, pos = ec.decode(v[16 * g:16 * g + 16], ck[g][0])
            if kind == 'corr':
                out[16 * g:16 * g + 16] = d
                self.ecc_event(t, False, '%s:%d' % (row, stream), g, pos)
            elif kind == 'unc':
                self.eb |= 1 << g
                self.last_bad |= 1 << g
                self.ecc_event(t, True, '%s:%d' % (row, stream), g, pos)
        return bytes(out)

    def _read_n(self, row, base, w, g, t, chk=True):
        out, err = [], None
        self.rbad = [False] * w                                 # which of the streams read had an uncorrectable superlane
        for b in range(w):
            k, v, ck = self.srf.read_ck(row, base + b, g, t)
            if k == 'ok':
                sp = self.srf.spare(row, base + b, g, t)
                out.append(self.ecc_check_value(t, row, base + b, v, ck, sp, g) if chk else (v, ec.with_spare(v, ck, sp, self.sl)))
                self.rbad[b] = chk and self.last_bad != 0
            else:
                out.append(self.zero if chk else (self.zero, None))
                if k == 'up':
                    err = D['DIR']
                elif err is None:
                    err = D['TIM']
        if err:
            raise Diag(err)
        return out

    def read_planes16(self, s, t):
        row = srow(s)
        g = self.lay.read_gap(self.unit, self.slice, row)
        if g is None:
            raise Diag(D['DIR'])
        if sid(s) % 16:
            raise Diag(D['OPR_ALIGN'])
        return self._read_n(row, sid(s), 16, g, t)

    def read_many(self, ops):
        """Read several stream operands, each (stream, width, cycle), all before the first error is reported: the model
        evaluates every read of an instruction (its ECC events too) and then reports the first diagnostic."""
        res, err = [], None
        for (s_, w_, t_) in ops:
            try:
                res.append(self.read_planes(s_, w_, t_))
            except Diag as x:
                res.append(None)
                err = err or x
        if err:
            raise err
        return res

    def poison_mask(self):
        """The superlanes whose results are poisoned (sec. 15): the uncorrectable operand superlanes, every
        superlane for a unit whose output depends on every input superlane."""
        return ((1 << self.sl) - 1 if self.edep and self.eb else self.eb) | self.einh | self.elink

    def link_flit_mask(self, f):
        """The superlanes flit f of a link transfer covers (microarchitecture.md sec. 17): a header (0) or tail flit every
        superlane, a payload flit its 8 lanes, so superlane (f - 1) // 2."""
        n = 1 + (self.vl + 7) // 8
        if f == 0 or f == n - 1:
            return (1 << self.sl) - 1
        return sum(1 << g for g in {byte // 16 for byte in range(5 + 8 * (f - 1), 13 + 8 * (f - 1))})

    def write_planes(self, s, planes, w, c, fwd=None):
        """fwd: the check bits of plane 0 forwarded as they are (the MEM Read), for the active tiles."""
        row = srow(s)
        g = self.lay.write_gap(self.unit, self.slice, row)
        if g is None:
            raise Diag(D['DIR'])
        if not self.aligned(s, w):
            raise Diag(D['OPR_ALIGN'])
        lst = self.pending.setdefault(c, [])
        mem = self.q <= 87
        for b in range(w):        # newest first, the order commits are applied and traced
            v = self.gate(planes[b])
            if fwd is not None and b == 0:
                ent = [fwd[x] if self.tile_on(x, mem) else (0, 0) for x in range(self.sl)]
                ck = ec.norm(v, tuple(ent), self.sl)
            else:
                ck = ec.poisoned(v, None, self.poison_mask(), self.sl)
            lst.insert(0, (row, sid(s) + b, g, v, self.q, b, ck))

    def mem_report(self, prio, tiles, data, mp, c):
        """The error report of a Gather (prio 3) or Scatter (prio 2) for the tiles in `tiles` (superlane -> the raw 13-bit address): S[0]
        or S[1], S[4..16] the address, S[17..22] the map stream (row bit W = 1, then the 5-bit id), S[23..28] the data stream, written in
        the commit cycle at the slice's write gap on the row of the data stream."""
        row = srow(data)
        g = self.lay.write_gap(self.unit, self.slice, row)
        if g is None:
            return
        sp = [0] * 30
        xs = 0
        for x, a in tiles.items():
            xs |= 1 << x
            sp[0 if prio == 3 else 1] |= 1 << x
            for i in range(13):
                if (a >> i) & 1:
                    sp[4 + i] |= 1 << x
            if srow(mp) == 'W':
                sp[17] |= 1 << x
            if row == 'W':
                sp[23] |= 1 << x
            for i in range(5):
                if (sid(mp) >> i) & 1:
                    sp[18 + i] |= 1 << x
                if (sid(data) >> i) & 1:
                    sp[24 + i] |= 1 << x
        if self.srf.report(row, g, c, prio, xs, sp, self.sl):
            self.lint(self.q, self.rc(), LINT['RAS_EVICT'])

    def bank_report(self, q, c, a, b):
        """The bank-conflict report (sec. 15.5) of the pair a, b = (name, fields, address offset, stream offset) dispatched in cycle c:
        S[2] of superlane 0, S[4..16] the write address and S[17..29] the read address (zero for an indirect reference), at the
        write gap of the slice in the commit cycle (dispatch + d_func) on the row of the read's data stream. An IfetchM pair
        reports nothing in the spare bits."""
        if self.dry:
            return
        rd, wr = (a, b) if a[0] in ('Read', 'Gather') else (b, a)
        if rd[0] not in ('Read', 'Gather'):
            return
        s = rd[1]['dst']
        e = sid(s) + rd[3]
        if not 0 <= e <= 31:
            return
        row = srow(s)
        unit, sl, _, _ = queue_unit(q)
        g = self.lay.write_gap(unit, sl, row)
        if g is None:
            return
        wa = (wr[1]['addr'] + wr[2]) & 0x1FFF if wr[0] == 'Write' else 0
        ra = (rd[1]['addr'] + rd[2]) & 0x1FFF if rd[0] == 'Read' else 0
        sp = [0] * 30
        sp[2] = 1
        for i in range(13):
            sp[4 + i] = (wa >> i) & 1
            sp[17 + i] = (ra >> i) & 1
        if self.srf.report(row, g, c + self.lat(rd[0])[0], 1, 1, sp, self.sl):
            self.lint(q, c, LINT['RAS_EVICT'])

    def sram_write(self, c, sl, a, v, ck=None):
        self.sram_revision[sl] += 1
        self.sram[(sl, a)] = v
        if ck is None:
            self.sck.pop((sl, a), None)
        else:
            self.sck[(sl, a)] = ck
        self.events.append(('sram', c, sl, a, bytes(v)))

    def sram_read(self, sl, a):
        return self.sram.get((sl, a), self.zero)

    def sram_ent(self, sl, a):
        """The explicit check bits of every superlane of the stored word."""
        return ec.full(self.sram_read(sl, a), self.sck.get((sl, a)), self.sl)

    def ecc_check_sl(self, t, sl, a, g, d16, c):
        """A Gather or Scatter tile's check of superlane g of the stored word at (sl, a): (the data after the
        single-bit correction, whether the superlane is uncorrectable)."""
        kind, d, pos = ec.decode(d16, c)
        if kind == 'corr':
            self.ecc_event(t, False, 'sram %d:%d' % (sl, a), g, pos)
            return d, False
        if kind == 'unc':
            self.eb |= 1 << g
            self.ecc_event(t, True, 'sram %d:%d' % (sl, a), g, pos)
            return d16, True
        return d16, False

    # ------------------------------------------------------------ execute
    def execute(self, name, f):
        """Run one reference; raises Diag on a diagnostic."""
        e = self.isa.by_name[name]
        getattr(self, 'x_' + e.slice)(name, f)

    # CU (MA 3)
    def x_CU(self, name, f):
        q = self.q
        if name == 'Ifetch':
            first, = self.read_planes(f['stream'], 1, self.rc())
            if len(self.fills.get(q, [])) >= self.staging:
                raise Diag(D['IQ_STAGING'])
            self.stream_reads[q] = (self.rc() + 1, f['stream'], first, self.eb)
        elif name == 'IfetchM':                              # words a, a+1 of the slice
            if q > 87:
                raise Diag(D['QUE_UNIT'])
            e = f['addr'] + 1
            if e >= self.segment[q] or e >= self.words:
                raise Diag(D['OPR_RANGE'])
            now = self.rc()
            if now < self.local_busy.get(q, 0):
                raise Diag(D['QUE_BUSY'])
            if len(self.fills.get(q, [])) >= self.staging:
                raise Diag(D['IQ_STAGING'])
            cks = self.sram_ent(self.slice, f['addr']) if self.itext is not None else None
            self.local_reads[q] = (now + 1, self.slice, e, self.sram_read(self.slice, f['addr'])[:], cks)
            self.local_busy[q] = now + 2
        elif name == 'SetVL':
            self.qvl[q] = max(1, min(f['n'], self.sl))
        elif name == 'Config' and q <= 87:
            v = f['v']
            self.split[q] = v + 1 if v < self.sl else self.sl

    # MEM (MA 4)
    def eff(self, a):
        e = a + self.offset
        return e if 0 <= e < self.segment[self.slice] and e < self.words else None

    def eff_stream(self, s):
        e = sid(s) + self.soffset
        if not 0 <= e <= 31:
            raise Diag(D['OPR_RANGE'])
        return (s & 32) | e

    def x_MEM(self, name, f):
        rc, cc = self.rc(), self.cc()
        s = self.slice
        if name in LOOPABLE:
            self.thread = f['thread']
        if name in ('Read', 'Write'):
            a = self.eff(f['addr'])
            if a is None:
                raise Diag(D['OPR_RANGE'])
            if name == 'Read':
                st = self.eff_stream(f['dst'])
                # the Read does not check: it forwards the stored check bits with the data (sec. 15)
                self.write_planes(st, [self.sram_read(s, a)], 1, cc, fwd=self.sram_ent(s, a))
            else:
                st = self.eff_stream(f['src'])
                # the Write does not check either: it stores the stream's data bits with the stream's check bits
                p, pck = self.read_plane_raw(st, rc)
                w = bytearray(self.sram_read(s, a))
                ent = self.sram_ent(s, a)
                pent = ec.full(p, pck, self.sl)
                for g in range(self.sl):
                    if self.tile_on(g, True):
                        w[16 * g:16 * g + 16] = p[16 * g:16 * g + 16]
                        ent[g] = pent[g]
                self.sram_write(rc, s, a, bytes(w), ec.norm(bytes(w), tuple(ent), self.sl))
        elif name == 'Gather':
            try:
                d = self.eff_stream(f['dst'])
            except Diag:                                      # the map is read all the same (its ECC events), the OPR-001 wins
                try:
                    self.read_planes(f['map'], 1, rc)
                except Diag:
                    pass
                raise
            m = self.read_planes(f['map'], 1, rc)[0]          # the map is an operand like any other: checked
            map_bad = self.eb
            k = f['k']
            out = bytearray(self.vl)
            bad = False
            bounds = {}
            for g in range(self.sl):
                if not self.tile_on(g, True) or (map_bad >> g) & 1:
                    continue                # a tile whose map superlane is uncorrectable makes no access
                a = self.eff((((m[16 * g + 1] << 8) | m[16 * g]) & self.addr_mask))
                if a is None:
                    bad = True          # this tile drives zero, the others act, one OPR-001
                    bounds[g] = (((m[16 * g + 1] << 8) | m[16 * g]) & self.addr_mask)
                else:
                    w = self.sram_read(s, a)
                    d2, _ = self.ecc_check_sl(rc, s, a, g, w[16 * g:16 * g + 16], self.sram_ent(s, a)[g][0])
                    out[16 * g + k] = d2[k]
            try:
                self.write_planes(d, [bytes(out)], 1, cc)
            except Diag:
                if not bad:
                    raise
            if bad:
                self.mem_report(3, bounds, d, f['map'], cc)
                self.eafter = True
                raise Diag(D['OPR_RANGE'])
        elif name == 'Scatter':
            st = self.eff_stream(f['src'])
            src = self.read_planes(st, 1, rc)[0]
            data_bad = self.eb
            self.eb = 0
            m = self.read_planes(f['map'], 1, rc)[0]          # checked: the uncorrectable map superlanes
            map_bad = self.eb
            self.eb = data_bad | map_bad
            k = f['k']
            tiles = [g for g in range(self.sl) if self.tile_on(g, True)]
            addrs = {g: self.eff((((m[16 * g + 1] << 8) | m[16 * g]) & self.addr_mask)) for g in tiles if not (map_bad >> g) & 1}
            for g in tiles:             # a bad tile writes nothing, the good tiles write
                a = addrs.get(g)
                if a is None:
                    continue
                w = bytearray(self.sram_read(s, a))
                ent = self.sram_ent(s, a)
                # the tile checks the old superlane, replaces its byte, and regenerates the superlane's check
                # bits; an uncorrectable old superlane or data byte is stored poisoned
                d2, unc = self.ecc_check_sl(rc, s, a, g, bytes(w[16 * g:16 * g + 16]), ent[g][0])
                w[16 * g:16 * g + 16] = d2
                w[16 * g + k] = src[16 * g + k]
                poisoned = unc or (self.eb >> g) & 1
                nv = bytes(w)
                ent[g] = (ec.encode(nv[16 * g:16 * g + 16]) ^ (ec.POISON_XOR if poisoned else 0), 0)
                self.sram_write(rc, s, a, nv, ec.norm(nv, tuple(ent), self.sl))
            if None in addrs.values():
                self.mem_report(2, {g: (((m[16 * g + 1] << 8) | m[16 * g]) & self.addr_mask) for g in tiles if g in addrs and addrs[g] is None}, st, f['map'], cc)
                self.eafter = True
                raise Diag(D['OPR_RANGE'])
        elif name == 'SetSegment':
            self.segment[s] = min(f['n'], self.words)
        # Countdown/Step/StreamStep/Iterations/PowerConfig/DeepSleep: no effect when executed

    # VXM (MA 5)
    def x_VXM(self, name, f):
        rc, cc = self.rc(), self.cc()
        t = f['type']
        srf = self.srf

        def read_planes(s, w, t_, chain=0):
            """MA 2/5: an operand without its chain bit follows
            the flow rule; a chained operand reads only the write made AT the VXM gap in cycle
            t - 1 (none: TIM-001)."""
            if not chain:
                return self.read_planes(s, w, t_)
            row = srow(s)
            g = self.lay.read_gap(self.unit, self.slice, row)
            if g is None:
                raise Diag(D['DIR'])
            if not self.aligned(s, w):
                raise Diag(D['OPR_ALIGN'])
            out, err = [], None
            for b in range(w):          # every plane found is checked; the first missing one is TIM-001 at the end
                v = srf.w.get((row, sid(s) + b), {}).get((g, g - t_ + 1))
                if v is None:
                    err = err or Diag(D['TIM'])
                    out.append(self.zero)
                else:
                    out.append(self.ecc_check_value(t_, row, sid(s) + b, v, srf.ck[(row, sid(s) + b)].get((g, g - t_ + 1))))
            if err:
                raise err
            return out
        def read_all(ops, t_):
            """Every operand is read (and its ECC checked) before the first error is reported (as the model)."""
            res, err = [], None
            for (s_, w_, ch) in ops:
                if s_ is None:
                    res.append(None)
                    continue
                try:
                    res.append(read_planes(s_, w_, t_, ch))
                except Diag as x:
                    res.append(None)
                    err = err or x
            if err:
                raise err
            return res
        if name == 'Reduce':
            w = nx.WIDTH[t]
            xa = self._lanes(read_planes(f['lhs'], w, rc, f.get('lhs_chain', 0)), w)
            kop = {0: 0, 1: 9, 2: 10}[f['mode']]
            out = [0] * self.vl
            for g in range(self.sl):
                acc = xa[16 * g]
                for j in range(1, 16):
                    acc = nx.k_binary(kop, t, acc, xa[16 * g + j])
                out[16 * g] = acc
            self.write_planes(f['dst'], self._planes(out), w, cc)
            return
        ws = wd = nx.WIDTH[t]
        b = m = None
        bc = mc = 0
        if name in VXM_BIN:
            op = VXM_BIN.index(name)
            b, bc = f['rhs'], f.get('rhs_chain', 0)
            fn = lambda x, y, z: nx.k_binary(op, t, x, y)
        elif name in VXM_UN:
            op = VXM_UN[name]
            fn = lambda x, y, z: nx.k_unary(op, t, x)
        elif name == 'Mask':
            b, bc = f['mask'], f.get('mask_chain', 0)
            fn = lambda x, y, z: x if nx.nonzero(t, y) else 0
        elif name == 'Select':
            b, m = f['rhs'], f['mask']
            bc, mc = f.get('rhs_chain', 0), f.get('mask_chain', 0)
            fn = lambda x, y, z: x if nx.nonzero(t, z) else y
        elif name == 'Cmp':
            b, bc = f['rhs'], f.get('rhs_chain', 0)
            mode = f['mode']
            fn = lambda x, y, z: nx.k_cmp(mode, t, x, y)
        elif name in VXM_CONV:
            op = VXM_CONV[name]
            st, dt = nx.CONV[op]
            ws, wd = nx.WIDTH[st], nx.WIDTH[dt]
            fn = lambda x, y, z: nx.k_convert(op, x)
        elif name == 'Convert':
            dt = nx.TYPE_NAME[f['dtype']]
            wd = nx.WIDTH[dt]
            fn = lambda x, y, z: nx.k_convert_any(t, dt, x)
        else:
            raise AssertionError(name)
        if f.get('sg4', 0):
            # MA 5: every operand is the SG4 group at its stream; the
            # ALU applies the operation to each ws-byte part of the 32-bit group operand.
            pa, pb, pm = read_all([(f['lhs'], 4, f.get('lhs_chain', 0)), (b, 4, bc), (m, 4, mc)], rc)
            parts = 4 // ws
            out = [0] * self.vl
            for i in range(parts):
                sl = slice(i * ws, (i + 1) * ws)
                xa = self._lanes(pa[sl], ws)
                xb = self._lanes(pb[sl], ws) if pb else [0] * self.vl
                xm = self._lanes(pm[sl], ws) if pm else [0] * self.vl
                mask = (1 << (8 * ws)) - 1
                for l in range(self.vl):
                    out[l] |= (fn(xa[l], xb[l], xm[l]) & mask) << (8 * i * ws)
            self.write_planes(f['dst'], self._planes(out), 4, cc)
            return
        pa, pb, pm = read_all([(f['lhs'], ws, f.get('lhs_chain', 0)), (b, ws, bc), (m, ws, mc)], rc)
        xa = self._lanes(pa, ws)
        xb = self._lanes(pb, ws) if pb else [0] * self.vl
        xm = self._lanes(pm, ws) if pm else [0] * self.vl
        cache = {}
        out = []
        for l in range(self.vl):
            k = (xa[l], xb[l], xm[l])
            r = cache.get(k)
            if r is None:
                r = cache[k] = fn(*k)
            out.append(r)
        self.write_planes(f['dst'], self._planes(out), wd, cc)

    def _lanes(self, planes, w):
        out = [0] * self.vl
        for b in range(w):
            p = planes[b]
            sh = 8 * b
            for l in range(self.vl):
                out[l] |= p[l] << sh
        return out

    def _planes(self, words):
        return [bytes((x >> (8 * b)) & 0xff for x in words) for b in range(4)]

    # MXM (MA 6). Instructions read the plane as it was at the start of the cycle (self.mxm)
    # and stage the parts they change; tick applies them (unit-state flush, MA 1).
    def stage_mxm(self, part, value):
        self.mxm_staged.append((self.q, self.plane, part, value))

    def cap(self, t):
        """The stream values a weight buffer or matrix of type t holds: mxm.rows, or two per f16 or bf16 row."""
        return self.wvals if t in PAIR else self.rows

    def plane_blocked(self, t):
        pl = self.plane
        if self.planes != 2:                         # a pair needs both planes of the hemisphere: with one, no plane runs f16 or bf16
            return t in PAIR
        if pl not in (1, 3):
            return False
        even = self.mxm[pl - 1]
        holds = (even.rows > 0 and even.wtype in PAIR) or (even.buf_rows > 0 and even.btype in PAIR)
        return t in PAIR or holds

    def mxm_check(self, p, t, f, x, faults):
        """The sum check (microarchitecture.md sec. 18) of an ABC whose exact sums carry the injected flips. Exact sums:
        an integer product sum as it is, a floating-point one in units of 2^-48 (f16) or 2^-32 (fp8), finite products only.
        Independent column-major nominal row products are compared against the final faulted primary row products.
        Classify changed installed rows, not injected fault entries: zero clean, one corrected, two or more uncorrectable."""
        vl, nr = self.vl, p.rows
        if t == 'bfp8':
            fm, sc = nx.E4M3, 17
        elif t in PAIR:
            fm, sc = (nx.F16, 24) if t == 'f16' else (nx.BF16, 133)
        elif t == 'fp8':
            fm, sc = (nx.E4M3 if f['fmt'] == 0 else nx.E5M2), 16
        else:
            fm = None

        def elem(lo, hi, k, sgn):
            if fm is None:
                return ((lo[k] - 256 if lo[k] > 127 else lo[k]) if sgn else lo[k])
            c, n, m = nx.fclass(fm, (lo[k] | (hi[k] << 8)) if t in ('f16', 'bf16') else lo[k])
            if c in ('nan', 'inf'):
                return None
            if t == 'bfp8':
                return int((-m if n else m) * Fraction(2) ** (sc + nx.block_exp(hi, k)))
            return int((-m if n else m) * (1 << sc))
        zero = bytes(vl)
        row = lambda r: (p.weights[2 * r], p.weights[2 * r + 1]) if t in PAIR else (p.weights[r], zero)
        ws, xs = p.wtype == 'i8', t == 'i8'
        xv = [elem(x[0], x[1] if t in PAIR else zero, k, xs) for k in range(vl)]
        ps = [0] * 320
        for r in range(nr):
            lo, hi = row(r)
            for k in range(vl):
                w = elem(lo, hi, k, ws)
                if w is not None and xv[k] is not None:
                    ps[r] += w * xv[k]
        expected = [0] * nr
        for k in range(vl):
            if xv[k] is None:
                continue
            for r in range(nr):
                lo, hi = row(r)
                w = elem(lo, hi, k, ws)
                if w is not None:
                    expected[r] += w * xv[k]
        for lane, bit in faults:
            if lane < nr:
                u = ps[lane] % (1 << 96)
                u ^= 1 << bit
                ps[lane] = u - (1 << 96) if u >= (1 << 95) else u
        wrong = [r for r in range(nr) if ps[r] != expected[r]]
        if not wrong:
            return 0, 0
        if len(wrong) == 1:
            return 1, wrong[0]
        return 2, 0

    def x_MXM(self, name, f):
        p = self.mxm[self.plane]
        if name == 'LW':
            t = MXM_WT[f['type']]
            if self.plane_blocked(t):
                raise Diag(D['RES'])
            rows = self.read_planes16(f['src'] << 4, self.rc())
            if p.buf_rows > 0 and p.btype != t:
                raise Diag(D['RES'])
            n = p.buf_rows
            if n + 16 > self.cap(t):
                raise Diag(D['RES_FULL'])
            v = p.copy()
            for i in range(16):
                v.buf[n + i] = rows[i]
                v.buf_bad[n + i] = self.rbad[i]
            v.btype = t
            v.buf_rows = n + 16
            v.ready_lw = self.cc()
            self.stage_mxm('BUF', v)
        elif name == 'IW':
            t = MXM_WT[f['type']]
            if self.plane_blocked(t) or p.buf_rows == 0 or self.issue < p.ready_lw or p.btype != t:
                raise Diag(D['RES'])
            v = p.copy()
            v.wtype = t
            v.weights = list(p.buf)
            v.w_bad = list(p.buf_bad)
            v.buf_bad = [False] * self.wvals
            v.rows = p.buf_rows // 2 if t in PAIR else p.buf_rows
            v.iw_open = False
            v.ready_iw = self.cc()
            v.buf_rows = 0
            self.stage_mxm('W', v)
            self.stage_mxm('BUF', v)
        elif name == 'IWS':
            t = MXM_WT[f['type']]
            if self.plane_blocked(t):
                raise Diag(D['RES'])
            rows = self.read_planes16(f['src'] << 4, self.rc())
            # an ABC dispatched after the last stream install starts a new matrix (MA 6)
            open_ = p.iw_open and not (p.abc and p.abc[0][0] > p.iws_cycle)
            n = (2 * min(p.rows, self.f16_rows) if p.wtype in PAIR else p.rows) if open_ else 0
            if open_ and p.wtype != t:
                raise Diag(D['RES'])
            if n + 16 > self.cap(t):
                raise Diag(D['RES_FULL'])
            v = p.copy()
            for i in range(16):
                v.weights[n + i] = rows[i]
                v.w_bad[n + i] = self.rbad[i]
            v.wtype = t
            v.rows = (n + 16) // 2 if t in PAIR else n + 16
            v.iw_open = True
            v.iws_cycle = self.issue
            v.ready_iw = self.cc()
            self.stage_mxm('W', v)
        elif name in ('ABC', 'ABC8'):
            t = 'fp8' if name == 'ABC8' else MXM_WT[f['type']]
            w = 2 if t in PAIR else 1
            # an f16 ABC holds the plane's MACCs timing.mxm_abc.occupancy_f16 cycles (two in ref and mini; MA 6)
            busy = any(e[2] in PAIR and e[0] < self.issue < e[0] + self.occ_f16 for e in p.abc)
            # integer weights and activations may differ in sign (MA 6)
            types_ok = p.wtype == t if (p.wtype in PAIR or t in PAIR or 'fp8' in (p.wtype, t)) else True
            if self.plane_blocked(t) or p.rows == 0 or self.issue < p.ready_iw or not types_ok or busy:
                raise Diag(D['RES'])
            x = self.read_planes(f['src'], w, self.rc())
            self.abcs.append((self.rc(), self.plane))
            mine = [(m[3], m[4]) for m in self.mfaults if m[1] == self.rc() and m[2] == self.plane]
            self.mfaults = [m for m in self.mfaults if not (m[1] == self.rc() and m[2] == self.plane)]
            if mine:
                verdict, lane = self.mxm_check(p, t, f, x, mine)
                if verdict == 2:
                    self.ras_irq(self.rc(), 5)
                    raise Diag(D['ECC_MXM'])
                if verdict == 1:
                    self.events.append(('ecc_mxm', self.rc(), self.q, self.plane, lane))
                    self.ras_irq(self.rc(), 4)
            out = [0] * max(320, self.vl)           # a product has mxm.rows (at most 320) lanes; the vector may be wider
            wide = [nx.WIDE_ZERO] * max(320, self.vl)     # the exact products, for the wide accumulator (only when declared)
            vl = self.vl
            if t in ('f16', 'bf16'):
                xs = [x[0][k] | (x[1][k] << 8) for k in range(vl)]
                dot = nx.dot_f16 if t == 'f16' else nx.dot_bf16
                for r in range(p.rows):
                    lo, hi = p.weights[2 * r], p.weights[2 * r + 1]
                    xl = [lo[k] | (hi[k] << 8) for k in range(vl)]
                    out[r] = dot(xl, xs)
                    if self.wide and t == 'f16':
                        wide[r] = nx.wide_of(nx.exact_f16(xl, xs))
            elif t == 'bfp8':
                for r in range(p.rows):
                    d = nx.exact_bfp8(p.weights[2 * r][:vl], p.weights[2 * r + 1][:vl], x[0][:vl], x[1][:vl])
                    out[r] = nx.round_exact(d)
                    if self.wide:
                        wide[r] = nx.wide_of(d)
            elif t == 'fp8':
                ff = nx.E4M3 if f['fmt'] == 0 else nx.E5M2      # mode bit 13: the format of weights and activations
                for r in range(p.rows):
                    out[r] = nx.dot_fp8(ff, p.weights[r][:vl], x[0][:vl])
                    if self.wide:
                        wide[r] = nx.wide_of(nx.exact_fp8(ff, p.weights[r][:vl], x[0][:vl]))
            else:
                sgn = lambda b: b - 256 if b > 127 else b
                sx = sgn if t == 'i8' else (lambda b: b)
                sw = sgn if p.wtype == 'i8' else (lambda b: b)
                xs = [sx(b) for b in x[0]]
                for r in range(p.rows):
                    wr = p.weights[r]
                    out[r] = sum(sw(wr[k]) * xs[k] for k in range(vl)) & 0xffffffff
            # an uncorrectable activation poisons every superlane of the product, an uncorrectable weight row r the
            # superlane r // 16 (an f16 row is buffer rows 2r and 2r + 1)
            poison = (1 << self.sl) - 1 if self.eb else 0
            if not poison:
                for r in range(p.rows):
                    rb = (p.w_bad[2 * r] or p.w_bad[2 * r + 1]) if t in PAIR else p.w_bad[r]
                    if rb and r // 16 < self.sl:
                        poison |= 1 << (r // 16)
            v = p.copy()
            v.abc = [(self.issue, self.cc(), t, out, poison, self.q, wide)] + list(p.abc)
            self.stage_mxm('ABC', v)
        elif name == 'ACC':
            a = f['acc']
            at = {0: 'x128', 2: 'i32', 3: 'f32'}[f['type']]
            if self.plane_blocked(None):
                raise Diag(D['RES'])
            if a >= self.acc_depth:                  # the decoder does not admit it
                raise Diag(D['OPR_RANGE'])
            e = next((e for e in p.abc if e[1] <= self.issue), None)
            if e is None:
                raise Diag(D['RES'])
            # two FP8 results of one dispatch cycle (mxm.fp8_rate 2): an even accumulator takes the lower queue's
            i = p.abc.index(e)
            if i + 1 < len(p.abc) and p.abc[i + 1][0] == e[0] and p.abc[i + 1][1] <= self.issue:
                f2 = p.abc[i + 1]
                lo, hi = (e, f2) if e[5] < f2[5] else (f2, e)
                e = lo if a % 2 == 0 else hi
            if (at == 'i32') != (e[2] in ('i8', 'u8')) or (at == 'x128' and e[2] in ('i8', 'u8', 'bf16')):
                raise Diag(D['OPR_RANGE'])
            prod = e[3]
            if at == 'x128':
                # the wide accumulator keeps the exact sum; the stream gets its read-out, rounded once to f32
                wnew = list(e[6]) if f['mode'] == 0 else [nx.wide_add(y, x) for x, y in zip(e[6], p.acc128[a])]
                out = [nx.wide_read(w) for w in wnew]
            elif f['mode'] == 0:
                out = list(prod)
            elif at == 'i32':
                out = [(x + y) & 0xffffffff for x, y in zip(prod, p.acc[a])]
            else:
                out = [nx.fp_add(nx.F32, y, x) for x, y in zip(prod, p.acc[a])]
            pm = e[4] if f['mode'] == 0 else e[4] | p.acc_poison[a]       # inherited, so reported by the reader only
            self.einh = pm
            self.write_planes(f['dst'], self._planes((out + [0] * self.vl)[:self.vl]), 4, self.cc())      # a product has mxm.rows lanes: zero above them
            v = p.copy()
            if at == 'x128':
                v.acc128[a] = wnew
            else:
                v.acc[a] = out
            v.acc_poison[a] = pm
            self.stage_mxm('ACC%d' % a, v)

    # SXM (MA 7)
    def x_SXM(self, name, f):
        rc, cc = self.rc(), self.cc()
        vl = self.vl
        maps = self.maps[self.sx]

        def emit(d, v):
            self.write_planes(d, [bytes(v)], 1, cc)
        if name == 'Shift':
            x = self.read_planes(f['src'], 1, rc)[0]
            self.edep = True           # an output superlane takes lanes of its neighbours (sec. 15)
            k = f['amount']
            v = bytearray(vl)
            for l in range(vl):
                j = l - k if f['dir'] == 0 else l + k
                if 0 <= j < vl:
                    v[l] = x[j]
            emit(f['dst'], v)
        elif name in ('Permute', 'Distribute'):
            m = maps[f['map']]
            x = self.read_planes(f['src'], 1, rc)[0]
            v = bytearray(vl)
            if name == 'Permute':
                # the operand read comes first (its diagnostic wins); then the map's readiness
                # and its bijective flag, both recorded by its LoadMap
                if rc < self.map_ready[self.sx][f['map']]:
                    raise Diag(D['RES'])
                if not self.map_bij[self.sx][f['map']]:
                    raise Diag(D['SXM_PERM'])
                self.edep = True       # a lane may come from any superlane (sec. 15)
                for l in range(vl):
                    if m[l] < vl:
                        v[l] = x[m[l]]
            else:
                for g in range(self.sl):
                    for j in range(16):
                        e = m[16 * g + j]
                        if e < 16:
                            v[16 * g + j] = x[16 * g + e]
            emit(f['dst'], v)
        elif name == 'Rotate':
            n = f['n']
            d = f['dst']
            if sid(d) + n * n > 32:
                raise Diag(D['OPR_ALIGN'])
            x = self.read_planes(f['src'], 1, rc)[0]
            err = None
            for t in range(n * n):
                tr, tc = t // n, t % n
                v = bytearray(vl)
                for g in range(self.sl):
                    for r in range(n):
                        for c in range(n):
                            v[16 * g + r * n + c] = x[16 * g + ((r + tr) % n) * n + (c + tc) % n]
                try:
                    emit((d & 32) | ((sid(d) + t) & 31), v)
                except Diag as e:
                    err = e
            if err:
                raise err
        elif name == 'Transpose':
            rows = self.read_planes16(f['src'] << 4, rc)
            d = f['dst'] << 4
            err = None
            for c in range(16):
                v = bytearray(vl)
                for g in range(self.sl):
                    for r in range(16):
                        v[16 * g + r] = rows[r][16 * g + c]
                try:
                    emit((d & 32) | ((sid(d) + c) & 31), v)
                except Diag as e:
                    err = e
            if err:
                raise err
        elif name in ('LoadMap', 'LoadMapWide'):
            if name == 'LoadMapWide':
                (lo,), (hi,) = self.read_many([(f['src'], 1, rc), (f['src2'], 1, rc)])
            else:
                lo, hi = self.read_planes(f['src'], 1, rc)[0], self.zero
            m = [0] * max(320, vl)
            for l in range(vl):
                m[l] = (hi[l] % 4) * 256 + lo[l]
            seen = [e for e in m[:vl] if e < vl]          # bijective: the entries below vlanes are distinct
            self.sxm_staged.append((self.q, self.sx, f['map'], m, len(seen) == len(set(seen)), rc + self.map_delay))
        elif name == 'SelectS':
            (a,), (b,), (c3,), (s,) = self.read_many([(f['a'], 1, rc), (f['b'], 1, rc), (f['c'], 1, rc), (f['sel'], 1, rc)])
            # A lane whose sel is above 2 is zero, the others are selected, one OPR-001
            bad = any(x > 2 for x in s)
            try:
                emit(f['dst'], [(a, b, c3)[s[l]][l] if s[l] <= 2 else 0 for l in range(vl)])
            except Diag:
                if not bad:
                    raise
            if bad:
                raise Diag(D['OPR_RANGE'])

    # C2C (MA 10): one chip, every link loops back to itself (link l TX -> link l RX) with
    # the nominal latency; the peer of every link is chip 0. link >= c2c.links: OPR-001.
    # A link accepts one vector per cpv cycles (LNK-001); its receive FIFO holds rx_depth
    # vectors and a vector that arrives at a full one is dropped (LNK-002, link_arrivals).
    def x_C2C(self, name, f):
        if name in ('Send', 'Transmit', 'Receive') and (f['link'] >= self.n_links or f['chip_id'] != 0):
            raise Diag(D['OPR_RANGE'])
        if name in ('Send', 'Transmit'):
            last = self.link_acc[f['link']]
            if last is not None and last + self.cpv > self.rc():
                raise Diag(D['LNK_BUSY'])                    # checked before the stream is read
            if name == 'Send':
                v = self.read_planes(f['stream'], 1, self.rc())[0]
                poison = self.eb                              # the uncorrectable superlanes go out with the vector
            else:
                v = bytes([1]) + bytes(self.vl - 1)
                poison = 0
            self.link_acc[f['link']] = self.rc()
            fc = sorted(x[3] for x in self.lfaults if x[1] == self.rc() and x[2] == f['link'] and not x[4])     # the F link lines of this vector
            fu = sorted(x[3] for x in self.lfaults if x[1] == self.rc() and x[2] == f['link'] and x[4])
            self.sends.append((self.rc(), f['link']))
            self.links[f['link']].append([self.rc() + self.link_lat, v, LV_PENDING, 0, fc, fu, False, poison])
        elif name == 'Receive':
            fifo = self.links[f['link']]
            self.link_decide(f['link'], self.rc())           # the arrivals up to this cycle are decided first
            head = next((e for e in fifo if e[2] in (LV_QUEUED, LV_PENDING)), None)
            if head is None or head[2] != LV_QUEUED or head[0] > self.rc():
                raise Diag(D['TIM'])
            if not self.aligned(f['stream'], 1):
                raise Diag(D['OPR_ALIGN'])
            if self.lay.write_gap(self.unit, self.slice, srow(f['stream'])) is None:
                raise Diag(D['DIR'])                         # no effect, FIFO kept
            head[2], head[3] = LV_TAKEN, self.rc()
            self.einh = head[7]                              # arrives poisoned in the same superlanes
            for fl in head[5]:                               # the superlanes of the uncorrectable flits are poisoned (sec. 17)
                self.elink |= self.link_flit_mask(fl)
            self.write_planes(f['stream'], [head[1]], 1, self.cc())
        elif name == 'PciReceive':
            raise Diag(D['TIM'])                             # no host script: nothing ever arrived
        elif name == 'PciSend':
            if self.pci_acc is not None and self.pci_acc + self.host_cpv > self.rc():
                raise Diag(D['HIO_BUSY'])                    # checked before the stream is read
            v = self.read_planes(f['stream'], 1, self.rc())[0]
            self.pci_acc = self.rc()
            self.events.append(('dma_out', self.rc(), bytes(v)))
        elif name == 'HostWait':
            if f['i'] >= self.n_doorbells:
                raise Diag(D['OPR_RANGE'])
        elif name == 'HostNotify':
            self.events.append(('irq', self.rc(), f['n']))   # IRQ_MASK resets to all-ones: the vector is delivered
        # Deskew, RuntimeDeskew: only their queue occupancy (dispatch)

    def poll_arrival(self, l):
        """The arrival cycle of the oldest vector of link l that is neither taken nor dropped (None: none)."""
        return next((e[0] for e in self.links[l] if e[2] in (LV_QUEUED, LV_PENDING)), None)

    def poll_ready(self, l, c):
        """A vector has arrived on link l by cycle c (what Receive tests)."""
        self.link_decide(l, c)
        head = next((e for e in self.links[l] if e[2] in (LV_QUEUED, LV_PENDING)), None)
        return head is not None and head[2] == LV_QUEUED and head[0] <= c

    def link_decide(self, l, upto):
        """Decide the arrivals of link l up to cycle upto, in arrival order. A vector that arrives at
        cycle a finds the earlier ones that are queued, or taken at a cycle >= a (a Receive in the arrival
        cycle takes after it); at rx_depth of them it is dropped and the chip reports LNK-002 at a."""
        fifo = self.links[l]
        for i, e in enumerate(fifo):
            if e[0] > upto:
                break
            if e[2] != LV_PENDING:
                continue
            held = sum(1 for o in fifo[:i] if o[2] == LV_QUEUED or (o[2] == LV_TAKEN and o[3] >= e[0]))
            if held >= self.rx_depth:
                e[2] = LV_DROPPED
                self.overruns.append((e[0], l))
            else:
                e[2] = LV_QUEUED

    def link_arrivals(self, upto):
        """The harness before tick(upto): decide every arrival up to upto, then report the overruns due,
        ascending (cycle, link), queue 65535."""
        for l in range(len(self.links)):
            self.link_decide(l, upto)
        due = [(a, l, 1) for a, l in self.overruns if a <= upto]
        self.overruns = [o for o in self.overruns if o[0] > upto]
        fec = []                                             # the link FEC events of the vectors that arrived (sec. 17), before the overrun of their link and cycle
        for l in range(len(self.links)):
            for e in self.links[l]:
                if e[0] <= upto and not e[6] and (e[4] or e[5]):
                    e[6] = True
                    fec.append((e[0], l, 0, e))
        for a, l, k in sorted(due + [x[:3] for x in fec]):
            if k == 1:
                self.events.append(('diag', a, CTRL_ID, D['LNK_OVERRUN']))
            else:
                e = next(x[3] for x in fec if x[0] == a and x[1] == l)
                for fl in sorted(e[4] + e[5]):
                    self.events.append(('ecc_link', a, l, fl in e[5], fl))
                    self.ras_irq(a, 7 if fl in e[5] else 6)

    # ------------------------------------------------------------ the cycle driver
    def run_element(self, q, name, f, c0, k, off, soff):
        """Issue one reference at c0 + k; it executes d_skew cycles later."""
        self.elems.append((q, name, f, c0, k, off, soff))
        if self.dry:
            return
        e = (q, name, f, c0, k, off, soff)
        ds = self.lat(name, f.get('type'))[1]
        if ds == 0:
            self.execute_element(e)
        else:
            self.ring.setdefault(c0 + k + ds, []).append(e)

    def execute_element(self, e):
        q, name, f, c0, k, off, soff = e
        self.issue, self.elem, self.offset, self.soffset = c0, k, off, soff
        self.df, self.ds = self.lat(name, f.get('type'))
        self.q, self.thread = q, 0
        self.unit, self.slice, self.plane, self.sx = queue_unit(q)
        self.eb, self.edep, self.eafter, self.einh, self.elink = 0, False, False, 0, 0       # ECC: bad superlanes of this instruction (sec. 15); einh: inherited poison; elink: of a received vector (sec. 17)
        try:
            self.execute(name, f)
            if self.elink:
                self.report(q, self.rc(), D['LNK_UNC'])
            if self.eb:
                self.report(q, self.rc(), D['ECC_UNC'])
        except Diag as x:
            self.report(q, self.rc(), x.code)
            if self.eafter and self.eb:
                self.report(q, self.rc(), D['ECC_UNC'])

    def execute_due(self, c):
        for e in self.ring.pop(c, []):
            self.execute_element(e)

    # ------------------------------------------------------------ the front end (MA 3)
    def fetch(self, q, t, payload, cks=None):
        """A fetch whose payload is complete at t (an IfetchM: its second read cycle; an Ifetch: the cycle of its second vector): its bytes enter the IQ of q from t + fetch_latency, at the
        fetch rate (the payload queues behind the earlier fetches of q)."""
        if len(self.fills.get(q, [])) >= self.staging:      # no flow control: nowhere to wait (IQ-003, the fetch has no effect)
            raise Diag(D['IQ_STAGING'])
        us = [int.from_bytes(payload[k:k + 4], 'little') for k in range(0, len(payload), 4)]
        base = self.iprod.get(q, 0)
        self.iprod[q] = base + len(us)
        if self.itext is not None:                           # the stored text words of the payload: data and explicit check bits
            for k in range(len(payload) // 16):
                c_, sp = cks[k] if cks is not None else (ec.encode(payload[16 * k:16 * k + 16]), 0)
                self.itext.setdefault(q, {})[(base + 4 * k) // 4] = [payload[16 * k:16 * k + 16], c_, sp]
        self.fills.setdefault(q, []).append([t + self.fetch_lat, us, base, 0])

    # ECC on instruction text (microarchitecture.md sec. 16.1): self.itext[q][word] = [the 16 data bytes as stored, the 9 check bits, the spare bit]
    # of every text word of the queue (None: no fault was injected in the program, so every word is valid); self.itag[q][i] is the
    # production index of unit i of the IQ (None: no word), four consecutive indices are a word.
    def vunit(self, q, i):
        """Unit i of the IQ of q as the decoder sees it: its lane of the word after the correction of a single error."""
        u = self.iq[q][i]
        if self.itext is None:
            return u
        t = self.itag[q][i]
        ent = self.itext.get(q, {}).get(t // 4) if t is not None else None
        if ent is None:
            return u
        kind, d, _ = ec.decode(ent[0], ent[1])
        data = d if kind == 'corr' else ent[0]
        return int.from_bytes(data[4 * (t % 4):4 * (t % 4) + 4], 'little')

    def word_of(self, q, i):
        t = self.itag[q][i] if self.itext is not None else None
        return None if t is None else t // 4

    def decode_at(self, q, pos, w):
        """The instruction of w units at pos of the IQ of q: OTS-E-ECC-002 when a word it lies in is uncorrectable."""
        if self.itext is not None:
            for k in range(w):
                wd = self.word_of(q, pos + k)
                ent = self.itext.get(q, {}).get(wd) if wd is not None else None
                if ent is not None and ec.decode(ent[0], ent[1])[0] == 'unc':
                    return False, D['ECC_TXT']
        return self.decode_text([self.vunit(q, pos + k) for k in range(w)])

    def text_check(self, q, c, units):
        """The head parcel (units of them) leaves the IQ at cycle c: every instruction of it checks the one or two words it lies in."""
        if self.itext is None:
            return
        pos = 0
        while pos < units:
            w = 2 if (self.vunit(q, pos) >> 28) & 7 in (1, 2, 4) else 1
            a = self.word_of(q, pos)
            b = self.word_of(q, pos + 1) if w == 2 else None
            self.word_check(q, c, a)
            if b != a:
                self.word_check(q, c, b)
            pos += w

    def word_check(self, q, c, wd):
        ent = self.itext.get(q, {}).get(wd) if wd is not None else None
        if ent is None:
            return
        kind, d, pos = ec.decode(ent[0], ent[1])
        if kind == 'ok':
            return
        self.ecc_event(c, kind == 'unc', 'text word %d' % wd, None, pos, q)
        if kind == 'corr':                                   # written back: the word is corrected once
            ent[0], ent[1], ent[2] = d, ec.encode(d), 0

    def flip_text(self, q, wd, bit):
        """F iq: False when no unit of the word is in the IQ."""
        if not any(self.itag[q][i] is not None and self.itag[q][i] // 4 == wd for i in range(len(self.iq.get(q, [])))):
            return False
        ent = self.itext[q][wd]
        if bit < 128:
            b = bytearray(ent[0])
            b[bit // 8] ^= 1 << (bit % 8)
            ent[0] = bytes(b)
        elif bit < 137:
            ent[1] ^= 1 << (bit - 128)
        else:
            ent[2] ^= 1
        return True

    def head(self, q):
        """The head parcel of queue q, or None when it is not all in the IQ: (units taken, sum of
        the NOP prefix, items = [(ok, instruction or code)]). NOPs with bundle bit 0 before the
        first instruction are the timing prefix; the parcel ends with the first instruction whose
        bundle bit is 0."""
        us = self.iq.get(q, [])
        pos, nops, items = 0, 0, []
        while pos < len(us):
            u0 = self.vunit(q, pos)
            w = 2 if (u0 >> 28) & 7 in (1, 2, 4) else 1
            if pos + w > len(us):
                return None
            bundle = u0 >> 31
            ok, i = self.decode_at(q, pos, w)
            pos += w
            if ok and i[0] == 'NOP' and not bundle and not items and i[1]['n'] > 0:
                nops += i[1]['n']
                continue
            items.append((ok, i))
            if not bundle:
                return pos, nops, items
        return None

    def earliest(self, q, p):
        return self.last_disp.get(q, -1) + 1 + p[1]

    def free(self, q, t):
        """MA 3: queue q is free at t (nothing holds a head parcel): the underflow rule."""
        if q == CTRL:
            return True
        return (not self.parked(q, t) and not self.host_parked(q, t) and t >= self.busy.get(q, 0) and q not in self.repeats and q not in self.polls
                and all(self.agt_state(q, p) == 'IDLE' and not self.path_busy(q, p, t) and not self.path_tail(q, p, t)
                        for p in (0, 1)))

    def enqueue(self, q, c):
        """The fetched text that is due enters the IQ at the start of cycle c, at the fetch rate;
        padding is dropped; a unit that does not fit is lost (IQ-002, once per queue and cycle);
        an IQ that runs dry (the head parcel completes after its earliest cycle, the queue free in
        the cycle before) reports lint IQ-001 when the parcel dispatches."""
        fl = self.fills.get(q)
        if not fl or fl[0][0] > c:
            return
        before = bool(self.iq.get(q)) and self.head(q) is not None
        budget, pushed, lost = self.fetch_rate, False, False
        while budget and fl and fl[0][0] <= c:
            u = fl[0][1].pop(0)
            tag = fl[0][2] + fl[0][3]
            fl[0][3] += 1
            budget -= 1
            if not fl[0][1]:
                fl.pop(0)
            if u != PADDING:
                if len(self.iq.setdefault(q, [])) < self.iq_cap:
                    self.iq[q].append(u)
                    self.itag.setdefault(q, []).append(tag)
                    self.active[q] = c
                    pushed = True
                else:
                    lost = True
        if lost:
            self.report(q, c, D['IQ_OVERFLOW'])
        if pushed and not before and c > 0:
            p = self.head(q)
            if p is not None and c > self.earliest(q, p) and self.free(q, c - 1):
                self.dry_iq.add(q)

    def occupancy(self, name, f, c):
        if name == 'NOP':
            return f['n']
        if name == 'Deskew':
            return self.hac - c % self.hac
        if name == 'RuntimeDeskew':
            return max(f['t'], self.rtd_floor)
        return 1

    def agt_configure(self, a, field, v):
        fp = AGT_FIELD[field]
        if fp <= a['pos']:
            if a['level'] >= 3:
                return False
            a['level'] += 1
            a['pos'] = -1
        l = a['levels'][a['level']]
        if field == 'Countdown':
            l['cd'] = max(v, 1)
        elif field == 'Step':
            l['step'] = v
        elif field == 'StreamStep':
            l['sstep'] = v
        else:
            l['iters'] = max(v, 1)
        a['pos'] = fp
        return True

    @staticmethod
    def agt_advance(levels, n):
        """MA 4 AGT: counters, delay, step and stream step of the next reference over the
        first n levels (a reference takes a cycle and every countdown wait is unconditional);
        None when no level can advance. Works on a copy."""
        ls = [dict(l) for l in levels]
        wait = 1
        for l in ls[:n]:
            wait += l['cd']
            if l['idx'] + 1 < l['iters']:
                l['idx'] += 1
                return ls, wait, l['step'], l['sstep']
            l['idx'] = 0
        return None

    @staticmethod
    def agt_nlevels(a):
        return 0 if a['pos'] < 0 else a['level'] + 1

    def agt_refresh(self, q, p):
        a = self.agts[(q, p)]
        if a['state'] == 'RUN':
            st = self.agt_advance(a['levels'], self.agt_nlevels(a))
            if st is None:
                # the trailing Wait(countdown) of every level holds the path
                n = self.agt_nlevels(a)
                self.ptail[(q, p)] = a['last'] + 1 + sum(l['cd'] for l in a['levels'][:n])
                del self.agts[(q, p)]
            else:
                a['next'] = a['last'] + st[1]

    def agt_due(self, q, p, c, parked):
        """The AGT reference of path p due at c (agt_advance result), also while the
        configuration is still arriving; None when none is due."""
        a = self.agts.get((q, p))
        if not a or parked:
            return None
        st = self.agt_advance(a['levels'], self.agt_nlevels(a))
        return st if st and a['last'] + st[1] <= c else None

    # MA 3/4 dispatch paths: a MEM queue has an R path (Read, Gather) and a W path
    # (Write, Scatter), each with its own busy state and AGT; self.agts is keyed (q, path).
    R_PATH = {'Read': 0, 'Gather': 0, 'Write': 1, 'Scatter': 1}

    def reserve_local_fetch(self, q, ins, c):
        """An IfetchM issued at cycle c, as a new instruction or as a Repeat iteration: it holds the R path through the
        cycle of its second read (c + 1) and takes the bank of word a + 1 in that cycle (postlude/tick.sail reserve_local_fetch)."""
        self.pbusy[(q, 0)] = c + 2
        if ins[0] == 'IfetchM':
            addr = ins[1]['addr'] + 1
            if q <= 87 and addr < self.segment[q] and addr < self.words:
                self.local_banks[q] = (c + 1, addr)

    def repeat_issue(self, q, prev, c):
        """A Repeat iteration of prev = (name, fields) at cycle c (postlude/tick.sail repeat_issue)."""
        self.run_element(q, prev[0], prev[1], c, 0, 0, 0)
        if prev[0] == 'IfetchM':
            self.reserve_local_fetch(q, prev, c)

    def path_busy(self, q, p, c):
        return c < self.pbusy.get((q, p), 0)

    def path_tail(self, q, p, c):
        return c < self.ptail.get((q, p), 0)

    def agt_state(self, q, p):
        a = self.agts.get((q, p))
        return a['state'] if a else 'IDLE'

    def hold(self, q, c, name, path=None, strict=False):
        """MA 3 "Waiting": is this instruction of the head parcel held at c? Held while its
        queue is parked by a Sync (a queue's own Sync and Notify are the barrier), busy (c below
        the busy-until: Deskew, an instruction issued this cycle) or running a Repeat; an
        instruction on dispatch path p while that path is busy, in the tail of a finished loop,
        running a loop, or holding an open configuration that the instruction does not continue;
        any other instruction while either path has an AGT state (a loop running or its
        configuration open: an AGT reference due this cycle holds the queue for it)
        or is busy or in a tail (Ifetch on a MEM queue needs no path). strict: the parcel breaks
        the slot rule, so every instruction needs the whole queue, Ifetch included."""
        if name not in ('Sync', 'Notify') and self.parked(q, c):
            return True
        if self.host_parked(q, c):
            return True
        if c < self.busy.get(q, 0) or q in self.repeats or q in self.polls:
            return True
        if path is not None:
            st = self.agt_state(q, path)
            return self.path_busy(q, path, c) or self.path_tail(q, path, c) or st == 'RUN' or (st == 'CONFIG' and name not in AGT_FIELD)
        if not strict and name in FETCH and q <= 87:
            return False
        return any(self.path_busy(q, p, c) or self.path_tail(q, p, c) or self.agt_state(q, p) != 'IDLE' for p in (0, 1))

    def parcel_held(self, q, c, r, items):
        """A parcel dispatches together: it waits when any of its instructions is held."""
        if r['n'] == 0:
            return False
        if not r['ok']:
            return any(self.hold(q, c, i[0] if ok else '', None, True) for ok, i in items)
        if r['single']:
            return self.hold(q, c, r['single'][0])
        if r['ifetch'] and self.hold(q, c, r['ifetch'][0], 0 if r['ifetch'][0] == 'IfetchM' else None):
            return True
        if r['cfg'] and self.hold(q, c, r['cfg'][0], self.cfg_path(q)):
            return True
        return any(r['refs'][k] and self.hold(q, c, r['refs'][k][0], k // 2) for k in range(4))

    def check(self, q, c, name, path=None):
        """The unit check of a dispatch (the instruction is not held: MA 3 "Dispatch")."""
        if not self.isa.issued_by(name, queue_unit(q)[0]):
            return D['QUE_UNIT']
        return None

    def dispatch_one(self, q, c, name, f):
        """A record that needs the whole queue."""
        err = self.check(q, c, name)
        if err:
            self.report(q, c, err)
            return
        if name == 'Repeat':
            prev = self.last.get(q)
            if prev is None:
                self.report(q, c, D['RES'])
                return
            self.busy[q] = c + 1
            # iteration k at p + k (d + 1) (p: the repeated instruction's issue
            # cycle; d cycles between iterations), the first
            # not before the Repeat's own cycle
            gap = f['d'] + 1
            first = max(self.last_at.get(q, 0) + gap, c)
            if first == c:
                self.repeat_issue(q, prev, c)
                self.last_at[q] = c
                if f['n'] > 1:
                    self.repeats[q] = dict(i=prev, left=f['n'] - 1, gap=gap, next=c + gap)
            else:
                self.repeats[q] = dict(i=prev, left=f['n'], gap=gap, next=first)
            return
        if name in ('Sync', 'Notify', 'Deskew', 'RuntimeDeskew', 'HostWait', 'DeskewRecv'):    # not repeatable
            self.last.pop(q, None)
        elif name != 'NOP' and name not in AGT_FIELD:
            self.last[q] = (name, f)
            self.last_at[q] = c
        self.busy[q] = c + self.occupancy(name, f, c)
        if name == 'Sync':
            self.barrier_sync(q, c)
        elif name == 'Notify':
            self.barrier_notify(c)
        elif name == 'HostWait' and f['i'] < self.n_doorbells:
            self.hw_since[q], self.hw_release[q] = c, None
        elif name == 'DeskewRecv':
            # : the diagnostics of Receive that do not depend on an arrival are the dispatch's (nothing held); else the
            # queue polls at each epoch boundary after this cycle
            if f['link'] >= self.n_links or f['chip_id'] != 0:
                self.report(q, c, D['OPR_RANGE'])
            elif self.lay.write_gap(queue_unit(q)[0], queue_unit(q)[1], srow(f['stream'])) is None:
                self.report(q, c, D['DIR'])
            else:
                self.polls[q] = dict(chip_id=f['chip_id'], link=f['link'], stream=f['stream'], since=c)
        self.run_element(q, name, f, c, 0, 0, 0)

    def classify(self, q, items):
        """The instructions of a parcel: one Ifetch, per thread one R and one W reference, one
        configuration record; or one instruction that needs the whole queue, alone."""
        lst = items
        r = dict(ok=True, n=len(lst), single=None, ifetch=None, refs=[None] * 4, cfg=None, err=None)

        def place(i):
            name, f = i
            if name in AGT_FIELD:
                slot = 'cfg'
            elif name in self.R_PATH:
                k = 2 * self.R_PATH[name] + f['thread']
                if r['refs'][k] is not None:
                    return False
                r['refs'][k] = i
                return True
            elif name in FETCH:
                slot = 'ifetch'
            else:
                return False
            if r[slot] is not None:
                return False
            r[slot] = i
            return True

        if not lst:
            return r
        if len(lst) == 1:
            ok, i = lst[0]
            if not ok:
                r['ok'] = False
                r['err'] = i
            elif q > 87 or not place(i):
                r['single'] = i
            return r
        if q > 87:
            r['ok'] = False
            return r
        for ok, i in lst:
            if not ok or not place(i):
                r['ok'] = False
        # IfetchM reads through the R port (all tiles): no R-type reference
        if r['ifetch'] and r['ifetch'][0] == 'IfetchM' and (r['refs'][0] or r['refs'][1]):
            r['ok'] = False
        return r

    def addr_bank(self, e):
        """The bank an address uses; NO_BANK outside the slice."""
        return (e >> self.bank_bit) & 1 if 0 <= e < self.words else NO_BANK

    def tag(self, name, f, off):
        """(thread, bank) of a reference; bank None (any bank) for an indirect one."""
        bank = self.addr_bank(f['addr'] + off) if name in ('Read', 'Write') else None
        return (f['thread'], bank)

    @staticmethod
    def meet(a, b):
        """Two bank uses meet: never with NO_BANK, always with None (any bank)."""
        if a == NO_BANK or b == NO_BANK:
            return False
        return a is None or b is None or a == b

    @classmethod
    def clash(cls, a, b):
        return a[0] == b[0] and cls.meet(a[1], b[1])

    def loop_tag(self, q, p, st):
        if st is None:
            return None
        a = self.agts[(q, p)]
        return self.tag(a['name'], a['f'], a['offset'] + st[2])

    def cfg_path(self, q):
        r, w = self.agt_state(q, 0) == 'CONFIG', self.agt_state(q, 1) == 'CONFIG'
        if r and w:
            return 1 if self.records.get((q, 1), 0) < self.records.get((q, 0), 0) else 0
        return 1 if w else 0

    def dispatch_cfg(self, q, c, p, name, f):
        err = self.check(q, c, name, p)
        if err:
            self.report(q, c, err)
            return
        a = self.agts.get((q, p))
        if not a or a['state'] != 'CONFIG':
            self.report(q, c, D['OPR_RANGE'])
            return
        self.records[(q, p)] = self.records.get((q, p), 0) + 1
        if not self.agt_configure(a, name, f[AGT_VALUE[name]]):
            self.report(q, c, D['OPR_RANGE'])
            del self.agts[(q, p)]

    def loop_clash_ref(self, q, p, st):
        a = self.agts[(q, p)]
        return (a['name'], a['f'], a['offset'] + st[2], a['soffset'] + st[3])

    def run_loop(self, q, c, p, st, blocked, partner=None):
        if st is None:
            return
        a = self.agts[(q, p)]
        if blocked and partner is not None:
            self.bank_report(q, c, self.loop_clash_ref(q, p, st), partner)
        a['levels'] = st[0]
        a['offset'] += st[2]
        a['soffset'] += st[3]
        a['last'] = c
        if blocked:
            self.report(q, c, D['MEM_BANK'])
        else:
            self.run_element(q, a['name'], a['f'], a['t0'], max(c - a['t0'], 0), a['offset'], a['soffset'])
        self.pbusy[(q, p)] = c + 1
        self.agt_refresh(q, p)

    def dispatch_slots(self, q, c, r, parked):
        issued, last = 0, None
        prev = self.local_banks.get(q)
        fm = {self.addr_bank(prev[1])} if prev and prev[0] == c else None
        fm_new = False
        if r['ifetch']:
            name, f = r['ifetch']
            err = self.check(q, c, name, 0 if name == 'IfetchM' else None)
            if err:
                self.report(q, c, err)
            elif name == 'IfetchM':
                a = f['addr']
                fm = {self.addr_bank(a)} - {NO_BANK}
                fm_new = True
            else:
                self.run_element(q, name, f, c, 0, 0, 0)
                issued, last = issued + 1, r['ifetch']
        if r['cfg']:
            self.dispatch_cfg(q, c, self.cfg_path(q), *r['cfg'])
        # due references are worked out after the configuration record
        due = [self.agt_due(q, p, c, parked) for p in (0, 1)]
        loop = [self.loop_tag(q, p, due[p]) for p in (0, 1)]
        lref = [self.loop_clash_ref(q, p, due[p]) if due[p] is not None else None for p in (0, 1)]    # before any loop runs
        refs = r['refs']
        go = [False] * 4
        for k in range(4):
            if refs[k]:
                err = self.check(q, c, refs[k][0], k // 2)
                if err:
                    self.report(q, c, err)
                else:
                    go[k] = True
        bank = [False] * 4
        for k in range(4):
            if refs[k] and go[k]:
                p, t = k // 2, k % 2
                o = 2 * (1 - p) + t
                me = self.tag(*refs[k], 0)
                hit = refs[o] is not None and go[o] and self.clash(me, self.tag(*refs[o], 0))
                hit = hit or (loop[1 - p] is not None and self.clash(me, loop[1 - p]))
                bank[k] = hit
        fm_clash = fm_loop = False
        if fm is not None:
            for t in (0, 1):
                j = refs[2 + t]
                if j and go[2 + t]:
                    b = self.tag(*j, 0)[1]
                    if any(self.meet(b, x) for x in fm):
                        bank[2 + t] = True
                        fm_clash = True
            if loop[1] is not None and any(self.meet(loop[1][1], x) for x in fm):
                fm_loop = fm_clash = True
            if fm_new:
                if fm_clash:
                    self.report(q, c, D['MEM_BANK'])
                else:
                    self.run_element(q, r['ifetch'][0], r['ifetch'][1], c, 0, 0, 0)
                    self.reserve_local_fetch(q, r['ifetch'], c)
                    issued, last = issued + 1, r['ifetch']
        lblock = [False, False]
        lpart = [None, None]
        for p in (0, 1):
            if loop[p] is not None:
                hit = loop[1 - p] is not None and self.clash(loop[p], loop[1 - p])
                if hit:
                    lpart[p] = lref[1 - p]
                for t in (0, 1):
                    j = refs[2 * (1 - p) + t]
                    if j and go[2 * (1 - p) + t] and self.clash(loop[p], self.tag(*j, 0)):
                        hit = True
                        lpart[p] = (j[0], j[1], 0, 0)
                lblock[p] = hit or (p == 1 and fm_loop)
        for p in (0, 1):
            self.run_loop(q, c, p, due[p], lblock[p], lpart[p])
        for k in range(4):
            if refs[k] and go[k]:
                if bank[k]:
                    self.report(q, c, D['MEM_BANK'])
                    po = 1 if k < 2 else 0
                    o = 2 * po + k % 2
                    partner = None
                    if loop[po] is not None and self.clash(self.tag(*refs[k], 0), loop[po]):
                        partner = lref[po]
                    if refs[o] and go[o] and self.clash(self.tag(*refs[k], 0), self.tag(*refs[o], 0)):
                        partner = (refs[o][0], refs[o][1], 0, 0)
                    if partner is not None:
                        self.bank_report(q, c, (refs[k][0], refs[k][1], 0, 0), partner)
                    continue
                p = k // 2
                name, f = refs[k]
                self.pbusy[(q, p)] = c + 1
                issued, last = issued + 1, refs[k]
                alone = refs[2 * p + 1 - k % 2] is None
                self.run_element(q, name, f, c, 0, 0, 0)
                # the loop opens at dispatch, whatever reference 0's execution reports (MA 8)
                if name in LOOPABLE and alone:
                    # defaults when not received: countdown 0, step 0, numIter 2
                    self.agts[(q, p)] = dict(state='CONFIG', name=name, f=f, t0=c, level=0, pos=-1, offset=0, soffset=0,
                                             last=c, next=0, levels=[dict(cd=0, step=0, sstep=0, iters=2, idx=0) for _ in range(4)])
                    self.records[(q, p)] = 0
        if issued > 1:
            self.last.pop(q, None)
        elif issued == 1:
            self.last[q] = last
            self.last_at[q] = c

    def dispatch(self, q, c, r, parked):
        n = r['n']
        if n and ((q <= 87 and not self.lay.present(q)) or (104 <= q <= 115 and not mxm_queue_present(q))):
            # The queue of an absent MEM slice has no unit: every instruction of the
            # parcel (also one that does not decode, or breaks the slot rule) is QUE-002.
            for _ in range(n):
                self.report(q, c, D['QUE_UNIT'])
            return
        slots = n != 0 and r['ok'] and r['single'] is None
        if not slots:
            due = [self.agt_due(q, p, c, parked) for p in (0, 1)]
            both = due[0] is not None and due[1] is not None and self.clash(self.loop_tag(q, 0, due[0]), self.loop_tag(q, 1, due[1]))
            lref = [self.loop_clash_ref(q, p, due[p]) if due[p] is not None else None for p in (0, 1)]
            for p in (0, 1):
                self.run_loop(q, c, p, due[p], both, lref[1 - p])
        if n == 0:
            return
        if not r['ok']:
            if n == 1:
                self.report(q, c, r['err'])
            else:
                for _ in range(n):
                    self.report(q, c, D['QUE_BUSY'])
        elif r['single']:
            self.dispatch_one(q, c, *r['single'])
        else:
            self.dispatch_slots(q, c, r, parked)

    def control(self, c):
        """The control channel (MA 3): Sync, Notify and Config; its parcels are never held."""
        p = self.head(CTRL) if self.iq.get(CTRL) else None
        if p is None or self.earliest(CTRL, p) > c:
            return
        self.text_check(CTRL, c, p[0])
        del self.iq[CTRL][:p[0]]
        del self.itag[CTRL][:p[0]]
        self.last_disp[CTRL] = c
        self.active[CTRL] = c
        items = p[2]
        if len(items) != 1:
            for _ in items:
                self.report(CTRL, c, D['QUE_BUSY'])
            return
        ok, i = items[0]
        if not ok:
            self.report(CTRL, c, i)
        elif i[0] == 'Sync':
            self.barrier_sync(CTRL, c)
        elif i[0] == 'Notify':
            self.barrier_notify(c)
        elif i[0] != 'Config':
            self.report(CTRL, c, D['QUE_UNIT'])

    def commit(self, c):
        """MA 2: collisions report SRF-002 on every colliding writer's queue (once per
        collided stream, ascending queue order); the lowest (queue, plane) writer wins. The
        writes of one (queue, plane) are one writer: they merge.
        A group of exactly two writes, one from an SXM hemisphere's SEL queue and one
        from a TSP queue of the same hemisphere, merges: no SRF-002, the register takes the
        OR of both values (HC34 p.31)."""
        lst = self.pending.pop(c, [])
        # The writes of one (queue, plane) are one writer (a MEM queue's two liquid
        # threads, each zero on the other's tiles): merge them by OR into the first of them.
        own = {}
        for e in lst:
            k = (e[:3], e[4], e[5])
            if k in own:
                mv = bytes(x | y for x, y in zip(own[k][0], e[3]))
                own[k] = (mv, ec.merge(own[k][0], own[k][1], e[3], e[6], mv, self.sl))
            else:
                own[k] = (e[3], e[6])
        seen, uniq = set(), []
        for e in lst:
            k = (e[:3], e[4], e[5])
            if k not in seen:
                seen.add(k)
                uniq.append(e[:3] + (own[k][0],) + e[4:6] + (own[k][1],))
        lst = uniq
        groups = {}
        for e in lst:
            groups.setdefault(e[:3], []).append(e)
        hits = {}
        for e in lst:
            row, s, g, v, q, b, ck = e
            grp = groups[(row, s, g)]
            others = [o for o in grp if (o[4], o[5]) != (q, b)]
            first = all((q, b) < (o[4], o[5]) for o in others)
            if len(others) == 1 and sxm_or_pair(q, others[0][4]):
                if first:
                    merged = bytes(x | y for x, y in zip(v, others[0][3]))
                    self.srf.write(row, s, g, c, merged, ec.merge(v, ck, others[0][3], others[0][6], merged, self.sl))
                    self.events.append(('srf', c, row, s, g, merged))
                continue
            if others:
                hits[q] = hits.get(q, 0) + 1
            if first:
                self.srf.write(row, s, g, c, v, ck)
                self.events.append(('srf', c, row, s, g, v))
        for q in sorted(hits):
            for _ in range(hits[q]):
                self.report(q, c, D['SRF'])

    def flush(self, c):
        """MA 1 step 6: apply the staged MXM plane parts and SXM maps; clashes are RES-003."""
        bad = set()
        for (q, pl, part, v) in self.mxm_staged:
            others = [o for o in self.mxm_staged if o[0] != q and o[1] == pl and o[2] == part]
            if part == 'ABC':
                # mxm.fp8_rate 2: a plane accepts two FP8 ABCs in one cycle; a third, or an ABC of another type, clashes
                two_fp8 = self.fp8_rate == 2 and len(others) == 1 and v.abc[0][2] == 'fp8' and others[0][3].abc[0][2] == 'fp8'
                if others and not two_fp8:
                    bad.add(q)
            elif others:
                bad.add(q)
        for (q, sx, mid, m, _bij, _rdy) in self.sxm_staged:
            if any(o[0] != q and o[1] == sx and o[2] == mid for o in self.sxm_staged):
                bad.add(q)
        for q in sorted(bad):
            self.report(q, c, D['RES_STATE'])
        for (q, pl, part, v) in reversed(self.mxm_staged):
            if q in bad:
                continue
            p = self.mxm[pl]
            if part.startswith('ACC'):
                a = int(part[3:])
                p.acc[a] = v.acc[a]
                p.acc_poison[a] = v.acc_poison[a]
                p.acc128[a] = v.acc128[a]
            elif part == 'ABC':
                # a second entry of the same dispatch cycle from another queue goes in front of the list as it stands
                if p.abc and v.abc and v.abc[0][0] == p.abc[0][0] and v.abc[0][5] != p.abc[0][5]:
                    p.abc = [v.abc[0]] + list(p.abc)
                else:
                    p.abc = v.abc
            else:
                for fld in PART_FIELDS[part]:
                    setattr(p, fld, getattr(v, fld))
        for (q, sx, mid, m, bij, rdy) in reversed(self.sxm_staged):
            if q not in bad:
                self.maps[sx][mid] = m
                self.map_bij[sx][mid] = bij
                self.map_ready[sx][mid] = rdy
        if bad:
            df, ds = self.lat('ACC')
            k = c - ds + df
            if k in self.pending:
                self.pending[k] = [e for e in self.pending[k] if not (e[4] in bad and queue_unit(e[4])[0].startswith('MXM'))]
        self.mxm_staged, self.sxm_staged = [], []

    def apply_faults(self, c):
        """ECC fault injection (microarchitecture.md sec. 12 and 15): the flips of cycle c, in input order, before the tick."""
        while self.faults and self.faults[0][1] <= c:
            f = self.faults.pop(0)
            if f[0] == 'sram':
                _, _, sl, a, g, bit = f
                if g < self.sl:
                    v, ck = ec.flip(self.sram_read(sl, a), self.sck.get((sl, a)), g, bit, self.sl)
                    self.sram[(sl, a)] = v
                    self.sram_revision[sl] += 1
                    if ck is None:
                        self.sck.pop((sl, a), None)
                    else:
                        self.sck[(sl, a)] = ck
            elif f[0] == 'iq':
                _, _, q, wd, bit = f
                if not self.flip_text(q, wd, bit):
                    raise RuntimeError('oracle: F iq of a word that is not in the IQ')
            else:
                _, _, row, st, gap, g, bit = f
                self.srf.flip(row, st, gap, c, g, bit, self.sl)

    def patrol_until(self, upto):
        period = self.cfg.get('scrub', {}).get('period', 0)
        if not period or self.dry:
            return
        while self.patrol_next <= upto:
            c = self.patrol_next
            self.patrol_next += period
            dispatch = c - self.lat('Read')[1]
            for s in range(88):
                if not self.lay.present(s):
                    continue
                if dispatch >= 0 and (self.path_busy(s, 0, dispatch) or self.path_busy(s, 1, dispatch)
                                      or (s in self.last and self.last_at.get(s) == dispatch)):
                    continue
                pending = self.patrol_pending.pop(s, None)
                if pending is not None:
                    address, value, checks, revision = pending
                    if revision == self.sram_revision[s]:
                        self.sram[(s, address)] = value
                        self.sram_revision[s] += 1
                        if checks is None:
                            self.sck.pop((s, address), None)
                        else:
                            self.sck[(s, address)] = checks
                    continue
                a = self.patrol_pointer[s]
                self.patrol_pointer[s] = (a + 1) % self.words
                checks = self.sck.get((s, a))
                if checks is None:
                    continue
                checks = list(checks)
                value = bytearray(self.sram_read(s, a))
                fixed = False
                for g in range(self.sl):
                    kind, corrected, _ = ec.decode(value[16*g:16*g+16], checks[g][0])
                    if kind == 'ok':
                        continue
                    self.events.append(('scrub', c, s, a, g, kind == 'unc'))
                    self.ras_irq(c, 1 if kind == 'unc' else 0)
                    if kind == 'corr':
                        value[16*g:16*g+16] = corrected
                        checks[g] = (ec.encode(corrected), checks[g][1])
                        fixed = True
                if fixed:
                    value = bytes(value)
                    self.patrol_pending[s] = (a, value, ec.norm(value, checks, self.sl), self.sram_revision[s])

    def tick(self, c):
        self.patrol_until(c - 1)
        self.link_arrivals(c)                                # the harness's LNK-002, before the tick
        self.apply_faults(c)
        for q, (at, ms, addr, first, cks) in list(self.local_reads.items()):
            if at <= c:
                del self.local_reads[q]
                checks = cks + self.sram_ent(ms, addr) if cks is not None else None
                try:
                    self.fetch(q, c, first + self.sram_read(ms, addr), checks)
                except Diag as ex:
                    self.report(q, c, ex.code)
        for q in sorted(self.fills):
            self.enqueue(q, c)
        self.barrier_retire(c)
        self.commit(c)
        for q in sorted(list(self.stream_reads)):
            at, stream, first, first_bad = self.stream_reads[q]
            if at > c:
                continue
            del self.stream_reads[q]
            self.q, self.eb = q, 0
            self.unit, self.slice, self.plane, self.sx = queue_unit(q)
            try:
                second, = self.read_planes(stream, 1, c)
                checks = None
                if self.itext is not None:
                    checks = [(ec.encode(v[16*g:16*g+16]) ^ (ec.POISON_XOR if (mask >> g) & 1 else 0), (mask >> g) & 1)
                              for v, mask in ((first, first_bad), (second, self.eb)) for g in range(self.sl)]
                self.fetch(q, c, first + second, checks)
                if self.eb:
                    self.report(q, c, D['ECC_UNC'])
            except Diag as ex:
                self.report(q, c, ex.code)
        self.execute_due(c)
        self.control(c)
        active = {k[0] for k in self.agts} | set(self.repeats) | set(self.polls) | {q for q, us in self.iq.items() if us and q < CTRL}
        for q in sorted(active):
            hp = self.head(q) if self.iq.get(q) else None
            due = hp is not None and c >= self.earliest(q, hp)
            items = hp[2] if due else []
            r = self.classify(q, items)
            parked = self.parked(q, c)
            pl = self.polls.get(q)
            if pl and c > pl['since'] and c % self.hac == 0 and not parked and self.poll_ready(pl['link'], c):
                del self.polls[q]                            # the poll becomes a Receive dispatched in this cycle
                self.busy[q] = c + 1
                self.run_element(q, 'Receive', dict(chip_id=pl['chip_id'], link=pl['link'], stream=pl['stream']), c, 0, 0, 0)
            rp = self.repeats.get(q)
            if rp and rp['next'] <= c and not parked:
                self.repeat_issue(q, rp['i'], c)
                self.last_at[q] = c
                self.busy[q] = c + 1
                if rp['left'] <= 1:
                    del self.repeats[q]
                else:
                    rp['left'] -= 1
                    rp['next'] = c + rp['gap']
            go = due and not self.parcel_held(q, c, r, items)
            # A configuration ends at the first cycle in which the queue dispatches no configuration
            # instruction; the ended configuration no longer holds the head parcel.
            cfg = go and r['ok'] and r['cfg'] is not None
            ended = False
            for p in (0, 1):
                a = self.agts.get((q, p))
                if a and a['state'] == 'CONFIG' and not cfg:
                    a['state'] = 'RUN'
                    self.agt_refresh(q, p)
                    ended = True
            if ended and due and not go:
                go = not self.parcel_held(q, c, r, items)
            r2 = self.classify(q, [])
            if go:
                r2 = r
                self.text_check(q, c, hp[0])
                del self.iq[q][:hp[0]]
                del self.itag[q][:hp[0]]
                self.last_disp[q] = c
                self.active[q] = c
                if q in self.dry_iq:
                    self.dry_iq.discard(q)
                    self.lint(q, c, LINT['IQ_UNDERFLOW'])
            self.dispatch(q, c, r2, parked)
        self.flush(c)
        self.patrol_until(c)

    def stuck(self, q):
        def one(k):
            return k in self.parked_since and self.release_at.get(k) is None
        return one(q) or one(CTRL) or (q in self.hw_since and self.hw_release.get(q) is None)

    def busy_work(self):
        if self.stream_reads or self.local_reads or self.pending or self.ring or any(self.fills.values()) or self.faults:
            return True
        if any(not self.stuck(q) for q in {k[0] for k in self.agts} | set(self.repeats)):
            return True
        if any(not self.stuck(q) and self.poll_arrival(pl['link']) is not None for q, pl in self.polls.items()):
            return True                                      # a poll with a vector on its way (one without waits for good)
        # a complete head parcel: the control channel's always runs; a data queue's unless parked for good
        # (a Sync or Notify is no wait: a parked queue's own barrier instruction still dispatches,
        # unless the queue's Repeat or AGT state holds it, which a queue parked for good never ends)
        for q in list(self.iq):
            p = self.iq[q] and self.head(q)
            barrier = all(ok and i[0] in ('Sync', 'Notify') for ok, i in p[2]) if p else False
            frozen = q in self.repeats or q in self.polls or any(k[0] == q for k in self.agts)
            if p and (q == CTRL or (not self.stuck(q) and q not in self.polls) or (barrier and not frozen)):
                return True
        return False

    def stranded(self, q):
        """MA 3: the text left in the IQ of q cannot form a parcel: after leading NOPs (a timing prefix with
        nothing to time is not a truncation) it holds half of a two-unit instruction, or instructions up to
        one with the bundle bit set and nothing after it."""
        us = self.iq.get(q, [])
        pos, items = 0, 0
        while pos < len(us):
            u0 = self.vunit(q, pos)
            w = 2 if (u0 >> 28) & 7 in (1, 2, 4) else 1
            if pos + w > len(us):
                return True
            bundle = u0 >> 31
            ok, i = self.decode_at(q, pos, w)
            pos += w
            if ok and i[0] == 'NOP' and not bundle and not items and i[1]['n'] > 0:
                continue
            items += 1
            return True
        return False

    def finish(self):
        """MA 12: after the last cycle, IQ-004 for each queue (control channel last) whose IQ holds text that
        cannot dispatch, unless the queue is parked for good; at the cycle of the queue's last IQ activity."""
        for q in sorted(self.iq):
            if self.iq[q] and not self.stuck(q) and q not in self.polls and self.stranded(q):
                self.report(q, self.active.get(q, 0), D['IQ_STRANDED'])

    def skip(self, t):
        """The next cycle that can matter when nothing is in motion: the earliest dispatch cycle of
        a head parcel (time passes in the NOP prefix and the queues' free cycles)."""
        if self.stream_reads or self.local_reads or self.pending or self.ring or any(self.fills.values()) or self.agts or self.repeats:
            return t
        best = self.faults[0][1] if self.faults else None   # a fault cycle is an event of the run (sec. 12)
        for q, pl in self.polls.items():                     # the epoch boundary at which a poll finds the vector on its way
            a = self.poll_arrival(pl['link'])
            if a is not None and not self.stuck(q):
                b = max(a, t, pl['since'] + 1)
                b += -b % self.hac
                best = b if best is None else min(best, b)
        for q in self.iq:
            p = self.head(q) if self.iq[q] else None
            if p is not None:
                e = self.earliest(q, p)
                best = e if best is None else min(best, e)
        return best if best is not None and best > t else t

    def run(self, images, preload=None, dry=False, faults=()):
        """images: {queue: [32-bit unit, ...]} (queue 0xffff or 144: the control channel), the
        initial IQ contents (padding units are dropped). dry: only record which elements execute
        (self.elems), skip their effects."""
        self.dry = dry
        self.elems = []
        self.iq = {(CTRL if q == CTRL_ID else q): [u for u in us if u != PADDING] for q, us in images.items()}
        self.itag = {q: list(range(len(us))) for q, us in self.iq.items()}
        self.iprod = {q: 4 * ((len(us) + 3) // 4) for q, us in self.iq.items()}
        self.estat = 0                                       # ERR_STATUS (the oracle has no host script: the bits only set)
        self.itext = None
        if faults and not dry:                               # a program with faults: every text word keeps its stored data and explicit check bits
            self.itext = {}
            for q, us in self.iq.items():
                for k in range((len(us) + 3) // 4):
                    d = b''.join(u.to_bytes(4, 'little') for u in us[4 * k:4 * k + 4]).ljust(16, b'\0')
                    self.itext.setdefault(q, {})[k] = [d, ec.encode(d), 0]
        self.fills, self.last_disp, self.dry_iq = {}, {}, set()
        self.stream_reads = {}
        self.local_reads, self.local_banks, self.local_busy = {}, {}, {}
        self.active = {}                                     # queue -> last cycle a unit entered its IQ or it dispatched (default 0)
        self.sram = dict(preload or {})
        self.sram_revision = [0] * 88
        self.patrol_next, self.patrol_pointer, self.patrol_pending = 0, [0] * 88, {}
        self.sck = {}                                        # (slice, addr) -> explicit check bits of a faulted word (oracle/ecc.py)
        self.mfaults = [] if dry else [f for f in faults if f[0] == 'mxm']          # ('mxm', c, plane, lane, bit): a bit of the exact sum of a lane of the ABC of cycle c (sec. 18)
        self.abcs = []                                       # (cycle, plane) of every ABC that read its activation
        self.faults = [] if dry else sorted((f for f in faults if f[0] not in ('link', 'mxm')), key=lambda f: f[1])      # ('sram', c, slice, addr, sl, bit) / ('srf', c, row, stream, gap, sl, bit)
        self.lfaults = [] if dry else [f for f in faults if f[0] == 'link']        # ('link', c, link, flit, uncorrectable): marks a flit of the vector sent in cycle c (sec. 17)
        self.sends = []                                      # (cycle, link) of every vector the chip put on a link
        self.eb, self.edep, self.eafter, self.einh, self.elink = 0, False, False, 0, 0
        self.events = []
        self.srf = SRF()
        self.pending = {}
        self.ring = {}                                       # execute cycle -> elements
        self.busy, self.agts, self.repeats, self.last = {}, {}, {}, {}
        self.polls = {}                                      # : queue -> its DeskewRecv that has not found its vector
        self.pbusy, self.records = {}, {}
        self.parked_since, self.release_at, self.last_notify = {}, {}, None
        self.hw_since, self.hw_release, self.pci_acc = {}, {}, None
        self.qvl, self.split = {}, {}
        self.ptail, self.last_at = {}, {}
        self.segment = [self.words] * 88
        self.mxm = [Plane(self.vl, self.wvals, self.acc_depth) for _ in range(2 * self.planes)]
        self.mxm_staged, self.sxm_staged = [], []
        self.maps = [[list(range(max(320, self.vl))) for _ in range(16)] for _ in range(2)]
        self.map_bij = [[True] * 16 for _ in range(2)]       # reset maps are bijective ...
        self.map_ready = [[0] * 16 for _ in range(2)]        # ... and ready
        self.links = [[] for _ in range(max(16, self.n_links))]   # inbound FIFOs: [arrival, vector, state, taken_at], per link
        self.link_acc = [None] * 16            # the cycle each link last accepted a vector
        self.overruns = []                     # (arrival, link) of the dropped vectors not yet reported
        self.q, self.thread = CTRL, 0
        t = 0
        ticks = 0
        while self.busy_work():
            t = self.skip(t)
            self.tick(t)
            t += 1
            ticks += 1
            if ticks > 3000000:
                raise RuntimeError('oracle: run did not finish')
        self.link_arrivals(1 << 62)                          # the arrivals after the last tick
        self.finish()
        return self.events
