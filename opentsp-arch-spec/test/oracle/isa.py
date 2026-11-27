"""Table-driven decoder and encoder over isa-db/generated/encoding.json (encoding.md).

An instruction is a pair (name, fields): name is the catalogue name, fields maps each
catalogue variable to its integer value (the element type to its name, e.g. 'i16').
"""
import json
import os

from . import numerics as nx

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOGUE = os.path.join(HERE, '..', '..', 'isa-db', 'generated', 'encoding.json')


def _loc(s):
    if '-' in s:
        hi, lo = s.split('-')
        return int(hi), int(lo)
    return int(s), int(s)


class Entry:
    def __init__(self, d):
        self.name = d['name']
        self.slice = d['slice']
        self.units = d['units']
        self.width = d['width']
        self.timing = d['timing']
        m = d['match']
        assert len(m) == self.width, self.name
        self.mask = self.value = 0
        for i, ch in enumerate(m):
            bit = self.width - 1 - i
            if ch in '01':
                self.mask |= 1 << bit
                self.value |= int(ch) << bit
        self.vars = []
        for v in d['variables']:
            hi, lo = _loc(v['location'])
            extra = {k: v[k] for k in v if k not in ('name', 'location', 'kind')}
            if 'hi_location' in extra:
                extra['hi'] = _loc(extra['hi_location'])
            if 'top_location' in extra:
                extra['top'] = _loc(extra['top_location'])
            self.vars.append((v['name'], hi, lo, v['kind'], extra))

    def fields_of(self, w, vlanes, sram_words=8192, acc_depth=2, declared=()):
        """Decode the variables of word w; None when a value constraint fails. `max_expr` is evaluated with the profile variables
        `vlanes`, `sram_words` and `acc_depth` (for a signed field it bounds the magnitude); `declared` are the profile's extensions."""
        f = {}
        for name, hi, lo, kind, ex in self.vars:
            n = hi - lo + 1
            v = (w >> lo) & ((1 << n) - 1)
            for part in ('hi', 'top'):
                if part in ex:
                    h1, l1 = ex[part]
                    v |= ((w >> l1) & ((1 << (h1 - l1 + 1)) - 1)) << n
                    n += h1 - l1 + 1
            for ext, codes in ex.get('extension_codes', {}).items():
                if v in codes and ext not in declared:
                    return None
            if kind == 'eltype_code':
                if v not in ex['codes']:
                    return None
                f[name] = nx.TYPE_NAME[v]
                continue
            if kind == 'acc_index' and v >= acc_depth:   # bit 12 and bits 11..8 above it (joined above); below the accumulator depth
                return None
            if kind == 'simm16':
                v = nx.sval(v, 16)
            if 'not_values' in ex and v in ex['not_values']:
                return None
            if 'values' in ex and v not in ex['values']:
                return None
            if 'codes' in ex and v not in ex['codes']:
                return None
            if 'min' in ex and v < ex['min']:
                return None
            if 'max' in ex and v > ex['max']:
                return None
            if kind == 'lane_count' and v >= vlanes:
                return None
            if 'max_expr' in ex:
                var, _, off = ex['max_expr'].partition('-')
                bound = {'vlanes': vlanes, 'sram_words': sram_words, 'acc_depth': acc_depth}[var.strip()] - (int(off) if off else 0)
                if abs(v) > bound if kind == 'simm16' else v > bound:
                    return None
            f[name] = v
        # boolean element types allow only the bool_values of a mode field
        for name, hi, lo, kind, ex in self.vars:
            if 'bool_values' in ex and nx.is_bool(f.get('type', 'i8')) and f[name] not in ex['bool_values']:
                return None
        return f

    def encode(self, **f):
        w = self.value
        for name, hi, lo, kind, ex in self.vars:
            v = f[name]
            if kind == 'eltype_code':
                v = nx.CODE[v] if isinstance(v, str) else v
            n = hi - lo + 1
            w |= (v & ((1 << n) - 1)) << lo
            for part in ('hi', 'top'):
                if part in ex:
                    h1, l1 = ex[part]
                    w |= ((v >> n) & ((1 << (h1 - l1 + 1)) - 1)) << l1
                    n += h1 - l1 + 1
        return w


class ISA:
    def __init__(self, path=CATALOGUE, vlanes=320, addr_bits=13, acc_depth=2, extensions=()):
        with open(path) as fh:
            d = json.load(fh)
        self.entries = [Entry(e) for e in d['instructions']]
        self.by_name = {e.name: e for e in self.entries}
        self.vlanes = vlanes
        self.sram_words = 1 << addr_bits
        self.acc_depth = acc_depth
        self.declared = tuple(extensions)
        self.w32 = [e for e in self.entries if e.width == 32]
        self.w64 = [e for e in self.entries if e.width == 64]

    def _decode(self, table, w):
        for e in table:
            if w & e.mask == e.value:
                f = e.fields_of(w, self.vlanes, self.sram_words, self.acc_depth, self.declared)
                if f is not None:
                    return (e.name, f)
        return None

    def decode32(self, w):
        return self._decode(self.w32, w & 0xffffffff)

    def decode64(self, w):
        return self._decode(self.w64, w)

    def encode(self, name, **f):
        return self.by_name[name].encode(**f)

    def issued_by(self, name, unit):
        u = self.by_name[name].units
        return 'any' in u or unit in u

    def timing(self, name):
        return self.by_name[name].timing
