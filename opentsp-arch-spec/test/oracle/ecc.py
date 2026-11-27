# SPDX-License-Identifier: Apache-2.0
"""The SECDED code of microarchitecture.md sec. 15, written from its definition for the oracle.

It is independent of the model's representation: the model keeps an overlay (the stored check bits
XOR the code of the stored data); here a stream value or an SRAM word carries the check bits
explicitly, per superlane, as the pair (c, spare): c the 9 check bits c[0..8] (bit i of c is c[i]), spare
the spare bit. A value with no entry for a superlane has the valid check bits of its data (spare 0).

The code (137, 128): data bit j sits at codeword position POS[j], the j-th integer >= 3 that is not a
power of two; check bit c[i] (i < 8) is at position 2^i; c[8] is the overall parity of positions 1..136.
The syndrome of a stored word is the XOR of the positions of its set bits; q is the parity of all 137
bits (data and c[0..8]).
"""

POS = []
_p = 3
while len(POS) < 128:
    if _p & (_p - 1):
        POS.append(_p)
    _p += 1
assert POS[0] == 3 and POS[127] == 136
INV = {p: j for j, p in enumerate(POS)}

POISON_XOR = 0b000000011          # the poison inverts c[0] and c[1]; the spare bit is the error report's (microarchitecture.md sec. 15.5)


def parity(x):
    return bin(x).count('1') & 1


def encode(d16):
    """The valid check bits (9 bits) of 16 data bytes."""
    x = int.from_bytes(d16, 'little')
    c = 0
    for i in range(8):
        b = 0
        for j in range(128):
            if (x >> j) & 1 and (POS[j] >> i) & 1:
                b ^= 1
        c |= b << i
    q = parity(x) ^ parity(c)
    return c | (q << 8)


def decode(d16, c):
    """('ok' | 'corr' | 'unc', the data bytes after the correction, the position the syndrome names)."""
    x = int.from_bytes(d16, 'little')
    s = 0
    for j in range(128):
        if (x >> j) & 1:
            s ^= POS[j]
    for i in range(8):
        if (c >> i) & 1:
            s ^= 1 << i
    q = parity(x) ^ parity(c)
    if s == 0 and q == 0:
        return 'ok', d16, 0
    if q == 1:
        if s == 0:
            return 'corr', d16, 0                          # c[8] itself
        if s & (s - 1) == 0:
            return 'corr', d16, s                          # a check bit c[0..7]
        if s <= 136:
            return 'corr', (x ^ (1 << INV[s])).to_bytes(16, 'little'), s
    return 'unc', d16, s


def full(v, ck, sl):
    """The explicit (c, spare) of every superlane of the value v."""
    if ck is None:
        return [(encode(v[16 * g:16 * g + 16]), 0) for g in range(sl)]
    return list(ck)


def norm(v, ck, sl):
    """None when every superlane has the valid code of its data."""
    if ck is None:
        return None
    ok = all(ck[g] == (encode(v[16 * g:16 * g + 16]), 0) for g in range(sl))
    return None if ok else tuple(ck)


def flip(v, ck, g, bit, sl):
    """Flip bit `bit` (0..127 data, 128..136 check bits c[0..8], 137 spare) of superlane g."""
    ent = full(v, ck, sl)
    c, sp = ent[g]
    if bit < 128:
        b = bytearray(v)
        b[16 * g + bit // 8] ^= 1 << (bit % 8)
        v = bytes(b)
    elif bit < 137:
        c ^= 1 << (bit - 128)
    else:
        sp ^= 1
    ent[g] = (c, sp)
    return v, norm(v, tuple(ent), sl)


def overlay(v, ck, g):
    """(stored check bits XOR the valid ones, spare) of superlane g: zero for a valid word."""
    if ck is None:
        return 0, 0
    return ck[g][0] ^ encode(v[16 * g:16 * g + 16]), ck[g][1]


def merge(v1, ck1, v2, ck2, vm, sl):
    """The check bits of the OR-merged value vm: the OR of the two values' overlays, applied to the valid
    check bits of vm (microarchitecture.md sec. 15, 'Merges')."""
    if ck1 is None and ck2 is None:
        return None
    ent = []
    for g in range(sl):
        o1, s1 = overlay(v1, ck1, g)
        o2, s2 = overlay(v2, ck2, g)
        ent.append((encode(vm[16 * g:16 * g + 16]) ^ (o1 | o2), s1 | s2))
    return norm(vm, tuple(ent), sl)


def poisoned(v, ck, mask, sl):
    """v's check bits with the superlanes of `mask` (a bit mask) poisoned: c[0], c[1] inverted (the spare bit is kept)."""
    if not mask:
        return ck
    ent = full(v, ck, sl)
    for g in range(sl):
        if (mask >> g) & 1:
            ent[g] = (encode(v[16 * g:16 * g + 16]) ^ POISON_XOR, 0)
    return norm(v, tuple(ent), sl)


def with_spare(v, ck, mask, sl):
    """v's check bits with the spare bit set in the superlanes of `mask` (an error report that flows with the value)."""
    if not mask:
        return ck
    ent = full(v, ck, sl)
    for g in range(sl):
        if (mask >> g) & 1:
            ent[g] = (ent[g][0], 1)
    return norm(v, tuple(ent), sl)
