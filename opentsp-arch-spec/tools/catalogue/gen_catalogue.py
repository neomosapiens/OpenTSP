"""Write isa-db/instructions/*.yaml (spec version 0.1). The catalogue is derived from the
Sail model and checked against it by tools/isa_db_check.sh; edit this table together with
the model's encdec/assembly clauses, then regenerate.
usage: gen_catalogue.py <isa-db/instructions>"""
import argparse, os, glob, yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", help="directory in which to regenerate instruction catalogue YAML")
OUT = parser.parse_args().output
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.yaml")):
    os.remove(f)

T4 = {"i8": 0, "f16": 1, "i32": 2, "f32": 3, "u8": 4, "i16": 5, "u16": 6, "u32": 7, "b8": 8, "b16": 9, "b32": 10,
      "e4m3": 11, "e5m2": 12, "bf16": 16}
ALL = list(T4)
BOOL = ["b8", "b16", "b32"]
FLT = ["f16", "f32"]
FP8 = ["e4m3", "e5m2"]     # : an implementation choice (microarchitecture.md sec. 5, "FP8 element types")
BF = ["bf16"]     # the profile extension `bf16` (type code 16, the type-extension bit 17 of the VXM word): an implementation choice
INTS = [t for t in ALL if t not in BOOL + FLT + FP8 + BF]
UNS = ["u8", "u16", "u32"]
SINT = [t for t in INTS if t not in UNS]
NOBOOL = [t for t in ALL if t not in BOOL]
NOFLT = [t for t in ALL if t not in FLT + FP8 + BF]


def V(name, msb, lsb, kind, **kw):
    d = {"name": name, "location": f"{msb}-{lsb}" if msb != lsb else str(msb), "kind": kind}
    d.update(kw)
    return d


def match(width, fixed, variables):
    bits = ["0"] * width
    for v in variables:
        for key in ("location", "hi_location", "top_location"):
            if key in v:
                a, _, b = v[key].partition("-")
                for i in range(int(b or a), int(a) + 1):
                    bits[i] = "-"
    for (msb, lsb, val) in fixed:
        for i in range(lsb, msb + 1):
            bits[i] = str((val >> (i - lsb)) & 1)
    return "".join(reversed(bits))


INS = []


def add(name, mnemonic, long_name, slice_, units, timing, width, fixed, variables, asm, op, extra=None):
    d = {"$schema": "opentsp.isa-db.instruction", "spec_version": "0.1", "name": name, "mnemonic": mnemonic, "long_name": long_name,
         "slice": slice_, "units": units, "timing": timing,
         "encoding": {"width": width, "match": match(width, fixed, variables), "variables": variables},
         "assembly": asm, "operation": op, "definedBy": "ref"}
    if extra:
        d.update(extra)
    INS.append(d)


ANY = ["any"]
# ---- ICU (common to every queue: HC34 p.27) ----
CU = [(31, 28, 0), (21, 16, 0)]
add("NOP", "nop", "No operation", "CU", ANY, "CU", 32, CU + [(27, 22, 0)],
    [V("n", 15, 0, "imm16", not_values=[0])], "nop {n}", "A timing prefix, not a dispatched instruction: at the start of a parcel (bundle bit 0) it delays the parcel by n cycles; the earliest dispatch of the next parcel is the previous parcel's cycle + 1 + the sum of the NOPs before it. It takes no dispatch slot and runs while an AGT loop or a Repeat runs. A NOP inside a bundle is a slot violation (OTS-E-QUE-001). n >= 1. ISCA20 sec. III-A1.")
add("Sync", "sync", "Barrier sync", "CU", ANY, "CU", 32, CU + [(27, 22, 1), (15, 0, 0)], [], "sync",
    "Parks the queue (control channel: every data queue) until the first Notify at or after it + 35 cycles. ISCA20 sec. III-A2.")
add("Notify", "notify", "Barrier notify", "CU", ANY, "CU", 32, CU + [(27, 22, 2), (15, 0, 0)], [], "notify",
    "Releases pending Syncs: dispatch resumes 35 cycles after the Notify. ISCA20 sec. III-A2.")
add("Repeat", "repeat", "Repeat previous instruction", "CU", ANY, "CU", 32, CU + [(27, 22, 3)],
    [V("n", 15, 4, "imm12", not_values=[0]), V("d", 3, 0, "imm4")], "repeat {n}, {d}",
    "Re-issues the queue's last instruction other than NOP, Repeat and AGT configuration n more times with d cycles between iterations (d = 0: none, consecutive cycles; d is 0..15): iteration k issues at p + k (d + 1), p the issue cycle of the repeated instruction, and the first not before the Repeat's own cycle; "
    "after Sync, Notify, Deskew or RuntimeDeskew there is nothing to repeat (OTS-E-RES-001). ISCA20 Table I.")
add("Ifetch", "ifetch", "Instruction fetch", "CU", ANY, "CU", 32, CU + [(27, 22, 4), (9, 0, 0)],
    [V("stream", 15, 10, "stream")], "ifetch {stream}",
    "Reads instruction text from the one stream operand stream: the vector at the execute cycle t and the next vector of the same stream at t + 1 (2 x vlanes bytes; any stream; a missing vector reports OTS-E-TIM-001 at its sample cycle and discards the pair) and appends it to the issuing queue's instruction queue (IQ): the text arrives icu.fetch_latency cycles after the second vector, at icu.fetch_bytes_per_cycle, padding units (0xFFFFFFFF) are dropped, and text that finds the IQ full (icu.iq_depth) is lost (OTS-E-IQ-002). An Ifetch that at either admission check (first sample or completion) finds the queue's fetch staging FIFO full (icu.fetch_staging_depth payloads) fetches nothing (OTS-E-IQ-003). ISCA20 sec. III-A3.")
add("IfetchM", "ifetchm", "Instruction fetch from local memory", "CU", ["MEM_W", "MEM_E"], "CU", 32, CU + [(27, 22, 7)],
    [V("addr", 15, 0, "sram_addr", max_expr="sram_words-1")], "ifetchm {addr}",
    "Reads instruction text from words addr and addr+1 of the issuing MEM slice in two successive SRAM read cycles (2 x vlanes bytes), sampling each word and its stored check bits at its own read; the R path is reserved for two cycles and each cycle uses its own bank; addr+1 outside the segment is OTS-E-OPR-001. The payload completes one cycle after the first read and then follows the same IQ latency, staging and overflow rules as Ifetch. A repeat that overlaps a pending local read is OTS-E-QUE-001. A repeated IfetchM (Repeat) reserves the R path for two cycles and the bank of its second read exactly as an explicit IfetchM issued in that cycle does. MEM queues only. Uses the Ifetch slot. ISCA20 Table I ('from streams or local memory'); the address rule is an opentsp rule.")
add("Config", "config", "Configure low-power mode", "CU", ANY, "CU", 32, CU + [(27, 22, 5)],
    [V("v", 15, 0, "imm16")], "config {v}", "On a MEM queue: liquid-thread split, tiles 0..v are thread 0 and the rest thread 1 (v >= geometry.superlanes - 1: all thread 0). On other queues it is an accepted power hint with no architectural state, result, diagnostic or timing effect; physical power state is not modelled. Config is named in ISCA20 Table I; the liquid-thread partition rule is an opentsp choice.")
add("SetVL", "setvl", "Set vector length", "CU", ANY, "CU", 32, CU + [(27, 22, 6), (15, 6, 0)],
    [V("n", 5, 0, "imm6", not_values=[0], max=32)], "setvl {n}",
    "Vector length of the issuing queue = n x 16 lanes, n = 1..32 on every profile (a profile has at most 32 superlanes; n above the profile's superlane count is the count: clamped). Inactive superlanes make no SRAM data changes and produce zero stream output; input integrity checks remain active. This is an opentsp output/storage mask, not a power-gating model. ISCA20 sec. II-F motivates activity control but does not specify this instruction or its masking semantics.")
# ---- MEM ----
MEMU = ["MEM_W", "MEM_E"]
THR = V("thread", 53, 53, "imm1")
add("Read", "read", "SRAM read", "MEM", MEMU, "MEM", 64, [(63, 60, 1), (59, 54, 0), (52, 48, 0), (25, 0, 0)],
    [V("addr", 47, 32, "sram_addr", max_expr="sram_words-1"), V("dst", 31, 26, "stream"), THR], "read {addr}, {dst}, {thread}",
    "Reads word addr (plus the AGT offset; outside the segment: OTS-E-OPR-001) of the issuing slice onto stream dst (plus the AGT StreamStep offset). Tiles outside the SetVL length or not in liquid thread `thread` (Config SL) drive zero. Loopable. A Write of the same liquid thread in the same parcel dispatches with it if the banks differ (else OTS-E-MEM-001; references of different threads never clash). ISCA20 Table I.")
add("Write", "write", "SRAM write", "MEM", MEMU, "MEM", 64, [(63, 60, 1), (59, 54, 1), (52, 48, 0), (25, 0, 0)],
    [V("addr", 47, 32, "sram_addr", max_expr="sram_words-1"), V("src", 31, 26, "stream"), THR], "write {addr}, {src}, {thread}",
    "Writes stream src to word addr (plus the AGT offsets); only the active tiles of liquid thread `thread` (within SetVL) change. Loopable. ISCA20 Table I.")
add("Gather", "gather", "SRAM gather", "MEM", MEMU, "MEM", 64, [(63, 60, 1), (59, 54, 3), (52, 52, 0), (35, 0, 0)],
    [V("dst", 47, 42, "stream"), V("map", 41, 36, "stream"), V("k", 51, 48, "imm4"), THR], "gather {dst}, {map}, {k}, {thread}",
    "Per active tile g: address = little-endian bytes 0..1 of superlane g of map, of which the bits from sram.addr_bits up are ignored (the address is the low sram.addr_bits bits; no diagnostic); lane 16g+k of dst = byte k of superlane g of that word; other lanes zero (one element per tile). Each tile acts alone: a tile whose address (plus the AGT offset) is outside the segment or the slice drives zero on its lane, the good tiles' lanes are written, and the instruction reports one OTS-E-OPR-001 (an exception to \"a diagnostic means no effect\"). Loopable.")
add("Scatter", "scatter", "SRAM scatter", "MEM", MEMU, "MEM", 64, [(63, 60, 1), (59, 54, 4), (52, 52, 0), (35, 0, 0)],
    [V("src", 47, 42, "stream"), V("map", 41, 36, "stream"), V("k", 51, 48, "imm4"), THR], "scatter {src}, {map}, {k}, {thread}",
    "Per active tile g: byte 16g+k of the word at map's tile-g address (the bits of the little-endian halfword in bytes 0..1 of the map from sram.addr_bits up are ignored: the address is the low sram.addr_bits bits; no diagnostic) = lane 16g+k of src; the rest of the word is unchanged (one element per tile). Each tile acts alone: a tile whose address (plus the AGT offset) is outside the segment or the slice writes nothing, the good tiles write, and the instruction reports one OTS-E-OPR-001 (an exception to \"a diagnostic means no effect\"). Loopable.")
AGT = [(63, 60, 1), (53, 32, 0), (15, 0, 0)]
add("Countdown", "countdown", "AGT inter-reference delay", "MEM", MEMU, "MEM", 64, AGT + [(59, 54, 5)],
    [V("d", 31, 16, "imm16")], "countdown {d}",
    "AGT configuration: idle cycles after each reference of the current loop level (Wait(countdown)): the period is d + 1; an explicit value below 1 counts as 1; 0 when not received. HC34 p.25.")
add("Step", "step", "AGT stride", "MEM", MEMU, "MEM", 64, AGT + [(59, 54, 6)],
    [V("a", 31, 16, "simm16", min=-32767, max=32767, max_expr="sram_words-1")], "step {a}",
    "AGT configuration: signed address stride of the current loop level, at most min(2^sram.addr_bits - 1, 32767) in magnitude (8191 with the reference 13 bits). HC34 p.25.")
add("StreamStep", "streamstep", "AGT stream stride", "MEM", MEMU, "MEM", 64, AGT + [(59, 54, 11)],
    [V("s", 31, 16, "imm16", max=31)], "streamstep {s}",
    "AGT configuration (optional): each reference of the current loop level advances the data stream id by s (past stream 31: OTS-E-OPR-001).")
add("Iterations", "iterations", "AGT loop bound", "MEM", MEMU, "MEM", 64, AGT + [(59, 54, 7)],
    [V("n", 31, 16, "imm16")], "iterations {n}",
    "AGT configuration: the current level makes n references (n < 1 means 1; 2 when not received); up to four levels, inner to outer (fields in the order Countdown, Step, StreamStep, Iterations). HC34 p.25.")
add("SetSegment", "setsegment", "Set segment size", "MEM", MEMU, "MEM", 64, [(63, 60, 1), (59, 54, 8), (53, 33, 0), (15, 0, 0)],
    [V("n", 32, 16, "imm17", max_expr="sram_words")], "setsegment {n}",
    "References at address >= n raise OTS-E-OPR-001. n is at most 2^sram.addr_bits (a 17-bit field, so that 65536 is expressible).")
add("PowerConfig", "powerconfig", "MEM power configuration", "MEM", MEMU, "MEM", 64, AGT + [(59, 54, 9)],
    [V("v", 31, 16, "imm16")], "powerconfig {v}", "Accepted power hint; no architectural state, result, stream, diagnostic or timing effect. No power or sleep/wake state is modelled.")
add("DeepSleep", "deepsleep", "MEM deep sleep", "MEM", MEMU, "MEM", 64, [(63, 60, 1), (59, 54, 10), (53, 0, 0)],
    [], "deepsleep", "Accepted power hint; no architectural state, result, stream, diagnostic or timing effect. No power or sleep/wake state is modelled.")
# ---- VXM ----
# Bits 21, 20, 19: the chain bits of the first, second and third source operand;
# a chain bit of an operand the instruction does not have is 0.
VX = [(63, 60, 2), (18, 18, 0), (15, 0, 0)]      # bits 17 (the type-extension bit) and 16 (the destination's, in Convert) are fields or reserved zero, set in vadd
CHAIN_BIT = (21, 20, 19)


def vsrcs(names):
    """The chain variables of an entry's source operands (in operand order) and the fixed
    zero chain bits of the operands it does not have."""
    vs = [V(n + "_chain", CHAIN_BIT[i], CHAIN_BIT[i], "chain") for i, n in enumerate(names)]
    return vs, [(b, b, 0) for b in CHAIN_BIT[len(names):]]


SG4_TEXT = (" Group bit (`.sg4` after the type, bit 18): every operand is the 4-stream group at"
            " its stream (SG4-aligned, else OTS-E-OPR-002) and the ALU performs 4 / width independent operations per"
            " lane, one per byte (8-bit types) or halfword (16-bit types) of the group (ISCA20 sec. III-C: 32-bit"
            " operands on SG4; 'four INT8 operations per cycle' per ALU); for 32-bit types it is the plain operation. The non-group forms of the same operations are the specification's own.")


def vadd(name, mnemonic, long_name, fixed, variables, asm, op, srcs, group=False):
    """A VXM entry: each source operand {x} in the template becomes {x}{x_chain}; with group,
    the type is followed by the group-bit suffix {sg4}."""
    cv, cz = vsrcs(srcs)
    for n in srcs:
        asm = asm.replace("{" + n + "}", "{" + n + "}{" + n + "_chain}")
    vx = VX
    if group:
        asm = asm.replace("{type}", "{type}{sg4}", 1)
        variables = variables + [V("sg4", 18, 18, "group4")]
        vx = [(63, 60, 2), (15, 0, 0)]
        op = op + SG4_TEXT
    if name != "Convert":
        vx = vx + [(16, 16, 0)]             # bit 16 is the type-extension bit of a Convert's destination type only
    add(name, mnemonic, long_name, "VXM", ["VXM"], "VXM", 64, vx + fixed + cz, variables + cv, asm,
        op + " A source written `x.chain` (chain bit set) is the VXM result committed at the VXM station in the cycle before the read (ALU chaining, ISCA20 sec. III-C); otherwise the stream value that flows to the station.")


def vtext(types):
    """The legal element types of a VXM entry, in words (must equal the model's legality)."""
    kinds = []
    if any(t in SINT for t in types):
        kinds.append("signed integer" if all(t in types for t in SINT) else "signed " + "/".join(t for t in types if t in SINT))
    if any(t in UNS for t in types):
        kinds.append("unsigned integer" if all(t in types for t in UNS) else "unsigned " + "/".join(t for t in types if t in UNS))
    if any(t in FLT for t in types):
        kinds.append("float" if all(t in types for t in FLT) else "/".join(t for t in types if t in FLT))
    if any(t in FP8 for t in types):
        kinds.append("fp8" if all(t in types for t in FP8) else "/".join(t for t in types if t in FP8))
    if any(t in BF for t in types):
        kinds.append("bf16 (extension)")
    if any(t in BOOL for t in types):
        kinds.append("boolean")
    txt = ", ".join(kinds[:-1]) + (" or " if len(kinds) > 1 else "") + kinds[-1]
    return txt + " (" + ", ".join(types) + ")"


def vt(types):
    d = V("type", 53, 52, "eltype_code", hi_location="23-22", top_location="17", codes=[T4[t] for t in types])
    if any(t in BF for t in types):
        d["extension_codes"] = {"bf16": [T4[t] for t in types if t in BF]}
    return d


BIN = [("Add", 0, NOBOOL), ("Add_sat", 1, INTS), ("Add_mod", 2, INTS), ("Sub", 3, NOBOOL), ("Sub_sat", 4, INTS),
       ("Sub_mod", 5, INTS), ("Mul", 6, NOBOOL), ("Mul_sat", 7, INTS), ("Mul_mod", 8, INTS), ("Max", 9, NOBOOL),
       ("Min", 10, NOBOOL), ("And", 11, NOFLT), ("Or", 12, NOFLT), ("Xor", 13, NOFLT)]
for nm, op, ty in BIN:
    vadd(nm, nm.lower(), f"Vector {nm.replace('_', ' ')}", [(59, 54, op), (33, 24, 0)],
        [vt(ty), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream"), V("rhs", 39, 34, "stream")], nm.lower() + ".{type} {dst}, {lhs}, {rhs}",
        f"Per lane: dst = lhs {nm.lower()} rhs in the element type; types: {vtext(ty)}.", ["lhs", "rhs"], group=True)
UN = [("Not", 14, NOFLT), ("Neg", 15, SINT + FLT + FP8 + BF), ("Abs", 16, SINT + FLT + FP8 + BF), ("ReLU", 20, SINT + FLT + FP8 + BF),
      ("TanH", 21, ["i8"] + FLT), ("Exp", 22, ["i8"] + FLT), ("RSqrt", 23, ["i8"] + FLT), ("Log", 24, ["i8"] + FLT)]
# There is no divide instruction: the two measured idioms and their domains are in docs/kernel-cookbook.md section 2.
UN_NOTE = {"RSqrt": " The ISA has no divide: 1/x is formed with RSqrt as rsqrt(x * x) (chained; 1/|x|, within 1 ulp for x in [2^-63, 2^64)) or"
                    " as rsqrt(x) * rsqrt(x) (1/x for x > 0 over the normal range, within 2 ulp); the domains, errors and edge values are"
                    " measured in docs/kernel-cookbook.md section 2."}
for nm, op, ty in UN:
    vadd(nm, nm.lower(), f"Vector {nm}", [(59, 54, op), (39, 24, 0)],
        [vt(ty), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream")], nm.lower() + ".{type} {dst}, {lhs}",
        f"Per lane: dst = {nm.lower()}(lhs)" + (", correctly rounded" if op >= 21 else "") + f"; types: {vtext(ty)}." + UN_NOTE.get(nm, ""), ["lhs"], group=True)
vadd("Mask", "mask", "Vector mask", [(59, 54, 17), (33, 24, 0)],
    [vt(ALL), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream"), V("mask", 39, 34, "stream")], "mask.{type} {dst}, {lhs}, {mask}",
    f"Per lane: dst = mask != 0 ? lhs : 0; types: {vtext(ALL)}.", ["lhs", "mask"], group=True)
vadd("Select", "select", "Vector select", [(59, 54, 18), (27, 24, 0)],
    [vt(ALL), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream"), V("rhs", 39, 34, "stream"), V("mask", 33, 28, "stream")],
    "select.{type} {dst}, {lhs}, {rhs}, {mask}", f"Per lane: dst = mask != 0 ? lhs : rhs; types: {vtext(ALL)}.", ["lhs", "rhs", "mask"], group=True)
vadd("Cmp", "cmp", "Vector compare", [(59, 54, 19), (33, 28, 0)],
    [vt(ALL), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream"), V("rhs", 39, 34, "stream"),
     V("mode", 27, 24, "cmp_mode", values=[0, 1, 2, 3, 4, 5], bool_values=[0, 1])],
    "cmp.{mode}.{type} {dst}, {lhs}, {rhs}", f"Per lane: all ones if lhs <mode> rhs, else zero; types: {vtext(ALL)}; booleans compare eq/ne only.", ["lhs", "rhs"], group=True)
CONV = [("Conv_I8_F32", 25, "i8"), ("Conv_F32_I8", 26, "f32"), ("Conv_I8_F16", 27, "i8"), ("Conv_F16_I8", 28, "f16"),
        ("Conv_I8_I32", 29, "i8"), ("Conv_I32_I8", 30, "i32"), ("Conv_F16_F32", 31, "f16"), ("Conv_F32_F16", 32, "f32")]
for nm, op, src in CONV:
    a, b = nm.split("_")[1:]
    vadd(nm, nm.lower(), f"Convert {a.lower()} to {b.lower()}", [(59, 54, op), (39, 24, 0)],
        [vt([src]), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream")], nm.lower() + ".{type} {dst}, {lhs}",
        f"Per lane: dst = convert(lhs) from {a.lower()} to {b.lower()}. The type field must equal the source type.", ["lhs"])
vadd("Convert", "convert", "Convert between any types", [(59, 54, 34), (39, 28, 0)],
    [vt(ALL), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream"), V("dtype", 27, 24, "eltype4", top_location="16", codes=list(range(13)) + [16], extension_codes={"bf16": [16]})],
    "convert.{type}.{dtype} {dst}, {lhs}",
    "Per lane: integers saturate, floats round to nearest even, bool = value != 0; to e4m3 an overflow or an infinity saturates to +-448, to e5m2 an overflow gives an infinity; a NaN gives the canonical NaN of the destination. Source type in the type field, destination in bits 27..24.", ["lhs"])
vadd("Reduce", "reduce", "Superlane reduction", [(59, 54, 33), (39, 28, 0)],
    [vt(NOBOOL), V("dst", 51, 46, "stream"), V("lhs", 45, 40, "stream"), V("mode", 27, 24, "reduce_op", values=[0, 1, 2])],
    "reduce.{mode}.{type} {dst}, {lhs}", f"Per superlane g: lane 16g = add/max/min over lanes 16g..16g+15, folded in ascending lane order (acc = lane 16g; acc = op(acc, lane 16g+j) for j = 1..15), each step rounded to nearest even in the element type for floats, wrapping for integers (opentsp decision: no source gives the order); other lanes zero; types: {vtext(NOBOOL)}.", ["lhs"])
# ---- MXM ----
MX = [(31, 28, 3)]
MXU = ["MXM_W", "MXM_E"]
add("LW", "lw", "Load weights", "MXM", MXU, "MXM_LW", 32, MX + [(27, 22, 0), (13, 13, 0), (11, 0, 0), (17, 14, 0)],
    [V("type", 21, 20, "mxm_wtype", hi_location="12", values=[0, 1, 2, 3, 4, 5], extension_codes={"bf16": [4], "block_fp8": [5]}), V("src", 19, 18, "stream_group16")], "lw.{type} {src}",
    "Appends 16 stream values src..src+15 to the weight buffer, which holds mxm.rows stream values for i8, u8 and fp8 and 2 x mxm.f16_rows for f16, bf16 and bfp8 (a longer one: OTS-E-RES-002; mxm.rows i8/u8/fp8 rows or mxm.f16_rows f16, bf16 or bfp8 rows). The type fp8 (code 3) stores bytes whose format (e4m3 or e5m2) the ABC gives. The type bf16 (code 4, the weight-type bit 12 set) exists only in a profile that declares the extension `bf16`: otherwise the word does not decode (OTS-E-DEC-003); a bf16 row is two stream values as an f16 row is, and takes the plane pair as f16 does. The type bfp8 (code 5, extension `block_fp8`; the weight-type bit 12 set, bits 21..20 = 1) takes two stream values per row, the e4m3 data and the block scales, like f16 (a plane pair, mxm.f16_rows rows); in a profile that does not declare `block_fp8` the word does not decode. ISCA20 Table I.")
add("IW", "iw", "Install weights from buffer", "MXM", MXU, "MXM_IW", 32, MX + [(27, 22, 1), (19, 13, 0), (11, 0, 0)],
    [V("type", 21, 20, "mxm_wtype", hi_location="12", values=[0, 1, 2, 3, 4, 5], extension_codes={"bf16": [4], "block_fp8": [5]})], "iw.{type}",
    "Installs the LW buffer once the last LW has committed. ISCA20 Table I.")
add("IWS", "iw", "Install weights from streams", "MXM", MXU, "MXM_IW", 32, MX + [(27, 22, 1), (13, 13, 1), (11, 0, 0), (17, 14, 0)],
    [V("type", 21, 20, "mxm_wtype", hi_location="12", values=[0, 1, 2, 3, 4, 5], extension_codes={"bf16": [4], "block_fp8": [5]}), V("src", 19, 18, "stream_group16")], "iw.{type} {src}",
    "Installs 16 stream values directly into the array, appending rows (an array holds mxm.rows stream values; beyond that OTS-E-RES-002). ISCA20 Table I ('from streams or LW buffer').")
add("ABC", "abc", "Activation buffer control", "MXM", MXU, "MXM_ABC", 32, MX + [(27, 22, 2), (13, 13, 0), (11, 0, 0)],
    [V("type", 21, 20, "mxm_wtype", hi_location="12", values=[0, 1, 2, 4, 5], extension_codes={"bf16": [4], "block_fp8": [5]}), V("src", 19, 14, "stream")], "abc.{type} {src}",
    "product[r] = sum_k W[r][k] * x[k] over vlanes inputs; up to mxm.rows output rows for i8/u8 and mxm.f16_rows for f16, remaining lanes zero (i8/u8: exact, 32-bit wrap; the weight sign is the installed type, the activation sign the ABC type, so i8 and u8 mix; f16 and bf16: exact sum rounded once to f32, weights of the same type only; bf16 (type code 4, the weight-type bit 12) exists only in a profile that declares the extension `bf16`, and has the capacity, plane pair and rate of f16; type 3 with mode bit 13 is the entry ABC8). Needs IW committed. FP16 and bf16 use an even/odd plane pair and accepts an ABC every timing.mxm_abc.occupancy_f16 cycles (1 in the shipped profiles); an earlier issue is OTS-E-RES-001. The type bfp8 (extension `block_fp8`, code 5: the weight-type bit 12 set with bits 21..20 = 1) reads the activation as a stream pair (e4m3 data at src, block scales at src + 1) and needs bfp8 weights; a block is the 16 lanes of one superlane with the scale of its first lane (the low four bits, a two's-complement exponent -8..7), the value of a lane is its e4m3 element times 2 to the power of the block scale, the products and the sum are exact, and the plane pair and the initiation interval are those of f16.")
add("ABC8", "abc", "Activation buffer control, FP8", "MXM", MXU, "MXM_ABC", 32, MX + [(27, 22, 2), (21, 20, 3), (12, 0, 0)],
    [V("fmt", 13, 13, "fp8_format"), V("src", 19, 14, "stream")], "abc.{fmt} {src}",
    "The fp8 form of ABC (type field 3, mode bit 13 = the format: 0 e4m3, 1 e5m2, for the weights and the activations alike): product[r] = sum_k W[r][k] * x[k] for r < mxm.rows, the exact sum of the exact products rounded once to f32 (round to nearest even), as the f16 form; fp8 weights only (an i8, u8, f16, bf16 or bfp8 plane gives OTS-E-RES-001, as does an fp8 ABC on any other weight type). Capacity and rate as i8: one ABC per cycle, mxm.rows rows, the plane is not held. A NaN operand, an infinity times zero or products of both infinities give the canonical NaN (e5m2 only: e4m3 has no infinity), else an infinite product gives that infinity; an exact zero sum is -0 only if every product is -0. ACC f32 reads the product. Implementation choice (microarchitecture.md sec. 6).")
add("ACC", "acc", "Accumulate", "MXM", MXU, "MXM_ACC", 32, MX + [(27, 22, 3), (7, 0, 0)],
    [V("type", 21, 20, "mxm_acctype", values=[0, 2, 3], extension_codes={"wide_acc": [0]}), V("dst", 19, 14, "stream"), V("mode", 13, 13, "acc_mode"),
     V("acc", 12, 12, "acc_index", hi_location="11-8", max_expr="acc_depth-1")],
    "acc.{type} {dst}, {mode}, {acc}",
    "Accumulator acc (0 .. mxm.acc_depth - 1; two per MACC in the shipped profiles) = product or acc + product; written to dst..dst+3. The index is bit 12 and, above it, bits 11..8 (acc = bit 12 + 2 x bits 11..8); an index at or above mxm.acc_depth does not decode (OTS-E-DEC-003). Needs ABC committed. i32 follows an i8 or u8 ABC; f32 an f16, bf16, fp8 or bfp8 ABC. The type x128 (extension `wide_acc`, code 0; a profile that does not declare it rejects the word, OTS-E-DEC-003) adds the exact product to a 128-bit fixed-point accumulator (units of 2^-48, wrapping at 128 bits, special values sticky) and writes its read-out, rounded once to f32 (round to nearest even); it follows an f16, fp8 or bfp8 ABC (an integer or bf16 ABC gives OTS-E-OPR-001: the exact bf16 sum is in units of 2^-266) and keeps its own accumulators (mxm.acc_depth of them per plane), apart from the i32 and f32 ones. With mxm.fp8_rate 2, two FP8 ABCs of one plane in one cycle leave two products of the same dispatch cycle: an even accumulator reads the one of the lower queue, an odd one the one of the higher queue.")
# ---- SXM ----
SX = [(63, 60, 4), (53, 52, 0), (17, 0, 0)]
# Shift and Permute move lanes across superlanes: timing class SXM_CROSS, whose d_func is at
# least geometry.superlanes (microarchitecture.md sec. 7).
add("Shift", "shift", "Lane shift", "SXM", ["SXM_W_SHF", "SXM_E_SHF"], "SXM_CROSS", 64, SX + [(59, 54, 0), (29, 18, 0)],
    [V("dst", 51, 46, "stream"), V("src", 45, 40, "stream"), V("dir", 39, 39, "shift_dir"), V("amount", 38, 30, "lane_count", max_expr="vlanes-1")],
    "shift {src}, {dir}, {amount}, {dst}", "up: dst[l] = src[l - amount]; down: dst[l] = src[l + amount]; zero fill. ISCA20 Fig. 8. Timing class SXM_CROSS: a lane may come from a later superlane, so the result commits no earlier than dispatch + superlanes.")
add("Permute", "permute", "Lane permute", "SXM", ["SXM_W_PRM", "SXM_E_PRM"], "SXM_CROSS", 64, SX + [(59, 54, 1), (39, 34, 0), (29, 18, 0)],
    [V("dst", 51, 46, "stream"), V("src", 45, 40, "stream"), V("map", 33, 30, "map_id")], "permute {src}, {map}, {dst}",
    "dst[l] = m[l] < vlanes ? src[m[l]] : 0; a map entry >= vlanes masks its lane (zero fill) and may repeat; the entries below vlanes must be distinct, i.e. the map must be bijective, which LoadMap and LoadMapWide record with the map (OTS-E-SXM-001: no effect). A map is ready for a Permute geometry.superlanes cycles after the execute cycle of its LoadMap or LoadMapWide; a Permute that executes earlier is OTS-E-RES-001 (no effect). Timing class SXM_CROSS, as Shift.")
add("Distribute", "distribute", "Superlane distribute", "SXM", ["SXM_W_DST", "SXM_E_DST"], "SXM", 64, SX + [(59, 54, 2), (39, 34, 0), (29, 18, 0)],
    [V("dst", 51, 46, "stream"), V("src", 45, 40, "stream"), V("map", 33, 30, "map_id")], "distribute {src}, {map}, {dst}",
    "Per superlane g: e = m[16g + j]; dst[16g + j] = e < 16 ? src[16g + e] : 0.")
add("Rotate", "rotate", "Tile rotate", "SXM", ["SXM_W_TSP", "SXM_E_TSP"], "SXM", 64, SX + [(59, 54, 3), (29, 18, 0)],
    [V("dst", 51, 46, "stream"), V("src", 45, 40, "stream"), V("n", 39, 30, "tile_size", values=[3, 4])], "rotate {src}, {n}, {dst}",
    "Per superlane g, lanes 0..n^2-1 of src form an n x n tile (row r at lanes r*n..r*n+n-1). Output t = tr*n + tc goes to stream dst+t: "
    "lane 16g + r*n + c = src lane 16g + ((r+tr) mod n)*n + (c+tc) mod n for r, c < n, other lanes zero (a cyclic 2-D shift of the tile). "
    "dst + n^2 > 32 is OTS-E-OPR-002. ISCA20 Table I.")
add("Transpose", "transpose", "Superlane transpose", "SXM", ["SXM_W_TSP", "SXM_E_TSP"], "SXM", 64, SX + [(59, 54, 4), (43, 40, 0), (49, 46, 0), (39, 18, 0)],
    [V("dst", 51, 50, "stream_group16"), V("src", 45, 44, "stream_group16")], "transpose {src}, {dst}",
    "16 streams to 16 streams; per superlane g: out_c[16g + r] = in_r[16g + c].")
add("LoadMap", "loadmap", "Load map", "SXM", ["SXM_W_PRM", "SXM_W_DST", "SXM_E_PRM", "SXM_E_DST"], "SXM", 64, SX + [(59, 54, 5), (51, 46, 0), (39, 34, 0), (29, 18, 0)],
    [V("src", 45, 40, "stream"), V("map", 33, 30, "map_id")], "loadmap {map}, {src}",
    "m[l] = src[l]. Records with the map whether it is bijective (the entries below vlanes are distinct); the map is ready for a Permute geometry.superlanes cycles after this execute cycle, and for a Distribute the next cycle.")
add("LoadMapWide", "loadmapwide", "Load wide map", "SXM", ["SXM_W_PRM", "SXM_W_DST", "SXM_E_PRM", "SXM_E_DST"], "SXM", 64, SX + [(59, 54, 6), (51, 46, 0), (39, 34, 0), (23, 18, 0)],
    [V("src", 45, 40, "stream"), V("map", 33, 30, "map_id"), V("src2", 29, 24, "stream")], "loadmapwide {map}, {src}, {src2}",
    "m[l] = (src2[l] mod 4) * 256 + src[l]. Records with the map whether it is bijective (the entries below vlanes are distinct); the map is ready for a Permute geometry.superlanes cycles after this execute cycle, and for a Distribute the next cycle.")
add("SelectS", "sxselect", "Shifter select", "SXM", ["SXM_W_SEL", "SXM_E_SEL"], "SXM", 64, [(63, 60, 4), (53, 52, 0), (59, 54, 7), (39, 30, 0), (11, 0, 0)],
    [V("dst", 51, 46, "stream"), V("a", 45, 40, "stream"), V("b", 29, 24, "stream"), V("sel", 23, 18, "stream"), V("c", 17, 12, "stream")],
    "sxselect {a}, {b}, {c}, {sel}, {dst}", "dst[l] = a[l], b[l] or c[l] for sel[l] = 0, 1 or 2 (unshifted, north-shifted, south-shifted); a lane whose sel is above 2 is zero, the other lanes are written, and the instruction reports one OTS-E-OPR-001 (an exception to \"a diagnostic means no effect\"). ISCA20 Fig. 8.")
# ---- C2C ----
CHIP = V("chip_id", 21, 16, "chip_id", hi_location="2-0")   # bits 8..6 of the chip number in the low reserved bits
LINK = V("link", 9, 6, "link_id", hi_location="5-3")        # bits 6..4 of the link number
add("Send", "send", "Chip-to-chip send", "C2C", ["C2C_W_TX", "C2C_E_TX"], "C2C", 32, [(31, 28, 5), (27, 22, 0)],
    [CHIP, V("stream", 15, 10, "stream"), LINK], "send {chip_id}, {link}, {stream}", "Reads stream at the TX gap and puts the vector on link `link` (0 .. c2c.links - 1; 16 links in the published chip, ISCA20 sec. II; at most 128: the link number is 4 bits plus 3 bits in the low reserved bits, and the chip number 6 bits plus 3 bits there); it arrives at chip_id after the link's latency (c2c.link_latency_min .. c2c.link_latency_max). link >= c2c.links or chip_id not the link's peer: OTS-E-OPR-001. A link accepts one vector per c2c.cycles_per_vector cycles: a Send on a link that accepted a vector fewer cycles before is OTS-E-LNK-001 and sends nothing (checked after OPR-001, before the stream is read); a Send released into the cycles of a csr or ctrl packet in flight on the link is OTS-E-LNK-003 and sends nothing (checked after LNK-001). ISCA20 Table I; ISCA22 sec. 2.2, 2.3, Table 2.")
add("Receive", "receive", "Chip-to-chip receive", "C2C", ["C2C_W_RX", "C2C_E_RX"], "C2C", 32, [(31, 28, 5), (27, 22, 1)],
    [CHIP, V("stream", 15, 10, "stream"), LINK], "receive {chip_id}, {link}, {stream}", "Writes the oldest vector arrived on link `link` to stream (a MEM Write stores it: ISCA20 Table I 'emplacing it in main memory', an opentsp choice); none arrived: OTS-E-TIM-001. link >= c2c.links or chip_id not the link's peer: OTS-E-OPR-001. ISCA20 Table I; ISCA22 sec. 2.3.")
add("Deskew", "deskew", "Link deskew", "C2C", ANY, "C2C", 32, [(31, 28, 5), (27, 22, 2), (21, 0, 0)],
    [], "deskew", "Holds the issuing queue until the next HAC epoch boundary (cycle mod c2c.hac_period = 0; 252 cycles, ISCA22 footnote 4). ISCA20 Table I; ISCA22 sec. 3.2.")
add("Transmit", "transmit", "Chip-to-chip notification", "C2C", ["C2C_W_TX", "C2C_E_TX"], "C2C", 32, [(31, 28, 5), (27, 22, 3), (15, 10, 0)],
    [CHIP, LINK], "transmit {chip_id}, {link}", "Sends the notification vector (lane 0 = 1, other lanes 0) to chip_id over link `link`. link >= c2c.links or chip_id not the link's peer: OTS-E-OPR-001; a link that accepted a vector fewer than c2c.cycles_per_vector cycles before is busy: OTS-E-LNK-001, nothing is sent; a Transmit released into the cycles of a csr or ctrl packet in flight on the link is OTS-E-LNK-003, nothing is sent. ISCA22 sec. 3.2.")
add("RuntimeDeskew", "runtime_deskew", "Runtime deskew", "C2C", ANY, "C2C", 32, [(31, 28, 5), (27, 22, 4), (21, 16, 0)],
    [V("t", 15, 0, "imm16")], "runtime_deskew {t}", "Holds the issuing queue max(t, 1) cycles (t plus the HAC-SAC drift, 0 for one modelled chip). ISCA22 sec. 3.3.")
# ---- host I/O (the PCIe host interface sits in the C2C slice) ----
HOSTSTREAMS = list(range(24, 32)) + list(range(56, 64))     # S24..S31 on either row
add("PciReceive", "pci_receive", "Host receive over PCIe", "C2C", ["C2C_W_RX", "C2C_E_RX"], "C2C", 32, [(31, 28, 5), (27, 22, 5), (21, 16, 0), (9, 0, 0)],
    [V("stream", 15, 10, "stream", values=HOSTSTREAMS)], "pci_receive {stream}",
    "Writes the oldest vector the host's DMA engine has pushed (the ingress FIFO of this chip's own host interface, host.cycles_per_vector cycles per vector, host.ingress_depth deep; a vector that finds it full waits in the DMA engine, nothing is dropped) to stream, which must be one of S24..S31 on either row (the upper eight streams, an implementation choice); a MEM Write commits it to SRAM (the two steps, the instruction and the Write, are the model's choice). None arrived: OTS-E-TIM-001. Implementation choice: the instruction, the encoding, the queues, the restriction to S24..S31. The published TSP sources: ISCA20 sec. II-6 (the PCIe host interface in the C2C module, a lightweight DMA engine that emplaces a model onto the TSP memory).")
add("PciSend", "pci_send", "Host send over PCIe", "C2C", ["C2C_W_TX", "C2C_E_TX"], "C2C", 32, [(31, 28, 5), (27, 22, 6), (21, 16, 0), (9, 0, 0)],
    [V("stream", 15, 10, "stream")], "pci_send {stream}",
    "Reads stream at the TX gap and gives the vector to the host. The PCIe interface accepts one vector per host.cycles_per_vector cycles; a PciSend on an interface that accepted a vector fewer cycles before is OTS-E-HIO-001 and sends nothing (checked before the stream is read). Optional; an implementation choice: the instruction and its pacing (the published sources give the PCIe host interface in the C2C module and the host's role of loading the model and collecting results, ISCA20 sec. II-6, and no instruction form or pacing for the chip-to-host direction).")
add("HostWait", "host_wait", "Wait for a host doorbell", "C2C", ANY, "CU", 32, [(31, 28, 5), (27, 22, 7), (21, 6, 0)],
    [V("i", 5, 0, "imm6")], "host_wait {i}",
    "Parks the queue until the host rings doorbell i (CSR 0x300 + i): the next parcel dispatches the cycle after the ring; a doorbell rung earlier is taken at once. i >= host.doorbells: OTS-E-OPR-001. Implementation choice.")
add("HostNotify", "host_notify", "Raise an MSI-X interrupt to the host", "C2C", ANY, "CU", 32, [(31, 28, 5), (27, 22, 8), (21, 4, 0)],
    [V("n", 3, 0, "imm4", not_values=[0])], "host_notify {n}",
    "Raises MSI-X vector n (1..15) towards the host at the execute cycle: IRQ_CAUSE bit n is set, and the interrupt is delivered at once if IRQ_MASK bit n is set; a masked vector keeps its cause bit and is delivered when the host enables it. Vector 0 is the error interrupt. The published sources: MSI-X interrupts from the C2C slice; implementation choice: the instruction and the numbering.")
# ---- the polling loop at program start ----
add("DeskewRecv", "deskew_recv", "Deskew and receive (the polling loop at program start)", "C2C", ["C2C_W_RX", "C2C_E_RX"], "C2C", 32, [(31, 28, 5), (27, 22, 9)],
    [CHIP, V("stream", 15, 10, "stream"), LINK], "deskew_recv {chip_id}, {link}, {stream}",
    "The data-dependent polling loop of ISCA22 sec. 3.2 (Fig. 7(b)) as one instruction: the queue is held until the next HAC epoch boundary after the dispatch (cycle mod c2c.hac_period = 0, as Deskew); if a vector has arrived on link `link` by then the instruction is a Receive dispatched at that boundary (same operands, same effect, same diagnostics after the dispatch), else the queue stays held and the vector is looked for again at the following boundary, and so on; no diagnostic is reported while it waits, and a vector that never comes leaves the queue held for the rest of the run (as an unreleased Sync). link >= c2c.links or chip_id not the link's peer is OTS-E-OPR-001, and a stream whose only writers are at or downstream of the C2C queue is OTS-E-DIR-001, each reported at the dispatch (the diagnostics of Receive that do not depend on the arrival) and then the instruction holds nothing and has no effect. The child therefore leaves the loop floor(L / c2c.hac_period) + 1 epochs after the parent's Transmit (ISCA22 sec. 3.2). Implementation choice: the loop as one instruction (the ISA has no branch).")
names = [(d["slice"], d["name"]) for d in INS]
assert len(set(names)) == len(names)
for d in INS:
    with open(os.path.join(OUT, f"{d['slice'].lower()}_{d['name'].lower()}.yaml"), "w") as f:
        f.write("# Instruction catalogue entry. Sail (model/) is normative; test/isa_db/check_isa_db.py\n"
                "# checks this file against the model's encdec and assembly mappings.\n")
        yaml.safe_dump(d, f, sort_keys=False, width=110)
print(len(INS), "files")
