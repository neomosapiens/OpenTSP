"""Negative configuration tests for the emulator .

Every configuration below breaks one invariant the model's types or config/schema.json
state. tsp_sim must refuse each with exit status 2 and a message
"tsp_sim: config <path>: <key>: <reason>" naming the offending key, before the model runs.
The shipped profiles must still load. ARGS: malformed harness arguments are exit 2 with the
usage line.

usage: config_negative.py <tsp_sim> <config.json>
"""
import copy, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from opentsp_reference.schema import causality_errors

sim, base_path = sys.argv[1], sys.argv[2]
base = json.load(open(base_path))
other_wps = 256 if base["sram"]["words_per_slice"] != 256 else 8192
QUADS = base["geometry"]["mem_slices_per_hemisphere"] // 4      # MEM quads per hemisphere


def setk(cfg, key, value):
    *path, last = key.split(".")
    for k in path:
        cfg = cfg[k]
    cfg[last] = value


def delk(cfg, key):
    *path, last = key.split(".")
    for k in path:
        cfg = cfg[k]
    del cfg[last]


# (name, key the message must name, mutation)
CASES = [
    ("d_func negative", "timing.mem.d_func", lambda c: setk(c, "timing.mem.d_func", -3)),
    ("d_func 64", "timing.vxm.d_func", lambda c: setk(c, "timing.vxm.d_func", 64)),
    ("d_skew 40", "timing.vxm.d_skew", lambda c: setk(c, "timing.vxm.d_skew", 40)),
    # The MXM structure keys: the plane count the queue map can name, the accumulator index width, the f16 capacity relation
    # (a check of the model, not of the schema: every harness refuses it with the key), the FP8 rate.
    ("planes_per_hemisphere 0", "mxm.planes_per_hemisphere", lambda c: setk(c, "mxm.planes_per_hemisphere", 0)),
    ("planes_per_hemisphere 3", "mxm.planes_per_hemisphere", lambda c: setk(c, "mxm.planes_per_hemisphere", 3)),
    ("acc_depth 1", "mxm.acc_depth", lambda c: setk(c, "mxm.acc_depth", 1)),
    ("acc_depth 33", "mxm.acc_depth", lambda c: setk(c, "mxm.acc_depth", 33)),
    ("fp8_rate 0", "mxm.fp8_rate", lambda c: setk(c, "mxm.fp8_rate", 0)),
    ("fp8_rate 3", "mxm.fp8_rate", lambda c: setk(c, "mxm.fp8_rate", 3)),
    ("f16_rows below half the rows", "mxm.f16_rows", lambda c: setk(c, "mxm.f16_rows", c["mxm"]["rows"] // 2 - 8)),
    ("f16_rows above the rows", "mxm.f16_rows", lambda c: setk(c, "mxm.f16_rows", c["mxm"]["rows"] + 8)),
    ("f16_rows 12 (not a multiple of 8)", "mxm.f16_rows", lambda c: setk(c, "mxm.f16_rows", 12)),
    ("missing planes_per_hemisphere", "mxm.planes_per_hemisphere", lambda c: delk(c, "mxm.planes_per_hemisphere")),
    ("superlanes 0", "geometry.superlanes", lambda c: setk(c, "geometry.superlanes", 0)),
    ("superlanes 33", "geometry.superlanes", lambda c: setk(c, "geometry.superlanes", 33)),
    ("superlanes 40", "geometry.superlanes", lambda c: setk(c, "geometry.superlanes", 40)),
    ("release_latency negative", "barrier.release_latency", lambda c: setk(c, "barrier.release_latency", -3)),
    ("release_latency 2^70", "barrier.release_latency", lambda c: setk(c, "barrier.release_latency", 2**70)),
    ("bank_bit 20", "sram.bank_bit", lambda c: setk(c, "sram.bank_bit", 20)),
    ("bank_bit of the other profile", "sram.bank_bit",
     lambda c: setk(c, "sram.bank_bit", 12 if c["sram"]["bank_bit"] != 12 else 7)),
    ("words_per_slice of the other profile", "sram.bank_bit", lambda c: setk(c, "sram.words_per_slice", other_wps)),
    ("words_per_slice 100", "sram.words_per_slice", lambda c: setk(c, "sram.words_per_slice", 100)),
    ("words_per_slice 0", "sram.words_per_slice", lambda c: setk(c, "sram.words_per_slice", 0)),
    # Any power of two from 256 to 65536, with bank_bit = log2(words_per_slice) - 1 and at most 2^addr_bits words (addr_bits 13 to 16).
    ("words_per_slice 128", "sram.words_per_slice", lambda c: setk(c, "sram.words_per_slice", 128)),
    ("words_per_slice 384", "sram.words_per_slice", lambda c: setk(c, "sram.words_per_slice", 384)),
    ("words_per_slice 131072", "sram.words_per_slice", lambda c: setk(c, "sram.words_per_slice", 131072)),
    ("words_per_slice 16384 and bank_bit 13 on a 13-bit address", "sram.bank_bit",
     lambda c: (setk(c, "sram.words_per_slice", 16384), setk(c, "sram.bank_bit", 13), setk(c, "sram.addr_bits", 13))),
    ("words_per_slice 65536 and bank_bit 15 on a 15-bit address", "sram.bank_bit",
     lambda c: (setk(c, "sram.words_per_slice", 65536), setk(c, "sram.bank_bit", 15), setk(c, "sram.addr_bits", 15))),
    ("addr_bits 12", "sram.addr_bits", lambda c: setk(c, "sram.addr_bits", 12)),
    ("addr_bits 17", "sram.addr_bits", lambda c: setk(c, "sram.addr_bits", 17)),
    ("addr_bits missing", "sram.addr_bits", lambda c: delk(c, "sram.addr_bits")),
    ("addr_bits 13.5", "sram.addr_bits", lambda c: setk(c, "sram.addr_bits", 13.5)),
    ("words_per_slice 1024 with bank_bit 7", "sram.bank_bit", lambda c: (setk(c, "sram.words_per_slice", 1024), setk(c, "sram.bank_bit", 7))),
    ("words_per_slice 512 with bank_bit 12", "sram.bank_bit", lambda c: (setk(c, "sram.words_per_slice", 512), setk(c, "sram.bank_bit", 12))),
    ("mem_slices_per_hemisphere 0", "geometry.mem_slices_per_hemisphere", lambda c: setk(c, "geometry.mem_slices_per_hemisphere", 0)),
    ("mem_slices_per_hemisphere 2", "geometry.mem_slices_per_hemisphere", lambda c: setk(c, "geometry.mem_slices_per_hemisphere", 2)),
    ("mem_slices_per_hemisphere 6 (not a multiple of 4)", "geometry.mem_slices_per_hemisphere", lambda c: setk(c, "geometry.mem_slices_per_hemisphere", 6)),
    ("mem_slices_per_hemisphere 45", "geometry.mem_slices_per_hemisphere", lambda c: setk(c, "geometry.mem_slices_per_hemisphere", 45)),
    ("mem_slices_per_hemisphere 48", "geometry.mem_slices_per_hemisphere", lambda c: setk(c, "geometry.mem_slices_per_hemisphere", 48)),
    ("mem_slices_per_hemisphere missing", "geometry.mem_slices_per_hemisphere", lambda c: delk(c, "geometry.mem_slices_per_hemisphere")),
    ("mxm.rows 0", "mxm.rows", lambda c: setk(c, "mxm.rows", 0)),
    ("mxm.rows 8", "mxm.rows", lambda c: setk(c, "mxm.rows", 8)),
    ("mxm.rows 17 (not a multiple of 16)", "mxm.rows", lambda c: setk(c, "mxm.rows", 17)),
    ("mxm.rows 321", "mxm.rows", lambda c: setk(c, "mxm.rows", 321)),
    ("mxm.rows 336", "mxm.rows", lambda c: setk(c, "mxm.rows", 336)),
    ("mxm missing", "mxm", lambda c: delk(c, "mxm")),
    ("mxm.acc_bits 64", "mxm.acc_bits", lambda c: setk(c, "mxm.acc_bits", 64)),
    ("mxm.acc_bits 16", "mxm.acc_bits", lambda c: setk(c, "mxm.acc_bits", 16)),
    ("mxm.acc_bits 128 without wide_acc", "extensions", lambda c: setk(c, "mxm.acc_bits", 128)),
    ("wide_acc without mxm.acc_bits 128", "extensions", lambda c: setk(c, "extensions", ["wide_acc"])),
    ("mxm.acc_bits missing", "mxm.acc_bits", lambda c: delk(c, "mxm.acc_bits")),
    ("extensions missing", "extensions", lambda c: delk(c, "extensions")),
    ("extensions not a list", "extensions", lambda c: setk(c, "extensions", {"block_fp8": True})),
    ("extensions entry not a string", "extensions", lambda c: setk(c, "extensions", [1])),
    ("an unknown extension", "extensions", lambda c: setk(c, "extensions", ["block_fp9"])),
    ("an extension listed twice", "extensions", lambda c: setk(c, "extensions", ["block_fp8", "block_fp8"])),
    # the layout relations use the number of quads: the last quad's gaps must keep their order
    ("last MEM quad's gap reaches the C2C receiver", "layout.c2c_rx_gap",
     lambda c: setk(c, "layout.c2c_rx_gap", c["layout"]["mem_up_gap_base"] + 1 - c["geometry"]["mem_slices_per_hemisphere"] // 4)),
    ("vxm_gap 99", "layout.vxm_gap", lambda c: setk(c, "layout.vxm_gap", 99)),
    ("mem_up_gap_base 0 (quads unreachable)", "layout.c2c_rx_gap", lambda c: setk(c, "layout.mem_up_gap_base", 0)),
    ("vxm_gap outside the MEM gaps", "layout.mem_up_gap_base", lambda c: setk(c, "layout.vxm_gap", 10)),
    ("link_latency 0", "c2c.link_latency", lambda c: setk(c, "c2c.link_latency", 0)),
    ("link_latency negative", "c2c.link_latency", lambda c: setk(c, "c2c.link_latency", -100)),
    ("link_latency a string", "c2c.link_latency", lambda c: setk(c, "c2c.link_latency", "5")),
    ("link_latency a fraction", "c2c.link_latency", lambda c: setk(c, "c2c.link_latency", 1.5)),
    ("link_latency missing", "c2c.link_latency", lambda c: delk(c, "c2c.link_latency")),
    ("hac_period 0", "c2c.hac_period", lambda c: setk(c, "c2c.hac_period", 0)),
    ("hac_period negative", "c2c.hac_period", lambda c: setk(c, "c2c.hac_period", -5)),
    # The link bandwidth and the receive FIFO.
    ("cycles_per_vector 0", "c2c.cycles_per_vector", lambda c: setk(c, "c2c.cycles_per_vector", 0)),
    ("cycles_per_vector negative", "c2c.cycles_per_vector", lambda c: setk(c, "c2c.cycles_per_vector", -20)),
    ("cycles_per_vector 2^16", "c2c.cycles_per_vector", lambda c: setk(c, "c2c.cycles_per_vector", 65536)),
    ("cycles_per_vector missing", "c2c.cycles_per_vector", lambda c: delk(c, "c2c.cycles_per_vector")),
    ("rx_depth 0", "c2c.rx_depth", lambda c: setk(c, "c2c.rx_depth", 0)),
    ("rx_depth negative", "c2c.rx_depth", lambda c: setk(c, "c2c.rx_depth", -1)),
    ("rx_depth 2^16", "c2c.rx_depth", lambda c: setk(c, "c2c.rx_depth", 65536)),
    ("rx_depth missing", "c2c.rx_depth", lambda c: delk(c, "c2c.rx_depth")),
    # c2c.links: the link field is 7 bits (c2c_insts.sail c2c_links), N-2 / V-C5-01.
    ("links 0", "c2c.links", lambda c: setk(c, "c2c.links", 0)),
    ("links 129", "c2c.links", lambda c: setk(c, "c2c.links", 129)),
    ("links -1", "c2c.links", lambda c: setk(c, "c2c.links", -1)),
    # Stream-writing classes commit at dispatch + d_func >= dispatch + 1.
    ("mem d_func 0", "timing.mem.d_func", lambda c: setk(c, "timing.mem.d_func", 0)),
    ("vxm d_func 0", "timing.vxm.d_func", lambda c: setk(c, "timing.vxm.d_func", 0)),
    ("c2c d_func 0", "timing.c2c.d_func", lambda c: setk(c, "timing.c2c.d_func", 0)),
    ("mxm_acc d_func 0", "timing.mxm_acc.d_func", lambda c: setk(c, "timing.mxm_acc.d_func", 0)),
    ("sxm_cross d_func 0", "timing.sxm_cross.d_func", lambda c: setk(c, "timing.sxm_cross.d_func", 0)),
    # Link latency relations.
    ("link_latency_min 0", "c2c.link_latency_min", lambda c: setk(c, "c2c.link_latency_min", 0)),
    ("link_latency_max 0", "c2c.link_latency_max", lambda c: setk(c, "c2c.link_latency_max", 0)),
    ("link_latency_max 2^16", "c2c.link_latency_max", lambda c: setk(c, "c2c.link_latency_max", 65536)),
    ("link_latency_min above link_latency", "c2c.link_latency_min", lambda c: setk(c, "c2c.link_latency_min", 500)),
    ("link_latency above link_latency_max", "c2c.link_latency", lambda c: setk(c, "c2c.link_latency", 229)),
    ("link_latency_max not below hac_period", "c2c.link_latency_max",
     lambda c: (setk(c, "c2c.link_latency", 230), setk(c, "c2c.link_latency_max", 300))),
    ("hac_period 100 (below the link latency)", "c2c.link_latency_max", lambda c: setk(c, "c2c.hac_period", 100)),
    ("spec_version 9.9", "spec_version", lambda c: setk(c, "spec_version", "9.9")),
    ("unknown top-level key", "extra", lambda c: setk(c, "extra", 1)),
    ("unknown nested key", "timing.vxm.extra", lambda c: setk(c, "timing.vxm.extra", 1)),
    ("geometry missing", "geometry", lambda c: delk(c, "geometry")),
    # ras.diag_depth may be left out (it then is 16); a given value stays inside 1..64
    ("diag_depth 0", "ras.diag_depth", lambda c: setk(c, "ras.diag_depth", 0)),
    ("diag_depth 65", "ras.diag_depth", lambda c: setk(c, "ras.diag_depth", 65)),
    ("diag_depth negative", "ras.diag_depth", lambda c: setk(c, "ras.diag_depth", -1)),
    ("log_depth missing", "ras.log_depth", lambda c: delk(c, "ras.log_depth")),
]

def override(entry, d_func, d_skew):
    # a row of the profile under test for the same key gives way, so a case meets only its own fault (test/ref_hw has Max.i8)
    def f(c):
        rows = c["timing"]["overrides"]
        rows[:] = [r for r in rows if r["entry"] != entry]
        rows.append({"entry": entry, "d_func": d_func, "d_skew": d_skew})
    return f


def again(entry, d_func, d_skew):
    return lambda c: c["timing"]["overrides"].append({"entry": entry, "d_func": d_func, "d_skew": d_skew})


# timing.overrides: the schema bounds each row; the model checks the
# effective table (postlude/timing_check.sail) and names the offending row's key.
CASES += [
    ("override of an unknown entry", "timing.overrides: Frobnicate is not a catalogue entry", override("Frobnicate", 5, 0)),
    ("override listed twice", "timing.overrides: Add is listed more than once",
     lambda c: (override("Add", 5, 2)(c), again("Add", 6, 2)(c))),
    ("override d_func 64", "timing.overrides", override("Add", 64, 0)),
    ("override d_skew 16", "timing.overrides", override("Add", 5, 16)),
    ("override with an unknown key", "timing.overrides", lambda c: c["timing"]["overrides"].append({"entry": "Add", "d_func": 5, "d_skew": 0, "x": 1})),
    ("override d_skew equal to d_func", "timing.overrides.Add: Add: d_skew 3 must be below d_func 3", override("Add", 3, 3)),
    ("override of ACC with d_skew above d_func", "timing.overrides.ACC: ACC: d_skew 9 must be below d_func 4", override("ACC", 4, 9)),
    ("Write d_skew differs from Read's", "timing.overrides.Write: Write: d_skew 1 must equal Read's", override("Write", 5, 1)),
    ("mem d_skew without IfetchM's", "timing.cu: IfetchM: d_skew 0 must equal Read's 1", lambda c: setk(c, "timing.mem.d_skew", 1)),
    # per-type rows: an override row's entry is "<Entry>.<element type>"
    ("typed override of an entry without an element type", "timing.overrides: Read.i8: Read does not take element type i8", override("Read.i8", 5, 0)),
    ("typed override of a type the entry does not take", "timing.overrides: Max.b8: Max does not take element type b8", override("Max.b8", 5, 2)),
    ("typed override of a transcendental at i32", "timing.overrides: Exp.i32: Exp does not take element type i32", override("Exp.i32", 5, 2)),
    ("typed override of a conversion at its destination type", "timing.overrides: Conv_F16_I8.i8: Conv_F16_I8 does not take element type i8", override("Conv_F16_I8.i8", 5, 2)),
    ("typed override of an unknown type", "timing.overrides: Max.q9 is not a catalogue entry", override("Max.q9", 5, 2)),
    ("typed override with an empty type", "timing.overrides: Max. is not a catalogue entry", override("Max.", 5, 2)),
    ("typed override listed twice", "timing.overrides: Max.i8 is listed more than once",
     lambda c: (override("Max.i8", 3, 2)(c), again("Max.i8", 4, 2)(c))),
    ("typed override d_skew equal to d_func", "timing.overrides.Max.i8: Max.i8: d_skew 2 must be below d_func 2", override("Max.i8", 2, 2)),
]
if base["geometry"]["superlanes"] > 1:
    CASES.append(("Shift override below the superlane count", "timing.overrides.Shift: Shift: d_func",
                  override("Shift", base["geometry"]["superlanes"] - 1, 0)))
    CASES.append(("ABC8 override below the superlane count (: the FP8 ABC reads later superlanes like ABC)", "timing.overrides.ABC8: ABC8: d_func",
                  override("ABC8", base["geometry"]["superlanes"] - 1, 0)))
    # The MXM dot product sums lanes of every superlane, like Shift and Permute.
    CASES.append(("ABC override below the superlane count", "timing.overrides.ABC: ABC: d_func",
                  override("ABC", base["geometry"]["superlanes"] - 1, 0)))
    CASES.append(("mxm_abc d_func below the superlane count", "timing.mxm_abc.d_func",
                  lambda c: setk(c, "timing.mxm_abc.d_func", c["geometry"]["superlanes"] - 1)))
# Shift and Permute read lanes of later superlanes, so their result can commit no earlier than
# dispatch + superlanes. With one superlane every d_func >= 1 satisfies it.
if base["geometry"]["superlanes"] > 1:
    CASES.append(("sxm_cross d_func below the superlane count", "timing.sxm_cross.d_func",
                  lambda c: setk(c, "timing.sxm_cross.d_func", c["geometry"]["superlanes"] - 1)))

# Raw texts that are not configurations at all.
RAW = [
    ("not an object", "(top level)", "[]"),
    ("not JSON", "not valid JSON", "{"),
    ("empty file", "not valid JSON", ""),
    ("duplicate key", "spec_version", None),
]

# Harness arguments: a bad number is exit 2 with the usage line.
ARGS = [
    ("--max-cycles -1", ["--max-cycles", "-1"]),
    ("--max-cycles +5", ["--max-cycles", "+5"]),
    ("--max-cycles ' 5'", ["--max-cycles", " 5"]),
    ("--max-cycles empty", ["--max-cycles", ""]),
    ("--max-cycles 2^64", ["--max-cycles", str(2**64)]),
]

fails = 0
tmp = tempfile.mkdtemp()


def run(text, name, args=()):
    path = os.path.join(tmp, "cfg.json")
    with open(path, "w") as fh:
        fh.write(text)
    p = subprocess.run([sim, path, *args], input="", capture_output=True, text=True, timeout=120)
    return p.returncode, p.stderr


def expect_reject(name, key, text):
    global fails
    rc, err = run(text, name)
    ok = rc == 2 and "config" in err and key in err and err.count("\n") == 1
    print(f"{'ok  ' if ok else 'FAIL'} reject {name}: rc {rc}: {err.strip()[:160]}")
    fails += not ok


rc, err = run(json.dumps(base), "base")
print(f"{'ok  ' if rc == 0 else 'FAIL'} the profile itself loads: rc {rc} {err.strip()[:160]}")
fails += rc != 0

# The text of the message for a range and for a set of constants (the same text in tsp_sim, tsp-iss, ots-asm and tsp_as.py;
# the last two are pinned by ots-asm tests/config_validate.rs and tests/spec/test_harness_contract.py).
for name, mutate, text in [
        ("superlanes 33: the range", lambda c: setk(c, "geometry.superlanes", 33), "geometry.superlanes: 33 is above the maximum 32 (must be between 1 and 32)"),
        ("superlanes 0: the range", lambda c: setk(c, "geometry.superlanes", 0), "geometry.superlanes: 0 is below the minimum 1 (must be between 1 and 32)"),
        ("words_per_slice 100: the set", lambda c: setk(c, "sram.words_per_slice", 100), "sram.words_per_slice: must be one of 256, 512, 1024, 2048, 4096, 8192")]:
    cfg = copy.deepcopy(base)
    mutate(cfg)
    rc, err = run(json.dumps(cfg), name)
    ok = rc == 2 and text in err
    print(f"{'ok  ' if ok else 'FAIL'} reject {name}: rc {rc}: {err.strip()[:160]}")
    fails += not ok

for name, key, mutate in CASES:
    cfg = copy.deepcopy(base)
    mutate(cfg)
    expect_reject(name, key, json.dumps(cfg))

for name, key, text in RAW:
    if text is None:
        body = json.dumps(base)
        text = body[:-1] + ', "spec_version": "0.1"}'
    expect_reject(name, key, text)

for name, args in ARGS:
    rc, err = run(json.dumps(base), name, args)
    ok = rc == 2 and "usage" in err
    print(f"{'ok  ' if ok else 'FAIL'} reject {name}: rc {rc}: {err.strip()[:160]}")
    fails += not ok
# Legal overrides load: one entry slower than its class, a skewed MEM class with IfetchM's
# override, a Shift at exactly the superlane count.
legal = copy.deepcopy(base)
legal["timing"]["mem"]["d_skew"] = 1
legal["timing"]["overrides"] = [{"entry": "Add", "d_func": 9, "d_skew": 3}, {"entry": "IfetchM", "d_func": 0, "d_skew": 1},
                                {"entry": "Shift", "d_func": max(base["geometry"]["superlanes"], 1), "d_skew": 0}]
rc, err = run(json.dumps(legal), "legal overrides")
print(f"{'ok  ' if rc == 0 else 'FAIL'} legal timing.overrides load: rc {rc} {err.strip()[:160]}")
fails += rc != 0
# Legal per-type rows: the entry's row and rows of two of its types side by side, and a type of every kind of typed entry.
legal = copy.deepcopy(base)
typed = ("Max", "Max.i8", "Max.f32", "Add.e4m3", "Exp.f16", "Reduce.u32", "Conv_F16_I8.f16", "Convert.i8", "Cmp.b8")
legal["timing"]["overrides"] = [r for r in legal["timing"]["overrides"] if r["entry"] not in typed]
legal["timing"]["overrides"] += [{"entry": e, "d_func": 9, "d_skew": 3} for e in typed]
rc, err = run(json.dumps(legal), "legal typed overrides")
print(f"{'ok  ' if rc == 0 else 'FAIL'} legal per-type timing.overrides load: rc {rc} {err.strip()[:160]}")
fails += rc != 0
# Every legal value of the new keys loads (words_per_slice with its bank bit, the MEM slice
# counts, the MXM row counts at both ends).
for w in (256, 512, 1024, 2048, 4096, 8192):
    legal = copy.deepcopy(base)
    legal["sram"]["words_per_slice"], legal["sram"]["bank_bit"] = w, w.bit_length() - 2
    rc, err = run(json.dumps(legal), f"words {w}")
    print(f"{'ok  ' if rc == 0 else 'FAIL'} legal words_per_slice {w} (bank_bit {w.bit_length() - 2}) loads: rc {rc} {err.strip()[:160]}")
    fails += rc != 0
for m in (4, 8, 12, 24, 44):
    legal = copy.deepcopy(base)
    legal["geometry"]["mem_slices_per_hemisphere"] = m
    rc, err = run(json.dumps(legal), f"mem {m}")
    print(f"{'ok  ' if rc == 0 else 'FAIL'} legal mem_slices_per_hemisphere {m} loads: rc {rc} {err.strip()[:160]}")
    fails += rc != 0
for n in (16, 32, 160, 304, 320):
    legal = copy.deepcopy(base)
    legal["mxm"]["rows"] = n
    legal["mxm"]["f16_rows"] = n // 2           # the f16 capacity lies between half the rows and all of them
    rc, err = run(json.dumps(legal), f"rows {n}")
    print(f"{'ok  ' if rc == 0 else 'FAIL'} legal mxm.rows {n} loads: rc {rc} {err.strip()[:160]}")
    fails += rc != 0
rc, err = run(json.dumps(base), "--max-cycles 0x10", ["--max-cycles", "0x10"])
print(f"{'ok  ' if rc == 0 else 'FAIL'} accept --max-cycles 0x10: rc {rc} {err.strip()[:160]}")
fails += rc != 0

# =======================================================================================
# The boundary of every x-opentsp-relations entry of config/schema.json.
# For "a < b + o", a config with a = b + o (exactly at the bound) is refused naming a, and
# one with a = b + o - 1 loads; for "a <= b + o", a = b + o loads and a = b + o + 1 is
# refused. When moving a would break a schema bound or another relation, b is moved instead.
schema = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/schema.json")))


def getk(cfg, key):
    for k in key.split("."):
        cfg = cfg[k]
    return cfg


def in_bounds(cfg, node=None):
    """Every integer leaf of cfg inside its schema minimum/maximum."""
    node = node or schema
    if node.get("type") == "object":
        return all(in_bounds(cfg[k], v) for k, v in node.get("properties", {}).items() if isinstance(cfg, dict) and k in cfg)
    if node.get("type") == "integer":
        return node.get("minimum", cfg) <= cfg <= node.get("maximum", cfg)
    return True


def sram_skew(cfg):
    """IfetchM shares the SRAM ports' d_skew with the MEM class (timing_check.sail): give it an
    override when the MEM class's d_skew moves."""
    s = cfg["timing"]["mem"]["d_skew"]
    if s != cfg["timing"]["cu"]["d_skew"]:
        cfg["timing"]["overrides"] = [{"entry": "IfetchM", "d_func": 0, "d_skew": s}]
    return cfg


def eff(x):
    """The offset of relation x for the base profile (the "quads" term of schema_invariants.py)."""
    return x["offset"] + x.get("quads", 0) * QUADS


def at_bound(r, delta):
    """A copy of base with r's left side at r's bound + delta (moving a, else b), or None."""
    edge = 0 if r["op"] == "<=" else -1          # the largest a that satisfies the relation
    # moving a, else b, else both (b to the smallest value that works, a to the bound)
    for move in ["a", "b"] + [("a+b", v) for v in range(0, 64)]:
        cfg = copy.deepcopy(base)
        if move == "a":
            setk(cfg, r["a"], getk(cfg, r["b"]) + eff(r) + edge + delta)
        elif move == "b":
            setk(cfg, r["b"], getk(cfg, r["a"]) - eff(r) - edge - delta)
        else:
            setk(cfg, r["b"], move[1])
            setk(cfg, r["a"], move[1] + eff(r) + edge + delta)
            move = "a and b"
        sram_skew(cfg)
        if "sram.bank_bit" in (r["a"], r["b"]):
            # the bank bit is the top bit of the words: keep the pair of keys legal when the bank bit moves
            cfg["sram"]["words_per_slice"] = min(2 ** (cfg["sram"]["bank_bit"] + 1), 65536)
        if delta == 0 and r["a"] == "geometry.superlanes" and r["b"] == "timing.mxm_abc.d_func":
            # At d_func == superlanes, causality requires zero operand skew.
            cfg["timing"]["mxm_abc"]["d_skew"] = 0
            cfg["timing"]["mxm_abc"]["occupancy_f16"] = min(cfg["timing"]["mxm_abc"]["occupancy_f16"], cfg["timing"]["mxm_abc"]["d_func"])
        if r["a"] in ("timing.mxm_abc.d_skew", "timing.sxm_cross.d_skew"):
            # The generic skew < latency boundary is realizable only with one
            # superlane; larger geometries have a stronger causality bound.
            cfg["geometry"]["superlanes"] = 1
        if delta == 0 and causality_errors(cfg):
            continue
        others = [x for x in schema["x-opentsp-relations"] if x is not r]
        if in_bounds(cfg) and all((getk(cfg, x["a"]) < getk(cfg, x["b"]) + eff(x)) if x["op"] == "<" else
               (getk(cfg, x["a"]) <= getk(cfg, x["b"]) + eff(x)) for x in others):
            return cfg, move
    return None, None


for r in schema["x-opentsp-relations"]:
    name = f"{r['a']} {r['op']} {r['b']} {eff(r):+d}"
    cfg, move = at_bound(r, 0)
    if cfg is None and r["a"].endswith(".d_skew") and at_bound(r, 1)[0] is None:
        # d_skew <= 15 (schema): with this profile's other rules (sxm_cross.d_func >=
        # superlanes = 20) d_skew < d_func always holds, at no reachable bound
        print(f"ok   h6 {name}: implied by d_skew's maximum in this profile"); continue
    if cfg is None:
        print(f"FAIL relation {name}: no single-key config at the bound"); fails += 1; continue
    rc, err = run(json.dumps(cfg), name)
    ok = rc == 0
    print(f"{'ok  ' if ok else 'FAIL'} h6 accept {name} at its bound (moved {move}): rc {rc} {err.strip()[:120]}")
    fails += not ok
    cfg, move = at_bound(r, 1)
    if cfg is None:
        print(f"FAIL relation {name}: no single-key config one past the bound"); fails += 1; continue
    expect_reject(f"{name} one past its bound (moved {move})", r["a"], json.dumps(cfg))

print("config negative tests: " + ("FAILURES" if fails else "all pass"))
sys.exit(1 if fails else 0)
