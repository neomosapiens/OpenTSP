"""Golden conformance vectors produced by the Sail model (the specification), for the ISS
and the assembler to match bit for bit.

Each vector is a full program: SRAM preloads, the text of the queues (`text`: queue ->
the hex 32-bit units of its initial instruction queue, as the harness `Q` lines; the lowering
of the program's cycle view, assembly.md sec. 4), the SRAM words to compare, and the model's
outputs (every diagnostic with cycle and queue, the dumped words, and the trace of stream-register and SRAM
writes with their values: a SHA-256 digest for every vector and the events themselves for the vectors with few of them). Programs come from the directed scenarios (test/directed) and from the
The oracle program generator (test/oracle/diff_oracle.py): one program per template, then random
mixes, and from one minimal program per catalogue entry (coverage-NNN). Everything is
in-tree, so anyone can regenerate the vectors.

The generators build ref-geometry programs. For a smaller profile the preloads and dumps
are fitted to it (words cut to vlanes bytes, addresses wrapped to words_per_slice), and in
the oracle generator's programs the SRAM address field of each record is wrapped too, so they
exercise the instructions instead of reporting OTS-E-OPR-001 throughout. The directed
scenarios keep their instructions: their out-of-range addresses are what they test.

Each vector lists the catalogue instructions and slices its text uses: an instruction (the
text raised to a cycle view, tsp_as.raise_images) uses an entry when it decodes to that entry
under encoding.json decode_rule on a queue that may issue it. --check fails, naming them, on catalogue entries that no vector uses, and on
ref also on entries that no vector runs without a diagnostic at their cycle and queue.

The host path: a vector may carry `host`, the host's script lines as the harness reads them (`H ...`, `R ...`),
and `expect.ext_events`, the model's external events in output order: the `T dma`, `T irq`, `T csr` and `T pkt` lines without
the leading "T ", then the `Y` (egress vector) and `X` (CSR read) lines. They are not part of `trace_sha256`. A vector
without a host script has `ext_events` only when it uses PciSend or HostNotify. The host vectors are the programs of
test/host/host_directed.py, captured as directed_programs captures the directed scenarios.

The matrix-unit check vectors (microarchitecture.md sec. 18) are the programs of test/directed/mxm_check.py, appended after the link FEC ones.
The link FEC vectors (microarchitecture.md sec. 17) are `ecc-NNN` vectors too: the single-chip programs of test/directed/link_fec.py, appended
after the SECDED ones; their `F link` and `R` lines ride in the `host` field and the `T ecc ... link` lines and the `X` lines in `ext_events`.

The ECC vectors (SECDED, microarchitecture.md sec. 15; `ecc-NNN`): the programs of test/directed/ras.py, each with its
fault-injection lines (`F ...`, the same `host` field), and `expect.ext_events` carrying the `T ecc` lines (the corrected and
uncorrectable events) in output order; the trace digest and the dumped words are those of the faulted run.

  gen_model_vectors.py <tsp_sim> <config.json> <out.json> [n-random] [seed]
  gen_model_vectors.py --check <tsp_sim> <config.json> <out.json>   (fail if outputs differ)
  gen_model_vectors.py --extra [--check] <tsp_sim> <config.json> <out.json>   (the side vector set of conformance/model-extra/: one program per
                                                                      (VXM entry, element type), OTS-E-IQ-003 (IfetchM and Ifetch), OTS-E-IQ-002, patrol (scrub.period > 0), fault kinds)
  gen_model_vectors.py --refresh <tsp_sim> <config.json> <out.json> (new config digest and ext_events only; stops if anything else differs)
  gen_model_vectors.py --append <tsp_sim> <config.json> <out.json>  (add the host, timing and FP8 programs the file does not hold yet, at its end;
                                                                      every other vector stays as it is, e.g. the historical random ones)
"""
import hashlib, importlib.util, json, os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import tsp_as                                   # the normative lowering (assembly.md sec. 4)
from opentsp_reference import schema as schema_invariants                        # with_defaults: the configuration digest is over the normalised configuration


def write_vectors(doc, path):
    """The golden vectors as committed: `json.dumps(indent=2)` and a final newline."""
    with open(path, "w") as f:
        f.write(json.dumps(doc, indent=2) + "\n")


def text_json(images):
    """{"<queue>": "<hex units>"}: the units as 8-digit hex words, a space between them."""
    return {str(q): " ".join(f"{u:08x}" for u in us) for q, us in images.items()}


def images_of(text):
    return {int(q): [int(x, 16) for x in us.split()] for q, us in text.items()}


def canonical_trace(lines):
    """The `T srf` and `T sram` lines of a trace in canonical order (microarchitecture.md sec. 12):
    each cycle's run of consecutive `T srf` lines sorted by (row, stream, gap), E before W; all
    other lines keep their order. `T diag` lines are not part of the result (expect.diagnostics has them)."""
    out, run_ = [], []
    def key(ln):
        t = ln.split()                                      # T srf <cycle> <row>:<stream> gap <gap> <value>
        row, st = t[3].split(":")
        return (row, int(st), int(t[5]))
    def flush():
        out.extend(sorted(run_, key=key))
        run_.clear()
    for ln in lines:
        if ln.startswith("T srf "):
            if run_ and run_[0].split()[2] != ln.split()[2]:
                flush()
            run_.append(ln)
        else:
            flush()
            if ln.startswith("T sram "):
                out.append(ln)
    flush()
    return out


def trace_digest(events):
    """SHA-256 (hex) of the canonical `T srf`/`T sram` lines, each followed by a newline."""
    return hashlib.sha256("".join(e + "\n" for e in events).encode()).hexdigest()


EVENTS_MAX = 64     # a vector with at most this many trace events stores them all; every vector stores the digest


def expect_trace(events):
    """The trace part of a vector's `expect`: the SHA-256 of the canonical events,
    and the events themselves when there are at most EVENTS_MAX of them."""
    out = {"trace_sha256": trace_digest(events)}
    if len(events) <= EVENTS_MAX:
        out["events"] = [e[2:] for e in events]          # without the leading "T "
    return out


def config_digest(cfg):
    """SHA-256 (hex) of the profile's configuration as canonical JSON (sorted keys, no spaces), with the defaults of omitted keys inserted
    (schema_invariants.with_defaults): an implementation checks that it runs the profile the vectors were generated from."""
    return hashlib.sha256(json.dumps(schema_invariants.with_defaults(json.load(open(cfg))), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Inapplicable(SystemExit):
    """tsp_sim rejects the program as input (exit code 2). Uncaught it ends the generator as before; for the captured scenarios
    (written for the ref geometry) on a profile with fewer than 44 MEM slices per hemisphere, main() and append_host() skip the
    scenario and say so: it names a slice the profile does not have."""


def small_profile(cfg):
    return json.load(open(cfg))["geometry"]["mem_slices_per_hemisphere"] < 44


def run(sim, cfg, images, sram, dumps, host=()):
    """Run the text images {queue: units} on tsp_sim. Returns (diagnostics, words, events): the
    events are the canonical `T srf` and `T sram` lines with their values."""
    return run_ext(sim, cfg, images, sram, dumps, host)[:3]


def run_ext(sim, cfg, images, sram, dumps, host=()):
    """As run, with a fourth result: the external events (module docstring) of the host path."""
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    lines += tsp_as.q_lines(images) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(host)
    res = subprocess.run([sim, cfg, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    if res.returncode == 2:                              # an input error: the program names hardware this profile does not have
        raise Inapplicable(f"gen_model_vectors: tsp_sim exit code 2: {res.stderr.strip()}")
    if res.returncode != 0:
        raise SystemExit(f"gen_model_vectors: tsp_sim exit code {res.returncode}: {res.stderr.strip()}")
    out = res.stdout
    vb = 16 * json.load(open(cfg))["geometry"]["superlanes"]
    diags, words, ext, tail = [], {}, [], []
    for ln in out.splitlines():
        t = ln.split()
        if ln.startswith("T diag"):
            diags.append({"cycle": int(t[2]), "queue": int(t[4]), "code": t[5]})
        elif t and t[0] == "W":
            # the harness always prints 320 lanes; a profile's word is its vlanes bytes
            words[f"{int(t[1], 16)}:{int(t[2], 16)}"] = bytes(int(x, 16) for x in t[3:])[:vb].hex()
        elif t[:2] in (["T", "dma"], ["T", "irq"], ["T", "csr"], ["T", "pkt"], ["T", "ecc"]):
            ext.append(ln[2:])
        elif t and t[0] in ("Y", "X"):
            tail.append(ln.rstrip())
    return diags, words, canonical_trace(out.splitlines()), ext + tail


def ext_part(ext):
    """The `ext_events` entry of a vector's `expect` (empty: no entry)."""
    return {"ext_events": ext} if ext else {}


def host_programs(sim, cfg):
    """The programs of test/host/host_directed.py: (records, sram, dumps, units, host lines). Captured by
    running the test with a recording run(); only runs on the profile itself and without options are kept."""
    progs = []
    path = os.path.join(SPEC, "test", "host", "host_directed.py")
    src = open(path).read()
    src = src.replace("SIM = sys.argv[1]; CFG = sys.argv[2]", f"SIM = {sim!r}; CFG = {cfg!r}")
    src = src.replace("    ls = [f\"S {sl:x}", "    if cfg_path is None and not args:\n        _PROGS.append((list(recs), dict(sram), list(dumps), list(host)))\n    ls = [f\"S {sl:x}", 1)
    src = src.replace("sys.exit(0 if ok else 1)", "pass")
    g = {"_PROGS": progs, "__name__": "host_directed", "__file__": path}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, "exec"), g)
    if not progs:
        raise SystemExit("gen_model_vectors: no program captured from test/host/host_directed.py")
    return [(r, sm, d, {}, h) for r, sm, d, h in progs]


def ecc_programs(sim, cfg):
    """The programs of test/directed/ras.py, test/directed/ras_csr.py, test/directed/ras_poison.py and test/directed/ras_spare.py, test/directed/ras_diag.py and test/directed/ras_bank.py (the
    Gather, Scatter and bank-conflict error reports in the spare bits, which need no fault line; their `R` lines read the error registers): (records, sram, dumps, units,
    fault, register and host lines). Captured by running the tests with a recording run(); the lines ride in the `host` field of the
    vector (`F`, `R` and `H csr` lines). ras_csr.py programs run with options are not kept; ras_poison.py (MXM and C2C poison)
    has no dumps and ends its programs with a VXM consumer, so the poison shows in the trace."""
    progs = []
    import contextlib, io
    for name, lines_var, hook_if in (("ras.py", "faults", ""), ("ras_csr.py", "extra", "if not opts and expect == 0:\n        "), ("ras_poison.py", "extra", ""), ("ras_spare.py", "extra", ""), ("ras_bank.py", "extra", ""), ("ras_diag.py", "extra", "")):
        path = os.path.join(SPEC, "test", "directed", name)
        src = open(path).read()
        src = src.replace("SIM = sys.argv[1]; CFG = sys.argv[2]", f"SIM = {sim!r}; CFG = {cfg!r}")
        old = "    ls = [f\"S {sl:x}"
        assert old in src
        dumps_expr = "[]" if name == "ras_poison.py" else "list(dumps)"
        src = src.replace(old, f"    {hook_if}_PROGS.append((list(recs), dict(sram), {dumps_expr}, list({lines_var})))\n" + old, 1)
        src = src.replace("sys.exit(0 if ok else 1)", "pass")
        g = {"_PROGS": progs, "__name__": name[:-3], "__file__": path}
        n0 = len(progs)
        skipped = False
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(src, path, "exec"), g)
            except SystemExit as e:                  # a test that declares itself not applicable to this profile exits 0 ("PASS all (skipped: ...)")
                if e.code not in (0, None):
                    raise
                skipped = True
        if len(progs) == n0 and not skipped:
            raise SystemExit(f"gen_model_vectors: no program captured from test/directed/{name}")
    return [(r, sm, d, {}, h) for r, sm, d, h in progs]


def link_fec_programs(sim, cfg, name="link_fec"):
    """The single-chip programs of test/directed/link_fec.py (link FEC, microarchitecture.md sec. 17), or, with name="mxm_check", of
    test/directed/mxm_check.py (matrix-unit error detection and correction, sec. 18): (records, sram, dumps, units, host
    lines). The test records the input lines of each of its runs; the `S`, `Q` and `D` lines become the vector's preloads, text
    units and dumps, the `F` and `R` lines its `host` field. A program the file already holds (the fault-free one that every run of
    the test starts from) is kept once."""
    path = os.path.join(SPEC, "test", "directed", name + ".py")
    src = open(path).read()
    src = src.replace("SIM = sys.argv[1]; CFG = sys.argv[2]", f"SIM = {sim!r}; CFG = {cfg!r}")
    src = src.replace("sys.exit(0 if ok else 1)", "pass")
    g = {"__name__": name, "__file__": path}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(src, path, "exec"), g)
        except SystemExit as e:                      # the test declares itself not applicable to this profile: exit 0, no program
            if e.code not in (0, None):
                raise
            return []
    progs, seen = [], set()
    for lines in g["_PROGS"]:
        sram, dumps, units, host = {}, [], {}, []
        for ln in lines:
            t = ln.split()
            if t[0] == "S":
                sram[(int(t[1], 16), int(t[2], 16))] = bytes(int(x, 16) for x in t[3:])
            elif t[0] == "Q":
                units.setdefault(int(t[1], 16), []).extend(int(x, 16) for x in t[2:])
            elif t[0] == "D":
                dumps.append((int(t[1], 16), int(t[2], 16)))
            else:
                host.append(ln)
        k = json.dumps([sorted(units.items()), sorted((k, v.hex()) for k, v in sram.items()), dumps, host])
        if k not in seen:
            seen.add(k)
            progs.append(([], sram, dumps, units, host))
    if not progs:
        raise SystemExit(f"gen_model_vectors: no program captured from test/directed/{name}.py")
    return progs


def wide_programs(sim, cfg):
    """Programs of the profile extensions wide_acc (ACC x128) and block_fp8 (the MXM type bfp8), only for a profile that declares both (the test
    profiles mini_ext, ref_ext and all_ext; none otherwise, so that no golden vector of ref or mini changes): the programs of
    test/directed/ext_types.py (ACC x128 after an f16 ABC, bfp8 matrix products with ACC f32 and ACC x128 on random data and scales, ACC x128
    after an integer ABC), captured by a recording run() as bf16_programs captures bf16.py."""
    if not {"wide_acc", "block_fp8"} <= set(json.load(open(cfg)).get("extensions", [])):
        return []
    path = os.path.join(SPEC, "test", "directed", "ext_types.py")
    src = open(path).read()
    src = src.replace("SIM = sys.argv[1]; CFG = sys.argv[2]", f"SIM = {sim!r}; CFG = {cfg!r}")
    src = src.replace("def run(recs, sram={}, dumps=[]):", "def run(recs, sram={}, dumps=[]):\n    _PROGS.append((list(recs), dict(sram), list(dumps), {}))", 1)
    progs = []
    g = {"_PROGS": progs, "__name__": "ext_types", "__file__": path}
    try:
        exec(compile(src, path, "exec"), g)
    except SystemExit as stop:                         # the script ends with sys.exit(0 or 1)
        if stop.code:
            raise
    return progs


def timing_programs(sim, cfg):
    """Programs whose conformance vector pins a published timing rule that no other vector exercises: four back-to-back
    f16 ABCs, so that the vector's diagnostics show the configured MACC occupancy (1 in the shipped profiles: all four are
    accepted; research profiles may reserve the paired matrix longer). Captured from test/directed/timing_rows.py::burst by a recording run(), as
    host_programs captures the host programs; only the profile itself is run."""
    path = os.path.join(SPEC, "test", "directed", "timing_rows.py")
    src = open(path).read()
    src = src.replace("SIM, CFG = sys.argv[1], sys.argv[2]", f"SIM, CFG = {sim!r}, {cfg!r}")
    src = src.split("\ntest_map_ready()\n")[0]                    # the definitions, not the test calls
    src = src.replace("def run(cfg, recs, sram={}):", "def run(cfg, recs, sram={}):\n    if cfg == CFG:\n        _PROGS.append((list(recs), dict(sram), [], {}))", 1)
    progs = []
    g = {"_PROGS": progs, "__name__": "timing_rows", "__file__": path}
    exec(compile(src, path, "exec"), g)
    g["burst"](cfg, 1, 4)                                          # f16 type code 1, four ABCs on consecutive cycles
    if len(progs) != 1:
        raise SystemExit("gen_model_vectors: no program captured from test/directed/timing_rows.py")
    return progs


def cr27_programs(cfg):
    """The programs that pin the repeated-IfetchM rule and the report retention in the vectors (records, sram, dumps, units): a repeated IfetchM against the explicit
    second IfetchM it must equal (, test/directed/frontend.py section 16: a Read after it, a Write of the second beat's bank, two
    iterations, overlapping iterations), and the report-retention pair of test/directed/ras_spare.py section 5 (: 11 and 12
    cycles of unrelated reports, so that the report of cycle 100 is observed or evicted). Written for the ref geometry, fitted by fit()."""
    isa = tsp_as.ISA("ref")
    w = lambda text: isa.parse(text)[1]
    vl = 16 * json.load(open(os.path.join(SPEC, "config", "ref.json")))["geometry"]["superlanes"]
    a = 10
    fw = {(0, a): bytes([0xFF] * vl), (0, a + 1): bytes([0xFF] * vl)}                 # fetched text that is only padding
    ifm, rep = w(f"ifetchm {a:#x}"), lambda n, d: w(f"repeat {n:#x}, {d:#x}")
    read, write = w("read 0x20, E:7, 0x0"), w(f"write {a + 1:#x}, E:5, 0x0")
    prog = lambda *rs: ([(c, 0, 1, x, 0) for c, x in rs], dict(fw), [], {})
    out = [prog((0, ifm), (2, ifm), (3, read)), prog((0, ifm), (2, rep(1, 1)), (3, read)),
           prog((0, ifm), (2, ifm), (3, write)), prog((0, ifm), (2, rep(1, 1)), (3, write)),
           prog((0, ifm), (2, ifm), (4, ifm), (5, read)), prog((0, ifm), (2, rep(2, 1)), (5, read)),
           prog((0, ifm), (2, rep(2, 0)), (3, read))]
    mem = lambda op, ad, st: (1 << 60) | (op << 54) | (ad << 32) | (st << 26)
    for n in (11, 12):
        recs = [(100, 40, 1, mem(0, 0, 2), 0), (100, 40, 1, mem(1, 0, 6), 0), (100, 41, 1, mem(0, 0, 2), 0)]
        recs += [(101 + k, 40, 1, w_, 0) for k in range(n - 1) for w_ in (mem(0, 0, 34), mem(1, 0, 38))]
        recs += [(101 + k, q, 1, w_, 0) for q in range(0, 88, 4) if q != 40 for k in range(n) for w_ in (mem(0, 0, 34), mem(1, 0, 38))]
        recs.append((116, 88, 1, w("add.i8 E:8, E:2, E:2"), 0))
        out.append((recs, {}, [(44, 9)], {}))
    return out


def fp8_programs(sim, cfg):
    """Programs of the FP8 element types e4m3 and e5m2 (; microarchitecture.md sec. 5 and 6): one program per operation on the
    interesting codes of each format (zeros, the smallest subnormal, 1, the largest finite value, infinity and NaN, both signs), the
    conversions (f16, f32, i8, the other format), Reduce, an FP8 matrix product in each format (random finite values and the
    cancellation a float32 accumulation would lose), the refused type combination, and the codes that stay illegal. Captured from the
    definitions of test/directed/fp8.py by a recording run(), as timing_programs captures timing_rows.py; each is a small
    ref-geometry program (the full 65536-pair runs stay in fp8.py). Run on the profile `cfg`, which must have the ref geometry."""
    path = os.path.join(SPEC, "test", "directed", "fp8.py")
    src = open(path).read().split("# 1. binary operations")[0]
    src = src.replace("SIM = sys.argv[1]; CFG = sys.argv[2]", f"SIM = {sim!r}; CFG = {cfg!r}")
    src = src.replace("def run(recs, sram={}, dumps=[]):", "def run(recs, sram={}, dumps=[]):\n    _PROGS.append((list(recs), dict(sram), list(dumps), {}))", 1)
    progs = []
    g = {"_PROGS": progs, "__name__": "fp8", "__file__": path}
    exec(compile(src, path, "exec"), g)
    vv, vop, chunks, TY, VL, mxm_run, illegal, mxm = (g[k] for k in ("vxm_vectors", "vxm_op", "chunks", "TY", "VL", "mxm_run", "illegal", "mxm"))
    codes = {"e4m3": [0x00, 0x01, 0x07, 0x08, 0x30, 0x38, 0x3C, 0x77, 0x7E, 0x7F, 0x80, 0x81, 0xB8, 0xC0, 0xF7, 0xFE, 0xFF],
             "e5m2": [0x00, 0x01, 0x03, 0x04, 0x3C, 0x40, 0x7B, 0x7C, 0x7D, 0x7E, 0x80, 0x81, 0xBC, 0xFB, 0xFC, 0xFE, 0xFF]}
    pairs = {t: [(a, b) for a in c for b in c] for t, c in codes.items()}
    def two(t): return [(1, chunks([a for a, _ in pairs[t]])), (1, chunks([b for _, b in pairs[t]]))]
    for t, op in (("e4m3", 0), ("e5m2", 3), ("e4m3", 6), ("e5m2", 6), ("e4m3", 9), ("e5m2", 10)):          # add, sub, mul, max, min
        vv(lambda k, s, dst, op=op, t=t: vop(op, t, dst, s[0], s[1]), two(t), 1, 1)
    for t, mode in (("e4m3", 2), ("e5m2", 0)):
        vv(lambda k, s, dst, mode=mode, t=t: vop(19, t, dst, s[0], s[1], mode=mode), two(t), 1, 1)
    all256 = [(1, chunks(list(range(256))))]
    for t, op in (("e4m3", 15), ("e5m2", 16), ("e4m3", 20)):                                               # neg, abs, relu
        vv(lambda k, s, dst, op=op, t=t: vop(op, t, dst, s[0]), all256, 1, len(all256[0][1]))
    vv(lambda k, s, dst: vop(17, "e4m3", dst, s[0], b=s[1]), two("e4m3"), 1, 1)                           # mask
    vv(lambda k, s, dst: vop(18, "e5m2", dst, s[0], b=s[1], m=s[2]), two("e5m2") + [(1, chunks([a for a, _ in reversed(pairs["e5m2"])]))], 1, 1)   # select
    rnd = random.Random(86)
    a4, b4 = [[rnd.getrandbits(32) for _ in range(VL)]], [[rnd.getrandbits(32) for _ in range(VL)]]
    vv(lambda k, s, dst: vop(0, "e4m3", dst, s[0], s[1], g=1), [(4, a4), (4, b4)], 4, 1)                  # add.e4m3.sg4
    f16s = list(range(0, 65536, 103)) + [0x7C00, 0xFC00, 0x7E00, 0x5F40, 0x5F41, 0x5B80, 0x1400, 0x1401, 0x0001, 0x8001]
    f32s = [rnd.getrandbits(32) for _ in range(500)] + [0x43E00000, 0x43E80000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x37800000, 0x3B000000, 0x3A800000, 0x47600000, 0x47700000]
    def conv(st, dt, vals):
        vs = chunks(vals)
        vv(lambda k, s, dst: vop(34, st, dst, s[0], mode=TY[dt]), [(g["nx"].WIDTH[st], vs)], g["nx"].WIDTH[dt], len(vs))
    conv("f16", "e4m3", f16s); conv("f16", "e5m2", f16s); conv("f32", "e4m3", f32s); conv("f32", "e5m2", f32s)
    for st, dt in (("e4m3", "f16"), ("e5m2", "f32"), ("e4m3", "e5m2"), ("e5m2", "e4m3"), ("i8", "e4m3"), ("e5m2", "i8"), ("e4m3", "b8")):
        conv(st, dt, list(range(256)))
    for t in ("e4m3", "e5m2"):                                                                               # Reduce add over groups of 16 lanes
        vals = [codes[t][rnd.randrange(len(codes[t]))] for _ in range(VL)]
        vv(lambda k, s, dst, t=t: vop(33, t, dst, s[0], mode=0), [(1, [vals])], 1, 1)
    nx = g["nx"]
    for wfmt, t in ((0, "e4m3"), (1, "e5m2")):
        fin = [c for c in range(256) if nx.fclass(nx.fmt(t), c)[0] in ("fin", "zero")]
        rows = [[rnd.choice(fin) for _ in range(VL)] for _ in range(16)]
        mxm_run(wfmt, rows, [rnd.choice(fin) for _ in range(VL)])
        big, one = (0x7E, 0x38) if t == "e4m3" else (0x7B, 0x3C)
        rows = [[big, big, 0x01] + [0] * (VL - 3) for _ in range(16)]
        mxm_run(wfmt, rows, [big, big | 0x80, one] + [0] * (VL - 3))
    mxm_run(0, [[0] * VL] * 16, [0] * VL, abc_word=mxm(2, 0, 32 | 24))                                      # abc.i8 on fp8 weights: OTS-E-RES-001
    base = vop(0, "i8", 24, 0, 8)
    illegal((base & ~((3 << 52) | (3 << 22))) | (1 << 52) | (3 << 22))                                       # type code 13
    illegal(vop(11, "e4m3", 24, 0, 8)); illegal(vop(21, "e4m3", 24, 0))                                     # and.e4m3, tanh.e4m3
    illegal(mxm(2, 0, 32 | 24, 1), 105); illegal(mxm(0, 3, 32 | 0, 1), 105)                                  # abc type 0 with bit 13, lw.fp8 with bit 13
    return progs


def bf16_programs(sim, cfg):
    """Programs of the profile extension bf16 (microarchitecture.md sec. 5 and 6), only for a profile that declares it (the
    test profile ext_bf16; none otherwise, so that no golden vector of ref or mini changes): one program per operation on the interesting
    codes (zeros, the smallest subnormal, 1, the largest finite value, infinity and NaN, both signs), the conversions, Reduce, the
    matrix product W (64 x 320) . x, its cancellation and special-value cases, and the codes that stay illegal. Captured from the
    definitions of test/directed/bf16.py by a recording run(), as fp8_programs captures fp8.py."""
    if "bf16" not in json.load(open(cfg)).get("extensions", []):
        return []
    path = os.path.join(SPEC, "test", "directed", "bf16.py")
    src = open(path).read().split("rnd = random.Random(1633)")[0]
    src = src.replace("SIM = sys.argv[1]; CFG = sys.argv[2]", f"SIM = {sim!r}; CFG = {cfg!r}")
    src = src.replace("def run(recs, sram={}, dumps=[]):", "def run(recs, sram={}, dumps=[]):\n    _PROGS.append((list(recs), dict(sram), list(dumps), {}))", 1)
    progs = []
    g = {"_PROGS": progs, "__name__": "bf16", "__file__": path}
    exec(compile(src, path, "exec"), g)
    vv, vop, chunks, VL, matvec, illegal, mxm, mxm_bf, conv = (g[k] for k in ("vxm_vectors", "vxm_op", "chunks", "VL", "matvec", "illegal", "mxm", "mxm_bf", "conv"))
    codes = [0x0000, 0x0001, 0x007F, 0x0080, 0x3B80, 0x3F80, 0x3FC0, 0x4000, 0x7F7F, 0x7F80, 0x7FC0, 0x8000, 0x8001, 0xBF80, 0xFF7F, 0xFF80]
    pairs = [(a, b) for a in codes for b in codes]
    two = [(2, chunks([a for a, _ in pairs])), (2, chunks([b for _, b in pairs]))]
    for op in (0, 3, 6, 9, 10):                                                                             # add, sub, mul, max, min
        vv(lambda k, s, dst, op=op: vop(op, "bf16", dst, s[0], s[1]), two, 2, len(two[0][1]))
    for mode in (0, 2, 5):
        vv(lambda k, s, dst, mode=mode: vop(19, "bf16", dst, s[0], s[1], mode=mode), two, 2, len(two[0][1]))
    allv = [(2, chunks(codes))]
    for op in (15, 16, 20):                                                                                 # neg, abs, relu
        vv(lambda k, s, dst, op=op: vop(op, "bf16", dst, s[0]), allv, 2, 1)
    vv(lambda k, s, dst: vop(17, "bf16", dst, s[0], b=s[1]), two, 2, len(two[0][1]))                         # mask
    vv(lambda k, s, dst: vop(18, "bf16", dst, s[0], b=s[1], m=s[2]), two + [(2, chunks([a for a, _ in reversed(pairs)]))], 2, len(two[0][1]))   # select
    rnd = random.Random(1633)
    a2, b2 = [[rnd.getrandbits(32) for _ in range(VL)]], [[rnd.getrandbits(32) for _ in range(VL)]]
    vv(lambda k, s, dst: vop(6, "bf16", dst, s[0], s[1], g=1), [(4, a2), (4, b2)], 4, 1)                    # mul.bf16.sg4
    f16s = list(range(0, 65536, 103)) + [0x7C00, 0xFC00, 0x7E00, 0x0001, 0x8001]
    f32s = [rnd.getrandbits(32) for _ in range(500)] + [0x3F808000, 0x3F818000, 0x3F808001, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00010000, 0x00008000]
    conv("f32", "bf16", f32s); conv("f16", "bf16", f16s); conv("i8", "bf16", list(range(256)))
    for dt in ("f32", "f16", "i8", "u8", "i32", "b16", "e4m3"):
        conv("bf16", dt, list(range(0, 65536, 97)) + codes)
    vals = [codes[rnd.randrange(len(codes))] for _ in range(VL)]
    vv(lambda k, s, dst: vop(33, "bf16", dst, s[0], mode=0), [(2, [vals])], 2, 1)                           # reduce add
    def rand_bf(lo=-30, hi=30):
        e = rnd.randint(lo, hi)
        return (rnd.getrandbits(1) << 15) | ((127 + e) << 7) | rnd.getrandbits(7)
    matvec([[rand_bf() for _ in range(VL)] for _ in range(64)], [rand_bf() for _ in range(VL)])           # the proof-of-concept W (64 x 320) . x
    matvec([[rand_bf(-126, 127) for _ in range(VL)] for _ in range(16)], [rand_bf(-126, 127) for _ in range(VL)])
    rows = [[0x7F00, 0x3F80, 0xFF00] + [0] * (VL - 3) for _ in range(16)]
    matvec(rows, [0x3F80, 0x3F80, 0x3F80] + [0] * (VL - 3))                                                 # 2^127 + 1 - 2^127 = 1
    rows = [[rand_bf() for _ in range(VL)] for _ in range(16)]
    rows[0][5], rows[1][5], rows[2][5] = 0x7FC0, 0xFF80, 0x7F80
    matvec(rows, [rand_bf() if k != 5 else 0 for k in range(VL)])                                           # NaN, -inf, inf times zero
    matvec([[0] * VL] * 16, [0] * VL, abc_word=mxm(2, 1, 32 | 24))                                          # abc.f16 on bf16 weights: OTS-E-RES-001
    matvec([[0] * VL] * 16, [0] * VL, wtype_h=0, abc_word=mxm_bf(2, 32 | 24))                               # abc.bf16 on i8 weights
    base = vop(0, "bf16", 24, 0, 8)
    illegal(base | (1 << 52)); illegal(base | (1 << 22))                                                    # codes 17 and 20
    illegal(vop(11, "bf16", 24, 0, 8)); illegal(vop(21, "bf16", 24, 0))                                    # and.bf16, tanh.bf16
    illegal(mxm(0, 1, 32 | 0, 0, 0, 1), 105); illegal(mxm_bf(2, 32 | 24, 1), 105)                          # lw type 5, abc.bf16 with bit 13
    return progs


def directed_cfg(cfg):
    """The profile the directed and FP8 scenarios are captured on: the profile itself when it has the ref geometry (20 superlanes, 44 MEM
    slices per hemisphere: ref, ref_slow, ...), else ref (the scenarios use slices up to 44 and 320-byte words), and fit() cuts the
    captured programs to the profile."""
    g = json.load(open(cfg))["geometry"]
    return cfg if (g["superlanes"], g["mem_slices_per_hemisphere"]) == (20, 44) else os.path.join(SPEC, "config", "ref.json")


def directed_programs(sim, cfg):
    """Capture the programs the directed test builds, by running it with a recording run().
    It runs against the real simulator: its later scenarios use the results of earlier ones.
    Its checks may fail on a profile it was not written for; only the programs are kept. A
    crash is an error, since it would silently drop every later scenario."""
    progs = []
    spec = importlib.util.spec_from_file_location("directed", os.path.join(SPEC, "test", "directed", "directed.py"))
    src = open(spec.origin).read()
    src = src.replace("SIM=sys.argv[1]; CFG=sys.argv[2]", f"SIM={sim!r}; CFG={cfg!r}")
    src = src.replace("def run(recs, sram={}, dumps=[]):", "def run(recs, sram={}, dumps=[]):\n    _PROGS.append((list(recs), dict(sram), list(dumps)))")
    src = src.replace("sys.exit(0 if ok else 1)", "pass")
    g = {"_PROGS": progs, "__name__": "directed", "__file__": spec.origin}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):   
        exec(compile(src, spec.origin, "exec"), g)
    return progs


def random_programs(n, seed, cfg):
    """Programs of the oracle generator (test/oracle/diff_oracle.py): its directed templates,
    then n random mixes, timed by the oracle machine of the given profile (its operand
    widths and addresses stay those of ref; fit() adapts the preloads and dumps)."""
    sys.path.insert(0, os.path.join(SPEC, "test"))
    sys.path.insert(0, os.path.join(SPEC, "test", "oracle"))
    import diff_oracle as D
    from oracle import machine as mc
    m = mc.Machine(json.load(open(cfg)))
    rnd = random.Random(seed)
    kinds = [[k] for k in sorted(set(D.DIRECTED))] + [D.kinds_for(rnd, i, n) for i in range(n)]
    out = []
    for i, k in enumerate(kinds):
        p = D.build(m, seed * 1000003 + i, k)
        recs = [(c, 0xFFFF if q == 144 else q, wc, w0, w1) for (c, q, wc, w0, w1) in p.recs]
        out.append((recs, dict(p.pre), sorted(p.dump), dict(p.units)))
    return out


def link_programs(cfg):
    """Three programs on the link limits, timed from the profile: Transmits around
    c2c.cycles_per_vector (LNK-001, also from the second TX queue in one cycle); rx_depth + 1
    Transmits whose last vector arrives at a full FIFO (LNK-002) with a Receive in its arrival
    cycle (dropped) and one a cycle before (not dropped)."""
    c = json.load(open(cfg))["c2c"]
    cpv, depth, lat = c["cycles_per_vector"], c["rx_depth"], c["link_latency"]
    WTX, WTX1, ERX = 126, 127, 142

    def tx(t, q=WTX, link=0): return (t, q, 1, (5 << 28) | (3 << 22) | (link << 6), 0)
    def rx(t, link=0, s=33): return (t, ERX, 1, (5 << 28) | (1 << 22) | (s << 10) | (link << 6), 0)
    T = 100
    busy = [tx(T), tx(T + cpv - 1), tx(T + cpv), tx(T + cpv, WTX1), tx(T + cpv, WTX1, 1), rx(T + lat), rx(T + lat + cpv)]
    flood = [tx(T + k * cpv) for k in range(depth + 1)]
    arr = T + depth * cpv + lat
    return [(busy, {}, [], {}), (flood + [rx(arr), rx(arr + 1)], {}, [], {}), (flood + [rx(arr - 1), rx(arr + 1)], {}, [], {})]


def recs_of_text(images):
    """The cycle view (cycle, queue, 1, word, 0) of text images (tsp_as.raise_images)."""
    return [(c, q, 1, w, 0) for c, q, w, _ in tsp_as.raise_images(images)]


class Catalogue:
    """The text decode of encoding.json decode_rule, with the placement check."""

    def __init__(self, cfg):
        sys.path.insert(0, os.path.join(SPEC, "test"))
        from oracle import isa
        c = json.load(open(cfg))
        self.vlanes = 16 * c["geometry"]["superlanes"]
        self.isa = isa.ISA(vlanes=self.vlanes, addr_bits=c["sram"]["addr_bits"], acc_depth=c["mxm"]["acc_depth"], extensions=c.get("extensions", []))
        enc = json.load(open(isa.CATALOGUE))
        self.units, self.control = enc["units"], enc["control_channel"]
        self.narrow = enc["decode_rule"]["narrow_tags"]
        self.wide = enc["decode_rule"]["wide_tags"]

    def queues(self, e):
        return list(range(144)) if "any" in e.units else [q for u in e.units for q in self.units[u]]

    def entry(self, rec):
        """The catalogue entry the cycle-view word decodes to (by the lowering: one unit for a narrow tag
        with zero upper bits, two for a wide tag) and may issue on, or None."""
        _, q, wc, w0, w1 = rec
        if wc != 1 or w1 != 0 or (q > 143 and q != self.control["queue"]):
            return None
        ctrl = q == self.control["queue"]
        upper, tag32 = w0 >> 32, (w0 >> 28) & 15
        if not upper and tag32 in self.narrow:
            d = self.isa.decode32(w0)
        elif (w0 >> 60) in self.wide:
            d = self.isa.decode64(w0)
        else:
            d = None
        if d is None:
            return None
        e = self.isa.by_name[d[0]]
        placed = d[0] in self.control["instructions"] if ctrl else q in self.queues(e)
        return e if placed else None

    def tags(self, recs):
        es = [e for e in map(self.entry, recs) if e]
        return sorted({e.name for e in es}), sorted({e.slice for e in es})

    def word(self, e, rnd):
        """A word that decodes to entry e in this profile."""
        for _ in range(4000):
            f = {}
            for name, hi, lo, kind, ex in e.vars:
                pool = ex.get("values") or ex.get("codes")
                if pool:
                    f[name] = rnd.choice(pool)
                    continue
                top = min((1 << (hi - lo + 1)) - 1, ex.get("max", 7), 7)
                f[name] = rnd.randint(min(ex.get("min", 0), top), top)
            w = e.encode(**f)
            d = self.isa.decode32(w) if e.width == 32 else self.isa.decode64(w)
            if d and d[0] == e.name:
                return w
        raise SystemExit(f"gen_model_vectors: no legal word found for {e.name}")

    def coverage_programs(self, seed):
        rnd = random.Random(seed)
        return [([(10, self.queues(e)[0], 1, self.word(e, rnd), 0)], {}, [], {}) for e in self.isa.entries]


def wrap_addr(cat, rec, words):
    """The record with its SRAM address field reduced modulo words_per_slice."""
    e = cat.entry(rec)
    for _, hi, lo, kind, _ex in (e.vars if e else []):
        if kind == "sram_addr":
            f = (1 << (hi - lo + 1)) - 1
            a = (rec[3] >> lo) & f
            return (*rec[:3], (rec[3] & ~(f << lo)) | ((a % words) << lo), rec[4])
    return rec


def fit(prog, cfg, cat, wrap):
    """Fit a program to the geometry of the profile (module docstring)."""
    c = json.load(open(cfg))
    vb, words = 16 * c["geometry"]["superlanes"], c["sram"]["words_per_slice"]
    recs, sram, dumps = prog[:3]
    units = dict(prog[3]) if len(prog) > 3 else {}     # text units a front-end template gave directly (after the lowered records)
    if wrap:
        recs = [wrap_addr(cat, r, words) for r in recs]
    sram = {(sl, a % words): bytes(v)[:vb].ljust(vb, b"\0") for (sl, a), v in sram.items()}
    return recs, sram, list(dict.fromkeys((sl, a % words) for sl, a in dumps)), units


def lowered(recs, units):
    """The text images of a program: the lowering of its cycle view, then its direct units."""
    images = tsp_as.lower_records(recs)
    for q, us in units.items():
        images[q] = images.get(q, []) + us
    return dict(sorted(images.items(), key=lambda kv: (kv[0] == tsp_as.CONTROL, kv[0])))


def nop_runs(images):
    """(cycle, queue) of every parcel that follows a NOP prefix: the NOP is not an instruction of
    the parcel (it is the timing before it), so it runs when its queue reports nothing from the
    parcel's cycle on."""
    out, last = [], {}
    for c, q, _, _ in tsp_as.raise_images(images):
        if c > last.get(q, -1) + 1:
            out.append((c, q))
        last[q] = max(last.get(q, -1), c)
    return out


def coverage(cat, vectors, sim_diags):
    """(entries in no vector, entries in no vector where they issue without a diagnostic). An
    instruction issues cleanly when its queue reports nothing at or after its cycle (an
    instruction can wait, so its dispatch cycle is not its text cycle)."""
    used, clean = set(), set()
    nop = "NOP"
    for v, d in zip(vectors, sim_diags):
        for c, q in nop_runs(images_of(v["text"])):          # a NOP prefix is the NOP entry running
            used.add(nop)
            if not any(x["queue"] == q and x["cycle"] >= c for x in d):
                clean.add(nop)
        for r in recs_of_text(images_of(v["text"])):
            e = cat.entry(r)
            if e:
                used.add(e.name)
                if not any(x["queue"] == r[1] and x["cycle"] >= r[0] for x in d):
                    clean.add(e.name)
    names = [e.name for e in cat.isa.entries]
    return [n for n in names if n not in used], [n for n in names if n not in clean]


CLEAN_TEMPLATE = {"DeskewRecv": "deskewrecv", "Send": "c2c", "Receive": "c2c", "Transmit": "c2c", "Permute": "sxm_wide", "LoadMapWide": "sxm_wide", "Ifetch": "front_ifetch", "IfetchM": "front_fetch"}


def clean_programs(sim, cfg, cat, seed, vectors, tries=300):
    """Programs that run, without a diagnostic at their cycle and queue, the entries no vector
    so far runs cleanly. Template: one_<entry> of the oracle generator, else CLEAN_TEMPLATE."""
    sys.path.insert(0, os.path.join(SPEC, "test"))
    sys.path.insert(0, os.path.join(SPEC, "test", "oracle"))
    import diff_oracle as D
    from oracle import machine as mc
    m = mc.Machine(json.load(open(cfg)))
    diags = [v["expect"]["diagnostics"] for v in vectors]
    _, unclean = coverage(cat, vectors, diags)
    nop = "NOP"
    out = []
    for name in unclean:
        kind = "one_" + name.lower() if "one_" + name.lower() in D.TEMPLATES else CLEAN_TEMPLATE.get(name)
        if not kind:
            continue
        for i in range(tries):
            p = D.build(m, seed * 7919 + 1000 * len(out) + i, [kind])
            recs = [(c, 0xFFFF if q == 144 else q, wc, w0, w1) for (c, q, wc, w0, w1) in p.recs]
            recs, sram, dumps, units = fit((recs, dict(p.pre), sorted(p.dump), dict(p.units)), cfg, cat, True)
            images = lowered(recs, units)
            d, _, _ = run(sim, cfg, images, sram, dumps)
            if any((e := cat.entry(r)) and e.name == name and not any(x["queue"] == r[1] and x["cycle"] >= r[0] for x in d) for r in recs_of_text(images)) or \
                    (name == nop and any(not any(x["queue"] == q and x["cycle"] >= c for x in d) for c, q in nop_runs(images))):
                out.append((recs, sram, dumps, units))
                break
    return out


FETCH_INTEGRITY_NAME = "ecc-901"


def fetch_integrity_vector(sim, cfg, cat):
    """Poison the second sample without changing its opcode. Keep the first sample clean."""
    assembler = tsp_as.ISA(os.path.basename(cfg)[:-5], cfg)
    def units(text):
        return tsp_as.word_units(assembler.parse(text)[1])
    reads = units("read 0x0, E:0, 0x0") + units("read 0x1, E:0, 0x0")
    def word(first):
        return first.to_bytes(4, "little") + bytes([255]) * (cat.vlanes - 4)
    sram = {(0, 0): word(0x52000002), (0, 1): word(0x52000001)}
    # Where and when: run the two reads alone and take, from the T trace, the cycle and gap of each read's write to E:0; the value
    # then moves toward the Ifetch's station (layout.vxm_gap) one gap per cycle. The Ifetch is timed to latch the first value
    # there, and the fault lands on the second one at that gap, whatever the profile's MEM latency and gaps are.
    _, _, ev, _ = run_ext(sim, cfg, {0: reads}, sram, [])
    writes = [(int(t[2]), int(t[5])) for t in (e.split() for e in ev) if t[:2] == ["T", "srf"] and t[3] == "E:0"]
    if len(writes) != 2 or writes[0][1] != writes[1][1] or writes[1][0] != writes[0][0] + 1:
        raise ValueError(f"fetched-text vector: expected two consecutive writes to E:0, got {writes}")
    (c1, g1), (c2, _) = writes
    station = json.load(open(cfg))["layout"]["vxm_gap"]
    if station < g1:
        raise ValueError(f"fetched-text vector: the Ifetch station (gap {station}) is upstream of the read's gap {g1}")
    images = {0: reads, 88: [c1 + station - g1] + units("ifetch E:0")}
    cyc = f"{c2 + station - g1:x}"
    host = [f"F srf {cyc} e 0 {station:x} 0 80", f"F srf {cyc} e 0 {station:x} 0 81"]
    d, w, ev, ext = run_ext(sim, cfg, images, sram, [], host)
    irq = [int(line.split()[-1]) for line in ext if line.startswith("irq ")]
    if 1 in irq or 0 not in irq or 2 not in irq:
        raise ValueError(f"fetched-text containment failed: {irq}")
    ins, uni = cat.tags(recs_of_text(images))
    return {"name": FETCH_INTEGRITY_NAME, "instructions": ins, "units": uni, "text": text_json(images),
            "sram": {f"{sl}:{a}": v.hex() for (sl, a), v in sram.items()}, "dump": [], "host": host,
            "expect": {"diagnostics": d, "words": w, **expect_trace(ev), **ext_part(ext)}}


# ---------------------------------------------------------------------------------------------------------------------------------
# The side vector sets (--extra; conformance/model-extra/, outside the freeze set): what the frozen vectors of conformance/model/ do not
# pin. The vector format is the one of the frozen files; the names are vxmtype-, iq003-, iq002-, patrol- and fault-NNN.

def fault_host_lines(faults):
    """The harness F lines of a diff_oracle fault list."""
    out = []
    for f in faults:
        if f[0] == "sram":
            out.append("F sram %x %x %x %x %x" % f[1:])
        elif f[0] == "link":
            out.append("F link %x %x %x %s" % (f[1], f[2], f[3], "uncorr" if f[4] else "corr"))
        elif f[0] == "mxm":
            out.append("F mxm %x %x %x %x" % f[1:])
        elif f[0] == "iq":
            out.append("F iq %x %x %x %x" % f[1:])
        else:
            out.append("F srf %x %s %x %x %x %x" % (f[1], f[2].lower(), f[3], f[4], f[5], f[6]))
    return out


def vxm_type_pairs(cat, declared=()):
    """Every (VXM entry, allowed element type code) of the catalogue (encoding.json `codes` of the entry's eltype_code field), without the
    codes of an extension that the profile (`declared`: its `extensions`) does not name."""
    return [(e.name, code) for e in cat.isa.entries if e.slice == "VXM" for (_, _, _, kind, ex) in e.vars if kind == "eltype_code" for code in ex["codes"]
            if not any(code in cs and x not in declared for x, cs in ex.get("extension_codes", {}).items())]


def vxm_type_programs(cfg, cat):
    """One program per (VXM entry, allowed element type): the oracle generator's one-instruction template for the entry with the type forced, whose
    operands the generator feeds and whose result it drains. The assertion: the text of the program holds the pair."""
    sys.path.insert(0, os.path.join(SPEC, "test"))
    sys.path.insert(0, os.path.join(SPEC, "test", "oracle"))
    import diff_oracle as D
    from oracle import machine as mc
    m = mc.Machine(json.load(open(cfg)))
    D.WORDS = json.load(open(cfg))["sram"]["words_per_slice"]                  # the generator's address range (init_worker sets it for a run)
    saved = D.pick_type
    out = []
    try:
        for name, code in vxm_type_pairs(cat, json.load(open(cfg)).get("extensions", [])):
            t = D.nx.TYPE_NAME[code]
            D.pick_type = lambda rnd, isa, nm, t=t, name=name, saved=saved: t if nm == name else saved(rnd, isa, nm)
            for i in range(80):
                p = D.build(m, 9100 + 7 * len(out) + i, ["one_" + name.lower()])
                has = False
                for (_, _, wc, w0, _) in p.recs:
                    d = cat.isa.decode64(w0) if wc == 1 and (w0 >> 60) in cat.wide else None
                    if d and d[0] == name and (d[1].get("type") in (t, code)):
                        has = True
                if has:
                    break
            else:
                raise SystemExit(f"gen_model_vectors --extra: no program for {name} with type {t}")
            out.append(([(c, 0xFFFF if q == 144 else q, wc, w0, w1) for (c, q, wc, w0, w1) in p.recs], dict(p.pre), sorted(p.dump), dict(p.units)))
    finally:
        D.pick_type = saved
    return out


def iq003_program(cfg):
    """More IfetchM in flight than icu.fetch_staging_depth: the first ones are staged, the next finds the FIFO full (OTS-E-IQ-003) and fetches nothing."""
    c = json.load(open(cfg))
    depth = c["icu"]["fetch_staging_depth"]
    assembler = tsp_as.ISA(os.path.basename(cfg)[:-5], cfg)
    units, sram = [], {}
    pad = bytes([255]) * (16 * c["geometry"]["superlanes"])
    for k in range(depth + 1):
        units += tsp_as.word_units(assembler.parse(f"ifetchm {0x40 + 2 * k:#x}")[1])
        sram[(0, 0x40 + 2 * k)] = pad            # padding text words: the fetch adds nothing
        sram[(0, 0x41 + 2 * k)] = pad
    return ([], sram, [], {0: units})


def iq003_ifetch_program(cfg):
    """An Ifetch on a VXM queue every cycle, from a stream that a MEM slice feeds every cycle: the payloads (2 x vlanes bytes) arrive faster than the
    staging FIFO drains them (icu.fetch_bytes_per_cycle), so the FIFO fills and the next Ifetch finds it full (OTS-E-IQ-003). Unlike IfetchM,
    which needs the R path for two cycles, it reaches the bound on a profile whose payload drains in two cycles (mini)."""
    c = json.load(open(cfg))
    depth = c["icu"]["fetch_staging_depth"]
    assembler = tsp_as.ISA(os.path.basename(cfg)[:-5], cfg)
    word = lambda text: assembler.parse(text)[1]
    lay = c["layout"]
    transit = lay["vxm_gap"] - lay["mem_up_gap_base"]                 # slice 0 (quad 0), row E, to the VXM
    first = 40
    n = depth + 8
    recs = [(first + k - c["timing"]["mem"]["d_func"] - transit, 0, 1, word("read 0x40, E:0, 0x0"), 0) for k in range(n + 1)]
    recs += [(first + k, 88, 1, word("ifetch E:0"), 0) for k in range(n)]
    pad = bytes([255]) * (16 * c["geometry"]["superlanes"])
    return (recs, {(0, 0x40): pad}, [], {})


def iq002_program(cfg):
    """Fetched text arrives at a full IQ: the initial image fills icu.iq_depth except for one unit and holds the queue behind a long NOP, and its
    IfetchM brings 2 x vlanes bytes of text that cannot be stored (OTS-E-IQ-002, once per cycle for the units that do not fit)."""
    c = json.load(open(cfg))
    cap = c["icu"]["iq_depth"] // 4
    vb = 16 * c["geometry"]["superlanes"]
    assembler = tsp_as.ISA(os.path.basename(cfg)[:-5], cfg)
    units = lambda text: tsp_as.word_units(assembler.parse(text)[1])
    image = units("ifetchm 0x40") + units("nop 0x7530")
    filler = units("read 0x0, E:0, 0x0")
    while len(image) + len(filler) <= cap - 1:
        image += filler
    text = b"".join(u.to_bytes(4, "little") for u in (filler * (2 * vb // 4))[:2 * vb // 4])
    return ([], {(0, 0x40): text[:vb], (0, 0x41): text[vb:2 * vb]}, [], {0: image})


def patrol_programs(cfg):
    """Programs for a profile with scrub.period > 0: single-bit errors in slice 0 that the patrol corrects (one pass over addresses 0..2), and a
    double-bit one that it reports uncorrectable. Each ends with a Read of the word, long after the patrol visited it."""
    period = json.load(open(cfg))["scrub"]["period"]
    if period <= 0:
        raise SystemExit("gen_model_vectors --extra: the patrol programs need scrub.period > 0")
    cat = Catalogue(cfg)
    vb = cat.vlanes
    word = lambda k: bytes((17 * l + k) & 0xFF for l in range(vb))
    def prog(flips, addrs):
        sram = {(0, a): word(a) for a in addrs}
        recs = [(4 * period + 40, 0, 1, (1 << 60) | (0 << 54) | (a << 32) | (a << 26), 0) for a in addrs[:1]]
        host = [f"F sram 0 0 {a:x} 0 {b:x}" for a, b in flips]
        return (recs, sram, [(0, a) for a in addrs], {}, host)
    return [prog([(0, 5), (1, 9), (2, 3)], [0, 1, 2]),          # three single-bit errors, three passes
            prog([(1, 5), (1, 9)], [1]),                         # a double-bit one: reported uncorrectable
            prog([], [0])]                                       # a clean pass


def fault_programs(cfg, n_per_kind=3):
    """Programs of the oracle generator with injected faults (gen_faults), picked until every F kind (sram, srf, iq, link, mxm) occurred in at
    least n_per_kind programs. Returns (records, sram, dumps, units, host lines)."""
    sys.path.insert(0, os.path.join(SPEC, "test"))
    sys.path.insert(0, os.path.join(SPEC, "test", "oracle"))
    import diff_oracle as D
    from oracle import machine as mc
    m = mc.Machine(json.load(open(cfg)))
    D.WORDS = json.load(open(cfg))["sram"]["words_per_slice"]
    kinds = ["vxm", "mem", "mxm", "mxm_f16", "c2c", "front_fetch", "vxm_chain", "mxm_full"]
    count, out = {"sram": 0, "srf": 0, "iq": 0, "link": 0, "mxm": 0}, []
    for i in range(600):
        if min(count.values()) >= n_per_kind:
            break
        p = D.build(m, 5100 + i, [kinds[i % len(kinds)]])
        p.faults = D.gen_faults(m, p, 5100 + i)
        have = {f[0] for f in p.faults}
        if not have or all(count[k] >= n_per_kind for k in have):
            continue
        for k in have:
            count[k] += 1
        out.append(([(c, 0xFFFF if q == 144 else q, wc, w0, w1) for (c, q, wc, w0, w1) in p.recs], dict(p.pre), sorted(p.dump), dict(p.units), fault_host_lines(p.faults)))
    missing = [k for k, v in count.items() if v < n_per_kind]
    if missing and not any(k in missing for k in ("link", "mxm")):
        raise SystemExit(f"gen_model_vectors --extra: no program with F {' '.join(missing)} faults")
    return out


def extra_vectors(sim, cfg):
    cat = Catalogue(cfg)
    c = json.load(open(cfg))
    progs = [("vxmtype", p) for p in vxm_type_programs(cfg, cat)] + [("iq003", iq003_program(cfg)), ("iq003", iq003_ifetch_program(cfg)), ("iq002", iq002_program(cfg))]
    if c.get("scrub", {}).get("period", 0) > 0:
        progs += [("patrol", p) for p in patrol_programs(cfg)]
    else:
        progs += [("fault", p) for p in fault_programs(cfg)]
    vectors, n = [], {}
    for kind, prog in progs:
        recs, sram, dumps, units = fit(prog, cfg, cat, False)
        host = list(prog[4]) if len(prog) > 4 else []
        images = lowered(recs, units)
        d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, host)
        ins, uni = cat.tags(recs_of_text(images))
        if kind == "iq002" and not any(x["code"] == "OTS-E-IQ-002" for x in d):
            raise SystemExit(f"gen_model_vectors --extra: the IQ overflow program reports no OTS-E-IQ-002 on {cfg}")
        if kind == "iq003" and not any(x["code"] == "OTS-E-IQ-003" for x in d):
            # a profile whose payload (2 x vlanes bytes) leaves the staging FIFO before the next IfetchM can issue (mini: 32 bytes at 16 per cycle)
            # cannot fill it with IfetchM: no vector (the directed test of test/directed/frontend.py sets a depth of 1 to reach it there)
            continue
        n[kind] = n.get(kind, -1) + 1
        vectors.append({"name": f"{kind}-{n[kind]:03d}", "instructions": ins, "units": uni, "text": text_json(images),
                        "sram": {f"{sl}:{a}": bytes(v).hex() for (sl, a), v in sram.items()},
                        "dump": [f"{sl}:{a}" for sl, a in dumps], **({"host": host} if host else {}),
                        "expect": {"diagnostics": d, "words": w, **expect_trace(ev), **ext_part(ext)}})
    return {"schema": "opentsp.conformance.model", "spec_version": "0.1", "profile": os.path.basename(cfg)[:-5], "config_sha256": config_digest(cfg),
            "text": "queue (0..143, 65535 = control channel) -> the 32-bit units of its initial instruction queue, hex, space separated",
            "vectors": vectors}


def extra_main(sim, cfg, path, check):
    if not check:
        doc = extra_vectors(sim, cfg)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(doc, open(path, "w"), indent=0)
        print(f"side vectors: {len(doc['vectors'])} written to {path}")
        return 0
    doc = json.load(open(path))
    bad = 0
    if doc.get("config_sha256") != config_digest(cfg):
        bad += 1
        print("MISMATCH config_sha256: the vectors were generated from another configuration than", cfg)
    cat = Catalogue(cfg)
    for v in doc["vectors"]:
        images = images_of(v["text"])
        sram = {tuple(map(int, k.split(":"))): bytes.fromhex(x) for k, x in v["sram"].items()}
        dumps = [tuple(map(int, k.split(":"))) for k in v["dump"]]
        d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, v.get("host", []))
        ins, uni = cat.tags(recs_of_text(images))
        want_trace = {k: v["expect"][k] for k in ("trace_sha256", "events") if k in v["expect"]}
        if d != v["expect"]["diagnostics"] or w != v["expect"]["words"] or ins != v["instructions"] or uni != v["units"] \
                or expect_trace(ev) != want_trace or ext != v["expect"].get("ext_events", []):
            bad += 1
            print("MISMATCH", v["name"])
    print(f"side vectors: {len(doc['vectors'])} checked, {bad} mismatches")
    return 1 if bad else 0


def append_host(sim, cfg, path):
    """--append: add the vectors of the host programs (test/host/host_directed.py) and of the timing programs (timing_programs)
    that the file does not hold yet, named host-<n> after the ones it has and directed-<index> (the next free index); the existing vectors are not touched."""
    doc = json.load(open(path))
    cat = Catalogue(cfg)
    key = lambda text, sram, dump, host: json.dumps([text, sram, dump, host], sort_keys=True)
    have = {key(v["text"], v["sram"], v["dump"], v.get("host", [])) for v in doc["vectors"]}
    nums = {k: 1 + max([int(v["name"].split("-")[1]) for v in doc["vectors"] if v["name"].startswith(k + "-") and v["name"] != FETCH_INTEGRITY_NAME] or [0]) for k in ("host", "timing", "fp8", "ecc", "cr27")}   # timing, fp8, cr27: only counts
    added = 0
    dcfg = directed_cfg(cfg)      # the FP8 programs are ref-geometry programs, fitted by fit()
    for kind, prog in [("host", p) for p in host_programs(sim, cfg)] + [("timing", p) for p in timing_programs(sim, cfg)] + [("fp8", p) for p in fp8_programs(sim, dcfg)] + [("ecc", p) for p in ecc_programs(sim, cfg)] + [("ecc", p) for p in link_fec_programs(sim, cfg)] + [("ecc", p) for p in link_fec_programs(sim, cfg, "mxm_check")] + [("cr27", p) for p in cr27_programs(cfg)]:
        n = nums[kind]
        if kind == "ecc" and f"ecc-{n:03d}" == FETCH_INTEGRITY_NAME:
            n += 1
        recs, sram, dumps, units = fit(prog, cfg, cat, False)
        host = list(prog[4]) if len(prog) > 4 else []
        images = lowered(recs, units)
        v_sram = {f"{sl}:{a}": bytes(v).hex() for (sl, a), v in sram.items()}
        v_dump = [f"{sl}:{a}" for sl, a in dumps]
        if key(text_json(images), v_sram, v_dump, host) in have:
            continue
        d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, host)
        ins, uni = cat.tags(recs_of_text(images))
        name = f"{kind}-{n:03d}" if kind in ("host", "ecc") else f"directed-{len(doc['vectors']):03d}"    # the schema's name classes: a timing program is a directed one
        doc["vectors"].append({"name": name, "instructions": ins, "units": uni, "text": text_json(images), "sram": v_sram, "dump": v_dump,
                               **({"host": host} if host else {}),
                               "expect": {"diagnostics": d, "words": w, **expect_trace(ev), **ext_part(ext)}})
        nums[kind] = n + 1; added += 1
    # A catalogue entry that no vector holds yet (a new instruction): one coverage program, and one clean program where the
    # coverage program reports a diagnostic. The vectors keep their names and traces; the new ones go to the end.
    diags = [v["expect"]["diagnostics"] for v in doc["vectors"]]
    missing, _ = coverage(cat, doc["vectors"], diags)
    rnd = random.Random(7)
    for e in cat.isa.entries:
        if e.name in missing:
            recs, sram, dumps, units = fit(([(10, cat.queues(e)[0], 1, cat.word(e, rnd), 0)], {}, [], {}), cfg, cat, False)
            images = lowered(recs, units)
            d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, [])
            ins, uni = cat.tags(recs_of_text(images))
            doc["vectors"].append({"name": f"coverage-{len(doc['vectors']):03d}", "instructions": ins, "units": uni, "text": text_json(images),
                                   "sram": {}, "dump": [], "expect": {"diagnostics": d, "words": w, **expect_trace(ev), **ext_part(ext)}})
            added += 1
    diags = [v["expect"]["diagnostics"] for v in doc["vectors"]]
    for recs, sram, dumps, units in clean_programs(sim, cfg, cat, 7, doc["vectors"]):
        images = lowered(recs, units)
        d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, [])
        ins, uni = cat.tags(recs_of_text(images))
        doc["vectors"].append({"name": f"clean-{len(doc['vectors']):03d}", "instructions": ins, "units": uni, "text": text_json(images),
                               "sram": {f"{sl}:{a}": bytes(v).hex() for (sl, a), v in sram.items()},
                               "dump": [f"{sl}:{a}" for sl, a in dumps],
                               "expect": {"diagnostics": d, "words": w, **expect_trace(ev), **ext_part(ext)}})
        added += 1
    if not any(v["name"] == FETCH_INTEGRITY_NAME for v in doc["vectors"]):
        doc["vectors"].append(fetch_integrity_vector(sim, cfg, cat))
        added += 1
    write_vectors(doc, path)
    print(f"golden vectors: {added} added to {path}, {len(doc['vectors'])} in all")
    return 0


def refresh(sim, cfg, path):
    """--refresh: re-run every vector of the file and rewrite only what a change of the model that adds an external event (or a
    profile key) can change: `config_sha256` and `expect.ext_events`. Everything else (diagnostics, words, trace digest, events,
    instruction and unit tags) must be as it is, or this stops. Prints the vectors whose ext_events changed."""
    doc = json.load(open(path))
    cat = Catalogue(cfg)
    changed = []
    for v in doc["vectors"]:
        images = images_of(v["text"])
        recs = recs_of_text(images)
        sram = {tuple(map(int, k.split(":"))): bytes.fromhex(x) for k, x in v["sram"].items()}
        dumps = [tuple(map(int, k.split(":"))) for k in v["dump"]]
        d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, v.get("host", []))
        ins, uni = cat.tags(recs)
        want_trace = {k: v["expect"][k] for k in ("trace_sha256", "events") if k in v["expect"]}
        if d != v["expect"]["diagnostics"] or w != v["expect"]["words"] or ins != v["instructions"] or uni != v["units"] or expect_trace(ev) != want_trace:
            raise SystemExit(f"gen_model_vectors --refresh: vector {v['name']} differs in more than ext_events")
        if ext != v["expect"].get("ext_events", []):
            changed.append(v["name"])
            if ext: v["expect"]["ext_events"] = ext
            else: v["expect"].pop("ext_events", None)
    doc["config_sha256"] = config_digest(cfg)
    write_vectors(doc, path)
    print(f"golden vectors: {len(doc['vectors'])} refreshed in {path}; ext_events changed in {len(changed)}: {' '.join(changed)}")
    return 0


def main(argv):
    check = "--check" in argv
    append = "--append" in argv
    args = [a for a in argv[1:] if a not in ("--check", "--append", "--refresh", "--extra")]
    sim, cfg, path = args[0], args[1], args[2]
    if "--extra" in argv:
        return extra_main(sim, cfg, path, check)
    if append:
        return append_host(sim, cfg, path)
    if "--refresh" in argv:
        return refresh(sim, cfg, path)
    if check:
        doc = json.load(open(path))
        bad = 0
        if doc.get("config_sha256") != config_digest(cfg):
            bad += 1
            print("MISMATCH config_sha256: the vectors were generated from another configuration than", cfg)
        cat = Catalogue(cfg)
        all_diags = []
        for v in doc["vectors"]:
            images = images_of(v["text"])
            recs = recs_of_text(images)
            sram = {tuple(map(int, k.split(":"))): bytes.fromhex(x) for k, x in v["sram"].items()}
            dumps = [tuple(map(int, k.split(":"))) for k in v["dump"]]
            d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, v.get("host", []))
            all_diags.append(d)
            ins, uni = cat.tags(recs)
            want_trace = {k: v["expect"][k] for k in ("trace_sha256", "events") if k in v["expect"]}
            if d != v["expect"]["diagnostics"] or w != v["expect"]["words"] or ins != v["instructions"] or uni != v["units"] \
                    or expect_trace(ev) != want_trace or ext != v["expect"].get("ext_events", []):
                bad += 1
                print("MISMATCH", v["name"])
        missing, unclean = coverage(cat, doc["vectors"], all_diags)
        print(f"golden vectors: {len(doc['vectors'])} checked, {bad} mismatches")
        print(f"catalogue coverage ({doc['profile']}): {len(cat.isa.entries) - len(missing)}/{len(cat.isa.entries)} entries in a vector, "
              f"{len(cat.isa.entries) - len(unclean)} issue without a diagnostic in one")
        if unclean:
            print("  only with a diagnostic or in no vector: " + " ".join(unclean))
        if missing:
            print("  UNCOVERED: " + " ".join(missing))
        # Every entry must also run cleanly in some vector, on every profile.
        # A clean run of Gather, Scatter and the matrix instructions needs distinct MEM slices to feed (16 weight streams, map and data words);
        # on a profile with fewer than 8 MEM slices per hemisphere (mid_min, fetch_alt) the templates cannot build one, so there the list is advisory.
        few = json.load(open(cfg))["geometry"]["mem_slices_per_hemisphere"] < 8
        if unclean:
            print(f"  NOT CLEAN on {doc['profile']}: every entry must run without a diagnostic in some vector" + (" (advisory: fewer than 8 MEM slices per hemisphere)" if few else ""))
        return 1 if bad or missing or (unclean and not few) else 0
    n = int(args[3]) if len(args) > 3 else 40
    seed = int(args[4]) if len(args) > 4 else 7
    cat = Catalogue(cfg)
    # The directed scenarios are written for the ref geometry (their preloads are 320 bytes, which tsp_sim
    # rejects on a narrower profile): they are captured on ref and fitted to the profile by fit().
    dcfg = directed_cfg(cfg)
    progs = [("directed", p) for p in directed_programs(sim, dcfg)] + [("generated", p) for p in random_programs(n, seed, cfg)] + [("coverage", p) for p in cat.coverage_programs(seed)] + [("link", p) for p in link_programs(cfg)]
    progs += [("host", p) for p in host_programs(sim, cfg)]
    progs += [("directed", p) for p in timing_programs(sim, cfg)]
    progs += [("directed", p) for p in fp8_programs(sim, dcfg)]
    progs += [("directed", p) for p in bf16_programs(sim, dcfg)]
    progs += [("directed", p) for p in wide_programs(sim, cfg)]
    progs += [("ecc", p) for p in ecc_programs(sim, cfg)]
    progs += [("ecc", p) for p in link_fec_programs(sim, cfg)]
    progs += [("ecc", p) for p in link_fec_programs(sim, cfg, "mxm_check")]
    vectors, skipped = [], []
    for i, (kind, prog) in enumerate(progs):
        recs, sram, dumps, units = fit(prog, cfg, cat, kind == "generated")
        host = list(prog[4]) if len(prog) > 4 else []
        images = lowered(recs, units)
        try:
            d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps, host)
        except Inapplicable as e:
            if "initial IQ image longer" not in str(e) and (kind == "generated" or not small_profile(cfg)):
                raise
            skipped.append(f"{kind}-{i:03d}")
            continue
        ins, uni = cat.tags(recs_of_text(images))
        number = i + 1 if kind == "ecc" and i >= 901 else i
        vectors.append({"name": f"{kind}-{number:03d}", "instructions": ins, "units": uni, "text": text_json(images),
                        "sram": {f"{sl}:{a}": bytes(v).hex() for (sl, a), v in sram.items()},
                        "dump": [f"{sl}:{a}" for sl, a in dumps],
                        **({"host": host} if host else {}),
                        "expect": {"diagnostics": d, "words": w, **expect_trace(ev), **ext_part(ext)}})
    # Clean coverage: for each entry that no vector runs without a diagnostic, search the oracle
    # template for that entry (deterministic seeds) for a program that runs it cleanly.
    cleaned = clean_programs(sim, cfg, cat, seed, vectors)
    for recs, sram, dumps, units in cleaned:
        images = lowered(recs, units)
        d, w, ev, ext = run_ext(sim, cfg, images, sram, dumps)
        ins, uni = cat.tags(recs_of_text(images))
        vectors.append({"name": f"clean-{len(vectors):03d}", "instructions": ins, "units": uni, "text": text_json(images),
                        "sram": {f"{sl}:{a}": bytes(v).hex() for (sl, a), v in sram.items()},
                        "dump": [f"{sl}:{a}" for sl, a in dumps],
                        "expect": {"diagnostics": d, "words": w, **expect_trace(ev), **ext_part(ext)}})
    vectors.append(fetch_integrity_vector(sim, cfg, cat))
    doc = {"schema": "opentsp.conformance.model", "spec_version": "0.1", "profile": os.path.basename(cfg)[:-5], "config_sha256": config_digest(cfg),
           "text": "queue (0..143, 65535 = control channel) -> the 32-bit units of its initial instruction queue, hex, space separated",
           "vectors": vectors}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_vectors(doc, path)
    print(f"golden vectors: {len(vectors)} written to {path}")
    if skipped:
        print(f"  {len(skipped)} programs skipped: they name MEM slices this profile does not have, or their text is longer than icu.iq_depth (first: {' '.join(skipped[:6])})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
