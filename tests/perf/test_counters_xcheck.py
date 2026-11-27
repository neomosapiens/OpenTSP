#!/usr/bin/env python3
"""Independent check of the ISS activity counters against the reference simulator.

usage: test_counters_xcheck.py [--sim PATH] [--iss PATH] [--asm PATH] [CASE ...]

For each program the check runs `tsp_sim --trace` and `tsp-iss run --stats` and derives from the reference simulator's trace
(`T srf`, `T sram`, `T diag` lines and the `E` line) the counts the ISS reports in the `activity` object of its statistics:

  * stream-register writes per row (`activity.srf.writes_east` / `writes_west`) = the `T srf` lines of that row;
  * stream-register writes per row and write gap (`activity.srf.by_gap`) = the `T srf` lines with that stream row and `gap`:
    the gap is the position of the writing unit along the row, so this is the count per unit (a MEM slice's Reads, the VXM's
    results, an MXM plane's ACC, an SXM's results);
  * SRAM word writes per MEM slice (`activity.mem.per_slice[].write_events`) = the `T sram` lines of that slice;
  * diagnostics (`diagnostics`) = the `T diag` lines = the count of the `E` line.

The trace carries no read and no arithmetic event, so the Reads are derived from where they write: a MEM Read or Gather
puts one vector on a stream at its slice's write gap, and the MEM gaps (`layout.mem_up_gap_base - quad`,
`layout.mem_down_gap_base + quad`, read from the configuration file, not from the ISS) are written by no other unit, so the
`T srf` lines at MEM gaps are the Reads and Gathers (`activity.mem.reads` less a Scatter's read and an IfetchM's two words,
which write no stream). The MAC and VXM element-operation counters have no trace event and are checked by the arithmetic of
tests/counters.rs. A merge of two writes (liquid threads, SEL and TSP) gives one `T srf` line for two scheduled vectors: the
five programs have none.

The counters with no trace event are checked against the program text instead, for the programs that are straight-line (no
AGT record, no Repeat, no Ifetch): the executed instructions are the instructions of the `.ots` source, so the check counts
its mnemonics (a second, textual derivation, independent of the ISS) and predicts the issued instructions per entry, the MEM
reads and writes and their bytes, the VXM element operations per type, the MXM instruction counts and MACs, and the SXM
vectors. The energy arithmetic is checked by pricing the ISS's counters again in Python from `opentsp-iss/perf/energy.yaml`
read by PyYAML and comparing the joules of `--energy`. Exit status 1 on any difference.
"""
import argparse
import importlib.util
import json
import os
import sys
import re
import tempfile
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROGRAMS = ["mem/agt_levels", "mem/gather_scatter", "mxm/weighted_matmul", "vxm/cmp_select_mask", "sxm/transpose16"]
PROFILE = "ref"
STATIC_CHECKED = []


def load_module():
    spec = importlib.util.spec_from_file_location("programs_run", os.path.join(ROOT, "tests", "programs", "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trace_counts(trace):
    srf_row, srf_gap, sram, diags = Counter(), Counter(), Counter(), 0
    for ln in trace:
        f = ln.split()
        if f[1] == "srf":
            row = f[3].split(":")[0]
            srf_row[row] += 1
            srf_gap[(row, int(f[5]))] += 1
        elif f[1] == "sram":
            sram[int(f[4])] += 1
        elif f[1] == "diag":
            diags += 1
    return srf_row, srf_gap, sram, diags


UNIT_LINE = re.compile(r"^\s+([A-Za-z0-9_.\[\]]+):\s+(\S+)")
TYPES = {"i8": 1, "u8": 1, "b8": 1, "e4m3": 1, "e5m2": 1, "f16": 2, "bf16": 2, "i16": 2, "u16": 2, "b16": 2, "i32": 4, "f32": 4,
         "u32": 4, "b32": 4}
FLOAT_TYPES = {"f16", "f32", "e4m3", "e5m2", "bf16"}   # EType::is_float of the ISS: the float scale applies to these, not to the integers and masks
ENTRY = {"read": "Read", "write": "Write", "gather": "Gather", "scatter": "Scatter", "lw": "LW", "iw": "IW", "abc": "ABC",
         "acc": "ACC", "cmp": "Cmp", "select": "Select", "mask": "Mask", "transpose": "Transpose"}
LOOPY = {"step", "iterations", "countdown", "streamstep", "repeat", "ifetch", "ifetchm", "setvl", "config", "sync", "notify"}


def static_expectation(source, vlanes, tile_bytes):
    """What the straight-line program `source` does, counted from its text; None for a program with loops."""
    mn = Counter()
    vxm = Counter()
    abc_types = Counter()
    for ln in open(source):
        m = UNIT_LINE.match(ln)
        if not m:
            continue
        unit, mnem = m.groups()
        parts = mnem.split(".")
        base = parts[0]
        if base in LOOPY or base not in ENTRY:
            return None
        mn[ENTRY[base]] += 1
        if unit.startswith("VXM"):
            ty = next(p for p in parts[1:] if p in TYPES)
            vxm[ty] += vlanes * (4 // TYPES[ty] if "sg4" in parts else 1)
        if base == "abc":
            abc_types[parts[1]] += 1
    reads = mn["Read"] + mn["Gather"] + mn["Scatter"]
    writes = mn["Write"] + mn["Scatter"]
    rows = 16 * mn["LW"] if mn["IW"] else 0
    return {
        "issued": dict(mn), "reads": reads, "writes": writes, "read_bytes": reads * vlanes, "write_bytes": writes * vlanes,
        "vxm": dict(vxm), "abc": dict(abc_types), "macs": sum(abc_types.values()) * rows * vlanes,
        "lw": mn["LW"], "iw": mn["IW"], "acc": mn["ACC"], "sxm_vectors": 16 * mn["Transpose"],
    }


def price_python(table, act, fetch_bytes, cycles, clock_mhz):
    """The joules of `act` under `table`, computed here from the definitions in energy.yaml (not by the ISS code)."""
    ev, cyc = table["events"], table["cycles"]
    vb = act["vector_bytes"]
    pj = act["mem"]["read_bytes"] * ev["mem_read_pj_per_byte"] + act["mem"]["write_bytes"] * ev["mem_write_pj_per_byte"]
    pj += act["srf"]["write_bytes"] * ev["srf_write_pj_per_byte"]
    for key, n in act["vxm"]["by_class_type"].items():
        cls, ty = key.split(".")
        pj += n * ev["vxm_op_pj"][cls] * ev["vxm_width_scale"]["w%d" % TYPES[ty]] * (ev["vxm_float_scale"] if ty in FLOAT_TYPES else 1)
    for ty, d in act["mxm"]["abc_by_type"].items():
        pj += d["macs"] * ev["mxm_mac_pj"][ty]
    pj += (act["mxm"]["lw_rows"] + act["mxm"]["iws_rows"]) * vb * ev["mxm_load_pj_per_byte"]
    pj += act["mxm"]["iw_rows"] * vb * ev["mxm_install_pj_per_byte"] + act["mxm"]["acc_lanes"] * ev["mxm_acc_pj_per_lane"]
    pj += (act["sxm"]["in_vectors"] + act["sxm"]["out_vectors"]) * vb * ev["sxm_pj_per_byte"]
    pj += (act["c2c"]["sent_bytes"] + act["c2c"]["received_bytes"] + act["c2c"]["sync_packets"] * vb) * ev["c2c_pj_per_byte"]
    pj += (act["host"]["in_bytes"] + act["host"]["out_bytes"]) * ev["host_pj_per_byte"] + fetch_bytes * ev["ifetch_pj_per_byte"]
    for cls, n in act["active_cycles"].items():
        pj += n * cyc[cls]["active_pj"] + (cycles - n) * cyc[cls]["idle_pj"]
    return pj * 1e-12 + table["static_w"] * cycles / (clock_mhz * 1e6)


def check(mod, runner, args, name):
    path = os.path.join(ROOT, "tests", "programs", f"{name}.{PROFILE}.case.json")
    case = json.load(open(path))
    if case["preload"] or case.get("host"):
        return [f"{name}: the native ISS path takes no preload or host vectors"]
    with tempfile.TemporaryDirectory(prefix="counters-xcheck-") as work:
        _, (obj, pkg) = runner.build_input(case, os.path.dirname(path), work)
        text = open(os.path.join(work, "input.txt")).read()
        sim = mod.parse_out(
            mod.sh([args.sim, mod.config_of(PROFILE), "--trace", "--max-cycles", str(case["budget"]["cycles"])],
                      stdin=text.encode()).stdout.decode())
        stats = os.path.join(work, "stats.json")
        table_path = os.path.join(ROOT, "opentsp-iss", "perf", "energy.yaml")
        cmd = [args.iss, "run", "--config", mod.config_of(PROFILE), f"--stats={stats}", "--energy", table_path,
               "--clock-mhz", "900", "--max-cycles", str(case["budget"]["cycles"])]
        if pkg:
            cmd += ["--weights", pkg]
        mod.sh(cmd + [obj])
        doc = json.load(open(stats))
    act = doc["activity"]
    srf_row, srf_gap, sram, diags = trace_counts(sim["trace"])
    bad = []

    def eq(what, want, got):
        if want != got:
            bad.append(f"{name}: {what}: tsp_sim trace {want} != ISS counters {got}")

    eq("stream writes, row E", srf_row["E"], act["srf"]["writes_east"])
    eq("stream writes, row W", srf_row["W"], act["srf"]["writes_west"])
    iss_gap = Counter({(g["row"], g["gap"]): g["writes"] for g in act["srf"]["by_gap"]})
    eq("stream writes per (row, gap)", dict(sorted(srf_gap.items())), dict(sorted(iss_gap.items())))
    iss_sram = {s["slice"]: s["write_events"] for s in act["mem"]["per_slice"] if s["write_events"]}
    eq("SRAM word writes per slice", dict(sorted(sram.items())), dict(sorted(iss_sram.items())))
    eq("diagnostics", diags, doc["diagnostics"])
    eq("E line count", int(sim["E"].split()[3]), doc["diagnostics"])
    lay = json.load(open(mod.config_of(PROFILE)))
    quads = lay["geometry"].get("mem_slices_per_hemisphere", 44) // 4
    lay = lay["layout"]
    mem_gaps = {lay["mem_up_gap_base"] - q for q in range(quads)} | {lay["mem_down_gap_base"] + q for q in range(quads)}
    others = {v for k, v in lay.items() if k.endswith("_gap") and not k.startswith("mem_")}
    assert not (mem_gaps & others), "a MEM gap is shared with another unit: the Read derivation does not hold"
    trace_reads = sum(n for (_, g), n in srf_gap.items() if g in mem_gaps)
    issued = act["issued"]
    iss_reads = act["mem"]["reads"] - issued.get("Scatter", 0) - 2 * issued.get("IfetchM", 0)
    eq("MEM Reads and Gathers (stream writes at MEM gaps)", trace_reads, iss_reads)
    if not act["mem"]["per_slice"] and not act["vxm"]["element_ops"]:
        bad.append(f"{name}: the program did nothing")
    # the energy arithmetic, priced again here
    table = yaml.safe_load(open(table_path))
    cycles = doc["last_cycle"] + 1
    want = price_python(table, act, doc["fetch_bytes"], cycles, 900.0)
    got = doc["energy"]["joules"]
    if abs(want - got) > 1e-9 * max(want, got):
        bad.append(f"{name}: joules: priced in Python {want!r} != ISS {got!r}")
    if doc["energy"]["calibration"] != table["calibration"]:
        bad.append(f"{name}: the calibration the ISS echoes differs from the table read by PyYAML")
    # the program text
    src = os.path.join(os.path.dirname(path), case["program"])
    exp = static_expectation(src, int(act["vector_bytes"]), 16)
    if exp is not None:
        m = act["mem"]
        eq("static: instructions issued per entry", exp["issued"], {k: v for k, v in act["issued"].items() if k in exp["issued"]})
        eq("static: MEM reads", exp["reads"], m["reads"])
        eq("static: MEM writes", exp["writes"], m["writes"])
        eq("static: MEM read bytes", exp["read_bytes"], m["read_bytes"])
        eq("static: MEM write bytes", exp["write_bytes"], m["write_bytes"])
        eq("static: VXM element operations per type", exp["vxm"], act["vxm"]["by_type"])
        eq("static: ABC per type", exp["abc"], {k: v["abc"] for k, v in act["mxm"]["abc_by_type"].items() if v["abc"]})
        eq("static: MACs", exp["macs"], act["mxm"]["macs"])
        eq("static: LW, IW, ACC", (exp["lw"], exp["iw"], exp["acc"]), (act["mxm"]["lw"], act["mxm"]["iw"], act["mxm"]["acc"]))
        eq("static: SXM vectors in and out", (exp["sxm_vectors"],) * 2, (act["sxm"]["in_vectors"], act["sxm"]["out_vectors"]))
        STATIC_CHECKED.append(name)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.environ.get("TSP_SIM", os.path.join(ROOT, "target", "spec", "tsp_sim")))
    ap.add_argument("--iss", default=os.environ.get("TSP_ISS", os.path.join(ROOT, "target", "release", "tsp-iss")))
    ap.add_argument("--asm", default=os.environ.get("OTS_ASM", os.path.join(ROOT, "target", "release", "ots-asm")))
    ap.add_argument("cases", nargs="*", default=PROGRAMS)
    a = ap.parse_args()
    for what, p in (("tsp_sim", a.sim), ("tsp-iss", a.iss), ("ots-asm", a.asm)):
        if not os.access(p, os.X_OK):
            sys.exit(f"test_counters_xcheck: {what} not found at {p} (build it, or pass --sim/--iss/--asm)")
    mod = load_module()
    runner = mod.Runner(argparse.Namespace(asm=a.asm, sim=a.sim, iss=a.iss, timeout=600, keep=None))
    failed = 0
    for name in a.cases:
        try:
            bad = check(mod, runner, a, name)
        except Exception as e:  # a tool that does not run is a failure, not a skip
            bad = [f"{name}: {type(e).__name__}: {e}"]
        print(("FAIL " if bad else "PASS ") + name)
        for b in bad:
            print("  " + b)
        failed += bool(bad)
    print(f"{len(a.cases) - failed}/{len(a.cases)} programs agree with tsp_sim; counted from the program text too: {', '.join(STATIC_CHECKED)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
