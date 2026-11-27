#!/usr/bin/env python3
"""Peaks, bandwidths, roofline knees and the weight-install bound derived from a profile .

usage: derive_peaks.py <profile.json> [--clock-mhz 900] [--json]

Everything printed is *derived*: it follows from the profile keys by the formulas below, and from the clock, which is an
argument of this tool and never a profile key (a clock key would be a specification change). Every value carries the label
"derived; clock is an assumption". Nothing here is measured, and no number is a promise of silicon behaviour.

The formulas are the usual peak-rate, bandwidth and roofline metrics, written over this model's keys. With S = geometry.superlanes, V = 16 S the vector length (lanes), R = mxm.rows,
P = mxm.planes_per_hemisphere (1 or 2: 2 P planes), F = mxm.f16_rows, A = mxm.fp8_rate (1 or 2), D = mxm.acc_depth and f the clock:

  INT8      2 P planes x R x V MACCs x 2 ops x f
  FP8       A x the INT8 rate: a plane accepts A FP8 ABCs per cycle (microarchitecture.md section 6)
  FP16      one matrix per hemisphere of F x V MACCs, so 2 x F x V MACCs x 2 ops x f / timing.mxm_abc.occupancy_f16 (the f16 pair is the two
            planes of a hemisphere, so with P = 1 no plane can hold f16 and the FP16 peak is 0)
  bf16, bfp8  the FP16 rate when `extensions` lists bf16 or block_fp8 (0 otherwise): both take the capacity, the plane pair and the rate of f16
  accumulators  2 P planes x D accumulators x R rows x (4 bytes, plus 16 with mxm.acc_bits 128): state, not a rate (no peak depends on D: this is a throughput bound that
            prices no ACC drain, and a dot product longer than R keeps at most D partial rows before they go through a stream)
  SRAM      present slices x sram.words_per_slice x V bytes; bandwidth present slices x V bytes x (1 read + 1 write) x f
            (present slices = 2 x geometry.mem_slices_per_hemisphere: the other queue ids of sram.slices have no slice, topology.md 1)
  streams   2 x geometry.streams_per_row x V bytes x f (the two directions of the stream registers)
  C2C       V bytes per c2c.cycles_per_vector cycles per link x f; the chip has c2c.links of them, but it injects at most 4 vectors per
            cycle (4 transmit queues, 2 per hemisphere, and a Send holds its queue for one cycle): the total is the smaller of the sum of the
            links and 4 V f (96 links of 24 cycles per vector already reach it)
  host      V bytes per host.cycles_per_vector cycles x f
            (a SerDes lane rate does not follow the width of the chip: when a rate exceeds the published link figure, 15 GB/s per
            link at the most, or the PCIe Gen4 x16 payload of 31.5 GB/s, a note says so; derive_profile.py rule R5 keeps both rates
            at their published byte rates for any width)
  knees     peak ops/s over a byte rate: the arithmetic intensity (ops per byte) above which the peak, not that rate, binds
  install   the weights of all planes (2 P R V bytes, 409,600 on ref) come from the nearest MEM slices of each hemisphere, one slice per
            stream; an IW takes 16 streams, so a hemisphere uses at most 16 P streams and 2 x readers x V bytes arrive per cycle
            (readers = min(streams, MEM slices, 16 P): 32 on ref, 16 with one plane per hemisphere); the first ABC can issue once the last install has landed:
            (issues - 1) + timing.mxm_iw.d_func + timing.mem.d_func + transit, the formula of test/directed/published_bounds.py
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass

LABEL = "derived; clock is an assumption"
LANES_PER_SUPERLANE = 16
HEMISPHERES = 2
F16_MATRICES = 2  # one FP16 matrix per hemisphere, on the even plane of a plane pair (microarchitecture.md section 6)
STREAMS_PER_PLANE = 16  # an IW from streams carries one row per stream, 16 rows per issue (mxm.rows / 16 issues per plane)
C2C_TX_QUEUES = 4  # 2 transmit queues per hemisphere (topology.md section 2): a chip injects at most 4 vectors per cycle
DEFAULT_CLOCK_MHZ = 900.0
# The published rates a profile's links are compared with (performance.md section 3): a link of 4 lanes is 25 Gb/s per lane in the system
# whose latencies the profile uses (12.5 GB/s) and 30 Gb/s per lane at the most (15 GB/s); the host link is PCIe Gen4 x16, about 31.5 GB/s of payload.
LINK_CAPABILITY_GB_S = 15.0
HOST_LINK_GB_S = 31.5

REQUIRED = [
    "geometry.superlanes", "geometry.streams_per_row", "geometry.mem_slices_per_hemisphere",
    "sram.words_per_slice", "mxm.rows", "mxm.planes_per_hemisphere", "mxm.f16_rows", "mxm.fp8_rate", "mxm.acc_depth",
    "timing.mxm_abc.occupancy_f16", "timing.mem.d_func", "timing.mxm_iw.d_func",
    "layout.mem_down_gap_base", "layout.mxm_in_gap", "c2c.links", "c2c.cycles_per_vector", "host.cycles_per_vector",
]


class ProfileError(Exception):
    """A profile that cannot be read or lacks a key this tool needs; main prints it and exits 2."""


@dataclass
class Quantity:
    name: str
    value: float
    unit: str
    formula: str
    label: str = LABEL
    clock_dependent: bool = False

    def as_dict(self):
        return {"value": self.value, "unit": self.unit, "formula": self.formula, "label": self.label,
                "clock_dependent": self.clock_dependent}


def f16_matrices(planes_per_hemisphere):
    """The f16 matrices of a chip: one per hemisphere, on a plane pair; with one plane per hemisphere no pair exists."""
    return F16_MATRICES if planes_per_hemisphere == 2 else 0


def lookup(cfg, key):
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ProfileError(f"the profile has no key {key}")
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int) or node <= 0:
        raise ProfileError(f"{key} must be a positive integer here (got {node!r})")
    return node


def load_profile(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError as e:
        raise ProfileError(f"{path}: cannot read the profile ({e.strerror})")
    except ValueError as e:
        raise ProfileError(f"{path}: not valid JSON ({e})")


def derive(cfg, clock_mhz=DEFAULT_CLOCK_MHZ):
    """The derived quantities of a profile at an assumed clock: an ordered dict name -> Quantity."""
    if not (isinstance(clock_mhz, (int, float)) and math.isfinite(clock_mhz) and clock_mhz > 0):
        raise ProfileError(f"--clock-mhz must be a positive finite number (got {clock_mhz!r})")
    k = {key: lookup(cfg, key) for key in REQUIRED}
    f = clock_mhz * 1e6
    sl = k["geometry.superlanes"]
    vlanes = LANES_PER_SUPERLANE * sl
    rows = k["mxm.rows"]
    planes = HEMISPHERES * k["mxm.planes_per_hemisphere"]
    f16_count = f16_matrices(k["mxm.planes_per_hemisphere"])
    streams = k["geometry.streams_per_row"]
    slices = 2 * k["geometry.mem_slices_per_hemisphere"]
    out = {}

    def add(name, value, unit, formula, clock=False):
        out[name] = Quantity(name, value, unit, formula, clock_dependent=clock)

    add("vlanes", vlanes, "lanes", "16 x geometry.superlanes")
    add("int8_tops", planes * rows * vlanes * 2 * f / 1e12, "TOPS",
        "2 x mxm.planes_per_hemisphere x mxm.rows x vlanes x 2 x clock", True)
    occ = k["timing.mxm_abc.occupancy_f16"]
    add("fp16_tflops", f16_count * k["mxm.f16_rows"] * vlanes * 2 * f / occ / 1e12, "TFLOPS",
        "2 matrices (none with one plane per hemisphere) x mxm.f16_rows x vlanes x 2 x clock / timing.mxm_abc.occupancy_f16", True)
    add("fp8_tops", k["mxm.fp8_rate"] * out["int8_tops"].value, "TOPS",
        "mxm.fp8_rate x the INT8 rate (opentsp resource assumption, microarchitecture.md section 6)", True)
    ext = cfg.get("extensions", [])
    f16_rate = out["fp16_tflops"].value
    add("bf16_tflops", f16_rate if "bf16" in ext else 0.0, "TFLOPS",
        "the FP16 rate when `extensions` lists bf16 (the capacity, plane pair and rate of f16), else 0", True)
    add("bfp8_tflops", f16_rate if "block_fp8" in ext else 0.0, "TFLOPS",
        "the FP16 rate when `extensions` lists block_fp8: bfp8 takes the storage (a stream pair per row) and the rate of f16, not those of FP8", True)
    wide = cfg["mxm"].get("acc_bits", 32) == 128
    add("mxm_accumulator_bytes", planes * k["mxm.acc_depth"] * rows * (4 + 16 * wide), "B",
        "2 x mxm.planes_per_hemisphere x mxm.acc_depth x mxm.rows x (4, plus 16 when mxm.acc_bits is 128: the wide accumulator is held beside the "
        "32-bit one; state, no peak depends on it)")

    cap = slices * k["sram.words_per_slice"] * vlanes
    add("sram_present_slices", slices, "slices", "2 x geometry.mem_slices_per_hemisphere")
    add("sram_capacity_bytes", cap, "B", "present slices x sram.words_per_slice x vlanes")
    add("sram_capacity_mib", cap / 2**20, "MiB", "capacity / 2^20")
    sram_bw = slices * vlanes * 2 * f
    add("sram_bw_tb_s", sram_bw / 1e12, "TB/s", "present slices x vlanes x (1 read + 1 write) x clock", True)
    add("sram_bw_tib_s", sram_bw / 2**40, "TiB/s", "the same in binary units", True)
    stream_bw = 2 * streams * vlanes * f
    add("stream_bw_tb_s", stream_bw / 1e12, "TB/s", "2 x geometry.streams_per_row x vlanes x clock", True)
    add("stream_bw_tib_s", stream_bw / 2**40, "TiB/s", "the same in binary units", True)

    add("c2c_bytes_per_cycle_per_link", vlanes / k["c2c.cycles_per_vector"], "B/cycle", "vlanes / c2c.cycles_per_vector")
    add("host_bytes_per_cycle", vlanes / k["host.cycles_per_vector"], "B/cycle", "vlanes / host.cycles_per_vector")
    link = vlanes / k["c2c.cycles_per_vector"] * f
    add("c2c_gb_s_per_link", link / 1e9, "GB/s", "vlanes / c2c.cycles_per_vector x clock", True)
    links_sum = link * k["c2c.links"]
    injection = C2C_TX_QUEUES * vlanes * f
    add("c2c_gb_s_links", links_sum / 1e9, "GB/s", "c2c.links x the per-link rate (one direction)", True)
    add("c2c_gb_s_injection", injection / 1e9, "GB/s", "4 transmit queues x vlanes x clock (one vector per queue and cycle)", True)
    binds = "the links bind" if links_sum <= injection else "the injection limit binds"
    add("c2c_gb_s_total", min(links_sum, injection) / 1e9, "GB/s", f"the smaller of the two above: {binds}", True)
    host = vlanes / k["host.cycles_per_vector"] * f
    add("host_gb_s", host / 1e9, "GB/s", "vlanes / host.cycles_per_vector x clock", True)

    int8_ops, fp16_ops = out["int8_tops"].value * 1e12, out["fp16_tflops"].value * 1e12
    add("knee_int8_sram", int8_ops / sram_bw, "ops/B", "INT8 peak / SRAM bandwidth", True)
    add("knee_fp16_sram", fp16_ops / sram_bw, "FLOP/B", "FP16 peak / SRAM bandwidth", True)
    add("knee_int8_stream", int8_ops / stream_bw, "ops/B", "INT8 peak / stream-register bandwidth", True)
    add("knee_int8_c2c", int8_ops / (out["c2c_gb_s_total"].value * 1e9), "ops/B", "INT8 peak / total C2C bandwidth", True)
    add("knee_int8_host", int8_ops / host, "ops/B", "INT8 peak / host link bandwidth", True)

    # Weight install bound. The weights of all four planes come from the MEM slices nearest the MXM, one slice per
    # stream and hemisphere; a slice is a reader once per cycle. Quads are numbered from the VXM outward, so the nearest
    # `readers` slices are the last ones, and the farthest of them sits at gap mem_down_gap_base + (quads - used quads).
    mem = k["geometry.mem_slices_per_hemisphere"]
    readers = min(streams, mem, STREAMS_PER_PLANE * k["mxm.planes_per_hemisphere"])  # an IW takes 16 streams: a plane installs 16 rows per cycle
    quads, used = mem // 4, -(-readers // 4)
    far_gap = k["layout.mem_down_gap_base"] + quads - used
    transit = k["layout.mxm_in_gap"] - far_gap
    nbytes = planes * rows * vlanes
    issue = -(-nbytes // (2 * readers * vlanes))
    first = k["timing.mem.d_func"] + transit
    iw = k["timing.mxm_iw.d_func"]
    add("install_bytes", nbytes, "B", "2 x mxm.planes_per_hemisphere x mxm.rows x vlanes (all planes)")
    add("install_readers", readers, "slices per hemisphere", "min(geometry.streams_per_row, geometry.mem_slices_per_hemisphere, 16 x mxm.planes_per_hemisphere): "
        "one IW carries 16 rows, so a plane takes 16 streams")
    add("install_issue_cycles", issue, "cycles", "ceil(install bytes / (2 x readers x vlanes))")
    add("install_transit_cycles", transit, "cycles",
        "layout.mxm_in_gap - (layout.mem_down_gap_base + quads - used quads): from the farthest used quad to the MXM input")
    add("install_first_arrival_cycle", first, "cycles after the first read",
        "timing.mem.d_func + transit")
    add("install_last_arrival_cycle", first + issue - 1, "cycles after the first read",
        "first arrival + issue cycles - 1")
    add("install_first_abc_cycle", first + issue - 1 + iw, "cycles after the first read",
        "(issue cycles - 1) + timing.mxm_iw.d_func + timing.mem.d_func + transit: the first ABC after the last install has landed "
        "(41 on ref: 19 + 1 + 5 + 16; test/directed/published_bounds.py measures the same cycle on tsp_sim, on ref and on variants; "
        "no faster schedule was shown impossible)")
    return out


def rate_notes(quantities):
    """Notes for a profile whose link or host rate exceeds the published figure at the assumed clock (a wider chip with the ref cycle counts)."""
    q = quantities
    notes = []
    link, host = q["c2c_gb_s_per_link"].value, q["host_gb_s"].value
    if link > LINK_CAPABILITY_GB_S:
        notes.append(f"note: the link rate of {link:.2f} GB/s is above the published capability of {LINK_CAPABILITY_GB_S:g} GB/s per link "
                     "(4 lanes at 30 Gb/s) at this clock: the profile's c2c.cycles_per_vector gives a wide vector more link bandwidth than "
                     "any published figure")
    if host > HOST_LINK_GB_S:
        notes.append(f"note: the host rate of {host:.2f} GB/s is above the {HOST_LINK_GB_S} GB/s of payload of PCIe Gen4 x16 at this clock: "
                     "the profile's host.cycles_per_vector gives a wide vector more host bandwidth than the published link")
    return notes


def structure_notes(quantities):
    """Notes for a profile whose MXM structure changes what the chip can do (no f16 with one plane per hemisphere)."""
    notes = []
    if quantities["bfp8_tflops"].value > 0:
        notes.append("note: bfp8 (extension block_fp8) takes the storage and the rate of f16, 2 bytes per weight on chip and "
                     f"{quantities['bfp8_tflops'].value:,.1f} TFLOPS, not the FP8 rate: the block-scaled format saves no bytes and gives no speed over f16 here")
    if quantities["fp16_tflops"].value == 0:
        notes.append("note: with mxm.planes_per_hemisphere 1 no plane can hold f16 (an f16 pair is the two planes of a hemisphere): "
                     "the FP16 peak is 0 and an f16 model has no MAC rate")
    return notes


def fmt(v):
    """Four significant digits for a small value, thousands separators for a large one."""
    if float(v).is_integer():
        return f"{int(v):,}"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:,.{max(0, 3 - math.floor(math.log10(abs(v))))}f}"


def render(quantities, clock_mhz, path):
    lines = [f"{path}: {LABEL}", f"assumed clock: {clock_mhz:g} MHz (an argument of this tool, not a profile key)", ""]
    width = max(len(q.name) for q in quantities.values())
    for q in quantities.values():
        lines.append(f"  {q.name:<{width}}  {fmt(q.value):>12} {q.unit:<24} {q.formula}")
    notes = rate_notes(quantities) + structure_notes(quantities)
    lines += [""] + notes + ([""] if notes else []) + [f"every value above: {LABEL}"]
    return "\n".join(lines)


def positive_clock(text):
    try:
        v = float(text)
    except ValueError:
        v = float("nan")
    if not (math.isfinite(v) and v > 0):
        raise argparse.ArgumentTypeError(f"--clock-mhz must be a positive finite number of MHz (got {text!r})")
    return v


def run(main, name):
    """Run a tool's main: an interrupt or a closed pipe ends it quietly (exit 130, or 141 for a closed pipe), not with a traceback."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"{name}: interrupted", file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        sys.stderr.close()
        sys.exit(141)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("--clock-mhz", type=positive_clock, default=DEFAULT_CLOCK_MHZ,
                    help="assumed clock in MHz (default 900; an assumption, never a profile key)")
    ap.add_argument("--json", action="store_true", help="print the quantities as JSON")
    args = ap.parse_args(argv)
    try:
        q = derive(load_profile(args.profile), args.clock_mhz)
    except ProfileError as e:
        print(f"derive_peaks: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"profile": args.profile, "clock_mhz": args.clock_mhz, "label": LABEL,
                          "quantities": {n: x.as_dict() for n, x in q.items()},
                          "notes": rate_notes(q) + structure_notes(q)}, indent=2))
    else:
        print(render(q, args.clock_mhz, args.profile))
    return 0


if __name__ == "__main__":
    run(main, "derive_peaks")
