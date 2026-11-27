// Specification version (0.1 while in initial development). The configuration file
// must name the version it was written for.
let spec_version : string = config spec_version

// Machine geometry. Values come from the configuration file (config/ref.json,
// config/mini.json), validated against the schema Sail generates with --output-schema.

// Superlanes: 20 in the reference profile (ISCA 2020 sec. II), 1 in the mini profile, any count from 1 to 32 in a
// profile. 32 is the most the 5-bit superlane field of an error record holds (core/csr.sail ras_record) and the most that
// the 9-bit lane count of Shift reaches (512 lanes); a larger count changes those layouts and is not part of this version.
type superlanes : Int = config geometry.superlanes
constraint superlanes >= 1 & superlanes <= 32
let superlanes = sizeof(superlanes)

// Lanes per stream value: one byte per lane, 16 lanes per superlane.
type vlanes : Int = 16 * superlanes
let vlanes = sizeof(vlanes)

// Stream registers per row (positions 0 .. srf_positions - 1) and streams per row.
type srf_positions : Int = config geometry.srf_positions
constraint srf_positions == 45
let srf_positions = sizeof(srf_positions)

type streams_per_row : Int = config geometry.streams_per_row
constraint streams_per_row == 32

// SRAM: per-slice word count and bank select bit.
type mem_slices : Int = config sram.slices
constraint mem_slices == 88
let mem_slices = sizeof(mem_slices)

// SRAM address width: 13 bits in the reference profile (ISCA 2020 sec. II item 5), 13 to 16 in a profile. It is the width of the
// address of a MEM instruction (bits 47..32 hold the 16-bit field, whose bits above addr_bits are zero), of IfetchM (bits 15..0), of
// the address a Gather or Scatter takes from its map (the low addr_bits bits of a halfword), and the number of words a slice can address.
type addr_bits : Int = config sram.addr_bits
constraint addr_bits >= 13 & addr_bits <= 16
let addr_bits = sizeof(addr_bits)

// Words per slice: any power of two from 256 to 65536 that the address width can address (at most 2^addr_bits, which
// tools/schema_invariants.py adds to the schema because Sail cannot write it there).
type words_per_slice : Int = config sram.words_per_slice
constraint words_per_slice in {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}
let words_per_slice = sizeof(words_per_slice)

// The bank bit is the upper-most address bit of the words (bit 12 of the 8192 words of ISCA 2020 13-bit
// addressing, an inference). opentsp decision for every profile: the same rule,
// bank_bit = log2(words_per_slice) - 1, so both banks hold half the slice. Sail cannot write this relation into the schema;
// tools/schema_invariants.py adds it.
type bank_bit : Int = config sram.bank_bit
constraint (words_per_slice == 256 & bank_bit == 7) | (words_per_slice == 512 & bank_bit == 8)
         | (words_per_slice == 1024 & bank_bit == 9) | (words_per_slice == 2048 & bank_bit == 10)
         | (words_per_slice == 4096 & bank_bit == 11) | (words_per_slice == 8192 & bank_bit == 12)
         | (words_per_slice == 16384 & bank_bit == 13) | (words_per_slice == 32768 & bank_bit == 14)
         | (words_per_slice == 65536 & bank_bit == 15)
constraint bank_bit < addr_bits
let bank_bit = sizeof(bank_bit)

// MEM slices per hemisphere: 44 in the reference profile (ISCA
// 2020 sec. II). The queue map does not depend on it (144 ids in every profile, so the ISA,
// the queue names and encoding.json are the same everywhere): MEM slice s (0 .. 87) exists
// when (s mod 44) < mem_slices_per_hemisphere (core/layout.sail mem_slice_present). A
// multiple of 4 because slices come in quads (the unit of distance from the VXM). Sail
// cannot write "multiple of 4" into the schema; tools/schema_invariants.py adds it.
let mem_slices_per_hemisphere : range(4, 44) = config geometry.mem_slices_per_hemisphere

// MXM rows: stream values held by a plane's weight buffer and by an installed matrix:
// that many i8 or u8 rows, or half as many f16 rows (two byte planes each). 320 in the
// reference profile (ISCA 2020 sec. III-D: 320 x 320 planes). A multiple of 16 because LW
// and IW from streams move 16 stream values at a time (tools/schema_invariants.py).
type mxm_rows : Int = config mxm.rows
constraint mxm_rows >= 16 & mxm_rows <= 320
let mxm_rows = sizeof(mxm_rows)

// MXM structure keys (topology.md "Profile parameters", microarchitecture.md sec. 6).
//
// mxm.planes_per_hemisphere: planes per hemisphere, 2 in the reference profile (ISCA 2020 sec. III-D: four planes; HC34 p.28:
// planes 0 and 1 west, 2 and 3 east). The queue map is fixed (3 queues per plane, core/queues.sail), so 1 or 2 are the counts it
// can name: with 1 the second plane of each hemisphere and its three queues do not exist (the queues report OTS-E-QUE-002 as the
// queues of an absent MEM slice do), and an f16 pair (planes 2i and 2i + 1 of a hemisphere) exists only when the hemisphere has
// both: with 1 plane per hemisphere no plane can hold f16 weights.
type mxm_planes : Int = config mxm.planes_per_hemisphere
constraint mxm_planes == 1 | mxm_planes == 2
let mxm_planes = sizeof(mxm_planes)

// mxm.acc_depth: accumulators per plane, 2 in the reference profile (the published MACC has two 32-bit accumulators). The ACC
// field takes any index below the depth: 0..1 in bit 12 and, for a deeper profile, the higher bits in 11..8 (a 5-bit index).
type mxm_acc_depth : Int = config mxm.acc_depth
constraint mxm_acc_depth >= 2 & mxm_acc_depth <= 32
let mxm_acc_depth = sizeof(mxm_acc_depth)

// mxm.f16_rows: the rows of an f16 matrix (two byte planes per row), mxm.rows / 2 in the reference profile (HC34 p.28: one
// 160 x 320 f16 matrix per hemisphere) or mxm.rows (ISCA 2020 sec. III-D: both byte planes of a pair in tandem). Capacity only:
// the f16 rate is timing.mxm_abc.occupancy_f16. The weight buffer and the installed matrix hold mxm_wvals = 2 * f16_rows stream
// values (the f16 rows, two per row; at least mxm.rows, so also the i8 rows). The relation rows <= 2 * f16_rows and f16_rows <= rows
// is checked by the configuration check of every harness (postlude/timing_check.sail `timing_config_error`), not by the schema.
type mxm_f16_rows : Int = config mxm.f16_rows
constraint mxm_f16_rows >= 8 & mxm_f16_rows <= 320
constraint mxm_f16_rows <= mxm_rows & mxm_rows <= 2 * mxm_f16_rows
let mxm_f16_rows = sizeof(mxm_f16_rows)
type mxm_wvals : Int = 2 * mxm_f16_rows
let mxm_wvals = sizeof(mxm_wvals)

// mxm.fp8_rate: FP8 ABCs a plane accepts in one cycle, relative to INT8 (1: one, as i8; 2: two, from two queues).
let mxm_fp8_rate : range(1, 2) = config mxm.fp8_rate

// Barrier: first data dispatch allowed at notify_cycle + release_latency. opentsp
// decision: at most 4095 cycles, so that a configuration cannot push a release past the
// emulator's cycle range.
let barrier_release_latency : range(0, 4095) = config barrier.release_latency

// Accumulator width of a matrix plane, in bits: 32 (the i32 and f32 accumulators only) or 128 (the wide accumulator A_X128 too:
// a 128-bit fixed point sum per lane, `mxm.acc_depth` per plane). The value 128 needs the extension wide_acc (below).
// Sail cannot write "32 or 128" into the schema; tools/schema_invariants.py adds it.
let mxm_acc_bits : range(32, 128) = config mxm.acc_bits
