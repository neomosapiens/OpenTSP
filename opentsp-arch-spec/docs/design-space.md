# TSP reference architecture: design space

This document separates what the architecture fixes from what the reference chip happens to choose and from what a profile or the encoding lets a port vary. It is a map for an architect who wants to ask "what if the chip were different", and for a reader who wants to know which numbers carry a reason. It adds no behaviour: every rule is stated where it is cited, and a disagreement with another document is a defect of this one.

**Scope.** This chapter distinguishes architectural invariants, profile-selectable choices and fixed implementation limits. A classification without a published basis is an explicit modeling choice.

## 1. Three layers

- **Paradigm:** a property of the programming model. Changing it makes another architecture, and every artifact (model, assemblers, compilers) changes.
- **Reference floorplan:** a choice of the reference chip, taken from the published TSP sources (ISCA20, ISCA22, HC34) or made by this specification where they are silent. It can be varied by a profile key, by the cap table of `topology.md` §1, or by an explicit specification change.
- **Profile and encoding:** the key that varies it today, and the frozen encoding field that limits it with its growth path.

| Structure | Paradigm | Reference floorplan | Profile and encoding |
|---|---|---|---|
| Execution | no branch, no speculation, no cache; the compiler places every instruction at its cycle (ISCA20 §III-A) | one parcel per queue per cycle (HC34 p.23) | `icu.*` keys; the object carries text, not cycles (`assembly.md` §4) |
| Scheduling equation | operand arrival = producer dispatch + d_func + distance − consumer d_skew (ISCA20 Eq. 4) | d_func and d_skew per instruction entry | `timing.*` rows and `timing.overrides`; d_func < 64, d_skew < 16 |
| Stream flow | values flow one register per cycle on two opposite rows, and a value is read at one cycle only (ISCA20 §II; HC34 p.24) | 45 registers per row, 32 streams per row | `geometry.srf_positions` and `streams_per_row` are fixed; stream field 6 bits |
| Functional slicing | one unit kind per slice, MEM, VXM, MXM, SXM, C2C, instructions bound to their unit (ISCA20 §II) | 44 MEM slices per hemisphere, 4 MXM planes, 16 VXM ALUs, one SXM per hemisphere | `geometry.mem_slices_per_hemisphere`, `mxm.rows`, `mxm.planes_per_hemisphere` (1 or 2); ALUs and SXMs are caps (`topology.md` §1) |
| Symmetry and the bisection | two mirror hemispheres around the VXM (HC34 p.14) | the order MXM, SXM, MEM, VXM; quads of 4 MEM slices share a station | the ten `layout.*` keys move stations within the nine ordering relations of the schema |
| Lock-step stagger | superlane k runs k cycles after superlane 0, so all timing is stated for superlane 0 (ISCA20 §II) | 20 superlanes of 16 lanes | `geometry.superlanes` 1 to 32; cross-superlane rows need `d_func − d_skew ≥ superlanes` |
| Integrity | SECDED on SRAM words, streams and text, with correction and a record in a CSR (ISCA20 §II-D) | 9 check bits per 128-bit word, 138-bit stream registers; poison propagation after an uncorrectable error is an opentsp choice (`isa-db/provenance.yaml`, `uncorrectable_poison`; the published sources give no behaviour after detection) | the scheme is fixed; the `ras.*` and `scrub.*` keys size logs and patrol |
| Memory | pseudo-dual-port SRAM, one read and one write per cycle in different banks (ISCA20 §III-B) | 2 banks per slice by the top address bit, 8192 words of 16 bytes per tile | `sram.words_per_slice` 256..65536 and `bank_bit` (not free: `topology.md` §6, "Banking"); `sram.addr_bits` 13..16  |
| Matrix | 320-wide dot products of stream vectors against installed weights, accumulated in the unit (HC34 p.28) | 320 × 320 planes, 2 accumulators, FP16 in plane pairs | `mxm.rows`; `timing.mxm_abc.occupancy_f16`; `mxm.acc_depth` (2..32), `mxm.f16_rows` (`mxm.rows`/2 to `mxm.rows`) and `mxm.fp8_rate` (1 or 2), the profile keys that `derive_peaks.py` reads (`performance.md` §2); `mxm.acc_bits` and `extensions` (bf16, the wide accumulator, block-scaled fp8; `performance.md` §2 gives their peaks and the cost of bfp8, which takes the storage and rate of f16); more than 2 planes per hemisphere is a cap (the queue map) |
| Vector | element-wise operations on streams, chained through one operand bit | 16 ALUs per lane in a 4 × 4 mesh | `timing.vxm`, per-entry overrides; ALU count is fixed by the queue map |
| Switch | lane movement between superlanes and rows (ISCA20 §III-E) | distributor, transposer, shifter, permuter, select | `timing.sxm`, `timing.sxm_cross`, `timing.sxm_map_ready` |
| Chip to chip | static scheduling across chips with the hardware-aligned counter and Deskew (ISCA22 §3) | 16 links, one vector per 24 cycles (12 GB/s per link at 900 MHz for a 320-byte vector: a wider vector needs more cycles for the same link, `performance.md` §6), latency 209..228 | `c2c.*` keys, `c2c.links` 1..128 (7 bits of `link`); `chip_id` 9 bits: a run takes 1..264 chips, a topology file up to 512 (`topology.md` §7); a chip injects at most 4 vectors per cycle (4 transmit queues), which caps the total of its links (`performance.md` §2) |
| Host | a doorbell and DMA interface outside the instruction stream | one PCIe-like interface per chip | `host.*` keys; the CSR space (`platform.md`) |
| Barrier | a chip-wide Sync and Notify with a fixed release latency (ISCA20 §III-A2) | 35 cycles | `barrier.release_latency` 0..4095 |
| Numerics | the element formats and rounding of the catalogue | correctly rounded transcendentals (`microarchitecture.md` §5) | the element-type code is 5 bits: codes 13..15 and 17..31 are free (`encoding.md`) |

Where a cell says "fixed" the Sail model, the ISS and the harness assume it; where it names a key, the key is validated by `config/schema.json` and the relations of `tools/schema_invariants.py`.

## 2. The implementation-choice timing rows

`microarchitecture.md` §1 classifies the 78 rows of the timing table: 1 published, 2 derived and 75 implementation choices. A choice with no source is not a defect, but a reader should not mistake it for evidence. The rows below are the 75 choices, grouped by family, with the reason where one exists and "no reason: free" where the specification would accept another value within the schema bounds. The reasons are the ones recorded in `isa-db/provenance.yaml` `timing_rows` and `config`; `check_provenance.py` checks that this section lists every implementation-choice row of the provenance with the same d_func and d_skew.

### Control unit (`timing.cu`, d_func 0, d_skew 0)

A control instruction puts no value on a stream, so d_func 0 is allowed (every other class needs d_func ≥ 1). The skew of 0 is chosen so that a fetch completes after its second read; no source gives either number. Notify and Sync carry their published rule in `barrier.release_latency` (35), not in these rows.

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `HostWait` | 0 | 0 | no reason: free |
| `HostNotify` | 0 | 0 | no reason: free |
| `Config` | 0 | 0 | no reason: free |
| `Ifetch` | 0 | 0 | no reason: free |
| `IfetchM` | 0 | 0 | no reason: free |
| `NOP` | 0 | 0 | no reason: free |
| `Notify` | 0 | 0 | no reason: free |
| `Repeat` | 0 | 0 | no reason: free |
| `SetVL` | 0 | 0 | no reason: free |
| `Sync` | 0 | 0 | no reason: free |

### Memory slice (`timing.mem`, d_func 5, d_skew 0)

No published value; d_func 5 is a model default, not a measured SRAM latency, and d_skew 0 is unsourced. Any d_func ≥ 1 is legal if Read, Write, Gather, Scatter and IfetchM keep one d_skew (the shared SRAM port). The AGT instructions and SetSegment share the class row and have no timing of their own.

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `Countdown` | 5 | 0 | no reason: free |
| `DeepSleep` | 5 | 0 | no reason: free |
| `Gather` | 5 | 0 | no reason: free |
| `Iterations` | 5 | 0 | no reason: free |
| `PowerConfig` | 5 | 0 | no reason: free |
| `Read` | 5 | 0 | no reason: free |
| `Scatter` | 5 | 0 | no reason: free |
| `SetSegment` | 5 | 0 | no reason: free |
| `Step` | 5 | 0 | no reason: free |
| `StreamStep` | 5 | 0 | no reason: free |
| `Write` | 5 | 0 | no reason: free |

### Matrix, FP8 multiply (`timing.mxm_abc`)

ABC8 takes the published 20 of ABC (HC34 p.28) for lack of a source of its own; it must also satisfy `d_func − d_skew ≥ superlanes`.

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `ABC8` | 20 | 0 | as ABC |

### Matrix, accumulate (`timing.mxm_acc`)

The reference chooses 20 cycles for ACC. The published lower bound is one cycle. See `microarchitecture.md` §1 for scheduling.

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `ACC` | 20 | 0 | reference timing choice |

### Matrix, install weights (`timing.mxm_iw`)

The floor of 1: installing moves stream values into the plane buffer. No source gives a latency. The install bound of the published sources is recorded in `isa-db/provenance.yaml` `timing_bounds`.

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `IW` | 1 | 0 | floor of 1 |
| `IWS` | 1 | 0 | floor of 1 |

### Matrix, load weights (`timing.mxm_lw`)

20 follows the 20-supercell stagger of the plane; no source gives LW a latency. Whether the published weight-install time includes this lag is not stated (`timing_bounds`).

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `LW` | 20 | 0 | stagger of the plane |

### Switch unit, same-superlane operations (`timing.sxm`, d_func 5, d_skew 0)

No published value; 5 equals the memory and vector classes for lack of a reason to differ. The cross-superlane rows (Shift, Permute) are derived, not choices.

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `Distribute` | 5 | 0 | no reason: free |
| `LoadMap` | 5 | 0 | no reason: free |
| `LoadMapWide` | 5 | 0 | no reason: free |
| `Rotate` | 5 | 0 | no reason: free |
| `SelectS` | 5 | 0 | no reason: free |
| `Transpose` | 5 | 0 | no reason: free |

### Vector unit (`timing.vxm`, d_func 5, d_skew 2)

The skew of 2 is the operand-read pipeline of the vector unit, a stated choice and not a circuit delay; d_func 5 has no source. Without a typed row one row serves every type of an entry (§1); a profile may give one element type of an entry its own row (`timing.overrides` entry `<Entry>.<type>`, for example `Max.i8`); the transcendental and conversion entries share it although their circuits would differ, so a profile that wants a slower Exp overrides it by name (the shipped profiles do for Exp, Log, RSqrt and TanH).

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `Abs` | 5 | 2 | no reason: free |
| `Add` | 5 | 2 | no reason: free |
| `Add_mod` | 5 | 2 | no reason: free |
| `Add_sat` | 5 | 2 | no reason: free |
| `And` | 5 | 2 | no reason: free |
| `Cmp` | 5 | 2 | no reason: free |
| `Conv_F16_F32` | 5 | 2 | no reason: free |
| `Conv_F16_I8` | 5 | 2 | no reason: free |
| `Conv_F32_F16` | 5 | 2 | no reason: free |
| `Conv_F32_I8` | 5 | 2 | no reason: free |
| `Conv_I32_I8` | 5 | 2 | no reason: free |
| `Conv_I8_F16` | 5 | 2 | no reason: free |
| `Conv_I8_F32` | 5 | 2 | no reason: free |
| `Conv_I8_I32` | 5 | 2 | no reason: free |
| `Convert` | 5 | 2 | no reason: free |
| `Exp` | 5 | 2 | no reason: free |
| `Log` | 5 | 2 | no reason: free |
| `Mask` | 5 | 2 | no reason: free |
| `Max` | 5 | 2 | no reason: free |
| `Min` | 5 | 2 | no reason: free |
| `Mul` | 5 | 2 | no reason: free |
| `Mul_mod` | 5 | 2 | no reason: free |
| `Mul_sat` | 5 | 2 | no reason: free |
| `Neg` | 5 | 2 | no reason: free |
| `Not` | 5 | 2 | no reason: free |
| `Or` | 5 | 2 | no reason: free |
| `RSqrt` | 5 | 2 | no reason: free |
| `ReLU` | 5 | 2 | no reason: free |
| `Reduce` | 5 | 2 | no reason: free |
| `Select` | 5 | 2 | no reason: free |
| `Sub` | 5 | 2 | no reason: free |
| `Sub_mod` | 5 | 2 | no reason: free |
| `Sub_sat` | 5 | 2 | no reason: free |
| `TanH` | 5 | 2 | no reason: free |
| `Xor` | 5 | 2 | no reason: free |

### Chip-to-chip and host (`timing.c2c`, d_func 1, d_skew 0)

The floor of 1. The real timing of these instructions is a rule, not a row: link rate, HAC epoch, host pacing (`microarchitecture.md` §10, §14).

| Entry | d_func | d_skew | Reason |
|---|---|---|---|
| `Deskew` | 1 | 0 | floor of 1; the rule governs |
| `Receive` | 1 | 0 | floor of 1; the rule governs |
| `RuntimeDeskew` | 1 | 0 | floor of 1; the rule governs |
| `Send` | 1 | 0 | floor of 1; the rule governs |
| `Transmit` | 1 | 0 | floor of 1; the rule governs |
| `PciReceive` | 1 | 0 | floor of 1; the rule governs |
| `PciSend` | 1 | 0 | floor of 1; the rule governs |
| `DeskewRecv` | 1 | 0 | floor of 1; the rule governs |

## 3. How to vary a layer

- A **profile key** changes a number in a profile and keeps the frozen interface: write `config/test/<name>.json`, check it with `tools/schema_invariants.py` and run the area recipes of `AGENTS.md` section 1 on it.
- A **cap** (`topology.md` §1) changes the model, the ISS, the harness and the encoding together, as one coherent specification change.
- A **paradigm** property is not varied in this repository: a chip with another rule is another specification.
- Adding an instruction, a unit or a queue follows `extending.md`.

## 4. Architectural alternatives

Three choices of the reference floorplan decide how fast a kernel can run, and none of them rests on a published number. Each line says what is fixed today, what the published sources give, the alternative, and what would change. These alternatives are outside the current contract unless selected by an existing profile key.

| Choice | Fixed today, and its source | Alternative | What would change |
|---|---|---|---|
| FP8 rate | FP8 weights take the capacity and timing of i8 by default: 320 rows, one ABC per cycle, 20 cycles, and `ABC8` has the row of `ABC` (`microarchitecture.md` §6). The published sources name no 8-bit float format, so the rate is an opentsp assumption (no published circuit result). `mxm.fp8_rate`  selects 2 ABCs per plane and cycle, and `test/fp8_rate2` is that profile. | FP8 at the f16 rate (an even/odd plane pair, `timing.mxm_abc.occupancy_f16`) | It needs no new key: FP8 would use the f16 occupancy rule. The rate-2 alternative is the key `mxm.fp8_rate`, which is done: it moves the derived FP8 peak and the mapper's FP8 MAC rate (`opentsp-arch-spec/tools/perf/derive_peaks.py`, `model_map.py` with `weight_type` `fp8`, `performance.md` §2) and leaves the encoding as it is |
| 16 uniform ALUs per lane | Every VXM queue executes every VXM instruction (decision `queues.vxm_alu_kinds`, `microarchitecture.md` §5). The published floorplan carries the labels "Small-ALU" and "Big-ALU" without saying which unit they name or which operations need the large ones (ATPESC 2022, Track 1 Talk 4, p.20 and 21), and ISCA20 §III-C gives a 4 x 4 mesh of vector ALUs per superlane | A split in two kinds, for example the large ALUs in the first two positions of each of the four mesh stages, with Max, Min, Mul, the transcendentals and the float arithmetic on the large ones only (not implemented) | `issued_by` per queue in the Sail model and the catalogue's `units` per entry; a large-only instruction on a small queue would raise OTS-E-QUE-002; a kernel would have to place large operations on the large queues, and the VXM throughput of those operations would fall to the share of large ALUs |
| Accumulators per plane | `ACC` selects one of `mxm.acc_depth` accumulators (2 on `ref` and `mini`; a five-bit index, bit 12 as its low bit and bits 11 to 8 above it), each holding one 320-lane vector. The published sources do not settle the depth (`microarchitecture.md` §6, `abstractions.mxm_accumulator_depth` in `isa-db/provenance.yaml`) | More accumulators per plane, or an accumulator vector of several rows | The index field needs more bits, so the encoding, the catalogue, the Sail model, the ISS and the assemblers change together;  did that for 2 to 32 accumulators (`mxm.acc_depth`, profile `test/acc8`). A dot product longer than 320 over several activation rows keeps up to `mxm.acc_depth` partial rows in the plane instead of passing them through a stream and the VXM or MEM; the performance model counts the accumulator state (`performance.md` §2) and prices no `ACC` drain, so it does not yet show what the depth gains |
