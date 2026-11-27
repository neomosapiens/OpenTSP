# TSP reference architecture: performance model

This document gives the **derived** performance figures of a profile (peak rates, bandwidths, roofline knees, the weight-install bound), the published values they are compared with, and the analytic mapping of a transformer onto several chips. It is hand-written and informative: it adds no key, instruction or behaviour to specification 0.1, and the model has no clock. A figure is *derived* when a formula over profile keys gives it, *measured* when a command run on this repository gave it, and an *estimate* otherwise. Every figure below carries one of these three words or sits in a table whose caption names it.

**The clock is an assumption.** `config/schema.json` has no clock key and this document does not add one. Cycle counts and capacities do not depend on it; every rate in this document does. The published sources run the chip at about 900 MHz (ISCA20 §I); ISCA20's own throughput figures are quoted at 1 GHz. The tools take `--clock-mhz` (default 900) and label every output "derived; clock is an assumption".

## 1. Tools

| Tool | Does | Test |
|---|---|---|
| `opentsp-arch-spec/tools/perf/derive_peaks.py <profile.json> [--clock-mhz 900] [--json]` | the tables of sections 2 to 6 for any profile | `tests/perf/test_derive_peaks.py` |
| `opentsp-arch-spec/tools/perf/derive_profile.py [--superlanes N] [--rows R] [--slices-per-hemisphere M] [--set KEY=VALUE]` | a schema-valid profile from the `ref` template by stated, overridable rules; prints the rules and every number they changed | `tests/perf/test_derive_profile.py` |
| `opentsp-arch-spec/tools/perf/model_map.py <model.json> <profile.json> --chips N` | footprint, split, cycles, chip-to-chip cost and tokens per second of a decoder-only transformer (section 7) | `tests/perf/test_model_map.py` |

`tools/just/perf-model.just` defines `just perf-model` (the three tests, the fixture check and the activity-counter field check, about one to two minutes; it builds `tsp_sim` and the ISS when they are missing) and `just perf-peaks`; the root `justfile` pulls it in with `import 'tools/just/perf-model.just'`, and `just test` runs `perf-model`, so a change to a profile key, the schema or a perf tool cannot leave it red unnoticed.

`derive_profile.py` is a tool, not a model change. A profile is any legal value of the independent keys of the schema; no schema relation derives a latency from the geometry, and `mini` and the test profiles say so. The tool offers a convention so that "a chip with N superlanes, R MXM rows and M MEM slices per hemisphere" has a starting point: `mxm.rows` = min(16 N, 320) (320 rows is the schema's cap, so every N from 21 to 32 gets 320 rows unless `--rows` says otherwise: the planes are not square above 20 superlanes); `mxm.f16_rows` = `mxm.rows` / 2 at every width (the ratio of `ref`, 160 of 320; the keys allow half the rows to all of them, so `--set mxm.f16_rows=<rows>` gives the tandem capacity of, and a row count set by hand brings its half); with `--set sram.words_per_slice=W`, `sram.bank_bit` = log₂ W − 1 and `sram.addr_bits` = max(13, log₂ W), so 16,384 words give 14 address bits (rule R4); `c2c.cycles_per_vector` and `host.cycles_per_vector` keep their published byte rates for any width (rule R5, section 6); `timing.sxm_map_ready`, `timing.mxm_abc.d_func − d_skew` and `timing.sxm_cross.d_func − d_skew` (and the per-entry rows of Shift, Permute, ABC and ABC8) equal N, which is exactly what the schema relations need and what `ref` has; and the units outside the MEM quads keep their distance to them (`layout.*` gaps move with the number of quads). Every such default is printed as a "derived default" with the numbers it changed. It refuses a geometry outside the schema (today 1 to 32 superlanes, at most 44 MEM slices per hemisphere, at most 320 rows) with the key and the reason; `--allow-invalid` prints it labelled `NOT SCHEMA-VALID`. The latencies of the functional units (MEM, VXM, SXM d_func) remain circuit choices: the tool does not scale them with the geometry, because no published source gives a rule.

## 2. Derived peaks

Derived at an assumed 900 MHz unless the column says otherwise (`derive_peaks.py <profile> --clock-mhz 900`). V = 16 × `geometry.superlanes` lanes, R = `mxm.rows`, P = `mxm.planes_per_hemisphere` (1 or 2, so 2 P planes), F = `mxm.f16_rows`, A = `mxm.fp8_rate` (1 or 2), f = the clock. The four MXM structure keys of  enter the peaks as the formulas say, so `test/one_plane`, `test/fp8_rate2` and `test/f16_tandem` each give their own peaks (the table after the notes).

| Quantity | Formula | `ref` | `mini` | `ref` at 1 GHz |
|---|---|---|---|---|
| INT8 peak | 2 P × R × V × 2 × f | 737.3 TOPS | 1.843 TOPS | 819.2 TOPS |
| FP16 peak | 2 × F × V × 2 × f / `occupancy_f16` (0 when P = 1) | 184.3 TFLOPS | 0.4608 TFLOPS | 204.8 TFLOPS |
| FP8 peak | A × the INT8 rate (an opentsp resource assumption, `microarchitecture.md` §6) | 737.3 TOPS | 1.843 TOPS | 819.2 TOPS |
| SRAM capacity | present slices × `words_per_slice` × V | 220 MiB | 0.3438 MiB | 220 MiB |
| SRAM bandwidth (1 read + 1 write per slice per cycle) | present slices × V × 2 × f | 50.69 TB/s = 46.10 TiB/s | 2.534 TB/s | 56.32 TB/s = 51.22 TiB/s |
| Stream-register bandwidth | 2 × `streams_per_row` × V × f | 18.43 TB/s = 16.76 TiB/s | 0.9216 TB/s | 20.48 TB/s = 18.63 TiB/s |
| C2C per link, one direction | V / `cycles_per_vector` × f | 12.00 GB/s | 0.6000 GB/s | 13.33 GB/s |
| C2C, 16 links, one direction | the smaller of links × the per-link rate and 4 × V × f (the injection limit) | 192.0 GB/s | 9.600 GB/s | 213.3 GB/s |
| Host link | V / `host.cycles_per_vector` × f | 28.80 GB/s | 1.440 GB/s | 32.00 GB/s |

The link and host rates are also printed in bytes per cycle (`c2c_bytes_per_cycle_per_link` 13.33 and `host_bytes_per_cycle` 32.00 on `ref`), which do not depend on the clock. Present slices = 2 × `geometry.mem_slices_per_hemisphere` (`topology.md` §1: the other queue ids of `sram.slices` have no slice). A peak is the product of the unit counts and the clock; no program reaches it for every cycle (section 8 says what the counters show).

**The chip-to-chip total is capped by injection.** A chip has four transmit queues (two per hemisphere, `topology.md` §2) and a `Send` holds its queue for one cycle, so it injects at most 4 vectors per cycle: 4 × V × f = 1,152 GB/s on `ref` at 900 MHz (derived). The total printed is the smaller of that and the sum of the links, and its formula says which binds: 16 links of `ref` are far below it (192.0 GB/s); 96 links of 24 cycles per vector reach it exactly, so a profile with 128 links  prints 1,152 GB/s, not 1,536, and its chip-to-chip knee is 640 operations per byte, not 480. A narrow derived chip with 16 links is bound by the injection too: with one superlane (`c2c.cycles_per_vector` 2, rule R5) the links sum to 115.2 GB/s and the chip injects 57.6 GB/s (derived).

**The MXM structure keys.** Each of the four keys moves its own figure and no other (derived at 900 MHz; every figure is derived from the keys, none is measured; `test/acc8` is the accumulator profile):

| Profile | Key | INT8 peak | FP8 peak | FP16 peak | Install bytes | Accumulator state |
|---|---|---|---|---|---|---|
| `ref` | P 2, F 160, A 1, D 2 | 737.3 TOPS | 737.3 TOPS | 184.3 TFLOPS | 409,600 | 10,240 B |
| `test/one_plane` | P 1 | 368.6 TOPS | 368.6 TOPS | 0 (no plane can hold f16) | 204,800 | 5,120 B |
| `test/fp8_rate2` | A 2 | 737.3 TOPS | 1,474.6 TOPS | 184.3 TFLOPS | 409,600 | 10,240 B |
| `test/f16_tandem` | F 320 | 737.3 TOPS | 737.3 TOPS | 368.6 TFLOPS | 409,600 | 10,240 B |
| `test/acc8` | D 8 | 737.3 TOPS | 737.3 TOPS | 184.3 TFLOPS | 409,600 | 40,960 B |

The accumulator depth D = `mxm.acc_depth` is state (2 P × D × R × (4 + 16 · [`mxm.acc_bits` = 128]) bytes: the model's accumulators are vectors of `mxm.rows`, one 32-bit and one 128-bit form per accumulator when the wide accumulator is declared, 51,200 B on `test/ref_ext`) and sets no peak: this is a throughput bound that prices no `ACC` drain, so where D matters is a dot product longer than R, which keeps at most D partial rows in the accumulators before they pass through a stream, and that cost is not modelled. One plane per hemisphere installs from 16 streams (an IW carries 16 rows, so a plane takes 16 streams), so its install rate is half of `ref`'s and the 204,800 bytes take the same 20 issue cycles; the first ABC is at cycle 37 (derived: 19 + 1 + 5 + 12, the four nearest quads being nearer the matrix unit). This figure has not been run on `tsp_sim`.

**The extension types.** `derive_peaks.py` prints `bf16_tflops` and `bfp8_tflops`: each is the FP16 rate when `extensions` lists `bf16` or `block_fp8` and 0 otherwise, because both types take the capacity (a stream pair per row, `mxm.f16_rows` rows), the plane pair and the rate (`timing.mxm_abc.occupancy_f16`) of f16. **Storage and rate of bfp8.** a block-scaled FP8 model here has 2 bytes per weight on chip and 184.3 TFLOPS at the ref geometry when the extension is enabled, against 1 byte and 737.3 TOPS for FP8, so the format does not give the fewer bytes and the higher rate that block floating point usually serves. `model_map.py` accepts `weight_type` `bf16` and `bfp8` (2 bytes, the f16 rate) on a profile that declares the extension and refuses it elsewhere.

### Throughput per unit and type (`ref`, per cycle)

| Unit | Type | Rate per cycle | At 900 MHz | Basis |
|---|---|---|---|---|
| MXM, 2 P planes (4 on `ref`) | i8, u8 | 2 P × R × V = 409,600 MACC | 737.3 TOPS (a MACC is 2 operations) | `topology.md` §2; one ABC per plane per cycle |
| MXM, 2 FP16 matrices (planes 0 and 2; none when P = 1) | f16 | 2 × F × V = 102,400 MACC (2 × 160 × 320, `occupancy_f16` 1) | 184.3 TFLOPS | `microarchitecture.md` §6: a quarter of the INT8 rate; 204,800 MACC (368.6 TFLOPS) with F = 320 |
| MXM | fp8 | A × 409,600 MACC | 737.3 TOPS (1,474.6 with A = 2) | opentsp assumption (the INT8 capacity and timing, and A ABCs per plane and cycle); not a published circuit |
| VXM, 16 ALU queues | i8 to i32, u8 to u32, f16, f32 | 5,120 lane operations (16 × 320) | 4.608 Tops lane operations/s | one instruction per queue per cycle applied to all lanes (`microarchitecture.md` §5); the same lane rate for every type, because the model gives every entry its own latency, not its own rate; a chained operation counts as one per ALU. The published sources give no per-type VXM rate and none is claimed |
| VXM, 16 ALU queues, `.sg4` group forms of the 25 elementwise operations (`Add` to `Log`) | 8-bit types (i8, u8, e4m3, e5m2): 4 operations per lane; 16-bit types (i16, u16, f16, and bf16 with the extension): 2 per lane; 32-bit types: the plain operation | 20,480 operations (16 × 320 × 4) at 8 bits; 10,240 (16 × 320 × 2) at 16 bits | 18.43 Tops at 8 bits; 9.216 Tops at 16 bits | derived from `isa.md:40` (the group bit, `4 / width` independent operations per lane, one per byte or halfword of the 4-stream group); an estimate of the peak, not measured, and restricted to the types and operations that have the group bit (the conversions and `Reduce` have none) |
| MEM, 88 slices | bytes | 28,160 B read and 28,160 B written | 50.69 TB/s (read plus write) | one read and one write reference per slice per cycle, in different banks (`topology.md` §6) |
| Stream registers | bytes | 20,480 B (2 directions × 32 streams × 320) | 18.43 TB/s | `topology.md` §4 |
| Weight install | bytes | 20,480 B (2 hemispheres × 32 readers × 320) | 18.43 TB/s | section 5 |
| C2C | bytes | 13.33 B per link (320 / 24) | 12.00 GB/s per link | `microarchitecture.md` §10 (LNK-001) |
| Host | bytes | 32 B (320 / 10) | 28.80 GB/s | `microarchitecture.md` §14 |

SXM operations move one 320-byte vector per queue per cycle; the model has no figure for them that is not this issue rule.

## 3. Published values and deviations

The published column lists figures reported in the TSP literature. The derived column uses the OpenTSP ref parameters.

| Quantity | Derived (`ref`) | Published | Deviation |
|---|---|---|---|
| INT8 peak at 900 MHz | 737.3 TOPS | about 750 TOPS | −1.7 % (the published figure may include the VXM, a rounding, or a different clock) |
| INT8 peak at 1 GHz | 819.2 TOPS | 820 TOPS (ISCA20, quoted) | −0.1 % |
| FP16 peak at 900 MHz | 184.3 TFLOPS | 188 TFLOPS (quoted) | −2.0 % |
| SRAM capacity | 220 MiB | 220 MiB | none |
| SRAM bandwidth at 1 GHz | 56.3 TB/s = 51.2 TiB/s | 55 TiB/s (ISCA20 Eq. 2, quoted) | the published figure is 7.4 % above what its own parameters give (2 × 44 × 2 × 320 B × 1 GHz = 56.3 TB/s = 51.2 TiB/s): the paper's number is loose, and this model keeps the formula |
| Stream bandwidth at 1 GHz | 20.48 TB/s = 18.63 TiB/s | about 20 TiB/s (ISCA20, quoted) | agrees only if the unit is decimal TB |
| C2C per link | 12.00 GB/s (24 cycles per 320-byte vector) | 4 lanes × 25 Gb/s = 12.5 GB/s (ISCA22 §2.3, used for the profile's latencies); the capability of 4 × 30 Gb/s = 15 GB/s (ISCA20 §II) | −4.0 % against 25 Gb/s (23.04 cycles rounded up to 24), −20 % against 30 Gb/s |
| C2C, 16 links, one direction | 192.0 GB/s | 200 GB/s at 25 Gb/s; 240 GB/s at 30 Gb/s | −4.0 %; −20 % |
| Host link | 28.80 GB/s (10 cycles per vector) | PCIe Gen4 x16: about 31.5 GB/s of payload (16 GT/s × 16 lanes × 128/130), derived from the link standard, not a published TSP figure | −8.6 % (9.1 cycles rounded up to 10, `microarchitecture.md` §14) |
| Barrier release | 35 cycles (`barrier.release_latency`) | 35 cycles | none (the key is the published value) |
| Weight install (section 5) | first ABC after 41 cycles; arrivals spread over cycles 21 to 40 | "less than 40 cycles" (so at most 39), counting convention undefined | 41 exceeds "less than 40" by 2 cycles on the first-read to first-ABC convention; met by the 20-cycle arrival spread |
| ResNet-50 latency and throughput (ISCA20) | not reproduced: no workload in this repository | "less than 49 µs" per image and 20.4 K images per second at batch 1 in the introduction (section I); "< 43 µs" for the query and "an inference latency of only 49 µs for a single image sample" in the discussion (section V): the source quotes both figures | not compared |

## 4. The roofline

The knee of a unit is its peak rate over a byte rate: the arithmetic intensity (operations per byte) above which the peak, not that byte rate, binds. Derived on `ref` at 900 MHz:

| Knee | Value | Reading |
|---|---|---|
| INT8 against SRAM bandwidth | 14.55 operations per byte | below this a kernel is limited by memory references |
| FP16 against SRAM bandwidth | 3.636 FLOP per byte | |
| INT8 against stream bandwidth | 40.00 operations per byte | the matrix unit consumes operands from stream registers |
| FP16 against stream bandwidth | 10.00 FLOP per byte | |
| INT8 against C2C (16 links, one direction) | 3,840 operations per byte | work that crosses chips needs this much arithmetic per byte moved |
| INT8 against the host link | 25,600 operations per byte | |

The weight-stationary matrix unit makes the stream knee concrete: an INT8 weight byte is installed once and then used by one MACC (2 operations) per activation vector, so a layer is bound by the weight install until each installed weight serves 20 activation vectors (the 40 operations per byte of the table); an FP16 weight is two bytes, and the break-even is 10 vectors. A decode step of a single sequence therefore sits far below the knee, whatever the model: it is install-bound (or, on the measured program, latency-bound). A prefill of 20 or more tokens per weight installation reaches the MACC peak.

## 5. Weight-install lower bound

`derive_peaks.py` prints the time to install all planes (2 P × R × V bytes, 409,600 on `ref`) from the MEM slices nearest the matrix unit. A slice is one reader per stream and hemisphere, so 2 × readers × V bytes arrive per cycle (readers = min(`streams_per_row`, `mem_slices_per_hemisphere`, 16 P) = 32: an IW carries 16 rows, so a plane takes 16 streams; the formula and the fixtures below vary `timing.mxm_iw.d_func` and `timing.mem.d_func`):

- issue cycles = ⌈install bytes / (2 × readers × V)⌉ = 20;
- transit = `layout.mxm_in_gap` − (`layout.mem_down_gap_base` + quads − used quads) = 43 − 27 = 16 gaps from the farthest of the used quads (quads 3 to 10 on `ref`) to the matrix unit;
- first arrival = `timing.mem.d_func` + transit = 21; last arrival = first arrival + issue cycles − 1 = 40; the last install lands `timing.mxm_iw.d_func` cycles after the last arrival, and the first ABC that sees every installed weight issues then: the **first ABC = (issue cycles − 1) + `timing.mxm_iw.d_func` + `timing.mem.d_func` + transit = 19 + 1 + 5 + 16 = 41**. This is the formula of the directed test `test/directed/published_bounds.py`, which measures the same cycle on `tsp_sim`; on `ref` with `timing.mxm_iw.d_func` 3 both give 43 (`tests/perf/profiles/iw3.json`, `tests/perf/test_derive_peaks.py`).

Checked by running, with a schedule written for the purpose (the commands are in the report of this change, not in the repository): on `ref` a schedule with the first ABC at 41 runs on `tsp_sim` without a diagnostic, and the same schedule built for a MEM latency one cycle shorter (first install word read one cycle late) is OTS-E-TIM-001; on `test/ref_slow` (MEM `d_func` 7) the formula gives 43, a schedule at 43 runs clean and the schedule built for `d_func` 6 is OTS-E-TIM-001. These runs show that the first arrival cannot be earlier and that the formula's value is reachable; they do not show that no schedule can issue the first ABC earlier than the formula says, and they do not replace the directed test of the published bound.

The published statement ("less than 40 cycles", including SRAM and network transit) has no defined counting convention. By the arrival spread (first to last arrival) the model needs 20 cycles; by first read to first ABC it needs 41, which exceeds "less than 40" (at most 39) by 2. The shipped latencies are kept: MEM `d_func` 5 is an implementation choice. A profile with MEM `d_func` 3 meets the bound by this formula (the first ABC at 39), and `published_bounds.py` runs that variant of the profile on every `model_test.sh` run and measures 39.

## 6. Profiles of other sizes

`derive_peaks.py` runs on any profile, including a derived one:

```
python3 opentsp-arch-spec/tools/perf/derive_profile.py --superlanes 4 --rows 64 --slices-per-hemisphere 8 > /tmp/p.json
python3 opentsp-arch-spec/tools/perf/derive_peaks.py /tmp/p.json
```

A profile that the schema refuses cannot be loaded by the harness, so its figures are only the arithmetic of the formulas: the schema allows any count from 1 to 32 superlanes.

**Link and host rates do not follow the width of the chip.** `c2c.cycles_per_vector` and `host.cycles_per_vector` count cycles per *vector*, and a vector is 16 × superlanes bytes, so a profile that widens the vector and keeps the 24 and 10 of `ref` gives the links more bandwidth for free: the shipped conformance profile `test/sl32` has 19.20 GB/s per link and 46.08 GB/s of host bandwidth, 60 % above the published link rates (4 lanes at 25 Gb/s = 12.5 GB/s, 30 Gb/s at the most = 15 GB/s, ISCA22 §2.3 and ISCA20 §II; PCIe Gen4 x16, about 31.5 GB/s of payload). `derive_peaks.py` prints a note when a rate is above those figures, and `model_map.py` warns. `derive_profile.py --superlanes N` applies rule R5, so that a derived chip has the published byte rates: `c2c.cycles_per_vector` = ⌈16 N / 13.89⌉ (12.5 GB/s at the assumed 900 MHz; 24 for 20 superlanes, 37 for 32) and `host.cycles_per_vector` = ⌈16 N / 32⌉ (32 bytes per cycle of the PCIe Gen4 x16 link at 900 MHz, `microarchitecture.md` §14; 10 for 20 superlanes, 16 for 32). The conformance profiles under `config/test/` keep their values: they are test profiles, not design points, and no figure of this document is a figure of them. The question "is a wider chip more input/output bound?" is answered with a derived profile, not with `test/sl32` (section 7).

The serving programs written for the vLLM binding (`opentsp-vllm`) run on `ref`-geometry profiles only (20 superlanes, 320 MXM rows, 44 MEM slices per hemisphere); another geometry is refused with the first number that differs. The showcase generators derive their layout from the profile and build and run the batched serving step on `test/sl32` and `test/sl8` as well (`tests/showcase/tinystories/test_serve_geometry.py`, bit for bit against the host reference; `docs/kernel-cookbook.md` §7), so the programs are generalised to the geometries that table lists and not to every legal profile. Profiles that the table does not list, derived or not, are for the architecture tools, the directed tests and the oracle, so no figure of this document is a serving figure of any other profile.

## 7. Mapping a transformer onto chips

`model_map.py <model.json> <profile.json> --chips N` takes a decoder-only transformer (`layers`, `hidden`, `heads`, `kv_heads`, `ffn`, `ffn_matrices` 2 or 3, `vocab`, `context`, `weight_bytes` 1 or 2, `kv_bytes` 1 or 2, optionally `weight_type` (`int8` or `fp8` for 1 byte, `f16`, `bf16` or `bfp8` for 2: the MAC rate the weights use, so `mxm.fp8_rate` moves an FP8 model and `mxm.f16_rows` and `mxm.planes_per_hemisphere` an f16 one; an f16 model on a profile with one plane per hemisphere is refused, since no plane can hold f16), and optionally `act_bytes` 1, 2 or 4, default 2: the bytes per element that cross a link, which is the width of a partial sum in an all-reduce, of an activation in a pipeline hop and of a logit in the all-gather; the default is an f16 partial sum, because the INT8 products accumulate in 32 bits and the tool prices no conversion, and `act_bytes` 4 sends the accumulators unconverted; 1 needs a requantisation that is not modelled and is a sensitivity run only). It computes the footprint (weights plus the KV cache of `--batch` sequences against 220 MiB of SRAM per chip, `--sram-fraction` to reserve some), and for three splits: a layer pipeline, tensor parallelism inside every layer, and a hybrid of pipeline stages that are tensor groups inside one node (groups of 2, 4 or 8 chips; the best grouping is reported). For each: the weight and KV install cycles of a layer from the derived install rate, the MACC cycles from the derived MACC rates, the chip-to-chip bytes and cycles per token, and tokens per second with the binding bound named.

**The wiring the collectives are priced on.** The default is the wiring `tsp_sim --chips 8N` builds (`topology.md` §7): nodes of 8 chips, fully connected by 7 local links, and one global link between every pair of nodes on global ports numbered 4 per chip, so that with 8 nodes two chips of a node own its seven global links. A chip has no link to a partner in another node: data crosses through the chip that owns the global link, and each receive-then-send turnaround costs a relay of `timing.c2c.d_func` + `layout.c2c_tx_gap` − `layout.c2c_rx_gap` = 34 cycles (a relay of two hops measures 2 L + 34 = 466 cycles, `test/c2c/relay_directed.py`). An all-reduce over more than one node is hierarchical: an all-reduce inside each node, the node sum over the global links (one relay), an exchange between the owner chips (one relay), and the owners sending the result to the other chips of the node (one relay). For the 7B model's 4,096 partial sums of 2 bytes (8,192 bytes, 26 vectors) on 64 chips that is 840 + 3 × (34 + 840) = 3,462 cycles (with 1 byte per element, 13 vectors: 528 + 3 × (34 + 528) = 2,214), which the tool prints phase by phase. A pipeline hop between consecutive chips of one node is one link; across a node boundary it is three link hops and two relays. The all-gather of the logits is priced the same way. The link latency (`c2c.link_latency`, 216 cycles) and the cycles per vector (24) come from the profile; the reduction arithmetic is not counted, and neither is the conversion of the accumulators to f16 before the reduction. The tokens-per-second "with the pipeline full" of a pipeline or hybrid split needs one sequence per stage in flight, so the report gives it only with that number of sequences and the cache they need, and gives no rate when that cache does not fit (the 7B hybrid on 64 chips needs 8 sequences and 368.1 MiB per chip, and gives none).

Assumptions that bound the claims: the links of a chip transfer in parallel (a chip has four transmit and four receive queues, so it injects at most 4 vectors per cycle: the per-link rate of 24 cycles per vector is what binds up to 96 links, and the injection limit above them, section 2; the mapper applies the same limit: a direct exchange with more peers than 4 × `c2c.cycles_per_vector` takes the injection time of all its vectors at 4 per cycle, and `--links-per-hop` above that many links is priced as that many, with a warning); KV and attention shard only over the KV heads and heads (so a model with few KV heads stops sharding its cache early); the embedding table stays on the host; install and MACC overlap perfectly within a layer (the larger of the two counts); no overlap of a collective with computation; in the hybrid, one copy of the activations crosses between two stages and the first chip of the next group serves the others. **No 64-chip collective was run on `tsp_sim`: every collective figure is an estimate from the keys and the measured relay cost.** The limits are those of `topology.md` §7: a run takes 1 to 264 chips (`tsp_sim --chips`, the last node may hold fewer than 8), a topology file up to 512 (the 9-bit `chip_id`), and a profile has up to 128 links. A chip count above 264 is warned about (a run of that size needs a topology file), above 512 it cannot be addressed, more than 33 nodes need more global links per node than its 32 global ports give, a profile with fewer links than the wiring uses is warned about, and a partial last node is priced as a full one and said so; every warning goes to standard error and into the report.

`--topology flat` prices the same run with a direct link between every pair of chips (a ring, recursive doubling or a direct exchange), and `--topology FILE` reads the topology file that `tsp_sim --topology` reads (`chips` and `links`, as `tools/gen_topology.py` writes them): it derives the nodes (groups of fully connected chips), the global links per node pair and the global ports per chip from the link list, labels the wiring "simulator wiring", and takes `--chips` from the file. The dragonfly files (`gen_topology.py dragonfly K`, `node8`) are priced; a wiring that is not nodes of fully connected chips joined by global links (`torus R`, `tray8`) is refused with the reason, because the collectives are priced hierarchically on nodes and the tool has no pricing for a torus (open). `--wiring-params FILE` prices a parametric wiring given by `name`, `chips_per_node`, `global_links_per_node_pair` (default 1) and `global_ports_per_chip` (default 4). `flat` and the parametric wiring are **hypothetical** wirings that no harness builds, and every line of their report says so.

**Check against one measured program.** The model is a throughput bound. The only measured evidence is the TinyStories showcase model on one chip (8 layers, hidden 64, 64 KV slots, FP16 weights), decode of one token:

| Program | Measured | Model | Error |
|---|---|---|---|
| serial layer of the TinyStories block program | 11,362 cycles | 6 cycles | −99.95 % |
| 8-layer decode step  | 26,398 cycles | 363 cycles | −98.62 % |

The bound is far below the measurement because that program is latency bound: the matrix unit is busy about 2 % of the cycles and each dependent stage waits for a chain of unit latencies. A model with 3 M parameters has no throughput problem. `--layer-overhead-cycles` adds a measured per-layer floor to the model (the calibration 11,356 cycles reproduces the layer it was taken from by construction); using it for another shape is an estimate, because the length of a layer's dependency chain depends on the shape.

**Examples** (`tests/perf/models/dense-7b.json` and `dense-70b.json` are the shapes of published 7- and 70-billion-parameter decoders at 1 byte per weight, a 4,096-token context and a 2-byte KV cache; **every row is an estimate**, 900 MHz assumed, one sequence unless stated, the modelled wiring; the 70B rows beyond 264 chips carry the tool's warning):

| Model, chips, split | Fits (MiB per chip of 220) | Layer cycles | Tokens per second | Binding bound |
|---|---|---|---|---|
| 7B, 64, pipeline (32 layer chips) | no (382) | 13,159 | not given: does not fit | |
| 7B, 64, tensor parallel | yes (162.5) | 7,181 (257 install + 2 all-reduces of 3,462) | 3,856 | C2C collective latency |
| 7B, 64, hybrid (8 stages of tensor groups of 8) | yes (144.1) | 3,325 (1,645 install + 2 all-reduces of 840) | 7,287 | C2C collective latency |
| 7B, 64, tensor parallel, batch 16 | no (1,122) | 82,445 | not given: does not fit | |
| 7B, 40, tensor parallel | no (221.5) | 5,526 | not given: does not fit | |
| 70B, 400 (warning: beyond the 264 of a run, a topology file wires it), tensor parallel | no (323.8) | 9,127 | not given: does not fit | |
| 70B, 400 (warning: beyond the 264 of a run, a topology file wires it), hybrid (50 stages of tensor groups of 8) | yes (208.0) | 8,253 | 1,081 | weight and KV install |

At least 38 chips hold the 7B weights and one sequence's cache with a perfect split, and at least 304 chips hold the 70B model; 400 chips are addressable (at most 512) and can be wired by a topology file, but `tsp_sim --chips` stops at 264 and 50 nodes need more global links than 32 ports give, so the two 70B rows price the node wiring extended past what the harness builds and show only that a hybrid split changes the answer. The 7B tensor split on 64 chips is bound by the collective latency (two all-reduces per layer of 3,462 cycles, against 257 install cycles and 11 MACC cycles), not by arithmetic. The hybrid keeps the all-reduces inside a node (840 cycles); at the f16 payload it is bound by the collectives too, by a small margin (54,576 cycles of all-reduces and all-gather per step against 52,640 of weight and KV install), and at one byte per element the install binds. With pure tensor parallelism, sharding the cache across only eight KV heads exceeds the per-chip memory budget at 400 chips. The hybrid split uses 208.0 MiB per chip on the extended wiring.

**Payload and topology sensitivity.** The model defaults to two bytes per element for f16 partial sums. One byte per element assumes requantisation, whose work is outside the estimate. A hypothetical flat wiring uses a direct link to every partner. With one byte per element, its 64-chip tensor estimate is 3,168 cycles per all-reduce, 4,190 tokens per second. At two bytes it gives 5,040 cycles and 2,671 tokens per second for tensor parallelism, and 10,445 tokens per second for the hybrid. The simulated topology uses hierarchical collectives through the node links. The tool does not price chip-level recursive doubling relayed through those links.

Sensitivity to the payload (7B, 64 chips, `ref`, tokens per second, estimates):

| `act_bytes` | Meaning | All-reduce, tensor split | Tensor split | Hybrid split |
|---|---|---|---|---|
| 1 | requantised partial sums (requantisation outside the estimate) | 2,214 cycles | 5,915 | 9,101 |
| 2 (default) | f16 partial sums | 3,462 cycles | 3,856 | 7,287 |
| 4 | the 32-bit accumulators sent unconverted | 5,958 cycles | 2,273 | 5,209 |

Sensitivity to the width of the chip (7B, 64 chips, `act_bytes` 2, estimates; the derived profile is `derive_profile.py --superlanes 32` with rule R5, the shipped profile is the conformance profile `test/sl32`):

| Profile | Link and host cycles per vector | Link rate | Tensor split | Hybrid split |
|---|---|---|---|---|
| `ref` (20 superlanes) | 24 and 10 | 12.00 GB/s | 3,856 | 7,287 |
| derived, 32 superlanes | 37 and 16 | 12.45 GB/s | 4,054 | 8,914 |
| `test/sl32`, as shipped | 24 and 10 | 19.20 GB/s (above any published figure; the tool warns) | 5,363 | 10,647 |

A 32-superlane chip with the published link rates gains 22 % on the hybrid split and 5 % on the tensor split; the larger gains of the shipped profile (46 % and 39 %) are link bandwidth that no circuit supplies. Widening the chip makes the collectives a larger share of the step, because the arithmetic and install rates grow with the width and the link rate does not. No figure of this section is a measurement, and none is a serving figure.

## 8. Reading the activity counters

The activity counters are in this tree. `tsp-iss run --stats` writes them as the `activity` object of its JSON report, and the counters are off unless `--stats` or `--energy` is given (with `--energy` the report also holds the priced run, `energy`; `opentsp-iss/perf/README.md` describes both). The fields this section uses: `activity.active_cycles` (per unit class), `activity.mxm.abc_by_type.<type>.macs` (multiply-accumulates per operand type i8, u8, f16, fp8; `activity.mxm.macs` is their total), `activity.mem.read_bytes` and `activity.mem.write_bytes`, `activity.c2c.sent_vectors` and `received_vectors` (with `activity.c2c.per_link`), `activity.host`, `activity.vxm.element_ops` and the run length `last_cycle`. With `cycles` the run length (`last_cycle` + 1: cycle 0 counts, the length the priced energy uses) and `peak` the derived per-cycle rates of section 2:

- matrix-unit utilisation = counted MACCs (`activity.mxm.abc_by_type.<type>.macs`) / (`cycles` × the per-cycle MACC rate of the type), for example INT8 against 409,600 on `ref`; the measured showcase program is at about 2 %;
- memory-bandwidth utilisation = (`activity.mem.read_bytes` + `activity.mem.write_bytes`) / (`cycles` × 56,320 B on `ref`);
- C2C utilisation per link = counted vectors (`activity.c2c.per_link`) / (`cycles` / `c2c.cycles_per_vector`);
- arithmetic intensity of a run = counted operations / counted bytes moved at the level of the knee in section 4; a run to the left of that knee cannot exceed the bandwidth line, and a run well below both lines is latency bound, as the measured program is.

Energy figures, when they exist, are estimates with their calibration targets stated; no power semantics are part of specification 0.1.
