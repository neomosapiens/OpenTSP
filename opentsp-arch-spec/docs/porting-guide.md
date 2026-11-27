# Porting guide: ISS and assembler

This guide is for engineers who implement or maintain an ISS and an assembler for the reference architecture: the ISS `opentsp-iss` and the assembler `opentsp-sw-toolchain` are such implementations. The artifacts below define the inputs for both ports. Their tests compare behavior against the executable model.

## 1. What to read

| Topic | Document |
|---|---|
| Geometry, stations, queue map | `topology.md` |
| Per-unit behaviour, timing, diagnostics, abstractions | `microarchitecture.md` |
| Every instruction (operation, units, timing class) | `isa.md` (generated) |
| Bit layouts, opcode allocation, decode rule | `encoding.md` (generated) |
| Assembly syntax, acceptance rules, program file and object format | `assembly.md`; the reference assembler `opentsp-arch-spec/tools/tsp_as.py` implements it and is the executable reference for every acceptance question |
| Sources and provenance | `opentsp-arch-spec/isa-db/provenance.yaml` |

Resolve instruction and state questions against Sail, external storage and host events against the normative harness, and profile-dependent behavior against the selected configuration. Report disagreements with the prose against the relevant part of this executable contract.

## 2. Machine-readable inputs

| File | Use |
|---|---|
| `opentsp-arch-spec/isa-db/generated/encoding.json` | Decode/encode tables and every text-level rule (§2.1). Generate the ISS decoder and the assembler's encoder from it; do not retype it. `test/isa_db/json_decoder.py` is a text decoder built from this file alone and compared with the model; `tools/tsp_as.py` is a complete assembler built from it |
| `opentsp-arch-spec/config/ref.json`, `mini.json` | Geometry, station gaps, latencies (`schema.json` validates them) |
| `opentsp-arch-spec/conformance/model/ref.json` | Golden vectors from the Sail model (§2.2; JSON Schema `conformance/model/schema.json`) |
| `opentsp-arch-spec/test/isa_db/corpus/<profile>/*.txt` | Assembler acceptance corpora with expected results (§3) |

### 2.1 `encoding.json`

Top-level keys (`schema` = `opentsp.isa-db.encoding`, `spec_version` = `0.1`). Every key that ends in `text` is prose for people; the other keys are the data.

| Key | Content |
|---|---|
| `slice_tags` | slice name → 4-bit tag (CU 0, MEM 1, VXM 2, MXM 3, SXM 4, C2C 5) |
| `eltype_codes` | element type name → 5-bit code (`i8` 0 … `b32` 10, `e4m3` 11, `e5m2` 12, `bf16` 16, an extension); 13..15 and 17..31 are reserved |
| `boolean_codes` | the codes of the boolean types (8, 9, 10), used by `bool_values` |
| `decode_rule` | the text-level decode rule (below): `narrow_tags` [0, 3, 5], `wide_tags` [1, 2, 4], `tag_bits` `30-28`, `bundle_bit` 31, the ordered `steps`, `no_match` (`OTS-E-DEC-003`) |
| `text_format` | the text of a queue (`encoding.md`, "Word formats"; `microarchitecture.md` §3): `unit_bits` 32, `byte_order` `little-endian`, `bundle_bit` 31, `tag_bits` `30-28`, `padding` `0xFFFFFFFF`, `nop` (tag 0, opcode 0, operand bits 15-0 in 1..65535, at the start of a parcel with bundle bit 0), `undecodable_unit` `0x6FFFFFFF`, and `lowering`, the normative function from a cycle view to text (`lower(entries)` of `tools/tsp_as.py`) |
| `field_rule` | how a field value is extracted and checked (below); `signed_kinds` lists the kinds whose `min`/`max` are two's complement (`simm16`) |
| `stream_encoding` | stream fields: `row_bit` 5, `row_codes` {E: 0, W: 1}, `id_bits` `4-0`, `ids` 32 |
| `profile_variables` | name → `expr` over the profile configuration; `vlanes` = `16 * geometry.superlanes` |
| `agt` | AGT placement: `loopable` (Read, Write, Gather, Scatter), `config` order (Countdown, Step, StreamStep, Iterations), `levels` 4 |
| `mem_slots` | instructions per parcel (`microarchitecture.md` §3, "Instruction front end"): `units` (the MEM queues), `slot_of` (instruction → slots: `fetch`, `R`, `W`, `config`; an instruction not listed needs the whole queue and is alone), `per_thread` (slots keyed by the instruction's `thread` field), `paths` (R 0, W 1) and `indirect` (references whose bank is unknown at dispatch: Gather, Scatter). The assembler, the object loader and the disassembler apply it |
| `sram_access` | `words`: instruction → number of SRAM words read from its `sram_addr` operand (IfetchM 2: a and a+1); 1 when not listed. Used for the address warning and the symbol bound |
| `stations` | per unit (every name of `units` except `any`) and row, the gap at which the unit reads and writes, as an expression over the profile's `layout` keys and the MEM quad; null where the unit cannot use the row (an operand there is DIR-001). What a scheduler needs to compute transit (`topology.md` §4); checked against the model on every data queue (`check_isa_db.py`) |
| `control_channel` | `queue` 65535 (0xFFFF) and the only `instructions` it accepts (NOP, Sync, Notify, Config) |
| `units` | unit name used in `instructions[].units` → list of queue ids; `any` = 0..143 (not the control channel) |
| `queues` | `names` (canonical name → queue id, including `CU` = 65535) and `aliases` (accepted on input, never printed) |
| `diagnostics` | every diagnostic code → its one-line meaning (from `model/core/diagnostics.sail`) |
| `operand_kinds` | kind → lexical class: `syntax` (`hex`, `dec`, `stream`, `name`), and optionally `group` (stream group size), `names` (name → code), `signed`, `symbol` (an address that may be a symbol) |
| `instructions` | one entry per encoding (below) |

An entry of `instructions` has `slice`, `name` (model name), `mnemonic`, `units`, `timing` (class), `assembly` (template with `{field}` placeholders, in operand order), `width` (32 or 64), `match` (`width` characters, MSB first: `0`/`1` fixed, `-` a field or reserved-zero bit) and `variables`. A variable has:

| Key | Meaning |
|---|---|
| `name`, `kind` | placeholder name in `assembly`; key of `operand_kinds` |
| `location` | bit range `msb-lsb` (or one bit `n`); bit 0 is the least significant bit of the word (of bits 31..0 for a 32-bit layout) |
| `hi_location` | optional second range that holds the **high** bits: value = word[`location`] \| word[`hi_location`] << width(`location`). The VXM `type` field: type = bits 23..22 (high) @ bits 53..52 (low); the MXM weight type: bit 12 above bits 21..20 |
| `top_location` | optional third range above the second: value = word[`location`] \| word[`hi_location`] << width(`location`) \| word[`top_location`] << (width(`location`) + width(`hi_location`)). The VXM `type`: bit 17 is bit 4 of the code (the type-extension bit); the Convert destination `dtype`: bit 16 above bits 27..24 |
| `extension_codes` | a map from an extension name to the entries of `codes` or `values` that are legal only in a profile whose `extensions` list names it (bf16: code 16, MXM type 4); a decoder that is given a profile applies it after `codes` / `values` (`encoding.json` `extensions` lists the names) |
| `codes`, `values` | the value must be one of these (`codes` for element types) |
| `not_values` | the value must not be one of these |
| `min`, `max` | inclusive range. For a kind in `field_rule.signed_kinds` the bits are read as two's complement first (Step: −8191..8191 in a 16-bit field on `ref`; a signed field with a `max_expr` is bounded in magnitude by it); otherwise unsigned |
| `max_expr` | inclusive upper bound computed from the profile: grammar `<variable> [ "-" <decimal> ]`, variable from `profile_variables` (`vlanes-1`, Shift amount; `sram_words-1`, `sram_words`: the SRAM address operands, SetSegment and Step). A variable is a product of factors, each an integer, a configuration key or `<int> ^ <key>` (`sram_words` = `2 ^ sram.addr_bits`). For a signed kind the bound is on the magnitude: −e..e |
| `streams`, `align` | on every stream operand (kinds `stream`, `stream_group16`): the number of consecutive streams it uses from its base, and the alignment of the base; the base needs base mod align = 0 and base + streams ≤ 32, else OTS-E-OPR-002. Expressions: an integer; `width(F)`, the byte width of the element type field F; `mxm_width(F)`, 2 when the MXM type field F is f16, bf16 or bfp8 (codes 1, 4, 5: an aligned pair of streams, OTS-E-OPR-002 on an odd base), else 1; `sg4 ? 4: E`; `n * n` (Rotate). Checked against the Python oracle's operand model (`test/isa_db/check_footprints.py`) |
| `bool_values` | when the entry's `type` field holds a code of `boolean_codes`, the value must also be one of these (Cmp on b8/b16/b32: eq, ne only) |

A `name` kind also requires the value to have a name in its table; a `stream` kind with `group` g holds (32 × row + id) / g. A word decodes to an entry when it matches `match` and every field is legal; every bit not in a field is reserved and fixed at `0` in `match`, so a word with a reserved bit set does not decode. At most one entry can match a word (checked pairwise).

**Text-level decode rule** (`decode_rule`; `model/postlude/iq.sail` `decode_text`), applied at the first unit of an instruction in a queue's text, in this order:

1. clear the bundle bit (bit 31 of the unit). The tag is bits 30..28 of the unit.
2. tag 0, 3 or 5 (CU, MXM, C2C): the instruction is this one unit; decode bits 31..0 (bit 31 clear) against the 32-bit entries.
3. tag 1, 2 or 4 (MEM, VXM, SXM): the instruction is this unit and the next unit of the queue's text; the first unit's bits 30..0 are word bits 62..32 and the second unit is bits 31..0; decode the 64-bit word against the 64-bit entries.
4. tag 6 or 7: OTS-E-DEC-003, one unit consumed.
5. A word that matches no entry is OTS-E-DEC-003. Then placement: on the control channel only `control_channel.instructions`, on a data queue only the queues of the entry's `units`; otherwise OTS-E-QUE-002. The queue does not affect decoding.

**Parcel rule** (`text_format`): the parcel is the instructions up to and including the first one whose bundle bit is 0, after any leading `NOP n` units with bundle bit 0 (each NOP is a timing prefix, not an instruction). The model's front end then decides when the parcel dispatches (`microarchitecture.md` §3).


### 2.2 Conformance vectors (`conformance/model/<profile>.json`)

JSON Schema: `conformance/model/schema.json`, checked by `test/isa_db/check_conformance_schema.py`. Generated by `tools/gen_model_vectors.py` from the model; `tools/model_test.sh` (golden vectors) checks they are current.

- Top level: `schema` = `opentsp.conformance.model`, `spec_version` = `0.1`, `profile` (the `config/<profile>.json` used), `config_sha256` (SHA-256 of that configuration as canonical JSON, keys sorted, separators `,` and `:`, after the keys the file leaves out are set to their defaults: today `ras.diag_depth` = 16, so a profile without the key and the same profile with 16 have one digest), `text` (a prose reminder of the vectors' `text` field), `vectors`.
- A vector: `name` (`directed-NNN`, `generated-NNN`, `coverage-NNN`, `clean-NNN` (a program that runs an entry without a diagnostic) or `link-NNN` (the  link-limit programs, LNK-001 and LNK-002)), `instructions` and `units` (the catalogue names and slices its text uses, sorted; a text instruction uses an entry when it decodes to it on a queue that may issue it, even if it then raises a diagnostic), `text`, `sram`, `dump`, `expect`.
- `text`: an object from the queue (decimal key: 0..143 is a data queue, 65535 the control channel) to the queue's initial instruction queue: its 32-bit units as 8 lower-case hex digits each, separated by one space, in program order (at most `icu.iq_depth` / 4 units, no padding unit; a queue with no text has no key). A vector has no cycle numbers: when each instruction dispatches follows from the text (`microarchitecture.md` §3).
- `sram`: the preload, `"<slice>:<address>"` (both decimal) → the word as lower-case hex, two digits per lane byte, lane 0 first, `vlanes` bytes. Every other word is zero at reset.
- `dump`: the `"<slice>:<address>"` words to read after the run.
- `expect.diagnostics`: every diagnostic in the order the model raises it, each `{cycle, queue, code}` (queue as in the `text` keys). `expect.words`: the dumped words, keyed as `dump`, in the `sram` format.
- `expect.trace_sha256` (every vector) and `expect.events` (the vectors with at most 64 events): the stream-register commits and SRAM writes of the run **with their values**, in canonical order (`microarchitecture.md` §12). The digest is the SHA-256 of the lines `T srf <cycle> <E|W>:<stream> gap <g> <value>` and `T sram <cycle> slice <s> addr <a> <value>`, each followed by a newline, each cycle's run of `T srf` lines sorted by (row, stream, gap); `expect.events` holds the same lines without `T `. A `<value>` is a space and `vlanes` bytes as lower-case hex pairs, lane 0 first.
- An implementation passes a vector when it produces exactly `expect.diagnostics` (same order), exactly `expect.words`, and a canonical trace with the digest `expect.trace_sha256`. That is the pass rule of the model and the ISS. A silicon RTL is held to the narrower obligation of [`platform.md` §7.1](platform.md#71-rtl-conformance), and `conformance/rtl/<profile>.json` (generated by `tools/gen_rtl_obligations.py`) says per vector whether it binds an RTL (`pass`), not (`none`) or only after a witness run (`undecided`).
- **`mxm.rows` above `vlanes`.** A profile whose `mxm.rows` exceeds `vlanes` is legal; the rows beyond `vlanes` only add buffer capacity. The advisory that says so is printed by `tools/schema_invariants.py` (an `advisory:` line of `--check`), and not by `derive_profile.py`, `tsp_sim` or `tsp-iss`. A silicon RTL is bound by the five comparisons of [`platform.md` §7.1](platform.md#71-rtl-conformance), on the vectors whose obligation is `pass`: the frozen vector files carry no such field, so `conformance/rtl/<profile>.json` (generated by `tools/gen_rtl_obligations.py`) lists `pass`, `none` or `undecided` for every vector, and [`bring-up.md`](bring-up.md) §2 says how to use it.
- **A profile whose `mxm.rows` exceeds `vlanes`** gets an advisory from `opentsp-arch-spec/tools/schema_invariants.py` (the rows beyond `vlanes` only add buffer capacity); `derive_profile.py`, `tsp_sim` and `tsp-iss` print nothing and accept the profile.
- **Profile and conformance.** The vectors depend on the profile: every cycle number follows from its latencies, gaps and constants, and the specification fixes no particular numbers. An implementation **declares its profile**, a configuration file valid under `config/schema.json`, and conforms when it matches the vectors that the frozen model generates from that file. `conformance/model/ref.json` and `mini.json` are those for the two shipped profiles; for any other profile (your own, or a test profile such as `config/test/ref_slow.json`), build `tsp_sim` and run `tools/gen_model_vectors.py <tsp_sim> <config.json> <out.json>`. The same programs are used, so the vectors differ only where the profile changes a result (cycles, hence also which cycle a value lands in). `gen_model_vectors.py --check` verifies a file against the model and `config_sha256` against the configuration. An implementation that is hard-wired to one set of latencies conforms to the profile of that set, and to no other.

### 2.3 Harness and trace protocol (`tsp_sim`)

`tools/build_sim.sh <out>` builds `tsp_sim` from the model (`c_emulator/tsp_sim.c`). An ISS that speaks the same protocol can be compared with it line by line. **`microarchitecture.md` §12 is the complete protocol**: every field, every range, the order of lines and the link FIFO rule. This section is the index. Where the two disagree, §12 decides.

Usage, as `tsp_sim --help` prints it: `tsp_sim <config.json> [--trace] [--observe] [--lint] [--on-irq stop|continue|replay] [--max-cycles N] [--chips K] [--window N] [--link-latency N | --link-latency-seed S | --link-jitter-seed S]`. `--on-irq` is the host policy for the error interrupt (`microarchitecture.md` §16.2): `continue` (the default) runs on, `stop` ends the run at the first error interrupt with exit code 5, and `replay` runs again with only the faults marked `p`. `stop` and `replay` need a single chip. `--window N` shortens the multi-chip lock-step window; the results do not depend on it. `--link-latency N` sets every C2C link to latency N, `--link-latency-seed S` draws each directed link's latency from [`c2c.link_latency_min`, `c2c.link_latency_max`] with a fixed generator, and `--link-jitter-seed S` draws a latency per transfer; at most one of the three may be given (`microarchitecture.md` §10). Input numbers and the `W` line are hex without `0x`. The numbers in the `E` and `T` lines are decimal, except the CSR address and value of `T csr`, which are hex.

Input, one item per line on stdin. Blank lines and lines starting with `#` are ignored. The words of an `H`, `P` or `F` line (`start`, `dma`, `db`, `csr`, `inject`, `r`, `w`, `sram`, `srf`, `iq`, `link`, `mxm`, `corr`, `uncorr`, and `e`, `w`, `t`, `p`) are lower case, not hex.

| Line | Meaning |
|---|---|
| `S <slice> <addr> <b0> … <bn>` | preload one SRAM word, lane 0 first (missing lanes are zero) |
| `Q <queue> <u0> <u1> …` | the initial instruction queue of a queue: 32-bit text units, hex, in program order (queue `0`..`8f`, or `ffff` = control channel); several `Q` lines of one queue append; a unit `ffffffff` is padding and is dropped |
| `D <slice> <addr>` | dump this word after the run |
| `C <chip>` | the following lines are for chip `<chip>` (with `--chips`) |
| `R <csr>` | print this CSR (address `0`..`fff`) after the run, as an `X` line (§12, `microarchitecture.md` §14) |
| `H start\|dma\|db\|csr\|inject …` | a host action on the current chip: the dispatch start cycle, a DMA vector into the ingress FIFO, a doorbell, a CSR read or write, or text pushed into a queue (§14) |
| `P csr\|ctrl …` | the chip's control unit sends a csr or ctrl packet on a connected link (§10, §14) |
| `F sram\|srf\|iq\|link\|mxm …` | fault injection: an ECC bit of an SRAM word or a stream register, a bit of instruction text, a link-FEC flit (`corr` or `uncorr`), or a bit of a matrix-unit sum. An optional `t` or `p` marks the fault for `--on-irq replay` (§15, §16, §17, §18) |

An `H`, `R`, `P` or `F` line acts on the chip of the last `C` line before it (chip 0 when there is none). The ranges (slice, address, queue, bit numbers, flit numbers) and the input-error messages are §12 and the harness contract of §10. Anything else is an input error (stderr, exit 2).

Output on stdout, after the run, in the order §12 gives:

- With `--observe`, each `O` line identifies a site at the tick where it was observed. See [Debugging and profiling a dynamic instruction](#debugging-and-profiling-a-dynamic-instruction) for fields and examples.
- With `--trace`, one line per event: `T srf`, `T sram`, `T diag` (a diagnostic; these lines are the `expect.diagnostics` of §2.2), `T ecc` (an ECC, link-FEC or matrix-check; not part of `trace_sha256`), and the host lines `T dma`, `T irq`, `T replay`, `T csr` and `T pkt`. A `<value>` on `T srf` and `T sram` is `vlanes` bytes, lane 0 first, lower-case hex, with no padding to 320 lanes. The order of one cycle's `T srf` lines is not specified; the canonical order is ascending (row, stream, gap), `E` before `W` (§12).
- `E <first-code|-> <first-cycle> <count>`: the first diagnostic and its cycle (`- -1 0` if there is none), and how many diagnostics there were. Printed with or without `--trace`.
- `W <slice> <addr> <b0> … <b(N-1)>`: one line per `D` line, two hex digits per byte. The line has max(320, vlanes) bytes; lanes at or beyond `vlanes` are `00` (on `mini`, bytes 16..319; on `sl32`, all 512 bytes carry data).
- `Y <cycle> <bytes>`: one line per vector `PciSend` gave the host, `vlanes` bytes, lane 0 first.
- `X <csr> <value>`: one line per `R` line, the address as three hex digits and the value in hex.

With `--lint` (independent of `--trace`; either, both or neither), `tsp_sim` also prints, in the same order of events, `T lint <cycle> queue <q> <code>`: a **simulator lint** (`validation.md` §5), a note with a code `OTS-W-<area>-<nnn>`, for example the IQ underflow of `microarchitecture.md` §3. A lint is **non-normative**: it is not a diagnostic, it is not counted in the `E` line, it is not in the golden vectors, and **an implementation need not report any lint**. Do not compare lint lines when checking an implementation against the vectors, or against `tsp_sim` unless you have implemented them; a differential test that does (the oracle’s `--impl-lint`) says so. The hardware behaviour a lint reports (for example that a parcel arrived late) is normative. To add a lint to the model, see `validation.md` §5.

With `--chips K > 1`, each chip's output is preceded by `C <chip>`, chips in ascending order.

The run ends when the model is quiescent: nothing is left in any IQ that can still dispatch, no fetch is in flight, and nothing is pending. Exit codes: 0 success, 2 input error, 3 `--max-cycles` exceeded (the output is incomplete), 4 an internal error, 5 an error interrupt ended the run under `--on-irq stop`, or a fault was still there in the second run of `--on-irq replay`.

### Porting a serving backend

Keep first-fatal evidence available to management after a guest clears status or masks an interrupt. For each attempt, collect settled detector reports from every participant and confirmation that its DMA writes reached private staging. Publish only when the matching attempt completes safely. On a fault, revoke or drain outstanding writers before reusing buffers and invalidate resident KV. The clocked gate assumes the board supplies trustworthy acknowledgements. Run `just ras-prove` for the gate and `just ras-qualify` for the two simulated deployments. [Interfaces and target-specific qualification](production-ras.md)

## 3. Acceptance gates

**ISS:**
1. Decode every instruction of a queue's text as the model does. Build `isa_query` with `ISA_QUERY_OUT=<path> tools/isa_db_check.sh` and compare your decoder with it. Use the text-level mode: feed it `text <u0> <u1>` lines (hex units; `u1` is used only for a two-unit instruction) and it answers `<code|-> TAB <units 1|2> TAB <text>`. The answer applies the §2.1 decode rule and gives OTS-E-DEC-003 or `-`; placement (OTS-E-QUE-002) is not included. The modes `32 <hex>` and `64 <hex>` decode a word at one fixed layout; use them only for single entries. There are no 128-bit words. `test/isa_db/corpus/<profile>/text.txt` (below) lists 1500 starts of text per profile with the expected decode-level code, the placement code and the text.
2. Run every vector of the committed file of your declared profile (`conformance/model/ref.json` or `mini.json`; for another profile, the file `gen_model_vectors.py` generates, §2.2) and produce exactly the expected diagnostics and words and a trace with the expected digest (stream and SRAM values included).
3. Then match random programs differentially against the Sail emulator (`tools/build_sim.sh`, protocol §2.3), with the same method as `test/oracle/diff_oracle.py` (Python): generate programs, run both, and compare every diagnostic, every dumped SRAM word, and the value of every `T srf` and `T sram` line; not the simulator lints, which an implementation need not report (§2.3, `validation.md` §5; `--impl-lint` compares them when the implementation has `--lint`). `diff_oracle.py <tsp_sim> config/ref.json <n> <seed> --impl "<your ISS>"` does this directly: it runs `<your ISS> <config.json> --trace` with each generated program on stdin, in place of the Python oracle, and compares its output with tsp_sim's (the generator needs the ref geometry). The comparison is line by line except within one cycle's run of `T srf` lines: the order of those is unspecified (`microarchitecture.md` §12), so both sides are first sorted into the canonical order (row, stream, gap) and the run is compared as a set, values included. Run it on your declared profile; `--impl` works on `config/ref.json` and, with the generator's ref-geometry programs fitted to it, on the other profiles the oracle supports (`config/test/ref_slow.json`). Your ISS may emit a cycle's commits in any order, but the `T srf` lines of cycle t must be consecutive, after the LNK-002 `T diag` lines and the IQ-002 `T lint` lines of t and before its other `T sram`, `T diag` and `T lint` lines, and all other lines must follow the order of §1 and §8.
4. **Multi-chip**. The golden vectors are single-chip. Run `test/c2c/c2c_directed.py <your ISS> config/<profile>.json` with your ISS in place of `tsp_sim`: it uses only the protocol of §2.3 (`--chips K`, `--link-latency`, `--link-latency-seed`, `--link-jitter-seed`, `C <chip>` lines) and checks the topology wiring, per-link latency, the seeded generators and Deskew. `test/c2c/deskewrecv_directed.py` (`DeskewRecv`, the polling loop at program start) takes the same arguments: the ISS must hold the queue until the epoch boundary at which the vector has arrived, wake at those boundaries (a chip with nothing else to do runs on until its vector comes) and end the run with the queue held when none comes. `c2c_directed.py` also checks the harness contract (`microarchitecture.md` §10), so the ISS must also: put `line <n>` in its stderr message for an input error, with empty stdout and exit 2; put `max-cycles` in its stderr message on exit 3; skip idle cycles (a record at cycle 0x5F5E000 must finish within seconds); accept the test configuration with `c2c.links` = 11; and accept `--link-latency` within the active profile's inclusive `c2c.link_latency_min`..`c2c.link_latency_max` range (209..228 on `ref` and `mini`). The same test checks the two link limits: LNK-001 (the sender's `c2c.cycles_per_vector` window, a model rule) and LNK-002 (the receiver's `c2c.rx_depth` FIFO, a harness rule with the exact decision of `microarchitecture.md` §12, "Link FIFOs": it also needs the cycle of a dropped arrival to be a cycle the run reaches, even when the chip has nothing else to do). The golden vectors carry single-chip programs of both codes (`link-NNN`). A multi-chip run built from a manifest (`tsp_as.py mpack`, `mobjinput`, `assembly.md` §4) is the same run as one built from the objects with `C <chip>` lines. Multi-chip behavior also requires the dedicated C2C tests; single-chip vectors do not establish it.

**Standard error is generally not normative.** What a port must reproduce is stdout, the exit status and the files it writes. The text a program prints on stderr is a courtesy to its user: the only parts a test reads are the tokens named above (`line <n>` in an input-error message, `max-cycles` on exit 3) and the `<code>:` prefix of an assembler rejection. The rest of a message (quoting, the wording of a range or relation, the order of clauses) may differ between the reference harness, the ISS and the assemblers, and the differential harnesses treat it as free. The exception is the topology-file semantic diagnostics: they must agree as specified in [topology.md §7](topology.md), after normalizing executable and file-path prefixes. That exception does not make unrelated diagnostic wording normative.

**Assembler:**
1. Agree with the reference assembler `tools/tsp_as.py` on every text: accept what it accepts and encode the same word; reject what it rejects with the same code and the same line (the message text after the code is not normative: `"tsp"` against `'tsp'`, or two wordings of one range rule, are not a disagreement). The corpora are checked-in files in `opentsp-arch-spec/test/isa_db/corpus/<profile>/`. They are tab-separated, and `#` lines are comments:
   - `accept.txt`: `<text> TAB <width> TAB <hex>`. The hex is bits 31..0 for a 32-bit layout.
   - `reject.txt`: `<text> TAB <code>`.
   - `text.txt`: `<queue> TAB <unit0> TAB <unit1|-> TAB <code|-> TAB <text>`, all hex (`unit1` only for a two-unit instruction): the start of one queue's text through the §2.1 decode rule and placement. The text is `illegal32/64 0x…` for OTS-E-DEC-003.
   - `groups.txt`: `<group> TAB <queue> TAB <word0> TAB <code|-> TAB <text>` (group decimal; queue and word0 hex): the rows of one group number are the instructions of one parcel of a queue, with the model's group code per instruction (QUE-001, MEM-001 or `-`; `microarchitecture.md` §3, "Instruction front end"). `--against` also runs this file.
   - A rejected text may itself contain tabs (the lexical rule of `assembly.md` §1 rejects them), so split a `reject.txt` line at its **last** tab.

   `test/isa_db/asm_corpus.py --against "<your assembler>" [profile]` runs them. It calls `<assembler> insn --profile <p> <text>`, which must print `<width> <hex>`, or exit non-zero with `<code>:` on stderr. It also calls `<assembler> dis --profile <p> <file>` on a `Q` text file and expects the `tsp_as.py dis` listing with `; <code>` annotations. `asm_corpus.py --check` (run by `check_asm_examples.py`) fails if the files differ from what `tsp_as.py` produces now. `--verify <isa_query> <tsp_sim>` checks every line against the model. `--write` regenerates the corpora. `asm_roundtrip.py` additionally draws random corpora against `isa_query`.
2. Apply the lexical rules of `assembly.md` §1 exactly:
   - separators are spaces only, with zero or more around `,`;
   - comments are `;` only;
   - hex digits may be upper or lower case, and leading zeros are accepted;
   - out-of-range values are errors and never wrap.
3. Accept the program format of `assembly.md` §3, including `.extern` and symbol operands. Emit the versioned `.elf` object of §4: little-endian ELF32, an `OpenTSP` note with ABI 1, the profile and full SHA-256 digests of `encoding.json` and the profile configuration file, and one text image per queue (`assembly.md` §4: the normative lowering `lower` of `tools/tsp_as.py`, the NOP gaps, the bundle bit). Read and bind `.safetensors` weight packages (§4).
4. Loaders require OpenTSP ELF ABI 1 and the matching profile and digests. Unrelated ELF files are rejected.

The ISS uses `mxm.acc_depth` wide accumulators per MXM plane when `wide_acc` is enabled. Element types use the 5-bit code space in §2.1, including bf16 when its extension is enabled.

## 4. Keeping implementations aligned

Consume the generated encoding database and validate configurations with the same schema and timing relations. Match the specified trace/diagnostic protocol and run vectors, directed checks and differential programs on each supported profile. A behavior change must update Sail, catalogue/provenance, generated and hand-written documentation, reference tools, oracle, vectors, ISS and assembler together.

## Debugging and profiling a dynamic instruction

Use the optional `--observe` switch of the Sail harness or the ISS to obtain site records alongside any requested effect trace. Its text fields are decimal except where the source token is a label. Run the following from a built checkout.

```sh
printf 'Q 58 51c00000 01800001\nH db 7 0\n' |
  target/spec/tsp_sim opentsp-arch-spec/config/ref.json --trace --observe
```

```text
O 0 0 88 dispatch 0 image:0
O 0 0 88 HostWait 0 0
O 0 7 88 wait-host 1 image:1
O 0 8 88 dispatch 1 image:1
O 0 8 88 SetVL 8 0
E - -1 0
```

Removing the doorbell leaves queue 88 parked. Both runs produce the same standard effect summary. In the released run, unit 1 became eligible at cycle 1 and dispatched at 8. Seven elapsed chip cycles separate eligibility and issue. Idle cycles can be skipped, so counting printed wait lines would give the wrong answer.

| Columns after `O chip cycle queue` | How to read them |
|---|---|
| `dispatch earliest source…` | The due parcel leaves this IQ. Source tokens are in wire order, including timing prefixes and both units of a long instruction. Dispatch can include an instruction that subsequently reports a diagnostic. |
| `wait-host`, `wait-barrier`, `wait-poll`, `wait-repeat`, or `wait-resource`, then earliest and sources | At this visited edge the complete, due head is held. The label reports the first applicable reason in that order. A parked run can end without a subsequent visited edge. An incomplete parcel has no dispatch identity yet. |
| `fetch base units` | A completed payload was admitted. The per-queue production base counts 32-bit units, including padding. A source token `fetch:g:i` identifies the g-th admitted fetch after the initial image and its original offset i. |
| `repeat previous` | Reissue of this queue's previous dynamic operation. Follow the earlier issue-cycle link through prior repeats to its original parcel. |
| `entry dispatch_cycle reference_offset` | A functional element entered execute, before operand checks. The first field is the catalogue entry, e.g. SetVL. Use queue, dispatch cycle, entry and reference offset to join it to the parcel/AGT site. Slot order disambiguates same-entry elements that execute on the same edge. A diagnostic can preclude this stage. |

Pin the executable image, profile, host input and fault schedule. Identify a dispatch by run or replay attempt, chip, queue, cycle and ordinal in its parcel. The observer uses queue 144 for control, which the text-image protocol spells 65535. `image:i` indexes the image supplied to the model. The text harness removes `ffffffff` padding from initial `Q` lines before assigning image offsets. Fetched payloads retain their original offsets across padding. A compiler can retain a side table keyed by image digest, queue and unit offset to resolve source, symbols and inlined locations. Dynamic fetch requires matching the actual payload to an emitted text fragment. For `host` or an unmatched payload report unknown source. Use observed issue cycles, operand/destination and ordered diagnostics to locate the consumer. A source-line number cannot be recovered from opcode bytes alone. Cold replay starts a new sequence of occurrences. Within a multi-chip trace use the chip field on each observation.

For an ISS embedding, `Observer::dispatch`, `wait(c,queue,why)`, `fetch` and `exec_begin` expose useful boundaries. The native statistics observer accounts for skipped cycles between holds. Its hold categories and denominators are defined in the ISS source. To measure queue delay use [first eligible, actual dispatch), make disjoint intervals when the reason changes, and identify simultaneous causes as such. For chip-wide stalled time take the union of intervals across queues under an explicit definition of idle. Summed per-queue waits have units of queue-cycles and can exceed elapsed chip cycles. Use execution and commit timestamps for functional latency. Use a host timer for simulator throughput. The ISS CLI emits the same site stream as the Sail harness. [ISS observer](../../opentsp-iss/crates/tsp-iss/src/observe.rs) · [Statistics implementation](../../opentsp-iss/crates/tsp-iss/src/harness/stats.rs)

The documented host interface can inspect selected SRAM words, CSR values and the outputs at stop. Quiescence can include a queue waiting forever for an external event. The harness's error-stop policy ends the run at an edge. Investigating an earlier state requires a new run with its inputs and observation enabled. Neither the object nor this observer defines a hardware debug port, architectural breakpoint, checkpoint or instruction-resume protocol.

### Joining a compiler listing and reconciling profile statistics

The two instructions in the HostWait example assemble as `VXM[0]: host_wait 0x0` in the cycle-0 bundle and `VXM[0]: setvl 0x1` in the cycle-1 bundle. For this object, a compiler side table can associate (queue 88, unit 0) with the HostWait source span and (queue 88, unit 1) with the SetVL span. At runtime `image:1` at dispatch 8 joins to the second span even though its scheduled bundle said 1. Bind the table to the object digest. When an instruction occupies two units, include both offsets in the span. The serving package need not carry this optional compiler metadata.

A barrier example can be compared directly in the two simulators. From the repository root run

```sh
printf 'Q 58 00400000 01800001\nQ 59 8 00800000\n' > barrier.txt
target/spec/tsp_sim opentsp-arch-spec/config/ref.json --observe < barrier.txt
target/release/tsp-iss run --profile ref --observe --stats=stats.json barrier.txt
```

Queue 88 dispatches Sync at 0. Its next instruction is eligible at 1. Queue 89 dispatches Notify at 8. With ref's release latency of 35 the next dispatch of queue 88 occurs at 43. Both simulators print one wait observation at 8. The ISS statistics report three dispatches, three visited ticks, last cycle 43, and 42 parked queue-cycles on queue 88. Subtracting eligibility 1 from actual issue 43 reproduces that count. Native ISS text input in this example contains only Q records. Timed H records use the harness input interface.
