# TSP reference architecture: assembly language

The **instruction syntax** is the Sail `assembly` mapping (`opentsp-arch-spec/model/`) in the print direction. For every instruction word it defines the one canonical text. Sail 0.20.3 cannot execute the opposite direction (string → instruction): it rewrites a mapping clause that uses `^` into a match failure. So the **acceptance rules** are the written rules of §1–§3 below, and they are normative. The per-instruction templates are listed in `isa.md` and in `isa-db/generated/encoding.json` (`instructions[].assembly`, with the lexical class of each field in `operand_kinds`).

**Reference assembler:** `opentsp-arch-spec/tools/tsp_as.py` implements this document. Its grammar is derived from `encoding.json`; no instruction is hand-coded. The checks that hold it to the model are in §6.

**Order of authority**: the Sail model decides what a word means and prints; this document decides what text is accepted and which code a rejected text gets; `tsp_as.py` and the corpora of §6 are an executable reading of this document. Where `tsp_as.py` and this document differ, one of them has a defect, and a port follows this document.

## 1. Lexical rules

- **Characters:** a line is ASCII, comments included. Tokens are separated by the space character (U+0020) only. A tab or a non-ASCII byte anywhere on a line, including inside a comment, is an error (OTS-E-ASM-001). Lines end with LF; a CR is an ordinary non-space character, so CRLF files are rejected.
- **Comments:** `;` starts a comment that runs to the end of the line. There is no other comment form (`#` and `//` are not comments).
- **Mnemonics** are lower case. A type or mode suffix is joined with `.`, as in `add.i8`, `cmp.lt.f32`, `reduce.max.i16`, `convert.u8.f32`, `lw.f16`, `acc.i32`. Upper-case mnemonics and type names are not accepted.
- **Separators**, exactly as the Sail library (`mapping.sail`):
  - After the mnemonic: one or more spaces (`spc`).
  - Between operands: zero or more spaces, `,`, then zero or more spaces (`sep` = `opt_spc "," def_spc`).
  - Leading and trailing spaces on a line, and around a single instruction text (`tsp_as.py insn " sync "`), are ignored.
  - The printed form uses exactly one space after the mnemonic and `, ` between operands.

  So `add.i8 E:8,E:0,E:4` and `add.i8   E:8, E:0,  E:4` are accepted, and both print as `add.i8 E:8, E:0, E:4`.
- **Numbers.** These follow Sail's `hex_bits_N` and `dec_bits_N` parsers:
  - **Hex**: `0x` followed by one or more hex digits. Digits `a-f` and `A-F` are both accepted, as are leading zeros (`0x0010` = `0x10`). The value must fit the field. Output is lower case with no leading zeros (`0x0` for zero). `0X` is not accepted.
  - **Decimal** (stream numbers and the accumulator index): one or more digits `0-9`. Leading zeros are accepted (`E:007` = `E:7`; accumulator `01` = `1`). Output has no leading zeros.
  - No signs and no other bases. A negative step is written in 16-bit two's complement.
- **Operands:**

| Kind (`operand_kinds`) | Syntax | Range |
|---|---|---|
| `stream` | `E:n` or `W:n`, n decimal | n = 0..31; row E is eastward, W westward |
| `stream_group16` (LW, IW from streams, Transpose) | `E:0`, `E:16`, `W:0`, `W:16` | any other n is an error |
| `imm1/4/6/12/16/17`, `map_id`, `tile_size`, `chip_id` | hex | the field width (`imm1`: the liquid thread, 0 or 1); plus the instruction's `values`, `not_values` and `max` (`isa.md`) |
| `sram_addr` | hex, or a symbol `name` or `name+0x<off>` with no spaces around `+` (§3) | 0..`sram_words` − 1 (0x1fff on `ref`; 2^`sram.addr_bits` − 1 in general, 13..16 bits) |
| `simm16` (`step`) | hex, 16-bit two's complement | magnitude at most min(2^`sram.addr_bits` − 1, 32767): −8191..8191 on `ref`, i.e. `0x0`..`0x1fff` and `0xe001`..`0xffff` |
| `lane_count` (`shift` amount) | hex | `max_expr` `vlanes-1`, with `vlanes` = 16 × `geometry.superlanes` (`profile_variables` in `encoding.json`): 0..0x13f on `ref`, 0..0xf on `mini`, 0..0x1ff on a 32-superlane profile |
| `eltype_code`, `eltype4` | `i8 f16 i32 f32 u8 i16 u16 u32 b8 b16 b32 e4m3 e5m2 bf16` | the instruction's `codes`; `bf16` only in a profile whose `extensions` names it (else OTS-E-DEC-003) |
| `mxm_wtype` / `mxm_acctype` | `i8 f16 u8 fp8 bf16 bfp8` / `i32 f32 x128` | `bf16`, `bfp8` and `x128` only in a profile whose `extensions` names `bf16`, `block_fp8` and `wide_acc` respectively (else OTS-E-DEC-003; `microarchitecture.md` §6.1) |
| `chain` (VXM source operands) | empty, or `.chain` written directly after the stream, no space: `E:8.chain` | empty = 0, `.chain` = 1 (`microarchitecture.md` §5) |
| `group4` (VXM group bit) | empty, or `.sg4` written directly after the type, no space: `max.i8.sg4`, `cmp.lt.i16.sg4` | empty = 0, `.sg4` = 1: every operand is a 4-stream group (`microarchitecture.md` §5) |
| `cmp_mode` | `eq ne lt le gt ge` | on boolean types `eq` and `ne` only |
| `reduce_op` | `add max min` | |
| `shift_dir` | `up down` | |
| `acc_mode`, `acc_index` | `replace` / `accumulate`, decimal `0` to `mxm.acc_depth` − 1 (`0` / `1` on `ref` and `mini`; an index at or above the profile's depth is OTS-E-OPR-001) | |

- **Templates.** An `assembly` template of `encoding.json` (`isa.md`) becomes the grammar of its instruction: `, ` stands for a separator (`sep` above), a single space for one or more spaces (`spc`), `{field}` for one token of the field's kind, and every other character for itself. The `chain` and `group4` kinds have an empty or suffix form, so their placeholder follows the stream or type with no separator (`{lhs}{lhs_chain}`). A text must match a template completely.
- **Out-of-range values are errors. They never wrap.** A value wider than its field, a stream number above 31, a group base other than 0 or 16, an excluded value (`nop 0x0`, `repeat 0x0, 0x1`) or one not listed (`setvl 0x21`, `rotate E:0, 0x5, W:0`), a step outside −8191..8191, and an illegal type for the instruction are all rejected.
- **Static and run-time limits.** Range checks fall into two kinds:
  - **Static, decode-level.** The assembler rejects these because the model does not decode the word. All the limits above are of this kind. Decode bounds may depend on the profile: the Shift amount is below 16 × `geometry.superlanes`, the MEM address-field width is `sram.addr_bits`, and an ACC index is below `mxm.acc_depth`. The model decodes `shift` only when amount < the configured lane count. That limit does not change at run time, and **SetVL does not change it**. The same text is therefore legal on `ref` and illegal on `mini`. `setvl` accepts `0x1`..`0x20` on every profile; at run time a value above the configured superlane count is clamped to that count (`microarchitecture.md`).
  - **Run-time.** These decode, so the assembler accepts them, and the model reports them when the instruction executes. A conforming assembler must accept them; it may warn, with the model's code:
    - an SRAM address at or above `sram.words_per_slice`, or at or above the slice's SetSegment bound (OTS-E-OPR-001). `tsp_as` warns about the first case, for every word the instruction accesses (`ifetchm a` reads a and a+1, so it warns at a+1 ≥ `words_per_slice`);
    - a misaligned operand (OTS-E-OPR-002): a 16- or 32-bit operand, or any `.sg4` operand, whose stream is not a multiple of its width; an f16 `abc` source at an odd stream; an `acc` destination not a multiple of 4; a `rotate` with dst + n² > 32. The rule for every operand is `streams` and `align` in `encoding.json`; `tsp_as` does not warn about these;
    - a link at or above `c2c.links`, or a `chip_id` that is not the link's peer (OTS-E-OPR-001), and every condition that depends on data, unit state or timing (`microarchitecture.md` §8, "Execution checks").

## 2. Instructions

`isa.md` lists every template. Examples (each line is checked: §6):

```
read 0x10, E:0, 0x0            ; MEM: SRAM word 0x10 -> E:0 (liquid thread 0)
step 0x2                       ; AGT: stride 2 for the preceding read
iterations 0x2                 ; AGT: 2 references in total (reference 0 included)
gather W:3, E:1, 0x2, 0x0      ; dst W:3, map E:1, element (lane) 2 of each tile, thread 0
add_sat.u8 E:8, E:0, E:4
max.i8.sg4 E:0, E:0, E:4          ; four int8 maxes on the groups E:0..3, E:4..7 (one ALU)
convert.i32.f32 W:4, W:0
cmp.ge.i16 E:2, E:0, E:4
lw.f16 W:16
iw.i8 W:0                      ; IW from streams (no operand: from the LW buffer)
acc.i32 E:8, accumulate, 1
sxselect W:0, W:1, W:2, W:3, E:4 ; SXM select: sel 0 -> a, 1 -> b (north), 2 -> c (south)
repeat 0x36, 0x1               ; previous instruction 54 more times, 2 cycles apart (d + 1)
```

## 3. Program files (`.ots`)

Grammar (EBNF; `SP` is one or more spaces, `S` zero or more, `SEP` = `S "," S`, and every line may end with spaces and a `; comment`):

```
program   = { blank } arch profile { extern } section { bundle | blank } ;
arch      = ".arch" SP "opentsp-tsa" ;
profile   = ".profile" SP name ;                    (* config/<name>.json: ref or mini *)
extern    = ".extern" SP ident SEP "slice=" dec SEP "addr=" hex SEP "size=" hex ;
section   = ".section" SP "text" ;
bundle    = "bundle" SP "@cycle=" dec S "{" { slot | blank } "}" ;   (* "{" ends its line; "}" stands alone *)
slot      = queue S ":" S instruction ;              (* instruction: sec. 1 and 2; queue: any text up to the first ":" *)
ident     = letter { letter | digit | "_" } ;        (* at most 15 characters; letter includes "_" *)
```

The directives appear once each, in this order: `.arch`, `.profile`, the `.extern` lines, then `.section text`. An unknown directive is an error. Blank lines and comment-only lines may appear anywhere. The `.profile` name is a `config/<name>.json` of this specification (unknown: OTS-E-ASM-003). **Which profile applies:** `--config F` wins over everything (the object is assembled for F and carries its digest, whatever `.profile` says); otherwise the `.profile` line names the profile, and `--profile P` supplies one only for input that names none (an instruction text, a Q text, a manifest); an object names its own profile and `--profile` must equal it. `ots-asm asm` and `tsp_as.py asm` give the same object in each of these cases . Extern names are case-sensitive. A missing final line feed is accepted. A bundle with no slot is accepted and emits nothing, but its cycle still counts for the strictly increasing rule. A line that is not a directive, a bundle line, `}` or `<queue>: <instruction>` is OTS-E-ASM-001; an instruction text that matches no template is OTS-E-DEC-003 (§5).

**Bundles:**
- A bundle holds the instructions the compiler intends to dispatch at one cycle (§4, "Program text": the assembler lowers it to text, and the model dispatches the instructions of one queue in one bundle as one **parcel**). The instructions of one queue in one bundle follow the model's **slot rule** (`microarchitecture.md` §3, "Instruction front end"; `model/postlude/tick.sail` `place`/`classify`; `encoding.json` `mem_slots`):
  - a non-MEM queue takes one instruction per cycle;
  - a MEM queue takes at most one `ifetch` or `ifetchm` (the fetch slot); per liquid thread (the `thread` operand) one R-type reference (`read`, `gather`) and one W-type reference (`write`, `scatter`); and one AGT configuration instruction (`countdown`, `step`, `streamstep`, `iterations`). Any other instruction (`repeat`, `sync`, `notify`, `config`, `setvl`, `setsegment`, …) needs the whole queue and must be alone. A `nop` alone in its bundle on its queue is a timing instruction, not a dispatched one: it adds nothing to the text (§4, "Program text"), and a `nop` with other instructions of its queue in the bundle is a slot-rule violation.

  Instructions that do not fit (two in one slot, a whole-queue instruction with another) are OTS-E-QUE-001. An R and a W reference of the **same** thread must address different banks (address bit `sram.bank_bit`), and a `gather` or `scatter` clashes with any reference of its thread on the other path; otherwise both are OTS-E-MEM-001 (`microarchitecture.md` §4). References of different threads never clash. So `read 0x0, E:0, 0x0` + `read 0x1, E:1, 0x1` + `ifetch E:2` is one legal bundle slot set on `MEM[4]`. An `ifetchm` takes the R slot of **both** threads, so it fits with `write` references only.
  **IfetchM bank timing.** The per-parcel bank check uses address a in the issue cycle. Its second read at a+1 occupies the following cycle; conflicts with that read, and the two-cycle R-path reservation, are checked at runtime. A write in the issue parcel may use the bank of a+1 when it differs from the bank of a.

- Bundle cycles are decimal, below 2³², and **strictly increasing** through the file. Empty cycles are omitted.
- An instruction must be one its queue's unit executes, as in `isa.md` "Units" and `encoding.json` `units` (unit → queue ids) (OTS-E-QUE-002). On a profile with fewer than 44 MEM slices per hemisphere, every instruction on the queue of an absent MEM slice (`MEM[n]` with n mod 44 ≥ `geometry.mem_slices_per_hemisphere`) is OTS-E-QUE-002 as well (`microarchitecture.md` §3).
- The **control channel** `CU` takes only `nop`, `sync`, `notify` and `config` (`encoding.json` `control_channel`): every other instruction (`repeat`, `ifetch`, `ifetchm`, `setvl`, every MEM, VXM, SXM, MXM and C2C instruction) is OTS-E-QUE-002. A 64-bit instruction is not a separate case: it decodes by its tag and is QUE-002 like any other instruction the channel lacks.
- **AGT configuration** (`countdown`, `step`, `streamstep`, `iterations`) goes in the bundles of the following consecutive cycles, on the same queue as the loopable instruction (`read`, `write`, `gather`, `scatter`) it configures (`microarchitecture.md` §4, `encoding.json` `agt`). Each dispatch path (R, W) has its own AGT: a reference alone **on its path** opens that path's configuration (a reference beside an `ifetch`, or a `read` + `write` pair, opens one per path). A configuration instruction goes to the configuring path; when both are configuring, to the one that has received fewer, R on a tie, so a pair's instructions alternate R, W (the example of `microarchitecture.md` §4). A cycle of the queue without a configuration instruction ends the open configurations. The assembler follows these rules statically and **warns** with OTS-E-OPR-001 (the model's run-time code) when a configuration instruction has no open configuration (no reference alone on a path, nor a configuration instruction, on its queue in the previous cycle) or would open a fifth level, and with OTS-E-QUE-001 when a reference is placed on a path whose configuration is open. These are warnings, not errors, because the program is still a valid object; the model reports a stray or fifth-level configuration instruction when it runs, and a reference on a configuring path makes the model wait for the loop (it is not a model diagnostic: the warning tells the compiler that its cycle will slip).

**Externs and symbols:**
- `.extern name, slice=<n>, addr=0x<a>, size=0x<s>` declares `s` SRAM words at `a.. a+s−1` of MEM slice `n`. The loader fills them from a separate `.safetensors` weight; weights are never in the object.
- A symbol operand `name` or `name+0x<off>` may be used wherever an `sram_addr` is: in `read`, `write` and `ifetchm`. Its value is `a + off`, and the assembler checks that:
  - `off + n ≤ s`, where n is the number of words the instruction accesses (`encoding.json` `sram_access`: 2 for `ifetchm`, which reads a and a+1; 1 otherwise);
  - the instruction issues on `MEM[n]`, the slice the extern lives in.

  A violation, an undefined name, a duplicate name, an extern in an absent MEM slice (n mod 44 ≥ `geometry.mem_slices_per_hemisphere`), an extern outside the slice's `words_per_slice`, or two externs whose words overlap in one slice is OTS-E-ASM-002.

**Queue names** map to queue ids (`topology.md` §5; `encoding.json` `queues`). An index may carry leading zeros (`MEM[007]` = `MEM[7]`). Any other name, including a lower-case or spaced one (`cu`, `MEM [0]`), or an index outside its range, is OTS-E-ASM-005.

| Name | Queue id |
|---|---|
| `MEM[s]`, s = 0..87 | s |
| `MEM.W[i]`, `MEM.E[i]`, i = 0..43 (aliases) | i, 44 + i |
| `VXM[a]`, a = 0..15 | 88 + a |
| `MXM[p].LW`, `MXM[p].ABC`, `MXM[p].ACC`, p = 0..3 | 104 + 3p + 0, 1, 2 |
| `SXM.W.DST[k]`, k = 0..3 | 116 + k |
| `SXM.W.TSP[k]`, k = 0..1 | 120 + k |
| `SXM.W.SHF[k]`, k = 0..1 | 122 + k |
| `SXM.W.PRM` | 124 |
| `SXM.W.SEL` | 125 |
| `C2C.W.TX[k]`, k = 0..1 | 126 + k |
| `C2C.W.RX[k]`, k = 0..1 | 128 + k |
| `SXM.E.DST[k]`, `SXM.E.TSP[k]`, `SXM.E.SHF[k]`, `SXM.E.PRM`, `SXM.E.SEL`, `C2C.E.TX[k]`, `C2C.E.RX[k]` | the W id + 14 (130..143) |
| `CU` | 0xFFFF (control channel) |

The MXM sub-queue names are labels (LW for LW and IW, ABC, ACC; W elsewhere means west): any instruction of the plane can issue on any of its three queues. A disassembler prints `MEM[s]`, never the aliases.

**Example.** This program adds two words. It is checked on the emulator (§6): with `x` = 3 and `y` = 4 in every lane, SRAM word 0x9 of slice 44 becomes 7. MEM slices 4 and 5 read at cycle 33, so their values reach the VXM's station when it reads its operands at cycle 42, d_skew = 2 cycles after its dispatch at 40. The east slice 44 writes the sum when it arrives, at cycle 48 (station gaps and latencies: `topology.md` §4, `microarchitecture.md`).

```
.arch opentsp-tsa
.profile ref                          ; config/ref.json
.extern x, slice=4, addr=0x10, size=0x1
.extern y, slice=5, addr=0x0, size=0x1 ; placed by the loader (.safetensors)
.section text
bundle @cycle=33 {
  MEM[4]: read x, E:0, 0x0            ; = read 0x10, E:0, 0x0
  MEM[5]: read y+0x0, E:4, 0x0
}
bundle @cycle=40 {
  VXM[0]: add.i8 E:8,E:0, E:4         ; any spacing after ","
}
bundle @cycle=48 {
  MEM.E[0]: write 0x9, E:8, 0x0       ; MEM[44]
}
```

To run the example, save the block above as `add.ots` in a working directory. Set `bin` to the absolute path of the repository's `target/release` directory after building `ots-asm` and `tsp-iss`. The two input tensors each contain one 320-byte ref word.

```sh
python3 -c 'from pathlib import Path; Path("x.bin").write_bytes(bytes([3])*320); Path("y.bin").write_bytes(bytes([4])*320)'
"$bin/ots-asm" asm -o add.elf add.ots
"$bin/ots-asm" wpack --profile ref -o add.safetensors x=i8:x.bin y=i8:y.bin
"$bin/tsp-iss" run --weights add.safetensors --dump 2c:9 add.elf
```

Expect `E - -1 0` and a `W 2c 9` row containing 320 bytes of `07`. The dump uses hexadecimal slice numbers, so `2c` selects the decimal slice 44 named by the program. To reuse the program with another set of inputs, replace the tensor files and run the package and execution commands again.


## 4. Objects

**Program text** (the unit of assembler output, [Word formats](encoding.md#word-formats)). An object carries, for each queue that has instructions, the **text image** of that queue: its 32-bit units in program order, in the format of `encoding.md` ("Word formats") (one unit per 32-bit instruction, two per 64-bit one with the high half first, bit 31 the bundle bit, `NOP n` as the unit n, `0xFFFFFFFF` the padding that is never enqueued). The program has no cycle numbers: time is in the text. The assembler lowers the bundles with the normative function `lower(entries)` of `tsp_as.py` (the model's tests and every generator use it):
- the entries of one queue are ordered by bundle cycle, and the instructions of one bundle are **one parcel**, in **source order** (not in word order: the assembler sorts the cycle view stably by (cycle, queue)), every instruction but the last with the bundle bit set; a disassembler lists a parcel in image order, so the listing of an object reassembles to the same images (and the same cycle view);
- before each parcel come the `NOP n` units for the gap g = (the parcel's cycle) − (the previous parcel's cycle) − 1, where the previous cycle is −1 before the first parcel: n ≤ 65535 each, none when g = 0;
- a 64-bit word (the cycle-view form of an instruction, a 32-bit instruction being in bits 31..0) that fits neither layout lowers to the unit `0x6FFFFFFF` (tag 6: it decodes to nothing, OTS-E-DEC-003);
- a `nop` that is alone in its bundle on its queue adds no unit (its time is in the gaps); a `nop` with other instructions of its queue stays in the parcel (a slot-rule violation, OTS-E-QUE-001);
- the images are in ascending queue id with `CU` = 0xFFFF last; a queue without instructions has no image.

**Theorem** (tested, `test/directed/frontend.py`): for a program none of whose instructions meets a wait condition (`microarchitecture.md` §3: a parked queue, a busy queue or path, a running loop or Repeat) and whose text is in its IQ when due, the model dispatches every instruction at its bundle's cycle. For a program that waits, the model dispatches later and shifts what follows (the bundle cycle is the cycle the compiler intends). Because the text of a queue is limited to `icu.iq_depth` / 4 units as an initial image, the assembler warns (OTS-E-ASM-003) when an image is longer, and a loader refuses it (below): the rest of the program is brought in by `ifetch` and `ifetchm` in the program itself (the compiler's omniscient prefetching, ISCA20 §III-A3), from SRAM preloaded through externs. The fetched text has the same format; padding fills the unused part of a fetched vector. **Fetching a word that was never initialised** delivers zero units: a zero unit is `NOP 0`, which does not decode (the count of a NOP is 1..65535), so every unit is OTS-E-DEC-003 at its own dispatch cycle, one report per unit (80 units per word on `ref`: one `ifetchm` of two zero words is 160 reports). The first report is the mistake and the others follow from it; the unused tail of a block must be padding (`0xFFFFFFFF`), never zero.

Text form (the `Q` lines of the `tsp_sim` input protocol, all fields hex, without `0x`): `Q <queue_id> <unit> <unit> …`, queue 0..8f or ffff, each unit at most 32 bits; several lines of one queue append (the assembler writes at most 64 units a line). A text file holds only `Q` lines and blank lines (it has no `S`, `D`, `C` or `#` lines, which only the harness reads); a file is read as text when its first non-blank byte is `Q`, else as a binary object.

**Binary object** (`.elf`). The resolved program is little-endian ELF32 with a 52-byte header, `ET_REL` and `EM_NONE`. It has no program headers, symbols or relocations. The section directory and names follow ELF32. OpenTSP names its private sections as follows.

| Section | Uncompressed contents |
|---|---|
| `.note.opentsp` | ELF note type 1 with owner `OpenTSP` including its terminating NUL. Its 84-byte descriptor has ABI u16 = 1, reserved u16 = 0, the NUL-padded 16-byte ASCII profile name, the 32-byte SHA-256 of the shipped encoding database, and the 32-byte SHA-256 of the profile file. |
| `.tsp.iq` | A u32 image count, then each queue's entry (queue u16, reserved u16 = 0, unit count u32), then each image's contiguous u32 instruction units in entry order. |
| `.tsp.load` | A u32 extent count, then one 32-byte record per extern in source order. The fields are name (16 bytes, ASCII and zero padding), slice u16, reserved u16, word address u32, word count u32 and reserved u32. Reserved fields are zero. |
| `.tsp.text` | Optional named fetched text extents, encoded as physical Safetensors. |
| `.tsp.const` | Optional named program literals in the same physical representation. |
| `.tsp.zero` | Optional zero-filled extents. Each record has a 16-byte name followed by a u32 event mask. Mask 1 means cold or recovery load. Mask 3 additionally means variant installation. |
| `.tsp.recipe` | Optional workload-specific logical tensor bindings. The workload loader resolves this section to physical words before installing the program. |

The writers compress payload sections independently with `SHF_COMPRESSED` and `ELFCOMPRESS_ZSTD` (type 2). Each section expands to at most 64 MiB. The loaders accept uncompressed payload sections as well. Every section must be unique, bounded by the ELF directory and nonoverlapping. Files with the same compiler, input and compressor emit deterministic bytes. Consumers that need compression-independent identity hash decoded sections in name order, with each name terminated by a zero byte and each payload preceded by its u64 length.

An invalid ELF directory, ABI note, interface identity, configuration, image or extern is OTS-E-ASM-003. Queue images have ascending queue order with CU last, positive unit counts, no duplicates, and no more than `icu.iq_depth / 4` non-padding units. The slot rules of §3 apply to every parcel. Same-bank R + W is reported when the model runs. Externs are unique identifiers of at most 15 characters within the selected SRAM slice and address range. Externs on one slice may touch at their boundaries.

A load source binds to an extern of the same name and exact word count. A program can draw from embedded program data and separate resident weight data. No name can come from two sources. Direct native and reference CLI loading emits the selected extern writes in placement order before the initial queue images. An IQ-only interface loads the queue images without SRAM side effects. Serving loaders may defer paged text until the corresponding bucket runs and preserve resident KV across variant installation. Pending execution, commits, fetches and external events drain before final results are collected (microarchitecture §§3 and 12).

**Manifest** (`.otsm`, little-endian, format revision 2). A manifest is the multi-chip container: it names the K chips of a run and embeds one object per chip, all of one profile. It has a 64-byte header, then K entries of 8 bytes, then the K embedded `.elf` objects back to back in chip order:

| Offset | Field | Value |
|---|---|---|
| 0 | magic | `OTSM` |
| 4 | spec version major, minor (u8, u8) | 0, 1 |
| 6 | header size (u16) | 64 |
| 8 | profile (16 bytes: ASCII, then NUL bytes only) | the profile of every embedded object |
| 24 | chip count K (u32) | 1..512: a reader takes any count of a topology file; a writer takes 1..264 (the counts `tsp_sim --chips` takes) or, with `mpack --topology`, the `chips` of that file |
| 28 | topology (u32) | 0: the manifest names no wiring; the harness takes `--chips K` (`microarchitecture.md` §10) or `--topology FILE` (`topology.md` §7) at run time, and the chip count must be K (the only value of 0.1) |
| 32 | format revision (u32) | 2 |
| 36 | reserved (u32) | 0 |
| 40 | encoding digest (12 bytes) | the first 12 bytes of the matching ELF note digest |
| 52 | profile digest (12 bytes) | the first 12 bytes of the matching ELF note digest |
| 64 + 8 i | entry i (chip i): offset u32 (from the start of the file), size u32 | entry 0 starts at 64 + 8 K; entry i + 1 starts where entry i ends; the last ends at the end of the file |

Chip i of the run is entry i, and an object's `chip_id` operands follow the `--chips K` wiring, as for single objects. A loader rejects a manifest with OTS-E-ASM-003 when the magic, version, header size, format revision, reserved word or topology is wrong; when the profile is not ASCII followed only by NUL bytes, names an unknown profile or is not the one being loaded; when a digest differs from the loader's (the 12-byte prefixes of the matching ELF note digests); when K is not a chip count above (0 or above 512); when an entry has size 0, does not follow its predecessor, or runs past the file, or the file has bytes after the last entry; or when an embedded object is not a valid object (the rules above) of the manifest's profile (the message names the chip). **Time and clock:** the manifest carries no start offsets or drift. By default, the simulator starts chips at cycle 0 with aligned HACs. The harness can set an individual chip's start cycle with `H start`. External weights, when supplied, use one `.safetensors` package per chip in chip order.

`tools/tsp_as.py mpack --profile P [--topology T.json] -o m.otsm chip0.elf chip1.elf …` builds a manifest (each object is checked first; K must be a chip count above; with `--topology` K must be the `chips` of the topology file, whose other contents the harness checks at run time). `tools/tsp_as.py mobjinput m.otsm [--weights chip0.safetensors --weights chip1.safetensors …]` prints the harness input for `tsp_sim --chips K`: for every chip i, in order, a line `C <i>` (hex), the `S` lines of its externs filled from its weight, then its `Q` lines (as `objinput` prints them for one object). `--weights` may be omitted (only embedded extents are written); when given it must be repeated K times, else OTS-E-ASM-004. A tensor missing from a package is OTS-E-ASM-004 as for `objinput`.

**Physical weight package** (`.safetensors`). The standard Safetensors header begins with a little-endian u64 JSON byte count, followed by the padded UTF-8 JSON document and the flat tensor data. Each named tensor is `U8`, with shape `[words, vlanes]` and a contiguous `data_offsets` range relative to the data start. Each word lists lane 0 first. Starting with offset zero, the image ranges cover the payload exactly. The reader limits JSON to 16 MiB.

The string metadata identifies `opentsp.profile` and, for each tensor, `dtype.<name>` (decimal architecture element type code, or 255 for raw bytes) and `sha256.<name>` (full digest of that tensor's data bytes). The type code records meaning and performs no conversion. Names must match `[A-Za-z_][A-Za-z0-9_]{0,14}`. Invalid framing, metadata, shape, duplicate keys or name is OTS-E-ASM-003. A mismatched digest or absent or wrongly sized binding is OTS-E-ASM-004. Both `wpack` writers generate these word arrays. `objinput --weights` reads them as the independent Sail harness's ordered S and Q input.

**Logical model weights.** A serving bundle carries one ordinary Safetensors file of model-shaped tensors, shared by all related programs and host embedding or verification code. The model loader applies the named recipe and architecture placement before handing physical words to the ISS. For the released GPT-2 and TinyStories recipes and optional cache, see the [demo guide](../../docs/demos.md).

**Disassembly** prints a program file from text images (the units of each queue, padding dropped), using the canonical text of every instruction and the queue names above. It raises the text as when no instruction waits (`raise_images`: a parcel's cycle is the previous parcel's cycle + 1 + the NOPs before it, −1 before the first; a trailing parcel whose last instruction has the bundle bit set is listed, NOPs after the last parcel are dropped). It applies the model's decode and placement checks in the model's order and annotates every instruction the model would reject with the model's code (`; OTS-E-…`), counting it and exiting with status 1:
- two or more instructions in one parcel: OTS-E-QUE-001 on each when they do not fit the slot rule of §3; otherwise OTS-E-MEM-001 on both references of a same-thread R + W pair in one bank or with a `gather`/`scatter`;
- otherwise, per instruction: OTS-E-DEC-003 (no entry matches, or tag 6 or 7; printed `illegal32 0x…` / `illegal64 0x…`), then OTS-E-QUE-002 (placement).

`dis` reads the profile from the object (or `--profile` for text images). For instructions without an annotation, assembling the output lowers to the same text back (the same images, apart from the NOPs of the gaps that were split differently).

**Listing format** (tested exactly by the corpora of §6):

```
listing   = ".arch opentsp-tsa" LF ".profile " name LF { extern-line LF } ".section text" LF { bundle } ;
bundle    = "bundle @cycle=" dec " {" LF { instr-line LF } "}" LF ;   (* one per cycle that has instructions, ascending *)
instr-line = "  " queue ": " text [ "   ; " code [ ": " detail ] ] ;   (* two spaces; exactly three spaces before ";" *)

text      = canonical instruction text | "illegal32 0x" hex | "illegal64 0x" hex ;
```

Within a cycle, the instructions are listed by queue (in the order of the images: ascending id, `CU` last) and, within a queue, in program order. `queue` is the canonical name of §3 (never an alias). `extern-line` is `.extern name, slice=n, addr=0x…, size=0x…` with lower-case hex and no leading zeros. An instruction that does not decode prints as `illegal32 0x…` (bits 31..0) or `illegal64 0x…`. The annotation is the instruction's code: the parcel's QUE-001 or MEM-001 when its queue has several instructions in the cycle, else DEC-003, then QUE-002.

## 5. Errors the assembler reports

The assembler checks statically everything the Sail decode checks. Any text it accepts encodes to a word that the model decodes to the same instruction and prints identically.


**Patching an assembled object.** These checks run on source text. A loader checks the format of an object (OTS-E-ASM-003), not the parcels in it, so a tool that changes the units of an assembled object afterwards (for example by patching an operand address into the text) bypasses them: the object is valid, and the model finds the fault only at dispatch. The case that matters is the bank rule: patching the address of one reference of a legal R + W pair into the other's bank makes the parcel OTS-E-MEM-001 at run time, both references are dropped, and the Read's stream commit is missing without any message at load time. **Rule: a program patched after assembly must re-check every parcel it touched** with the checks of the disassembler (`dis`: decode, placement, the slot rule and OTS-E-MEM-001 of §3, applied to the parcels raised from the images), or with an equivalent per-parcel check, before the object is run or stored. The directed test `test/directed/directed.py` runs a legal parcel and the same text patched into one bank on the model. The toolchain command `ots-asm patch`  applies a patch table to an assembled object and re-runs these per-parcel checks on every parcel it touched; a producer that patches objects uses it rather than writing units directly. **Why a patch site can keep a parcel legal.** A site declares the values it accepts (`min`, `max`, and `stride` with `span`: a value v is accepted when (v - min) mod stride < span). The bank of a MEM address is the bank bit of the address, so a grid whose accepted values all have one bank bit (the stride is a multiple of the bank size and the span lies inside one bank) keeps the bank of the reference fixed for every accepted value: a parcel that was a legal R + W pair stays legal, and the site needs no per-value argument. A site whose values lie in both banks (a lookup table that is read from either bank) can turn a legal pair into OTS-E-MEM-001 for some values, so `ots-asm patch` re-runs the bank rule on every touched parcel for the value it was given and refuses the value; a producer that patches objects by another route re-checks the parcel in the same way (the rule above).

| Code | Meaning |
|---|---|
| OTS-E-ASM-001 | Syntax: unknown directive, misplaced or duplicate directive, a line that is not a directive, bundle line, `}` or `<queue>: <instruction>`, tab or non-ASCII byte, unterminated bundle, cycles not strictly increasing, ambiguous text (two templates accept it) |
| OTS-E-ASM-002 | Symbol: undefined, duplicate, offset outside the extern, extern on another slice than the issuing queue, extern outside the slice, overlapping externs |
| OTS-E-ASM-003 | Format: unknown architecture or profile, profile or object of another specification version, bad object magic, version, format revision, digest, size, profile field, image entry, image order, an image longer than `icu.iq_depth` or extern entry; a malformed text file or weight package (§4); a warning for a queue's text longer than `icu.iq_depth` |
| OTS-E-ASM-004 | Weights: an extern with no tensor of its name, a tensor of another word count, an image that does not match its digest, a tensor name that is not an identifier when packing (a package that carries such a name is OTS-E-ASM-003 on reading), or a word of the wrong size when packing (§4) |
| OTS-E-ASM-005 | Queue: an unknown queue name or an index out of range (the model has no queue-id diagnostic: its input cannot name a queue that does not exist) |
| OTS-E-DEC-003 | No instruction matches the text, or an illegal type or mode for the instruction |
| OTS-E-OPR-001 | An operand value out of range (§1); a warning for an accessed word beyond the profile's `words_per_slice` (for `ifetchm a`: word a+1) and for a misplaced AGT configuration instruction (§3) |
| OTS-E-MEM-001 | An R-type and a W-type reference of the same liquid thread on one MEM queue in one bundle addressing the same bank, or one of them a `gather`/`scatter` (§3) |
| OTS-E-QUE-001 | Instructions on one queue in one bundle that do not fit the slot rule of §3 (`microarchitecture.md` §3, "Instruction front end"): a second instruction on a non-MEM queue; on a MEM queue two in one slot (fetch; R or W of one thread; configuration) or a whole-queue instruction with another. A warning for a reference on a path whose AGT configuration is open |
| OTS-E-QUE-002 | An instruction on a queue whose unit lacks it, or on the queue of an absent MEM slice (on `CU`: any instruction other than nop, sync, notify, config, including every 64-bit instruction) |

**The command line.** A tool that is given something it cannot use says so with a code and exits 1; it never prints a stack trace. A `word` command whose `<width>` or `<hex>` is not an integer is OTS-E-ASM-001 (`word: OTS-E-ASM-001: expected <width> <hex>, two integers`); an input or output file that is missing or cannot be read or written is OTS-E-ASM-003 (`<command>: OTS-E-ASM-003: <path>: <reason>`), for every command that reads or writes a file; a missing argument or option value is the usage text and exit 2. `--config <profile.json>` names a profile file other than `config/<P>.json`; both assemblers (`tools/tsp_as.py`, `ots-asm`) validate it as `tsp_sim` and `tsp-iss` validate a configuration (the schema `config/schema.json`: types, bounds and relations; then the timing rules of `microarchitecture.md` §1) and refuse a profile the model refuses: `<tool>: config <file>: <key>: <reason>`, exit 2. `--no-validate` skips the check. `opentsp-sw-toolchain/tests/diff_tsp_as.py` (`bad_inputs`) runs both assemblers on these cases and compares status and code.

Run-time diagnostics (timing, barrier, bank conflicts, MXM readiness, segment bounds, alignment: §1) belong to the ISS and the model. The assembler may warn about them, using the same codes.

**Message text is not normative.** A rejection is identified by its code and its line number. The words after the code (the quoting of names, the wording of a range or relation) are not part of the specification: the two assemblers and the simulators word some messages differently (a profile name in double or single quotes, two phrasings of the rule `d_func - d_skew` at least the superlane count), and the differential harnesses treat the text as free (`porting-guide.md` §3).

**Which code a rejected instruction text gets**. The text is tried against every template (§1, Templates):
1. No template matches the text's shape: OTS-E-DEC-003.
2. Otherwise, for each matching entry in `encoding.json` order, the operands are checked **left to right in template order**, and the first failing operand gives that entry's error:
   - a stream above 31, a group base other than 0 or 16, a value wider than its field, a value above `max_expr`, an excluded (`not_values`) or out-of-range (`max`, signed `min`/`max`) value, or a value not in `values`: OTS-E-OPR-001;
   - a name (type, mode, direction) not in the entry's `codes` or `values`: OTS-E-DEC-003;
   - a symbol error: OTS-E-ASM-002.

   Then the joint rule: a `bool_values` field on a boolean type, OTS-E-DEC-003.
3. If exactly one entry has no error, the text is accepted; if two or more have none, OTS-E-ASM-001 (ambiguous; the catalogue has no such text); if none, the error of the **first** matching entry.

So `cmp.lt.b8 E:0, E:1, E:40` is OTS-E-OPR-001 (the third operand, `rhs`, fails before the joint rule), while `cmp.lt.b8 E:0, E:1, E:4` is OTS-E-DEC-003 (`lt` on a boolean type). In a program file an unknown queue name (OTS-E-ASM-005) is reported before the text is tried. The placement codes (QUE-001, then QUE-002, then MEM-001) are checked when the bundle closes, in the order of §3.

## 6. Validation of this document

All the following run in `tools/model_test.sh` (`validation.md` reference assembler checks):
- `test/isa_db/asm_roundtrip.py` (ref and mini) checks three directions:
  - **Accept:** for every catalogue entry, at its corners and on random legal fields, the model prints the word; `tsp_as` parses the text with exactly one entry, encodes the same word, and disassembles it to the same text.
  - **Reject:** for every field, out-of-range texts (wider than the field, with and without leading zeros; above 31; not a group base; excluded; not listed; outside the step range; at or above the lane count) are rejected and never wrapped.
  - **Decode:** random words, legal and illegal, disassemble exactly as the model prints them.
- `test/isa_db/check_asm_examples.py` checks the program files:
  - every §2 example assembles and prints back through the model;
  - the §3 example assembles, and its text runs on the emulator with the stated result;
  - text images and binary objects round-trip, and a disassembled program reassembles to the same text;
  - the lexical rules of §1 hold (spaces, tabs, comments, case, leading zeros);
  - every queue name in the table maps to its id;
  - every error class of §5 is reported;
  - unrelated ELF objects, wrong ABIs and wrong profiles are rejected, and every malformed profile field, extern entry, image entry, image order and parcel of §4 is OTS-E-ASM-003, never a crash;
  - placement: every catalogue entry on every data queue and on `CU` gets the model's code (QUE-002), and the disassembler annotates the same instructions;
  - the disassembler's codes equal the model's on instructions with DEC-003, QUE-001, MEM-001 and QUE-002, including every accepted and rejected slot combination of §3 (Ifetch/IfetchM + references, R and W of two threads, Gather/Scatter, a reference + a configuration instruction, two fetches, two configuration instructions, a whole-queue instruction with another); the assembler and the object loader give the same verdict on each group, and a clean disassembly reassembles;
  - AGT placement per path: the Read + Write pair configuration is silent; stray, late, fifth-level and same-path cases warn;
  - text images: a malformed `Q` line is OTS-E-ASM-003, and a file is text when its first non-blank byte is `Q`;
  - lowering: the unit stream of `lower` for NOP gaps (including gaps above 65535), bundles, wide instructions and the lone-NOP rule, and `raise_images` as its inverse;
- `test/isa_db/json_decoder.py` (ref and mini, run by `check_asm_examples.py`) builds a text decoder from `encoding.json` alone (the `text_format` and `decode_rule` keys) and compares its texts with `isa_query`, and its decode diagnostics with `tsp_sim`, on random text.
- `test/isa_db/asm_corpus.py --check` keeps the checked-in corpora `test/isa_db/corpus/<profile>/` (`accept.txt`, `reject.txt`, `text.txt`, `groups.txt`) equal to the reference assembler's answers; `--verify` checks them against the model, `--against <cmd>` runs another assembler over them (porting-guide.md §3).

Profile identifiers use `[a-z0-9_]{1,16}`. ELF objects and multi-chip manifests store the identifier in a 16-byte field. A 16-byte identifier occupies the field without a trailing NUL. Physical Safetensors stores it as the `opentsp.profile` metadata string. All three writers reject longer identifiers without truncation.

## Scheduling constraints

Neither assembler checks timing: they check text (§1 to §5) and leave every constraint that depends on when an instruction executes to the machine. This chapter is the one place that lists those constraints. Each row names the hazard, the formula a schedule must satisfy (in profile keys, so it holds on every profile), the diagnostic the model reports when the formula is violated, whether `tsp-iss verify` finds the violation without data, and a directed test that pins the row. The numbering is stable: a new hazard takes the next number.

`tsp-iss verify <obj.elf | manifest.otsm> [--weights W.safetensors ...] [--config C.json] [--profile P] [--host FILE] [--topology FILE] [--max-cycles N] [--allow-parked]` is the data-independent check. It runs the program on the ISS without data and prints one line per diagnostic with its class. **Its domain.** A manifest is judged on every chip, and the program runs at the shortest and at the longest `c2c.link_latency` of its profile (the receive rules need the longest, the overrun rule the shortest; the paragraph below says when it runs more); a lone object with text on a C2C queue is refused (exit 2), because the vectors it receives come from other chips. `--host FILE` gives the `H` and `C` lines (the doorbells and DMA a producer intends), so that the text behind a HostWait is judged. Text that ends the run parked for good behind a wait nothing released (a HostWait, a DeskewRecv, a Sync) is printed as `parked` and was not judged: it fails (exit 1) unless `--allow-parked`. **Its three exit codes.** 0, **well-formed**: no `schedule` code and no `data` code but the end-of-run IQ-004, nothing parked. 1, **ill-formed**: a `schedule` code that no `data` code precedes, or parked text. 3, **undecided** (data-dependent): a `data` code occurred and no `schedule` code precedes it, so a `schedule` code that follows can be its cascade and a `schedule` code that the data hid was never raised; or text fetched from SRAM did not decode; or the schedule of the manifest depends on the link latencies and no hazard was found at the latencies that were run (the paragraph below); or the run needs more than `--max-cycles` (the option of `tsp-iss run`, passed on). Give `--weights` (the program's data) and run again. 2 is an input error and 4 an internal error. [platform.md](platform.md#71-rtl-conformance) §7.1 defines the three verdicts. A code is `schedule` when the rule that raises it reads only the text, the configuration and the timing state (the class column of `microarchitecture.md` §8, "Diagnostic classes"): a run in which every instruction took effect raises every `schedule` code the object has, so an object that passes (exit 0) raised none of them for any data. That holds for the object that passes, not for every hazard of the table below: the column "Found by `tsp-iss verify`" says under which conditions a row is found (a link rule needs the manifest, a host-driven packet needs the host script). The two other classes need data or faults and `verify` cannot decide them: `data` codes read SRAM or stream values (a Gather's map, an sxselect selector, a Permute map, fetched text), and `fault` codes read the faults the harness injects. `tsp-iss run` reports the same diagnostics as `E` lines and exits 0; only `verify` turns a schedule hazard into a failing exit status. The verifier is not reimplemented in an assembler: a second implementation of the timing rules is the divergence the faithfulness audits look for.

**The domain of `verify` for a manifest** is every chip, at every per-link latency corner, under the host script given with `--host`. The program runs once with every link at `c2c.link_latency_min` and once with every link at `c2c.link_latency_max` (the receive rules need the longest, the overrun rule the shortest); these two runs decide the whole latency domain of the specification (each directed link with its own latency, also varying per transfer) as long as the schedule of the program does not depend on a latency, and the only instruction whose timing does is DeskewRecv, which is released at the first epoch boundary after its vector arrives. `verify` records the release cycle of every DeskewRecv; when the two runs release one at different cycles (the epoch boundary falls inside the latency range), the program is judged again at every combination of the two bounds over the directed links it sends on (at most 10 links, 2^10 runs; each line then names the links, `link-latency 0:0=209+2:1=228`, chip:link=latency). A hazard found is exit 1. A clean result is exit 3, never 0: the latencies between the bounds and the jitter of single transfers were not judged, and a hazard can need such a latency (a second vector sent just before the boundary). A lone object that sends or receives on a C2C link (Send, Receive, Transmit, DeskewRecv) cannot be judged alone: the vectors it receives come from other chips, so `verify` refuses it (exit 2) and asks for the manifest (`ots-asm mpack`); a lone object that only talks to the host (PciSend, PciReceive) is judged as it is. In the table below, a pin is the stable token of its label (for example, `H-13`): the free text after the token is not checked against the test.

The table is generated: `opentsp-arch-spec/tools/gen_hazard_table.py` builds it from the diagnostic table of `microarchitecture.md` §8 and the diagnostics the model raises (`model/core/diagnostics.sail` and its use sites), and `gen_hazard_table.py --check` fails when a code has no row, a row names a code that does not exist, a row's class disagrees with the §8 column, a pinning label is missing from its test file, or this block is stale (`--write` regenerates it). To add a hazard, add its row to the generator and run `--write`.

<!-- hazard-table:begin (generated by opentsp-arch-spec/tools/gen_hazard_table.py; do not edit by hand) -->

| No. | Hazard | Formula | Diagnostic | Found by `tsp-iss verify` | Pinning test |
|---|---|---|---|---|---|
| H-01 | A stream operand is read when no value of it is at the reader's gap | read cycle t_r = d_r + k_r + d_skew_r; the value written at gap g_w that commits at c_w = d_w + k_w + d_func_w sits at gap G in cycle c_w + (G - g_w), so t_r = c_w + (G_r - g_w) exactly (G_r > g_w); a read one cycle early or late finds no value | OTS-E-TIM-001 | yes | `directed/directed.py` "A reference 0 that fails still opens its loop" |
| H-02 | A unit reads or writes a stream on a row where it has no station, or upstream of the writer | the reader's gap must be larger than the writer's on the same row (values move toward larger register numbers on both rows: the E row flows east, the W row west); a unit with no station on the row cannot use it | OTS-E-DIR-001 | yes | `directed/directed.py` "SXM shift writes the turned-round row" |
| H-03 | Two writes commit to one (row, stream, gap) in one cycle | d_a + k_a + d_func_a = d_b + k_b + d_func_b and g_a = g_b on one row and stream; the lowest (queue, byte plane) wins; the two MEM threads of one queue and a SEL + TSP pair merge instead (`microarchitecture.md` section 2) | OTS-E-SRF-002 | yes | `directed/directed.py` "SRF-002 on both writers" |
| H-04 | A multi-stream operand is not aligned to its width | a w-byte element at stream s needs s mod w = 0 and s + w <= 32 (w = 2, 4, 16; `.sg4` groups: 4); a Rotate needs dst + n^2 <= 32; an f16 ABC source needs an even stream | OTS-E-OPR-002 | yes | `directed/directed.py` "max.i8.sg4 with a source group at E:1 is OPR-002" |
| H-05 | An operand value is out of range at run time | SRAM address (after the AGT stride) inside the SetSegment bound and `sram.words_per_slice` (IfetchM: a + 1 too); stream 0..31 after StreamStep; link < `c2c.links` and chip_id the link's peer; AGT configuration instruction only after a reference on its path and at most four levels; ACC type equal to the ABC product's | OTS-E-OPR-001 | yes | `directed/directed.py` "stray AGT config is OPR-001"; `directed/directed.py` "stride past the top of the slice is OPR-001" |
| H-06 | A Gather or Scatter tile address, or an sxselect selector, is out of range | every tile address inside the segment and `sram.words_per_slice`, every `sel` <= 2: values of the map and selector streams, so only data decides; the good tiles and lanes still take effect | OTS-E-OPR-001 | no (data-dependent) | `directed/directed.py` "sxselect with sel 3 is OPR-001" |
| H-07 | A Read and a Write of one liquid thread are dispatched in one bank, or with an indirect reference | bank(a) = (a >> `sram.bank_bit`) & 1; R and W of one thread in one parcel need bank(a_r) != bank(a_w); a Gather or Scatter clashes with any reference of its thread on the other path; IfetchM reads a and a + 1 in consecutive cycles | OTS-E-MEM-001 | yes | `directed/directed.py` "Read + Write in one bank are MEM-001" |
| H-08 | A parcel breaks the slot rule | non-MEM queue: one instruction per parcel; MEM queue: at most one fetch, per thread one R and one W reference, one AGT configuration instruction, and a whole-queue instruction (Repeat, Sync, Notify, Config, SetVL, SetSegment, ...) alone | OTS-E-QUE-001 | yes | `directed/directed.py` "R1.1 three instructions: QUE-001 x3" |
| H-09 | An instruction is placed on a queue whose unit lacks it, or on an absent MEM slice | unit(instruction) in units(queue) (`encoding.json` `units`); MEM queue s needs (s mod 44) < `geometry.mem_slices_per_hemisphere`; the control channel takes Sync, Notify and Config only | OTS-E-QUE-002 | yes | `directed/directed.py` "R1.5 wrong unit on a free queue: QUE-002" |
| H-10 | A word does not decode | tags 6 and 7, an unknown opcode, a reserved bit, an illegal type or mode; text the assembler writes always decodes, so only text fetched from SRAM (a data word) can raise it | OTS-E-DEC-003 | no (data-dependent) | `directed/directed.py` "R1.2 undecodable on a parked queue waits for the release" |
| H-11 | An MXM stage is used before its producer's result is ready | IW dispatch >= LW dispatch + `timing.mxm_lw.d_func`; ABC dispatch >= IW dispatch + `timing.mxm_iw.d_func`; ACC dispatch >= ABC dispatch + `timing.mxm_abc.d_func`; a following ABC >= an f16 ABC + `timing.mxm_abc.occupancy_f16`; f16 uses the even plane of a pair, and an integer ABC needs integer weights | OTS-E-RES-001 | yes | `directed/directed.py` "ACC with no committed ABC is RES-001"; `directed/timing_rows.py` "occupancy_f16" |
| H-12 | A Permute uses its map too early, or a Repeat has nothing to repeat | Permute execute cycle >= LoadMap execute cycle + `timing.sxm_map_ready` (>= `geometry.superlanes`); a Repeat needs a preceding instruction other than NOP, Repeat and AGT configuration on its queue | OTS-E-RES-001 | yes | `directed/directed.py` "Repeat with no previous instruction is OTS-E-RES-001" |
| H-13 | An MXM weight buffer is filled beyond `mxm.rows` stream values | at most `mxm.rows` stream values in a plane's buffer (20 LW of 16 streams = 320 on `ref`) | OTS-E-RES-002 | yes | `directed/directed.py` "LWs fill the buffer; the next is RES-002" |
| H-14 | Two queues change one unit-state part in one cycle | no two staged changes to one MXM plane part (buffer, installed weights, ABC results, accumulator 0 or 1) or to one SXM map in one cycle | OTS-E-RES-003 | yes | `directed/directed.py` "two installs on one plane in one cycle are RES-003" |
| H-15 | A Permute map is not a bijection | the 16-bit entries below `vlanes` of the map loaded by LoadMap are pairwise distinct: contents of the map, so only data decides | OTS-E-SXM-001 | no (data-dependent) | `directed/directed.py` "a ready map that is not bijective: SXM-001 and no effect" |
| H-16 | A link is offered a vector before its previous one has left | two Send or Transmit execute cycles on one link at least `c2c.cycles_per_vector` apart (the same cycle counts as 0) | OTS-E-LNK-001 | yes, on the manifest of the program (every chip, at both link latency bounds) | `c2c/c2c_directed.py` "LNK-001" |
| H-17 | More vectors arrive on a link than its receive FIFO holds | vectors sent but not yet taken by a Receive that executed before their arrival <= `c2c.rx_depth`; a vector sent at c arrives at c + L, `c2c.link_latency_min` <= L <= `c2c.link_latency_max` | OTS-E-LNK-002 | yes, on the manifest of the program (every chip, at both link latency bounds) | `c2c/c2c_directed.py` "LNK-002" |
| H-18 | A Send or Transmit is released into a control packet in flight | no Send or Transmit execute cycle inside the cycles a csr or ctrl packet occupies on the link (host-driven packets) | OTS-E-LNK-003 | yes, on the manifest of the program and with the host script (`--host`) that sends the packets | `host/host_directed.py` "LNK-003" |
| H-19 | A Receive runs before its vector has arrived | Receive execute cycle >= the Send's execute cycle + L; make this hold for every L by a Deskew on the sender and on the receiver (`c2c.hac_period` = 252 > `c2c.link_latency_max`) rather than by a fixed offset | OTS-E-TIM-001 | yes, on the manifest of the program (a lone object with text on a link is refused) | `c2c/c2c_directed.py` "Receive before arrival is TIM-001" |
| H-20 | A host send follows its predecessor too closely | two PciSend execute cycles on one interface at least `host.cycles_per_vector` apart | OTS-E-HIO-001 | yes | `host/host_directed.py` "two PciSend within" |
| H-21 | Fetched text overflows the instruction queue | units in the IQ (text already enqueued + text delivered by Ifetch and IfetchM) <= `icu.iq_depth`; the units that do not fit are lost | OTS-E-IQ-002 | no (data-dependent) | `directed/frontend.py` "IQ-002: a full IQ loses the payload" |
| H-22 | Too many fetch payloads are staged | fetches issued but not yet delivered per queue <= `icu.fetch_staging_depth`; delivery starts `icu.fetch_latency` cycles after the second read | OTS-E-IQ-003 | yes | `directed/frontend.py` "IQ-003" |
| H-23 | Text is left in an IQ that never completes a parcel | the last unit of a queue's text has its bundle bit set, or half of a two-unit instruction arrived: fetched text decides, so it is data-dependent; the image an assembler writes cannot strand text | OTS-E-IQ-004 | no (data-dependent) | `directed/frontend.py` "IQ-004" |
| H-24 | An uncorrectable error is injected into a stream operand, an SRAM word or a link flit | no formula: the injected fault state (`F` lines) decides, never a program value; the result carries poison and the consumer reports it again (`microarchitecture.md` section 15) | OTS-E-ECC-001, OTS-E-LNK-004 | no (fault-dependent) | `directed/ras.py` "SRAM double flip: one OTS-E-ECC-001 at the VXM"; `directed/link_fec.py` "uncorr on flit" |
| H-25 | An uncorrectable error is injected into instruction text or into a matrix sum | no formula: the injected fault state decides; ECC-002 consumes the instruction's units without effect, ECC-003 gives the ABC no effect (the ACC then reports RES-001) | OTS-E-ECC-002, OTS-E-ECC-003 | no (fault-dependent) | `directed/ras_csr.py` "text ECC: a double flip in text word 0"; `directed/mxm_check.py` "OTS-E-ECC-003 on the ABC, the ABC has no effect" |
| H-26 | A parcel is due while its queue is parked, busy or held | no report: the parcel dispatches at the first cycle >= its due cycle at which the queue is not parked (the Sync window ends at N + `barrier.release_latency` for the first Notify N at or after the Sync), not busy (Deskew, Repeat, AGT loop and its tail) and its path is free; a late parcel delays every later parcel of its queue and no other queue | none | not applicable | `directed/directed.py` "R1.6 wrong unit while an AGT loop runs waits for the loop and its tail" |
| H-27 | Fetched text arrives after its parcel was due (IQ underflow) | lint OTS-W-IQ-001 only (`tsp_sim --lint`): not a diagnostic and not normative; the prefetch must deliver the parcel's units (`icu.fetch_latency` + units / `icu.fetch_bytes_per_cycle` cycles after the fetch's second read) before the parcel is due | none | not applicable | `directed/frontend.py` "underflow" |

<!-- hazard-table:end -->

## Tensor layout in weight packages and SRAM

The object and package formats of §4 carry words, not tensors. A package tensor is a run of `vlanes`-byte words, lane 0 first, and `wpack` copies the bytes of its input file: the dtype tag (`t=f16:file`) is informational and converts nothing. What the bytes mean to a program is fixed by the stream and memory rules of `topology.md` §3 and §4. This chapter states the layout a producer must write so that a program reads the tensor it intends.

- **Element vectors and byte planes.** Lane l of a stream value is element l of a vector. A w-byte element (w = 1, 2, 4) is split into w **byte planes**: plane k holds byte k, least significant first (little-endian), of every lane. On the wire plane k is stream s + k of an aligned group (s a multiple of w; §1, `topology.md` §3); in SRAM it is one word (a `vlanes`-byte word of the plane's lanes). So a vector of f16 elements is two words (plane 0 = the low bytes of all lanes, plane 1 = the high bytes), a vector of i32 or f32 is four. A file of the element-interleaved bytes a framework writes (element 0 byte 0, element 0 byte 1,...) is **not** in this layout.
- **`wpack` and planes.** `tsp_as.py wpack` and `ots-asm wpack` copy bytes. A multi-byte tensor must therefore be given as plane words. `ots-asm wpack --planes` converts an element-interleaved little-endian file into plane words (plane k holds byte k of each element, as `opentsp_tools.programs.data` `to_planes` does for the reference programs) and `ots-asm wpack` warns when a multi-byte dtype is used without it or without `--raw-bytes`. Example: 320 f16 values in a file of 640 bytes become two words with `--planes` (word 0 the 320 low bytes, word 1 the 320 high bytes); without it the same file is two words of interleaved bytes, which an f16 `abc` would read as 320 different elements. A tensor of more than one vector is written **vector-major**, the placement (a) below: plane k of vector j is word j × w + k of the tensor. Example: 640 f16 values (two vectors v0 and v1) become four words, [v0 plane 0, v0 plane 1, v1 plane 0, v1 plane 1]; a producer that places the planes of one vector in w slices (placement (b)) cuts the words with stride w.
- **Placement of the planes.** The package format has no notion of planes: it places words in an extern of one MEM slice. Two placements are in use. (a) The w plane words of one vector are consecutive words of one slice, read one after another. (b) The planes of one vector are in the w slices of one MEM quad at the same address, one extern per slice; the four slices of a quad share a station gap (`topology.md` §4), so their reads reach any consumer in the same cycle as an aligned stream group (s..s+3, or a pair). The reference compiler uses (b) for operands that a VXM or MXM consumes as a group.
- **`mxm.rows` advisory.** When `mxm.rows` is larger than `vlanes` (16 × `geometry.superlanes`), the rows beyond `vlanes` only add to the buffer capacity (RES-002): a product has `vlanes` lanes. The advisory that says so is printed by `opentsp-arch-spec/tools/schema_invariants.py` (an `advisory:` line of `--check`); `derive_profile.py`, `tsp_sim`, `tsp-iss` and the assemblers print nothing and accept the profile.
- **2-D weights.** An MXM weight row is one stream value, so a weight matrix W of K inputs by N outputs is packed as its transpose: **row r of the installed matrix is output r, and its columns are the K inputs** (the ABC computes `product[r] = Σ_k W[r][k] · x[k]`, `microarchitecture.md` §6). An i8 or u8 row is one word (one plane); an f16 row is a stream pair (two words: low-byte plane, high-byte plane).
- **LW row order.** An `lw` appends 16 stream values (8 f16 rows) to the buffer: row `n + i` is the value on stream `src + i`, in stream order, where n counts the rows already in the buffer; `mxm.rows` / 16 LWs fill a plane (20 on `ref`). The installed row index is the buffer index, so the row that arrives on stream `src + i` of the j-th LW (j from 0) is row 16 j + i. For f16 the pair (2i, 2i + 1) of the LW is row 8 j + i.
- **Striping over slices.** One LW reads 16 streams in one cycle, so the 16 rows of one LW come from 16 different MEM slices (a slice puts one word per read on one stream): the reference compiler stripes the rows of a weight tile over 16 slices, row r in slice r mod 16, and reads each slice with one AGT loop of 20 references (one per LW); slices in different quads start their loops at cycles that differ by their station-gap difference so that the 16 values meet at the matrix input station together. One extern holds the words of one slice, so a striped tile is 16 externs (names of at most 15 characters).
- **What is not specified.** The word order of a tensor larger than one matrix tile (which tile is which extern), quantisation and the transposition itself belong to the producer's layout recipe; the specification fixes only the rules above.

## Program-data packages

Long queues use `ifetch` and `ifetchm` to bring the rest of their instruction units from SRAM. Serving units put their fetched text in `.tsp.text` and program literals in `.tsp.const`. Assemblers and ISS also accept separately authored Safetensors of physical words for diagnostic and bring-up programs. The extern placement and page order determine what reaches the IQ.

- **Text blocks.** A fetch brings a payload of 2 × `vlanes` bytes, which is two SRAM words (`ifetchm a` reads words a and a + 1 of its own slice; `ifetch s` reads the stream s at two consecutive cycles). A block is those two words: lane 0 of the first word first, little-endian 32-bit units, four lanes per unit, the unused tail filled with `0xFFFFFFFF` (padding, never enqueued). The last block of a queue is padded; a block is a raw tensor (dtype 255) of two words, with one extern per block (name of at most 15 characters), in the slice of the queue for `ifetchm` (word a + 1 must be inside the slice, else OTS-E-OPR-001) and in slices upstream of the queue's read station for `ifetch`, put on the stream pair by two Reads. The concatenation of the initial image and the blocks in fetch order, padding dropped, equals the lowering `lower` of the queue's bundles: the split changes where text comes from, never what it is.
- **What the compiler must keep true.** Every unit of a parcel must be in the IQ by the cycle the parcel is due (else the parcel dispatches late: the queue waits for its text, everything behind it on that queue shifts, and the lint OTS-W-IQ-001 reports it), and an arriving unit must find room (a unit that finds `icu.iq_depth` / 4 units in the IQ is lost: OTS-E-IQ-002). The plan therefore needs the actual dispatch cycles of the queue, which equal the bundle cycles only when nothing waits (the theorem of §4). Four events make them differ, so a compiler must plan the fetches against the actual cycles, not against the bundle cycles:
  - **Sync and Notify.** A queue parked by a Sync dispatches its next parcel at the release cycle (Notify N + `barrier.release_latency`), later than the bundle cycle if that was earlier; the parcels behind it shift with it. The reference pager (`opentsp-sw-toolchain/python/opentsp_tools/pager.py`) pages only programs with no Sync or Notify for this reason.
  - **Deskew and RuntimeDeskew.** The queue is held to the next HAC epoch boundary (`c2c.hac_period`), so later parcels shift by the hold. The reference pager does not page a program with either.
  - **Repeat.** A Repeat holds its queue through its last iteration: a parcel due inside the busy window waits, and so does a fetch instruction placed there. The reference pager does not page a queue that has a Repeat.
  - **Loops and busy paths.** An AGT loop holds its path through its trailing waits, and a whole-queue instruction waits for both paths (`microarchitecture.md` §3).
- **Slices that host text.** The words used for text are not used by anything else; a slice whose words are read onto a stream as text must run no `setvl` or `config` while it does (both gate or split the stream a Read puts out). The control channel has no image and is not paged.

## Scheduling a kernel against a profile

Starting from a producer and consumer, look up the operation's encoding, legal queue, stream group width and alignment. Resolve any per-entry timing override before assigning cycles. A producer dispatched at d with reference offset k commits its stream at d+k+d_func. Add the distance to the consumer station, then subtract the consumer's d_skew and reference offset to determine its dispatch. Check bus direction, banks, queue slots and intervening writes on that same path. The worked addition in §3 uses a VXM dispatch of 40, an operand read at 42 and a MEM write at 48. The object carries the profile identity. Load the matching profile and weight placements before comparing results.

Lower the per-queue schedule using NOP prefixes relative to the previous **actual** parcel dispatch. The next complete head becomes eligible at previous_dispatch+1+sum(prefix), subject to fetch availability, barrier, doorbell, Repeat, AGT and occupied resource holds. A hold shifts subsequent parcels on this queue. Another queue may still issue at its original time and miss an operand. For example, releasing HostWait with a doorbell at 7 allows its next eligible site at 8. An instruction that decodes can still fail at placement, dispatch or operand read. The partial effects and their diagnostic order are specified in the [execution and resource contract](microarchitecture.md). The idioms a kernel writer needs and this specification does not spell out are in the [kernel cookbook](../../docs/kernel-cookbook.md): division and reciprocal from `RSqrt` (three idioms with their measured domains), a matrix-vector tile, a softmax and a LayerNorm, long and patched programs.

| FP8 kernel operation | Ordinary encoding | Four-stream group |
|---|---|---|
| Add/Sub/Mul, Max/Min, Cmp, Mask/Select, Neg/Abs/ReLU | One FP8 element per byte lane | `.sg4` uses a four-aligned base and four operations per lane |
| Reduce add/max/min | `reduce.add.e4m3` and corresponding modes/formats. Each group of 16 lanes folds in increasing order. | No group encoding |
| Convert | `convert.e4m3.f16` and other legal source/destination pairs. Align each operand for its own width. | No group encoding |
| Bitwise, saturating/modular arithmetic, transcendental or fixed legacy conversion | The corresponding FP8 type encoding is illegal. | No FP8 encoding |

For a rounding-sensitive example, put binary16 values 2048, 1 and −2048 (0x6800, 0x3c00, 0xe800) in the first three lanes and +0 in the remainder. The low and high bytes travel in adjacent streams. Reduce add rounds 2049 to the even 2048, then cancels to +0. Contracting the same inputs against exact unit weights in the FP16 MXM dot preserves the unit contribution and rounds once to binary32 1 (0x3f800000). Reordering the reduction to cancel the two large inputs first also gives 1. A compiler needs explicit permission to change those results. For FP8, e4m3 0x7e is finite 448 and 0x7f is NaN, whereas e5m2 0x7c is infinity. Use the exact destination format for saturation and canonical NaN rules. [Generated instruction definitions](isa.md) · [Numeric semantics](microarchitecture.md)
