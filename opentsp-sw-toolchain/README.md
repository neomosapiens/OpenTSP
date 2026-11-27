# OpenTSP software toolchain

The [platform setup guide](../docs/installation.md) covers native macOS, native Linux and the development container.

The assembler implements the [OpenTSP 0.1 assembly and binary formats](../opentsp-arch-spec/docs/assembly.md). Its output is checked against the [reference assembler](../opentsp-arch-spec/tools/tsp_as.py).

- [Assemble and run a program](../opentsp-arch-spec/docs/assembly.md) using the worked example.
- [Build kernels](../docs/kernel-cookbook.md) using cycle-annotated examples.
- [Inspect a schedule](tools/README.md#schedule-explorer) or open the [bundled comparison](tools/schedule-example.html) in your browser.

## Crates

- `ots-profile`: the machine profile (`config/*.json`): schema, validation, derived geometry and timing.
- `ots-isa`: the instruction set from the catalogue: instruction words, queue map, element types, diagnostic codes.
- `ots-format`: the file formats: instruction text, `.elf` objects with extern placements and optional embedded program data, physical `.safetensors` packages for external SRAM data, and `.otsm` multi-chip manifests. Serving model weights use separate logical Safetensors.
- `ots-asm`: the assembler library and the `ots-asm` command.

## Commands

```bash
cargo build --release -p ots-asm
export PATH="$PWD/target/release:$PATH"
```

Command reference (replace placeholders with your inputs):

```text
ots-asm insn  [--profile P] [--config F] <text>               # one instruction -> "<width> <hex>"
ots-asm word  [--profile P] [--config F] <width> <hex>        # one word -> instruction text
ots-asm asm   [--config F] [--input] [-o out.elf] <prog.ots>  # assemble; --input prints the Q lines of the lowering
ots-asm dis   [--profile P] [--config F] <obj.elf | text.txt>
ots-asm objinput [--profile P] [--config F] [--weights W.safetensors] <obj.elf | text.txt>
ots-asm wpack --profile P [--config F] [--planes | --raw-bytes] -o W.safetensors name=dtype:file...
ots-asm mpack [--profile P] [--config F] [--topology T.json] -o M.otsm <chip0.elf>...   # --topology: the chips are those of the file (at most 512); otherwise K = 1..264 objects
ots-asm mobjinput [--config F] <M.otsm> [--weights chip0.safetensors...]
ots-asm patch [--profile P] [--config F] --table T.json --values V.json -o out.elf <obj.elf>
```

Every command that is given `--config F` validates the file as `tsp-iss` and `tsp_sim` do (the schema, its relations and the timing rules), and prints `ots-asm: config F: <message>` with status 2 for a configuration the model refuses; `--no-validate` skips the check. The Python reference assembler also validates `--config` inputs and supports `--no-validate`.

### Weight packages: `--planes` and `--raw-bytes`

`wpack` copies each file as the words of its tensor (the file is a whole number of `vlanes`-byte words). A multi-byte element type needs its bytes as plane words (word k of a tensor of `w`-byte elements holds byte k of `vlanes` consecutive elements, plane 0 first). The tag after `name=` does not convert anything. `--planes` converts a file of little-endian elements, one after the other, into plane words, vector-major: for each vector of `vlanes` elements, plane k (byte k of its `vlanes` elements) is word j × w + k of the tensor for vector j, so the `w` plane words of one vector are consecutive and the next vector follows; the file must hold a whole number of `vlanes` elements, and the dtype must be an element type, not `raw`. Without `--planes`, a multi-byte dtype (`f16`, `i16`, `u16`, `b16`, `i32`, `u32`, `f32`, `b32`) prints a warning on stderr and copies the file; `--raw-bytes` states that the file already holds plane words and silences the warning. `--planes` and `--raw-bytes` exclude one another.

### Patching an object: `ots-asm patch`

`patch` rewrites operand fields of an object's text, as a host does between runs of one program (the memory addresses of a read, the map of a permute), and checks the result again; the object format does not change. The table names each site, the values file gives the new values:

```json
{"format": 1, "sites": [
  {"name": "rd", "queue": 4, "unit": 1, "op": "read", "field": "addr", "kind": "sram_addr", "min": 0, "max": 4095}
]}
```

`queue` is the queue id and `unit` the index of the instruction's first text unit in that queue's text (every unit counts, NOP units included); `op`, `field` and `kind` are the instruction's mnemonic and the operand's name and kind in the catalogue and must match what the object holds there; `min` and `max` bound the values; the optional `stride` and `span` (default 1 and 1) accept `v` when `(v - min) % stride < span`. The values file is `{"rd": 32}`. A site without a value keeps its operand. A value outside the range or the grid, an unknown name and a site that does not match its instruction are refused with a coded diagnostic and nothing is written. The patched object is then checked as the assembler checks a bundle: every parcel that holds a patched instruction must decode (`OTS-E-DEC-003`), fit the queue's slots (`OTS-E-QUE-001`), sit on a queue of its unit (`OTS-E-QUE-002`) and obey the bank rule (`OTS-E-MEM-001`), so a patch that moves a read and a write of one thread into one bank produces no object. One line per patched site is printed on stdout.

### Checking an object without data

`tsp-iss verify <obj.elf | manifest.otsm> [--weights W.safetensors ...] [--config C.json] [--profile P] [--host FILE] [--topology FILE] [--max-cycles N] [--allow-parked]` runs the program on the ISS without data and prints one line per diagnostic with its class (`schedule`, `data-dependent`, `data-fetched`, `fault`). A manifest is judged on every chip and runs with every link at the shortest and at the longest `c2c.link_latency` of its profile; when a DeskewRecv is released at different cycles in the two runs, the program is run again at every combination of the two bounds over the links it sends on (up to 10 links; a line then names `link-latency chip:link=latency`), and a clean result is then exit 3, not 0, because the latencies between the bounds were not judged ([assembly](../opentsp-arch-spec/docs/assembly.md), "Scheduling constraints", states the domain). A lone object that sends or receives on a C2C link is refused (exit 2) and the manifest asked for; one that only talks to the host (PciSend, PciReceive) is judged as it is. A queue that ends the run parked with text it never dispatched (a HostWait, a DeskewRecv, a Sync nothing releases) is printed as `parked queue Q cycle C CAUSE` and fails unless `--allow-parked` is given; `--host FILE` takes the `H` lines of the harness (a doorbell, a DMA vector) so the text behind the wait is judged under the producer's intended timing. The exit status is 0 when the program is clean (no schedule code and no data code but the end-of-run IQ-004), 1 when a schedule code occurs that no data code precedes (the codes of the `schedule` class of `microarchitecture.md` section 8; a DEC-003 of text fetched from SRAM, and an OPR-001 of a Gather or Scatter tile address or of an `sxselect` select, are data) or text is parked, 3 when it is undecided (a data code occurred and no schedule code precedes it, because a data code can hide the schedule code that the instruction it removed would have raised, or text fetched from SRAM did not decode so the text behind it was never judged: give the data with `--weights`; or the schedule depends on the link latencies and no hazard was found; or the run needs more than `--max-cycles`) and 2 for a bad input. The assemblers do not check timing.

When loading a profile, the assembler checks the effective Shift, Permute, ABC and ABC8 latencies, map dissemination time and instruction fetch delivery against tile capture. A causal violation yields `OTS-E-ASM-003` before object generation. The ISS also validates the full simulator configuration when constructing a machine through its CLI or Rust platform API.

`objinput` and `mobjinput` print the input of `tsp_sim` and `tsp-iss`; the ISS also reads objects, weight packages and manifests directly (`opentsp-iss/README.md`).

## Source essentials

```asm
.arch opentsp-tsa
.profile ref
.extern W, slice=4, addr=0x0, size=0x1
.section text
bundle @cycle=0 {
  MEM[4]: read W, E:0, 0x0
}
```

The full syntax, the object and package formats and the diagnostics are in [assembly](../opentsp-arch-spec/docs/assembly.md). A declared extern is bound to a `.safetensors` package when the object is loaded; no weight bytes are ever part of an object.

## Schedule inspection

[Schedule Explorer and bubble analysis](tools/README.md#schedule-explorer) reads assembled objects and sealed routes, ranks exact queue gaps in JSON, and generates interactive offline timelines. [Open the bundled four-chip comparison](tools/schedule-example.html).

## Tests

Build the assembler and reference simulator before running the worked example. See [installation](../docs/installation.md#build-and-exercise-each-component).

```bash
cargo test --workspace
just coverage-asm                  # per-crate line and branch coverage floor (90 %)
just asm-diff                      # every ots-asm command against tsp_as.py (tests/diff_tsp_as.py), seeds in parallel
bash opentsp-sw-toolchain/tests/e2e_example.sh "$PWD/target/release/ots-asm" "$PWD/target/spec/tsp_sim"
```

An assembly error emits a coded diagnostic and exits 1. Invalid arguments and invalid configuration files exit 2. Ctrl-C interrupts the foreground command. See the [assembly example](../opentsp-arch-spec/docs/assembly.md) for a complete program, input tensors and execution.
