# OpenTSP instruction-set simulator

The Rust ISS implements specification 0.1 and its reference-harness input/output protocol. It models the cycle driver, instruction queues, SRAM, stream registers, CU/MEM/VXM/MXM/SXM/C2C units, links and host events. It is checked against the Sail model through directed cases, golden vectors and differential tests. Those checks do not prove equivalence for every history; see [assurance](../docs/assurance.md).

## Build and run

Install the [shared prerequisites](../docs/installation.md). From the repository root:

~~~bash
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
cargo build --release -p tsp-iss -p ots-asm
target/release/ots-asm asm -o /tmp/program.elf program.ots
target/release/tsp-iss run --profile ref --trace /tmp/program.elf
~~~

The [assembly example](../opentsp-arch-spec/docs/assembly.md) includes source and data. ELF objects contain initial instruction images and extern placements. They can also carry fetched text, literals and zero-initialized extents. Supply external physical words in a `.safetensors` package with `--weights`. An `.otsm` manifest groups the ELF files in chip order. Select wiring separately with `--chips` or `--topology`.

~~~bash
target/release/tsp-iss run --profile ref --weights weights.safetensors --stats /tmp/program.elf
target/release/tsp-iss opentsp-arch-spec/config/mini.json --trace < input.txt
target/release/tsp-iss vectors opentsp-arch-spec/conformance/model/mini.json
~~~

The second command uses the reference-compatible harness. The native `run` command accepts `--config` for a custom profile. Both forms support explicit multi-chip wiring:

~~~text
tsp-iss CONFIG.json [--chips K | --topology FILE] ...
tsp-iss run [--chips K | --topology FILE] ... PROGRAM
~~~

`--chips` selects the built-in wiring for 1–264 chips. Explicit topologies support up to 512 chips (`MAX_CHIPS`) and 128 chip-to-chip links per chip (`MAX_LINKS`). Each selected profile sets the actual link count and `mxm.planes_per_hemisphere` sets its matrix plane count. `--jobs` controls independent work between link deliveries. `--max-cycles` bounds a run. Invalid input returns a nonzero exit with details on stderr. Use `--help` for the full option list and Ctrl-C to interrupt a foreground process.

The [platform](../opentsp-arch-spec/docs/platform.md), [topology](../opentsp-arch-spec/docs/topology.md) and [microarchitecture](../opentsp-arch-spec/docs/microarchitecture.md) chapters define the protocol, state and diagnostics.

## Verification and diagnosis

~~~bash
just iss-vectors
just iss-diff
just programs
cargo test -p tsp-iss -p tsp-numerics
~~~

Trace output records requested architectural writes and diagnostics. Optional observers and activity reports are separate from architectural stdout. --stats prints a summary; --stats=FILE writes JSON. --obs-out DIR selects observer artifacts, with --obs-summary, --obs-diagnostics, --obs-utilization and --obs-events enabling individual families. --obs-summary alone does not turn on unrequested artifact families.

Utilization JSON uses numeric schema 1 for ordinary dense timelines. If any execution bin index is at least 2,000,000, every chip in that file uses schema opentsp.utilization.sparse.v1, with timeline_encoding absolute-nonzero-v1, exact timeline_bins, sorted absolute timeline_nonzero indices and timeline_omitted_bins_are_zero true. The legacy timeline field is absent. Consumers must reject unknown schemas and preserve exact integer precision for cycle/count values, including a possible run length of 2**64 cycles.

For observer files, wait for exit 0 and read the files named by the run’s `manifest.json`. Only one process can publish to a directory at a time. A write failure returns exit 4 and leaves no completed manifest for that publication. Reuse may leave older files outside the current manifest.

Optional observer wait records sample visited cycles and may differ when idle cycles are skipped. Required architectural effects must conform. Other data-dependent holds, injected faults and partial effects follow the specification contracts.

## Activity and schedule inspection

Use the [energy model](perf/README.md) to interpret energy outputs and the [Schedule Explorer](../opentsp-sw-toolchain/tools/README.md#schedule-explorer) to inspect static instruction issue gaps. Static gaps and sampled runtime waits describe different observations. [Testing](../docs/testing.md) maps verification commands to their evidence.

## Implementation

`crates/tsp-iss` contains the simulator library and CLI. `crates/tsp-numerics` contains the numerical kernels. The shared software-toolchain crates supply profiles, ISA decoding and file formats.

Rust callers use `Machine::load_image`, `Machine::preload`, and the other typed run and input APIs to operate a machine. The geometry, ISA, timing table and state readers are read-only. Advanced debuggers can use `Machine::edit_state` to alter architectural state before continuing. This discards recorded schedules and rebuilds the scheduler’s derived state. Callers must keep their edits within the selected profile and preserve architectural invariants.

Opt-in matrix threading and schedule replay preserve the modeled result and fall back when the execution cannot use them. Activity counters and the [energy table](perf/README.md) provide estimates, not physical power or timing measurements. CPU simulator performance depends on host load and workload; no physical accelerator throughput follows from it.
