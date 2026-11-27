# Python toolchain library

Install the workspace packages with `just python-setup`. Import the library as `opentsp_tools`.

- `profile`, `stations`, `queues` and `sched` describe the instruction set, profile and scheduling constraints.
- `emit`, `pager` and `patch` emit programs, page instruction text into SRAM and validate operand patches.
- `refasm` loads the architecture’s `tools/tsp_as.py` directly from the source checkout. The pager and patch tool call `assemble`, `word_units`, and `queue_id` in process. `refasm` verifies the required symbols without changing `sys.path`.
- `programs` contains model-independent helpers and kernels for producing OpenTSP programs.

See the [toolchain README](../README.md) for build and execution commands and the [kernel cookbook](../../docs/kernel-cookbook.md) for worked examples.

## Schedule Explorer

[Open the bundled four-chip schedule comparison](schedule-example.html) in a browser. It shows the **Two-chunk attention** and **Single-page pairs** schedules on the same cycle axis. Select a stage, unit, or queue, then choose a ranked gap to inspect its inclusive cycle range and the neighboring instructions. Full wave repeats each sealed template at its measured panel cadence. The comparison shows device schedule cycles.

After `just python-setup`, inspect an assembled object from the repository root:

```bash
python -m opentsp_tools.schedule --object Program program.elf \
  --html schedule.html --json bubbles.json
```

The explorer reads fetched instruction pages from the ELF. For a program with separately supplied pages, use `--weights Program pages.safetensors` with a physical Safetensors package. The reader understands the pager's `tx<queue>b<block>` externs, including paired `a` and `b` halves. Programs that fetch additional text without a recognized page payload report incomplete coverage. The parser reads the object profile and uses the installed specification.

For sealed stage packages, compare up to two inputs:

```bash
python -m opentsp_tools.schedule \
  --route Before /path/to/before --route After /path/to/after \
  --html comparison.html --json bubbles.json
```

The route loader checks each stage's manifest entry and payload digests. The stage selector follows route order. The gap detail identifies the physical chip. Use `--window FIRST LAST` for an inclusive excerpt and `--top N` to change the number of ranked gaps in the JSON report.

### Read the gap report

A static gap starts after the previous explicit issue and ends before the next issue on the same queue. Instructions in one parcel count as one issue cycle. Queue gap percentages use the first-to-last issue span of that queue. Unit percentages sum those spans across their queues. The stage ranking also reports each gap's share of the stage issue envelope. Gaps can overlap across queues, so their percentages can sum above 100%. In-flight work, Repeat, address generation, and activity on other queues can occupy those cycles.

The generated HTML works offline. The optional `--timing LABEL timing.json` accepts an ISS measurement with `source_sha256`, `period`, `cycles`, `rows`, and `boot`. Cycles and period are in device cycles, rows are the measured total, and boot maps template cycle zero into the ISS clock. Full wave is available for complete measured panels that fit the supplied cadence. Use the source digest printed in a JSON report when preparing timing evidence. For a route it is the SHA-256 of its manifest. For an object with weights it binds both inputs.

For optional sampled waits, `--trace LABEL trace.json` accepts `source_sha256`, `chip` (zero-based stage index), `extent` (half-open window), `queues` (the sampled queue IDs), and `waits` (rows of `[cycle, queue, reason, first_due_cycle]`). Normalize trace clocks to template cycles first. Waits only cover the supplied queues and interval. They are displayed separately from static gaps.
