# Contributing to OpenTSP

Read the [project README](README.md), the relevant component README and the [architecture specification](opentsp-arch-spec/README.md) before changing behavior. The [testing guide](docs/testing.md) maps behavior to qualification entry points.

Source the shared toolchain in every shell:

~~~bash
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
~~~

Preserve version 0.1 semantics, file formats, diagnostics and trace/timing behavior unless the change deliberately updates the specification and every affected implementation. Keep weights separate from objects. Coordinate changes to Sail, catalogue/provenance, generated ISA/encoding, reference tools, ISS/assembler, oracle and golden vectors.

For a bug, reproduce the failure, make the smallest causal correction, verify it passes and verify a relevant adverse control fails. Run the touched tests, just lint and the relevant area recipe. Integration uses just test, just programs and just e2e; semantic model changes also require just model-test and just model-prove. Formal results must state their actual hypotheses and tool/native trust boundaries. Never equate proof counts or passing tests with whole-machine assurance.

Keep public documentation focused on current contracts and reproducible usage. Preserve source attribution and licenses. Store temporary build products and local qualification evidence outside tracked source.
