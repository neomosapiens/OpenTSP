# OpenTSP architecture specification

Version **0.1** defines an executable TSP-inspired architecture. The Sail model specifies instruction and architectural-state semantics; the normative C harness, configuration schema and host/link input/output protocols supply the execution environment. The instruction catalogue, generated encoding tables and conformance vectors accompany that combined contract.

The published TSP sources are cited where they establish a feature. Choices made by this independent model are identified explicitly. Whole-machine formal correctness is [not proved](../docs/assurance.md).

## Architecture sources

OpenTSP is inspired by Groq's original Tensor Streaming Processor papers.
The architecture chapters cite them as ISCA20 for the processor and ISCA22 for the interconnect and scale-out design.

- Dennis Abts et al., [*Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads*](https://doi.org/10.1109/ISCA45697.2020.00023), ISCA 2020.
- Dennis Abts, Garrin Kimmell, Andrew Ling, et al., [*A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning*](https://doi.org/10.1145/3470496.3527405), ISCA 2022.

## Start here

| Role | Reading path |
|---|---|
| Chip architect | Start with [topology](docs/topology.md) and [microarchitecture](docs/microarchitecture.md), then use [design space](docs/design-space.md) and the [performance model](docs/performance.md). |
| Circuit designer | Start with the [platform and RTL contract](docs/platform.md), then use [bring-up](docs/bring-up.md), [validation](docs/validation.md) and the [management gate](docs/production-ras.md). |
| Toolchain developer | Start with [assembly and formats](docs/assembly.md), [ISA](docs/isa.md) and [encoding](docs/encoding.md). Use the [porting guide](docs/porting-guide.md) for trace and diagnostic conformance. |

## Architecture chapters

| Chapter | Contents |
|---|---|
| [Topology](docs/topology.md) | Geometry, profiles, queues, stations, SRAM banks and chip wiring |
| [Microarchitecture](docs/microarchitecture.md) | Instruction semantics, timing, streams, diagnostics, numerics and fault behavior |
| [ISA](docs/isa.md) / [Encoding](docs/encoding.md) | Instruction reference and bit layouts |
| [Assembly](docs/assembly.md) | Source syntax, worked example, object/weight/manifest formats and scheduling constraints |
| [Platform](docs/platform.md) | Reset, host/link environment, observable effects and RTL conformance |
| [Production RAS](docs/production-ras.md) | Publication gate, quarantine, recovery and operator procedures |
| [Bring-up](docs/bring-up.md) | Conformance vectors and a model-to-RTL comparison workflow |
| [Validation](docs/validation.md) | Checks, proof statements, domains and limitations |
| [Porting guide](docs/porting-guide.md) | Implementing the specification in another simulator or hardware model |
| [Performance](docs/performance.md) | Derived rates and explicit modeling assumptions |
| [Design space](docs/design-space.md) | Profile parameters, fixed limits and architectural alternatives |
| [Extending](docs/extending.md) | Adding a coherent instruction/model/toolchain extension |

## Build and use

Install the [shared toolchain](../docs/installation.md), then run these commands from the repository root:

~~~bash
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
just python-setup
opentsp-arch-spec/tools/build_sim.sh
target/spec/tsp_sim opentsp-arch-spec/config/mini.json < input.txt
just model-test
just model-prove
~~~

The harness consumes the input protocol described in the microarchitecture chapter and emits diagnostics, requested trace records and state reads. A malformed input is rejected; --trace and --observe enable additional records. Interrupt a foreground run with Ctrl-C.

| Directory | Contents |
|---|---|
| model/ | Sail instruction/state semantics, cycle driver, unit tests and SMT obligations |
| c_emulator/ | C storage, input/output and host/link harness |
| config/ | ref and mini profiles, schema and directed test profiles |
| isa-db/ | Catalogue, provenance and generated machine-readable encoding |
| conformance/ | Reference vectors and RTL-obligation sidecars |
| platform/ras/ | Companion result-publication Sail contract and management RTL |
| test/, tools/ | Differential/directed checks and generation/build tools |

Run just model-exhaustive for every 32-bit decoder input and just ras-prove for the management gate. Those results apply to their stated domains and environmental premises. A hardware implementation must independently establish detector, isolation, DMA and timing assumptions. The [ISS](../opentsp-iss/README.md) and [assembler](../opentsp-sw-toolchain/README.md) provide downstream implementations.
