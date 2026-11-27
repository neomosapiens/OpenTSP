# OpenTSP vLLM plugin

The plugin runs TinyStories on one simulated OpenTSP chip and GPT-2 small through a four-chip pipeline. It targets **vLLM 0.33.0**. The host performs tokenization, token/position embedding, scheduling and sampling; simulated chips execute the transformer kernels and retain device KV. This is a functional simulator integration, not a hardware-throughput claim.

## Install and check

From the repository root, follow [installation](../docs/installation.md), then:

~~~bash
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
just vllm-env
just vllm-binding
just vllm-surface
just vllm-unit
~~~

The external environment defaults to ~/.local/opt/opentsp-vllm-venv; override OPENTSP_VLLM_VENV. The editable install points at this source directory. Installing it does not select the OpenTSP platform for unrelated vLLM runs: activation requires OPENTSP_VLLM=1, which the launcher supplies.

## Generate and serve

The [demo guide](../docs/demos.md) owns checkpoint setup, program packages, launch commands, HTTP usage and shutdown.

- [Four-chip GPT-2](../docs/demos.md#four-chip-gpt-2-small) shows live batching, chip activity and KV residency in the browser. Assemble its sealed program bundle from the included schedules and the GPT-2 checkpoint. The [four-chip design](../docs/gpt2-four-chip-design.md) covers compilation, placement, resident KV and the vLLM worker.
- [Single-chip TinyStories](../docs/demos.md#single-chip-tinystories) builds its serving package from the checkpoint and runs in the terminal or over HTTP.
- [The serving package layout](../docs/demos.md#serving-package-layout) explains which files move between deployments.

## Packages, reliability and boundaries

Programs are digest-keyed, locked and atomically published in the shared cache. The GPT-2 launcher validates the separate sealed four-chip bundle. Checkpoints are external to programs and objects. Twin checking compares simulated outputs bit for bit when enabled.

Unsafe steps terminate affected requests before publishing their new tokens. Keep a stable OPENTSP_RAS_DOMAIN per independent deployment and preserve its durable incident state. Unclean shutdown may require offline investigation and clearance. See the [RAS contract](../opentsp-arch-spec/docs/production-ras.md).

System.run uses an inclusive cycle limit: 0 through 2**64-2 are accepted; 2**64-1 is rejected before execution. A limit return is resumable. Device cycles exclude host embedding, scheduling, sampling and simulation overhead. Traffic counters are interface counts, not hardware bandwidth. Masked stale nonfinite KV cannot contaminate attention; active nonfinite values may propagate. Whole-machine formal assurance remains [unproved](../docs/assurance.md).

Source lives in [src/opentsp_vllm](src/opentsp_vllm), the Rust binding in [native](native), and tests in [tests](tests). See the [serving test tiers](../docs/testing.md) and [license](LICENSE-APACHE-2.0).
