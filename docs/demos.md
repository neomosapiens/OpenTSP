# OpenTSP demos

Both demos run actual model inference through vLLM and the OpenTSP instruction-set simulator.

| Demo | What to try | Runtime inputs |
|---|---|---|
| [Four-chip GPT-2 small](#four-chip-gpt-2-small) | Follow eight live request streams, mixed prefill and decode, and KV residency in the browser. | Four-chip ELF bundle and shared GPT-2 Safetensors. Build both from the included schedules and the downloaded checkpoint. |
| [Single-chip TinyStories](#single-chip-tinystories) | Generate a story from the terminal, explore prefix reuse in a batch, or call the local HTTP API. | TinyStories ELF bundle, shared Safetensors and GPT-2 tokenizer. Build the bundle locally from the TinyStories checkpoint. |

## Shared setup

Install the [native or container prerequisites](installation.md), then run from the repository root in Bash:

```bash
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
just vllm-env
just vllm-binding
```

These recipes install the pinned CPU vLLM environment and build the native ISS binding for this source directory. Use the Hugging Face CLI (`hf`) from the specification environment to obtain checkpoint files. Both HTTP servers bind to loopback by default. They are intended for local development.

## Four-chip GPT-2 small

Four simulated chips run GPT-2 small. Chips 0–2 each own four transformer layers and their resident KV. Chip 3 runs the final normalization and the full vocabulary projection. The host tokenizes inputs, computes token-plus-position embeddings, schedules rows and samples logits. See the [four-chip design](gpt2-four-chip-design.md) for tensor mapping, the compiled package and the vLLM step lifecycle.

The [GPT-2 small checkpoint](https://huggingface.co/openai-community/gpt2) has 124,439,808 unique parameters, 12 layers, 12 attention heads per layer, hidden width 768 and 1,024 model positions. Embedding and vocabulary weights are tied.

The route has a shared pool of 256 positions across 16 usable 16-slot pages. vLLM admits up to eight requests and eight scheduled rows per step. The sealed single-page paired schedule issues a pair every 30,534 modeled cycles. Longer and unevenly grouped requests use the applicable sealed fallback.

### GPT-2 setup

```bash
hf download gpt2 model.safetensors config.json generation_config.json vocab.json merges.txt tokenizer.json tokenizer_config.json \
  --local-dir "$HOME/.cache/opentsp/checkpoints/gpt2"
```

The verified `model.safetensors` has SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.

Build the programs from the released schedules and the downloaded checkpoint:

```bash
just gpt2-programs
```

The builder verifies the checkpoint tensors, assembles all five variants from the released schedules and writes one shared `weights.safetensors`. The bundle includes the vLLM config and tokenizer under `model/`. The upstream checkpoint is a build input. The default runtime only needs the generated bundle.

Use `just gpt2-programs --checkpoint /path/to/gpt2 --output /path/to/new-bundle` for other locations. The output path must be new. Set `OPENTSP_GPT2_4_PROGRAMS` to that path when running the demo. Each of the five variant directories contains a manifest and four stage units. All deployment inputs fit under 512 MiB on the initial checked build, separately from the upstream checkpoint and optional device layout cache.

To prepare optional device images while building, add `--prebuild-layout /path/to/private-cache`. Set `OPENTSP_DEVICE_LAYOUT_CACHE` to that directory when launching inference. Use a private directory owned by the serving account. A missing, stale or unreadable image is rebuilt from verified canonical files. The cache contains derived immutable SRAM payloads and holds no incident history or live KV.

### Four-chip browser demo

~~~bash
just gpt2-demo
~~~

Open [the loopback demo](http://127.0.0.1:18765) and choose **Run 8 users**. Each card shows its prompt, the streamed answer, and the originating batch and output event. In **Live execution**, the batch header shows scheduled prefill and decode. Rows keep their user color while the view marks observed host and chip arrivals. The selected request rail follows prompt prefill and successive decode steps. Token text appears when vLLM streams it. **Compact** folds the guide, and **Custom prompt** submits your own request.

Open **Execution details** for batch history, timings and resident KV. Select a KV page to see its per-layer address range. Use **Warm prefix**, **Run staggered pair** and **Grow KV** to explore allocator behavior. **Stop** cancels the active browser run. Ctrl-C shuts down the demo server and its vLLM engine. Set `DEMO_PORT` and `VLLM_PORT` to choose other loopback ports.

The two throughput panels show aggregate output tokens per second and the average per user across eight concurrent requests. **Software ISS** measures browser wall time from submission. **Projected TSP silicon** scales the corresponding observed simulation cycles at the selected frequency and excludes host overhead. The projection makes no physical hardware claim. Inspect the same schedule with the [Schedule Explorer](../opentsp-sw-toolchain/tools/schedule-example.html). [SHIP diagram inspiration, MLSys slides 15–18](https://mlsys.org/media/mlsys-2026/Slides/3834_VmkjzHq.pdf#page=15)

### GPT-2 HTTP API

After [GPT-2 setup](#gpt-2-setup), start the OpenAI-compatible endpoint:

~~~bash
"${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}/bin/python" -m opentsp_vllm.serve \
  --tsp-programs "${OPENTSP_GPT2_4_PROGRAMS:-$HOME/.cache/opentsp/gpt2-four-chip}" --port 8000
~~~

In another terminal:

~~~bash
curl -fsS http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gpt2","prompt":"The capital of France is","max_tokens":4,"temperature":0}'
~~~

Use `--tsp-check` to compare each step with the numerical twin. The server defaults to four sequences. Set `--max-num-seqs 8` for eight. An invalid serving limit returns exit 2 before startup. GPT-2 uses the completions route. Ctrl-C requests graceful shutdown. Review the [RAS contract](../opentsp-arch-spec/docs/production-ras.md) before clearing durable incident state after unclean shutdown.

## Single-chip TinyStories

The [eight-layer TinyStories checkpoint](https://huggingface.co/calum/tinystories-gpt2-3M) generates text through vLLM on one simulated OpenTSP chip. The host tokenizes, adds the token and position embeddings, schedules work and samples the output. The chip executes all transformer layers, retains the KV cache in SRAM and computes the final normalization and all 50,257 logits.

### TinyStories setup

```bash
hf download calum/tinystories-gpt2-3M config.json generation_config.json pytorch_model.bin
hf download gpt2 vocab.json merges.txt tokenizer.json tokenizer_config.json
just showcase-prepare
```

`just showcase-prepare` reads the checkpoint and GPT-2 tokenizer from the local Hugging Face cache. It writes the tokenizer and model configuration under `model/` and one digest-keyed serving package under `programs/`. The `serving` link selects the exact published package. Set `OPENTSP_SHOWCASE_DIR` to choose another root, or `JOBS` to set the number of stage builders. A repeat verifies and reuses a matching package. Run `just showcase-prepare` again after changing OpenTSP source. If startup reports that an object was built against another `encoding.json`, rebuild and select the package with this checkout before launching.

### Interactive and one-shot generation

```bash
just showcase                                  # an interactive session
PROMPT="Once upon a time" just showcase        # one shot: the story continues and the program exits
```

The session accepts the following commands.

| command | effect |
|---|---|
| `/max N` | at most N new tokens (the prompt and the new tokens share 128 positions) |
| `/temp X` | sampler temperature, 0 (greedy) to 2 |
| `/topk K` | sample among the K likeliest tokens; 0 turns the filter off |
| `/seed N` or `/seed none` | the sampler's seed (default 1, so a prompt gives the same story every time; `none`: random) |
| `/verbose [on|off]` | one line per scheduler step (token count, cycles, seconds) instead of the streamed text |
| `/start` | a story from nothing: the prompt is just `<|endoftext|>` |
| `/set`, `/help`, `/quit` | show the settings, help, leave (Ctrl-D also leaves) |

Ctrl-C stops a generation after the step that is running; the request is aborted in vLLM and its KV blocks are freed. Ctrl-C at the prompt only reminds you how to leave. `//text` starts a prompt with a slash. For one shot the sampler can be set with `MAX_TOKENS`, `TEMPERATURE`, `TOP_K`, `SEED` and `VERBOSE=1` (or the flags of `python -m opentsp_vllm.showcase.cli chat`).

While the prompt is being prefilled (no token yet), a dim note shows `prefilling the prompt on the chip: 16/34 tokens`; it is erased when the first token arrives. After every generation one line reports the generated and cached token counts, scheduler steps, total ISS cycles, wall time, time to first token, throughput and stop reason.

The fixed-program showcase produced this greedy continuation.

```text
$ PROMPT="Once upon a time" MAX_TOKENS=4 TEMPERATURE=0 just showcase
Once upon a time, there was a
```

### TinyStories batching and HTTP API

```bash
MAX_TOKENS=2 TEMPERATURE=0 just showcase-batch
N=6 just showcase-batch
PORT=8001 just showcase-serve
```

The scheduler accepts up to four sequences and eight tokens per step. The chip handles real tokens in causal order. A request can use up to 128 positions including its prompt and generated text. The cache contains 127 allocatable pages of 16 tokens and a reserved null page. Requests can share completed prefix pages. The batch command reports how many prompt tokens each request reused.

```bash
curl -s http://127.0.0.1:8001/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"tinystories","prompt":"Deep in the woods","max_tokens":3,"temperature":0}'
```

The service returns HTTP 400 for a prompt beyond the context limit. Ctrl-C shuts down the server. Within the interactive showcase it aborts the current request and returns to the prompt.

If an unsafe chip step interrupts a stream, the terminal error ends that request. Text already delivered remains valid under the documented fault model. The service discards the failed step and the chip’s KV cache. A client can submit a fresh request after recovery. If the chip stays quarantined, inspect the incident under the same deployment identity before clearing it offline. [Fault policy and operator commands](../opentsp-arch-spec/docs/production-ras.md)

### TinyStories execution and KV

1. The host writes the embedded hidden vector, physical visibility masks and destination maps into designated SRAM words. The loader selects immutable program stages. An instruction page may be reloaded when stages share the same text area.
2. Each layer projects Q, K and V. Scatter uses the maps to write the new f16 K row and f32 V row into the device pool. The scheduler's explicit block-copy operation also runs on chip, using Gather and Scatter with source and destination maps. The host writes the two address maps for each copied word; it does not read or write the KV payload through the host word interface. Identity copies launch no stages.
3. Attention processes the request's occupied physical chunks in ascending order. Each chunk covers 128 positions. Masked scores become negative infinity and masked values contribute zero, including stale nonfinite data. Softmax combines chunks before its within-chunk reduction. The context sum uses two interleaved f32 chains carried across chunk boundaries. A request contained in one chunk uses its own fixed softmax stage.
4. The chip runs the output projection, residuals and MLP. For a token requiring logits it runs the final normalization and tied head. Only the head's output regions are returned to the host. Warm restarts retain SRAM between stages. The binding's explicit replay mode copies SRAM through host memory for simulator conformance.

An engine step can launch many chip stages. The displayed ISS cycles sum their simulated execution and exclude host work. The patched-bucket backend uses a different arithmetic order and cycle count. The [package layout](#serving-package-layout) describes how to recognize the two packages.

## Serving package layout

TinyStories and GPT-2 small use the same on-disk serving schema, `opentsp.serving` version 1. A unit is a separately compiled program. TinyStories executes units successively on one chip. GPT-2 places its layer and head units on different chips in a pipeline. The host reads the manifests and bindings. The device executes the instruction images. The ISA and binary headers are documented in the [assembler reference](../opentsp-arch-spec/docs/assembly.md).

| Path | Format | What the loader does with it |
|---|---|---|
| `manifest.json` for TinyStories, `<variant>/manifest.json` for GPT-2 | JSON text | Identifies the model, profile, route and compiled stages. TinyStories also records its SRAM regions for inputs and cache management. GPT-2 variant manifests bind to the digest of the model tensors at the shared bundle root. |
| `weights.safetensors` at the bundle root | Safetensors | Holds the shared logical model tensors, including the tables the host reads for embedding. |
| `<unit>/program.elf` | ELF32 | Holds queue images, placements, fetched instruction parcels, literals, zero ranges and any logical tensor bindings. |
| `<unit>/bindings.npz` | Binary ZIP of NumPy arrays | TinyStories declares host readback regions. The head also maps the returned byte planes to vocabulary indices. Internal stages return zero regions. GPT-2 records scheduled host DMA cycles. |
| `<unit>/program.json` | JSON text | Identifies the stage, its cycle budget and SHA-256 checksums for the binary files and stage metadata. The model manifest identifies the descriptors that belong to its build. |

For example, TinyStories' `pre0/` projects the hidden state for layer zero and writes the new K and V into SRAM. `score0_15/` computes that layer's scores for physical chunk 15. `head/` returns logits. Each GPT-2 four-chip variant has `s0/` through `s3/`. The first three units each contain four transformer layers. The fourth holds final normalization and the vocabulary head.

### What changes between tokens

The host computes the token-plus-position embedding and supplies data describing visible positions and the destination KV slot. TinyStories writes these inputs through the SRAM aperture. GPT-2 also uses scheduled DMA. The host selects and launches the compiled stages, and the loader may restore their immutable instruction pages into the shared SRAM text area on a stage switch. The fixed TinyStories showcase and the sealed GPT-2 route use precompiled instructions.

The optional TinyStories patched-bucket backend reads instruction operand sites from `bindings.npz`. The showcase uses fixed programs. Run `just showcase-prepare` to build the fixed-program package and update the `serving` link to that package.

### Moving a workload

Copy the complete deployment directory including `weights.safetensors`, config, tokenizer and all published program units. Keep every GPT-2 variant together under one bundle root. Keep TinyStories' `model/` directory and the linked serving package. The optional local cache can be rebuilt at the destination.

On cold recovery the loader checks the programs, metadata and weights against the identity recorded at healthy startup. The package contains neither live KV nor the incident journal. Keep the deployment’s stable RAS domain and durable state directory across package moves so its recovery history and one-attempt limit continue to apply. [Recovery and operator procedures](../opentsp-arch-spec/docs/production-ras.md)

## Verification

**GPT-2.** Run `just vllm-surface` and `just vllm-unit` to exercise the plugin and the serving interface. `just vllm-ship-e2e` covers the scheduler, resident KV and numerical parity with a small test model. Use the eight-user browser demo to exercise the full model from the sealed bundle. Run the full-model recovery campaign separately.

**TinyStories.** `just showcase-test` exercises the session and the fixed backend through vLLM. `just vllm-binding-test` includes a two-layer test comparing logits and the device's context accumulator bits with a physical-order oracle at the 128-position limit. It also checks a device-only block copy, stale masked data, and the ISS patch counters. Use `just showcase-decode` to compare the logical-page-order driver with its numerical reference.

See [testing](testing.md) for test tiers and [assurance limitations](assurance.md) for the scope of correctness claims.
