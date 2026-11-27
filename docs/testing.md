# Testing OpenTSP

Start with the [toolchain setup](installation.md). The [architecture validation chapter](../opentsp-arch-spec/docs/validation.md) defines the modeled obligations and their oracles. Run checks from the source directory with its packages installed. Use the same executable, profile, weights, host input, and seed when reproducing a result.

| Tier | Commands | Evidence |
|---|---|---|
| Source and build | `just lint`, `just check-docs`, `just check-freeze` | Formatting, compiler lints, test wiring, documentation and specification source consistency. The gates work in a source directory without Git metadata. |
| Workspace | `just test` | Rust integration tests, the replay feature, performance-model formulas and formal infrastructure controls. It needs the native reference toolchain for tests that use the C emulator. |
| Assembler and simulator parity | `just asm-diff`, `just iss-diff`, `just iss-vectors`, `just harness-contract` | Independent assembler outputs and refusal codes, model/ISS diagnostic order, simulated cycles, traces and data for the selected profiles and programs. |
| Program journeys | `just e2e`, `just programs` | Complete paths from assembly and weight packing through both simulators and native ISS execution where its input supports the case. |
| Reference and component proof | `just model-test`, `just model-prove`, `just verify-all`, `just ras-prove` | Model-linked vectors and directed checks, SMT, exhaustive decode, Lean and refinement checks within their stated domains. Lean and management RTL proofs have their own installation requirements. |
| Serving | `just vllm-surface`, `just vllm-unit`, `just vllm-binding-test`, `just showcase-test`, `just vllm-serve-test` | Pinned vLLM API, model runner, Rust binding and real engine on both simulated backends. These require the serving environment. Some cases require locally cached checkpoints or cold package generation. |
| GPT-2 and fault recovery | `just vllm-ship-e2e`, `just ras-qualify`, `just gpt2-demo` | The small fixture exercises scheduling, resident KV and numerical parity. The browser runs the full four-chip model from the [sealed deployment bundle](demos.md#gpt-2-setup). Building the bundle requires the upstream checkpoint. The recovery campaign exercises result fencing and starts both real models. Set `OPENTSP_SHIP_PACKAGE` when the bundle is outside its default location. See [serving-domain validation](../opentsp-arch-spec/docs/validation.md#serving-domain-ras-verification). |
| Extended campaigns | `just prove-nightly`, `just model-coverage`, `just model-transcendental` | Mutation sensitivity, profile sweeps, proof and test reach, and large numerical campaigns. Requirements and runtimes depend on the campaign and host. |

For a behavioral change, start with the smallest assertion that would fail for the old behavior. Derive the expected result from the specification, a distinct oracle, or a preserved known result. A negative control should expose the actual failure. Advance to an external entry point when the change affects wiring, configuration, build artifacts, or shutdown.

Keep the reference, candidate, and test data identities with a failure. Use a fixed seed for reproduction and record the variants exercised. Compare actual outputs and statuses. A matching timeout, missing backend, or empty test selection does not establish equivalence. A selected suite qualifies only the checks it lists. General serving tests report skips when an upstream checkpoint required for comparison is missing. Run the dedicated suites to qualify each full model from its deployment bundle.

For the real-model recovery campaign, after preparing both demos:

```bash
OPENTSP_SHIP_PACKAGE="${OPENTSP_GPT2_4_PROGRAMS:-$HOME/.cache/opentsp/gpt2-four-chip}" just ras-qualify
```

Set `OPENTSP_SHOWCASE` and `OPENTSP_GPT2` for non-default TinyStories and GPT-2 test paths. Keep the sealed-package override local to this command. The two-layer GPT-2 test fixtures use their own generated cache.

The worker and device tests use controlled failures to exercise error reporting and cleanup. Serving qualification also runs the actual engine and simulated chips. Cross-implementation comparison can catch divergent execution but inherits errors shared by the specification and its generated data. The separate numerical oracle and component proofs address specific parts of that risk. The [assurance limitations](assurance.md) define the remaining whole-machine and hardware scope.
