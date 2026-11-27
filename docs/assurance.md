# Assurance and limitations

OpenTSP has executable tests, differential checks, bounded SMT obligations and native Lean component/composition proofs. Each result applies to its actual statement, source, domain and assumptions. They do **not** compose into a machine-checked correctness theorem for the complete Sail/C machine over all admitted histories. **Whole-machine formal correctness remains unproved.**

The model and tools support architecture exploration, software-toolchain development, simulation and RTL reference work within their documented contracts. They are not a fully formally verified machine or hardware/board signoff.

## Missing whole-machine guarantees

| Area | Connection not established by an end-to-end machine-checked theorem |
|---|---|
| Grants and admission | Grants survive actual fetch, AGT, dispatch and other-path activity, and every consumer uses admitted resources |
| Scheduled ownership and timing | Complete queue histories preserve payload, issuer, absolute execution/commit times and ring/resource lifetimes together |
| Stream-register behavior | Upstream selection, lane merging, distinct-writer priority, poison, lifetime, eviction, reports and commit compose across all producer/consumer histories |
| Instruction queues and callers | Reset, delivery, fetch, receive, Repeat, AGT and consumption preserve contents, tags, ECC and allocation through partial failures |
| External environment | Actual C callbacks, GMP storage, allocation, host/link values, addresses, time and IRQ effects satisfy every proof contract |
| Event census | Every relevant control, dispatch and external frontier satisfies the global reachable-state next-event argument |
| Event skipping | Accelerated histories preserve architectural state and required ordered effects, including barrier and patrol catch-up |
| Internal failures and matrix integrity | All caller-domain arguments and expected-row/checker reasoning compose into actual complete-machine execution |
| Diagnostics and partial effects | Every subsystem preserves report order, first-error/count behavior and retained side effects through actual callbacks |

These are missing guarantees, not assertions that each corresponding defect exists. Untested combinations could expose wrong results, ordering, diagnostic behavior or internal failures. Current evidence does not establish a numerical probability of escaped defects.

Numerical proof coverage also lacks a general complete implementation decomposition for floating-point operations and a full termination/rounding argument for the capped transcendental procedure. Numerical regression and differential checks remain necessary.

## Proof and implementation trust

Lean results use the permitted core axioms and their stated semantic hypotheses. Some require explicit port/frame, representation, image-value, ordering or producer-domain assumptions not derived from complete implementation histories. The checker must cover the actual public declarations and source identities; an allowed-axiom check alone does not establish the missing hypotheses.

The Sail/Lean frontend, proof checker, generated code, native compiler, C harness, GMP, operating system and platform remain part of the trust boundary. Local C contracts do not supply a complete C-to-Lean refinement bridge. Tests that pass on a configured host do not prove fresh-OS installation or every supported platform.

The matrix checker compares faulted rows with independently recomputed expected rows. Its contract assumes reliable captured operands and checker computation in the documented injected-fault model. It does not protect against arbitrary simultaneous corruption of both computation and checker.

## Interpreting validation

A successful mutation ratchet means all observed outcomes match its registered expectations. Two registered expected survivors remain; a pass does not mean every mutation is detected. Component proof counts and test counts are not coverage percentages.

Optional observer wait records sample visited cycles and may differ under event skipping; required architectural effects still must conform. Model cycles, energy estimates, interface traffic and host simulator elapsed time are not physical hardware performance measurements.

Downstream users should retain exact source/profile/input identities and their own differential and regression cases. Future formal work can reveal defects requiring semantic corrections or downstream changes; compatibility of unknown fixes is not promised. See the [validation chapter](../opentsp-arch-spec/docs/validation.md), [platform contract](../opentsp-arch-spec/docs/platform.md) and [RAS contract](../opentsp-arch-spec/docs/production-ras.md) for the precise local boundaries.
