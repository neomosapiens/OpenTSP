// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Simulator lints: a general, NON-NORMATIVE channel for
// sanity checks that help the user of a simulator but are not architectural behaviour.
//
//  - A lint never changes machine state, never counts as a diagnostic (the `E` line of tsp_sim),
//    is not part of the golden vectors, and an implementation need not report it.
//  - The code is `OTS-W-<area>-<nnn>` (W for warning, as OTS-E for the diagnostics).
//  - A lint reaches the harness through lint_callback (core/callbacks.sail); tsp_sim prints
//    `T lint <cycle> queue <q> <code>` with --lint.
//
// To add a lint (opentsp-arch-spec/docs/validation.md, "Simulator lints"):
//  1. ONE line here: `let LINT_<NAME> : string = "OTS-W-<AREA>-<nnn>"   // what it detects`;
//  2. ONE call where the model detects it: `lint(q, c, LINT_<NAME>)`;
//  3. one row in the lint table of validation.md (tools/check_lints.py fails until every code
//     defined here is in the table and every row is defined here).

let LINT_IQ_UNDERFLOW : string = "OTS-W-IQ-001"   // a parcel's text arrived after the queue could have issued it (the IQ ran dry)
let LINT_RAS_EVICT : string = "OTS-W-RAS-001"   // a live spare-bit error report was evicted from the 256-entry report array by newer reports

// q is a queue 0..143, or 144 for the control channel (reported as 65535).
function lint(q : range(0, 144), c : nat, code : string) -> unit =
  lint_callback(c, if q == 144 then 65535 else q, code)
