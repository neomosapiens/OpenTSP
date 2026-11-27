// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Obligations on the hold helpers that bind the model definitions in
// postlude/tick.sail. The prove_extra hold project loads postlude, so
// hold_decide / view_free / path_view / hold_view / mem_path are the model's — this
// file does not redeclare them.
//
// View construction uses a Sail vector literal for vp (highest index first) so the
// obligations never call the model's path_views (vector_init), which Sail's SMT
// backend mis-emits. Path arguments stay option(mem_path); free parameters avoid
// literal Some(0)/Some(1), which Sail SMT also mis-emits for option(range).

function hold_idle_path_probe() -> path_view =
  struct { pbusy = false, ptail = false, pstate = AGT_IDLE }

function hold_free_view_probe() -> hold_view =
  struct { vparked = false, vhost = false, vbusy = false, vrepeat = false, vpoll = false,
           vp = [hold_idle_path_probe(), hold_idle_path_probe()] }

// A free view is free for every path / cfg_record (covers None and Some).
$property
function p_hold_free(path : option(mem_path), cfg : bool) -> bool = {
  let hv = hold_free_view_probe();
  view_free(hv) & not(hold_decide(hv, false, false, path, cfg))
}

// A parked queue holds a non-barrier instruction and does not hold its own Sync/Notify.
$property
function p_hold_parked_barrier() -> bool = {
  let hv = { hold_free_view_probe() with vparked = true };
  hold_decide(hv, false, false, None(), false) &
    not(hold_decide(hv, true, false, None(), false)) &
    not(view_free(hv))
}
