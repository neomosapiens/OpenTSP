// internal_error on a reachable path: in a proof build (-D OPENTSP_PROOF_BUILD) the SMT backend must translate it;
// in the default build it stops at dec_str (jib_smt.ml:245). export_check.sh runs both.
val pick : bits(8) -> bits(8)
function pick(x) = if unsigned(x) < 200 then x else internal_error(__FILE__, __LINE__, "unreachable")

$property
function p_pick(x : bits(8)) -> bool = pick(x) == x
