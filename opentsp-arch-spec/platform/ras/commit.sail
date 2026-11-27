// Platform management interface. Each call samples one management-clock edge.
// The chip's architectural interrupt mask controls notification independently.
struct attempt = {
  running : bool, fenced : bool, epoch : bits(64), members : bits(16), acks : bits(16), remaining : bits(32)
}
struct evidence = {
  reset_seen : bool, fatal : bool, bad_state : bool, needs_service : bool, report_epoch : bits(64), settled : bits(16),
  write_valid : bool, write_epoch : bits(64), visible : bool, visible_epoch : bits(64)
}

function ras_arrivals(s : attempt, e : evidence) -> bits(16) =
  if e.report_epoch == s.epoch then e.settled else 0x0000

function ras_live(s : attempt, e : evidence) -> bool =
  s.running & not(e.bad_state) & not(e.needs_service) & s.remaining != 0x00000001 & not(s.fenced) & not(e.reset_seen) & not(e.fatal)

function ras_write(s : attempt, e : evidence) -> bool =
  ras_live(s, e) & e.write_valid & e.write_epoch == s.epoch

function ras_publish(s : attempt, e : evidence) -> bool =
  ras_live(s, e) & e.visible & e.visible_epoch == s.epoch & not(ras_write(s, e)) &
  ((s.acks | ras_arrivals(s, e)) & s.members) == s.members

function ras_reopen(isolated : bool, healthy : bool, permitted : bool, epoch : bits(64)) -> bool =
  isolated & healthy & permitted & epoch != 0xffffffffffffffff

$property
function p_fault_suppresses_effects(s : attempt, e : evidence) -> bool =
  not(e.fatal | e.bad_state | e.needs_service | e.reset_seen | s.fenced | not(s.running)) | (not(ras_write(s, e)) & not(ras_publish(s, e)))

$property
function p_deadline_wins(s : attempt, e : evidence) -> bool =
  s.remaining != 0x00000001 | (not(ras_write(s, e)) & not(ras_publish(s, e)))

$property
function p_stale_visibility(s : attempt, e : evidence) -> bool =
  e.visible_epoch == s.epoch | not(ras_publish(s, e))

$property
function p_stale_writes(s : attempt, e : evidence) -> bool =
  e.write_epoch == s.epoch | not(ras_write(s, e))

$property
function p_missing_participant(s : attempt, e : evidence, i : bits(4)) -> bool = {
  let k = unsigned(i);
  not(ras_publish(s, e)) | s.members[k] == bitzero | s.acks[k] == bitone |
    (e.report_epoch == s.epoch & e.settled[k] == bitone)
}

$property
function p_last_write_is_private(s : attempt, e : evidence) -> bool =
  not(ras_write(s, e) & ras_publish(s, e))

$property
function p_reopen_needs_evidence(a : bool, b : bool, c : bool, ep : bits(64)) -> bool =
  not(ras_reopen(a,b,c,ep)) | (a & b & c & ep != 0xffffffffffffffff)

// Existence of a normal completion under the above predicates.
$counterexample
function c_no_success(s : attempt, e : evidence) -> bool =
  not(ras_publish(s, e) & s.members != 0x0000)

// The refinement check binds these external observations to the synthesized
// RTL signals. With unconstrained observations a mismatch must be possible.
struct gate_outputs = { write_allowed : bool, published : bool }
$counterexample
function c_gate_always_agrees(s : attempt, e : evidence, o : gate_outputs) -> bool =
  o.write_allowed == ras_write(s, e) & o.published == ras_publish(s, e)
