// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// ICU instructions (ISCA 2020 sec. III-A and Table I; HC34 p.25, p.27): NOP, Sync, Notify,
// Repeat, Ifetch, Config, SetVL. They are common to all functional slices, so they issue on
// every data queue. The control channel (queue_id 0xFFFF) takes NOP, Sync, Notify and
// Config only (postlude/tick.sail control). The barrier takes effect when Sync and Notify
// dispatch, in the cycle driver (postlude/tick.sail), and Repeat is applied there too, so
// execute does little here. NOP n is not dispatched: it is a timing prefix of the next
// parcel in the instruction queue (postlude/iq.sail, design sec. 7.3); its execute clause
// is never reached from the queue.

// 32-bit layout: slice 31..28 = 0, opcode 27..22, reserved 21..16, imm 15..0.

union clause instruction = NOP : bits(16)
union clause instruction = SYNC : unit
union clause instruction = NOTIFY : unit

mapping clause encdec32 = NOP(n)     <-> 0x0 @ 0b000000 @ 0b000000 @ n when n != 0x0000
mapping clause encdec32 = SYNC()     <-> 0x0 @ 0b000001 @ 0b000000 @ 0x0000
mapping clause encdec32 = NOTIFY()   <-> 0x0 @ 0b000010 @ 0b000000 @ 0x0000

// Repeat n, d: re-issue the queue's previous instruction n more times with d cycles between
// iterations, a spacing of d + 1 (ISCA 2020 Table I). Layout: n 15..4 (1..4095), d 3..0 (0..15; d = 0 is no cycle between, consecutive cycles)
// - opentsp field split.
union clause instruction = REPEAT : (bits(12), bits(4))
mapping clause encdec32 = REPEAT(n, d) <-> 0x0 @ 0b000011 @ 0b000000 @ n @ d
  when n != 0x000

// Ifetch s: fetch instruction text from the ONE stream operand s (ISCA 2020 sec. III-A3: "a
// single stream operand which carries the text of the instructions in their program order,
// filling the IQ with 640-bytes (a pair of 320-byte vectors)"). The pair is two consecutive
// vectors of that stream: the first is read at the Ifetch's execute cycle t, the second at
// t + 1 (a stream carries one vector per cycle); any stream is legal. Implementation choices
// (isa-db/provenance.yaml CU.Ifetch): a missing beat reports TIM-001 on its sample
// edge and discards the pair. The payload enters the IQ from t+1+icu.fetch_latency.
// Layout: s 15..10.
union clause instruction = IFETCH : streamref
mapping clause encdec32 = IFETCH(Streamref(st)) <-> 0x0 @ 0b000100 @ 0b000000 @ st @ 0b0000000000

// IfetchM a: the local-memory form of Ifetch (ISCA 2020 Table I: "Ifetch Fetch
// instructions from streams or local memory"). The published sources give no
// operand or address rule, so opentsp defines it: it reads the two consecutive words a, a+1 of the issuing
// queue's own MEM slice, the same 640 bytes (two 320-byte vectors) as the stream form;
// only MEM queues issue it (elsewhere OTS-E-QUE-002); a+1 must be inside the slice's
// segment (SetSegment), else OTS-E-OPR-001. Its text enters the IQ like the stream form's.
// It takes no AGT offset and uses the Ifetch slot
// of a MEM queue and reads through the R port (ISCA 2020 sec. III-B, one read
// and one write per cycle in different banks): with a Read or Gather of either thread in
// its parcel it is OTS-E-QUE-001 (all the parcel's instructions); while the R path is held
// it waits at the head of the queue. It reserves two consecutive R-port cycles, reading
// a then a+1, including each word's integrity state at its read. A Write/Scatter or W-path
// AGT reference must avoid the bank of that cycle's read. A first-cycle conflict cancels
// both new operations; a second-cycle conflict rejects only the later W operation, since
// the fetch is already in flight. Repeat iterations that overlap its two-cycle read
// occupancy report QUE-001. Layout: opcode 7, a 15..0 (the 16-bit address field; bits above addr_bits are zero).
union clause instruction = IFETCHM : sram_addr
mapping clause encdec32 = IFETCHM(a) <-> 0x0 @ 0b000111 @ 0b000000 @ sram_field(a)

// Config v: the published Config name has a low-power motivation (ISCA 2020 Table I).
// OpenTSP defines the MEM liquid-thread split below; that rule is not published there.
// On other queues this is an accepted power hint with no architectural state, result,
// diagnostic or timing effect. No physical power state is modelled. Layout: imm 15..0.
union clause instruction = CONFIG : bits(16)
mapping clause encdec32 = CONFIG(v) <-> 0x0 @ 0b000101 @ 0b000000 @ v

// SetVL n: the vector length of the issuing queue's slice becomes n x 16 lanes; the
// other superlanes are masked. This instruction and its masking semantics are OpenTSP
// choices; ISCA 2020 sec. II-F motivates activity control, not these encoding/masking choices.
// Masked MEM storage is unchanged and masked result superlanes are zero (core/sram.sail).
// This does not disable whole-vector unit state or input integrity checks under the owning
// instruction's rules, and it models no physical power gating. n = 1..32 on every profile (a 6-bit field: bits 5..0); a value above the profile's
// superlane count is the count (the clamp of execute), so a program written for more superlanes runs on a narrower profile.
union clause instruction = SETVL : bits(6)
mapping clause encdec32 = SETVL(n) <-> 0x0 @ 0b000110 @ 0b000000 @ 0b0000000000 @ n
  when unsigned(n) >= 1 & unsigned(n) <= 32

function clause execute NOP(_) = Issued()
function clause execute SYNC() = Issued()
function clause execute NOTIFY() = Issued()
function clause execute REPEAT(_, _) = Issued()
// Config v: on a MEM queue, v = SL (0..superlanes - 1) splits the slice's tiles into liquid threads:
// tiles 0..SL are thread 0, the rest thread 1 (with 20 superlanes Config 6 puts tiles 0-6 in thread 0 and
// tiles 7-19 in thread 1). A value of superlanes - 1 or more puts
// every tile in thread 0. On other queues Config is an architectural no-op; power hints do not model a state or timing change.
function clause execute CONFIG(v) = {
  let q = issue_queue;
  if q <= 87 then {
    let sl = unsigned(v);
    mem_thread_split[q] = if sl < superlanes then sl + 1 else superlanes
  };
  Issued()
}
function clause execute SETVL(n) = {
  let q = issue_queue;
  if q <= 143 then queue_vl[q] = max(1, min(unsigned(n), superlanes));
  Issued()
}
// The text words of the payload carry the check bits the operands had when read (core/ecc.sail): corrected operands
// give valid words, a poisoned superlane an uncorrectable text word.
struct stream_fetch = { ready : nat, source : streamref, first : vec, ov : ecc_ov }
register stream_fetches : vector(145, option(stream_fetch)) = vector_init(None())

function stream_fetch_arm(q : range(0, 144), st : streamref, t : nat, data : vec, ov : ecc_ov) -> unit =
  stream_fetches[q] = Some(struct { ready = t + 1, source = st, first = data, ov = ov })

function stream_fetch_take(q : queue_id, c : nat) -> option(stream_fetch) =
  match stream_fetches[q] {
    Some(f) => if f.ready <= c then { stream_fetches[q] = None(); Some(f) } else None(),
    None() => None(),
  }

function stream_fetch_start(st : streamref) -> ExecutionResult =
  match read_planes(st, 1, exec_cycle()) {
    Err(d) => Diagnostic(d),
    Ok(a) => if fill_len(fill_fifo[issue_queue]) >= iq_staging_depth then Diagnostic(IQ_STAGING)
      else {
        stream_fetch_arm(issue_queue, st, exec_cycle(), a[0], ecc_poison_ov(ecc_bad));
        Issued()
      },
  }

function clause execute IFETCH(st) = stream_fetch_start(st)

// Complete after commits, at the same read phase as other stream consumers.
// Finish the previous pair before this queue can start its next one.
function stream_fetch_service(q : queue_id, c : nat) -> unit =
  match stream_fetch_take(q, c) {
    Some(f) => {
      issue_queue = q;
      issue_unit = issuer_of(q);
      ecc_begin();
      match read_planes(f.source, 1, c) {
        Err(d) => diag_report(c, q, d),
        Ok(b) => {
          if not(iq_fetch_ov(q, c, f.first, b[0], f.ov, ecc_poison_ov(ecc_bad))) then
            diag_report(c, q, IQ_STAGING)
          else if ecc_bad != sail_zeros(superlanes) then diag_report(c, q, ECC_UNC)
        },
      }
    },
    None() => (),
  }

function stream_fetch_complete(c : nat) -> unit =
  foreach (q from 0 to 143) { stream_fetch_service(q, c) }

function clause timing_of NOP(_) = T_CU
function clause timing_of SYNC() = T_CU
function clause timing_of NOTIFY() = T_CU
function clause timing_of REPEAT(_) = T_CU
function clause timing_of IFETCH(_) = T_CU
function clause timing_of IFETCHM(_) = T_CU
function clause timing_of CONFIG(_) = T_CU
function clause timing_of SETVL(_) = T_CU

function clause entry_of NOP(_) = "NOP"
function clause entry_of SYNC() = "Sync"
function clause entry_of NOTIFY() = "Notify"
function clause entry_of REPEAT(_) = "Repeat"
function clause entry_of IFETCH(_) = "Ifetch"
function clause entry_of IFETCHM(_) = "IfetchM"
function clause entry_of CONFIG(_) = "Config"
function clause entry_of SETVL(_) = "SetVL"

// ICU instructions are common to every functional slice (HC34 p.27).
function clause issued_by(NOP(_), _) = true
function clause issued_by(SYNC(), _) = true
function clause issued_by(NOTIFY(), _) = true
function clause issued_by(REPEAT(_), _) = true
function clause issued_by(IFETCH(_), _) = true
function clause issued_by(IFETCHM(_), u) = u == U_MEM_W | u == U_MEM_E
function clause issued_by(CONFIG(_), _) = true
function clause issued_by(SETVL(_), _) = true

mapping clause assembly = NOP(n)   <-> "nop" ^ spc() ^ hex_bits_16(n)
mapping clause assembly = SYNC()   <-> "sync"
mapping clause assembly = NOTIFY() <-> "notify"
mapping clause assembly = REPEAT(n, d) <-> "repeat" ^ spc() ^ hex_bits_12(n) ^ sep() ^ hex_bits_4(d)
mapping clause assembly = IFETCH(st) <-> "ifetch" ^ spc() ^ stream_name(st)
mapping clause assembly = IFETCHM(a) <-> "ifetchm" ^ spc() ^ sram_text(a)
mapping clause assembly = CONFIG(v) <-> "config" ^ spc() ^ hex_bits_16(v)
mapping clause assembly = SETVL(n) <-> "setvl" ^ spc() ^ hex_bits_6(n)
