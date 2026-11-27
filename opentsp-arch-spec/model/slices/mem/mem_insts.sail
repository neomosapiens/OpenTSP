// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// MEM slice instructions (ISCA 2020 sec. III-B; HC34 p.25). Every MEM queue
// accesses its own slice (issue_unit.slice). Results go to the slice's write gap on the
// destination stream's row; operands come from its read gap on the source's row.
//
// Instruction categories: (1) address generation
// table (AGT) configuration: Countdown, Step, StreamStep, Iterations, SetSegment; (2) direct references
// Read and Write (a MEM queue issues on two dispatch paths, R and W, one per SRAM port: one
// R and one W per liquid thread per cycle, in different banks; see "Dispatch paths" below
// and postlude/tick.sail), and indirect references Gather and Scatter, which are loopable;
// (3) power management, not modelled. Iteration is performed by the cycle driver
// (postlude/tick.sail): a loopable instruction followed by Countdown/Step/StreamStep/Iterations
// re-issues with a stepped address. issue_addr_offset is that step, applied here, and
// issue_stream_offset the StreamStep added to the data stream's id.
//
// Tile gating (core/sram.sail): Read, Write, Gather and Scatter carry a
// liquid-thread tag t in bit 53. A tile executes the instruction only when it is active:
// below the queue's SetVL length and in thread t (Config SL on the queue). An inactive tile
// changes no SRAM and its superlane of the result stream is zero. Direct Read/Write still
// validate their common effective address even if every tile is inactive; Gather/Scatter
// validate only the indirect addresses of active tiles.

private function mem_unit(u : unit_kind) -> bool = u == U_MEM_W | u == U_MEM_E

// The execute and commit cycles of the current reference (reference k: from dispatch + k).
private function read_cycle() -> int = exec_cycle()
private function commit_cycle() -> nat = commit_at()

// Segment size per slice for bounds checks (SetSegment). Reset: whole slice.
register mem_segment : vector(88, nat) = vector_init(0)

function mem_reset() -> unit = mem_segment = vector_init(words_per_slice)

// The effective address of the current reference: the instruction's address plus the
// AGT offset, bounds-checked against the segment: below 0 or at or above the segment size
// it is OTS-E-OPR-001 (no wrap-around).
// Tile g of the issuing slice executes the current MEM instruction (thread issue_thread).
private function tile_on(g : int) -> bool = {
  let q = issue_queue;
  if q <= 87 then tile_active(queue_vl[q], mem_thread_split[q], true, issue_thread, g) else true
}

// The data stream of the current reference: the instruction's stream plus the AGT
// StreamStep offset, on the same row; past stream 31 or below 0 it is OTS-E-OPR-001.
private function eff_stream(Streamref(s) : streamref) -> option(streamref) = {
  let e = unsigned(s[4 .. 0]) + issue_stream_offset;
  if 0 <= e & e <= 31 then Some(Streamref([s[5]] @ get_slice_int(5, e, 0))) else None()
}

private function eff_addr(SramAddr(a) : sram_addr) -> option(sram_addr) = {
  let e = unsigned(a) + issue_addr_offset;
  if 0 <= e & e < mem_segment[issue_unit.slice] & e < 2 ^ addr_bits then Some(sram_addr_of(e)) else None()
}

// ---------------------------------------------------------------------------------------
// Read addr, dst, t / Write addr, src, t
// 64-bit layout: slice 63..60 = 1, opcode 59..54, thread 53, reserved 52..48, addr 47..32 (the 16-bit address field: the
// address in the low addr_bits bits, zero above; a profile with 13 bits uses 44..32 only),
// stream 31..26, reserved 25..0.

mapping encdec_mem_dir : mem_dir <-> bits(6) = {
  MEM_READ  <-> 0b000000,
  MEM_WRITE <-> 0b000001,
}

union clause instruction = MEM_DIRECT : (mem_dir, sram_addr, streamref, bits(1))

mapping clause encdec64 = MEM_DIRECT(op, a, Streamref(s), t)
  <-> 0x1 @ encdec_mem_dir(op) @ t @ 0b00000 @ sram_field(a) @ s @ 0b00000000000000000000000000

function clause execute MEM_DIRECT(op, a0, s0, t) = {
  issue_thread = t;
  match (eff_addr(a0), eff_stream(s0)) {
    (None(), _) => Diagnostic(OPR_RANGE),
    (_, None()) => Diagnostic(OPR_RANGE),
    (Some(a), Some(s)) => match op {
      // Inactive tiles' superlanes are zeroed by write_planes.
      MEM_READ => match sram_read(issue_unit.slice, a) {
        None()  => Diagnostic(OPR_RANGE),
        // The Read does not check the ECC: it forwards the stored check bits with the data.
        Some(v) => match write_planes_fwd(s, one_plane(v), 1, commit_cycle(), Some(sram_read_ov(issue_unit.slice, a))) {
          Ok(())  => Issued(),
          Err(d)  => Diagnostic(d),
        },
      },
      // The Write does not check the ECC either: it stores the stream's 138 bits as they are.
      MEM_WRITE => match read_plane_raw(s, read_cycle()) {
        Err(d) => Diagnostic(d),
        Ok((p, pov))  => match sram_read(issue_unit.slice, a) {
          None() => Diagnostic(OPR_RANGE),
          Some(old) => {
            var w : vec = old;
            var wov : ecc_ov = sram_read_ov(issue_unit.slice, a);
            foreach (g from 0 to (superlanes - 1)) {
              if tile_on(g) then {
                w[128 * g + 127 .. 128 * g] = slice(p, 128 * g, 128);
                wov = ecc_ov_set(wov, g, ecc_ov_get(pov, g))
              }
            };
            if sram_write_ov(exec_cycle(), issue_unit.slice, a, w, wov) then Issued() else Diagnostic(OPR_RANGE)
          },
        },
      },
    },
  }
}

function clause timing_of MEM_DIRECT(_) = T_MEM
function clause entry_of MEM_DIRECT(MEM_READ, _, _, _) = "Read"
function clause entry_of MEM_DIRECT(MEM_WRITE, _, _, _) = "Write"
function clause issued_by(MEM_DIRECT(_), u) = mem_unit(u)

mapping mem_dir_mnemonic : mem_dir <-> string = {
  MEM_READ  <-> "read",
  MEM_WRITE <-> "write",
}

mapping clause assembly = MEM_DIRECT(op, a, s, t)
  <-> mem_dir_mnemonic(op) ^ spc() ^ sram_text(a) ^ sep() ^ stream_name(s) ^ sep() ^ hex_bits_1(t)

// ---------------------------------------------------------------------------------------
// Gather dst, map, k, t / Scatter src, map, k, t: stream-indirect references (ISCA 2020
// Table I). Each tile g (superlane g) takes its address from the bottom bytes of its own
// 16-byte slice of the map stream, little-endian (lanes 16g, 16g+1; the low addr_bits bits, the bits above are ignored), and moves ONE
// element, byte k (each tile produces one element of the vector):
// Gather puts byte k of word SRAM[addr_g] (lane 16g+k) on lane 16g+k of dst, every other
// lane zero; Scatter writes lane 16g+k of src into byte k (lane 16g+k) of SRAM[addr_g], the
// rest of the word unchanged. Layout: thread 53, reserved 52, k 51..48, stream 47..42,
// map 41..36, reserved 35..0.

union clause instruction = MEM_GATHER : (streamref, streamref, bits(4), bits(1))
union clause instruction = MEM_SCATTER : (streamref, streamref, bits(4), bits(1))

mapping clause encdec64 = MEM_GATHER(Streamref(d), Streamref(m), k, t)
  <-> 0x1 @ 0b000011 @ t @ 0b0 @ k @ d @ m @ 0b000000_000000000000_000000000000000000
mapping clause encdec64 = MEM_SCATTER(Streamref(s), Streamref(m), k, t)
  <-> 0x1 @ 0b000100 @ t @ 0b0 @ k @ s @ m @ 0b000000_000000000000_000000000000000000

private function lane_byte(v : vec, l : int) -> bits(8) =
  if 0 <= l & l < vlanes then slice(v, 8 * l, 8) else 0x00

private function set_lane_byte(v : vec, l : int, b : bits(8)) -> vec =
  if 0 <= l & l < vlanes then { var w = v; w[8 * l + 7 .. 8 * l] = b; w } else v

private function tile_addr(m : vec, g : int) -> sram_addr =
  sram_addr_of(emod_int(unsigned(lane_byte(m, 16 * g + 1) @ lane_byte(m, 16 * g)), 2 ^ addr_bits))

// The error report of a Gather or Scatter (microarchitecture.md sec. 15.5): for the tiles in `bounds` (those with an address outside
// the segment or the slice), S[0] (Gather) or S[1] (Scatter) of the tile's superlane, and the auxiliary streams: S[4..16] the low 13 bits of the
// address of the tile (a profile with a wider address does not report the bits above), S[17..22] the map stream (row bit W = 1, then the 5-bit id, low bit first), S[23..28] the data stream.
// Written in the commit cycle of the instruction at its write gap, on the row of its data stream. The diagnostic still reports.
private function mem_report(prio : int, bounds : bits(superlanes), addrs : vector(superlanes, int), data : streamref, map : streamref, c : nat) -> unit = {
  let r = stream_row(data);
  match write_gap(issue_unit.unit, issue_unit.slice, r) {
    None() => (),
    Some(g) => {
      var sp : vector(30, bits(superlanes)) = vector_init(sail_zeros(superlanes));
      let ms = stream_id(map);
      let ds = stream_id(data);
      foreach (x from 0 to (superlanes - 1)) {
        if bounds[x] == bitone then {
          sp[if prio == 3 then 0 else 1][x] = bitone;
          let a : bits(13) = get_slice_int(13, addrs[x], 0);   // the low 13 bits
          foreach (i from 0 to 12) { if a[i] == bitone then sp[4 + i][x] = bitone };
          if stream_row(map) == RowW then sp[17][x] = bitone;
          foreach (i from 0 to 4) { if get_slice_int(5, ms, 0)[i] == bitone then sp[18 + i][x] = bitone };
          if r == RowW then sp[23][x] = bitone;
          foreach (i from 0 to 4) { if get_slice_int(5, ds, 0)[i] == bitone then sp[24 + i][x] = bitone }
        }
      };
      if srf_report(r, g, c, prio, bounds, sp) then lint(issue_queue, exec_cycle(), LINT_RAS_EVICT)
    },
  }
}

function clause execute MEM_GATHER(d0, m, kb, t) = {
  issue_thread = t;
  let k = unsigned(kb);
  // The map stream is an operand like any other: it is checked, and a tile whose map superlane is uncorrectable
  // makes no access (its lane is zero and poisoned, and the instruction reports OTS-E-ECC-001).
  match (eff_stream(d0), read_planes(m, 1, read_cycle())) {
    (None(), _) => Diagnostic(OPR_RANGE),
    (_, Err(e)) => Diagnostic(e),
    (Some(d), Ok(pms)) => {
      let pm : vec = pms[0];
      let map_bad : bits(superlanes) = ecc_bad;
      // DIAG-EFFECT: gather per-tile (an exception to "a diagnostic means no effect", microarchitecture.md section 8).
      // Each active tile acts alone: a tile whose effective address is outside the segment
      // or the slice drives zero on its lane, the good tiles' lanes are written, and the
      // instruction reports one OPR-001 (an exception to "a diagnostic means no effect").
      var out : vec = zero_vec;
      var bad : bool = false;
      var bounds : bits(superlanes) = sail_zeros(superlanes);
      var taddr : vector(superlanes, int) = vector_init(0);
      foreach (g from 0 to (superlanes - 1)) {
        let SramAddr(tb) = tile_addr(pm, g);
        taddr[g] = unsigned(tb);
        if tile_on(g) & map_bad[g] == bitzero then match eff_addr(tile_addr(pm, g)) {
          None() => { bad = true; bounds[g] = bitone },
          Some(a) => match sram_read(issue_unit.slice, a) {
            Some(w) => {
              // The tile checks its own superlane of the stored word before it uses the byte.
              let SramAddr(ab) = a;
              let (d2, _) = ecc_check_sl(read_cycle(), issue_queue, issue_unit.slice, unsigned(ab), g, slice(w, 128 * g, 128), ecc_ov_get(sram_read_ov(issue_unit.slice, a), g));
              out = set_lane_byte(out, 16 * g + k, slice(d2, 8 * k, 8))
            },
            None()  => { bad = true; bounds[g] = bitone },
          },
        }
      };
      let r : ExecutionResult = match write_planes(d, one_plane(out), 1, commit_cycle()) {
        Ok(()) => Issued(),
        Err(e) => Diagnostic(e),
      };
      if bad then mem_report(3, bounds, taddr, d0, m, commit_cycle());
      if bad then { ecc_after_diag = true; Diagnostic(OPR_RANGE) } else r
    },
  }
}

function clause execute MEM_SCATTER(s0, m, kb, t) = {
  issue_thread = t;
  let k = unsigned(kb);
  match eff_stream(s0) {
    None() => Diagnostic(OPR_RANGE),
    Some(s) => match read_planes(s, 1, read_cycle()) {
      Err(e) => Diagnostic(e),
      Ok(ps) => {
      // The data operand's uncorrectable superlanes are set aside while the map is read, so that the map's
      // can be told apart: a tile whose map superlane is uncorrectable writes nothing.
      let data_bad : bits(superlanes) = ecc_bad;
      ecc_bad = sail_zeros(superlanes);
      match read_planes(m, 1, read_cycle()) {
      Err(e) => Diagnostic(e),
      Ok(pms) => {
        let pm : vec = pms[0];
        let map_bad : bits(superlanes) = ecc_bad;
        ecc_bad = data_bad | map_bad;
        // DIAG-EFFECT: scatter per-tile (an exception to "a diagnostic means no effect", microarchitecture.md section 8).
        // Each active tile acts alone: the tiles with an effective address inside the
        // segment write their lane, the others write nothing, and the instruction reports one
        // OPR-001 if any tile was bad (an exception to "a diagnostic means no effect").
        // Tiles write in ascending order (each writes only its own lane of the word).
        var bad : bool = false;
        var bounds : bits(superlanes) = sail_zeros(superlanes);
        var taddr : vector(superlanes, int) = vector_init(0);
        foreach (g from 0 to (superlanes - 1)) {
          let SramAddr(tb) = tile_addr(pm, g);
          taddr[g] = unsigned(tb);
          if tile_on(g) & map_bad[g] == bitzero then match eff_addr(tile_addr(pm, g)) {
            None() => { bad = true; bounds[g] = bitone },
            Some(a) => match sram_read(issue_unit.slice, a) {
              Some(w) => {
                // The tile checks the old superlane, replaces its byte, and regenerates the superlane's check
                // bits; a superlane with an uncorrectable old word or data byte is stored poisoned.
                let SramAddr(ab) = a;
                let wov = sram_read_ov(issue_unit.slice, a);
                let (d2, unc) = ecc_check_sl(read_cycle(), issue_queue, issue_unit.slice, unsigned(ab), g, slice(w, 128 * g, 128), ecc_ov_get(wov, g));
                var nw : vec = w;
                nw[128 * g + 127 .. 128 * g] = d2;
                nw = set_lane_byte(nw, 16 * g + k, lane_byte(ps[0], 16 * g + k));
                let poisoned = unc | ecc_bad[g] == bitone;
                let _ = sram_write_ov(exec_cycle(), issue_unit.slice, a, nw, ecc_ov_set(wov, g, if poisoned then ecc_poison_sl else 0b0000000000))
              },
              None() => { bad = true; bounds[g] = bitone },
            },
          }
        };
        if bad then mem_report(2, bounds, taddr, s0, m, commit_cycle());
        if bad then { ecc_after_diag = true; Diagnostic(OPR_RANGE) } else Issued()
      },
      }
      },
    },
  }
}

function clause timing_of MEM_GATHER(_) = T_MEM
function clause timing_of MEM_SCATTER(_) = T_MEM
function clause entry_of MEM_GATHER(_) = "Gather"
function clause entry_of MEM_SCATTER(_) = "Scatter"
function clause issued_by(MEM_GATHER(_), u) = mem_unit(u)
function clause issued_by(MEM_SCATTER(_), u) = mem_unit(u)

mapping clause assembly = MEM_GATHER(d, m, k, t)
  <-> "gather" ^ spc() ^ stream_name(d) ^ sep() ^ stream_name(m) ^ sep() ^ hex_bits_4(k) ^ sep() ^ hex_bits_1(t)
mapping clause assembly = MEM_SCATTER(s, m, k, t)
  <-> "scatter" ^ spc() ^ stream_name(s) ^ sep() ^ stream_name(m) ^ sep() ^ hex_bits_4(k) ^ sep() ^ hex_bits_1(t)

// ---------------------------------------------------------------------------------------
// AGT configuration (HC34 p.25):
//   Countdown d  - idle cycles after each reference of the loop level ("Wait(countdown)"):
//                  period d + 1; d < 1 means 1 (rule (3)); default 0 when not received
//   Step a       - signed stride added to the address between references of the level;
//                  default 0
//   StreamStep s - (optional) the level's references advance the data stream id by s
//                  s = 0..31, default 0
//   Iterations n - the level makes n references (n < 1 means 1, opentsp decision);
//                  default 2 when not received
// They configure the loopable instruction (Read, Write, Gather, Scatter) dispatched before
// them on the same queue; the cycle driver applies them (postlude/tick.sail). Executed
// on their own, they have no effect. Layout: opcode, value in bits 31..16.

// Step a: |a| <= step_limit, the largest stride the signed 16-bit field holds that stays inside the address space.
let step_limit : int = if 2 ^ addr_bits - 1 < 32767 then 2 ^ addr_bits - 1 else 32767

union clause instruction = MEM_COUNTDOWN : bits(16)
union clause instruction = MEM_STEP : bits(16)
union clause instruction = MEM_ITERATIONS : bits(16)
union clause instruction = MEM_SETSEGMENT : bits(17)
union clause instruction = MEM_STREAMSTEP : bits(16)

mapping clause encdec64 = MEM_COUNTDOWN(d) <-> 0x1 @ 0b000101 @ 0x00000 @ 0b00 @ d @ 0x0000
mapping clause encdec64 = MEM_STEP(a) <-> 0x1 @ 0b000110 @ 0x00000 @ 0b00 @ a @ 0x0000
  when signed(a) >= -(step_limit) & signed(a) <= step_limit
mapping clause encdec64 = MEM_ITERATIONS(n) <-> 0x1 @ 0b000111 @ 0x00000 @ 0b00 @ n @ 0x0000
mapping clause encdec64 = MEM_STREAMSTEP(v) <-> 0x1 @ 0b001011 @ 0x00000 @ 0b00 @ v @ 0x0000
  when unsigned(v) <= 31
// SetSegment n: references at address >= n raise OTS-E-OPR-001 (n <= 2^addr_bits). The value is in bits 32..16 (17 bits, so that
// 65536 is expressible); a profile with 13 bits uses 29..16 only.
mapping clause encdec64 = MEM_SETSEGMENT(n) <-> 0x1 @ 0b001000 @ 0b000000000000000000000 @ n @ 0x0000
  when unsigned(n) <= 2 ^ addr_bits

// PowerConfig and DeepSleep are accepted power hints, with no architectural state,
// result, stream, diagnostic or timing effect and no modelled power or sleep/wake state.
// An implementation may act on them only while preserving that functional contract
// (OpenTSP decision; the cited instruction summaries do not specify these encodings).
union clause instruction = MEM_POWERCONFIG : bits(16)
union clause instruction = MEM_DEEPSLEEP : unit
mapping clause encdec64 = MEM_POWERCONFIG(v) <-> 0x1 @ 0b001001 @ 0x00000 @ 0b00 @ v @ 0x0000
mapping clause encdec64 = MEM_DEEPSLEEP() <-> 0x1 @ 0b001010 @ 0x00000 @ 0b00 @ 0x0000 @ 0x0000
function clause execute MEM_POWERCONFIG(_) = Issued()
function clause execute MEM_DEEPSLEEP() = Issued()
function clause timing_of MEM_POWERCONFIG(_) = T_MEM
function clause timing_of MEM_DEEPSLEEP() = T_MEM
function clause entry_of MEM_POWERCONFIG(_) = "PowerConfig"
function clause entry_of MEM_DEEPSLEEP() = "DeepSleep"
function clause issued_by(MEM_POWERCONFIG(_), u) = mem_unit(u)
function clause issued_by(MEM_DEEPSLEEP(), u) = mem_unit(u)
mapping clause assembly = MEM_POWERCONFIG(v) <-> "powerconfig" ^ spc() ^ hex_bits_16(v)
mapping clause assembly = MEM_DEEPSLEEP() <-> "deepsleep"

function clause execute MEM_COUNTDOWN(_) = Issued()
function clause execute MEM_STEP(_) = Issued()
function clause execute MEM_ITERATIONS(_) = Issued()
function clause execute MEM_STREAMSTEP(_) = Issued()
function clause timing_of MEM_STREAMSTEP(_) = T_MEM
function clause entry_of MEM_STREAMSTEP(_) = "StreamStep"
function clause issued_by(MEM_STREAMSTEP(_), u) = mem_unit(u)
mapping clause assembly = MEM_STREAMSTEP(v) <-> "streamstep" ^ spc() ^ hex_bits_16(v)
function clause execute MEM_SETSEGMENT(n) = {
  mem_segment[issue_unit.slice] = min(unsigned(n), words_per_slice);
  Issued()
}

function clause timing_of MEM_COUNTDOWN(_) = T_MEM
function clause timing_of MEM_STEP(_) = T_MEM
function clause timing_of MEM_ITERATIONS(_) = T_MEM
function clause timing_of MEM_SETSEGMENT(_) = T_MEM
function clause entry_of MEM_COUNTDOWN(_) = "Countdown"
function clause entry_of MEM_STEP(_) = "Step"
function clause entry_of MEM_ITERATIONS(_) = "Iterations"
function clause entry_of MEM_SETSEGMENT(_) = "SetSegment"
function clause issued_by(MEM_COUNTDOWN(_), u) = mem_unit(u)
function clause issued_by(MEM_STEP(_), u) = mem_unit(u)
function clause issued_by(MEM_ITERATIONS(_), u) = mem_unit(u)
function clause issued_by(MEM_SETSEGMENT(_), u) = mem_unit(u)

mapping clause assembly = MEM_COUNTDOWN(d) <-> "countdown" ^ spc() ^ hex_bits_16(d)
mapping clause assembly = MEM_STEP(a) <-> "step" ^ spc() ^ hex_bits_16(a)
mapping clause assembly = MEM_ITERATIONS(n) <-> "iterations" ^ spc() ^ hex_bits_16(n)
mapping clause assembly = MEM_SETSEGMENT(n) <-> "setsegment" ^ spc() ^ hex_bits_17(n)

// Which instructions the AGT can loop: Read, Write, Gather and Scatter.
function is_loopable(i : instruction) -> bool =
  match i {
    MEM_DIRECT(_)  => true,
    MEM_GATHER(_)  => true,
    MEM_SCATTER(_) => true,
    _              => false,
  }

// The AGT field an instruction configures, if any.
enum agt_field = {AGT_COUNTDOWN, AGT_STEP, AGT_STREAMSTEP, AGT_ITERATIONS}

// ---------------------------------------------------------------------------------------
// Dispatch paths (postlude/tick.sail). Each MEM slice has two dispatch paths, one per
// port of the pseudo-dual-port SRAM (an opentsp model): the R path issues the R-type references Read and Gather, the W path the W-type
// references Write and Scatter. Each path has its own busy state and its own AGT.
//
// Configuration records are routed to a path (opentsp decision; postlude/tick.sail
// cfg_path): to the one path whose AGT is configuring; when both are (a Read/Write pair),
// to the one that has received fewer configuration records so far, the R path on a tie.
// So a pair's records alternate R, W, R, W, ...: a Read and a Write pair with step and
// iterations records is written read + write at 10, step at 11 (R), step at 12 (W),
// iterations at 13 (R) and iterations at 14 (W). The encodings need no path field.

function agt_config(i : instruction) -> option((agt_field, bits(16))) =
  match i {
    MEM_COUNTDOWN(d)  => Some((AGT_COUNTDOWN, d)),
    MEM_STEP(a)       => Some((AGT_STEP, a)),
    MEM_STREAMSTEP(v) => Some((AGT_STREAMSTEP, v)),
    MEM_ITERATIONS(n) => Some((AGT_ITERATIONS, n)),
    _                 => None(),
  }

// The dispatch path of a reference: 0 (R path: Read, Gather) or 1 (W path: Write, Scatter).
type mem_path = range(0, 1)

function ref_path(i : instruction) -> option(mem_path) =
  match i {
    MEM_DIRECT(MEM_READ, _, _, _)  => Some(0),
    MEM_GATHER(_)                  => Some(0),
    MEM_DIRECT(MEM_WRITE, _, _, _) => Some(1),
    MEM_SCATTER(_)                 => Some(1),
    _                              => None(),
  }

// The liquid-thread tag of a reference (0 for anything else).
function ref_thread(i : instruction) -> bits(1) =
  match i {
    MEM_DIRECT(_, _, _, t) => t,
    MEM_GATHER(_, _, _, t)  => t,
    MEM_SCATTER(_, _, _, t) => t,
    _                       => 0b0,
  }

// The SRAM bank an access uses, as the dispatch sees it: none (an address outside the
// slice accesses no bank; its execution is OTS-E-OPR-001), one bank, or
// any bank (an indirect reference: its addresses come from a stream).
union bank_use = { BankNone : unit, BankOf : bits(1), BankAny : unit }

function addr_bank(e : int) -> bank_use =
  if 0 <= e & e < words_per_slice then BankOf(sram_bank(sram_addr_of(e))) else BankNone()

// Two accesses meet in a bank: never when one uses no bank, always when one may use any.
function banks_meet(a : bank_use, b : bank_use) -> bool =
  match (a, b) {
    (BankNone(), _)        => false,
    (_, BankNone())        => false,
    (BankOf(x), BankOf(y)) => x == y,
    _                      => true,
  }

// The bank of a direct reference at address offset off; any bank for an indirect one.
function ref_bank(i : instruction, off : int) -> bank_use =
  match i {
    MEM_DIRECT(_, SramAddr(a), _, _) => addr_bank(unsigned(a) + off),
    _ => BankAny(),
  }
