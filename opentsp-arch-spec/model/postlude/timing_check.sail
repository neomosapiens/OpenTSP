// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// The timing table (microarchitecture.md sec. 1): one row per
// catalogue entry, the entry's timing.overrides row or else its class's. config/schema.json
// bounds each value; the rules below relate rows, so the model checks them on the effective
// table. Every harness calls timing_config_error after loading the configuration and
// refuses a configuration it rejects (exit 2, like a schema violation).

// Every catalogue entry (isa-db/instructions/*.yaml "name") and its timing class.
// test/isa_db/check_isa_db.py checks that entry_of and timing_of agree with it and with
// the catalogue.
let catalogue_entries : list((string, timing_class)) = [|
  ("Config", T_CU), ("Ifetch", T_CU), ("IfetchM", T_CU), ("NOP", T_CU), ("Notify", T_CU),
  ("Repeat", T_CU), ("SetVL", T_CU), ("Sync", T_CU),
  ("Countdown", T_MEM), ("DeepSleep", T_MEM), ("Gather", T_MEM), ("Iterations", T_MEM),
  ("PowerConfig", T_MEM), ("Read", T_MEM), ("Scatter", T_MEM), ("SetSegment", T_MEM),
  ("Step", T_MEM), ("StreamStep", T_MEM), ("Write", T_MEM),
  ("Abs", T_VXM), ("Add", T_VXM), ("Add_mod", T_VXM), ("Add_sat", T_VXM), ("And", T_VXM),
  ("Cmp", T_VXM), ("Conv_F16_F32", T_VXM), ("Conv_F16_I8", T_VXM), ("Conv_F32_F16", T_VXM),
  ("Conv_F32_I8", T_VXM), ("Conv_I32_I8", T_VXM), ("Conv_I8_F16", T_VXM), ("Conv_I8_F32", T_VXM),
  ("Conv_I8_I32", T_VXM), ("Convert", T_VXM), ("Exp", T_VXM), ("Log", T_VXM), ("Mask", T_VXM),
  ("Max", T_VXM), ("Min", T_VXM), ("Mul", T_VXM), ("Mul_mod", T_VXM), ("Mul_sat", T_VXM),
  ("Neg", T_VXM), ("Not", T_VXM), ("Or", T_VXM), ("Reduce", T_VXM), ("ReLU", T_VXM),
  ("RSqrt", T_VXM), ("Select", T_VXM), ("Sub", T_VXM), ("Sub_mod", T_VXM), ("Sub_sat", T_VXM),
  ("TanH", T_VXM), ("Xor", T_VXM),
  ("ABC", T_MXM_ABC), ("ABC8", T_MXM_ABC), ("ACC", T_MXM_ACC), ("IW", T_MXM_IW), ("IWS", T_MXM_IW), ("LW", T_MXM_LW),
  ("Distribute", T_SXM), ("LoadMap", T_SXM), ("LoadMapWide", T_SXM), ("Permute", T_SXM_CROSS),
  ("Rotate", T_SXM), ("SelectS", T_SXM), ("Shift", T_SXM_CROSS), ("Transpose", T_SXM),
  ("Deskew", T_C2C), ("DeskewRecv", T_C2C), ("PciReceive", T_C2C), ("PciSend", T_C2C), ("Receive", T_C2C), ("RuntimeDeskew", T_C2C), ("Send", T_C2C), ("Transmit", T_C2C),
  ("HostNotify", T_CU), ("HostWait", T_CU),
|]

private function class_of_entry(name : string, xs : list((string, timing_class))) -> option(timing_class) =
  match xs {
    [||] => None(),
    (n, t) :: rest => if n == name then Some(t) else class_of_entry(name, rest),
  }

// The element type with code i of core/types.sail encdec_eltype (None: the code is unassigned).
private function type_of_code(i : int) -> option(eltype) = {
  let b : bits(5) = get_slice_int(5, i, 0);
  if encdec_eltype_backwards_matches(b) then Some(encdec_eltype_backwards(b)) else None()
}

// How a timing.overrides entry reads when it is not a catalogue entry: "<entry>.<type>", <entry> a catalogue
// entry and <type> an element type (core/types.sail eltype_name), as (entry, type, whether the entry takes
// the type: entry_takes_type). None for any other text.
private function typed_parts(s : string) -> option((string, string, bool)) = {
  var r : option((string, string, bool)) = None();
  var es = catalogue_entries;
  foreach (_step from 0 to (list_len(catalogue_entries) - 1)) {
    match es {
      [||] => (),
      (n, _) :: rest => {
        foreach (i from 0 to 31) {
          match type_of_code(i) {
            Some(t) => if s == n ^ "." ^ eltype_name(t) then r = Some((n, eltype_name(t), entry_takes_type(n, t))),
            None() => (),
          }
        };
        es = rest
      },
    }
  };
  r
}

private function count_override(name : string, xs : list(timing_override)) -> nat =
  match xs { [||] => 0, o :: rest => (if o.entry == name then 1 else 0) + count_override(name, rest) }

// The effective row of a catalogue entry.
function entry_row(name : string) -> latency =
  match class_of_entry(name, catalogue_entries) {
    Some(t) => entry_latency(name, t),
    None()  => internal_error(__FILE__, __LINE__, "entry_row: not a catalogue entry"),
  }

// Entries that access SRAM through the R and W ports. The port and bank rule is checked at
// dispatch (postlude/tick.sail), so their accesses must keep the dispatch spacing: one d_skew.
let sram_entries : list(string) = [| "Read", "Write", "Gather", "Scatter", "IfetchM" |]

private function first_error(a : string, b : string) -> string = if a == "" then b else a

private function timing_class_key(t : timing_class) -> string = match t {
  T_CU => "cu", T_MEM => "mem", T_VXM => "vxm", T_MXM_LW => "mxm_lw", T_MXM_IW => "mxm_iw",
  T_MXM_ABC => "mxm_abc", T_MXM_ACC => "mxm_acc", T_SXM => "sxm", T_SXM_CROSS => "sxm_cross", T_C2C => "c2c"
}

// The configuration key an entry's row comes from: its override, else its class row.
private function row_key(name : string) -> string =
  if count_override(name, timing_overrides) > 0 then "timing.overrides." ^ name
  else match class_of_entry(name, catalogue_entries) {
    Some(t) => "timing." ^ timing_class_key(t),
    None()  => "timing.overrides." ^ name,
  }

// "" when every name of `extensions` is an extension of this version (bf16, wide_acc, block_fp8) and none is listed twice.
private function extension_error(xs : list(string)) -> string =
  match xs {
    [||] => "",
    x :: rest =>
      if x != "bf16" & x != "wide_acc" & x != "block_fp8" then "extensions: " ^ x ^ " is not an extension of this version"
      else if has_extension(x, rest) then "extensions: " ^ x ^ " is listed more than once"
      else extension_error(rest),
  }

// "" when the effective table is valid; otherwise "<key>: <reason>" for the first violation,
// <key> the row's configuration key (timing.overrides.<entry> or timing.<class>).
//   - an override names a catalogue entry, at most once;
//   - every entry outside the control unit has d_skew < d_func: tick applies the commits of
//     cycle c before it executes the elements due at c, so a result must commit after its
//     execute cycle (dispatch + d_skew), and an MXM stage is ready from
//     its producer's dispatch + d_func, after the producer has executed;
//   - Shift, Permute and ABC have d_func >= geometry.superlanes (schema relations on the
//     sxm_cross and mxm_abc classes; also for an override);
//   - timing.mxm_abc.occupancy_f16 is at most ABC's d_func (schema relation on the class row);
//   - mxm.acc_bits is 32 or 128, and 128 exactly when the extension wide_acc is declared;
//   - Read, Write, Gather, Scatter and IfetchM share one d_skew.
function timing_config_error() -> string = {
  var err : string = extension_error(extensions);
  var xs = timing_overrides;
  foreach (_step from 0 to (list_len(timing_overrides) - 1)) {
    match xs {
      [||] => (),
      o :: rest => {
        if class_of_entry(o.entry, catalogue_entries) == None() then {
          match typed_parts(o.entry) {
            Some((n, tn, takes)) =>
              if not(takes) then
                err = first_error(err, "timing.overrides: " ^ o.entry ^ ": " ^ n ^ " does not take element type " ^ tn)
              else if count_override(o.entry, timing_overrides) > 1 then
                err = first_error(err, "timing.overrides: " ^ o.entry ^ " is listed more than once"),
            None() => err = first_error(err, "timing.overrides: " ^ o.entry ^ " is not a catalogue entry"),
          }
        } else if count_override(o.entry, timing_overrides) > 1 then
          err = first_error(err, "timing.overrides: " ^ o.entry ^ " is listed more than once");
        xs = rest
      },
    }
  };
  if err == "" then {
    var es = catalogue_entries;
    foreach (_step from 0 to (list_len(catalogue_entries) - 1)) {
      match es {
        [||] => (),
        (n, t) :: rest => {
          let l = entry_latency(n, t);
          if t != T_CU & not(l.d_skew < l.d_func) then
            err = first_error(err, row_key(n) ^ ": " ^ n ^ ": d_skew " ^ dec_str(l.d_skew) ^ " must be below d_func " ^ dec_str(l.d_func)
              ^ " (an instruction commits after it executes at dispatch + d_skew)");
          es = rest
        },
      }
    };
    // the rows keyed by an element type: their d_skew is below their d_func like every other row
    var ts = timing_overrides;
    foreach (_step from 0 to (list_len(timing_overrides) - 1)) {
      match ts {
        [||] => (),
        o :: rest => {
          if class_of_entry(o.entry, catalogue_entries) == None() & not(o.d_skew < o.d_func) then
            err = first_error(err, "timing.overrides." ^ o.entry ^ ": " ^ o.entry ^ ": d_skew " ^ dec_str(o.d_skew) ^ " must be below d_func "
              ^ dec_str(o.d_func) ^ " (an instruction commits after it executes at dispatch + d_skew)");
          ts = rest
        },
      }
    };
    foreach (j from 0 to 3) {
      let n : string = if j == 0 then "Shift" else if j == 1 then "Permute" else if j == 2 then "ABC" else "ABC8";
      let l = entry_row(n);
      if l.d_func < l.d_skew + superlanes then
        err = first_error(err, row_key(n) ^ ": " ^ n ^ ": d_func " ^ dec_str(l.d_func) ^ " must be at least geometry.superlanes "
          ^ dec_str(superlanes) ^ " plus d_skew " ^ dec_str(l.d_skew) ^ " (later-superlane operands must arrive before output)")
    };
    if not(fetch_profile_causal(superlanes, iq_fetch_latency, iq_fetch_rate)) then
      err = first_error(err, "icu.fetch_bytes_per_cycle: instruction text delivery precedes source tile capture");
    if timing_sxm_map_ready < superlanes then
      err = first_error(err, "timing.sxm_map_ready: must be at least geometry.superlanes (map dissemination)");
    if timing_mxm_occupancy_f16 > entry_row("ABC").d_func then
      err = first_error(err, "timing.mxm_abc.occupancy_f16: " ^ dec_str(timing_mxm_occupancy_f16) ^ " must be at most ABC's d_func " ^ dec_str(entry_row("ABC").d_func)
        ^ " (macc_busy finds the f16 entry that still holds the plane only while it has not committed)");
    if mxm_acc_bits != 32 & mxm_acc_bits != 128 then
      err = first_error(err, "mxm.acc_bits: " ^ dec_str(mxm_acc_bits) ^ " must be 32 or 128");
    if ext_wide_acc != (mxm_acc_bits == 128) then
      err = first_error(err, "extensions: wide_acc is declared exactly when mxm.acc_bits is 128");
    let s0 = entry_row("Read").d_skew;
    var ss = sram_entries;
    foreach (_step from 0 to (list_len(sram_entries) - 1)) {
      match ss {
        [||] => (),
        n :: rest => {
          if entry_row(n).d_skew != s0 then
            err = first_error(err, row_key(n) ^ ": " ^ n ^ ": d_skew " ^ dec_str(entry_row(n).d_skew) ^ " must equal Read's " ^ dec_str(s0)
              ^ " (Read, Write, Gather, Scatter and IfetchM share the SRAM ports' dispatch rule)");
          ss = rest
        },
      }
    }
  };
  // The f16 capacity of a plane pair lies between half the rows and all of them (the type-level bound of the weight storage).
  // (the integers are re-read through a bit vector: the typechecker takes the relation as given for the type-level keys and
  // would drop the test)
  let mxm_rows_raw : int = unsigned(get_slice_int(16, mxm_rows, 0));
  let mxm_f16_rows_raw : int = unsigned(get_slice_int(16, mxm_f16_rows, 0));
  if err == "" & not(2 * mxm_f16_rows_raw >= mxm_rows_raw & mxm_f16_rows_raw <= mxm_rows_raw) then
    err = "mxm.f16_rows: " ^ dec_str(mxm_f16_rows_raw) ^ " must be between mxm.rows / 2 and mxm.rows (" ^ dec_str(if mxm_rows_raw >= 0 then mxm_rows_raw / 2 else 0) ^ " and " ^ dec_str(mxm_rows_raw) ^ ")";
  err
}

// ISA queries (test/isa_db/check_isa_db.py): a word's catalogue entry and timing class name.
function isa_entry32(w : bits(32)) -> string = entry_of(encdec32(w))
function isa_entry64(w : bits(64)) -> string = entry_of(encdec64(w))
function isa_timing32(w : bits(32)) -> string = timing_class_name(timing_of(encdec32(w)))
function isa_timing64(w : bits(64)) -> string = timing_class_name(timing_of(encdec64(w)))
// The element type of a word's instruction, "" for an instruction without one (timing_type_of).
function isa_timing_type32(w : bits(32)) -> string = timing_type_of(encdec32(w))
function isa_timing_type64(w : bits(64)) -> string = timing_type_of(encdec64(w))
// Whether a timing row "<entry>.<type>" is accepted as naming a type the entry takes.
function isa_entry_takes_type(name : string, tname : string) -> bool =
  match typed_parts(name ^ "." ^ tname) { Some((_, _, takes)) => takes, None() => false }
// The class catalogue_entries gives an entry ("-": not in the table).
function isa_entry_class(name : string) -> string =
  match class_of_entry(name, catalogue_entries) { Some(t) => timing_class_name(t), None() => "-" }
