// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// The instruction queues (IQ) and the fetch path into them (opentsp-arch-spec/docs/
// assembly.md sec. 7.1, 7.4; ISCA 2020 sec. III-A3: Ifetch fills the IQ with 640
// bytes; HC34 p.27). Every queue (the 144 data queues and the control channel, index 144)
// has an IQ of 32-bit text units, capacity icu.iq_depth bytes, and a fill FIFO of fetched
// payloads that are still on their way.
//
// Storage of the IQ units is external, like the SRAM (the harness keeps one ring of
// iq_depth / 4 units per queue); the head, the length, the overflow rule and everything else
// are in this model. The initial IQ image of each queue (the host's program load
// via the I/O path) is read from the harness at reset.

let iq_depth : range(64, 4096) = config icu.iq_depth
let iq_fetch_latency : range(1, 64) = config icu.fetch_latency
let iq_fetch_rate : range(4, 64) = config icu.fetch_bytes_per_cycle
let iq_staging_depth : range(1, 64) = config icu.fetch_staging_depth

// Capacity and fill rate in 32-bit units.
let iq_cap : nat = iq_depth / 4
// Compare equivalent superlane-zero times. The first unit of a tile is the
// earliest delivered from its sixteen bytes. Each beat has its own tile stagger.
function fetch_tile_available(sl : nat, latency : nat, rate : range(4, 64),
                              beat : range(0, 1), tile : nat) -> bool =
  1 + latency + (16 * (beat * sl + tile)) / rate > beat + tile

function fetch_profile_causal(sl : range(1, 32), latency : nat, rate : range(4, 64)) -> bool =
  rate <= 16 | (fetch_tile_available(sl, latency, rate, 0, sl - 1) &
                fetch_tile_available(sl, latency, rate, 1, sl - 1))

let iq_rate : nat = iq_fetch_rate / 4

// One unit of the ring of queue q (0..144) at slot 0 .. iq_cap - 1.
val iq_unit_read = impure {c: "iq_unit_read", cpp: "iq_unit_read"} : (range(0, 144), nat) -> bits(32)
val iq_unit_write = impure {c: "iq_unit_write", cpp: "iq_unit_write"} : (range(0, 144), nat, bits(32)) -> unit

// The text-word tag of a slot of the ring: the production index of the unit (counted from 0 over the queue's text in
// production order: the initial image, then the units of each fetched payload; a padding unit of the image keeps its
// index), or iq_no_word for a unit that belongs to no protected text word (host-injected text). Four consecutive indices
// are one text word (16 bytes = one superlane of a vector).
val iq_tag_read = impure {c: "iq_tag_read", cpp: "iq_tag_read"} : (range(0, 144), nat) -> int
val iq_tag_write = impure {c: "iq_tag_write", cpp: "iq_tag_write"} : (range(0, 144), nat, int) -> unit
let iq_no_word : int = -1

// The initial IQ image of queue q: its length in units (padding already dropped, at most
// iq_cap, checked by the harness) and unit i.
val iq_image_len = impure {c: "iq_image_len", cpp: "iq_image_len"} : (range(0, 144)) -> range(0, 1024)
val iq_image_unit = impure {c: "iq_image_unit", cpp: "iq_image_unit"} : (range(0, 144), nat) -> bits(32)

// The padding unit (all ones): never enqueued.
let iq_padding : bits(32) = 0xFFFFFFFF

register iq_head : vector(145, nat) = vector_init(0)          // ring slot of the oldest unit
register iq_len : vector(145, nat) = vector_init(0)           // units held
register iq_last : vector(145, int) = vector_init(-1)         // cycle of the queue's last dispatch (-1: none)
register iq_dry : vector(145, bool) = vector_init(false)      // lint OTS-W-IQ-001 pending for the head parcel

// ECC on instruction text (microarchitecture.md section 16): the overlay (10 bits, as for a superlane word) of each text
// word that has one, by word index; a word that is not in the list has a valid code. iq_ecc_live: some overlay was ever
// set since reset, so a run without a text fault pays one test per unit read.
register iq_prod : vector(145, nat) = vector_init(0)
struct ov_ent = { ov_k : nat, ov_v : bits(10) }
register iq_wov : vector(145, list(ov_ent)) = vector_init([||])
register iq_ecc_live : bool = false

register iq_active : vector(145, nat) = vector_init(0)       // last cycle in which a unit entered the queue's IQ or the queue dispatched (0: the initial image)

// Fetched payloads in the order their fetches executed; the units of a payload are taken
// at icu.fetch_bytes_per_cycle from cycle ready = (execute cycle) + icu.fetch_latency. A
// payload leaves the FIFO when its last unit has entered the IQ, and the FIFO holds at most
// icu.fetch_staging_depth payloads: a fetch that finds it full fetches nothing (iq_fetch).
// base: the production index of the payload's first unit (-1: host-injected text, no word); taken: its units taken so
// far; ovw: the overlay of each of its text words.
struct fill_seg = { ready : nat, units : list(bits(32)), base : int, taken : nat, ovw : list(bits(10)) }
register fill_fifo : vector(145, list(fill_seg)) = vector_init([||])

// Ring slot x of the queue, for x below twice the capacity.
function ring(x : nat) -> nat = if x >= iq_cap then x - iq_cap else x

// The i-th unit from the head (i < iq_len <= iq_cap).
function iq_peek(q : range(0, 144), i : nat) -> bits(32) = iq_unit_read(q, ring(iq_head[q] + i))

function iq_pop(q : range(0, 144), n : nat) -> unit = {
  iq_head[q] = ring(iq_head[q] + n);
  iq_len[q] = max(iq_len[q] - n, 0)
}

// Append one unit; false (the unit is lost) when the IQ already holds iq_depth bytes.
function iq_push_tagged(q : range(0, 144), u : bits(32), tag : int) -> bool =
  if iq_len[q] >= iq_cap then false
  else {
    iq_unit_write(q, ring(iq_head[q] + iq_len[q]), u);
    iq_tag_write(q, ring(iq_head[q] + iq_len[q]), tag);
    iq_len[q] = iq_len[q] + 1;
    true
  }

function iq_push(q : range(0, 144), u : bits(32)) -> bool = iq_push_tagged(q, u, iq_no_word)

// The overlay of text word w of queue q.
function iq_word_ov(q : range(0, 144), w : nat) -> bits(10) = {
  var e : bits(10) = 0b0000000000;
  var xs = iq_wov[q];
  var go : bool = true;
  foreach (_step from 0 to (list_len(iq_wov[q]) - 1)) {
    if go then match xs {
      [||] => go = false,
      ent :: rest => if ent.ov_k == w then { e = ent.ov_v; go = false } else xs = rest,
    }
  };
  e
}

// Set the overlay of text word w (an overlay of zero removes the entry).
function iq_word_ov_set(q : range(0, 144), w : nat, e : bits(10)) -> unit = {
  var out : list(ov_ent) = [||];
  var xs = iq_wov[q];
  foreach (_step from 0 to (list_len(iq_wov[q]) - 1)) {
    match xs {
      [||] => (),
      ent :: rest => { if ent.ov_k != w then out = ent :: out; xs = rest },
    }
  };
  if e != 0b0000000000 then { out = struct { ov_k = w, ov_v = e } :: out; iq_ecc_live = true };
  iq_wov[q] = out
}

private function ov_nth(xs : list(bits(10)), n : nat) -> bits(10) =
  match xs { [||] => 0b0000000000, x :: rest => if n == 0 then x else ov_nth(rest, max(n - 1, 0)) }

// The fill rate in cycle c: up to iq_rate units from the head of the FIFO, from payloads
// that are ready. Padding is dropped (it still takes its place in the rate). Returns
// (a unit was enqueued, a unit was lost to a full IQ).
function iq_deliver(q : range(0, 144), c : nat) -> (bool, bool) = {
  var budget : nat = iq_rate;
  var pushed : bool = false;
  var lost : bool = false;
  var go : bool = true;
  // Every step takes one unit off the budget, drops one emptied payload or ends the fill; the last step is the one that ends it.
  foreach (_step from 0 to (iq_rate + list_len(fill_fifo[q]))) {
    if go & budget > 0 then match fill_fifo[q] {
      [||] => go = false,
      seg :: rest =>
        if seg.ready > c then go = false
        else match seg.units {
          [||] => fill_fifo[q] = rest,
          u :: us => {
            fill_fifo[q] = { seg with units = us, taken = seg.taken + 1 } :: rest;
            budget = max(budget - 1, 0);
            if u != iq_padding then {
              let tag : int = if seg.base >= 0 then seg.base + seg.taken else iq_no_word;
              if iq_push_tagged(q, u, tag) then {
                pushed = true;
                iq_active[q] = c;
                if seg.base >= 0 then {
                  let e = ov_nth(seg.ovw, ediv_int(seg.taken, 4));
                  if e != 0b0000000000 then iq_word_ov_set(q, ediv_int(max(seg.base + seg.taken, 0), 4), e)
                }
              } else lost = true
            }
          },
        }
    }
  };
  // Drop an emptied head payload.
  match fill_fifo[q] {
    seg :: rest => match seg.units { [||] => fill_fifo[q] = rest, _ => () },
    [||] => (),
  };
  (pushed, lost)
}

private function fill_append(xs : list(fill_seg), s : fill_seg) -> list(fill_seg) =
  match xs { [||] => [|s|], x :: rest => x :: fill_append(rest, s) }

// The 32-bit little-endian units of a vector, unit 0 (lanes 0..3) first, prefixed to rest.
private function vec_units(v : vec, rest : list(bits(32))) -> list(bits(32)) = {
  var xs = rest;
  foreach (k from (vlanes / 4 - 1) downto 0) { xs = slice(v, 32 * k, 32) :: xs };
  xs
}

function fill_len(xs : list(fill_seg)) -> nat = match xs { [||] => 0, _ :: rest => 1 + fill_len(rest) }

// A fetch whose payload is complete at cycle t on queue q (an Ifetch: its second vector is read at
// t = execute cycle + 1; an IfetchM: its second SRAM read at execute cycle + 1): payload a (lane 0 first) then b. False, with no
// effect, when the queue's staging FIFO already holds icu.fetch_staging_depth payloads (the
// published IQ has no flow control for fetches: the payload would have nowhere to wait).
// ova and ovb are the ECC overlays of the two vectors: each superlane of a vector is one text word.
function iq_fetch_ov(q : range(0, 144), t : nat, a : vec, b : vec, ova : ecc_ov, ovb : ecc_ov) -> bool =
  if fill_len(fill_fifo[q]) >= iq_staging_depth then false
  else {
    let payload = vec_units(a, vec_units(b, [||]));
    var ws : list(bits(10)) = [||];
    foreach (g from (superlanes - 1) downto 0) { ws = ecc_ov_get(ovb, g) :: ws };
    foreach (g from (superlanes - 1) downto 0) { ws = ecc_ov_get(ova, g) :: ws };
    let base = iq_prod[q];
    observe_site("fetch", t, q, 0, 8 * superlanes, base);
    iq_prod[q] = base + 8 * superlanes;
    fill_fifo[q] = fill_append(fill_fifo[q], struct { ready = t + iq_fetch_latency, units = payload, base = base, taken = 0, ovw = ws });
    true
  }

function iq_fetch(q : range(0, 144), t : nat, a : vec, b : vec) -> bool = iq_fetch_ov(q, t, a, b, ecc_ov_zero, ecc_ov_zero)

// Reset the queues and load the initial images.
function iq_reset() -> unit = {
  iq_head = vector_init(0);
  iq_len = vector_init(0);
  iq_last = vector_init(-1);
  iq_dry = vector_init(false);
  iq_active = vector_init(0);
  fill_fifo = vector_init([||]);
  iq_prod = vector_init(0);
  iq_wov = vector_init([||]);
  iq_ecc_live = false;
  foreach (q from 0 to 144) {
    let n = iq_image_len(q);
    foreach (i from 0 to (n - 1)) {
      let u = iq_image_unit(q, i);
      if u != iq_padding then { let _ = iq_push_tagged(q, u, i); () }
    };
    iq_prod[q] = 4 * ediv_int(n + 3, 4)
  }
}
