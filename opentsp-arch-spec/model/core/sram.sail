// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// SRAM: each of the 88 MEM slices holds words_per_slice words of one stream value each
// (ISCA 2020 sec. II-B). Storage is external, as sail-riscv keeps RAM outside the model
// (core/phys_mem_interface.sail): the C++ harness allocates it lazily.

val read_sram_word = impure {c: "read_sram_word", cpp: "read_sram_word"} : (mem_slice, nat) -> vec
val raw_write_sram_word = impure {c: "write_sram_word", cpp: "write_sram_word"} : (mem_slice, nat, vec) -> unit

// The ECC overlay of a stored word (core/ecc.sail): zero for a word whose check bits are valid.
// The harness keeps it beside the data (a word never faulted has none).
val read_sram_ecc = impure {c: "read_sram_ecc", cpp: "read_sram_ecc"} : (mem_slice, nat) -> ecc_ov
val raw_write_sram_ecc = impure {c: "write_sram_ecc", cpp: "write_sram_ecc"} : (mem_slice, nat, ecc_ov) -> unit

// A write notification invalidates a patrol scrubber's captured correction.
// This models write snooping, not an additional SRAM read port. Conservatively
// invalidate on any mutation of the slice, including integrity-only fault injection.
register sram_revision : vector(88, nat) = vector_init(0)
function write_sram_word(s : mem_slice, a : nat, v : vec) -> unit = {
  sram_revision[s] = sram_revision[s] + 1;
  raw_write_sram_word(s, a, v)
}
function write_sram_ecc(s : mem_slice, a : nat, v : ecc_ov) -> unit = {
  sram_revision[s] = sram_revision[s] + 1;
  raw_write_sram_ecc(s, a, v)
}

function sram_index(SramAddr(a) : sram_addr) -> option(nat) = {
  let i = unsigned(a);
  if i < words_per_slice then Some(i) else None()
}

// Bank of an address: bit bank_bit (ISCA 2020 sec. III-B: the bank bit is exposed).
function sram_bank(SramAddr(a) : sram_addr) -> bits(1) = [a[bank_bit]]

function sram_read(s : mem_slice, a : sram_addr) -> option(vec) =
  match sram_index(a) {
    Some(i) => Some(read_sram_word(s, i)),
    None()  => None(),
  }

// The overlay of the word at a (zero for an address outside the slice).
function sram_read_ov(s : mem_slice, a : sram_addr) -> ecc_ov =
  match sram_index(a) {
    Some(i) => read_sram_ecc(s, i),
    None()  => ecc_ov_zero,
  }

// Store a word and its overlay (the stream value's 138 bits as they are).
function sram_write_ov(c : cycle, s : mem_slice, SramAddr(a) : sram_addr, v : vec, ov : ecc_ov) -> bool =
  match sram_index(SramAddr(a)) {
    Some(i) => { write_sram_word(s, i, v); write_sram_ecc(s, i, ov); sram_write_callback(c, s, get_slice_int(16, unsigned(a), 0), v); true },
    None()  => false,
  }

// A word with generated (valid) check bits.
function sram_write(c : cycle, s : mem_slice, a : sram_addr, v : vec) -> bool = sram_write_ov(c, s, a, v, ecc_ov_zero)

// ---------------------------------------------------------------------------------------
// Tile-level output/storage masking (OpenTSP instruction and masking choices; ISCA 2020
// sec. II-F motivates activity control, not these encoding/masking choices).
//
// SetVL n on a data queue masks superlanes >= n of that queue's slice: masked MEM tiles
// leave SRAM unchanged and masked result superlanes are zero. This is not a promise to
// suppress input integrity checks or whole-vector unit state under their owning rules,
// and it is not a physical power-gating model. The vector length is per functional slice;
// the model keeps it per issuing queue.
// Liquid threads (MEM): Config(SL) on a MEM queue sets tiles 0..SL to thread 0 and
// SL+1..superlanes-1 to thread 1; a MEM instruction tagged with thread t writes only the tiles
// of thread t, so a vector shorter than VL does not clobber the other thread's words.
// The thread tag of a MEM instruction is bit 53 of its word (an opentsp encoding;
// the published papers give none). Reset: VL = all
// superlanes, every tile thread 0, thread tag 0, so programs that use neither feature
// behave exactly as before.

register queue_vl : vector(144, range(1, superlanes)) = vector_init(superlanes)
register mem_thread_split : vector(88, range(0, superlanes)) = vector_init(superlanes)   // tiles < split are thread 0

function masks_reset() -> unit = {
  queue_vl = vector_init(superlanes);
  mem_thread_split = vector_init(superlanes)
}

// Tile g is written when g < vl and (thread mask) g belongs to thread t.
function tile_active(vl : int, split : int, use_thread : bool, t : bits(1), g : int) -> bool =
  g < vl & (not(use_thread) | (if t == 0b0 then g < split else g >= split))

// Merge new into old, keeping old's bytes in inactive tiles.
function merge_tiles(old : vec, new : vec, vl : int, split : int, use_thread : bool, t : bits(1)) -> vec = {
  var out : vec = old;
  foreach (g from 0 to (superlanes - 1)) {
    if tile_active(vl, split, use_thread, t, g) then out[128 * g + 127 .. 128 * g] = slice(new, 128 * g, 128)
  };
  out
}
