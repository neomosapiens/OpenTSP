// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Basic TSP types. Read this file early: every slice uses these definitions.

// One stream value: one byte per lane (ISCA 2020 sec. II: 16 lanes per superlane).
type vec = bits(8 * vlanes)

// Cycle numbers. Object records carry u32 cycles.
type cycle = nat

// The two stream rows. Each row numbers its 45 stream registers 0..44 in flow order.
enum row = {RowE, RowW}

// A stream reference as encoded: bit 5 selects the row (0 = E, 1 = W), bits 4..0 the stream.
newtype streamref = Streamref : bits(6)

function stream_row(Streamref(s) : streamref) -> row = if s[5] == bitzero then RowE else RowW
function stream_id(Streamref(s) : streamref) -> range(0, 31) = unsigned(s[4 .. 0])

mapping stream_name : streamref <-> string = {
  Streamref(0b0 @ i) <-> "E:" ^ dec_bits_5(i),
  Streamref(0b1 @ i) <-> "W:" ^ dec_bits_5(i),
}

// A unit's position on a row: it reads register g and writes register g + 1 ("gap g").
type gap = range(0, 43)

// SRAM addresses are addr_bits (13 to 16, core/geometry.sail) bits per MEM slice (ISCA 2020 sec. II item 5: 13 in the reference profile).
newtype sram_addr = SramAddr : bits(addr_bits)

function sram_addr_of(n : int) -> sram_addr = SramAddr(get_slice_int(addr_bits, n, 0))

// The 16-bit address field of an encoding: the address in the low addr_bits bits, zero above. A field that holds a value of addr_bits + 1
// bits or more does not decode on the profile.
val sram_field : sram_addr <-> bits(16)
function sram_field_forwards(SramAddr(a)) = get_slice_int(16, unsigned(a), 0)
function sram_field_forwards_matches(_) = true
function sram_field_backwards(f) = sram_addr_of(unsigned(f))
function sram_field_backwards_matches(f) = unsigned(f) < 2 ^ addr_bits

// The assembly text of an address: hexadecimal, at most addr_bits bits.
val sram_text : sram_addr <-> string
function sram_text_forwards(SramAddr(a)) = hex_str(unsigned(a))
function sram_text_forwards_matches(_) = true
function sram_text_backwards(s) = SramAddr(parse_hex_bits(addr_bits, s))
function sram_text_backwards_matches(s) = valid_hex_bits(addr_bits, s)

// MEM slices 0..87: 0..43 west hemisphere, 44..87 east; slice 0 / 44 nearest the VXM.
type mem_slice = range(0, 87)

// Data queues 0..143 (the order of core/queues.sail); the CU control channel is encoded as 0xFFFF.
type queue_id = range(0, 143)

// Element types (HC34 p.26: UINT8, INT8, BOOL8, UINT16, INT16, BOOL16, FLOAT16, UINT32,
// INT32, BOOL32, FLOAT32) and the two FP8 formats (an implementation choice: microarchitecture.md
// sec. 5, FP8). An element of width w occupies byte planes in w consecutive
// streams, aligned to w. The type code is bit 4 @ ext @ type (5 bits): the original 2-bit field keeps
// i8/f16/i32/f32 at 0..3, and a 2-bit extension in a reserved field selects the rest. A fifth bit (code bit 4, a
// reserved bit of the VXM word) selects the profile extension types: bf16 is code 16 and exists only in a profile that
// declares the extension `bf16` (`extensions`).
enum eltype = {I8, F16, I32, F32, U8, I16, U16, U32, B8, B16, B32, E4M3, E5M2, BF16}

mapping encdec_eltype : eltype <-> bits(5) = {
  I8  <-> 0b00000,
  F16 <-> 0b00001,
  I32 <-> 0b00010,
  F32 <-> 0b00011,
  U8  <-> 0b00100,
  I16 <-> 0b00101,
  U16 <-> 0b00110,
  U32 <-> 0b00111,
  B8  <-> 0b01000,
  B16 <-> 0b01001,
  B32 <-> 0b01010,
  E4M3 <-> 0b01011,
  E5M2 <-> 0b01100,
  BF16 <-> 0b10000,
}

mapping eltype_name : eltype <-> string = {
  I8  <-> "i8",
  F16 <-> "f16",
  I32 <-> "i32",
  F32 <-> "f32",
  U8  <-> "u8",
  I16 <-> "i16",
  U16 <-> "u16",
  U32 <-> "u32",
  B8  <-> "b8",
  B16 <-> "b16",
  B32 <-> "b32",
  E4M3 <-> "e4m3",
  E5M2 <-> "e5m2",
  BF16 <-> "bf16",
}

function eltype_width(t : eltype) -> {1, 2, 4} =
  match t {
    I8  => 1, U8  => 1, B8  => 1, E4M3 => 1, E5M2 => 1,
    F16 => 2, I16 => 2, U16 => 2, B16 => 2, BF16 => 2,
    I32 => 4, F32 => 4, U32 => 4, B32 => 4,
  }

// Profile extensions (`extensions` in the profile: a list of names, empty in `ref` and `mini`). A word that uses an
// extension the profile does not declare does not decode (OTS-E-DEC-003), as a word with an unassigned type code.
let extensions : list(string) = config extensions

function has_extension(name : string, xs : list(string)) -> bool =
  match xs { [||] => false, x :: rest => x == name | has_extension(name, rest) }

let ext_wide_acc : bool = has_extension("wide_acc", extensions)
let ext_block_fp8 : bool = has_extension("block_fp8", extensions)
let ext_bf16 : bool = has_extension("bf16", extensions)

function is_unsigned(t : eltype) -> bool = t == U8 | t == U16 | t == U32
function is_bool(t : eltype) -> bool = t == B8 | t == B16 | t == B32
