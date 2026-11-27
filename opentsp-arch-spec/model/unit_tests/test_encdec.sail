// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Encoding tests against the bit positions in opentsp-arch-spec/docs/encoding.md, run on
// the model's own encdec mappings.

$[test]
function test_mem_read_layout() -> unit = {
  let w = encdec64(MEM_DIRECT(MEM_READ, sram_addr_of(4099), Streamref(0b100101), 0b0));
  assert(w[63 .. 60] == 0x1);
  assert(w[59 .. 54] == 0b000000);
  assert(w[44 .. 32] == 0b1_0000_0000_0011);
  assert(w[31 .. 26] == 0b100101);
  assert(w[53] == bitzero & w[52 .. 45] == 0b00000000 & w[25 .. 0] == sail_zeros(26));
}

$[test]
function test_mem_roundtrip_and_reserved() -> unit = {
  let i = MEM_DIRECT(MEM_WRITE, sram_addr_of(8191), Streamref(0b111111), 0b0);
  let w = encdec64(i);
  assert(encdec64(w) == i);
  // A reserved bit set does not decode to a MEM instruction.
  match encdec64(w | 0x0000_0000_0000_0001) {
    ILLEGAL64(_) => (),
    _            => assert(false, "reserved bit accepted"),
  }
}

$[test]
function test_cu_nop_zero_is_illegal() -> unit = {
  match encdec32(0x0000_0000) {
    ILLEGAL32(_) => (),
    _            => assert(false, "NOP 0 accepted"),
  };
  assert(encdec32(0x0000_0005) == NOP(0x0005))
}

// Illegal types and modes do not decode.
private function is_illegal64(w : bits(64)) -> bool = match encdec64(w) { ILLEGAL64(_) => true, _ => false }

$[test]
function test_vxm_illegal_type_and_mode() -> unit = {
  // Add.i8 E:8 <- E:0, E:4 decodes.
  let ok = encdec64(VXM_BIN(VADD, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  assert(not(is_illegal64(ok)));
  // Add with a non-zero mode field does not decode.
  assert(is_illegal64(ok | 0x0000_0000_0300_0000));
  // Add_sat.f16 does not decode (saturating ops are integer-only).
  assert(is_illegal64(0x2 @ 0b000001 @ 0b01 @ 0b001000 @ 0b000000 @ 0b000100 @ 0b000000 @ 0x0 @ 0x000000));
  // Cmp mode 6..15 does not decode; mode 5 does.
  let cmp = encdec64(VXM_CMP(CMP_GE, I32, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  assert(not(is_illegal64(cmp)));
  assert(is_illegal64((cmp & ~(0x0000_0000_0F00_0000)) | 0x0000_0000_0600_0000));
  // Reduce op 3 does not decode.
  let red = encdec64(VXM_REDUCE(RED_MIN, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)));
  assert(is_illegal64((red & ~(0x0000_0000_0F00_0000)) | 0x0000_0000_0300_0000));
  // A conversion with the wrong source type does not decode.
  let cv = encdec64(VXM_CONV(CONV_I8_F32, I8, Streamref(0b000000), Vsrc(Streamref(0b000100), 0b0)));
  assert(is_illegal64(cv | 0x0010_0000_0000_0000));
}
