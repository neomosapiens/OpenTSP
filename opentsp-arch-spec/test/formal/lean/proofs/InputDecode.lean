-- Independent modular bitfield foundation for the actual input decoder.
import Proofs.PublicNumerics
open Sail Out Out.Functions
set_option autoImplicit false
namespace InputDecodeProofs

theorem actual_field_modular (x : BitVec 32) (lsb n : Int) (hl : 0 ≤ lsb) (hn : 0 ≤ n) :
    field x lsb n=(BitVec.toNatInt x/pow2i lsb)%pow2i n := by
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hpow := NumericalProofs.pow2i_add_nonnegative lsb n hl hn
  have hA := IntervalProofs.pow2i_positive lsb hl
  have hB := IntervalProofs.pow2i_positive n hn
  have hq := Int.ediv_ediv_of_nonneg (x:=BitVec.toNatInt x) (y:=pow2i lsb) (z:=pow2i n) (by omega : 0 ≤ pow2i lsb)
  simp only [field,Int.tdiv_eq_ediv_of_nonneg hx]
  rw [hpow,Int.emod_def,hq]
  rw [Int.mul_comm (pow2i n)]

end InputDecodeProofs
