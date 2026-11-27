-- Admitted SRAM geometry and exact address/index transfer, not fixed-8192 assumptions.
import Proofs.PatrolOwnedCycle
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
namespace ResourceGeometryProofs

def indexFor {bits : Nat} (words : Nat) (a : BitVec bits) : Option Nat :=
  if a.toNat<words then some a.toNat else none

theorem actual_index_source (a : BitVec Out.Functions.addr_bits) :
    sram_index (.SramAddr a)=indexFor Out.Functions.words_per_slice a := by
  have ha := a.isLt
  simp only [Out.Functions.addr_bits] at ha
  simp [sram_index,indexFor,BitVec.toNatInt,Out.Functions.addr_bits,Out.Functions.words_per_slice,BitVec.setWidth_eq] <;> first | omega | (split <;> split <;> simp_all <;> omega)

theorem bank_admission_capacity (bits bank words : Nat) (hw : words=2^(bank+1))
    (hbank : bank < bits) : words≤2^bits := by
  rw [hw]
  exact Nat.pow_le_pow_right (by decide : 0<2) (by omega)

theorem admitted_max_words (bits words : Nat) (hbits : bits≤16) (hw : words≤2^bits) :
    words≤65536 := by
  have hp : 2^bits≤2^16 := Nat.pow_le_pow_right (by decide) hbits
  omega

theorem exact_valid_address (bits words i : Nat) (hcap : words≤2^bits) (hi : i<words) :
    indexFor words (BitVec.ofNat bits i)=some i := by
  have hb : i<2^bits := Nat.lt_of_lt_of_le hi hcap
  simp [indexFor,BitVec.toNat_ofNat,Nat.mod_eq_of_lt hb,hi]

theorem rejected_address {bits : Nat} (words : Nat) (a : BitVec bits) (hi : words≤a.toNat) :
    indexFor words a=none := by simp [indexFor]; omega

theorem flat_index_bound (words slice addr : Nat) (hw : 0<words) (hs : slice<88) (ha : addr<words) :
    slice*words+addr<88*words := by
  have hm := Nat.mul_le_mul_right words (show slice+1≤88 by omega)
  simp only [Nat.add_mul,Nat.one_mul] at hm
  omega

theorem flat_index_injective (words s t a b : Nat) (hw : 0<words) (ha : a<words) (hb : b<words)
    (he : s*words+a=t*words+b) : s=t ∧ a=b := by
  have hmod := congrArg (fun n => n%words) he
  have hdiv := congrArg (fun n => n/words) he
  simp [Nat.add_mod,Nat.mod_eq_of_lt ha,Nat.mod_eq_of_lt hb] at hmod
  have hd1 : (s*words+a)/words=s := by rw [Nat.mul_comm _ words,Nat.mul_add_div hw]; simp [Nat.div_eq_of_lt ha]
  have hd2 : (t*words+b)/words=t := by rw [Nat.mul_comm _ words,Nat.mul_add_div hw]; simp [Nat.div_eq_of_lt hb]
  exact ⟨by simpa [hd1,hd2] using hdiv,hmod⟩

theorem bounded_C_index (words slice addr : Nat) (hw : 0<words) (hmax : words≤65536)
    (hs : slice<88) (ha : addr<words) : slice*words+addr<5767168 := by
  have hb := flat_index_bound words slice addr hw hs ha
  omega

theorem exact_second_address (bits words seg a : Nat) (hcap : words≤2^bits)
    (ha : a+1<words) (hseg : a+1<seg) :
    indexFor words (BitVec.ofNat bits (a+1))=some (a+1) ∧ a+1<seg := by
  exact ⟨exact_valid_address bits words (a+1) hcap ha,hseg⟩

theorem boundary_no_second (words seg a : Nat) (ha : words≤a+1 ∨ seg≤a+1) :
    ¬(a+1<seg ∧ a+1<words) := by omega
end ResourceGeometryProofs
