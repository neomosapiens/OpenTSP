-- Actual signed integer RNE, public saturation and successful-cast composition.
import Proofs.GridBounds
open Sail Out Out.Functions
set_option autoImplicit false
namespace SignedRoundingProofs

def signedChoice (q r h : Int) : Int :=
  if h<r ∨ (r=h ∧ q.tmod 2≠0) then q+1 else q

theorem signed_choice_half (q r h : Int) (hh : 0 < h)
    (hr : 0 ≤ r ∧ r < 2*h) :
    -h ≤ q*(2*h)+r-signedChoice q r h*(2*h) ∧
      q*(2*h)+r-signedChoice q r h*(2*h) ≤ h := by
  by_cases hc : h<r ∨ (r=h ∧ q.tmod 2≠0)
  · simp [signedChoice,hc,Int.add_mul]; omega
  · simp [signedChoice,hc]; omega

theorem signed_choice_tie_even (q h : Int) : signedChoice q h h%2=0 := by
  have h0 := Int.emod_nonneg q (by decide : (2:Int)≠0)
  have h1 := Int.emod_lt_of_pos q (by decide : (0:Int)<2)
  by_cases he : q%2=0
  · have ht := (RootEnclosureProofs.truncated_even_iff q).mpr he
    simp [signedChoice,ht,he]
  · have ht : q.tmod 2≠0 := by intro h; exact he ((RootEnclosureProofs.truncated_even_iff q).mp h)
    have ho : q%2=1 := by omega
    simp [signedChoice,ht,Int.add_emod,ho]

theorem rne_int_source (v sh : Int) (hs : sh < 0) :
    rne_int v sh=
      let d := pow2i (-sh)
      let q := idiv_floor v d
      signedChoice q (v-q*d) (d.tdiv 2) := by
  simp [rne_int,signedChoice,show ¬sh ≥ 0 by omega]

theorem rne_int_signed_nearest (v sh : Int) (hs : sh < 0) (z : Int) :
    (v-rne_int v sh*pow2i (-sh)).natAbs ≤ (v-z*pow2i (-sh)).natAbs := by
  have hp := RoundingProofs.positive_power_even (-sh) (by omega)
  let h := pow2i (-sh-1)
  have hd : pow2i (-sh)=2*h := hp.1
  have hh : 0 < h := hp.2
  let q := idiv_floor v (2*h)
  let r := v-q*(2*h)
  have hf := IntervalProofs.floor_bounds v (2*h) (by omega)
  have hrem : 0 ≤ r ∧ r < 2*h := by
    have he : (q+1)*(2*h)=q*(2*h)+2*h := by grobner
    change q*(2*h) ≤ v ∧ v < (q+1)*(2*h) at hf
    dsimp [r]; omega
  have heval : rne_int v sh=signedChoice q r h := by
    rw [rne_int_source v sh hs,hd]
    have he : (2*h).tdiv 2=h := Int.mul_tdiv_cancel_left _ (by decide : (2:Int)≠0)
    dsimp only
    rw [he]
  have he := signed_choice_half q r h hh hrem
  have hsum : q*(2*h)+r=v := by dsimp [r]; grobner
  rw [hsum] at he
  rw [hd,heval]
  exact RoundingProofs.nearest_from_half v _ h hh he z

theorem rne_int_tie_even (v sh : Int) (hs : sh < 0)
    (ht : v - idiv_floor v (pow2i (-sh))*pow2i (-sh)=(pow2i (-sh)).tdiv 2) :
    rne_int v sh%2=0 := by
  rw [rne_int_source v sh hs]
  dsimp only
  rw [ht]
  exact signed_choice_tie_even _ _

theorem rne_int_nonnegative_shift (v sh : Int) (hs : 0 ≤ sh) :
    rne_int v sh=v*pow2i sh := by simp [rne_int,hs]

theorem int_of_fp_finite_source (f : fmt) (x : BitVec 32) (bits : Int)
    (hc : (fdecode f x).cls=fclass.FFinite) :
    int_of_fp f x bits=
      sat_s (if (fdecode f x).neg then -RoundingProofs.gridRound (fdecode f x).mag (fdecode f x).e 0
        else RoundingProofs.gridRound (fdecode f x).mag (fdecode f x).e 0) bits := by
  simp [int_of_fp,hc,sat_s,RoundingProofs.gridRound]

theorem sat_s_range (v bits : Int) (hb : 1 ≤ bits) :
    -pow2i (bits-1) ≤ sat_s v bits ∧ sat_s v bits ≤ pow2i (bits-1)-1 := by
  have h := IntervalProofs.pow2i_positive (bits-1) (by omega)
  simp [sat_s]
  split <;> (first | omega | (split <;> omega))

theorem sat_s_nearest (v bits z : Int) (hb : 1 ≤ bits)
    (hz : -pow2i (bits-1) ≤ z ∧ z ≤ pow2i (bits-1)-1) :
    (v-sat_s v bits).natAbs ≤ (v-z).natAbs := by
  have h := IntervalProofs.pow2i_positive (bits-1) (by omega)
  simp [sat_s]
  split <;> (first | omega | (split <;> omega))

theorem i8_transcendental_success (op : Int) (x : BitVec 32) (r : Int)
    (h : i8_tr_value op (s_of_bits x 8)=some r) :
    i8_transcendental op x=pure (bits_of_s (sat_s r 8) 8) := by
  simp [i8_transcendental,h]


theorem fp_transcendental_success (op : Int) (f : fmt) (x r : BitVec 32)
    (h : fp_tr_value op f x=some r) : fp_transcendental op f x=pure r := by
  simp [fp_transcendental,h]

theorem fp_tr_nan (op : Int) (f : fmt) (x : BitVec 32)
    (h : (fdecode f x).cls=fclass.FNaN) :
    fp_tr_value op f x=some (canon_nan f) := by simp [fp_tr_value,h]

theorem fp_tr_zero (op : Int) (f : fmt) (x : BitVec 32)
    (h : (fdecode f x).cls=fclass.FZero) :
    fp_tr_value op f x=some
      (if op=21 then zero_bits f (fdecode f x).neg
       else if op=22 then fp_of_int f 1
       else if op=23 then inf_bits f (fdecode f x).neg else inf_bits f true) := by
  simp [fp_tr_value,h]

theorem fp_tr_inf (op : Int) (f : fmt) (x : BitVec 32)
    (h : (fdecode f x).cls=fclass.FInf) :
    fp_tr_value op f x=some
      (if op=21 then fp_of_int f (if (fdecode f x).neg then -1 else 1)
       else if op=22 then (if (fdecode f x).neg then zero_bits f false else inf_bits f false)
       else if op=23 then (if (fdecode f x).neg then canon_nan f else zero_bits f false)
       else if (fdecode f x).neg then canon_nan f else inf_bits f false) := by
  cases hn : (fdecode f x).neg <;> simp [fp_tr_value,h,hn]

theorem fp_tr_negative_domain (op : Int) (f : fmt) (x : BitVec 32)
    (hc : (fdecode f x).cls=fclass.FFinite) (hn : (fdecode f x).neg=true)
    (hop : op=23 ∨ op=24) : fp_tr_value op f x=some (canon_nan f) := by
  rcases hop with rfl|rfl <;> simp [fp_tr_value,hc,hn]

theorem fp_tr_log_one (f : fmt) (x : BitVec 32)
    (hc : (fdecode f x).cls=fclass.FFinite) (hn : (fdecode f x).neg=false)
    (hm : (fdecode f x).mag=pow2i (-(fdecode f x).e)) (he : (fdecode f x).e ≤ 0) :
    fp_tr_value 24 f x=some (zero_bits f false) := by
  simp [fp_tr_value,hc,hn,hm,he]

theorem int_of_fp_nan (f : fmt) (x : BitVec 32) (bits : Int)
    (h : (fdecode f x).cls=fclass.FNaN) : int_of_fp f x bits=0 := by simp [int_of_fp,h]

theorem int_of_fp_zero (f : fmt) (x : BitVec 32) (bits : Int)
    (h : (fdecode f x).cls=fclass.FZero) : int_of_fp f x bits=0 := by simp [int_of_fp,h]

theorem int_of_fp_inf (f : fmt) (x : BitVec 32) (bits : Int)
    (h : (fdecode f x).cls=fclass.FInf) : int_of_fp f x bits=
      if (fdecode f x).neg then -pow2i (bits-1) else pow2i (bits-1)-1 := by simp [int_of_fp,h]

end SignedRoundingProofs
