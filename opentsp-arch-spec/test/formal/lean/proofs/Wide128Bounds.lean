-- Actual x128-admitted dot formats fit before direct ACC_REPLACE.
-- Decoder premises are the already universally proved native domains suite.
import Proofs.InputDecode
open Sail Out Out.Functions Out.fclass
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
set_option exponentiation.threshold 100000
namespace Wide128Proofs
open DotBoundProofs DotSemanticProofs

def TightBounds (a : fval) (bits emax : Int) : Prop :=
  a.cls=FFinite → 0 ≤ a.mag ∧ a.mag ≤ pow2i bits ∧ a.e ≤ emax

def DecoderBounds (f : fmt) (bits emax : Int) : Prop :=
  ∀ x : BitVec 32, TightBounds (fdecode f x) bits emax

theorem admitted_term_bound (a b : fval) (bits emax u : Int)
    (hb : 0 ≤ bits ∧ bits ≤ 40) (hu : 2*emax+u ≤ 80-2*bits)
    (ha : TightBounds a bits emax) (hbm : TightBounds b bits emax) :
    (exactTerm a b u).natAbs ≤ 2^80 := by
  cases hca : a.cls <;> cases hcb : b.cls <;> try (solve | simp [exactTerm,hca,hcb])
  have ha := ha hca
  have hbm := hbm hcb
  have hprod := CallerBoundProofs.nonnegative_product_bound a.mag b.mag (pow2i bits) (pow2i bits) ⟨ha.1,ha.2.1⟩ ⟨hbm.1,hbm.2.1⟩
  have hscale := CallerBoundProofs.pow2_any_upper (a.e+b.e+u) (80-2*bits) (by omega) (by omega)
  have hspos := PositivityProofs.pow2i_any_positive (a.e+b.e+u)
  have htwopow := NumericalProofs.pow2i_add_nonnegative bits bits hb.1 hb.1
  rw [show bits+bits=2*bits by omega] at htwopow
  rw [←htwopow] at hprod
  have h := CallerBoundProofs.nonnegative_product_bound (a.mag*b.mag) (pow2i (a.e+b.e+u))
    (pow2i (2*bits)) (pow2i (80-2*bits)) hprod ⟨by omega,hscale⟩
  have hp := NumericalProofs.pow2i_add_nonnegative (2*bits) (80-2*bits) (by omega) (by omega)
  rw [show 2*bits+(80-2*bits)=80 by omega] at hp
  rw [←hp] at h
  have hpow : pow2i 80=((2^80:Nat):Int) := by decide +kernel
  have habs : (a.mag*b.mag*pow2i (a.e+b.e+u)).natAbs ≤ 2^80 := by
    have he : ((a.mag*b.mag*pow2i (a.e+b.e+u)).natAbs:Int)=a.mag*b.mag*pow2i (a.e+b.e+u) := by omega
    apply Int.ofNat_le.mp
    rw [he,←hpow]
    exact h.2
  cases hn : neq_bool a.neg b.neg <;> simpa [exactTerm,hca,hcb,hn,Int.natAbs_neg] using habs

theorem indexed_sum_budget (term : Int → Int) (B : Nat) (ht : ∀ k, (term k).natAbs ≤ B)
    (i : Int) (n : Nat) : (indexedSum term i n).natAbs ≤ n*B := by
  induction n generalizing i with
  | zero => simp [indexedSum]
  | succ n ih =>
    simp only [indexedSum]
    have h := Int.natAbs_add_le (term i) (indexedSum term (i+1) n)
    have hi := ih (i+1)
    have hb := ht i
    rw [Nat.add_mul]
    simp only [Nat.one_mul]
    omega

theorem pair_replace_bound (f : fmt) (bits emax : Int) (wlo whi xlo xhi : vec)
    (hb : 0 ≤ bits ∧ bits ≤ 40) (hu : 2*emax+48 ≤ 80-2*bits)
    (hd : DecoderBounds f bits emax) :
    (pair_exact f 48 wlo whi xlo xhi).dsum.natAbs ≤ 2^89 := by
  rw [pair_exact_sum]
  have hterm : ∀ k, (exactTerm (pairA f wlo whi k) (pairB f xlo xhi k) 48).natAbs ≤ 2^80 := by
    intro k; exact admitted_term_bound _ _ bits emax 48 hb hu (hd _) (hd _)
  have h := indexed_sum_budget _ (2^80) hterm 0 Out.Functions.vlanes.toNat
  have hn : Out.Functions.vlanes.toNat ≤ 512 := by decide
  have hp : 512*(2^80)=(2:Nat)^89 := by decide +kernel
  have hm := Nat.mul_le_mul_right (2^80) hn
  rw [hp] at hm
  exact Nat.le_trans h hm

theorem fp8_replace_bound (f : fmt) (bits emax : Int) (wr xv : vec)
    (hb : 0 ≤ bits ∧ bits ≤ 40) (hu : 2*emax+48 ≤ 80-2*bits)
    (hd : DecoderBounds f bits emax) :
    (fp8_exact f wr xv).dsum.natAbs ≤ 2^89 := by
  rw [fp8_exact_sum]
  have hterm : ∀ k, (exactTerm (fp8A f wr k) (fp8B f xv k) 48).natAbs ≤ 2^80 := by
    intro k; exact admitted_term_bound _ _ bits emax 48 hb hu (hd _) (hd _)
  have h := indexed_sum_budget _ (2^80) hterm 0 Out.Functions.vlanes.toNat
  have hn : Out.Functions.vlanes.toNat ≤ 512 := by decide
  have hp : 512*(2^80)=(2:Nat)^89 := by decide +kernel
  have hm := Nat.mul_le_mul_right (2^80) hn
  rw [hp] at hm
  exact Nat.le_trans h hm

theorem bfp_replace_bound (wd ws xd xs : vec) (hd : DecoderBounds FMT_E4M3 4 5) :
    (bfp8_exact wd ws xd xs).dsum.natAbs ≤ 2^89 := by
  rw [bfp_exact_sum]
  have hterm : ∀ k, (exactTerm (fp8A FMT_E4M3 wd k) (fp8B FMT_E4M3 xd k)
      (blk_exp ws k+blk_exp xs k+48)).natAbs ≤ 2^80 := by
    intro k
    have hl := block_scale_range ws k.toNat
    have hr := block_scale_range xs k.toNat
    exact admitted_term_bound _ _ 4 5 _ (by decide) (by omega) (hd _) (hd _)
  have h := indexed_sum_budget _ (2^80) hterm 0 Out.Functions.vlanes.toNat
  have hn : Out.Functions.vlanes.toNat ≤ 512 := by decide
  have hp : 512*(2^80)=(2:Nat)^89 := by decide +kernel
  have hm := Nat.mul_le_mul_right (2^80) hn
  rw [hp] at hm
  exact Nat.le_trans h hm

theorem all_x128_admitted_replace_bounds (wlo whi xlo xhi wr xv wd ws xd xs : vec)
    (h16 : DecoderBounds FMT16 11 5) (hE4 : DecoderBounds FMT_E4M3 4 5)
    (hE5 : DecoderBounds FMT_E5M2 3 13) :
    (pair_exact FMT16 48 wlo whi xlo xhi).dsum.natAbs ≤ 2^89 ∧
    (fp8_exact FMT_E4M3 wr xv).dsum.natAbs ≤ 2^89 ∧
    (fp8_exact FMT_E5M2 wr xv).dsum.natAbs ≤ 2^89 ∧
    (bfp8_exact wd ws xd xs).dsum.natAbs ≤ 2^89 := by
  exact ⟨pair_replace_bound FMT16 11 5 _ _ _ _ (by decide) (by decide) h16,
    fp8_replace_bound FMT_E4M3 4 5 _ _ (by decide) (by decide) hE4,
    fp8_replace_bound FMT_E5M2 3 13 _ _ (by decide) (by decide) hE5,
    bfp_replace_bound _ _ _ _ hE4⟩

theorem replace_fits_signed128 (d : dsum) (hd : d.dsum.natAbs ≤ 2^89) :
    -pow2i 127 ≤ (dsum_wide d).wv ∧ (dsum_wide d).wv < pow2i 127 := by
  have hp : ((2^89:Nat):Int)<pow2i 127 := by decide +kernel
  have hb : -pow2i 127 ≤ d.dsum ∧ d.dsum < pow2i 127 := by omega
  cases hn : d.dnan <;> cases hpi : d.dpinf <;> cases hni : d.dninf <;>
    simp [dsum_wide,hn,hpi,hni] <;> omega

def StoredWide128 (w : wlane) : Prop :=
  DotSemanticProofs.WideClass w ∧ -pow2i 127 ≤ w.wv ∧ w.wv < pow2i 127

theorem reset_stored128 : StoredWide128 wlane_zero := by
  have hc := DotSemanticProofs.wide_zero_class
  have hp := IntervalProofs.pow2i_positive 127 (by decide)
  exact ⟨hc,by simp [wlane_zero]; omega⟩

theorem replace_stored128 (d : dsum) (hd : d.dsum.natAbs ≤ 2^89) : StoredWide128 (dsum_wide d) :=
  ⟨DotSemanticProofs.dot_wide_class d,replace_fits_signed128 d hd⟩

theorem accumulate_stored128 (a b : wlane) (ha : StoredWide128 a) (hb : StoredWide128 b) :
    StoredWide128 (wlane_add a b) := by
  have hc := DotSemanticProofs.wide_add_class_preserved a b ha.1 hb.1
  refine ⟨hc,?_⟩
  have hwrap := CallerBoundProofs.wrap_s_range (a.wv+b.wv) 128 (by decide)
  change -pow2i 127 ≤ wrap_s (a.wv+b.wv) 128 ∧ wrap_s (a.wv+b.wv) 128 < pow2i 127 at hwrap
  rw [DotSemanticProofs.complete_wide_class_add a b ha.1 hb.1]
  rcases ha.1 with hca|hca|hca|hca <;> rcases hb.1 with hcb|hcb|hcb|hcb <;>
    simp [DotSemanticProofs.wideAddSpec,hca,hcb]
  all_goals first | exact ha.2 | exact hb.2 | exact hwrap | omega

end Wide128Proofs
