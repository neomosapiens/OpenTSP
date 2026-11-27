-- Actual SRF upstream scan: largest eligible producer, including absence.
import Proofs.ResourceScan
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfNearestProofs
open ResourceScanProofs

def eligible (slot : srf_slot) (g : Nat) (d : Int) (w : Int) : Bool :=
  slot[w]!.valid && ((slot[w]!.diag == d) && (w  <  g))

def chooseStep (p : Int → Bool) (w best : Int) : Int :=
  if p w && (w > best) then w else best

def bestStep (slot : srf_slot) (g : Nat) (d w best : Int) : Int :=
  let e:=slot[w]!
  if e.valid && ((e.diag == d) && ((w  <  g) && (w > best))) then w else best

def bestFactory (slot : srf_slot) (g : Nat) (d upper : Int) : Int := Id.run do
  let mut best : Int := -1
  for w in [0:upper:1]i do
    best:=bestStep slot g d w best
  pure best

section
attribute [local irreducible] IntRange.forIn'
theorem actual_best_source (slot : srf_slot) (g : Nat) (d : Int) :
    srf_slot_best slot g d=bestFactory slot g d 43 := by
  simp only [srf_slot_best,bestFactory,bestStep,pure_bind,bind_pure]
end

theorem step_semantics (slot : srf_slot) (g : Nat) (d w best : Int) :
    bestStep slot g d w best=chooseStep (eligible slot g d) w best := by
  simp [bestStep,chooseStep,eligible,Bool.and_assoc]

def Winner (p : Int → Bool) (n best : Int) : Prop :=
  -1 ≤ best ∧ best < n ∧ (best= -1 ∨ (0 ≤ best ∧ p best=true)) ∧
    ∀ w,0 ≤ w → w < n → p w=true → w ≤ best

theorem initial_winner (p : Int → Bool) : Winner p 0 (-1) := by
  constructor; omega
  constructor; omega
  constructor; exact Or.inl rfl
  intro w hw hl
  omega

theorem next_winner (p : Int → Bool) (n best : Int) (hn : 0 ≤ n) (h : Winner p n best) :
    Winner p (n+1) (chooseStep p n best) := by
  rcases h with ⟨hlo,hhi,hvalid,hmax⟩
  cases hp : p n
  · simp only [chooseStep,hp,Bool.false_and,Bool.false_eq_true,ite_false]
    refine ⟨hlo,by omega,hvalid,?_⟩
    intro w hw hl hwP
    have hwne : w≠n := by intro he; subst w; rw [hp] at hwP; contradiction
    exact hmax w hw (by omega) hwP
  · simp only [chooseStep,hp,Bool.true_and,show decide (n > best)=true from by simp [hhi],ite_true]
    refine ⟨by omega,by omega,Or.inr ⟨hn,hp⟩,?_⟩
    intro w hw hl hwP
    omega

theorem fold_winner (p : Int → Bool) (n : Nat) (start best : Int)
    (hs : 0 ≤ start) (h : Winner p start best) :
    Winner p (start+n) (indexedFold (chooseStep p) start n best) := by
  induction n generalizing start best with
  | zero => simpa [indexedFold] using h
  | succ n ih =>
    have hnext:=next_winner p start best hs h
    have hf:=ih (start+1) (chooseStep p start best) (by omega) hnext
    simpa only [indexedFold,Int.natCast_add,Int.natCast_one,Int.add_assoc,Int.add_comm,Int.add_left_comm] using hf

theorem actual_nearest (slot : srf_slot) (g : Nat) (d : Int) :
    Winner (eligible slot g d) 44 (srf_slot_best slot g d) := by
  rw [actual_best_source]
  change Winner _ _ (IntRange.forIn' (m := Id) {start:=0,stop:=43,step:=1} (-1)
    (fun w _ b => pure (.yield (bestStep slot g d w b))))
  simp only [step_semantics]
  rw [actual_indexed_range_fold _ _ _ rfl]
  simpa using fold_winner (eligible slot g d) 44 0 (-1) (by decide) (initial_winner _)

theorem actual_found_eligible (slot : srf_slot) (g : Nat) (d : Int)
    (h : 0 ≤ srf_slot_best slot g d) :
    0 ≤ srf_slot_best slot g d ∧ srf_slot_best slot g d < 44 ∧
    eligible slot g d (srf_slot_best slot g d)=true := by
  have hw:=actual_nearest slot g d
  refine ⟨h,hw.2.1,?_⟩
  rcases hw.2.2.1 with hn|hp
  · omega
  · exact hp.2

theorem actual_none_iff (slot : srf_slot) (g : Nat) (d : Int) :
    srf_slot_best slot g d= -1 ↔ ∀w,0 ≤ w → w < 44 → eligible slot g d w=false := by
  have hw:=actual_nearest slot g d
  constructor
  · intro h w h0 h44
    cases hp : eligible slot g d w
    · rfl
    · have hh:=hw.2.2.2 w h0 h44 hp
      omega
  · intro h
    rcases hw.2.2.1 with hn|hp
    · exact hn
    · have hh:=h _ hp.1 hw.2.1
      rw [hp.2] at hh
      contradiction
theorem actual_slot_read_found (slot : srf_slot) (g : Nat) (d : Int)
    (h : 0≤srf_slot_best slot g d) :
    srf_slot_read_ov slot g d=(true,slot[srf_slot_best slot g d]!.value,slot[srf_slot_best slot g d]!.ov) := by
  have hb:=actual_found_eligible slot g d h
  have he : Int.emod (srf_slot_best slot g d) 44=srf_slot_best slot g d := by
    change srf_slot_best slot g d %44=srf_slot_best slot g d
    exact Int.emod_eq_of_lt h hb.2.1
  simp [srf_slot_read_ov,h,he]

theorem actual_slot_read_missing (slot : srf_slot) (g : Nat) (d : Int)
    (h : ∀w,0≤w → w < 44 → eligible slot g d w=false) :
    srf_slot_read_ov slot g d=(false,BitVec.zero (8 *i Out.Functions.vlanes),ecc_ov_zero) := by
  have he:=(actual_none_iff slot g d).mpr h
  simp [srf_slot_read_ov,he,Out.Functions.vlanes]

theorem actual_nearest_is_max (slot : srf_slot) (g : Nat) (d w : Int)
    (hw : 0≤w ∧ w < 44) (he : eligible slot g d w=true)
    (hm : ∀k,0≤k → k < 44 → eligible slot g d k=true → k≤w) :
    srf_slot_best slot g d=w := by
  have h:=actual_nearest slot g d
  have hl:=h.2.2.2 w hw.1 hw.2 he
  have h0:0≤srf_slot_best slot g d := by omega
  have hb:=actual_found_eligible slot g d h0
  have hh:=hm _ hb.1 hb.2.1 hb.2.2
  omega
end SrfNearestProofs
