-- All64×44 actual stream entries, independent existence/upstream semantics.
import Proofs.SrfNearest
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfUpstreamProofs
open ResourceScanProofs

theorem exists_interval_step (p : Int → Prop) (i : Int) (n : Nat) :
 (∃j,i≤j ∧ j<i+(n+1) ∧ p j) ↔ p i ∨ ∃j,i+1≤j ∧ j<i+1+n ∧ p j := by
 constructor
 · rintro ⟨j,hj,hb,hp⟩
   by_cases he:j=i
   · subst j; exact Or.inl hp
   · exact Or.inr ⟨j,by omega,by omega,hp⟩
 · rintro (hp|⟨j,hj,hb,hp⟩)
   · exact ⟨i,by omega,by omega,hp⟩
   · exact ⟨j,by omega,by omega,hp⟩

theorem scan_spec {α : Type} (step : Int → α → α) (field : α → Bool) (p : Int → Prop)
 (hs : ∀i a,field (step i a)=true ↔ field a=true ∨ p i) (i : Int) (n : Nat) (a : α) :
 field (indexedFold step i n a)=true ↔ field a=true ∨ ∃j,i≤j ∧ j<i+n ∧ p j := by
 induction n generalizing i a with
 | zero => simp [indexedFold]; omega
 | succ n ih =>
   rw [indexedFold,ih,hs]
   simp only [Int.natCast_add,Int.natCast_one]
   rw [exists_interval_step]
   exact or_assoc

def live (stream : srf_stream) (t k w : Int) : Bool :=
 let e:=stream[k]![w]!
 e.valid && decide (e.diag+t≤44)
def cellStep (stream : srf_stream) (g : Nat) (t k w : Int) (a : Bool×Bool) : Bool×Bool :=
 let e:=stream[k]![w]!
 if e.valid && decide (e.diag+t≤44) then (true,if w<g then true else a.2) else a

def rowFactory (stream : srf_stream) (g : Nat) (t k upper : Int) (a : Bool×Bool) : Bool×Bool := Id.run do
 let mut state:=a
 for w in [0:upper:1]i do state:=cellStep stream g t k w state
 pure state

def streamFactory (stream : srf_stream) (g : Nat) (t upper : Int) : Bool×Bool := Id.run do
 let mut state:=(false,false)
 for k in [0:upper:1]i do state:=rowFactory stream g t k 43 state
 pure state

section
attribute [local irreducible] IntRange.forIn'
theorem actual_stream_source (stream : srf_stream) (g : Nat) (t : Int) :
 srf_stream_only_downstream stream g t=((streamFactory stream g t 63).1 && !(streamFactory stream g t 63).2) := by
 simp only [srf_stream_only_downstream,streamFactory,rowFactory,cellStep,pure_bind,bind_pure,Out.Functions.not,decide_eq_true_eq]
 rfl
end

theorem cell_any (stream : srf_stream) (g : Nat) (t k w : Int) (a : Bool×Bool) :
 (cellStep stream g t k w a).1=true ↔ a.1=true ∨ live stream t k w=true := by
 dsimp only [cellStep,live]
 generalize h:(stream[k]![w]!.valid && decide (stream[k]![w]!.diag+t≤44))=p
 cases p <;> simp

theorem cell_upstream (stream : srf_stream) (g : Nat) (t k w : Int) (a : Bool×Bool) :
 (cellStep stream g t k w a).2=true ↔ a.2=true ∨ (live stream t k w=true ∧ w<g) := by
 dsimp only [cellStep,live]
 generalize h:(stream[k]![w]!.valid && decide (stream[k]![w]!.diag+t≤44))=p
 cases p <;> by_cases hw:w<g <;> simp [hw]

theorem row_fold (stream : srf_stream) (g : Nat) (t k : Int) (a : Bool×Bool) :
 rowFactory stream g t k 43 a=indexedFold (cellStep stream g t k) 0 44 a := by
 unfold rowFactory
 change IntRange.forIn' (m:=Id) {start:=0,stop:=43,step:=1} a (fun w _ a => pure (.yield (cellStep stream g t k w a)))=_
 rw [actual_indexed_range_fold _ _ _ rfl]
 rfl

theorem row_any (stream : srf_stream) (g : Nat) (t k : Int) (a : Bool×Bool) :
 (rowFactory stream g t k 43 a).1=true ↔ a.1=true ∨ ∃w,0≤w ∧ w<44 ∧ live stream t k w=true := by
 rw [row_fold]
 exact scan_spec _ Prod.fst _ (cell_any stream g t k) 0 44 a

theorem row_upstream (stream : srf_stream) (g : Nat) (t k : Int) (a : Bool×Bool) :
 (rowFactory stream g t k 43 a).2=true ↔ a.2=true ∨ ∃w,0≤w ∧ w<44 ∧ live stream t k w=true ∧ w<g := by
 rw [row_fold]
 exact scan_spec _ Prod.snd _ (cell_upstream stream g t k) 0 44 a

theorem stream_fold (stream : srf_stream) (g : Nat) (t : Int) :
 streamFactory stream g t 63=indexedFold (fun k a => rowFactory stream g t k 43 a) 0 64 (false,false) := by
 unfold streamFactory
 change IntRange.forIn' (m:=Id) {start:=0,stop:=63,step:=1} (false,false) (fun k _ a => pure (.yield (rowFactory stream g t k 43 a)))=_
 rw [actual_indexed_range_fold _ _ _ rfl]
 rfl

theorem stream_any (stream : srf_stream) (g : Nat) (t : Int) :
 (streamFactory stream g t 63).1=true ↔ ∃k,0≤k ∧ k<64 ∧ ∃w,0≤w ∧ w<44 ∧ live stream t k w=true := by
 rw [stream_fold]
 simpa using scan_spec _ Prod.fst _ (row_any stream g t) 0 64 (false,false)

theorem stream_upstream (stream : srf_stream) (g : Nat) (t : Int) :
 (streamFactory stream g t 63).2=true ↔ ∃k,0≤k ∧ k<64 ∧ ∃w,0≤w ∧ w<44 ∧ live stream t k w=true ∧ w<g := by
 rw [stream_fold]
 simpa using scan_spec _ Prod.snd _ (row_upstream stream g t) 0 64 (false,false)

theorem actual_only_downstream (stream : srf_stream) (g : Nat) (t : Int) :
 srf_stream_only_downstream stream g t=true ↔
 (∃k,0≤k ∧ k<64 ∧ ∃w,0≤w ∧ w<44 ∧ live stream t k w=true) ∧
 ¬(∃k,0≤k ∧ k<64 ∧ ∃w,0≤w ∧ w<44 ∧ live stream t k w=true ∧ w<g) := by
 rw [actual_stream_source,Bool.and_eq_true]
 rw [stream_any]
 have hu:=stream_upstream stream g t
 apply and_congr Iff.rfl
 cases h:(streamFactory stream g t 63).2 with
 | false =>
   constructor
   · intro _ hex
     have hit:=hu.mpr hex
     simp [h] at hit
   · intro _; rfl
 | true =>
   constructor
   · intro hf; cases hf
   · intro hn; exact False.elim (hn (hu.mp h))
end SrfUpstreamProofs
