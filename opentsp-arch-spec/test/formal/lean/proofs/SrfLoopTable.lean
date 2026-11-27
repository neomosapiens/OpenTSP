-- Each indexed array element is visited once; other elements retain their value.
import Proofs.SrfBitPrimitives
open Sail Out Out.Functions
set_option autoImplicit false
namespace SrfLoopTableProofs
open ResourceScanProofs

theorem fold_at {α β : Type} (atCell : α → Nat → β) (step : Int → α → α)
 (changeCell : Nat → β → β) (N : Nat)
 (hstep:∀k st j,k<N → j<N →
   atCell (step k st) j=(if j=k then changeCell k (atCell st j) else atCell st j))
 (n start j : Nat) (st : α) (hr:start+n≤N) (hj:j<N) :
 atCell (indexedFold step start n st) j=
 (if start≤j ∧ j<start+n then changeCell j (atCell st j) else atCell st j) := by
 induction n generalizing start st with
 | zero =>
   simp only [indexedFold,Nat.add_zero]
   rw [if_neg (by omega)]
 | succ n ih =>
   have hs:start<N:=by omega
   have ht:start+1+n≤N:=by omega
   have hv:=hstep start st j hs hj
   have hf:=ih (start+1) (step start st) ht
   simp only [indexedFold]
   have he:(start:Int)+1=((start+1:Nat):Int):=by omega
   rw [he,hf,hv]
   by_cases same:j=start
   · subst j
     simp [show ¬start+1≤start ∧ start<start+1+n from by omega,
       show start≤start ∧ start<start+(n+1) from by omega]
   · have hc:(start+1≤j ∧ j<start+1+n) ↔ (start≤j ∧ j<start+(n+1)):=by omega
     simp [same,hc]
end SrfLoopTableProofs
