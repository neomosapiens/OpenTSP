-- Actual Ziv-loop source binding and complete precision-call domain.
import Proofs.RootEnclosure
open Sail Out Out.Functions
set_option autoImplicit false
namespace ZivDomainProofs

def Allowed (p : Int) : Prop :=
  p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096 ∨ p=8192

def Active (p : Int) : Prop :=
  p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096

def zivStep {α : Type} (evaluate : Int → Option α → Option α)
    (state : Int × Option α) : Int × Option α :=
  let (p,res) := state
  let (p,res) : Int × Option α :=
    if (((is_none res) && (p ≤b 4096)) : Bool) then
      let res := evaluate p res
      let p := p *i 2
      (p,res)
    else (p,res)
  (p,res)

def zivFactory {α : Type} (evaluate : Int → Option α → Option α)
    (upper : Int) : Int × Option α := Id.run do
  let mut state : Int × Option α := (64,none)
  for _round in [0:upper:1]i do
    state := zivStep evaluate state
  return state

def fpEvaluate (op : Int) (f : fmt) (a : fval) (p : Int)
    (_old : Option (BitVec 32)) : Option (BitVec 32) :=
  let (lo,hi,sh) := tr_interval op a p
  round_iv f lo hi sh

def intEvaluate (op : Int) (a : fval) (p : Int) (old : Option Int) : Option Int :=
  let (lo,hi,sh) := tr_interval op a p
  let x := rne_int lo sh
  let y := rne_int hi sh
  if ((x == y) : Bool) then some x else old

theorem fp_ziv_source (op : Int) (f : fmt) (a : fval) :
    tr_eval op f a=(zivFactory (fpEvaluate op f a) 6).2 := by rfl

theorem int_ziv_source (op : Int) (a : fval) :
    tr_eval_int op a=(zivFactory (intEvaluate op a) 6).2 := by rfl

theorem active_of_allowed_guard (p : Int) (h : Allowed p) (hg : p ≤ 4096) :
    Active p := by dsimp [Allowed,Active] at *; omega

theorem doubled_precision_allowed (p : Int) (h : Active p) : Allowed (p*2) := by
  dsimp [Active,Allowed] at *; omega

theorem ziv_step_preserves_domain {α : Type} (evaluate : Int → Option α → Option α)
    (state : Int × Option α) (h : Allowed state.1) :
    Allowed (zivStep evaluate state).1 := by
  rcases state with ⟨p,res⟩
  by_cases hg : is_none res=true ∧ p ≤ 4096
  · have ha := active_of_allowed_guard p h hg.2
    simpa [zivStep,hg] using doubled_precision_allowed p ha
  · simpa [zivStep,hg] using h

theorem repeat_ziv_domain {α : Type} (evaluate : Int → Option α → Option α)
    (steps : Nat) (state : Int × Option α) (h : Allowed state.1) :
    Allowed (IntervalProofs.repeatStep (zivStep evaluate) steps state).1 := by
  induction steps generalizing state with
  | zero => exact h
  | succ steps ih => exact ih _ (ziv_step_preserves_domain evaluate state h)

theorem ziv_factory_iteration {α : Type} (evaluate : Int → Option α → Option α)
    (upper : Int) :
    zivFactory evaluate upper=IntervalProofs.repeatStep (zivStep evaluate) (upper+1).toNat (64,none) := by
  change IntRange.forIn' (m := Id) {start := 0,stop := upper,step := 1}
    (64,none) (fun _ _ state => pure (.yield (zivStep evaluate state))) = _
  simpa using IntervalProofs.unit_range_iterate {start := 0,stop := upper,step := 1}
    (zivStep evaluate) (64,none) rfl

theorem actual_ziv_precision_domain {α : Type} (evaluate : Int → Option α → Option α)
    (steps : Nat) :
    Allowed (IntervalProofs.repeatStep (zivStep evaluate) steps (64,none)).1 := by
  apply repeat_ziv_domain
  dsimp [Allowed]; omega

end ZivDomainProofs
