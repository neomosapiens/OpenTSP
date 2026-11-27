-- Actual report bit merge factories and invariant key preservation.
import Proofs.ResourceScan
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfMergeProofs
open ResourceScanProofs
def streamStep (sp : Vector (BitVec Out.Functions.superlanes) 30) (x s : Int) (keep take : Bool) (e : srf_rep) : srf_rep :=
  let m := (GetElem?.getElem! e.sp s)
                let m : (BitVec Out.Functions.superlanes) :=
                  if ((s <b 3) : Bool)
                  then
                    (if (((BitVec.access (GetElem?.getElem! sp s) x) == 1#1) : Bool)
                    then (BitVec.update m x 1#1)
                    else m)
                  else
                    (if ((s ≥b 4) : Bool)
                    then
                      (if (take : Bool)
                      then (BitVec.update m x (BitVec.access (GetElem?.getElem! sp s) x))
                      else
                        (if ((Out.Functions.not keep) : Bool)
                        then
                          (if (((BitVec.access (GetElem?.getElem! sp s) x) == 1#1) : Bool)
                          then (BitVec.update m x 1#1)
                          else m)
                        else m))
                    else m)
                { e with sp := (vectorUpdate e.sp s m) }

def streamLoop (e : srf_rep) (prio : Int) (sp : Vector (BitVec Out.Functions.superlanes) 30) (x : Int) (keep take : Bool) : srf_rep := Id.run do
 let mut out:=e
 for s in [0:29:1]i do out:=streamStep sp x s keep take out
 pure out

def laneStep (prio : Int) (xs : BitVec Out.Functions.superlanes) (sp : Vector (BitVec Out.Functions.superlanes) 30) (x : Int) (e : srf_rep) : srf_rep := Id.run do
  if (((BitVec.access xs x) == 1#1) : Bool)
      then
        (do
          let keep : Bool := ((GetElem?.getElem! e.pr x) >b prio)
          let take : Bool := ((GetElem?.getElem! e.pr x) <b prio)
          let e := streamLoop e prio sp x keep take
          let p3 := (GetElem?.getElem! e.sp 3)
          let n3 : Int :=
            (((if (((BitVec.access (GetElem?.getElem! e.sp 0) x) == 1#1) : Bool)
                then 1
                else 0) +i (if (((BitVec.access (GetElem?.getElem! e.sp 1) x) == 1#1) : Bool)
                then 1
                else 0)) +i (if (((BitVec.access (GetElem?.getElem! e.sp 2) x) == 1#1) : Bool)
              then 1
              else 0))
          let p3 : (BitVec Out.Functions.superlanes) :=
            (BitVec.update p3 x
              (if (((Int.emod n3 2) == 1) : Bool)
              then 1#1
              else 0#1))
          let e : srf_rep := { e with sp := (vectorUpdate e.sp 3 p3) }
          if ((prio >b (GetElem?.getElem! e.pr x)) : Bool)
          then (pure { e with pr := (vectorUpdate e.pr x prio) })
          else (pure e))
      else (pure e)

def mergeFactory (e : srf_rep) (prio : Int) (xs : BitVec Out.Functions.superlanes) (sp : Vector (BitVec Out.Functions.superlanes) 30) : srf_rep := Id.run do
 let mut out:=e
 for x in [0:(Out.Functions.superlanes:Int)-1:1]i do out:=laneStep prio xs sp x out
 pure out

section
attribute [local irreducible] IntRange.forIn'
theorem actual_merge_source (e : srf_rep) (prio : Int) (xs : BitVec Out.Functions.superlanes) (sp : Vector (BitVec Out.Functions.superlanes) 30) :
 srf_rep_merge e prio xs sp=mergeFactory e prio xs sp := by
 unfold srf_rep_merge mergeFactory
 simp only [Id.run,pure_bind,bind_pure]
 apply congrArg (fun (body : (i : Int) → i∈({start:=0,stop:=(Out.Functions.superlanes:Int)-1,step:=1}:IntRange) → srf_rep → Id (ForInStep srf_rep)) => IntRange.forIn' (m:=Id) {start:=0,stop:=(Out.Functions.superlanes:Int)-1,step:=1} e body)
 funext x hx state
 unfold laneStep streamLoop streamStep
 simp only [Id.run,pure_bind,bind_pure]
 by_cases selected:((BitVec.access xs x.toNat)==1#1)=true
 all_goals simp only [Out.Functions.superlanes] at selected ⊢
 all_goals simp only [selected,ite_true,ite_false]
 all_goals rfl

end

def Key (e : srf_rep) := (e.used,e.rw,e.diag,e.wgap)

theorem fold_keeps {α β : Type} (view : α → β) (step : Int → α → α)
 (hs:∀i a,view (step i a)=view a) (i : Int) (n : Nat) (a : α) :
 view (indexedFold step i n a)=view a := by
 induction n generalizing i a with
 | zero => rfl
 | succ n ih => exact (ih (i+1) (step i a)).trans (hs i a)

theorem stream_step_key (sp : Vector (BitVec Out.Functions.superlanes) 30) (x s : Int) (keep take : Bool) (e : srf_rep) :
 Key (streamStep sp x s keep take e)=Key e := by rfl

theorem stream_loop_key (e : srf_rep) (prio : Int) (sp : Vector (BitVec Out.Functions.superlanes) 30) (x : Int) (keep take : Bool) :
 Key (streamLoop e prio sp x keep take)=Key e := by
 unfold streamLoop
 change Key (IntRange.forIn' (m:=Id) {start:=0,stop:=29,step:=1} e (fun s _ a => pure (.yield (streamStep sp x s keep take a))))=Key e
 rw [actual_indexed_range_fold _ _ _ rfl]
 exact fold_keeps Key _ (fun i a => stream_step_key sp x i keep take a) _ _ e

attribute [local irreducible] streamLoop streamStep

theorem lane_step_key (prio : Int) (xs : BitVec Out.Functions.superlanes) (sp : Vector (BitVec Out.Functions.superlanes) 30) (x : Int) (e : srf_rep) :
 Key (laneStep prio xs sp x e)=Key e := by
 unfold laneStep
 dsimp only [Id.run]
 split
 · split <;> change Key (streamLoop e prio sp x _ _)=Key e <;> exact stream_loop_key _ _ _ _ _ _
 · rfl

theorem actual_merge_key (e : srf_rep) (prio : Int) (xs : BitVec Out.Functions.superlanes) (sp : Vector (BitVec Out.Functions.superlanes) 30) :
 Key (srf_rep_merge e prio xs sp)=Key e := by
 rw [actual_merge_source]
 unfold mergeFactory
 change Key (IntRange.forIn' (m:=Id) {start:=0,stop:=(Out.Functions.superlanes:Int)-1,step:=1} e (fun x _ a => pure (.yield (laneStep prio xs sp x a))))=Key e
 rw [actual_indexed_range_fold _ _ _ rfl]
 exact fold_keeps Key _ (fun i a => lane_step_key prio xs sp i a) _ _ e
end SrfMergeProofs
