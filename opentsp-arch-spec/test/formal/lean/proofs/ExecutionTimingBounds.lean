-- Actual generated timing classes and every entry/type override obey ring horizons.
import Proofs.ExecuteProducer
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace ExecutionTimingBoundProofs

def Bounded (l:latency):Prop:=l.d_skew≤15 ∧ l.d_func≤63

theorem class_bounded (t:timing_class):Bounded (latency_of t):=by
 cases t <;> unfold Bounded <;> decide +kernel

theorem overrides_bounded : ∀o,o∈timing_overrides→Bounded {d_func:=o.d_func,d_skew:=o.d_skew}:=by
 simp [timing_overrides,Bounded]

theorem found_bounded (name:String) (xs:List timing_override)
 (hx:∀o,o∈xs→Bounded {d_func:=o.d_func,d_skew:=o.d_skew})
 (l:latency) (h:find_override name xs=some l):Bounded l:=by
 induction xs with
 | nil=>simp [find_override] at h
 | cons o tail ih=>
  unfold find_override at h
  split at h
  · have he:{d_func:=o.d_func,d_skew:=o.d_skew}=l:=Option.some.inj h
    rw [←he]
    exact hx o (by simp)
  · exact ih (fun o ho=>hx o (List.mem_cons_of_mem _ ho)) h

theorem entry_bounded (name:String) (t:timing_class):Bounded (entry_latency name t):=by
 unfold entry_latency
 cases h:find_override name timing_overrides with
 | none=>exact class_bounded t
 | some l=>exact found_bounded name _ overrides_bounded l h

theorem typed_entry_bounded (name tname:String) (t:timing_class):Bounded (entry_latency_typed name tname t):=by
 unfold entry_latency_typed
 split
 · exact entry_bounded name t
 · cases h:find_override (name++("."++tname)) timing_overrides with
   | none=>exact entry_bounded name t
   | some l=>exact found_bounded _ _ overrides_bounded l h

theorem actual_instruction_timing_bounded (i:instruction):
 (latency_of_instr i).d_skew≤15 ∧ (latency_of_instr i).d_func≤63:=
 typed_entry_bounded (entry_of i) (timing_type_of i) (timing_of i)

end ExecutionTimingBoundProofs
