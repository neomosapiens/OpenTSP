-- Real AGT producer time origin: due references cannot issue before their dispatch.
import Proofs.ExecuteProducer
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace AgtOriginProofs

def TimeOrigin (a:agt) : Prop:=a.t0<=a.last

theorem idle_origin : TimeOrigin agt_idle := by change (0:Nat)<=0; decide
theorem opened_origin (i:instruction) (c:Nat) : TimeOrigin (agt_opened i c) := by
 simp [TimeOrigin,agt_opened]

theorem configure_fields (a:agt) (f:agt_field) (v:BitVec 16) :
 (agt_configure_pure a f v).1.t0=a.t0 ∧ (agt_configure_pure a f v).1.last=a.last := by
 unfold agt_configure_pure
 dsimp only [Id.run]
 repeat' split
 all_goals simp

theorem configure_origin (a:agt) (f:agt_field) (v:BitVec 16) (ha:TimeOrigin a) :
 TimeOrigin (agt_configure_pure a f v).1 := by
 have h:=configure_fields a f v
 simpa only [TimeOrigin,h.1,h.2] using ha

theorem refresh_origin (a:agt) (ha:TimeOrigin a) : TimeOrigin (agt_refresh_pure a).1 := by
 unfold agt_refresh_pure
 dsimp only [Id.run]
 split
 · split
   · exact idle_origin
   · exact ha
 · exact ha

theorem finish_origin (a:agt) (ha:TimeOrigin a) : TimeOrigin (agt_finish_config_pure a).1 :=
 refresh_origin {a with state:=.AGT_RUN} ha

theorem due_not_before_last (a:agt) (c:Nat) (parked:Bool)
 (step:Vector agt_level 4 × Nat × Int × Int)
 (hd:agt_due_pure a c parked=some step) : a.last<=c := by
 unfold agt_due_pure at hd
 split at hd
 · contradiction
 · split at hd
   · contradiction
   · split at hd
     · rename_i h
       simp only [decide_eq_true_eq] at h
       omega
     · contradiction

theorem actual_due_origin (a:agt) (c:Nat) (parked:Bool)
 (step:Vector agt_level 4 × Nat × Int × Int)
 (ha:TimeOrigin a) (hd:agt_due_pure a c parked=some step) :
 a.t0 + (max ((c:Int)-a.t0) 0).toNat=c := by
 have h:=due_not_before_last a c parked step hd
 exact ExecuteProducerProofs.agt_reference_origin a.t0 c (by unfold TimeOrigin at ha; omega)

theorem after_reference_origin (a:agt) (c:Nat) (parked:Bool)
 (ls:Vector agt_level 4) (delay:Nat) (off soff:Int)
 (ha:TimeOrigin a) (hd:agt_due_pure a c parked=some (ls,delay,off,soff)) :
 TimeOrigin (agt_after_reference a ls off soff c) := by
 have h:=due_not_before_last a c parked (ls,delay,off,soff) hd
 simpa only [TimeOrigin,agt_after_reference] using (Nat.le_trans ha h)

end AgtOriginProofs
