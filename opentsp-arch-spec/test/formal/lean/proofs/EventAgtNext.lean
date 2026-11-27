import Proofs.EventUnparked
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventUnparkedProofs
set_option autoImplicit false
set_option maxHeartbeats 2000000
open agt_state
namespace EventAgtNextProofs

def NextSound (a : agt) : Prop := a.state=AGT_RUN →
  ∃ls d st ss, agt_advance a.levels (agt_nlevels a)=some (ls,d,st,ss) ∧ a.next=a.last+d

theorem idle_next_sound : NextSound agt_idle := by
  simp [NextSound,agt_idle]

theorem opened_next_sound (i : instruction) (c : Nat) : NextSound (agt_opened i c) := by
  simp [NextSound,agt_opened,agt_idle]

theorem refresh_next_sound (a : agt) : NextSound (agt_refresh_pure a).1 := by
  unfold agt_refresh_pure
  dsimp only [Id.run]
  split
  · split
    · exact idle_next_sound
    · rename_i ls d st ss ha
      unfold NextSound
      intro _
      refine ⟨ls,d,st,ss,ha,?_⟩
      change ((a.last:Int)+(d:Int)).toNat=a.last+d
      omega
  · unfold NextSound
    intro hr
    change a.state=AGT_RUN at hr
    have hbeq : (a.state==AGT_RUN)=true := by rw [hr]; decide
    contradiction

theorem finish_next_sound (a : agt) : NextSound (agt_finish_config_pure a).1 := by
  exact refresh_next_sound _

theorem due_implies_frontier (a : agt) (t : Nat) (parked : Bool)
    (hs : NextSound a) (hr:a.state=AGT_RUN)
    (out : Vector agt_level 4 × Nat × Int × Int)
    (hd : agt_due_pure a t parked=some out) : a.next≤t := by
  obtain ⟨ls,d,st,ss,ha,he⟩:=hs hr
  simp only [agt_due_pure,ha] at hd
  split at hd
  · contradiction
  · split at hd
    · have hle : (a.last:Int)+(d:Int)≤(t:Int) := by simpa only [decide_eq_true_eq] using ‹((a.last+i d) ≤b t)=true›
      omega
    · contradiction

theorem configure_state (a : agt) (f : agt_field) (v : BitVec 16) :
    (agt_configure_pure a f v).1.state=a.state := by
  unfold agt_configure_pure
  dsimp only [Id.run]
  repeat' split
  all_goals simp

theorem configure_next_sound (a : agt) (f : agt_field) (v : BitVec 16)
    (hc:a.state=AGT_CONFIG) : NextSound (agt_configure_pure a f v).1 := by
  unfold NextSound
  rw [configure_state,hc]
  simp

-- Every actual production cell writer, with the checked dispatch guard explicit.
inductive CellStep : agt→agt→Prop where
  | keep (a : agt) : CellStep a a
  | clear (a : agt) : CellStep a agt_idle
  | opened (a : agt) (i : instruction) (c : Nat) : CellStep a (agt_opened i c)
  | configure (a : agt) (f : agt_field) (v : BitVec 16) (hc:a.state=AGT_CONFIG) :
      CellStep a (agt_configure_pure a f v).1
  | finish (a : agt) : CellStep a (agt_finish_config_pure a).1
  | reference (a : agt) (ls : Vector agt_level 4) (off soff : Int) (c : Nat) :
      CellStep a (agt_refresh_pure (agt_after_reference a ls off soff c)).1

theorem step_next_sound (a b : agt) (h : CellStep a b) (ha:NextSound a) : NextSound b := by
  cases h with
  | keep=>exact ha
  | clear=>exact idle_next_sound
  | opened=>exact opened_next_sound _ _
  | configure f v hc=>exact configure_next_sound _ f v hc
  | finish=>exact finish_next_sound _
  | reference=>exact refresh_next_sound _

inductive CellHistory : agt→Prop where
  | reset : CellHistory agt_idle
  | step (a b : agt) : CellHistory a→CellStep a b→CellHistory b

theorem history_next_sound (a : agt) (h:CellHistory a) : NextSound a := by
  induction h with
  | reset=>exact idle_next_sound
  | step a b h st ih=>exact step_next_sound a b st ih

theorem history_due_captured (a : agt) (h:CellHistory a) (t : Nat) (parked : Bool)
    (hr:a.state=AGT_RUN) (out : Vector agt_level 4 × Nat × Int × Int)
    (hd:agt_due_pure a t parked=some out) : a.next≤t :=
  due_implies_frontier a t parked (history_next_sound a h) hr out hd

#print axioms configure_state
#print axioms configure_next_sound
#print axioms step_next_sound
#print axioms history_next_sound
#print axioms history_due_captured

#print axioms idle_next_sound
#print axioms opened_next_sound
#print axioms refresh_next_sound
#print axioms finish_next_sound
#print axioms due_implies_frontier
end EventAgtNextProofs
