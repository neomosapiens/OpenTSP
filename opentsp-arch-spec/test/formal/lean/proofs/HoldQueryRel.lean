import Out
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace HoldQueryRel
abbrev St := SequentialState RegisterType trivialChoiceSource

def HoldRegs : List Register := [.agts,.agts_w,.path_busy_until,.path_tail_until,
  .queue_busy_until,.queue_repeat,.queue_poll,.parked_since,.release_at,.hw_since,.hw_release,.agt_records]
def Agree (s t : St) : Prop := ∀ r, r∈HoldRegs → s.regs.get? r=t.regs.get? r

def ResultRel {α : Type} (s t : St)
    (a b : EStateM.Result (Sail.Error exception) St α) : Prop :=
  match a,b with
  | .ok x u,.ok y v => x=y ∧ u=s ∧ v=t
  | .error e u,.error f v => e=f ∧ u=s ∧ v=t
  | _,_ => False
def Query {α : Type} (a : SailM α) : Prop :=
  ∀ s t, Agree s t → ResultRel s t (a.run s) (a.run t)

theorem pure_query {α : Type} (x : α) : Query (pure x) := by
  intro s t h
  exact ⟨rfl,rfl,rfl⟩

theorem read_query (r : Register) (hr : r∈HoldRegs) : Query (Out.readReg r) := by
  intro s t h
  simp only [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    get,getThe,MonadStateOf.get,EStateM.get]
  rw [h r hr]
  cases t.regs.get? r <;> exact ⟨rfl,rfl,rfl⟩

theorem bind_query {α β : Type} (a : SailM α) (f : α → SailM β)
    (ha : Query a) (hf : ∀x, Query (f x)) : Query (a >>= f) := by
  intro s t h
  have hp := ha s t h
  cases hsa : a.run s with
  | ok x u =>
    cases hta : a.run t with
    | ok y v =>
      have hv : x=y ∧ u=s ∧ v=t := by simpa only [hsa,hta,ResultRel] using hp
      rcases hv with ⟨hxy,hu,hv⟩
      subst y
      subst u
      subst v
      simpa only [EStateM.run,bind,EStateM.bind,show a s=.ok x s from hsa,
        show a t=.ok x t from hta] using hf x s t h
    | error e v => simp only [hsa,hta,ResultRel] at hp
  | error e u =>
    cases hta : a.run t with
    | ok y v => simp only [hsa,hta,ResultRel] at hp
    | error f v =>
      simpa only [EStateM.run,bind,EStateM.bind,show a s=.error e u from hsa,
        show a t=.error f v from hta,ResultRel] using hp

theorem forIn_query {β : Type} (range : IntRange) (b : β)
    (body : Int → β → SailM (ForInStep β)) (hb : ∀i x,Query (body i x)) :
    Query (forIn range b body) := by
  change Query (IntRange.forIn' range b (fun i _ x => body i x))
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive:=fun x i hs => Query (IntRange.forIn'.loop range (fun i _ x => body i x) x i hs))
  · intro x i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply bind_query _ _ (hb i x)
    intro step
    cases step
    · exact pure_query _
    · exact ih _
  · intro x i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact pure_query _

macro "hold_query" : tactic => `(tactic|
  repeat' first
  | exact pure_query _
  | (apply read_query; decide)
  | apply bind_query
  | apply forIn_query
  | split
  | (intro x)
  | simp [ResultRel,EStateM.run,pure,EStateM.pure])

theorem hold_view_query (q c : Nat) : Query (hold_view_of q c) := by
  simp only [hold_view_of,queue_parked,host_parked,parked_one,polling,path_views,
    path_view_of,path_busy,path_tail,agt_get,bind_assoc,pure_bind,bind_pure]
  hold_query

theorem hold_check_query (q c : Nat) (i : instruction) (path : Option Nat) (strict : Bool) :
    Query (hold_check q c i path strict) := by
  simp only [hold_check,bind_assoc,pure_bind,bind_pure]
  apply bind_query _ _ (hold_view_query q c)
  intro v
  exact pure_query _

theorem strict_held_query (q c : Nat) (items : List (Result instruction diag)) (held : Bool) :
    Query (strict_held q c items held) := by
  induction items generalizing held with
  | nil => exact pure_query _
  | cons x xs ih =>
    simp only [strict_held,bind_assoc,pure_bind,bind_pure]
    apply bind_query
    · cases x <;> exact hold_check_query _ _ _ _ _
    · intro next
      exact ih _

theorem parcel_held_query (q c : Nat) (r : cycle_recs)
    (items : List (Result instruction diag)) : Query (parcel_held q c r items) := by
  simp only [parcel_held,cfg_path,agt_get,bind_assoc,pure_bind,bind_pure]
  repeat' first
  | exact strict_held_query _ _ _ _
  | exact hold_check_query _ _ _ _ _
  | exact pure_query _
  | (apply read_query; decide)
  | apply bind_query
  | apply forIn_query
  | split
  | (intro x)
  | simp [ResultRel,EStateM.run,pure,EStateM.pure]

theorem successful_value_transfer {α : Type} (a : SailM α) (ha : Query a)
    (s t : St) (h : Agree s t) (x : α) (hx : a.run s=.ok x s) :
    a.run t=.ok x t := by
  have hp := ha s t h
  cases ht : a.run t with
  | ok y v =>
    have hv : x=y ∧ s=s ∧ v=t := by simpa only [hx,ht,ResultRel] using hp
    rcases hv with ⟨hxy,_,hvt⟩
    subst y
    subst v
    rfl
  | error e v => simp only [hx,ht,ResultRel] at hp

#print axioms strict_held_query
#print axioms parcel_held_query
#print axioms successful_value_transfer

#print axioms pure_query
#print axioms read_query
#print axioms bind_query
#print axioms forIn_query
#print axioms hold_view_query
#print axioms hold_check_query
end HoldQueryRel
