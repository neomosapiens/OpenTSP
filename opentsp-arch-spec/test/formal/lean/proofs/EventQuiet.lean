-- Register/memory-state identities for inactive production phases. Pure trace sinks
-- remain erased in this interpretation; these are not ordered-output or whole-tick claims.
import Proofs.EventCensus
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventProofs EventCensusProofs
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace EventQuietProofs

theorem insert_existing (s : St) (r : Register) (v : RegisterType r)
    (hr : s.regs.get? r = some v) : s.regs.insert r v = s.regs := by
  apply Std.ExtDHashMap.ext_get?
  intro k
  by_cases he : k=r
  · subst k
    simp [hr]
  · simp [Std.ExtDHashMap.get?_insert,he,Ne.symm he]

theorem vector_update_existing {α : Type} [Inhabited α] {n : Nat}
    (v : Vector α n) (i : Nat) (hi : i<n) : vectorUpdate v i v[i]! = v := by
  simpa [vectorUpdate,Vector.set!,Array.setIfInBounds_def,hi] using
    Array.set_getElem_self (xs:=v.toArray) (i:=i) (by simpa using hi)

theorem execute_due_empty (s : St) (c : Nat) (v : Vector (List exec_element) 16)
    (hr : s.regs.get? Register.execute_ring = some v) (he : v[c%16]! = []) :
    (execute_due c).run s = .ok () s := by
  have hu : vectorUpdate v (c%16) [] = v := by
    rw [←he]
    exact vector_update_existing v (c%16) (Nat.mod_lt _ (by decide))
  have hmod : SailTrunc.tmod c 16 = c % 16 := rfl
  have hm := insert_existing s Register.execute_ring v hr
  simp [execute_due,execute_all,tmod_int,Out.readReg,PreSail.readReg,Out.writeReg,
    PreSail.writeReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,
    MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,
    hmod,hr,he,hu,hm]

theorem register_value_of_read (s : St) (r : Register) (v : RegisterType r)
    (hr : (Out.readReg r).run s = .ok v s) : s.regs.get? r = some v := by
  simp only [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get] at hr
  cases h : s.regs.get? r <;>
    simp_all [throw,throwThe,MonadExceptOf.throw,EStateM.throw,EStateM.pure]

theorem execute_phase_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) : (execute_due t).run s = .ok () s := by
  have empty := execute_ring_empty_before_frontier s after v c t frontier hr hct ht
  have he : v.executes[t%16]! = [] := by simpa using empty
  exact execute_due_empty s t v.executes
    (register_value_of_read s Register.execute_ring v.executes v.execute_read) he

theorem empty_inclusive {β : Type} (b : β) (body : Int → β → SailM (ForInStep β)) :
    forIn [1:0:1]i b body = pure b := by
  change IntRange.forIn' ({start:=1,stop:=0} : IntRange) b
    (fun i _ x => body i x) = pure b
  unfold IntRange.forIn'
  rw [IntRange.forIn'.loop.eq_def]
  simp [Membership.mem]

theorem zero_vector_access (q : Int) :
    ((vectorInit 0 : Vector Nat 145)[q]!) = 0 := by
  change ((Vector.replicate 145 0)[q.toNat]!) = 0
  by_cases h : q < 145 <;> simp_all [getElem!_def,Vector.getElem?_replicate]

theorem unit_forIn (range : IntRange) (body : Int → Unit → SailM (ForInStep Unit))
    (hb : ∀ i x, body i x = pure (.yield ())) : forIn range () body = pure () := by
  funext s
  have hf := fixed_state_forIn range () s (fun _ _ => ()) body
    (by intro i x; rw [hb]; rfl)
  have hv : Id.run (forIn range () (fun (_ : Int) (_ : Unit) =>
    (pure (.yield ()) : Id (ForInStep Unit)))) = () := Subsingleton.elim _ _
  simpa only [hv,EStateM.run,pure,EStateM.pure] using hf

theorem commit_due_empty (s : St) (c : Nat) (v : Vector (List commit) 128)
    (hr : s.regs.get? Register.pending = some v) (he : v[c%128]! = []) :
    (commit_due c).run s = .ok () s := by
  have hu : vectorUpdate v (c%128) [] = v := by
    rw [←he]
    exact vector_update_existing v (c%128) (Nat.mod_lt _ (by decide))
  have hmod : SailTrunc.tmod c 128 = c % 128 := rfl
  have hm := insert_existing s Register.pending v hr
  simp [commit_due,commit_apply,tmod_int,Out.readReg,PreSail.readReg,Out.writeReg,
    PreSail.writeReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,
    MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,
    hmod,hr,he,hu,hm,zero_vector_access,empty_inclusive,unit_forIn]
  rw [unit_forIn _ _ (by intro i x; cases x; rfl)]
  simp [pure,EStateM.pure,hr,hu,hm]

theorem commit_phase_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) : (commit_due t).run s = .ok () s := by
  have empty := commit_ring_empty_before_frontier s after v c t frontier hr hct ht
  have he : v.commits[t%128]! = [] := by simpa using empty
  exact commit_due_empty s t v.commits
    (register_value_of_read s Register.pending v.commits v.commit_read) he

theorem bounded_unit_forIn (range : IntRange) (s : St)
    (body : Int → Unit → SailM (ForInStep Unit))
    (hb : ∀ i, i ∈ range → ∀ x, (body i x).run s = .ok (.yield ()) s) :
    (forIn range () body).run s = .ok () s := by
  change (IntRange.forIn' range () (fun i _ x => body i x)).run s = .ok () s
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive:=fun x i hs =>
      (IntRange.forIn'.loop range (fun i _ x => body i x) x i hs).run s = .ok () s)
  · intro x i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    have he : body i x s = .ok (.yield ()) s := hb i hi x
    simpa only [EStateM.run,bind,EStateM.bind,he] using ih ()
  · intro x i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    cases x
    simp [hi,EStateM.run,pure,EStateM.pure]

theorem stream_fetch_take_quiet (s : St) (c q : Nat)
    (v : Vector (Option stream_fetch) 145)
    (hr : s.regs.get? Register.stream_fetches = some v)
    (hf : ∀ f, v[q]! = some f → c<f.ready) :
    (stream_fetch_take q c).run s = .ok none s := by
  cases h : v[q]! with
  | none =>
    simp [stream_fetch_take,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr,h]
  | some f =>
    have hready := hf f h
    simp [stream_fetch_take,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr,h,show ¬f.ready≤c by omega]

theorem stream_fetch_service_quiet (s : St) (c q : Nat)
    (v : Vector (Option stream_fetch) 145)
    (hr : s.regs.get? Register.stream_fetches = some v)
    (hf : ∀ f, v[q]! = some f → c<f.ready) :
    (stream_fetch_service q c).run s = .ok () s := by
  have ht : stream_fetch_take q c s = .ok none s := stream_fetch_take_quiet s c q v hr hf
  simp [stream_fetch_service,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,ht]

theorem stream_phase_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) : (stream_fetch_complete t).run s = .ok () s := by
  simp only [stream_fetch_complete,bind_pure,pure_bind]
  apply bounded_unit_forIn
  intro i hi u
  have bounds : 0≤i ∧ i≤143 := by simpa [Membership.mem] using hi
  have cast : (i.toNat:Int)=i := by omega
  have hq : i.toNat<144 := by omega
  have hreg := register_value_of_read s Register.stream_fetches v.streams v.stream_read
  have hs : stream_fetch_service i.toNat t s = .ok () s :=
    stream_fetch_service_quiet s t i.toNat v.streams hreg (by
      intro f hf
      apply stream_not_due_before_frontier s after v c t i.toNat frontier f hr hct ht hq
      change v.streams[i.toNat]! = some f
      exact hf)
  cases u
  simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hs]

theorem local_phase_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) : (local_fetch_complete t).run s = .ok () s := by
  simp only [local_fetch_complete,bind_pure,pure_bind]
  apply bounded_unit_forIn
  intro i hi u
  have bounds : 0≤i ∧ i≤87 := by simpa [Membership.mem] using hi
  have cast : (i.toNat:Int)=i := by omega
  have hq : i.toNat<88 := by omega
  have hreg := register_value_of_read s Register.local_fetches v.locals v.local_read
  cases u
  cases h : v.locals[i]! with
  | none =>
    simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hreg,h]
  | some f =>
    have hf : v.locals[(i.toNat:Int)]! = some f := by simpa [cast] using h
    have hready := local_not_due_before_frontier s after v c t i.toNat frontier f hr hct ht hq hf
    simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hreg,h,show ¬f.ready≤t by omega]

theorem enqueue_quiet (s : St) (c q : Nat) (v : Vector (List fill_seg) 145)
    (hr : s.regs.get? Register.fill_fifo = some v)
    (hf : ∀ seg rest, v[q]! = seg::rest → c<seg.ready) :
    (enqueue q c).run s = .ok () s := by
  cases h : v[q]! with
  | nil =>
    simp [enqueue,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr,h]
  | cons seg rest =>
    have hready := hf seg rest h
    simp [enqueue,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr,h,show ¬seg.ready≤c by omega]

theorem enqueue_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t q : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) (hq : q<145) :
    (enqueue q t).run s = .ok () s := by
  apply enqueue_quiet s t q v.fills
    (register_value_of_read s Register.fill_fifo v.fills v.fill_read)
  intro seg rest hf
  apply fill_not_due_before_frontier s after v c t q frontier seg rest hr hct ht hq
  change v.fills[q]! = seg::rest
  exact hf

def enqueue_all (c : Nat) : SailM Unit := do
  let mut u := ()
  for q in [0:144:1]i do
    let () := u
    u ← enqueue q.toNat c
  pure u

theorem enqueue_phase_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) :
    (enqueue_all t).run s = .ok () s := by
  simp only [enqueue_all,bind_pure,pure_bind]
  apply bounded_unit_forIn
  intro i hi u
  have bounds : 0≤i ∧ i≤144 := by simpa [Membership.mem] using hi
  have hq : i.toNat<145 := by omega
  have hs : enqueue i.toNat t s = .ok () s :=
    enqueue_before_frontier s after v c t i.toNat frontier hr hct ht hq
  cases u
  simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hs]

#print axioms enqueue_quiet
#print axioms enqueue_before_frontier
#print axioms enqueue_phase_before_frontier

#print axioms stream_fetch_service_quiet
#print axioms stream_phase_before_frontier
#print axioms local_phase_before_frontier

#print axioms commit_phase_before_frontier
#print axioms bounded_unit_forIn
#print axioms stream_fetch_take_quiet

#print axioms zero_vector_access
#print axioms unit_forIn
#print axioms commit_due_empty

#print axioms register_value_of_read
#print axioms execute_phase_before_frontier
#print axioms empty_inclusive

#print axioms insert_existing
#print axioms vector_update_existing
#print axioms execute_due_empty
end EventQuietProofs
