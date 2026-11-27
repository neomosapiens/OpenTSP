-- Frames of actual event-query actions, retaining callback obligations explicitly.
import Proofs.Events
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
set_option autoImplicit false
namespace EventFrameProofs
abbrev St := SequentialState RegisterType trivialChoiceSource

def Frame {α : Type} (action : SailM α) : Prop :=
  ∀ st a st', action.run st = EStateM.Result.ok a st' → st'=st

theorem frame_pure {α : Type} (a : α) : Frame (pure a) := by
  intro st b st' h
  simp [EStateM.run, pure, EStateM.pure] at h
  exact h.2.symm

theorem frame_bind {α β : Type} (action : SailM α) (next : α → SailM β)
    (ha : Frame action) (hn : ∀ a, Frame (next a)) : Frame (action >>= next) := by
  intro st b st' h
  simp only [EStateM.run, bind, EStateM.bind] at h
  cases hs : action st with
  | error error state => simp [hs] at h
  | ok a state =>
    simp only [hs] at h
    have he := ha st a state hs
    have hf := hn a state b st' h
    exact hf.trans he

theorem frame_read (reg : Register) : Frame (Out.readReg reg) := by
  intro st a st' h
  simp [Out.readReg, PreSail.readReg, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get] at h
  cases he : st.regs.get? reg <;> simp_all [EStateM.pure, throw, throwThe, MonadExceptOf.throw, EStateM.throw]

theorem frame_forIn {β : Type} (range : IntRange) (b : β)
    (f : Int → β → SailM (ForInStep β))
    (hf : ∀ i x, Frame (f i x)) : Frame (forIn range b f) := by
  change Frame (IntRange.forIn' range b (fun i _ x => f i x))
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun x i hs => Frame (IntRange.forIn'.loop range (fun i _ x => f i x) x i hs))
  · intro x i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply frame_bind _ _ (hf i x)
    intro step
    cases step with
    | done y => exact frame_pure y
    | yield y => exact ih y
  · intro x i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact frame_pure x

attribute [irreducible] Frame

theorem next_boundary_frame (c : Nat) : Frame (next_boundary c) := by
  simp only [next_boundary]
  exact frame_pure _

#print axioms frame_pure
#print axioms frame_bind
#print axioms frame_read
#print axioms frame_forIn
#print axioms next_boundary_frame

theorem unparked_from_frame (q t : Nat) : Frame (unparked_from q t) := by
  simp only [unparked_from, bind_assoc, pure_bind, bind_pure]
  apply frame_forIn
  intro pass x
  apply frame_bind
  · apply frame_forIn
    intro j x
    apply frame_bind
    · exact frame_read _
    · intro parks
      apply frame_bind
      · exact frame_read _
      · intro releases
        split
        all_goals first | exact frame_pure _ | skip
        all_goals
          apply frame_bind
          · split <;> exact frame_pure _
          · intro y
            exact frame_pure _
  · intro y
    exact frame_pure _
#print axioms unparked_from_frame

theorem held_frame (q : Nat) : Frame (held q) := by
  simp only [held, bind_assoc, pure_bind, bind_pure]
  apply frame_forIn
  intro j x
  apply frame_bind _ _ (frame_read _)
  intro parks
  apply frame_bind _ _ (frame_read _)
  intro releases
  split <;> exact frame_pure _

theorem parked_one_frame (q c : Nat) : Frame (parked_one q c) := by
  simp only [parked_one, bind_assoc, pure_bind, bind_pure]
  apply frame_bind _ _ (frame_read _)
  intro parks
  apply frame_bind _ _ (frame_read _)
  intro releases
  exact frame_pure _

theorem queue_parked_frame (q c : Nat) : Frame (queue_parked q c) := by
  simp only [queue_parked, bind_assoc, pure_bind, bind_pure]
  apply frame_bind _ _ (parked_one_frame q c)
  intro a
  apply frame_bind _ _ (parked_one_frame 144 c)
  intro b
  exact frame_pure _

theorem c2c_link_frame (link : BitVec 7) : Frame (c2c_link link) := by
  simp only [c2c_link]
  split <;> exact frame_pure _

theorem c2c_link_bound (link : BitVec 7) :
    EventProofs.Post (fun result => ∀ l, result=some l → l<128) (c2c_link link) := by
  simp only [c2c_link]
  split
  · apply EventProofs.post_pure
    intro l hl
    cases hl
    have hh := link.isLt
    simpa [BitVec.toNatInt] using hh
  · apply EventProofs.post_pure
    intro l hl
    cases hl

theorem frame_bind_post {α β : Type} (P : α → Prop) (action : SailM α) (next : α → SailM β)
    (ha : Frame action) (hp : EventProofs.Post P action) (hn : ∀ a, P a → Frame (next a)) :
    Frame (action >>= next) := by
  unfold Frame at *
  unfold EventProofs.Post at hp
  intro st b st' h
  simp only [EStateM.run, bind, EStateM.bind] at h
  cases hs : action st with
  | error error state => simp [hs] at h
  | ok a state =>
    simp only [hs] at h
    exact (hn a (hp st a state hs) state b st' h).trans (ha st a state hs)

theorem deskewrecv_next_arrival_frame (link : BitVec 7)
    (hcallback : ∀ l, l < 128 → Frame (c2c_next_arrival l)) : Frame (deskewrecv_next_arrival link) := by
  simp only [deskewrecv_next_arrival, bind_assoc, pure_bind, bind_pure]
  apply frame_bind_post _ _ _ (c2c_link_frame link) (c2c_link_bound link)
  intro result hresult
  cases result with
  | none => exact frame_pure _
  | some l => exact hcallback l (hresult l rfl)

theorem model_poll_event_frame
    (hcallback : ∀ l, l < 128 → Frame (c2c_next_arrival l)) : Frame (model_poll_event ()) := by
  simp only [model_poll_event, bind_assoc, pure_bind, bind_pure]
  apply frame_forIn
  intro q best
  apply frame_bind _ _ (frame_read _)
  intro polls
  split
  · apply frame_bind _ _ (deskewrecv_next_arrival_frame _ hcallback)
    intro arrival
    apply frame_bind _ _ (held_frame q.toNat)
    intro held
    split
    · simp only [bind_assoc]
      apply frame_bind _ _ (frame_read _)
      intro last_tick
      apply frame_bind _ _ (next_boundary_frame _)
      intro boundary
      apply frame_bind _ _ (queue_parked_frame _ _)
      intro parked
      apply frame_bind
      · split
        · apply frame_bind _ _ (unparked_from_frame _ _)
          intro unparked
          exact next_boundary_frame _
        · exact frame_pure _
      · intro boundary
        exact frame_pure _
    · exact frame_pure _
  · exact frame_pure _

#print axioms c2c_link_bound
#print axioms frame_bind_post
#print axioms held_frame
#print axioms parked_one_frame
#print axioms queue_parked_frame
#print axioms c2c_link_frame
#print axioms deskewrecv_next_arrival_frame
#print axioms model_poll_event_frame

def Spec {α : Type} (P : St → Prop) (Q : α → Prop) (action : SailM α) : Prop :=
  ∀ st, P st → ∀ a st', action.run st = EStateM.Result.ok a st' → P st' ∧ Q a

theorem spec_pure {α : Type} (P : St → Prop) (Q : α → Prop) (a : α) (ha : Q a) :
    Spec P Q (pure a) := by
  intro st hp b st' h
  simp [EStateM.run, pure, EStateM.pure] at h
  rcases h with ⟨rfl,rfl⟩
  exact ⟨hp,ha⟩

theorem spec_frame {α : Type} (P : St → Prop) (action : SailM α) (hf : Frame action) :
    Spec P (fun _ => True) action := by
  unfold Frame at hf
  intro st hp a st' h
  exact ⟨by simpa [hf st a st' h] using hp, trivial⟩

theorem spec_post {α : Type} (P : St → Prop) (Q : α → Prop) (action : SailM α)
    (hf : Frame action) (hv : EventProofs.Post Q action) : Spec P Q action := by
  unfold Frame at hf
  unfold EventProofs.Post at hv
  intro st hp a st' h
  exact ⟨by simpa [hf st a st' h] using hp, hv st a st' h⟩

theorem spec_bind {α β : Type} (P : St → Prop) (Q : α → Prop) (R : β → Prop)
    (action : SailM α) (next : α → SailM β)
    (ha : Spec P Q action) (hn : ∀ a, Q a → Spec P R (next a)) : Spec P R (action >>= next) := by
  intro st hp b st' h
  simp only [EStateM.run, bind, EStateM.bind] at h
  cases hs : action st with
  | error error state => simp [hs] at h
  | ok a state =>
    simp only [hs] at h
    obtain ⟨hstate,ha⟩ := ha st hp a state hs
    exact hn a ha state hstate b st' h

theorem spec_forIn {β : Type} (range : IntRange) (P : St → Prop) (Q : β → Prop) (b : β)
    (f : Int → β → SailM (ForInStep β)) (hb : Q b)
    (hf : ∀ i x, Q x → Spec P (fun step => match step with | .done y => Q y | .yield y => Q y) (f i x)) :
    Spec P Q (forIn range b f) := by
  change Spec P Q (IntRange.forIn' range b (fun i _ x => f i x))
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun x i hs => Q x → Spec P Q (IntRange.forIn'.loop range (fun i _ x => f i x) x i hs))
  · intro x i hs hi ih hx
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply spec_bind P _ Q _ _ (hf i x hx)
    intro step hstep
    cases step with
    | done y => exact spec_pure P Q y hstep
    | yield y => exact ih y hstep
  · intro x i hs hi hx
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact spec_pure P Q x hx
  · exact hb

attribute [irreducible] Spec

def TickValue (t : Nat) (st : St) : Prop := st.regs.get? Register.last_tick = some t

theorem read_last_tick_spec (t : Nat) : Spec (TickValue t) (fun v => v=t) (Out.readReg Register.last_tick) := by
  unfold Spec
  intro st hp a st' h
  dsimp [TickValue] at hp ⊢
  simp [Out.readReg, PreSail.readReg, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get,hp] at h
  rcases h with ⟨rfl,rfl⟩
  exact ⟨hp,rfl⟩

theorem next_boundary_lower (c : Nat) : EventProofs.Post (fun b => c ≤ b) (next_boundary c) := by
  simp only [next_boundary]
  apply EventProofs.post_pure
  change c ≤ ((c : Int) + Int.emod (c2c_hac_period - Int.emod (c : Int) c2c_hac_period) c2c_hac_period).toNat
  have hnonnegative := Int.emod_nonneg (c2c_hac_period - Int.emod (c : Int) c2c_hac_period)
    (by decide : (c2c_hac_period : Int) ≠ 0)
  have hsum : (c : Int) ≤ (c : Int) + Int.emod (c2c_hac_period - Int.emod (c : Int) c2c_hac_period) c2c_hac_period :=
    Int.le_add_of_nonneg_right hnonnegative
  simpa using Int.toNat_le_toNat hsum

#print axioms spec_pure
#print axioms spec_frame
#print axioms spec_post
#print axioms spec_bind
#print axioms spec_forIn
#print axioms read_last_tick_spec
#print axioms next_boundary_lower

theorem model_poll_event_future (t : Nat)
    (hcallback : ∀ l, l < 128 → Frame (c2c_next_arrival l)) :
    Spec (TickValue t) (EventProofs.Future t) (model_poll_event ()) := by
  simp only [model_poll_event, bind_assoc, pure_bind, bind_pure]
  apply spec_forIn _ (TickValue t) (EventProofs.Future t) (-1)
  · left; omega
  · intro q best hb
    apply spec_bind (TickValue t) (fun _ => True)
    · exact spec_frame _ _ (frame_read _)
    · intro polls _
      split
      · apply spec_bind (TickValue t) (fun _ => True)
        · exact spec_frame _ _ (deskewrecv_next_arrival_frame _ hcallback)
        · intro arrival _
          apply spec_bind (TickValue t) (fun _ => True)
          · exact spec_frame _ _ (held_frame q.toNat)
          · intro held _
            split
            · simp only [bind_assoc]
              apply spec_bind (TickValue t) (fun v => v=t) _ _ _ (read_last_tick_spec t)
              intro last_tick heq
              subst last_tick
              apply spec_bind (TickValue t) (fun b => t < b)
              · apply spec_post _ _ _ (next_boundary_frame _)
                apply EventProofs.post_imp _ _ _ (next_boundary_lower _)
                intro boundary hbound
                have hh := Int.le_max_right (↑arrival.toNat) ((t:Int)+1)
                split at hbound <;> simp_all <;> omega
              · intro boundary hboundary
                apply spec_bind (TickValue t) (fun _ => True)
                · exact spec_frame _ _ (queue_parked_frame _ _)
                · intro parked _
                  apply spec_bind (TickValue t) (fun b => t < b)
                  · split
                    · apply spec_bind (TickValue t) (fun u => boundary ≤ u)
                      · exact spec_post _ _ _ (unparked_from_frame _ _) (EventProofs.unparked_from_mono _ _)
                      · intro unparked hu
                        apply spec_post _ _ _ (next_boundary_frame _)
                        apply EventProofs.post_imp _ _ _ (next_boundary_lower _)
                        intro b hb
                        omega
                    · exact spec_pure _ _ boundary hboundary
                  · intro final hf
                    apply spec_pure
                    exact EventProofs.earliest_future t _ _ hb (by right; omega)
            · exact spec_pure _ _ _ hb
      · exact spec_pure _ _ _ hb
#print axioms model_poll_event_future
end EventFrameProofs
