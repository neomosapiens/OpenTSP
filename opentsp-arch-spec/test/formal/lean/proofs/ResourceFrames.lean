-- Native effect composition for actual resource projections, including error states.
import Proofs.Resources
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace ResourceFrameProofs
abbrev St := SequentialState RegisterType trivialChoiceSource

def Keeps {α β : Type} (view : St → β) (action : SailM α) : Prop :=
  ∀ s, match action.run s with
    | .ok _ after => view after=view s
    | .error _ after => view after=view s

def RegKeeps {α : Type} (r : Register) (action : SailM α) : Prop :=
  Keeps (fun s => s.regs.get? r) action

theorem pure_keeps {α β : Type} (view : St → β) (x : α) : Keeps view (pure x) := by intro s; rfl

theorem bind_keeps {α β γ : Type} (view : St → γ) (a : SailM α) (f : α → SailM β)
    (ha : Keeps view a) (hf : ∀ x, Keeps view (f x)) : Keeps view (a >>= f) := by
  intro s
  cases h : a.run s with
  | ok x after =>
    have h1 := ha s
    have h2 := hf x after
    simp only [h] at h1
    cases hnext : (f x).run after with
    | ok y final =>
      simp only [hnext] at h2
      simpa [EStateM.run,bind,EStateM.bind,show a s = .ok x after from h,
        show f x after = .ok y final from hnext] using h2.trans h1
    | error err final =>
      simp only [hnext] at h2
      simpa [EStateM.run,bind,EStateM.bind,show a s = .ok x after from h,
        show f x after = .error err final from hnext] using h2.trans h1
  | error error after =>
    have h1 := ha s
    simpa [h,EStateM.run,bind,EStateM.bind,show a s = .error error after from h] using h1

theorem read_keeps {β : Type} (view : St → β) (r : Register) : Keeps view (Out.readReg r) := by
  intro s
  simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    get,getThe,MonadStateOf.get,EStateM.get]
  cases s.regs.get? r <;> rfl

theorem write_keeps {β : Type} (view : St → β) (r : Register) (v : RegisterType r)
    (h : ∀ s, view {s with regs := s.regs.insert r v}=view s) : Keeps view (Out.writeReg r v) := by
  intro s
  exact h s

theorem unrelated_register_write (r w : Register) (v : RegisterType w) (hne : r≠w) :
    RegKeeps r (Out.writeReg w v) := by
  apply write_keeps
  intro s
  simp [Std.ExtDHashMap.get?_insert,hne,Ne.symm hne]

theorem ite_keeps {α β : Type} (view : St → β) (p : Prop) [Decidable p] (a b : SailM α)
    (ha : Keeps view a) (hb : Keeps view b) : Keeps view (if p then a else b) := by
  split <;> assumption

theorem forIn_keeps {α β : Type} (view : St → β) (range : IntRange) (value : α)
    (f : (i : Int) → i ∈ range → α → SailM (ForInStep α))
    (hf : ∀ i hi x, Keeps view (f i hi x)) : Keeps view (IntRange.forIn' range value f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => Keeps view (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply bind_keeps _ _ _ (hf i hi value)
    intro step
    cases step with
    | done value => exact pure_keeps view value
    | yield value => exact ih value
  · intro value i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact pure_keeps view value

/-- Only the real external raw callback is trusted to leave this Sail register
    unchanged; actual revision reads/writes are composed here, not abstracted away. -/
theorem write_sram_word_other_reg (r : Register) (hne : r≠Register.sram_revision)
    (slice addr : Nat) (value : vec)
    (port : RegKeeps r (raw_write_sram_word slice addr value)) :
    RegKeeps r (write_sram_word slice addr value) := by
  unfold write_sram_word
  apply bind_keeps _ _ _ (read_keeps _ _)
  intro a
  apply bind_keeps _ _ _ (read_keeps _ _)
  intro b
  apply bind_keeps _ _ _ (unrelated_register_write r Register.sram_revision _ hne)
  intro u
  exact port

theorem write_sram_ecc_other_reg (r : Register) (hne : r≠Register.sram_revision)
    (slice addr : Nat) (value : ecc_ov)
    (port : RegKeeps r (raw_write_sram_ecc slice addr value)) :
    RegKeeps r (write_sram_ecc slice addr value) := by
  unfold write_sram_ecc
  apply bind_keeps _ _ _ (read_keeps _ _)
  intro a
  apply bind_keeps _ _ _ (read_keeps _ _)
  intro b
  apply bind_keeps _ _ _ (unrelated_register_write r Register.sram_revision _ hne)
  intro u
  exact port
def FieldView {β : Type} (r : Register) (pick : RegisterType r → β) (s : St) : Option β :=
  (s.regs.get? r).map pick

theorem unrelated_register_field_write {β : Type} (r w : Register) (pick : RegisterType r → β)
    (value : RegisterType w) (hne : r≠w) : Keeps (FieldView r pick) (Out.writeReg w value) := by
  apply write_keeps
  intro s
  simp [FieldView,Std.ExtDHashMap.get?_insert,Ne.symm hne]

theorem read_missing_run (s : St) (r : Register) (h : s.regs.get? r=none) :
    (Out.readReg r).run s=EStateM.Result.error Error.Unreachable s := by
  simp only [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    get,getThe,MonadStateOf.get,EStateM.get,h]
  rfl

theorem update_field_keeps {β : Type} (r : Register) (pick : RegisterType r → β)
    (update : RegisterType r → RegisterType r) (h : ∀ old, pick (update old)=pick old) :
    Keeps (FieldView r pick) (do let old ← Out.readReg r; Out.writeReg r (update old)) := by
  intro s
  cases hr : s.regs.get? r with
  | none =>
    have hm := read_missing_run s r hr
    change Out.readReg r s = .error Error.Unreachable s at hm
    simp only [EStateM.run,bind,EStateM.bind]
    rw [hm]
  | some old => simp [FieldView,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
      EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,throw,throwThe,EStateM.throw,EStateM.instMonadExceptOfOfBacktrackable,hr,
      modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,h]

theorem vector_update_other {α : Type} [Inhabited α] (n i j : Nat) (v : Vector α n) (x : α)
    (hi : i<n) (hj : j<n) (hne : i≠j) : (vectorUpdate v j x)[i]! = v[i]! := by
  simp [vectorUpdate,Vector.set!,hi,hj,hne,Ne.symm hne]

/-- A value built only by actual state reads can be inserted at another cell
    without accidentally assuming an arbitrary callback preserved the whole state. -/
theorem read_value_update_field_keeps {α β : Type} (r : Register) (pick : RegisterType r → β)
    (value : SailM α) (build : RegisterType r → α → RegisterType r)
    (hv : Keeps (fun s : St => s) value) (hb : ∀ old x, pick (build old x)=pick old) :
    Keeps (FieldView r pick) (do let old ← Out.readReg r; let x ← value; Out.writeReg r (build old x)) := by
  intro s
  cases hr : s.regs.get? r with
  | none =>
    have hm := read_missing_run s r hr
    change Out.readReg r s = .error Error.Unreachable s at hm
    simp only [EStateM.run,bind,EStateM.bind]
    rw [hm]
  | some old =>
    have hv0 := hv s
    cases hx : value.run s with
    | error err after =>
      have hs : after=s := by simpa [hx] using hv0
      subst after
      simp [FieldView,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
        get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,throw,throwThe,EStateM.throw,EStateM.instMonadExceptOfOfBacktrackable,hr,show value s = .error err s from hx]
    | ok x after =>
      have hs : after=s := by simpa [hx] using hv0
      subst after
      simp [FieldView,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
        EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,throw,throwThe,EStateM.throw,EStateM.instMonadExceptOfOfBacktrackable,hr,
        show value s = .ok x s from hx,modify,modifyGet,MonadStateOf.modifyGet,
        EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hb]

theorem reg_keeps_field {α β : Type} (r : Register) (pick : RegisterType r → β)
    (action : SailM α) (h : RegKeeps r action) : Keeps (FieldView r pick) action := by
  intro s
  have hs := h s
  cases hr : action.run s <;>
    simp only [RegKeeps,Keeps,hr] at hs <;> simpa [FieldView] using congrArg (Option.map pick) hs

theorem read_twice_update_field_keeps {β : Type} (r : Register) (pick : RegisterType r → β)
    (update : RegisterType r → RegisterType r → RegisterType r)
    (h : ∀ old, pick (update old old)=pick old) :
    Keeps (FieldView r pick) (do let old ← Out.readReg r; let seen ← Out.readReg r; Out.writeReg r (update old seen)) := by
  intro s
  cases hr : s.regs.get? r with
  | none =>
    have hm := read_missing_run s r hr
    change Out.readReg r s = .error Error.Unreachable s at hm
    simp only [EStateM.run,bind,EStateM.bind]
    rw [hm]
  | some old =>
    simp [FieldView,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
      EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,hr,
      modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,h]

theorem actual_word_write_other_revision_cell (q slice addr : Nat) (hq : q<88) (hs : slice<88)
    (hne : q≠slice) (value : vec)
    (port : Keeps (FieldView Register.sram_revision (fun v => v[q]!)) (raw_write_sram_word slice addr value)) :
    Keeps (FieldView Register.sram_revision (fun v => v[q]!)) (write_sram_word slice addr value) := by
  let beforeWrite : SailM PUnit := do
    let old ← Out.readReg Register.sram_revision
    let seen ← Out.readReg Register.sram_revision
    Out.writeReg Register.sram_revision (vectorUpdate old slice (((seen[slice]!:Nat):Int)+1).toNat)
  have hp : Keeps (FieldView Register.sram_revision (fun v => v[q]!)) beforeWrite := by
    apply read_twice_update_field_keeps
    intro old
    exact vector_update_other 88 q slice old _ hq hs hne
  have h := bind_keeps _ beforeWrite (fun _ => raw_write_sram_word slice addr value) hp (fun _ => port)
  simpa only [beforeWrite,write_sram_word,bind_assoc] using h

theorem actual_ecc_write_other_revision_cell (q slice addr : Nat) (hq : q<88) (hs : slice<88)
    (hne : q≠slice) (value : ecc_ov)
    (port : Keeps (FieldView Register.sram_revision (fun v => v[q]!)) (raw_write_sram_ecc slice addr value)) :
    Keeps (FieldView Register.sram_revision (fun v => v[q]!)) (write_sram_ecc slice addr value) := by
  let beforeWrite : SailM PUnit := do
    let old ← Out.readReg Register.sram_revision
    let seen ← Out.readReg Register.sram_revision
    Out.writeReg Register.sram_revision (vectorUpdate old slice (((seen[slice]!:Nat):Int)+1).toNat)
  have hp : Keeps (FieldView Register.sram_revision (fun v => v[q]!)) beforeWrite := by
    apply read_twice_update_field_keeps
    intro old
    exact vector_update_other 88 q slice old _ hq hs hne
  have h := bind_keeps _ beforeWrite (fun _ => raw_write_sram_ecc slice addr value) hp (fun _ => port)
  simpa only [beforeWrite,write_sram_ecc,bind_assoc] using h

end ResourceFrameProofs
