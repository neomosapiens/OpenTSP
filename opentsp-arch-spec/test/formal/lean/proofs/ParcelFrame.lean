import Proofs.StorageProduct
import Lean
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
set_option maxHeartbeats 500000
namespace Proofs.ParcelFrame
/-- The trusted read ports promise no state change only for the real bounded C
    arrays. Unit initialization is discharged by the separate product relation. -/
structure ReadPorts (cap : Nat) : Prop where
  unit : ∀ q slot (s : St), Proofs.StorageProduct.Good cap s →
    Proofs.Storage.Live cap s q slot → ∃ u, (iq_unit_read q slot).run s = .ok u s
  tag : ∀ q slot (s : St), q<145 → slot<cap → ∃ t, (iq_tag_read q slot).run s = .ok t s

def FrameAt {α : Type} (s : St) (action : SailM α) : Prop :=
  match action.run s with | .ok _ after => after=s | .error _ after => after=s

theorem pure_frame {α : Type} (s : St) (x : α) : FrameAt s (pure x) := by rfl

theorem bind_frame {α β : Type} (s : St) (a : SailM α) (f : α → SailM β)
    (ha : FrameAt s a) (hf : ∀ x, FrameAt s (f x)) : FrameAt s (a >>= f) := by
  cases he : a.run s with
  | ok x after =>
    have hs : after=s := by simpa [FrameAt,he] using ha
    subst after
    simpa [FrameAt,EStateM.run,bind,EStateM.bind,show a s = .ok x s from he] using hf x
  | error error after =>
    simpa [FrameAt,EStateM.run,bind,EStateM.bind,show a s = .error error after from he] using ha

theorem read_frame (s : St) (r : Register) : FrameAt s (Out.readReg r) := by
  simp [FrameAt,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    get,getThe,MonadStateOf.get,EStateM.get]
  cases s.regs.get? r <;> rfl

theorem ite_frame {α : Type} (s : St) (p : Prop) [Decidable p] (a b : SailM α)
    (ha : FrameAt s a) (hb : FrameAt s b) : FrameAt s (if p then a else b) := by
  split <;> assumption

theorem forIn_frame {β : Type} (s : St) (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β))
    (hf : ∀ i hi value, FrameAt s (f i hi value)) :
    FrameAt s (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => FrameAt s (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply bind_frame _ _ _ (hf i hi value)
    intro step
    cases step with
    | done value => exact pure_frame s value
    | yield value => exact ih value
  · intro value i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact pure_frame s value

theorem ring_domain (h i : Nat) (hh : h<iq_cap) (hi : i≤iq_cap) :
    ring (h + i)<iq_cap := by
  rw [Proofs.ring_spec]
  split <;> omega

theorem peek_frame (s : St) (q i : Nat) (ports : ReadPorts iq_cap)
    (H : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (L : Vector Nat 145) (hL : s.regs.get? Register.iq_len=some L)
    (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hi : i<L[q]!) :
    FrameAt s (iq_peek q i) := by
  obtain ⟨u,he⟩ := ports.unit q (ring (H[q]! + i)) s good ⟨H,L,hH,hL,hq,hh,hl,i,hi,rfl⟩
  change iq_unit_read q (ring (H[q]! + i)) s = .ok u s at he
  have hc : ((H[q]!:Int)+(i:Int)).toNat=H[q]! + i := by omega
  simp [FrameAt,iq_peek,Out.readReg,PreSail.readReg,hH,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hc,he]

theorem unit_word_frame (s : St) (q i : Nat) (ports : ReadPorts iq_cap)
    (H : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hq : q<145) (hh : H[q]!<iq_cap) (hi : i≤iq_cap) :
    FrameAt s (iq_unit_word q i) := by
  obtain ⟨t,he⟩ := ports.tag q (ring (H[q]! + i)) s hq (ring_domain _ _ hh hi)
  change iq_tag_read q (ring (H[q]! + i)) s = .ok t s at he
  have hc : ((H[q]!:Int)+(i:Int)).toNat=H[q]! + i := by omega
  simp [FrameAt,iq_unit_word,Out.readReg,PreSail.readReg,hH,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hc,he]
  by_cases ht : t=iq_no_word <;> simp [ht,EStateM.pure]

theorem unit_lane_frame (s : St) (q i : Nat) (ports : ReadPorts iq_cap)
    (H : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hq : q<145) (hh : H[q]!<iq_cap) (hi : i≤iq_cap) :
    FrameAt s (iq_unit_lane q i) := by
  obtain ⟨t,he⟩ := ports.tag q (ring (H[q]! + i)) s hq (ring_domain _ _ hh hi)
  change iq_tag_read q (ring (H[q]! + i)) s = .ok t s at he
  have hc : ((H[q]!:Int)+(i:Int)).toNat=H[q]! + i := by omega
  simp [FrameAt,iq_unit_lane,Out.readReg,PreSail.readReg,hH,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hc,he]

theorem word_overlay_frame (s : St) (q w : Nat) : FrameAt s (iq_word_ov q w) := by
  unfold iq_word_ov
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | with_reducible apply read_frame
    | with_reducible apply pure_frame
    | with_reducible apply ite_frame
    | with_reducible apply forIn_frame
    | with_reducible apply bind_frame
    | split
    | intro x

theorem view_frame (s : St) (q i : Nat) (ports : ReadPorts iq_cap)
    (H : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (L : Vector Nat 145) (hL : s.regs.get? Register.iq_len=some L)
    (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hi : i<L[q]!) :
    FrameAt s (iq_view q i) := by
  unfold iq_view
  repeat' first
    | exact peek_frame s q i ports H hH L hL good hq hh hl hi
    | exact unit_word_frame s q i ports H hH hq hh (by omega)
    | exact unit_lane_frame s q i ports H hH hq hh (by omega)
    | exact word_overlay_frame s _ _
    | with_reducible apply read_frame
    | with_reducible apply pure_frame
    | with_reducible apply ite_frame
    | with_reducible apply bind_frame
    | split
    | intro x

theorem word_unc_frame (s : St) (q : Nat) (w : Int) : FrameAt s (iq_word_unc q w) := by
  unfold iq_word_unc
  repeat' first
    | exact word_overlay_frame s _ _
    | with_reducible apply pure_frame
    | with_reducible apply ite_frame
    | with_reducible apply bind_frame
    | split
    | intro x

theorem text_unc_frame (s : St) (q pos w : Nat) (ports : ReadPorts iq_cap)
    (H : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hq : q<145) (hh : H[q]!<iq_cap) (hpos : pos<iq_cap) :
    FrameAt s (iq_text_unc q pos w) := by
  unfold iq_text_unc
  repeat' first
    | exact unit_word_frame s q pos ports H hH hq hh (by omega)
    | exact unit_word_frame s q (pos+1) ports H hH hq hh (by omega)
    | exact word_unc_frame s _ _
    | with_reducible apply read_frame
    | with_reducible apply pure_frame
    | with_reducible apply ite_frame
    | with_reducible apply bind_frame
    | split
    | intro x


theorem throw_frame {α : Type} (s : St) (e : Error Out.exception) :
    FrameAt s (throw e : SailM α) := by rfl

theorem assert_frame (s : St) (b : Bool) (message : String) :
    FrameAt s (Out.assert b message : SailM Unit) := by
  cases b <;> rfl

theorem ty_split_backwards_frame (s : St) (arg_ : ((BitVec 1) × (BitVec 2) × (BitVec 2))) : FrameAt s (ty_split_backwards arg_) := by
  unfold ty_split_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem dst_split_backwards_frame (s : St) (arg_ : ((BitVec 1) × (BitVec 4))) : FrameAt s (dst_split_backwards arg_) := by
  unfold dst_split_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_vxm_binop_backwards_frame (s : St) (arg_ : (BitVec 6)) : FrameAt s (encdec_vxm_binop_backwards arg_) := by
  unfold encdec_vxm_binop_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_vxm_unop_backwards_frame (s : St) (arg_ : (BitVec 6)) : FrameAt s (encdec_vxm_unop_backwards arg_) := by
  unfold encdec_vxm_unop_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_cmp_mode_backwards_frame (s : St) (arg_ : (BitVec 4)) : FrameAt s (encdec_cmp_mode_backwards arg_) := by
  unfold encdec_cmp_mode_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_vxm_conv_backwards_frame (s : St) (arg_ : (BitVec 6)) : FrameAt s (encdec_vxm_conv_backwards arg_) := by
  unfold encdec_vxm_conv_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_reduce_op_backwards_frame (s : St) (arg_ : (BitVec 4)) : FrameAt s (encdec_reduce_op_backwards arg_) := by
  unfold encdec_reduce_op_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_mapop_backwards_frame (s : St) (arg_ : (BitVec 6)) : FrameAt s (encdec_mapop_backwards arg_) := by
  unfold encdec_mapop_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_mem_dir_backwards_frame (s : St) (arg_ : (BitVec 6)) : FrameAt s (encdec_mem_dir_backwards arg_) := by
  unfold encdec_mem_dir_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem wtype_split_backwards_frame (s : St) (arg_ : ((BitVec 1) × (BitVec 2))) : FrameAt s (wtype_split_backwards arg_) := by
  unfold wtype_split_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_acctype_backwards_frame (s : St) (arg_ : (BitVec 2)) : FrameAt s (encdec_acctype_backwards arg_) := by
  unfold encdec_acctype_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

theorem encdec_c2c_op_backwards_frame (s : St) (arg_ : (BitVec 6)) : FrameAt s (encdec_c2c_op_backwards arg_) := by
  unfold encdec_c2c_op_backwards
  repeat' first
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | with_reducible apply throw_frame
    | with_reducible apply assert_frame
    | split
    | intro x

/-- Frame an outer option match without enumerating pure decoder predicates. -/
theorem option_frame {α β : Type} (s : St) (x : Option α) (someCase : α → SailM β)
    (noneCase : SailM β) (hs : ∀ a, FrameAt s (someCase a))
    (hn : FrameAt s noneCase) :
    FrameAt s (match x with | some a => someCase a | none => noneCase) := by
  cases x with
  | none => exact hn
  | some a => exact hs a

elab "split_action_match" : tactic => Lean.Elab.Tactic.liftMetaTactic fun goal => goal.withContext do
  let target ← Lean.instantiateMVars (← goal.getType)
  let args := target.getAppArgs
  unless target.isAppOfArity ``FrameAt 3 do
    throwError "expected FrameAt"
  let action ← Lean.Meta.whnf args[2]!
  let some _ ← Lean.Meta.matchMatcherApp? action
    | throwError "action is not an outer match"
  let target' ← Lean.Meta.mkAppM ``FrameAt #[args[1]!, action]
  let goal ← goal.change target'
  Lean.Meta.Split.splitMatch goal action

theorem decode_text_frame (s : St) (u0 u1 : BitVec 32) : FrameAt s (decode_text u0 u1) := by
  unfold decode_text
  repeat' first
    | with_reducible apply ty_split_backwards_frame
    | with_reducible apply dst_split_backwards_frame
    | with_reducible apply encdec_vxm_binop_backwards_frame
    | with_reducible apply encdec_vxm_unop_backwards_frame
    | with_reducible apply encdec_cmp_mode_backwards_frame
    | with_reducible apply encdec_vxm_conv_backwards_frame
    | with_reducible apply encdec_reduce_op_backwards_frame
    | with_reducible apply encdec_mapop_backwards_frame
    | with_reducible apply encdec_mem_dir_backwards_frame
    | with_reducible apply wtype_split_backwards_frame
    | with_reducible apply encdec_acctype_backwards_frame
    | with_reducible apply encdec_c2c_op_backwards_frame
    | with_reducible apply pure_frame
    | with_reducible apply ite_frame
    | with_reducible apply bind_frame
    | unfold encdec32_backwards
    | unfold encdec64_backwards
    | split_action_match
    | split
    | intro x


theorem decode_at_frame (s : St) (q pos w : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap)
    (hi : pos<L[q]!) (hw : w=1 ∨ w=2) (hend : pos+w≤L[q]!) :
    FrameAt s (iq_decode_at q pos w) := by
  have hnext : ((pos:Int)+1).toNat=pos+1 := by omega
  unfold iq_decode_at
  repeat' first
    | exact text_unc_frame s q pos w ports H hH hq hh (by omega)
    | exact view_frame s q pos ports H hH L hL good hq hh hl hi
    | simpa only [hnext] using view_frame s q (pos+1) ports H hH L hL good hq hh hl (by simp_all; omega)
    | exact decode_text_frame s _ _
    | with_reducible apply pure_frame
    | with_reducible apply bind_frame
    | split
    | intro x

theorem width_one_or_two (u : BitVec 32) : text_width u=1 ∨ text_width u=2 := by
  unfold text_width
  dsimp only
  split <;> simp

end Proofs.ParcelFrame
