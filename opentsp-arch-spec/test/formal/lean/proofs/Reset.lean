import Proofs.Capacity
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Frontend
namespace Proofs.Reset
open Out.Register
def resetTail : SailM Unit := do
  Out.writeReg iq_last (vectorInit (Neg.neg 1))
  Out.writeReg iq_dry (vectorInit false)
  Out.writeReg iq_active (vectorInit 0)
  Out.writeReg fill_fifo (vectorInit [])
  Out.writeReg iq_prod (vectorInit 0)
  Out.writeReg iq_wov (vectorInit [])
  Out.writeReg iq_ecc_live false
  let loop_q_lower := 0
  let loop_q_upper := 144
  let mut loop_vars := ()
  for q in [loop_q_lower:loop_q_upper:1]i do
    let () := loop_vars
    loop_vars ← do
      let n ← do (iq_image_len q)
      let loop_i_lower := 0
      let loop_i_upper := (n -i 1)
      let mut loop_vars_1 := ()
      for i in [loop_i_lower:loop_i_upper:1]i do
        let () := loop_vars_1
        loop_vars_1 ← do
          let u ← do (iq_image_unit q i)
          if ((u != iq_padding) : Bool)
          then
            (do
              let _ ← do (iq_push_tagged q u i)
              (pure ()))
          else (pure ())
      (pure loop_vars_1)
      Out.writeReg iq_prod (vectorUpdate (← Out.readReg iq_prod) q (4 *i (Int.ediv (n +i 3) 4)))
  (pure loop_vars)


theorem reset_split : iq_reset () = (do
  Out.writeReg Register.iq_head (vectorInit 0)
  Out.writeReg Register.iq_len (vectorInit 0)
  resetTail) := by rfl

/-- Only bounded queues are admitted. Image indices remain arbitrary natural numbers;
    the production loop must establish the immutable image-length bound before dereferencing. This states the trusted
    reader frame, not image order/content agreement with an object loader. -/
structure ImageEnv (length : Nat → Nat) : Prop where
  len_frame : ∀ q (s : St), q < 145 → (iq_image_len q).run s = .ok (length q) s
  unit_frame : ∀ q i (s : St), q < 145 → i < length q → ∃ u, (iq_image_unit q i).run s = .ok u s

theorem image_len_bind_preserves {α : Type} (q : Nat) (hq : q < 145)
    (length : Nat → Nat) (env : ImageEnv length) (f : Nat → SailM α)
    (hf : Preserves (f (length q))) : Preserves ((iq_image_len q) >>= f) := by
  intro s hs
  have he := env.len_frame q s hq
  have h := hf s hs
  change iq_image_len q s = .ok (length q) s at he
  simpa [EStateM.run, bind, EStateM.bind, he] using h

theorem image_unit_preserves (q i : Nat) (hq : q < 145)
    (length : Nat → Nat) (hi : i < length q) (env : ImageEnv length) : Preserves (iq_image_unit q i) := by
  intro s hs
  obtain ⟨u,h⟩ := env.unit_frame q i s hq hi
  simpa [h] using hs

theorem reset_tail_preserves
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv recU recT) (length : Nat → Nat) (image : ImageEnv length) : Preserves resetTail := by
  unfold resetTail
  dsimp only [ForIn.forIn, ForIn'.forIn', instForInOfForIn']
  repeat' first
    | exact preserves_push _ (by simp only [Membership.mem] at *; simp at *; omega) _ _ recU recT env
    | with_reducible apply image_len_bind_preserves _ (by simp only [Membership.mem] at *; simp at *; omega) length image
    | exact image_unit_preserves _ _ (by simp only [Membership.mem] at *; simp at *; omega) length (by simp only [Membership.mem] at *; simp at *; omega) image
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x
/-- The complete actual reset initializes all145 ring bounds from arbitrary state,
    including its image load loop and both successful/error return states, under
    the explicitly bounded trusted storage/image contracts. -/
theorem reset_initializes (s : St) (hcap : 0 < iq_cap)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv recU recT) (length : Nat → Nat) (image : ImageEnv length) :
    match (iq_reset ()).run s with
    | .ok _ after => IqBounds after
    | .error _ after => IqBounds after := by
  rw [reset_split]
  let s0 : St := {s with regs := (s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145)}
  have hzero : IqBounds s0 := by
    exact Proofs.Capacity.zero_bounds iq_cap hcap s
  have ht := reset_tail_preserves recU recT env length image s0 hzero
  change (match resetTail.run s0 with
    | .ok _ after => IqBounds after
    | .error _ after => IqBounds after)
  cases he : resetTail.run s0 <;> simpa only [he] using ht

end Proofs.Reset
