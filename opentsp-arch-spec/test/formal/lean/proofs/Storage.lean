-- Ordered ring-storage composition for the actual capacity-parametric push/pop bodies.
import Proofs.Capacity
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity
namespace Proofs.Storage

def put {α : Type} (m : Nat → Nat → α) (q slot : Nat) (value : α) : Nat → Nat → α :=
  fun r j => if r = q ∧ j = slot then value else m r j

def Live (cap : Nat) (s : St) (q slot : Nat) : Prop :=
  ∃ H L : Vector Nat 145, s.regs.get? Register.iq_head = some H ∧
    s.regs.get? Register.iq_len = some L ∧ q < 145 ∧ H[q]! < cap ∧ L[q]! ≤ cap ∧
    ∃ i, i < L[q]! ∧ slot = ringC cap (H[q]! + i)

theorem live_domain (cap : Nat) (s : St) (q slot : Nat) (h : Live cap s q slot) :
    q < 145 ∧ slot < cap := by
  obtain ⟨H,L,_,_,hq,hh,hl,i,hi,hs⟩ := h
  refine ⟨hq,?_⟩
  rw [hs,Proofs.Capacity.ring_spec]
  split <;> omega

/-- Trusted storage contract on initialized occupied slots. Lazy C allocation does
    NOT initialize other unit cells, so neither frames nor reads promise their values.
    Allocation succeeds, chip ownership is exclusive, and occupied cells have been
    initialized by load/push. All tags remain arbitrary Int. -/
structure StoreEnv (cap : Nat)
    (U : St → Nat → Nat → BitVec 32) (T : St → Nat → Nat → Int) : Prop where
  unit_write : ∀ q slot u (s : St), q < 145 → slot < cap →
    ∃ t, (iq_unit_write q slot u).run s = .ok () t ∧ t.regs = s.regs ∧
      U t q slot = u ∧
      (∀ r j, Live cap s r j → (r ≠ q ∨ j ≠ slot) → U t r j = U s r j) ∧ T t = T s
  tag_write : ∀ q slot tag (s : St), q < 145 → slot < cap →
    ∃ t, (iq_tag_write q slot tag).run s = .ok () t ∧ t.regs = s.regs ∧
      U t = U s ∧ T t = put (T s) q slot tag
  unit_read : ∀ q slot (s : St), Live cap s q slot →
    (iq_unit_read q slot).run s = .ok (U s q slot) s
  tag_read : ∀ q slot (s : St), Live cap s q slot →
    (iq_tag_read q slot).run s = .ok (T s q slot) s
  blind : ∀ (s : St) regs, U {s with regs := regs} = U s ∧ T {s with regs := regs} = T s

/-- The actual push changes exactly one unit/tag pair and length, with all other
    occupied external slots and all unrelated Sail registers preserved. -/
theorem push_store (cap q : Nat) (u : BitVec 32) (tag : Int) (s : St)
    (U : St → Nat → Nat → BitVec 32) (T : St → Nat → Nat → Int) (env : StoreEnv cap U T)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L)
    (hq : q < 145) (hh : H[q]! < cap) (hl : L[q]! < cap) :
    ∃ after, (pushC cap q u tag).run s = .ok true after ∧
      after.regs = s.regs.insert Register.iq_len (vectorUpdate L q (L[q]!+1)) ∧
      U after q (ringC cap (H[q]!+L[q]!)) = u ∧
      (∀ r j, Live cap s r j → (r ≠ q ∨ j ≠ ringC cap (H[q]!+L[q]!)) →
        U after r j = U s r j) ∧
      T after = put (T s) q (ringC cap (H[q]!+L[q]!)) tag := by
  have hslot : ringC cap (H[q]!+L[q]!) < cap := by
    rw [Proofs.Capacity.ring_spec]
    split <;> omega
  obtain ⟨t1,h1,r1,u1,frame1,t1'⟩ := env.unit_write q (ringC cap (H[q]!+L[q]!)) u s hq hslot
  obtain ⟨t2,h2,r2,u2,t2'⟩ := env.tag_write q (ringC cap (H[q]!+L[q]!)) tag t1 hq hslot
  have hn : ¬cap ≤ L[q]! := by omega
  have e1 : iq_unit_write q (ringC cap (H[q]!+L[q]!)) u s = .ok () t1 := h1
  have e2 : iq_tag_write q (ringC cap (H[q]!+L[q]!)) tag t1 = .ok () t2 := h2
  have hc : ((H[q]! : Int)+(L[q]! : Int)).toNat = H[q]!+L[q]! := by omega
  simp [pushC, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg, hL,hH,
    EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get,
    EStateM.get, modify, modifyGet, MonadStateOf.modifyGet, hn,hc,e1,e2,r1,r2]
  refine ⟨{t2 with regs := t2.regs.insert Register.iq_len (vectorUpdate L q (L[q]!+1))},rfl,?_,?_,?_,?_⟩
  · simp [r2,r1]
  · rw [(env.blind t2 _).1,u2,u1]
  · intro r j live different
    rw [(env.blind t2 _).1,u2]
    exact frame1 r j live different
  · rw [(env.blind t2 _).2,t2',t1']

def contents (cap : Nat) (U : Nat → Nat → BitVec 32) (T : Nat → Nat → Int)
    (q head len : Nat) : List (BitVec 32 × Int) :=
  (List.range len).map (fun i => (U q (ringC cap (head + i)), T q (ringC cap (head + i))))

theorem ring_distinct (cap h a b : Nat) (_hh : h < cap) (ha : a < cap) (hb : b < cap)
    (hne : a ≠ b) : ringC cap (h+a) ≠ ringC cap (h+b) := by
  simp only [Proofs.Capacity.ring_spec]
  repeat' split
  all_goals omega

/-- Physical stores at the actual append slot preserve every previous logical pair
    and append precisely the caller's unit/tag, including negative and huge tags. -/
theorem contents_append (cap q h len : Nat) (hh : h < cap) (hl : len < cap)
    (U : Nat → Nat → BitVec 32) (T : Nat → Nat → Int) (u : BitVec 32) (tag : Int) :
    contents cap (put U q (ringC cap (h+len)) u) (put T q (ringC cap (h+len)) tag) q h (len+1) =
      contents cap U T q h len ++ [(u,tag)] := by
  unfold contents
  rw [List.range_succ, List.map_append]
  congr 1
  · apply List.map_congr_left
    intro i hi
    have hi' : i < len := List.mem_range.mp hi
    have hne := ring_distinct cap h i len hh (by omega) hl (by omega)
    simp [put,hne]
  · simp [put]

theorem contents_append_frame (cap q h len : Nat)
    (U V : Nat → Nat → BitVec 32) (T : Nat → Nat → Int) (u : BitVec 32) (tag : Int)
    (hnew : V q (ringC cap (h+len)) = u)
    (hold : ∀ i, i < len → V q (ringC cap (h + i)) = U q (ringC cap (h + i)))
    (hh : h < cap) (hl : len < cap) :
    contents cap V (put T q (ringC cap (h+len)) tag) q h (len+1) =
      contents cap U T q h len ++ [(u,tag)] := by
  unfold contents
  rw [List.range_succ, List.map_append]
  congr 1
  · apply List.map_congr_left
    intro i hi
    have hi' := List.mem_range.mp hi
    have hne := ring_distinct cap h i len hh (by omega) hl (by omega)
    simp [put,hne,hold i hi']
  · simp [put,hnew]

/-- Composed production result: exact execution, register frame and logical FIFO append. -/
theorem actual_push_appends (q : Nat) (u : BitVec 32) (tag : Int) (s : St)
    (U : St → Nat → Nat → BitVec 32) (T : St → Nat → Nat → Int) (env : StoreEnv iq_cap U T)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L)
    (hq : q < 145) (hh : H[q]! < iq_cap) (hl : L[q]! < iq_cap) :
    ∃ after, (iq_push_tagged q u tag).run s = .ok true after ∧
      after.regs = s.regs.insert Register.iq_len (vectorUpdate L q (L[q]!+1)) ∧
      contents iq_cap (U after) (T after) q H[q]! (L[q]!+1) =
        contents iq_cap (U s) (T s) q H[q]! L[q]! ++ [(u,tag)] := by
  obtain ⟨after,he,hr,hu,hframe,ht⟩ := push_store iq_cap q u tag s U T env H L hH hL hq hh hl
  refine ⟨after,?_,hr,?_⟩
  · simpa only [push_production] using he
  · rw [ht]
    apply contents_append_frame iq_cap q H[q]! L[q]! (U s) (U after) (T s) u tag hu _ hh hl
    intro i hi
    apply hframe q (ringC iq_cap (H[q]! + i))
    · exact ⟨H,L,hH,hL,hq,hh,by omega,i,hi,rfl⟩
    · exact Or.inr (ring_distinct iq_cap H[q]! i L[q]! hh (by omega) hl (by omega))

theorem ring_shift (cap h len n i : Nat) (hh : h < cap) (hl : len ≤ cap)
    (hn : n ≤ len) (hi : i < len-n) :
    ringC cap (ringC cap (h+n) + i) = ringC cap (h + n + i) := by
  simp only [Proofs.Capacity.ring_spec]
  repeat' split
  all_goals omega

/-- Actual pop changes only head/length; every surviving logical pair retains its old order. -/
theorem pop_keeps_order (cap q n : Nat) (s : St)
    (U : St → Nat → Nat → BitVec 32) (T : St → Nat → Nat → Int) (env : StoreEnv cap U T)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L)
    (hh : H[q]! < cap) (hl : L[q]! ≤ cap) (hn : n ≤ L[q]!) :
    ∃ after, (popC cap q n).run s = .ok () after ∧
      after.regs = (s.regs.insert Register.iq_head (vectorUpdate H q (ringC cap (H[q]!+n)))).insert
        Register.iq_len (vectorUpdate L q (L[q]!-n)) ∧
      U after = U s ∧ T after = T s ∧
      ∀ i, i < L[q]!-n →
        (U after q (ringC cap (ringC cap (H[q]!+n) + i)),
         T after q (ringC cap (ringC cap (H[q]!+n) + i))) =
        (U s q (ringC cap (H[q]! + n + i)), T s q (ringC cap (H[q]! + n + i))) := by
  let after : St := {s with regs := (s.regs.insert Register.iq_head (vectorUpdate H q (ringC cap (H[q]!+n)))).insert Register.iq_len (vectorUpdate L q (L[q]!-n))}
  refine ⟨after,iq_pop_run q n s H L hH hL,rfl,(env.blind s _).1,(env.blind s _).2,?_⟩
  intro i hi
  rw [(env.blind s _).1,(env.blind s _).2,ring_shift cap H[q]! L[q]! n i hh hl hn hi]

/-- The actual decoder's primitive peek returns the stored unit at its logical position
    without changing Sail state, and its external storage index is proved in bounds. -/
theorem peek_value (q i : Nat) (s : St)
    (U : St → Nat → Nat → BitVec 32) (T : St → Nat → Nat → Int) (env : StoreEnv iq_cap U T)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L)
    (hq : q < 145) (hh : H[q]! < iq_cap) (hl : L[q]! ≤ iq_cap) (hi : i < L[q]!) :
    (iq_peek q i).run s = .ok (U s q (ringC iq_cap (H[q]! + i))) s := by
  have hr := env.unit_read q (ringC iq_cap (H[q]! + i)) s ⟨H,L,hH,hL,hq,hh,hl,i,hi,rfl⟩
  have hc : ((H[q]! : Int) + (i : Int)).toNat = H[q]! + i := by omega
  simp only [ring_production] at hr ⊢
  simp [iq_peek, Out.readReg, PreSail.readReg, hH, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get, hc]
  exact hr

end Proofs.Storage
