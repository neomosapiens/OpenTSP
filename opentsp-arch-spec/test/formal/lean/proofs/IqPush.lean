-- Lean counterpart of the SMT obligation p_iq_append (model/props/props.sail), proved about the real generated `iq_push_tagged`.
-- The obligation records the arguments of the two storage callbacks in registers (proof_slot, proof_data, proof_tag_slot, proof_tag) and
-- checks them. The callbacks here are opaque, so the same record is a ghost trace: two functions of the harness state, `recU` for the last
-- unit write and `recT` for the last tag write, which the environment assumption (IqEnv) says the callbacks set and leave to each other.
-- The theorem then states the slot, the payload and the source tag that were written, besides the register effect. Everything else is proved.
import Proofs.Common
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs

abbrev St := SequentialState RegisterType trivialChoiceSource

/-- The environment assumption. The queue-storage callbacks do not touch the register file (the storage lives in the harness, outside the
    Sail registers); `iq_unit_write q i u` makes `recU` read the call `(q, i, u)` and leaves `recT` alone, and `iq_tag_write q i tag` the same
    for `recT`; and the two records do not depend on the register file. The contract applies only to architectural queues and ring slots.
    It assumes successful external allocation and exclusive chip ownership. C implementation transfer and caller reachability are
    separate obligations; this proposition alone does not prove that implementation or all production call sites satisfy it. -/
structure IqEnv (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int)) : Prop where
  unit_write : ∀ q i u (s : St), q < 145 → i < iq_cap →
    ∃ t, (iq_unit_write q i u).run s = EStateM.Result.ok () t ∧ t.regs = s.regs ∧ recU t = some (q, i, u) ∧ recT t = recT s
  tag_write : ∀ q i tag (s : St), q < 145 → i < iq_cap →
    ∃ t, (iq_tag_write q i tag).run s = EStateM.Result.ok () t ∧ t.regs = s.regs ∧ recT t = some (q, i, tag) ∧ recU t = recU s
  blind : ∀ (s : St) r, recU { s with regs := r } = recU s ∧ recT { s with regs := r } = recT s

/-- A full queue refuses the unit: no callback, no register changes. -/
theorem iq_push_full (q : Nat) (u : BitVec 32) (tag : Int) (s : St)
    (L : Vector Nat 145) (hL : s.regs.get? Register.iq_len = some L) (hfull : L[q]! ≥ iq_cap) :
    (iq_push_tagged q u tag).run s = EStateM.Result.ok false s := by
  simp [iq_push_tagged, Out.readReg, PreSail.readReg, hL, EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe,
    MonadStateOf.get, EStateM.get, hfull]

/-- A queue with room takes the unit: the unit goes to the slot `ring (head + len)` with its payload, the tag to the same slot with the
    source tag it was given, the length grows by one and the head stays (the register file differs from the old one in the length only). -/
theorem iq_push_room (q : Nat) (u : BitVec 32) (tag : Int) (s : St)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int)) (env : IqEnv recU recT)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H) (hL : s.regs.get? Register.iq_len = some L)
    (hq : q < 145) (hhead : H[q]! < iq_cap) (hroom : L[q]! < iq_cap) :
    ∃ s', (iq_push_tagged q u tag).run s = EStateM.Result.ok true s' ∧
      s'.regs = s.regs.insert Register.iq_len (vectorUpdate L q (L[q]! + 1)) ∧
      recU s' = some (q, ring (H[q]! + L[q]!), u) ∧ recT s' = some (q, ring (H[q]! + L[q]!), tag) := by
  have hslot : ring (H[q]! + L[q]!) < iq_cap := by
    rw [ring_spec]
    split <;> omega
  obtain ⟨t1, h1, r1, u1, t1'⟩ := env.unit_write q (ring (H[q]! + L[q]!)) u s hq hslot
  obtain ⟨t2, h2, r2, t2', u2⟩ := env.tag_write q (ring (H[q]! + L[q]!)) tag t1 hq hslot
  have hn : ¬ (iq_cap ≤ L[q]!) := by omega
  have e1 : iq_unit_write q (ring (H[q]! + L[q]!)) u s = EStateM.Result.ok () t1 := h1
  have e2 : iq_tag_write q (ring (H[q]! + L[q]!)) tag t1 = EStateM.Result.ok () t2 := h2
  have c1 : ((H[q]! : Int) + (L[q]! : Int)).toNat = H[q]! + L[q]! := by omega
  simp [iq_push_tagged, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg, hL, hH, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get, modify, modifyGet, MonadStateOf.modifyGet, hn, c1,
    e1, e2, r1, r2]
  refine ⟨{ t2 with regs := t2.regs.insert Register.iq_len (vectorUpdate L q (L[q]! + 1)) }, rfl, ?_, ?_, ?_⟩
  · simp [r2, r1]
  · rw [(env.blind t2 _).1, u2, u1]
  · rw [(env.blind t2 _).2, t2']

/-- The slot both callbacks receive is the logical position `head + len` folded once into the ring, as in p_iq_append: inside the ring, and
    equal to `head + len` or to `head + len - iq_cap`. -/
theorem iq_push_slot (h l : Nat) (hh : h < iq_cap) (hl : l < iq_cap) :
    ring (h + l) < iq_cap ∧ (ring (h + l) = h + l ∨ ring (h + l) + iq_cap = h + l) := by
  rw [ring_spec]
  repeat' split
  all_goals omega
