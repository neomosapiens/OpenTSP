-- Capacity/rate-parametric exact-body factories. The four rfl equalities bind these
-- bodies to the actual generated production functions at the exported constants.
-- The ring, push, pop and complete delivery safety proofs quantify arbitrary Nat
-- capacities/rates; their invariant is nonvacuous only for positive capacity.
-- This proves no external-memory FIFO identity, callback C refinement, reset or
-- global reachability/dispatch/progress property. Other profile geometry is still
-- that of the exported RegisterType; source equality is checked on each export.
import Proofs.Frontend
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
namespace Proofs.Capacity
open Out.Register

def ringC (cap : Nat) (x : Nat) : Nat :=
  if ((x ≥b cap) : Bool)
  then (x -i cap)
  else x



def popC (cap : Nat) (q : Nat) (n : Nat) : SailM Unit := do
  Out.writeReg iq_head (vectorUpdate (← Out.readReg iq_head) q
    ((ringC cap) ((GetElem?.getElem! (← Out.readReg iq_head) q) +i n)))
  Out.writeReg iq_len (vectorUpdate (← Out.readReg iq_len) q
    (Max.max ((GetElem?.getElem! (← Out.readReg iq_len) q) -i n) 0))



def pushC (cap : Nat) (q : Nat) (u : (BitVec 32)) (tag : Int) : SailM Bool := do
  if (((GetElem?.getElem! (← Out.readReg iq_len) q) ≥b cap) : Bool)
  then (pure false)
  else
    (do
      (iq_unit_write q
        ((ringC cap)
          ((GetElem?.getElem! (← Out.readReg iq_head) q) +i (GetElem?.getElem! (← Out.readReg iq_len) q)))
        u)
      (iq_tag_write q
        ((ringC cap)
          ((GetElem?.getElem! (← Out.readReg iq_head) q) +i (GetElem?.getElem! (← Out.readReg iq_len) q)))
        tag)
      Out.writeReg iq_len (vectorUpdate (← Out.readReg iq_len) q
        ((GetElem?.getElem! (← Out.readReg iq_len) q) +i 1))
      (pure true))



def deliverC (cap rate : Nat) (q : Nat) (c : Nat) : SailM (Bool × Bool) := do
  let budget : Nat := rate
  let pushed : Bool := false
  let lost : Bool := false
  let go : Bool := true
  let (budget, go, lost, pushed) ← (( do
    let loop__step_lower := 0
    let loop__step_upper ← do
      (pure (rate +i (list_len (GetElem?.getElem! (← Out.readReg fill_fifo) q))))
    let mut loop_vars := (budget, go, lost, pushed)
    for _step in [loop__step_lower:loop__step_upper:1]i do
      let (budget, go, lost, pushed) := loop_vars
      loop_vars ← do
        let (budget, go, lost, pushed) ← (( do
          if ((go && ((budget >b 0) : Bool)) : Bool)
          then
            (do
              let (budget, go, lost, pushed) ← (( do
                match (GetElem?.getElem! (← Out.readReg fill_fifo) q) with
                | [] =>
                  (let go : Bool := false
                  (pure (budget, go, lost, pushed)))
                | (seg :: rest) =>
                  (do
                    let (budget, go, lost, pushed) ← (( do
                      if ((seg.ready >b c) : Bool)
                      then
                        (let go : Bool := false
                        (pure (budget, go, lost, pushed)))
                      else
                        (do
                          let (budget, lost, pushed) ← (( do
                            match seg.units with
                            | [] =>
                              (do
                                Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q rest)
                                (pure (budget, lost, pushed)))
                            | (u :: us) =>
                              (do
                                Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q
                                  ({ seg with units := us, taken := (seg.taken +i 1) } :: rest))
                                let budget : Nat := (Max.max (budget -i 1) 0)
                                let (lost, pushed) ← (( do
                                  if ((u != iq_padding) : Bool)
                                  then
                                    (do
                                      let tag : Int :=
                                        if ((seg.base ≥b 0) : Bool)
                                        then (seg.base +i seg.taken)
                                        else iq_no_word
                                      let (lost, pushed) ← (( do
                                        if ((← ((pushC cap) q u tag)) : Bool)
                                        then
                                          (do
                                            let pushed : Bool := true
                                            Out.writeReg iq_active (vectorUpdate (← Out.readReg iq_active)
                                              q c)
                                            if ((seg.base ≥b 0) : Bool)
                                            then
                                              (do
                                                let e := (ov_nth seg.ovw (Int.ediv seg.taken 4))
                                                if ((e != 0b0000000000#10) : Bool)
                                                then
                                                  (iq_word_ov_set q
                                                    (Int.ediv (Max.max (seg.base +i seg.taken) 0) 4)
                                                    e)
                                                else (pure ()))
                                            else (pure ())
                                            (pure (lost, pushed)))
                                        else
                                          (let lost : Bool := true
                                          (pure (lost, pushed))) ) : SailM (Bool × Bool) )
                                      (pure (lost, pushed)))
                                  else (pure (lost, pushed)) ) : SailM (Bool × Bool) )
                                (pure (budget, lost, pushed))) ) : SailM (Nat × Bool × Bool) )
                          (pure (budget, go, lost, pushed))) ) : SailM (Nat × Bool × Bool × Bool)
                      )
                    (pure (budget, go, lost, pushed))) ) : SailM (Nat × Bool × Bool × Bool) )
              (pure (budget, go, lost, pushed)))
          else (pure (budget, go, lost, pushed)) ) : SailM (Nat × Bool × Bool × Bool) )
        (pure (budget, go, lost, pushed))
    (pure loop_vars) ) : SailM (Nat × Bool × Bool × Bool) )
  match (GetElem?.getElem! (← Out.readReg fill_fifo) q) with
  | (seg :: rest) =>
    (do
      match seg.units with
      | [] => Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q rest)
      | _ => (pure ()))
  | [] => (pure ())
  (pure (pushed, lost))



theorem ring_production (x : Nat) : ringC iq_cap x = ring x := by rfl
theorem pop_production (q n : Nat) : popC iq_cap q n = iq_pop q n := by rfl
theorem push_production (q : Nat) (u : BitVec 32) (tag : Int) : pushC iq_cap q u tag = iq_push_tagged q u tag := by rfl
theorem deliver_production (q c : Nat) : deliverC iq_cap iq_rate q c = iq_deliver q c := by rfl

variable {cap rate : Nat}

theorem ring_spec (x : Nat) : (ringC cap) x = if x ≥ cap then x - cap else x := by
  unfold ringC
  by_cases h : x ≥ cap
  · simp [h]
  · simp [h]


theorem iq_pop_run (q n : Nat) (s : SequentialState RegisterType trivialChoiceSource) (H L : Vector Nat 145)
    (hH : s.regs.get? Register.iq_head = some H) (hL : s.regs.get? Register.iq_len = some L) :
    ((popC cap) q n).run s = EStateM.Result.ok ()
      { s with regs := (s.regs.insert Register.iq_head (vectorUpdate H q ((ringC cap) (H[q]! + n)))).insert Register.iq_len
                (vectorUpdate L q (L[q]! - n)) } := by
  simp [popC, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg, hH, hL, EStateM.run, bind,
    EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get, modify, modifyGet,
    MonadStateOf.modifyGet, EStateM.modifyGet, Std.ExtDHashMap.get?_insert]
  have h : ((H[q]! : Int) + (n : Int)).toNat = H[q]! + n := by omega
  simp [h]

/-- The lengths add up, the head stays inside the (ringC cap), and the surviving logical index denotes the same physical slot. -/
theorem iq_pop_conservation (q n : Nat) (hq : q < 145)
    (s : SequentialState RegisterType trivialChoiceSource) (H L : Vector Nat 145)
    (hH : s.regs.get? Register.iq_head = some H) (hL : s.regs.get? Register.iq_len = some L)
    (hhead : H[q]! < cap) (hold : L[q]! ≤ cap) (hpop : n ≤ L[q]!) :
    ∃ s', ((popC cap) q n).run s = EStateM.Result.ok () s' ∧
      ∃ H' L', s'.regs.get? Register.iq_head = some H' ∧ s'.regs.get? Register.iq_len = some L' ∧
        L'[q]! + n = L[q]! ∧ H'[q]! < cap ∧
        ∀ i, i < L'[q]! → (ringC cap) (H'[q]! + i) = (ringC cap) (H[q]! + n + i) := by
  refine ⟨_, iq_pop_run q n s H L hH hL, vectorUpdate H q ((ringC cap) (H[q]! + n)), vectorUpdate L q (L[q]! - n), ?_, ?_, ?_, ?_, ?_⟩
  · simp [Std.ExtDHashMap.get?_insert]
  · simp
  · rw [vectorUpdate_self _ _ _ hq]; omega
  · rw [vectorUpdate_self _ _ _ hq, ring_spec]
    repeat' split
    all_goals omega
  · intro i hi
    simp only [vectorUpdate_self _ _ _ hq] at hi ⊢
    simp only [ring_spec]
    repeat' split
    all_goals omega

structure IqEnv (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int)) : Prop where
  unit_write : ∀ q i u (s : St), q < 145 → i < cap →
    ∃ t, (iq_unit_write q i u).run s = EStateM.Result.ok () t ∧ t.regs = s.regs ∧ recU t = some (q, i, u) ∧ recT t = recT s
  tag_write : ∀ q i tag (s : St), q < 145 → i < cap →
    ∃ t, (iq_tag_write q i tag).run s = EStateM.Result.ok () t ∧ t.regs = s.regs ∧ recT t = some (q, i, tag) ∧ recU t = recU s
  blind : ∀ (s : St) r, recU { s with regs := r } = recU s ∧ recT { s with regs := r } = recT s

/-- A full queue refuses the unit: no callback, no register changes. -/
theorem iq_push_full (q : Nat) (u : BitVec 32) (tag : Int) (s : St)
    (L : Vector Nat 145) (hL : s.regs.get? Register.iq_len = some L) (hfull : L[q]! ≥ cap) :
    ((pushC cap) q u tag).run s = EStateM.Result.ok false s := by
  simp [pushC, Out.readReg, PreSail.readReg, hL, EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe,
    MonadStateOf.get, EStateM.get, hfull]

/-- A queue with room takes the unit: the unit goes to the slot `(ringC cap) (head + len)` with its payload, the tag to the same slot with the
    source tag it was given, the length grows by one and the head stays (the register file differs from the old one in the length only). -/
theorem iq_push_room (q : Nat) (u : BitVec 32) (tag : Int) (s : St)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int)) (env : IqEnv (cap := cap) recU recT)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H) (hL : s.regs.get? Register.iq_len = some L)
    (hq : q < 145) (hhead : H[q]! < cap) (hroom : L[q]! < cap) :
    ∃ s', ((pushC cap) q u tag).run s = EStateM.Result.ok true s' ∧
      s'.regs = s.regs.insert Register.iq_len (vectorUpdate L q (L[q]! + 1)) ∧
      recU s' = some (q, (ringC cap) (H[q]! + L[q]!), u) ∧ recT s' = some (q, (ringC cap) (H[q]! + L[q]!), tag) := by
  have hslot : (ringC cap) (H[q]! + L[q]!) < cap := by
    rw [ring_spec]
    split <;> omega
  obtain ⟨t1, h1, r1, u1, t1'⟩ := env.unit_write q ((ringC cap) (H[q]! + L[q]!)) u s hq hslot
  obtain ⟨t2, h2, r2, t2', u2⟩ := env.tag_write q ((ringC cap) (H[q]! + L[q]!)) tag t1 hq hslot
  have hn : ¬ (cap ≤ L[q]!) := by omega
  have e1 : iq_unit_write q ((ringC cap) (H[q]! + L[q]!)) u s = EStateM.Result.ok () t1 := h1
  have e2 : iq_tag_write q ((ringC cap) (H[q]! + L[q]!)) tag t1 = EStateM.Result.ok () t2 := h2
  have c1 : ((H[q]! : Int) + (L[q]! : Int)).toNat = H[q]! + L[q]! := by omega
  simp [pushC, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg, hL, hH, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get, modify, modifyGet, MonadStateOf.modifyGet, hn, c1,
    e1, e2, r1, r2]
  refine ⟨{ t2 with regs := t2.regs.insert Register.iq_len (vectorUpdate L q (L[q]! + 1)) }, rfl, ?_, ?_, ?_⟩
  · simp [r2, r1]
  · rw [(env.blind t2 _).1, u2, u1]
  · rw [(env.blind t2 _).2, t2']

/-- The slot both callbacks receive is the logical position `head + len` folded once into the (ringC cap), as in p_iq_append: inside the (ringC cap), and
    equal to `head + len` or to `head + len - cap`. -/
theorem iq_push_slot (h l : Nat) (hh : h < cap) (hl : l < cap) :
    (ringC cap) (h + l) < cap ∧ ((ringC cap) (h + l) = h + l ∨ (ringC cap) (h + l) + cap = h + l) := by
  rw [ring_spec]
  repeat' split
  all_goals omega

-- Production IQ preservation, with the external (ringC cap)'s actual admitted argument domain.


/-- The architectural (ringC cap) invariant includes every data queue and control queue144. -/
def IqBoundsC (cap : Nat) (s : St) : Prop :=
  ∃ H L : Vector Nat 145,
    s.regs.get? Register.iq_head = some H ∧ s.regs.get? Register.iq_len = some L ∧
    ∀ q : Nat, q < 145 → H[q]! < cap ∧ L[q]! ≤ cap

private theorem update_other {α : Type} [Inhabited α] {n : Nat}
    (V : Vector α n) (q r : Nat) (x : α) (_hq : q < n) (hr : r < n) (hne : r ≠ q) :
    (vectorUpdate V q x)[r]! = V[r]! := by
  simp [vectorUpdate, Vector.set!, hr, Array.getElem_setIfInBounds_ne, Ne.symm hne]

/-- The actual push either refuses with its entire state unchanged, or invokes unit/tag
    stores only at an admitted queue+slot and preserves all145 rings' bounds. -/
theorem iq_push_preserves_bounds (q : Nat) (hq : q < 145) (u : BitVec 32) (tag : Int)
    (s : St) (hs : (IqBoundsC cap) s)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv (cap := cap) recU recT) :
    ∃ b after, ((pushC cap) q u tag).run s = EStateM.Result.ok b after ∧ (IqBoundsC cap) after := by
  obtain ⟨H,L,hH,hL,hbounds⟩ := hs
  by_cases full : cap ≤ L[q]!
  · exact ⟨false, s, iq_push_full q u tag s L hL full, H,L,hH,hL,hbounds⟩
  · have room : L[q]! < cap := by omega
    obtain ⟨after,hr,regs,hu,ht⟩ := iq_push_room q u tag s recU recT env H L hH hL hq (hbounds q hq).1 room
    refine ⟨true,after,hr,H,vectorUpdate L q (L[q]!+1),?_,?_,?_⟩
    · rw [regs]; simp [Std.ExtDHashMap.get?_insert, hH]
    · rw [regs]; simp
    · intro r hr
      constructor
      · exact (hbounds r hr).1
      · by_cases same : r=q
        · subst r
          rw [vectorUpdate_self L q (L[q]!+1) hq]
          omega
        · rw [update_other L q r (L[q]!+1) hq hr same]
          exact (hbounds r hr).2

/-- The actual pop preserves the (ringC cap) invariant whenever it removes only existing units.
    The assertion is about the real generated transition, not a length-only surrogate. -/
theorem iq_pop_preserves_bounds (q n : Nat) (hq : q < 145) (s : St) (hs : (IqBoundsC cap) s)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L) (hn : n ≤ L[q]!) :
    ∃ after, ((popC cap) q n).run s = EStateM.Result.ok () after ∧ (IqBoundsC cap) after := by
  obtain ⟨H0,L0,hH0,hL0,hbounds⟩ := hs
  have eqH : H0=H := by simpa [hH] using hH0.symm
  have eqL : L0=L := by simpa [hL] using hL0.symm
  subst H0; subst L0
  refine ⟨_,iq_pop_run q n s H L hH hL,
    vectorUpdate H q ((ringC cap) (H[q]!+n)),vectorUpdate L q (L[q]!-n),?_,?_,?_⟩
  · simp [Std.ExtDHashMap.get?_insert]
  · simp
  · intro r hr
    by_cases same : r=q
    · subst r
      rw [vectorUpdate_self H q _ hq,vectorUpdate_self L q _ hq]
      constructor
      · rw [ring_spec]
        have hhead := (hbounds q hq).1
        have hlen := (hbounds q hq).2
        split <;> omega
      · have hlen := (hbounds q hq).2
        omega
    · rw [update_other H q r _ hq hr same,update_other L q r _ hq hr same]
      exact hbounds r hr



/-- Partial-effect safety includes error states; it does not assert callback termination. -/
def PreservesC (cap : Nat) {α : Type} (action : SailM α) : Prop :=
  ∀ s : St, (IqBoundsC cap) s → match action.run s with
    | .ok _ after => (IqBoundsC cap) after
    | .error _ after => (IqBoundsC cap) after

theorem preserves_pure {α : Type} (a : α) : (PreservesC cap) (pure a) := by
  intro s hs
  exact hs

theorem preserves_ite {α : Type} (p : Prop) [Decidable p] (a b : SailM α)
    (ha : (PreservesC cap) a) (hb : (PreservesC cap) b) : (PreservesC cap) (if p then a else b) := by
  split <;> assumption

theorem preserves_bind {α β : Type} (a : SailM α) (f : α → SailM β)
    (ha : (PreservesC cap) a) (hf : ∀ x, (PreservesC cap) (f x)) : (PreservesC cap) (a >>= f) := by
  intro s hs
  have h := ha s hs
  cases he : a.run s with
  | ok x after =>
    have hp : (IqBoundsC cap) after := by simpa [he] using h
    have hf' := hf x after hp
    simpa [EStateM.run, bind, EStateM.bind, show a s = .ok x after from he] using hf'
  | error error after =>
    simpa [EStateM.run, bind, EStateM.bind, show a s = .error error after from he] using h

theorem preserves_read (r : Register) : (PreservesC cap) (Out.readReg r) := by
  intro s hs
  simp [Out.readReg, PreSail.readReg, EStateM.run, bind, EStateM.bind, get, getThe,
    MonadStateOf.get, EStateM.get]
  cases s.regs.get? r <;> exact hs

theorem preserves_other_write (r : Register) (v : RegisterType r)
    (hh : r ≠ Register.iq_head) (hl : r ≠ Register.iq_len) : (PreservesC cap) (Out.writeReg r v) := by
  intro s hs
  obtain ⟨H,L,hH,hL,hbounds⟩ := hs
  simp [Out.writeReg, PreSail.writeReg, EStateM.run, modify, modifyGet,
    MonadStateOf.modifyGet, EStateM.modifyGet]
  exact ⟨H,L,by simpa [Std.ExtDHashMap.get?_insert, hh, Ne.symm hh] using hH,
    by simpa [Std.ExtDHashMap.get?_insert,hl,Ne.symm hl] using hL,hbounds⟩

theorem preserves_forIn {β : Type} (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β))
    (hf : ∀ i hi value, (PreservesC cap) (f i hi value)) : (PreservesC cap) (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => (PreservesC cap) (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply preserves_bind _ _ (hf i hi value)
    intro step
    cases step with
    | done value => exact preserves_pure value
    | yield value => exact ih value
  · intro value i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact preserves_pure value

theorem preserves_push (q : Nat) (hq : q < 145) (u : BitVec 32) (tag : Int)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv (cap := cap) recU recT) : (PreservesC cap) ((pushC cap) q u tag) := by
  intro s hs
  obtain ⟨b,after,h,hi⟩ := iq_push_preserves_bounds q hq u tag s hs recU recT env
  simpa [h] using hi

/-- Updating text-word integrity metadata preserves every queue's (ringC cap) bounds. -/
theorem overlay_preserves_bounds (q w : Nat) (e : BitVec 10) : (PreservesC cap) (iq_word_ov_set q w e) := by
  unfold iq_word_ov_set
  dsimp only [ForIn.forIn, ForIn'.forIn', instForInOfForIn']
  repeat' first
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x

/-- The complete production delivery function preserves all queue bounds, including
    padding, not-ready and empty payloads, overflow/refusal, tag writes and overlay updates.
    The invariant holds also if a model read errors; this is not a termination theorem. -/
theorem iq_deliver_preserves_bounds (q c : Nat) (hq : q < 145)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv (cap := cap) recU recT) : (PreservesC cap) ((deliverC cap rate) q c) := by
  unfold deliverC
  dsimp only [ForIn.forIn, ForIn'.forIn', instForInOfForIn']
  repeat' first
    | exact preserves_push q hq _ _ recU recT env
    | exact overlay_preserves_bounds _ _ _
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x

theorem zero_bounds (cap : Nat) (hcap : 0 < cap) (s : St) :
    IqBoundsC cap { s with regs := (s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145) } := by
  refine ⟨vectorInit 0, vectorInit 0, ?_, ?_, ?_⟩
  · simp [Std.ExtDHashMap.get?_insert]
  · simp
  · intro q hq
    simp [vectorInit, hq]
    exact hcap

end Proofs.Capacity
