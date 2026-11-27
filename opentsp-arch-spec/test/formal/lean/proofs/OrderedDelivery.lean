import Proofs.OrderedImage
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep Proofs.OrderedImage
set_option autoImplicit false
namespace Proofs.OrderedDelivery
open Out.Register
abbrev Flags := Nat × Bool × Bool × Bool

def stepC (cap q c : Nat) (flags : Flags) : SailM Flags := do
  let (budget,go,lost,pushed) := flags
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

def deliveryC (cap rate : Nat) (q : Nat) (c : Nat) : SailM (Bool × Bool) := do
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
        stepC cap q c (budget,go,lost,pushed)
    (pure loop_vars) ) : SailM (Nat × Bool × Bool × Bool) )
  match (GetElem?.getElem! (← Out.readReg fill_fifo) q) with
  | (seg :: rest) =>
    (do
      match seg.units with
      | [] => Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q rest)
      | _ => (pure ()))
  | [] => (pure ())
  (pure (pushed, lost))

theorem delivery_factory (cap rate q c : Nat) :
    deliveryC cap rate q c = deliverC cap rate q c := by
  simp only [deliveryC,deliverC,stepC,bind_assoc,pure_bind,bind_pure,Prod.eta]
  rfl

/-- A physical transfer consumes budget even for padding or a full destination.
    An empty front segment consumes one loop iteration but no transfer budget.
    A not-yet-ready front stops delivery; later segments cannot overtake it. -/
structure Plan where
  budget : Nat
  go : Bool
  lost : Bool
  pushed : Bool
  fifo : List fill_seg
  accepted : List (BitVec 32 × Int)
  offered : List (BitVec 32 × Int)

def flags (p : Plan) : Flags := (p.budget,p.go,p.lost,p.pushed)

def offer (cap used : Nat) (p : Plan) (pair : BitVec 32 × Int) : Plan :=
  if used+p.accepted.length<cap then
    {p with pushed:=true,accepted:=p.accepted++[pair],offered:=p.offered++[pair]}
  else {p with lost:=true,offered:=p.offered++[pair]}

def stepPlan (cap used c : Nat) (p : Plan) : Plan :=
  if p.go && decide (0 < p.budget) then
    match p.fifo with
    | [] => {p with go := false}
    | seg :: rest =>
      if c < seg.ready then {p with go := false}
      else match seg.units with
      | [] => {p with fifo := rest}
      | u :: us =>
        let updated : Plan := {p with budget := (p.budget - 1), fifo := ({seg with units := us, taken := (seg.taken + 1)} :: rest)}
        if u = iq_padding then updated
        else offer cap used updated (u,if 0 ≤ seg.base then seg.base + seg.taken else iq_no_word)
  else p


/-- Successful execution under an explicit reachable-register predicate. -/
def Total (P : St → Prop) {α : Type} (a : SailM α) : Prop :=
  ∀ s, P s → ∃ value after, a.run s = .ok value after ∧ P after

theorem total_pure (P : St → Prop) {α : Type} (value : α) : Total P (pure value) := by
  intro s hs
  exact ⟨value,s,rfl,hs⟩

theorem total_bind (P : St → Prop) {α β : Type} (a : SailM α) (f : α → SailM β)
    (ha : Total P a) (hf : ∀ value, Total P (f value)) : Total P (a >>= f) := by
  intro s hs
  obtain ⟨x,t,he,ht⟩ := ha s hs
  obtain ⟨y,u,hf,hu⟩ := hf x t ht
  refine ⟨y,u,?_,hu⟩
  change a s = .ok x t at he
  simpa [EStateM.run,bind,EStateM.bind,he] using hf

theorem total_read (P : St → Prop) (r : Register)
    (known : ∀ s, P s → ∃ value, s.regs.get? r = some value) : Total P (Out.readReg r) := by
  intro s hs
  obtain ⟨v,hv⟩ := known s hs
  refine ⟨v,s,?_,hs⟩
  simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,get,getThe,
    MonadStateOf.get,EStateM.get,hv,pure,EStateM.pure]

theorem total_write (P : St → Prop) (r : Register) (v : RegisterType r)
    (closed : ∀ s, P s → P {s with regs := s.regs.insert r v}) : Total P (Out.writeReg r v) := by
  intro s hs
  refine ⟨(),{s with regs := s.regs.insert r v},?_,closed s hs⟩
  rfl

theorem total_forIn (P : St → Prop) {β : Type} (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β))
    (hf : ∀ i hi value, Total P (f i hi value)) : Total P (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => Total P (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply total_bind _ _ _ (hf i hi value)
    intro step
    cases step with
    | done value => exact total_pure P value
    | yield value => exact ih value
  · intro value i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact total_pure P value

theorem total_ite (P : St → Prop) {α : Type} (p : Prop) [Decidable p] (a b : SailM α)
    (ha : Total P a) (hb : Total P b) : Total P (if p then a else b) := by
  split <;> assumption

/-- Real overlay writes preserve indexed storage and every register except the
    two actual destinations. A present overlay table supplies its only read. -/
def OverlayFrame (before : St) (s : St) : Prop :=
  s.mem = before.mem ∧ s.sailOutput = before.sailOutput ∧
  (∀ r, r ≠ Register.iq_wov → r ≠ Register.iq_ecc_live → s.regs.get? r = before.regs.get? r) ∧
  ∃ W : Vector (List ov_ent) 145, s.regs.get? Register.iq_wov = some W

theorem overlay_known (before s : St) (h : OverlayFrame before s) :
    ∃ W, s.regs.get? Register.iq_wov = some W := h.2.2.2

theorem overlay_write (before s : St) (h : OverlayFrame before s)
    (r : Register) (v : RegisterType r) (allowed : r=Register.iq_wov ∨ r=Register.iq_ecc_live) :
    OverlayFrame before {s with regs := s.regs.insert r v} := by
  rcases h with ⟨hm,ho,hr,W,hW⟩
  refine ⟨hm,ho,?_,?_⟩
  · intro t tw te
    have ne : t ≠ r := by rcases allowed with eq|eq <;> subst r <;> assumption
    simpa [Std.ExtDHashMap.get?_insert,ne,Ne.symm ne] using hr t tw te
  · rcases allowed with eq|eq
    · subst r; exact ⟨v,by simp⟩
    · subst r; exact ⟨W,by simpa [Std.ExtDHashMap.get?_insert] using hW⟩

theorem overlay_total (before : St) (q w : Nat) (e : BitVec 10) :
    Total (OverlayFrame before) (iq_word_ov_set q w e) := by
  unfold iq_word_ov_set
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact total_read _ Register.iq_wov (overlay_known before)
    | apply total_write <;> intro s hs <;> apply overlay_write before s hs <;> simp
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x


structure Seen (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s : St) : Prop where
  rep : Represents cap view s
  bounds : Proofs.Initialized.Bounds cap s
  head : s.regs.get? Register.iq_head = some H
  len : s.regs.get? Register.iq_len = some L
  fifo : s.regs.get? Register.fill_fifo = some F
  active : ∃ A : Vector Nat 145, s.regs.get? Register.iq_active = some A
  wov : ∃ W : Vector (List ov_ent) 145, s.regs.get? Register.iq_wov = some W

theorem seen_fifo (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F G : Vector (List fill_seg) 145) (s : St) (h : Seen cap view H L F s) :
    Seen cap view H L G {s with regs := s.regs.insert Register.fill_fifo G} := by
  obtain ⟨A,hA⟩ := h.active
  obtain ⟨W,hW⟩ := h.wov
  refine ⟨h.rep,?_,?_,?_,?_,?_,?_⟩
  · exact Proofs.Initialized.preserves_other_write (cap:=cap) Register.fill_fifo G (by decide) (by decide) s h.bounds
  · simpa [Std.ExtDHashMap.get?_insert] using h.head
  · simpa [Std.ExtDHashMap.get?_insert] using h.len
  · simp
  · exact ⟨A,by simpa [Std.ExtDHashMap.get?_insert] using hA⟩
  · exact ⟨W,by simpa [Std.ExtDHashMap.get?_insert] using hW⟩

theorem seen_active (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s : St) (h : Seen cap view H L F s)
    (A : Vector Nat 145) :
    Seen cap view H L F {s with regs := s.regs.insert Register.iq_active A} := by
  obtain ⟨W,hW⟩ := h.wov
  refine ⟨h.rep,?_,?_,?_,?_,?_,?_⟩
  · exact Proofs.Initialized.preserves_other_write (cap:=cap) Register.iq_active A (by decide) (by decide) s h.bounds
  · simpa [Std.ExtDHashMap.get?_insert] using h.head
  · simpa [Std.ExtDHashMap.get?_insert] using h.len
  · simpa [Std.ExtDHashMap.get?_insert] using h.fifo
  · exact ⟨A,by simp⟩
  · exact ⟨W,by simpa [Std.ExtDHashMap.get?_insert] using hW⟩

theorem seen_overlay (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s t : St) (h : Seen cap view H L F s)
    (frame : OverlayFrame s t) : Seen cap view H L F t := by
  rcases frame with ⟨hm,ho,hr,W,hW⟩
  have hd := hr Register.iq_head (by decide) (by decide)
  have hl := hr Register.iq_len (by decide) (by decide)
  have hf := hr Register.fill_fifo (by decide) (by decide)
  have ha := hr Register.iq_active (by decide) (by decide)
  refine ⟨?_,?_,hd.trans h.head,hl.trans h.len,hf.trans h.fifo,?_,⟨W,hW⟩⟩
  · simpa only [Represents,Consistent,Initialized,tags,readValue,Present,hm] using h.rep
  · obtain ⟨H0,L0,hH0,hL0,hb⟩ := h.bounds
    refine ⟨H0,L0,hd.trans hH0,hl.trans hL0,?_⟩
    simpa only [Initialized,Present,hm] using hb
  · obtain ⟨A,hA⟩ := h.active
    exact ⟨A,ha.trans hA⟩

theorem seen_overlay_run (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s : St) (h : Seen cap view H L F s)
    (q w : Nat) (e : BitVec 10) :
    ∃ after, (iq_word_ov_set q w e).run s = .ok () after ∧ Seen cap view H L F after := by
  have start : OverlayFrame s s := ⟨rfl,rfl,(by intros; rfl),h.wov⟩
  obtain ⟨v,after,he,hf⟩ := overlay_total s q w e s start
  exact ⟨after,he,seen_overlay cap view H L F s after h hf⟩

theorem push_seen (cap q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : Proofs.Initialized.WritePorts cap) (s : St) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145)
    (h : Seen cap view H L F s) (hh : H[q]!<cap) (room : L[q]!<cap) :
    ∃ after, (pushC cap q u tag).run s = .ok true after ∧
      Seen cap (appendStore cap view q H[q]! L[q]! u tag)
        H (vectorUpdate L q (L[q]!+1)) F after := by
  obtain ⟨after,he,rep,bounds,regs,_⟩ :=
    push_room_exact cap q hq u tag ports s view h.rep h.bounds H L h.head h.len hh room
  obtain ⟨A,hA⟩ := h.active
  obtain ⟨W,hW⟩ := h.wov
  refine ⟨after,he,rep,bounds,?_,?_,?_,?_,?_⟩
  · simpa [regs,Std.ExtDHashMap.get?_insert] using h.head
  · simp [regs]
  · simpa [regs,Std.ExtDHashMap.get?_insert] using h.fifo
  · exact ⟨A,by simpa [regs,Std.ExtDHashMap.get?_insert] using hA⟩
  · exact ⟨W,by simpa [regs,Std.ExtDHashMap.get?_insert] using hW⟩


def segmentTag (seg : fill_seg) : Int :=
  if 0 ≤ seg.base then seg.base + seg.taken else iq_no_word

def overlayC (q : Nat) (seg : fill_seg) : SailM Unit := do
  if ((seg.base ≥b 0) : Bool) then
    let e := ov_nth seg.ovw (Int.ediv seg.taken 4)
    if e != 0#10 then
      iq_word_ov_set q (Int.ediv (Max.max (seg.base + seg.taken) 0) 4) e
    else pure ()
  else pure ()

def markC (q c : Nat) (seg : fill_seg) : SailM Unit := do
  Out.writeReg iq_active (vectorUpdate (← Out.readReg iq_active) q c)
  overlayC q seg

def commitC (cap q c : Nat) (seg : fill_seg) (u : BitVec 32) : SailM Bool := do
  let tag : Int := if ((seg.base ≥b 0) : Bool) then seg.base + seg.taken else iq_no_word
  if (← pushC cap q u tag) then
    markC q c seg
    pure true
  else pure false

theorem overlayC_run (cap q : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s : St) (h : Seen cap view H L F s) (seg : fill_seg) :
    ∃ after, (overlayC q seg).run s = .ok () after ∧ Seen cap view H L F after := by
  unfold overlayC
  split
  · dsimp only
    split
    · exact seen_overlay_run cap view H L F s h _ _ _
    · exact ⟨s,rfl,h⟩
  · exact ⟨s,rfl,h⟩

theorem mark_run (cap q c : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s : St) (h : Seen cap view H L F s) (seg : fill_seg) :
    ∃ after, (markC q c seg).run s = .ok () after ∧ Seen cap view H L F after := by
  obtain ⟨A,hA⟩ := h.active
  let middle : St := {s with regs := s.regs.insert Register.iq_active (vectorUpdate A q c)}
  have hm : Seen cap view H L F middle := seen_active cap view H L F s h _
  obtain ⟨after,he,ha⟩ := overlayC_run cap q view H L F middle hm seg
  refine ⟨after,?_,ha⟩
  change overlayC q seg middle = .ok () after at he
  simpa [markC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,
    EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,
    MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,
    EStateM.modifyGet,hA,middle] using he

theorem commit_room (cap q c : Nat) (hq : q<145) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (h : Seen cap view H L F s) (ports : Proofs.Initialized.WritePorts cap)
    (hh : H[q]!<cap) (room : L[q]!<cap) (seg : fill_seg) (u : BitVec 32) :
    ∃ after, (commitC cap q c seg u).run s = .ok true after ∧
      Seen cap (appendStore cap view q H[q]! L[q]! u (segmentTag seg))
        H (vectorUpdate L q (L[q]!+1)) F after := by
  obtain ⟨middle,he,hm⟩ := push_seen cap q hq u (segmentTag seg) ports s view H L F h hh room
  obtain ⟨after,ha,hs⟩ := mark_run cap q c _ _ _ _ middle hm seg
  refine ⟨after,?_,hs⟩
  change pushC cap q u (segmentTag seg) s = .ok true middle at he
  change markC q c seg middle = .ok () after at ha
  have tag_eq : (if ((seg.base ≥b 0) : Bool) then seg.base + seg.taken else iq_no_word) =
      segmentTag seg := by simp [segmentTag]
  simp only [commitC,tag_eq]
  simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,he,ha]

theorem commit_full (cap q c : Nat) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (h : Seen cap view H L F s) (full : cap≤L[q]!) (seg : fill_seg) (u : BitVec 32) :
    (commitC cap q c seg u).run s = .ok false s := by
  have he := Proofs.Capacity.iq_push_full q u (segmentTag seg) s L h.len full
  change pushC cap q u (segmentTag seg) s = .ok false s at he
  have tag_eq : (if ((seg.base ≥b 0) : Bool) then seg.base + seg.taken else iq_no_word) =
      segmentTag seg := by simp [segmentTag]
  simp only [commitC,tag_eq]
  simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,he]


def stepD (cap q c : Nat) (flags : Flags) : SailM Flags := do
  let (budget,go,lost,pushed) := flags
  if go && ((budget >b 0) : Bool) then
    match (← Out.readReg fill_fifo)[q]! with
    | [] => pure (budget,false,lost,pushed)
    | seg :: rest =>
      if ((seg.ready >b c) : Bool) then pure (budget,false,lost,pushed)
      else match seg.units with
      | [] =>
        Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q rest)
        pure (budget,go,lost,pushed)
      | u :: us =>
        Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q
          ({seg with units := us, taken := (seg.taken +i 1)} :: rest))
        let remain : Nat := Max.max (budget -i 1) 0
        if u != iq_padding then
          if (← commitC cap q c seg u) then pure (remain,go,lost,true)
          else pure (remain,go,true,pushed)
        else pure (remain,go,lost,pushed)
  else pure flags


theorem bind_list_match {α β γ : Type} (xs : List α) (nil : SailM β)
    (cons : α → List α → SailM β) (f : β → SailM γ) :
    ((match xs with | [] => nil | h::t => cons h t) >>= f) =
      (match xs with | [] => nil >>= f | h::t => cons h t >>= f) := by
  cases xs <;> rfl


theorem same_bind {α β : Type} (a : SailM α) (f g : α → SailM β)
    (h : ∀ x, f x = g x) : (a >>= f) = (a >>= g) := by rw [funext h]

theorem step_factory (cap q c : Nat) (flags : Flags) :
    stepC cap q c flags = stepD cap q c flags := by
  rcases flags with ⟨budget,go,lost,pushed⟩
  simp only [stepC,stepD,commitC,markC,overlayC,bind_assoc,pure_bind,bind_pure,
    Prod.eta,bind_if,ite_true,Bool.false_eq_true,ite_false]
  split
  · apply same_bind
    intro F
    cases hf : F[q]! with
    | nil => simp only [hf]
    | cons seg rest =>
      simp only [hf]
      split
      · rfl
      · cases hu : seg.units with
        | nil => simp [hu,bind_assoc,pure_bind,bind_pure,Prod.eta]
        | cons u us =>
          simp only [hu,bind_assoc,pure_bind,bind_pure,Prod.eta,bind_if,ite_true,Bool.false_eq_true,ite_false]
  · rfl


/-- Ordered concrete storage interpretation of exactly the accepted transfer trace. -/
def historyStore (cap : Nat) (view : CView) (q h used : Nat) :
    List (BitVec 32 × Int) → CView
  | [] => view
  | pair :: rest => historyStore cap (appendStore cap view q h used pair.1 pair.2) q h (used+1) rest

theorem history_append (cap : Nat) (view : CView) (q h used : Nat)
    (xs : List (BitVec 32 × Int)) (u : BitVec 32) (tag : Int) :
    historyStore cap view q h used (xs ++ [(u,tag)]) =
      appendStore cap (historyStore cap view q h used xs) q h (used+xs.length) u tag := by
  induction xs generalizing view used with
  | nil => rfl
  | cons pair rest ih =>
    simp only [List.cons_append,historyStore,ih,List.length_cons]
    congr 1 <;> omega

theorem history_content (cap : Nat) (view : CView) (q h used : Nat)
    (xs : List (BitVec 32 × Int)) (hh : h<cap) (room : used+xs.length≤cap) :
    content cap (historyStore cap view q h used xs) q h (used+xs.length) =
      content cap view q h used ++ xs := by
  induction xs generalizing view used with
  | nil => simp [historyStore]
  | cons pair rest ih =>
    have one : used<cap := by simp only [List.length_cons] at room; omega
    have hr : used+1+rest.length≤cap := by simp only [List.length_cons] at room; omega
    rw [historyStore]
    have eq := ih (appendStore cap view q h used pair.1 pair.2) (used+1) hr
    have ix : used+(pair::rest).length=used+1+rest.length := by simp; omega
    rw [ix,eq,append_content cap view q h used pair.1 pair.2 hh one]
    simp only [List.append_assoc,List.singleton_append]

theorem update_current {α : Type} [Inhabited α] (V : Vector α 145) (q : Nat) (hq : q<145)
    (x : α) (hx : V[q]! =x) : vectorUpdate V q x=V := by
  subst x
  cases V
  simp_all [vectorUpdate,Vector.set!,Array.setIfInBounds]

def stepStore (cap : Nat) (view : CView) (q h used : Nat) (before after : Plan) :=
  historyStore cap view q h used (after.accepted.drop before.accepted.length)

theorem step_plan_exact (cap q c used : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (seen : Seen cap view H L F s) (p : Plan)
    (fifo : F[q]! =p.fifo) (count : used+p.accepted.length=L[q]!) (hh : H[q]!<cap) :
    let next := stepPlan cap used c p
    ∃ after, (stepC cap q c (flags p)).run s = .ok (flags next) after ∧
      Seen cap (stepStore cap view q H[q]! L[q]! p next)
        H (vectorUpdate L q (used+next.accepted.length))
        (vectorUpdate F q next.fifo) after := by
  dsimp only
  rw [step_factory]
  have sameL : vectorUpdate L q (used+p.accepted.length)=L := update_current L q hq _ count.symm
  have sameF : vectorUpdate F q p.fifo=F := update_current F q hq _ fifo
  by_cases go : p.go=true
  · by_cases budget : 0<p.budget
    · cases hf : p.fifo with
      | nil =>
        refine ⟨s,?_,?_⟩
        · simp [stepD,flags,stepPlan,offer,go,budget,hf,Out.readReg,PreSail.readReg,
            EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,
            MonadStateOf.get,EStateM.get,seen.fifo,fifo]
        · simpa [stepPlan,offer,go,budget,hf,stepStore,historyStore,sameL,
            show vectorUpdate F q []=F by simpa [hf] using sameF] using seen
      | cons seg rest =>
        by_cases ready : seg.ready≤c
        · cases hu : seg.units with
          | nil =>
            let middle : St := {s with regs := s.regs.insert Register.fill_fifo (vectorUpdate F q rest)}
            have hm := seen_fifo cap view H L F (vectorUpdate F q rest) s seen
            refine ⟨middle,?_,?_⟩
            · simp [stepD,flags,stepPlan,offer,go,budget,hf,ready,hu,Out.readReg,Out.writeReg,
                PreSail.readReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,
                pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,modify,
                modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,seen.fifo,fifo,middle,
                show ¬c<seg.ready by omega,show ¬seg.ready>c by omega]
            · simpa [stepPlan,offer,go,budget,hf,ready,hu,stepStore,historyStore,sameL,
                show ¬c<seg.ready by omega] using hm
          | cons u us =>
            let restF := vectorUpdate F q ({seg with units:=us,taken:=seg.taken+1}::rest)
            let middle : St := {s with regs := s.regs.insert Register.fill_fifo restF}
            have hm : Seen cap view H L restF middle := seen_fifo cap view H L F restF s seen
            by_cases padding : u=iq_padding
            · refine ⟨middle,?_,?_⟩
              · simp [stepD,flags,stepPlan,offer,go,budget,hf,ready,hu,padding,Out.readReg,Out.writeReg,
                  PreSail.readReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
                  get,getThe,MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,
                  EStateM.modifyGet,seen.fifo,fifo,middle,restF,
                  show ¬c<seg.ready by omega,show ¬seg.ready>c by omega]
              · simpa [stepPlan,offer,go,budget,hf,hu,padding,stepStore,historyStore,sameL,restF,
                  show ¬c<seg.ready by omega] using hm
            · by_cases room : used+p.accepted.length<cap
              · obtain ⟨after,he,ha⟩ := commit_room cap q c hq view H L restF middle hm ports hh
                    (by omega) seg u
                refine ⟨after,?_,?_⟩
                · have he' : commitC cap q c seg u middle=.ok true after := he
                  simp [stepD,flags,stepPlan,offer,go,budget,hf,ready,hu,padding,room,Out.readReg,Out.writeReg,
                    PreSail.readReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
                    get,getThe,MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,
                    EStateM.modifyGet,seen.fifo,fifo,middle,restF,he',
                    show ¬c<seg.ready by omega,show ¬seg.ready>c by omega]
                · simpa [stepPlan,offer,go,budget,hf,hu,padding,room,stepStore,historyStore,
                    List.drop_append,List.drop_length,segmentTag,restF,count,
                    show L[q]!<cap by omega,
                    show used+(p.accepted.length+1)=L[q]!+1 by omega,
                    show ¬c<seg.ready by omega,Nat.add_assoc] using ha
              · have he := commit_full cap q c view H L restF middle hm (by omega) seg u
                refine ⟨middle,?_,?_⟩
                · have he' : commitC cap q c seg u middle=.ok false middle := he
                  simp [stepD,flags,stepPlan,offer,go,budget,hf,ready,hu,padding,room,Out.readReg,Out.writeReg,
                    PreSail.readReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
                    get,getThe,MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,
                    EStateM.modifyGet,seen.fifo,fifo,middle,restF,he',
                    show ¬c<seg.ready by omega,show ¬seg.ready>c by omega]
                · simpa [stepPlan,offer,go,budget,hf,hu,padding,room,stepStore,historyStore,sameL,restF,
                    show ¬c<seg.ready by omega] using hm
        · refine ⟨s,?_,?_⟩
          · simp [stepD,flags,stepPlan,offer,go,budget,hf,ready,Out.readReg,PreSail.readReg,
              EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,
              MonadStateOf.get,EStateM.get,seen.fifo,fifo,show c<seg.ready by omega]
          · simpa [stepPlan,offer,go,budget,hf,ready,stepStore,historyStore,sameL,
              show vectorUpdate F q (seg::rest)=F by simpa [hf] using sameF,
              show c<seg.ready by omega] using seen
    · refine ⟨s,?_,?_⟩
      · simp [stepD,flags,stepPlan,offer,go,budget]
      · simpa [stepPlan,offer,go,budget,stepStore,historyStore,sameL,sameF] using seen
  · refine ⟨s,?_,?_⟩
    · simp [stepD,flags,stepPlan,offer,go]
    · simpa [stepPlan,offer,go,stepStore,historyStore,sameL,sameF] using seen


def iterateFlags (f : Flags → SailM Flags) : Nat → Flags → SailM Flags
  | 0, value => pure value
  | count+1, value => do
    let next ← f value
    iterateFlags f count next

def flagsBody (n : Nat) (f : Flags → SailM Flags)
    (_i : Int) (_hi : _i ∈ ({stop := (n:Int)} : IntRange)) (value : Flags) :
    SailM (ForInStep Flags) := do
  let next ← f value
  pure (ForInStep.yield next)

theorem native_flags_loop (n start count : Nat) (hn : start+count=n+1)
    (f : Flags → SailM Flags) (value : Flags) :
    IntRange.forIn'.loop ({stop := (n:Int)} : IntRange) (flagsBody n f) value
      (start:Int) (by simp) = iterateFlags f count value := by
  induction count generalizing start value with
  | zero =>
    have hnot : ¬(start:Int) ∈ ({stop := (n:Int)} : IntRange) := by
      simp only [Membership.mem]
      simp
      omega
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hnot,iterateFlags]
  | succ count ih =>
    have hmem : (start:Int) ∈ ({stop := (n:Int)} : IntRange) := by
      simp only [Membership.mem]
      simp
      omega
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hmem,flagsBody,iterateFlags,bind_assoc,pure_bind]
    have hnext : start+1+count=n+1 := by omega
    have eq := funext fun next => ih (start+1) hnext next
    simpa only [Int.natCast_add,Int.natCast_add_one] using
      congrArg (fun rest : Flags → SailM Flags => f value >>= rest) eq

def nativeFlags (upper : Nat) (f : Flags → SailM Flags) (value : Flags) : SailM Flags := do
  let mut value := value
  for _i in [0:(upper:Int):1]i do
    value ← f value
  pure value

theorem native_flags_eq (upper : Nat) (f : Flags → SailM Flags) (value : Flags) :
    nativeFlags upper f value = iterateFlags f (upper+1) value := by
  unfold nativeFlags
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  simp only [bind_pure,pure_bind]
  apply Eq.trans (b := IntRange.forIn' ({stop := (upper:Int)} : IntRange) value
    (flagsBody upper f))
  · rfl
  · exact native_flags_loop upper 0 (upper+1) (by omega) f value

def finishC (q : Nat) : SailM Unit := do
  match (← Out.readReg fill_fifo)[q]! with
  | [] => pure ()
  | seg::rest =>
    match seg.units with
    | [] => Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q rest)
    | _ => pure ()

def deliveryVia (cap rate q c : Nat) : SailM (Bool × Bool) := do
  let F ← Out.readReg fill_fifo
  let value ← nativeFlags (rate + list_len F[q]!) (stepC cap q c) (rate,true,false,false)
  finishC q
  pure (value.2.2.2,value.2.2.1)

theorem whole_delivery_factory (cap rate q c : Nat) :
    deliveryC cap rate q c = deliveryVia cap rate q c := by
  simp only [deliveryC,deliveryVia,nativeFlags,finishC,bind_assoc,pure_bind,bind_pure,Prod.eta,
    Int.natCast_add]
  apply same_bind
  intro F
  apply same_bind
  intro value
  apply same_bind
  intro G
  cases hf : G[q]! with
  | nil => simp [hf]
  | cons seg rest =>
    cases hu : seg.units <;> simp [hf,hu,bind_assoc,pure_bind,bind_pure]

def planLoop (cap used c : Nat) : Nat → Plan → Plan
  | 0, p => p
  | count+1, p => planLoop cap used c count (stepPlan cap used c p)

theorem project_if {α β : Type} (f : α → β) (p : Prop) [Decidable p] (x y : α) :
    f (if p then x else y) = if p then f x else f y := by
  by_cases h : p <;> simp [h]

theorem step_prefix (cap used c : Nat) (p : Plan) :
    ∃ extra, (stepPlan cap used c p).accepted=p.accepted++extra := by
  refine ⟨(stepPlan cap used c p).accepted.drop p.accepted.length,?_⟩
  cases hf : p.fifo with
  | nil => simp [stepPlan,offer,hf,project_if]
  | cons seg rest =>
    cases hu : seg.units with
    | nil => simp [stepPlan,offer,hf,hu,project_if]
    | cons u us =>
      by_cases go : p.go=true <;>
      by_cases budget : 0<p.budget <;>
      by_cases ready : c<seg.ready <;>
      by_cases padding : u=iq_padding <;>
      by_cases room : used+p.accepted.length<cap <;>
      simp [stepPlan,offer,hf,hu,go,budget,ready,padding,room,project_if]


theorem history_concat (cap : Nat) (view : CView) (q h used : Nat)
    (xs ys : List (BitVec 32 × Int)) :
    historyStore cap view q h used (xs++ys) =
      historyStore cap (historyStore cap view q h used xs) q h (used+xs.length) ys := by
  induction xs generalizing view used with
  | nil => rfl
  | cons pair rest ih =>
    simp only [List.cons_append,historyStore,ih,List.length_cons]
    congr 1 <;> omega

theorem step_store_history (cap : Nat) (view : CView) (q h used : Nat) (p : Plan) (c : Nat) :
    stepStore cap (historyStore cap view q h used p.accepted) q h
      (used+p.accepted.length) p (stepPlan cap used c p) =
        historyStore cap view q h used (stepPlan cap used c p).accepted := by
  obtain ⟨extra,eq⟩ := step_prefix cap used c p
  simp only [stepStore,eq,List.drop_append,List.drop_length,Nat.sub_self,List.drop_zero,
    List.nil_append]
  exact (history_concat cap view q h used p.accepted extra).symm

theorem update_twice {α : Type} [Inhabited α] (V : Vector α 145) (q : Nat) (x y : α) (hq : q<145) :
    vectorUpdate (vectorUpdate V q x) q y=vectorUpdate V q y := by
  simp [vectorUpdate,Vector.set!,hq]

theorem update_get {α : Type} [Inhabited α] (V : Vector α 145) (q : Nat) (x : α) (hq : q<145) :
    (vectorUpdate V q x)[q]! = x := by simp [vectorUpdate,Vector.set!,hq]

def LoopInv (cap q used : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (p : Plan) (s : St) : Prop :=
  Seen cap (historyStore cap view q H[q]! used p.accepted) H
    (vectorUpdate L q (used+p.accepted.length)) (vectorUpdate F q p.fifo) s

theorem one_iteration (cap q c used : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St) (p : Plan)
    (inv : LoopInv cap q used view H L F p s) (hh : H[q]!<cap) :
    ∃ after, (stepC cap q c (flags p)).run s = .ok (flags (stepPlan cap used c p)) after ∧
      LoopInv cap q used view H L F (stepPlan cap used c p) after := by
  obtain ⟨after,he,ha⟩ := step_plan_exact cap q c used hq ports _ H
    (vectorUpdate L q (used+p.accepted.length)) (vectorUpdate F q p.fifo) s inv p
    (update_get F q p.fifo hq) (update_get L q (used+p.accepted.length) hq).symm hh
  refine ⟨after,he,?_⟩
  simpa only [LoopInv,vectorUpdate_self _ _ _ hq,step_store_history,
    update_twice _ q _ _ hq] using ha

theorem loop_exact (count cap q c used : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St) (p : Plan)
    (inv : LoopInv cap q used view H L F p s) (hh : H[q]!<cap) :
    ∃ after, (iterateFlags (stepC cap q c) count (flags p)).run s =
        .ok (flags (planLoop cap used c count p)) after ∧
      LoopInv cap q used view H L F (planLoop cap used c count p) after := by
  induction count generalizing p s with
  | zero => exact ⟨s,rfl,inv⟩
  | succ count ih =>
    obtain ⟨middle,he,hm⟩ := one_iteration cap q c used hq ports view H L F s p inv hh
    obtain ⟨after,hr,ha⟩ := ih middle (stepPlan cap used c p) hm
    refine ⟨after,?_,ha⟩
    change stepC cap q c (flags p) s=.ok (flags (stepPlan cap used c p)) middle at he
    simpa [iterateFlags,planLoop,EStateM.run,bind,EStateM.bind,he] using hr


def cleanFront : List fill_seg → List fill_seg
  | [] => []
  | seg::rest => match seg.units with | [] => rest | _::_ => seg::rest

def finishPlan (p : Plan) : Plan := {p with fifo := cleanFront p.fifo}

theorem finish_exact (cap q : Nat) (hq : q<145) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (h : Seen cap view H L F s) :
    ∃ after, (finishC q).run s=.ok () after ∧
      Seen cap view H L (vectorUpdate F q (cleanFront F[q]!)) after := by
  cases hf : F[q]! with
  | nil =>
    have same : vectorUpdate F q []=F := update_current F q hq [] hf
    refine ⟨s,?_,?_⟩
    · simp [finishC,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,
        EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,h.fifo,hf]
    · simpa [cleanFront,hf,same] using h
  | cons seg rest =>
    cases hu : seg.units with
    | nil =>
      let after : St := {s with regs := s.regs.insert Register.fill_fifo (vectorUpdate F q rest)}
      refine ⟨after,?_,?_⟩
      · simp [finishC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,EStateM.run,
          bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,modify,
          modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,h.fifo,hf,hu,after]
      · simpa [cleanFront,hf,hu] using seen_fifo cap view H L F (vectorUpdate F q rest) s h
    | cons u us =>
      have same : vectorUpdate F q (seg::rest)=F := update_current F q hq _ hf
      refine ⟨s,?_,?_⟩
      · simp [finishC,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,
          EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,h.fifo,hf,hu]
      · simpa [cleanFront,hf,hu,same] using h

def startPlan (rate : Nat) (fifo : List fill_seg) : Plan :=
  ⟨rate,true,false,false,fifo,[],[]⟩

theorem seen_bound (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s : St) (h : Seen cap view H L F s)
    (q : Nat) (hq : q<145) : H[q]!<cap ∧ L[q]!≤cap := by
  obtain ⟨H0,L0,hH,hL,hb⟩ := h.bounds
  have eqH : H0=H := Option.some.inj (hH.symm.trans h.head)
  have eqL : L0=L := Option.some.inj (hL.symm.trans h.len)
  subst H0; subst L0
  exact ⟨(hb q hq).1,(hb q hq).2.1⟩

/-- Whole actual delivery: exact SAME initial store history, captured FIFO consumption,
    accepted pair sequence, length and pushed/lost result, including padding/refusal.
    The budget is physical units and the real inclusive iterator runs rate+segments+1
    iterations, followed by the actual one-empty-front cleanup. -/
theorem delivery_exact (cap rate q c : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (seen : Seen cap view H L F s) :
    let result := finishPlan (planLoop cap L[q]! c (rate+(list_len F[q]!)+1) (startPlan rate F[q]!))
    ∃ after, (deliverC cap rate q c).run s = .ok (result.pushed,result.lost) after ∧
      Seen cap (historyStore cap view q H[q]! L[q]! result.accepted) H
        (vectorUpdate L q (L[q]!+result.accepted.length)) (vectorUpdate F q result.fifo) after ∧
      content cap (historyStore cap view q H[q]! L[q]! result.accepted)
        q H[q]! (L[q]!+result.accepted.length) =
        content cap view q H[q]! L[q]! ++ result.accepted := by
  dsimp only
  let start := startPlan rate F[q]!
  let done := planLoop cap L[q]! c (rate+list_len F[q]!+1) start
  have init : LoopInv cap q L[q]! view H L F start s := by
    have hL := update_current L q hq L[q]! rfl
    have hF := update_current F q hq F[q]! rfl
    simpa [LoopInv,start,startPlan,historyStore,hL,hF] using seen
  have hh := (seen_bound cap view H L F s seen q hq).1
  obtain ⟨middle,he,hm⟩ := loop_exact (rate+list_len F[q]!+1) cap q c L[q]! hq ports
    view H L F s start init hh
  obtain ⟨after,hf,ha⟩ := finish_exact cap q hq _ _ _ _ middle hm
  have final : LoopInv cap q L[q]! view H L F (finishPlan done) after := by
    simpa only [LoopInv,finishPlan,update_get _ q _ hq,update_twice _ q _ _ hq] using ha
  have bound := (seen_bound _ _ _ _ _ _ final q hq).2
  have lengthBound : L[q]!+(finishPlan done).accepted.length≤cap := by
    simpa only [update_get _ q _ hq] using bound
  refine ⟨after,?_,final,history_content cap view q H[q]! L[q]!
    (finishPlan done).accepted hh lengthBound⟩
  rw [← delivery_factory,whole_delivery_factory]
  have he' : iterateFlags (stepC cap q c) (rate+list_len F[q]!+1) (flags start) s
      = .ok (flags done) middle := he
  have hf' : finishC q middle=.ok () after := hf
  have hstart : iterateFlags (stepC cap q c) (rate+list_len F[q]!+1)
      (rate,true,false,false) s=.ok (flags done) middle := by
    simpa only [start,startPlan,flags] using he'
  simp [deliveryVia,native_flags_eq,start,startPlan,finishPlan,flags,Out.readReg,
    PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,
    MonadStateOf.get,EStateM.get,seen.fifo,hstart,hf',done]

theorem actual_delivery_exact (q c : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts iq_cap) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (seen : Seen iq_cap view H L F s) :
    let result := finishPlan (planLoop iq_cap L[q]! c (iq_rate+(list_len F[q]!)+1) (startPlan iq_rate F[q]!))
    ∃ after, (iq_deliver q c).run s = .ok (result.pushed,result.lost) after ∧
      Seen iq_cap (historyStore iq_cap view q H[q]! L[q]! result.accepted) H
        (vectorUpdate L q (L[q]!+result.accepted.length)) (vectorUpdate F q result.fifo) after ∧
      content iq_cap (historyStore iq_cap view q H[q]! L[q]! result.accepted)
        q H[q]! (L[q]!+result.accepted.length) =
        content iq_cap view q H[q]! L[q]! ++ result.accepted := by
  simpa only [deliver_production] using delivery_exact iq_cap iq_rate q c hq ports view H L F s seen


/-- Eligible nonpadding offers include those lost to a full IQ. Accepted data is
    exactly their capacity-limited prefix, rather than an arbitrary append trace. -/
def Prefix (free : Nat) (p : Plan) : Prop :=
  p.accepted = p.offered.take free ∧
  p.lost = decide (free<p.offered.length) ∧
  p.pushed = decide (p.accepted≠[])

theorem take_snoc {α : Type} (xs : List α) (x : α) (n : Nat) :
    (xs++[x]).take n = if xs.length<n then xs.take n++[x] else xs.take n := by
  by_cases h : xs.length<n
  · have ht : ([x] : List α).length≤n-xs.length := by simp; omega
    simp only [List.take_append,if_pos h,List.take_of_length_le ht]
  · have hs : n-xs.length=0 := by omega
    simp [List.take_append,h,hs]

theorem offer_prefix (cap used : Nat) (hu : used≤cap) (p : Plan) (pair : BitVec 32 × Int)
    (hp : Prefix (cap-used) p) : Prefix (cap-used) (offer cap used p pair) := by
  rcases hp with ⟨ha,hl,hpush⟩
  have hlen : p.accepted.length=min (cap-used) p.offered.length := by
    rw [ha,List.length_take]
  have roomEq : (used+p.accepted.length<cap) ↔ p.offered.length<cap-used := by omega
  by_cases room : used+p.accepted.length<cap
  · have hsmall := roomEq.mp room
    simp only [Prefix,offer,if_pos room]
    refine ⟨?_,?_,?_⟩
    · simpa only [take_snoc,if_pos hsmall] using congrArg (fun xs => xs++[pair]) ha
    · have oldFalse : ¬cap-used<p.offered.length := by omega
      have newFalse : ¬cap-used<(p.offered++[pair]).length := by simp; omega
      simpa only [hl,decide_eq_false oldFalse,decide_eq_false newFalse]
    · simp
  · have hfull : ¬p.offered.length<cap-used := by omega
    have overflow : cap-used < (p.offered++[pair]).length := by simp; omega
    simp only [Prefix,offer,if_neg room]
    refine ⟨?_,?_,hpush⟩
    · simpa only [take_snoc,if_neg hfull] using ha
    · simp; omega


theorem prefix_metadata (free : Nat) (p : Plan) (budget : Nat) (fifo : List fill_seg) :
    Prefix free {p with budget:=budget,fifo:=fifo} ↔ Prefix free p := by rfl

theorem step_prefix_invariant (cap used c : Nat) (hu : used≤cap) (p : Plan)
    (hp : Prefix (cap-used) p) : Prefix (cap-used) (stepPlan cap used c p) := by
  unfold stepPlan
  split
  · cases hf : p.fifo with
    | nil => simpa [hf,Prefix] using hp
    | cons seg rest =>
      simp only [hf]
      split
      · simpa [Prefix] using hp
      · cases hi : seg.units with
        | nil => simpa [hi,Prefix] using hp
        | cons u us =>
          simp only [hi]
          split
          · simpa [Prefix] using hp
          · apply offer_prefix cap used hu
            simpa [Prefix] using hp
  · exact hp

theorem plan_prefix (count cap used c : Nat) (hu : used≤cap) (p : Plan)
    (hp : Prefix (cap-used) p) : Prefix (cap-used) (planLoop cap used c count p) := by
  induction count generalizing p with
  | zero => exact hp
  | succ count ih => exact ih (stepPlan cap used c p) (step_prefix_invariant cap used c hu p hp)

theorem initial_prefix (cap used rate : Nat) (fifo : List fill_seg) :
    Prefix (cap-used) (startPlan rate fifo) := by simp [Prefix,startPlan]

theorem finish_prefix (free : Nat) (p : Plan) (hp : Prefix free p) : Prefix free (finishPlan p) := hp


/-- Architectural prefix/loss statement composed with the SAME actual whole action. -/
theorem delivery_prefix_exact (cap rate q c : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (seen : Seen cap view H L F s) :
    let result := finishPlan (planLoop cap L[q]! c (rate+list_len F[q]!+1) (startPlan rate F[q]!))
    ∃ after, (deliverC cap rate q c).run s = .ok (result.pushed,result.lost) after ∧
      Seen cap (historyStore cap view q H[q]! L[q]! result.accepted) H
        (vectorUpdate L q (L[q]!+result.accepted.length)) (vectorUpdate F q result.fifo) after ∧
      Prefix (cap-L[q]!) result ∧
      content cap (historyStore cap view q H[q]! L[q]! result.accepted)
        q H[q]! (L[q]!+result.accepted.length) =
        content cap view q H[q]! L[q]! ++ result.offered.take (cap-L[q]!) := by
  dsimp only
  obtain ⟨after,he,hs,hc⟩ := delivery_exact cap rate q c hq ports view H L F s seen
  have hu := (seen_bound cap view H L F s seen q hq).2
  have hprefix := finish_prefix (cap-L[q]!) _
    (plan_prefix (rate+list_len F[q]!+1) cap L[q]! c hu _ (initial_prefix cap L[q]! rate F[q]!))
  exact ⟨after,he,hs,hprefix,hc.trans (congrArg (fun xs => content cap view q H[q]! L[q]! ++ xs) hprefix.1)⟩

theorem actual_delivery_prefix_exact (q c : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts iq_cap) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145) (s : St)
    (seen : Seen iq_cap view H L F s) :
    let result := finishPlan (planLoop iq_cap L[q]! c (iq_rate+list_len F[q]!+1) (startPlan iq_rate F[q]!))
    ∃ after, (iq_deliver q c).run s = .ok (result.pushed,result.lost) after ∧
      Seen iq_cap (historyStore iq_cap view q H[q]! L[q]! result.accepted) H
        (vectorUpdate L q (L[q]!+result.accepted.length)) (vectorUpdate F q result.fifo) after ∧
      Prefix (iq_cap-L[q]!) result ∧
      content iq_cap (historyStore iq_cap view q H[q]! L[q]! result.accepted)
        q H[q]! (L[q]!+result.accepted.length) =
        content iq_cap view q H[q]! L[q]! ++ result.offered.take (iq_cap-L[q]!) := by
  simpa only [deliver_production] using
    delivery_prefix_exact iq_cap iq_rate q c hq ports view H L F s seen

end Proofs.OrderedDelivery
