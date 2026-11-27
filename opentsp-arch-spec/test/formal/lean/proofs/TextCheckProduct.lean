import Proofs.ParcelBound
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
set_option autoImplicit false
namespace Proofs.TextCheckProduct
open Proofs.ParcelFrame
def Keep (H L : Vector Nat 145) (s : St) : Prop :=
  Proofs.StorageProduct.Good iq_cap s ∧
  s.regs.get? Register.iq_head=some H ∧ s.regs.get? Register.iq_len=some L
/-- The C interrupt callback changes host bookkeeping/output but no represented Sail
    state or IQ backing storage. This is a named small trusted interface, not a claim
    that the external host world is unchanged. Only architectural vectors0..15 are used. -/
structure IrqFrame : Prop where
  irq : ∀ c n (s : St), n<16 →
    match (host_irq_callback c n).run s with
    | .ok _ after => after.regs=s.regs ∧ after.mem=s.mem
    | .error _ after => after.regs=s.regs ∧ after.mem=s.mem

theorem keep_same_fields (H L : Vector Nat 145) (s t : St)
    (regs : t.regs=s.regs) (mem : t.mem=s.mem) : Keep H L t ↔ Keep H L s := by
  simp only [Keep, Proofs.StorageProduct.Good, Represents, Consistent, Present,
    Proofs.Initialized.Bounds, Proofs.StorageRep.Initialized, tags, readValue, regs, mem]

variable {H L : Vector Nat 145}
def Safe {α : Type} (action : SailM α) : Prop :=
  ∀ s : St, Keep H L s → match action.run s with
    | .ok _ after => Keep H L after
    | .error _ after => Keep H L after
theorem preserves_pure {α : Type} (a : α) : (Safe (H:=H) (L:=L)) (pure a) := by
  intro s hs
  exact hs

theorem preserves_ite {α : Type} (p : Prop) [Decidable p] (a b : SailM α)
    (ha : (Safe (H:=H) (L:=L)) a) (hb : (Safe (H:=H) (L:=L)) b) : (Safe (H:=H) (L:=L)) (if p then a else b) := by
  split <;> assumption

theorem preserves_bind {α β : Type} (a : SailM α) (f : α → SailM β)
    (ha : (Safe (H:=H) (L:=L)) a) (hf : ∀ x, (Safe (H:=H) (L:=L)) (f x)) : (Safe (H:=H) (L:=L)) (a >>= f) := by
  intro s hs
  have h := ha s hs
  cases he : a.run s with
  | ok x after =>
    have hp : (Keep H L) after := by simpa [he] using h
    have hf' := hf x after hp
    simpa [EStateM.run, bind, EStateM.bind, show a s = .ok x after from he] using hf'
  | error error after =>
    simpa [EStateM.run, bind, EStateM.bind, show a s = .error error after from he] using h

theorem preserves_read (r : Register) : (Safe (H:=H) (L:=L)) (Out.readReg r) := by
  intro s hs
  simp [Out.readReg, PreSail.readReg, EStateM.run, bind, EStateM.bind, get, getThe,
    MonadStateOf.get, EStateM.get]
  cases s.regs.get? r <;> exact hs

theorem preserves_other_write (r : Register) (v : RegisterType r)
    (hh : r≠Register.iq_head) (hl : r≠Register.iq_len) : Safe (H:=H) (L:=L) (Out.writeReg r v) := by
  intro s hs
  obtain ⟨good,hH,hL⟩ := hs
  have hg := Proofs.StorageProduct.preserves_other_write (cap:=iq_cap) r v hh hl s good
  simp [Out.writeReg,PreSail.writeReg,EStateM.run,modify,modifyGet,
    MonadStateOf.modifyGet,EStateM.modifyGet] at hg ⊢
  exact ⟨hg,by simpa [Std.ExtDHashMap.get?_insert,hh,Ne.symm hh] using hH,
    by simpa [Std.ExtDHashMap.get?_insert,hl,Ne.symm hl] using hL⟩

theorem preserves_forIn {β : Type} (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β))
    (hf : ∀ i hi value, (Safe (H:=H) (L:=L)) (f i hi value)) : (Safe (H:=H) (L:=L)) (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => (Safe (H:=H) (L:=L)) (IntRange.forIn'.loop range f value i hs))
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


theorem frame_safe {α : Type} (a : SailM α)
    (hf : ∀ s, Keep H L s → FrameAt s a) : Safe (H:=H) (L:=L) a := by
  intro s hs
  have h := hf s hs
  cases he : a.run s with
  | ok x after =>
    have eq : after=s := by simpa [FrameAt,he] using h
    simpa [he,eq] using hs
  | error error after =>
    have eq : after=s := by simpa [FrameAt,he] using h
    simpa [he,eq] using hs

theorem unit_write_safe (q slot : Nat) (u : BitVec 32) (hq : q<145) (hslot : slot<iq_cap)
    (ports : Proofs.Initialized.WritePorts iq_cap) :
    Safe (H:=H) (L:=L) (iq_unit_write q slot u) := by
  intro s hs
  obtain ⟨⟨c,rep,bounds⟩,hH,hL⟩ := hs
  have he := ports.unit q slot u s hq hslot
  let t := writeCell s (145* iq_cap) 0 (q* iq_cap+slot) u.toNat
  have ht : Proofs.Initialized.Bounds iq_cap t := by
    obtain ⟨H0,L0,hH0,hL0,hb⟩ := bounds
    refine ⟨H0,L0,hH0,hL0,?_⟩
    intro r hr
    refine ⟨(hb r hr).1,(hb r hr).2.1,?_⟩
    intro i hi
    apply initialized_preserved iq_cap s q slot u.toNat r _ hq hslot hr
    · rw [Proofs.Capacity.ring_spec]
      have hh := (hb r hr).1
      have hl := (hb r hr).2.1
      split <;> omega
    · exact (hb r hr).2.2 i hi
  have keep : Keep H L t := ⟨⟨cUnitWrite c q slot u,unit_represents iq_cap c s rep q slot u hq hslot,ht⟩,hH,hL⟩
  simpa [he] using keep

theorem view_safe (q pos : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (hi : pos<L[q]!) (ports : ReadPorts iq_cap) :
    Safe (H:=H) (L:=L) (iq_view q pos) := by
  apply frame_safe
  intro s hs
  exact view_frame s q pos ports H hs.2.1 L hs.2.2 hs.1 hq hh hl hi

theorem word_safe (q pos : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hi : pos≤iq_cap) (ports : ReadPorts iq_cap) :
    Safe (H:=H) (L:=L) (iq_unit_word q pos) := by
  apply frame_safe
  intro s hs
  exact unit_word_frame s q pos ports H hs.2.1 hq hh hi

theorem irq_safe (c n : Nat) (hn : n<16) (irq : IrqFrame) :
    Safe (H:=H) (L:=L) (host_irq_callback c n) := by
  intro s hs
  have frame := irq.irq c n s hn
  cases run : (host_irq_callback c n).run s with
  | ok value after =>
    simp only [run] at frame ⊢
    exact (keep_same_fields H L s after frame.1 frame.2).mpr hs
  | error error after =>
    simp only [run] at frame ⊢
    exact (keep_same_fields H L s after frame.1 frame.2).mpr hs

theorem host_msix_safe (c : Nat) (irq : IrqFrame) :
    Safe (H:=H) (L:=L) (host_msix c 0) := by
  unfold host_msix host_deliver
  repeat' first
    | exact irq_safe c 0 (by decide) irq
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_bind
    | split
    | intro x

theorem ras_safe (c : Nat) (q : Int) (kind : Nat) (loc : BitVec 43) (sl pos : Nat) (irq : IrqFrame) :
    Safe (H:=H) (L:=L) (ras_record c q kind loc sl pos) := by
  unfold ras_record
  repeat' first
    | exact host_msix_safe c irq
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_bind
    | split
    | intro x

theorem ras_ecc_safe (c q kind : Nat) (unc isw : Bool) (id addr sl pos : Nat) (irq : IrqFrame) :
    Safe (H:=H) (L:=L) (ras_ecc_event c q kind unc isw id addr sl pos) := by
  unfold ras_ecc_event
  exact ras_safe c q _ _ sl pos irq

theorem head_bind_safe {α : Type} (f : Vector Nat 145 → SailM α)
    (hf : Safe (H:=H) (L:=L) (f H)) :
    Safe (H:=H) (L:=L) ((Out.readReg Register.iq_head) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_head s = .ok H s := by
    simp [Out.readReg,PreSail.readReg,hs.2.1,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,he] using hf s hs

theorem length_bind_safe {α : Type} (f : Vector Nat 145 → SailM α)
    (hf : Safe (H:=H) (L:=L) (f L)) :
    Safe (H:=H) (L:=L) ((Out.readReg Register.iq_len) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_len s = .ok L s := by
    simp [Out.readReg,PreSail.readReg,hs.2.2,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,he] using hf s hs

theorem peek_safe (q pos : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (hi : pos<L[q]!) (ports : ReadPorts iq_cap) :
    Safe (H:=H) (L:=L) (iq_peek q pos) := by
  apply frame_safe
  intro s hs
  exact peek_frame s q pos ports H hs.2.1 L hs.2.2 hs.1 hq hh hl hi

theorem tag_read_safe (q slot : Nat) (hq : q<145) (hslot : slot<iq_cap) (ports : ReadPorts iq_cap) :
    Safe (H:=H) (L:=L) (iq_tag_read q slot) := by
  apply frame_safe
  intro s _
  obtain ⟨value,he⟩ := ports.tag q slot s hq hslot
  simp [FrameAt,he]

theorem overlay_read_safe (q w : Nat) :
    Safe (H:=H) (L:=L) (iq_word_ov q w) := by
  apply frame_safe
  intro s _
  exact word_overlay_frame s q w

theorem overlay_write_safe (q w : Nat) (e : BitVec 10) :
    Safe (H:=H) (L:=L) (iq_word_ov_set q w e) := by
  unfold iq_word_ov_set
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x


theorem length_map_bind_safe {α β : Type} (g : Vector Nat 145 → β) (f : β → SailM α)
    (hf : Safe (H:=H) (L:=L) (f (g L))) :
    Safe (H:=H) (L:=L) ((do let v ← Out.readReg Register.iq_len; pure (g v)) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_len s = .ok L s := by
    simp [Out.readReg,PreSail.readReg,hs.2.2,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,he] using hf s hs

theorem head_map_bind_safe {α β : Type} (g : Vector Nat 145 → β) (f : β → SailM α)
    (hf : Safe (H:=H) (L:=L) (f (g H))) :
    Safe (H:=H) (L:=L) ((do let v ← Out.readReg Register.iq_head; pure (g v)) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_head s = .ok H s := by
    simp [Out.readReg,PreSail.readReg,hs.2.1,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,he] using hf s hs

theorem ring_int_domain (h : Nat) (i : Int) (hh : h<iq_cap)
    (hi0 : 0≤i) (hi : i≤iq_cap) : ring ((h:Int) + i).toNat<iq_cap := by
  have he : ((h:Int) + i).toNat=h + i.toNat := by omega
  rw [he]
  exact ring_domain h i.toNat hh (by omega)

theorem word_check_safe (q c : Nat) (w : Int) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    Safe (H:=H) (L:=L) (iq_word_check q c w) := by
  unfold iq_word_check
  simp only [bind_assoc,pure_bind,bind_pure]
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact ras_ecc_safe _ _ _ _ _ _ _ _ _ irq
    | exact overlay_read_safe _ _
    | exact overlay_write_safe _ _ _
    | with_reducible apply head_map_bind_safe
    | with_reducible apply length_map_bind_safe
    | with_reducible apply head_bind_safe
    | with_reducible apply length_bind_safe
    | exact peek_safe q _ hq hh hl (by simp only [Membership.mem] at *; simp at *; omega) reads
    | exact unit_write_safe q _ _ hq (by
        simp only [Membership.mem] at *
        simp at *
        apply ring_int_domain H[q]! _ hh <;> omega) writes
    | exact tag_read_safe q _ hq (by
        simp only [Membership.mem] at *
        simp at *
        apply ring_int_domain H[q]! _ hh <;> omega) reads
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x


/-- Full actual text checking/repair preserves the represented initialized FIFO
    shape and the exact captured head/length, even on partial failure paths. -/
theorem text_check_safe (q c units : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (hu : units≤L[q]!) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    Safe (H:=H) (L:=L) (iq_text_check q c units) := by
  unfold iq_text_check
  simp only [bind_assoc,pure_bind,bind_pure]
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact word_check_safe q c _ hq hh hl reads writes irq
    | exact view_safe q _ hq hh hl (by simp_all <;> omega) reads
    | exact word_safe q _ hq hh (by simp_all <;> omega) reads
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x

/-- Actual pop after checking remains within the captured queue, preserves its
    external representation/initialized occupancy and decreases length exactly. -/
theorem pop_product (q units : Nat) (hq : q<145) (hu : units≤L[q]!)
    (s : St) (keep : Keep H L s) :
    ∃ after, (iq_pop q units).run s = .ok () after ∧
      Proofs.StorageProduct.Good iq_cap after ∧
      after.regs.get? Register.iq_len=some (vectorUpdate L q (L[q]!-units)) := by
  obtain ⟨⟨c,rep,bounds⟩,hH,hL⟩ := keep
  let after : St := {s with regs := (s.regs.insert Register.iq_head (vectorUpdate H q (Proofs.Capacity.ringC iq_cap (H[q]!+units)))).insert Register.iq_len (vectorUpdate L q (L[q]!-units))}
  have he : (Proofs.Capacity.popC iq_cap q units).run s = .ok () after :=
    Proofs.Capacity.iq_pop_run q units s H L hH hL
  obtain ⟨t,ht,hb⟩ := Proofs.Initialized.pop_preserves iq_cap q units hq s bounds H L hH hL hu
  rw [he] at ht
  cases ht
  refine ⟨after,?_,⟨c,rep,hb⟩,?_⟩
  · simpa only [Proofs.Capacity.pop_production] using he
  · simp [after]


/-- The exact check/pop subsequence used by both production dispatch paths. -/
def checkedPop (q c : Nat) (p : parcel) : SailM Unit := do
  iq_text_check q c p.units
  iq_pop q p.units

def checkedConsume (q c : Nat) : SailM parcel := do
  let p ← parcel_scan q
  checkedPop q c p
  pure p

/-- Parser output is safe for the actual repair/pop subsequence. Before pop, errors
    preserve the old head/length and represented initialized store; a successful pop
    decreases exactly by the parser's bounded count. No repair value-identity claim
    is inferred from this shape theorem. -/
theorem checked_consume_safe (q c : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame)
    (s : St) (keep : Keep H L s) :
    match (checkedConsume q c).run s with
    | .ok p after => p.units≤L[q]! ∧ Proofs.StorageProduct.Good iq_cap after ∧
        after.regs.get? Register.iq_len=some (vectorUpdate L q (L[q]!-p.units))
    | .error _ after => Keep H L after := by
  have hs := Proofs.ParcelBound.scan_bound s q reads H L keep.2.1 keep.2.2 keep.1 hq hh hl
  cases hp : (parcel_scan q).run s with
  | error error after =>
    have eq : after=s := by simpa [Proofs.ParcelBound.SpecAt,hp] using hs
    have hp' : parcel_scan q s = .error error after := hp
    simpa [checkedConsume,EStateM.run,bind,EStateM.bind,hp',eq] using keep
  | ok p after =>
    have hps : after=s ∧ p.units≤L[q]! := by simpa [Proofs.ParcelBound.SpecAt,hp] using hs
    obtain ⟨eq,bound⟩ := hps
    subst after
    have hp' : parcel_scan q s = .ok p s := hp
    have hc := text_check_safe q c p.units hq hh hl bound reads writes irq s keep
    cases check : (iq_text_check q c p.units).run s with
    | error error after =>
      have ka : Keep H L after := by simpa [check] using hc
      have check' : iq_text_check q c p.units s = .error error after := check
      simpa [checkedConsume,checkedPop,EStateM.run,bind,EStateM.bind,hp',check'] using ka
    | ok u after =>
      cases u
      have ka : Keep H L after := by simpa [check] using hc
      have check' : iq_text_check q c p.units s = .ok () after := check
      obtain ⟨t,pop,good,len⟩ := pop_product q p.units hq bound after ka
      have pop' : iq_pop q p.units after = .ok () t := pop
      simpa [checkedConsume,checkedPop,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
        hp',check',pop'] using And.intro bound (And.intro good len)


open vxm_unop
open vxm_lane_op
open vxm_conv
open vxm_binop
open vsrc
open unit_kind
open timing_class
open sxm_mapop
open streamref
open srf_lookup
open sram_addr
open shift_dir
open scrub_action
open row
open reduce_op
open option
open mxm_wtype
open mxm_part
open mxm_acctype
open mem_dir
open instruction
open fclass
open exception
open eltype
open ecc_class
open diag
open cmp_mode
open c2c_ptype
open c2c_op
open bank_use
open agt_state
open agt_field
open acc_mode
open Register
open ExecutionResult


def controlUsingCheckedPop (c : Nat) : SailM Unit := do
  if (((GetElem?.getElem! (← Out.readReg iq_len) 144) >b 0) : Bool)
  then
    (do
      let p ← do (parcel_scan 144)
      if ((p.complete && ((← (parcel_earliest 144 p)) ≤b c)) : Bool)
      then
        (do
          (observe_site "dispatch" c 144 (GetElem?.getElem! (← Out.readReg iq_head) 144) p.units
            (← (parcel_earliest 144 p)))
          (checkedPop 144 c p)
          Out.writeReg iq_last (vectorUpdate (← Out.readReg iq_last) 144 c)
          Out.writeReg iq_active (vectorUpdate (← Out.readReg iq_active) 144 c)
          match p.items with
          | (x :: rest) =>
            (do
              match rest with
              | [] =>
                (do
                  match x with
                  | .Err d => (report 144 c d)
                  | .Ok i =>
                    (do
                      match i with
                      | .SYNC () => (barrier_sync 144 c)
                      | .NOTIFY () => (barrier_notify c)
                      | .CONFIG _ => (pure ())
                      | _ => (report 144 c QUE_UNIT)))
              | _ =>
                (do
                  let loop_k_lower := 1
                  let loop_k_upper := p.n
                  let mut loop_vars := ()
                  for k in [loop_k_lower:loop_k_upper:1]i do
                    let () := loop_vars
                    loop_vars ← do (report 144 c QUE_BUSY)
                  (pure loop_vars)))
          | [] => (pure ()))
      else (pure ()))
  else (pure ())


theorem control_checked_pop_bridge (c : Nat) : Out.Functions.control c = controlUsingCheckedPop c := by
  simp only [Out.Functions.control,controlUsingCheckedPop,checkedPop,bind_assoc]
  rfl

def tickQueueUsingCheckedPop (q : Nat) (c : Nat) : SailM Unit := do
  let p : parcel :=
    { complete := false
      units := 0
      nops := 0
      n := 0
      items := [] }
  let due : Bool := false
  let (due, p) ← (( do
    if (((GetElem?.getElem! (← Out.readReg iq_len) q) >b 0) : Bool)
    then
      (do
        let p ← (parcel_scan q)
        let due ← (pure (p.complete && (c ≥b (← (parcel_earliest q p)))))
        (pure (due, p)))
    else (pure (due, p)) ) : SailM (Bool × parcel) )
  let recs : cycle_recs :=
    if (due : Bool)
    then (classify_items q p.items)
    else no_recs
  let parked ← do (queue_parked q c)
  match (GetElem?.getElem! (← Out.readReg queue_poll) q) with
  | .some p =>
    (do
      if ((← do if ((poll_epoch p.since c c2c_hac_period parked) : Bool)
                then deskewrecv_ready c p.link else pure false) : Bool)
      then
        (do
          Out.writeReg queue_poll (vectorUpdate (← Out.readReg queue_poll) q none)
          let rcv := (C2C_RECEIVE (p.chip, p.link, p.s))
          Out.writeReg queue_busy_until (vectorUpdate (← Out.readReg queue_busy_until) q
            (c +i (← (occupancy rcv c))))
          (run_element q rcv c 0 0 0))
      else (pure ()))
  | none => (pure ())
  match (GetElem?.getElem! (← Out.readReg queue_repeat) q) with
  | .some r =>
    (do
      if (((r.next ≤b c) && (Out.Functions.not parked)) : Bool)
      then
        (do
          (observe_site "repeat" c q 0 0 (GetElem?.getElem! (← Out.readReg queue_last_at) q))
          (repeat_issue q r.instr c)
          Out.writeReg queue_last_at (vectorUpdate (← Out.readReg queue_last_at) q c)
          Out.writeReg queue_busy_until (vectorUpdate (← Out.readReg queue_busy_until) q (c +i 1))
          Out.writeReg queue_repeat (vectorUpdate (← Out.readReg queue_repeat) q (repeat_after r c)))
      else (pure ()))
  | none => (pure ())
  let go ← (( do (pure (due && (Out.Functions.not (← (parcel_held q c recs p.items))))) ) : SailM Bool )
  let ended : Bool := false
  let ended ← (( do
    let loop_path_lower := 0
    let loop_path_upper := 1
    let mut loop_vars := ended
    for path in [loop_path_lower:loop_path_upper:1]i do
      let ended := loop_vars
      loop_vars ← do
        if ((config_ends (← (agt_get q path)).state go (config_for recs)) : Bool)
        then
          (do
            (agt_finish_config q path)
            (pure true))
        else (pure ended)
    (pure loop_vars) ) : SailM Bool )
  let go ← (( do
    if ((ended && (due && (Out.Functions.not go))) : Bool)
    then
      (do
        (pure (Out.Functions.not (← (parcel_held q c recs p.items)))))
    else (pure go) ) : SailM Bool )
  if ((due && (Out.Functions.not go)) : Bool)
  then
    (observe_site
      (wait_reason (← (host_parked q c)) parked (← (polling q))
        (bne (GetElem?.getElem! (← Out.readReg queue_repeat) q) none)) c q
      (GetElem?.getElem! (← Out.readReg iq_head) q) p.units (← (parcel_earliest q p)))
  else (pure ())
  let r2 : cycle_recs := no_recs
  let r2 ← (( do
    if (go : Bool)
    then
      (do
        let r2 : cycle_recs := recs
        (observe_site "dispatch" c q (GetElem?.getElem! (← Out.readReg iq_head) q) p.units
          (← (parcel_earliest q p)))
        (checkedPop q c p)
        Out.writeReg iq_last (vectorUpdate (← Out.readReg iq_last) q c)
        Out.writeReg iq_active (vectorUpdate (← Out.readReg iq_active) q c)
        if ((GetElem?.getElem! (← Out.readReg iq_dry) q) : Bool)
        then
          (do
            Out.writeReg iq_dry (vectorUpdate (← Out.readReg iq_dry) q false)
            (pure (lint q c LINT_IQ_UNDERFLOW)))
        else (pure ())
        (pure r2))
    else (pure r2) ) : SailM cycle_recs )
  (dispatch q c r2 parked)


theorem tick_queue_checked_pop_bridge (q c : Nat) : Out.Functions.tick_queue q c = tickQueueUsingCheckedPop q c := by
  simp only [Out.Functions.tick_queue,tickQueueUsingCheckedPop,checkedPop,bind_assoc]
  rfl

end Proofs.TextCheckProduct
