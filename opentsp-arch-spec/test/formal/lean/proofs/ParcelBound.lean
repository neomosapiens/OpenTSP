import Proofs.ParcelFrame
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
open Out.Register
namespace Proofs.ParcelBound
abbrev ScanVars := Bool × Bool × List (Result instruction diag) × Nat × Nat × Nat
def scanStep (q len : Nat) (v : ScanVars) : SailM ScanVars := do
  let (complete,done,items,n,nops,pos) := v
  let (complete, done, items, n, nops, pos) ← (( do
    if ((Out.Functions.not done) : Bool)
    then
      (do
        let (complete, done, items, n, nops, pos) ← (( do
          if ((pos ≥b len) : Bool)
          then
            (let done : Bool := true
            (pure (complete, done, items, n, nops, pos)))
          else
            (do
              let u0 ← do (iq_view q pos)
              let w := (text_width u0)
              let (complete, done, items, n, nops, pos) ← (( do
                if (((w == 2) && (((pos +i 1) ≥b len) : Bool)) : Bool)
                then
                  (let done : Bool := true
                  (pure (complete, done, items, n, nops, pos)))
                else
                  (do
                    let r ← do (iq_decode_at q pos w)
                    let bundle := ((BitVec.access u0 31) == 1#1)
                    let nop : Nat :=
                      match r with
                      | .Ok (.NOP k) =>
                        (if (((Out.Functions.not bundle) && ((n == 0) : Bool)) : Bool)
                        then (BitVec.toNatInt k)
                        else 0)
                      | _ => 0
                    let pos : Nat := (pos +i w)
                    let (complete, done, items, n, nops) : (Bool × Bool × (List (Result instruction diag)) × Nat × Nat) :=
                      if ((nop >b 0) : Bool)
                      then
                        (let nops : Nat := (nops +i nop)
                        (complete, done, items, n, nops))
                      else
                        (let items : (List (Result instruction diag)) := (r :: items)
                        let n : Nat := (n +i 1)
                        let (complete, done) : (Bool × Bool) :=
                          if ((Out.Functions.not bundle) : Bool)
                          then
                            (let complete : Bool := true
                            let done : Bool := true
                            (complete, done))
                          else (complete, done)
                        (complete, done, items, n, nops))
                    (pure (complete, done, items, n, nops, pos))) ) : SailM
                (Bool × Bool × (List (Result instruction diag)) × Nat × Nat × Nat) )
              (pure (complete, done, items, n, nops, pos))) ) : SailM
          (Bool × Bool × (List (Result instruction diag)) × Nat × Nat × Nat) )
        (pure (complete, done, items, n, nops, pos)))
    else (pure (complete, done, items, n, nops, pos)) ) : SailM
    (Bool × Bool × (List (Result instruction diag)) × Nat × Nat × Nat) )
  (pure (complete, done, items, n, nops, pos))

def scanLoop (q len : Nat) (v : ScanVars) : SailM ScanVars := do
  let (complete,done,items,n,nops,pos) := v
  let loop__step_lower := 0
  let loop__step_upper := len
  let mut loop_vars := (complete, done, items, n, nops, pos)
  for _step in [loop__step_lower:loop__step_upper:1]i do
    let (complete, done, items, n, nops, pos) := loop_vars
    loop_vars ← scanStep q len (complete,done,items,n,nops,pos)
  (pure loop_vars)

def scanWhole (q : Nat) : SailM parcel := do
  let len ← do (pure (GetElem?.getElem! (← Out.readReg iq_len) q))
  let pos : Nat := 0
  let nops : Nat := 0
  let n : Nat := 0
  let items : (List (Result instruction diag)) := []
  let done : Bool := false
  let complete : Bool := false
  let (complete, done, items, n, nops, pos) ← scanLoop q len (complete,done,items,n,nops,pos)
  if (complete : Bool)
  then
    (pure { complete := true
            units := pos
            nops := nops
            n := n
            items := (rev_items items []) })
  else (pure no_parcel)


theorem scan_production (q : Nat) : parcel_scan q = scanWhole q := by
  simp only [parcel_scan,scanWhole,scanLoop,scanStep,bind_assoc,pure_bind,bind_pure,Prod.eta]
  rfl

open Proofs.ParcelFrame
def SpecAt {α : Type} (s : St) (P : α → Prop) (action : SailM α) : Prop :=
  match action.run s with
  | .ok value after => after=s ∧ P value
  | .error _ after => after=s

theorem pure_spec {α : Type} (s : St) (P : α → Prop) (x : α) (h : P x) :
    SpecAt s P (pure x) := ⟨rfl,h⟩

theorem frame_true {α : Type} (s : St) (a : SailM α) (h : FrameAt s a) :
    SpecAt s (fun _ => True) a := by
  cases he : a.run s <;> simpa [FrameAt,SpecAt,he] using h

theorem bind_spec {α β : Type} (s : St) (P : α → Prop) (Q : β → Prop)
    (a : SailM α) (f : α → SailM β)
    (ha : SpecAt s P a) (hf : ∀ x, P x → SpecAt s Q (f x)) :
    SpecAt s Q (a >>= f) := by
  cases he : a.run s with
  | ok x after =>
    have hp : after=s ∧ P x := by simpa [SpecAt,he] using ha
    obtain ⟨hs,hx⟩ := hp
    subst after
    simpa [SpecAt,EStateM.run,bind,EStateM.bind,show a s = .ok x s from he] using hf x hx
  | error error after =>
    simpa [SpecAt,EStateM.run,bind,EStateM.bind,show a s = .error error after from he] using ha

theorem bind_frame_spec {α β : Type} (s : St) (Q : β → Prop)
    (a : SailM α) (f : α → SailM β)
    (ha : FrameAt s a) (hf : ∀ x, SpecAt s Q (f x)) :
    SpecAt s Q (a >>= f) :=
  bind_spec s (fun _ => True) Q a f (frame_true s a ha) (fun x _ => hf x)

theorem forIn_spec {β : Type} (s : St) (P : β → Prop) (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β)) (hb : P b)
    (hf : ∀ i hi value, P value →
      SpecAt s (fun step => match step with | .yield v => P v | .done v => P v) (f i hi value)) :
    SpecAt s P (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => P value → SpecAt s P (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih hp
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply bind_spec _ _ _ _ _ (hf i hi value hp)
    intro step hstep
    cases step with
    | done value => exact pure_spec s P value hstep
    | yield value => exact ih value hstep
  · intro value i hs hi hp
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact pure_spec s P value hp
  · exact hb

def Valid (len : Nat) (v : ScanVars) : Prop := v.2.2.2.2.2≤len

theorem step_valid (s : St) (q : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap)
    (v : ScanVars) (hv : Valid L[q]! v) :
    SpecAt s (Valid L[q]!) (scanStep q L[q]! v) := by
  rcases v with ⟨complete,done,items,n,nops,pos⟩
  change pos≤L[q]! at hv
  unfold scanStep
  dsimp only
  try simp only [pure_bind,bind_pure,Prod.eta]
  split
  · split
    · apply pure_spec
      exact hv
    · apply bind_frame_spec
      · exact view_frame s q pos ports H hH L hL good hq hh hl (by simp_all <;> omega)
      · intro u
        have hw := width_one_or_two u
        try dsimp only
        split
        · apply pure_spec
          exact hv
        · apply bind_frame_spec
          · exact decode_at_frame s q pos (text_width u) ports H L hH hL good hq hh hl
              (by simp_all <;> omega) hw (by simp_all <;> omega)
          · intro decoded
            try dsimp only
            apply pure_spec
            simp only [Valid]
            simp_all <;> omega
  · apply pure_spec
    exact hv



theorem loop_valid (s : St) (q : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap)
    (v : ScanVars) (hv : Valid L[q]! v) :
    SpecAt s (Valid L[q]!) (scanLoop q L[q]! v) := by
  unfold scanLoop
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  simp only [bind_pure,Prod.eta]
  apply forIn_spec s (Valid L[q]!) _ _ _ hv
  intro i hi value hvalue
  dsimp only
  apply bind_spec _ _ _ _ _ (step_valid s q ports H L hH hL good hq hh hl value hvalue)
  intro result hresult
  exact pure_spec s _ (ForInStep.yield result) hresult



/-- Complete actual parser: its successful consumption is bounded by the captured
    length, while all successful and failed returns preserve the whole state. -/
theorem scan_bound (s : St) (q : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) :
    SpecAt s (fun p => p.units≤L[q]!) (parcel_scan q) := by
  have hloop := loop_valid s q ports H L hH hL good hq hh hl
    (false,false,[],0,0,0) (by simp [Valid])
  rw [scan_production]
  unfold scanWhole SpecAt
  simp only [Out.readReg,PreSail.readReg,hL,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  cases he : (scanLoop q L[q]! (false,false,[],0,0,0)).run s with
  | ok value after =>
    have hp : after=s ∧ Valid L[q]! value := by simpa [SpecAt,he] using hloop
    obtain ⟨hs,hv⟩ := hp
    subst after
    have he' : scanLoop q L[q]! (false,false,[],0,0,0) s = .ok value s := he
    simp only [he']
    rcases value with ⟨complete,done,items,n,nops,pos⟩
    cases complete <;> simp_all [Valid,no_parcel,EStateM.pure]
  | error error after =>
    have hs : after=s := by simpa [SpecAt,he] using hloop
    have he' : scanLoop q L[q]! (false,false,[],0,0,0) s = .error error after := he
    simpa only [he'] using hs

theorem spec_frame {α : Type} (s : St) (P : α → Prop) (a : SailM α)
    (h : SpecAt s P a) : FrameAt s a := by
  cases he : a.run s with
  | ok x after =>
    have hs := (show after=s ∧ P x by simpa [SpecAt,he] using h).1
    simpa [FrameAt,he] using hs
  | error error after => simpa [FrameAt,SpecAt,he] using h

theorem scan_frame (s : St) (q : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) :
    FrameAt s (parcel_scan q) :=
  spec_frame s _ _ (scan_bound s q ports H L hH hL good hq hh hl)

end Proofs.ParcelBound
