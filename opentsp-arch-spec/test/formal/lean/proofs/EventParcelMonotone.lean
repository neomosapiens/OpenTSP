import Proofs.EventTimedView
import Proofs.SlotAdmission
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventBoundaryProofs EventTimedViewProofs
open HoldAdmissionProofs SlotAdmissionProofs
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace EventParcelMonotoneProofs
abbrev St := SequentialState RegisterType trivialChoiceSource

theorem check_monotone (a b : hold_view) (h : ViewLE a b)
    (q : Nat) (i : instruction) (path : Option Nat) (strict : Bool) :
    checkValue q i path strict a=true → checkValue q i path strict b=true := by
  exact hold_decide_monotone a b h _ _ _ _

theorem ref_monotone (a b : hold_view) (h : ViewLE a b)
    (q : Nat) (r : cycle_recs) (k : Int) :
    refValue q r a k=true → refValue q r b k=true := by
  unfold refValue
  cases r.refs[k]! with
  | none => simp
  | some i => exact check_monotone a b h q i _ _

theorem fetch_monotone (a b : hold_view) (h : ViewLE a b) (q : Nat) (r : cycle_recs) :
    fetchValue q r a=true → fetchValue q r b=true := by
  unfold fetchValue
  cases r.ifetch with
  | none => simp
  | some i => exact check_monotone a b h q i _ _

theorem cfg_monotone (a b : hold_view) (h : ViewLE a b) (q p : Nat) (r : cycle_recs) :
    cfgValue q r a p=true → cfgValue q r b p=true := by
  unfold cfgValue
  cases r.cfg with
  | none => simp
  | some i => exact check_monotone a b h q i _ _

theorem slots_monotone (a b : hold_view) (h : ViewLE a b) (q p : Nat) (r : cycle_recs) :
    slotValue q r a p=true → slotValue q r b p=true := by
  have h0:=ref_monotone a b h q r 0
  have h1:=ref_monotone a b h q r 1
  have h2:=ref_monotone a b h q r 2
  have h3:=ref_monotone a b h q r 3
  have hf:=fetch_monotone a b h q r
  have hc:=cfg_monotone a b h q p r
  simp only [slotValue,refs_value_closed,Bool.or_eq_true] at *
  grind

def itemInstr (x : Result instruction diag) : instruction :=
  match x with | .Ok i=>i | .Err _=>.ILLEGAL32 0#32

def strictValue (q : Nat) (v : hold_view) (xs : List (Result instruction diag)) (held : Bool) : Bool :=
  match xs with
  | []=>held
  | x::xs=>strictValue q v xs (held || checkValue q (itemInstr x) none true v)

theorem strict_run (s : St) (q c : Nat) (v : hold_view)
    (hv : (hold_view_of q c).run s=.ok v s)
    (xs : List (Result instruction diag)) (held : Bool) :
    (strict_held q c xs held).run s=.ok (strictValue q v xs held) s := by
  induction xs generalizing held with
  | nil => rfl
  | cons x xs ih =>
    have hc:=actual_check_from_view s q c (itemInstr x) none true v hv
    have hc': hold_check q c (itemInstr x) none true s=.ok (checkValue q (itemInstr x) none true v) s:=hc
    cases x
    all_goals simp only [itemInstr] at hc' ⊢
    all_goals simpa only [strict_held,strictValue,itemInstr,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hc'] using ih (held || checkValue q _ none true v)

theorem strict_monotone (a b : hold_view) (h : ViewLE a b)
    (q : Nat) (xs : List (Result instruction diag)) (ha hb : Bool)
    (hh : ha=true → hb=true) :
    strictValue q a xs ha=true → strictValue q b xs hb=true := by
  induction xs generalizing ha hb with
  | nil => exact hh
  | cons x xs ih =>
    apply ih
    have hc:=check_monotone a b h q (itemInstr x) none true
    simp only [Bool.or_eq_true]
    grind

def parcelValue (q p : Nat) (r : cycle_recs) (items : List (Result instruction diag)) (v : hold_view) : Bool :=
  if r.n=0 then false
  else if !r.ok then strictValue q v items false
  else match r.single with
    | some i=>checkValue q i none false v
    | none=>slotValue q r v p

theorem parcel_value_monotone (a b : hold_view) (h : ViewLE a b)
    (q p : Nat) (r : cycle_recs) (items : List (Result instruction diag)) :
    parcelValue q p r items a=true → parcelValue q p r items b=true := by
  unfold parcelValue
  split
  · simp
  · split
    · exact strict_monotone a b h q items false false (by simp)
    · cases r.single with
      | none => exact slots_monotone a b h q p r
      | some i => exact check_monotone a b h q i none false

theorem actual_parcel_run (s : St) (q c p : Nat) (r : cycle_recs)
    (items : List (Result instruction diag)) (v : hold_view)
    (hv : (hold_view_of q c).run s=.ok v s) (hp : (cfg_path q).run s=.ok p s) :
    (parcel_held q c r items).run s=.ok (parcelValue q p r items v) s := by
  by_cases hn:r.n=0
  · simp [parcel_held,parcelValue,hn,EStateM.run,pure,EStateM.pure]
  · cases hok:r.ok
    · simpa [parcel_held,parcelValue,hn,hok,Out.Functions.not] using strict_run s q c v hv items false
    · cases hs:r.single with
      | some i =>
        simpa [parcel_held,parcelValue,hn,hok,hs,Out.Functions.not] using actual_check_from_view s q c i none false v hv
      | none =>
        rw [actual_slots_source q c r items hn hok hs]
        simpa [parcelValue,hn,hok,hs,Out.Functions.not] using slots_run s q c r v p hv hp

theorem actual_candidate_unheld (s : St) (values : Values) (present : Present s values)
    (q base t p : Nat) (r : cycle_recs) (items : List (Result instruction diag))
    (hb : base≤t) (hp : (cfg_path q).run s=.ok p s)
    (ht : (parcel_held q t r items).run s=.ok false s) :
    (parcel_held q (floor base (ends values q base) t) r items).run s=.ok false s := by
  let k:=floor base (ends values q base) t
  have htv:=actual_view s values present q t
  have hkv:=actual_view s values present q k
  have htr:=actual_parcel_run s q t p r items _ htv hp
  have hkr:=actual_parcel_run s q k p r items _ hkv hp
  have hfalse:parcelValue q p r items (atTime values q t)=false:=by
    rw [ht] at htr
    exact (EStateM.Result.ok.inj htr).1.symm
  have hm:=parcel_value_monotone _ _ (candidate_view_monotone values q base t hb) q p r items
  have hkfalse:parcelValue q p r items (atTime values q k)=false:=by
    cases he:parcelValue q p r items (atTime values q k)
    · rfl
    · have hbad:=hm he; rw [hfalse] at hbad; contradiction
  simpa only [hkfalse] using hkr

#print axioms parcel_value_monotone
#print axioms actual_parcel_run
#print axioms actual_candidate_unheld

#print axioms check_monotone
#print axioms slots_monotone
#print axioms strict_run
#print axioms strict_monotone
end EventParcelMonotoneProofs
