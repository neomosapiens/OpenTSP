import Proofs.EventParcelMonotone
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventBoundaryProofs EventTimedViewProofs EventParcelMonotoneProofs
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace EventSearchProofs
abbrev St := SequentialState RegisterType trivialChoiceSource

def search (base : Nat) (held : Nat→Bool) (cands : List Nat) (best : Int) : Int :=
  match cands with
  | []=>best
  | t::rest=>search base held rest (if base≤t ∧ held t=false then earliest_int best t else best)

def Urgent (target : Nat) (best : Int) : Prop := 0 ≤ best ∧ best ≤ (target:Int)

theorem min_preserves (target : Nat) (best : Int) (t : Nat) (h : Urgent target best) :
    Urgent target (earliest_int best t) := by
  unfold Urgent at *
  simp only [earliest_int,decide_eq_true_eq]
  split <;> (try split) <;> (try split) <;> omega

theorem min_captures (best : Int) (t : Nat) : Urgent t (earliest_int best t) := by
  simp only [earliest_int,decide_eq_true_eq]
  unfold Urgent
  split <;> (try split) <;> (try split) <;> omega

theorem search_preserves (base : Nat) (held : Nat→Bool) (cands : List Nat)
    (target : Nat) (best : Int) (hb : Urgent target best) :
    Urgent target (search base held cands best) := by
  induction cands generalizing best with
  | nil=>exact hb
  | cons t rest ih=>
    simp only [search]
    split
    · exact ih _ (min_preserves target best t hb)
    · exact ih _ hb

theorem search_captures (base : Nat) (held : Nat→Bool) (cands : List Nat)
    (k : Nat) (hk : k∈cands) (hb : base≤k) (hf : held k=false) (best : Int) :
    Urgent k (search base held cands best) := by
  induction cands generalizing best with
  | nil=>simp at hk
  | cons t rest ih=>
    rcases List.mem_cons.mp hk with he|hr
    · subst t
      simp only [search,show base≤k ∧ held k=false from ⟨hb,hf⟩,if_true]
      exact search_preserves base held rest k _ (min_captures best k)
    · simp only [search]
      split <;> exact ih hr _

def heldAt (values : Values) (q p : Nat) (r : cycle_recs)
    (items : List (Result instruction diag)) (t : Nat) : Bool :=
  parcelValue q p r items (atTime values q t)

theorem actual_search (s : St) (values : Values) (present : Present s values)
    (q base p : Nat) (r : cycle_recs) (items : List (Result instruction diag))
    (hp : (cfg_path q).run s=.ok p s) (cands : List Nat) (best : Int) :
    (earliest_free q base r items cands best).run s=
      .ok (search base (heldAt values q p r items) cands best) s := by
  induction cands generalizing best with
  | nil=>rfl
  | cons t rest ih=>
    have hq:=actual_parcel_run s q t p r items _ (actual_view s values present q t) hp
    have hq':parcel_held q t r items s=.ok (heldAt values q p r items t) s:=hq
    cases hh:heldAt values q p r items t
    all_goals by_cases hb:base≤t
    all_goals simpa [earliest_free,search,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hq',hh,hb,Out.Functions.not] using ih (if base≤t ∧ heldAt values q p r items t=false then earliest_int best t else best)

def candidates (v : Values) (q base : Nat) : List Nat :=
  let cs:=[base,(v .queue_busy_until)[q]!, (v .path_busy_until)[2*q]!, (v .path_busy_until)[2*q+1]!,
    (v .path_tail_until)[2*q]!, (v .path_tail_until)[2*q+1]!]
  let cs:=match (v .release_at)[q]! with | some x=>x::cs | none=>cs
  let cs:=match (v .release_at)[144]! with | some x=>x::cs | none=>cs
  match (v .hw_release)[q]! with | some x=>x::cs | none=>cs

theorem candidates_members (v : Values) (q base x : Nat) :
    x∈candidates v q base ↔ x∈ends v q base := by
  cases hq:(v .release_at)[q]!
  all_goals cases hc:(v .release_at)[144]!
  all_goals cases hh:(v .hw_release)[q]!
  all_goals simp_all [candidates,ends]
  all_goals grind

@[simp] theorem get_int_nat {α : Type} [Inhabited α] {n : Nat} (v : Vector α n) (i : Nat) :
    GetElem?.getElem! v (i:Int)=v[i]! := by rfl

@[simp] theorem get_two_int {α : Type} [Inhabited α] {n : Nat} (v : Vector α n) (q : Nat) :
    GetElem?.getElem! v ((2:Int)*(q:Int))=v[2*q]! := by
  have he : (2:Int)*(q:Int)=((2*q:Nat):Int) := by omega
  rw [he]
  exact get_int_nat v (2*q)
@[simp] theorem get_two_add_int {α : Type} [Inhabited α] {n : Nat} (v : Vector α n) (q : Nat) :
    GetElem?.getElem! v ((2:Int)*(q:Int)+1)=v[2*q+1]! := by
  have he : (2:Int)*(q:Int)+1=((2*q+1:Nat):Int) := by omega
  rw [he]
  exact get_int_nat v (2*q+1)

theorem actual_dispatchable (s : St) (values : Values) (present : Present s values)
    (q base p : Nat) (r : cycle_recs) (items : List (Result instruction diag))
    (hp : (cfg_path q).run s=.ok p s) :
    (parcel_dispatchable q base r items).run s=
      .ok (search base (heldAt values q p r items) (candidates values q base) (-1)) s := by
  change (∀reg,s.regs.get? reg=some (values reg)) at present
  have hs:=actual_search s values present q base p r items hp (candidates values q base) (-1)
  cases hq:(values .release_at)[q]!
  all_goals cases hc:(values .release_at)[144]!
  all_goals cases hh:(values .hw_release)[q]!
  all_goals try simp only [getElem!_pos] at hc
  all_goals simp_all [parcel_dispatchable,candidates,Out.readReg,PreSail.readReg,EStateM.run,
    bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,←Int.natCast_mul,←Int.natCast_add]

theorem actual_dispatchable_captures (s : St) (values : Values) (present : Present s values)
    (q base t p : Nat) (r : cycle_recs) (items : List (Result instruction diag))
    (hb : base≤t) (hp : (cfg_path q).run s=.ok p s)
    (ht : (parcel_held q t r items).run s=.ok false s) :
    ∃next,(parcel_dispatchable q base r items).run s=.ok next s ∧ 0≤next ∧ next≤t := by
  let k:=floor base (ends values q base) t
  have hkbase:base≤k:=floor_ge base t _
  have hkt:k≤t:=floor_le base t _ hb
  have hkm:k∈ends values q base:=by
    rcases floor_member base t (ends values q base) with he|hm
    · simpa only [k,he] using (show base∈ends values q base by simp [ends])
    · exact hm
  have hkfalse:=actual_candidate_unheld s values present q base t p r items hb hp ht
  have hkr:=actual_parcel_run s q k p r items _ (actual_view s values present q k) hp
  have hvfalse:heldAt values q p r items k=false:=by
    rw [hkfalse] at hkr
    exact (EStateM.Result.ok.inj hkr).1.symm
  have hcand:k∈candidates values q base:=(candidates_members values q base k).mpr hkm
  have hu:=search_captures base (heldAt values q p r items) (candidates values q base) k hcand hkbase hvfalse (-1)
  refine ⟨_,actual_dispatchable s values present q base p r items hp,hu.1,?_⟩
  have hle:=hu.2
  omega

#print axioms actual_dispatchable_captures

#print axioms search_preserves
#print axioms search_captures
#print axioms actual_search
#print axioms candidates_members
#print axioms actual_dispatchable
end EventSearchProofs
