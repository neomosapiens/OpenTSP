import Proofs.DistinctWriterSeen
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace SrfCommitPolicyProofs
open DistinctWriterSeenProofs

def CommitDomain (xs : List commit) : Prop := ∀ e,e∈xs→e.q≤144 ∧ e.b≤3

def other (e o : commit) : Bool := same_target e o && !(o.q==e.q && o.b==e.b)
def otherKeys (e : commit) : List commit → List Nat
 | [] => []
 | o::rest => if other e o then key o.q o.b :: otherKeys e rest else otherKeys e rest

theorem other_keys_bounded (e : commit) (xs : List commit) (domain : CommitDomain xs) :
 ∀ k,k∈otherKeys e xs→k<580 := by
 induction xs with
 | nil => simp [otherKeys]
 | cons o rest ih =>
   have hd:=domain o (by simp)
   have tail:CommitDomain rest:=fun c hc=>domain c (by simp [hc])
   intro k hk
   by_cases h:other e o=true
   · simp only [otherKeys,h,if_true,List.mem_cons] at hk
     rcases hk with eq|member
     · subst k;exact key_bounded o.q o.b hd.1 hd.2
     · exact ih tail k member
   · simp only [otherKeys,h,if_false] at hk
     exact ih tail k hk

/-- Exact full production recursion: the native bitmap/count projection is the
    independently typed key scan over ALL same-target other writer records. -/
theorem actual_scan_count (e : commit) (i : Nat) (xs : List commit) (j : Nat) (a : win_acc) :
 let out:=commit_scan e i xs j a
 (out.seen,out.others)=scan (otherKeys e xs) (a.seen,a.others) := by
 induction xs generalizing j a with
 | nil => rfl
 | cons o rest ih =>
   cases ht:same_target e o
   · simpa only [commit_scan,ht,Bool.false_eq_true,if_false,otherKeys,other,ht,Bool.false_and] using ih (j+1) a
   · cases ho:(o.q==e.q && o.b==e.b)
     · simp only [commit_scan,ht,ho,Bool.false_eq_true,if_false,if_true,otherKeys,other,ht,ho,Bool.not_false,Bool.true_and]
       rw [ih]
       simp only [scan,sourceStep,key]
       have hk : (4 * (o.q:Int) + o.b).toNat = 4*o.q+o.b := by omega
       have hc : ((a.others:Int)+1).toNat = a.others+1 := by omega
       simp only [hk,hc]
       split <;> split <;> split <;> rfl
     · simp only [commit_scan,ht,ho,if_true,otherKeys,other,ht,ho,Bool.not_true,Bool.true_and,Bool.false_eq_true,if_false]
       split
       · split <;> simpa using ih (j+1) _
       · exact ih (j+1) a


/-- Mathematical cardinality witness: no bitmap or scan occurs in this predicate. -/
def DistinctCount (keys : List Nat) (n : Nat) : Prop :=
 ∃ reps : List Nat,reps.Nodup ∧ reps.length=n ∧ ∀k,k∈reps↔k∈keys

theorem actual_scan_distinct (e : commit) (i : Nat) (xs : List commit)
 (domain : CommitDomain xs) (a : win_acc) (hz : a.seen=0#580) (hn : a.others=0) :
 DistinctCount (otherKeys e xs) (commit_scan e i xs 0 a).others := by
 have h:=actual_scan_count e i xs 0 a
 obtain ⟨reps,nd,cnt,mem⟩:=complete_distinct_other_count (otherKeys e xs)
  (other_keys_bounded e xs domain)
 have hc:=congrArg Prod.snd h
 simp only [hz,hn] at hc
 exact ⟨reps,nd,cnt.symm.trans hc.symm,mem⟩

def own (e o : commit) : Bool := same_target e o && (o.q==e.q && o.b==e.b)
def allPaired (e : commit) (xs : List commit) : Bool :=
 xs.all (fun o=> !other e o || or_pair e o)
def leastWriter (e : commit) (xs : List commit) : Bool :=
 xs.all (fun o=> !other e o || !writer_before o e)
def noEarlier (e : commit) (i : Nat) : List commit → Nat → Bool
 | [],_=>true
 | o::rest,j=>(if j<i then !own e o else true) && noEarlier e i rest (j+1)
def collectOwn (e : commit) (i : Nat) : List commit → Nat → List commit
 | [],_=>[]
 | o::rest,j=>if own e o && (j != i) then o::collectOwn e i rest (j+1)
              else collectOwn e i rest (j+1)
def collectOther (e : commit) (xs : List commit) : List commit := xs.filter (other e)
def orValues (xs : List commit) (v : vec) : vec :=
 xs.foldl (fun a o=>a ||| o.value) v
def orOverlays (xs : List commit) (v : ecc_ov) : ecc_ov :=
 xs.foldl (fun a o=>a ||| o.ov) v

/-- Declarative policy projections: all writers paired; least writer wins;
 the first own occurrence stands; every payload and overlay record contributes. -/
theorem actual_scan_policy_fields (e : commit) (i : Nat) (xs : List commit) (j : Nat) (a : win_acc) :
 let out:=commit_scan e i xs j a
 out.paired = (a.paired && allPaired e xs) ∧
 out.first = (a.first && leastWriter e xs) ∧
 out.stands = (a.stands && noEarlier e i xs j) ∧
 out.own = orValues (collectOwn e i xs j) a.own ∧
 out.own_ov = orOverlays (collectOwn e i xs j) a.own_ov ∧
 out.merged = orValues (collectOther e xs) a.merged ∧
 out.merged_ov = orOverlays (collectOther e xs) a.merged_ov := by
 induction xs generalizing j a with
 | nil=>simp [commit_scan,allPaired,leastWriter,noEarlier,collectOwn,collectOther,orValues,orOverlays]
 | cons o rest ih=>
   have hj : ((j:Int)+1).toNat=j+1 := by omega
   simp only [commit_scan,hj]
   generalize hn : (if same_target e o then _ else a) = next
   have h := ih (j+1) next
   simp only [h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2]
   subst next
   cases ht:same_target e o <;> cases ho:(o.q==e.q && o.b==e.b) <;>
    cases hp:or_pair e o <;> cases hf:writer_before o e <;>
    cases hs:((Sail.BitVec.access a.seen (4*(o.q:Int)+o.b).toNat)==0#1) <;>
    by_cases he:j=i <;> by_cases hl:j<i <;>
    simp only [allPaired,leastWriter,List.all_cons,noEarlier,collectOwn,collectOther,List.filter_cons,other,own,ht,ho,hp,hf,he,hl,hs,Functions.not] <;>
    simp_all [allPaired,leastWriter,noEarlier,collectOwn,collectOther,other,own,
     orValues,orOverlays,Functions.not,Bool.and_assoc,
     Int.natCast_add,Int.toNat_natCast]


/-- The independent output rule contains no production scan or bitmap. -/
def policyOutput (e : commit) (i : Nat) (xs : List commit) (n : Nat) : Bool × Bool × vec × ecc_ov :=
 let ownv := orValues (collectOwn e i xs 0) e.value
 let owno := orOverlays (collectOwn e i xs 0) e.ov
 if !noEarlier e i xs 0 then (false,false,ownv,owno)
 else if n==0 then (false,true,ownv,owno)
 else if n==1 && allPaired e xs then
  (false,leastWriter e xs, ownv ||| orValues (collectOther e xs) zero_vec,
   owno ||| orOverlays (collectOther e xs) ecc_ov_zero)
 else (true,leastWriter e xs,ownv,owno)

def GroupPolicy (e : commit) (i : Nat) (xs : List commit) (result : Bool × Bool × vec × ecc_ov) : Prop :=
 ∃ n,DistinctCount (otherKeys e xs) n ∧ result=policyOutput e i xs n

theorem actual_commit_wins_policy (e : commit) (i : Nat) (xs : List commit)
 (domain : CommitDomain xs) : GroupPolicy e i xs (commit_wins e i xs) := by
 let a : win_acc := {others:=0,seen:=0#580,paired:=true,first:=true,stands:=true, own:=e.value,merged:=zero_vec,own_ov:=e.ov,merged_ov:=ecc_ov_zero}
 have distinct := actual_scan_distinct e i xs domain a rfl rfl
 have fields := actual_scan_policy_fields e i xs 0 a
 refine ⟨(commit_scan e i xs 0 a).others,distinct,?_⟩
 unfold commit_wins
 change (if Functions.not (commit_scan e i xs 0 a).stands then
    (false,false,(commit_scan e i xs 0 a).own,(commit_scan e i xs 0 a).own_ov)
   else if (commit_scan e i xs 0 a).others==0 then
    (false,true,(commit_scan e i xs 0 a).own,(commit_scan e i xs 0 a).own_ov)
   else if (commit_scan e i xs 0 a).others==1 && (commit_scan e i xs 0 a).paired then
    (false,(commit_scan e i xs 0 a).first,
     (commit_scan e i xs 0 a).own ||| (commit_scan e i xs 0 a).merged,
     (commit_scan e i xs 0 a).own_ov ||| (commit_scan e i xs 0 a).merged_ov)
   else (true,(commit_scan e i xs 0 a).first,(commit_scan e i xs 0 a).own,(commit_scan e i xs 0 a).own_ov)) = _
 simp only [policyOutput,fields.1,fields.2.1,fields.2.2.1,fields.2.2.2.1,
  fields.2.2.2.2.1,fields.2.2.2.2.2.1,fields.2.2.2.2.2.2,a,Bool.true_and,Functions.not]

-- These source bindings make target and priority selector mutations proof failures.
theorem actual_same_target (e o : commit) :
 same_target e o = (e.r==o.r && e.id==o.id && e.g==o.g) := by simp only [same_target,Bool.and_assoc]

theorem actual_writer_before (e o : commit) :
 writer_before e o = (decide (e.q<o.q) || ((e.q==o.q) && decide (e.b<o.b))) := by rfl


theorem nodup_cardinality_ext (xs ys : List Nat) (hx : xs.Nodup) (hy : ys.Nodup)
 (mem : ∀ k,k∈xs↔k∈ys) : xs.length=ys.length := by
 induction xs generalizing ys with
 | nil=>
   cases ys with
   | nil=>rfl
   | cons y rest=>have h := (mem y).mpr (by simp);simp at h
 | cons x rest ih=>
   obtain ⟨hn,ht⟩ := List.nodup_cons.mp hx
   have hm : x∈ys := (mem x).mp (by simp)
   have he : ∀k,k∈rest↔k∈ys.erase x := by
    intro k
    rw [hy.mem_erase_iff]
    constructor
    · intro hk
      refine ⟨?_,(mem k).mp (List.mem_cons_of_mem _ hk)⟩
      intro eq;subst k;exact hn hk
    · intro ⟨hne,hk⟩
      rcases List.mem_cons.mp ((mem k).mpr hk) with eq|hr
      · exact False.elim (hne eq)
      · exact hr
   have hh := ih (ys.erase x) ht (hy.erase x) he
   have hl := List.length_erase_of_mem hm
   change rest.length+1=ys.length
   rw [hh]
   cases ys with
   | nil=>simp at hm
   | cons y tail=>simp only [List.length_cons] at hl ⊢; omega

theorem distinct_count_unique (keys : List Nat) (n m : Nat)
 (hn : DistinctCount keys n) (hm : DistinctCount keys m) : n=m := by
 obtain ⟨xs,hx,lx,mx⟩:=hn
 obtain ⟨ys,hy,ly,my⟩:=hm
 have hl:=nodup_cardinality_ext xs ys hx hy (fun k=>(mx k).trans (my k).symm)
 omega

theorem group_policy_unique (e : commit) (i : Nat) (xs : List commit)
 (a b : Bool×Bool×vec×ecc_ov) (ha : GroupPolicy e i xs a) (hb : GroupPolicy e i xs b) : a=b := by
 obtain ⟨n,hn,ha⟩:=ha
 obtain ⟨m,hm,hb⟩:=hb
 have eq:=distinct_count_unique (otherKeys e xs) n m hn hm
 simpa [ha,hb,eq]

#print axioms actual_commit_wins_policy
#print axioms group_policy_unique
end SrfCommitPolicyProofs


