-- Universal distinct-writer counting invariant for the actual bounded bitmap transition.
import Proofs.SrfBitPrimitives
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace DistinctWriterSeenProofs

def key (q b:Nat):Nat:=4*q+b

theorem key_bounded (q b:Nat) (hq:q≤144) (hb:b≤3):key q b<580:=by unfold key;omega

theorem key_injective (q b q2 b2:Nat) (hb:b≤3) (hb2:b2≤3)
 (he:key q b=key q2 b2):q=q2 ∧ b=b2:=by unfold key at he;omega

structure Witness (seen:BitVec 580) (count:Nat) (writers:List Nat):Prop where
 nodup:writers.Nodup
 bounded:∀k,k∈writers→k<580
 represented:∀k,k<580→(seen.getLsbD k=true ↔ k∈writers)
 counted:count=writers.length

theorem empty_witness:Witness (0#580) 0 []:=by
 constructor
 · simp
 · simp
 · intro k hk;simp
 · rfl

def sourceStep (seen:BitVec 580) (count k:Nat):BitVec 580×Nat:=
 if (Sail.BitVec.access seen k)==0#1 then (Sail.BitVec.update seen k 1#1,count+1) else (seen,count)

theorem actual_zero_query (seen:BitVec 580) (k:Nat) (hk:k<580):
 ((Sail.BitVec.access seen k)==0#1)= !seen.getLsbD k:=by
 simp only [Sail.BitVec.access,getElem!_pos seen k hk,←BitVec.getLsbD_eq_getElem hk]
 cases seen.getLsbD k <;> rfl

theorem actual_mark_bit (seen:BitVec 580) (k j:Nat) (hj:j<580):
 (Sail.BitVec.update seen k 1#1).getLsbD j=(if j=k then true else seen.getLsbD j):=by
 exact SrfBitPrimitiveProofs.actual_update_bit seen k j 1#1 hj

theorem actual_step_witness (seen:BitVec 580) (count k:Nat) (writers:List Nat)
 (hw:Witness seen count writers) (hk:k<580):
 ∃nextwriters,Witness (sourceStep seen count k).1 (sourceStep seen count k).2 nextwriters ∧
  ∀j,j∈nextwriters ↔ j=k ∨ j∈writers:=by
 rw [sourceStep,actual_zero_query seen k hk]
 cases hs:seen.getLsbD k
 · simp only [Bool.not_false,if_true]
   have hn:k∉writers:=by
    intro hm
    have ht:seen.getLsbD k=true:=(hw.represented k hk).mpr hm
    rw [hs] at ht;cases ht
   refine ⟨k::writers,?_,?_⟩
   · constructor
     · exact List.nodup_cons.mpr ⟨hn,hw.nodup⟩
     · intro j hj
       rcases List.mem_cons.mp hj with he|he
       · simpa [he] using hk
       · exact hw.bounded j he
     · intro j hj
       rw [actual_mark_bit seen k j hj]
       by_cases same:j=k
       · simp [same]
       · simp [same,hw.represented j hj,List.mem_cons]
     · simp [hw.counted]
   · intro j;exact List.mem_cons
 · simp only [Bool.not_true,Bool.false_eq_true,if_false]
   have hm:k∈writers:=(hw.represented k hk).mp hs
   refine ⟨writers,hw,?_⟩
   intro j
   constructor
   · exact Or.inr
   · intro h;rcases h with he|he
     · simpa [he] using hm
     · exact he

def scan:List Nat→(BitVec 580×Nat)→(BitVec 580×Nat)
 | [],state=>state
 | k::rest,state=>scan rest (sourceStep state.1 state.2 k)

theorem actual_scan_witness (input:List Nat) (seen:BitVec 580) (count:Nat) (writers:List Nat)
 (hw:Witness seen count writers) (hi:∀k,k∈input→k<580):
 ∃out,Witness (scan input (seen,count)).1 (scan input (seen,count)).2 out ∧
  ∀j,j∈out ↔ j∈input ∨ j∈writers:=by
 induction input generalizing seen count writers with
 | nil=>exact ⟨writers,hw,by simp⟩
 | cons k rest ih=>
   obtain ⟨next,hn,hm⟩:=actual_step_witness seen count k writers hw (hi k (by simp))
   obtain ⟨out,ho,hmem⟩:=ih (sourceStep seen count k).1 (sourceStep seen count k).2 next hn
    (fun j hj=>hi j (List.mem_cons_of_mem _ hj))
   refine ⟨out,ho,?_⟩
   intro j
   rw [hmem,hm,List.mem_cons]
   simp only [or_assoc,or_comm,or_left_comm]

theorem complete_distinct_other_count (otherKeys:List Nat) (hi:∀k,k∈otherKeys→k<580):
 ∃representatives:List Nat,representatives.Nodup ∧
  (scan otherKeys (0#580,0)).2=representatives.length ∧
  ∀k,k∈representatives ↔ k∈otherKeys:=by
 obtain ⟨out,hw,hm⟩:=actual_scan_witness otherKeys (0#580) 0 [] empty_witness hi
 exact ⟨out,hw.nodup,hw.counted,by simpa using hm⟩

end DistinctWriterSeenProofs
