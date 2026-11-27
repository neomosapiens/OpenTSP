-- All stream cells and all mask bits of one report lane, actual source loops.
import Proofs.SrfLoopTable
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfMergeSemanticProofs
open ResourceFrameProofs SrfMergeProofs ResourceScanProofs SrfLoopTableProofs SrfBitPrimitiveProofs

def wordStep (old new : BitVec Out.Functions.superlanes) (x s : Int) (keep take : Bool) :=
 if (s <b 3) then (if (Sail.BitVec.access new x)==1#1 then Sail.BitVec.update old x 1#1 else old)
 else if (s ≥b 4) then
   (if take then Sail.BitVec.update old x (Sail.BitVec.access new x)
    else if !keep then (if (Sail.BitVec.access new x)==1#1 then Sail.BitVec.update old x 1#1 else old) else old)
 else old

theorem actual_stream_cell (sp : Vector (BitVec Out.Functions.superlanes) 30) (x : Int)
 (keep take : Bool) (e : srf_rep) (k j : Nat) (hk:k<30) (hj:j<30) :
 (streamStep sp x k keep take e).sp[j]! =
 (if j=k then wordStep e.sp[j]! sp[k]! x k keep take else e.sp[j]!) := by
 unfold streamStep wordStep
 simp only [Int.toNat_natCast,Out.Functions.not]
 by_cases he:j=k
 · subst j
   simp [vectorUpdate,Vector.set!,hk]
   rfl
 · rw [vector_update_other 30 j k e.sp _ hj hk he]
   simp [he]

theorem actual_stream_table (e : srf_rep) (prio : Int) (sp : Vector (BitVec Out.Functions.superlanes) 30)
 (x : Int) (keep take : Bool) (j : Nat) (hj:j<30) :
 (streamLoop e prio sp x keep take).sp[j]! =wordStep e.sp[j]! sp[j]! x j keep take := by
 unfold streamLoop
 change (IntRange.forIn' (m:=Id) {start:=0,stop:=29,step:=1} e
   (fun s _ a => pure (.yield (streamStep sp x s keep take a)))).sp[j]! =_
 rw [actual_indexed_range_fold _ _ _ rfl]
 have h:=fold_at (fun (e:srf_rep) j => e.sp[j]!) (streamStep sp x · keep take)
  (fun j old => wordStep old sp[j]! x j keep take) 30
  (fun k st j hk hj => actual_stream_cell sp x keep take st k j hk hj) 30 0 j e (by decide) hj
 simpa [show 0≤j ∧ j<0+30 from by omega] using h

def bitPolicy (s : Nat) (keep take old new : Bool) : Bool :=
 if s<3 then old||new else if s=3 then old else if take then new else if keep then old else old||new

theorem word_bit_policy (old new : BitVec Out.Functions.superlanes) (x : Nat) (s : Nat)
 (keep take : Bool) (hx:x<Out.Functions.superlanes) :
 (wordStep old new x s keep take).getLsbD x=
 bitPolicy s keep take (old.getLsbD x) (new.getLsbD x) := by

 have upd (b : BitVec 1) : (Sail.BitVec.update old x b).getLsbD x=b.getLsbD 0 := by
   rw [actual_update_bit old x x b hx]
   simp only [ite_true]
 have acc:(Sail.BitVec.access new x).getLsbD 0=new.getLsbD x := by
   simp [Sail.BitVec.access,BitVec.getLsbD_ofBool,getElem!_pos new x hx,←BitVec.getLsbD_eq_getElem hx]
 have cond:=access_one new x hx
 by_cases hs:s<3
 · have hi:(s:Int)<3:=Int.ofNat_lt.mpr hs
   simp only [wordStep,bitPolicy,Int.toNat_natCast,show decide ((s:Int)<3)=true from by simp [hi],hs,ite_true]
   cases hn:new.getLsbD x <;> simp_all [upd]
 · by_cases he:s=3
   · subst s
     simp [wordStep,bitPolicy]
   · have hnat:(4:Nat)≤s:=by omega
     have hi:(4:Int)≤(s:Int):=Int.ofNat_le.mpr hnat
     have hn:¬(s:Int)<(3:Int):=Int.not_lt.mpr (Int.ofNat_le.mpr (show (3:Nat)≤s from by omega))
     simp only [wordStep,bitPolicy,Int.toNat_natCast,show decide ((s:Int)<3)=false from by simp [hn],hs,he,ite_false,
       show decide ((s:Int)≥4)=true from by simp [hi],ite_true]
     cases keep <;> cases take <;> cases hv:new.getLsbD x <;> simp_all [upd]

theorem word_other_bit (old new : BitVec Out.Functions.superlanes) (x j : Nat) (s : Int)
 (keep take : Bool) (hj:j<Out.Functions.superlanes) (hne:j≠x) :
 (wordStep old new x s keep take).getLsbD j=old.getLsbD j := by
 have hu (b:BitVec 1):(Sail.BitVec.update old x b).getLsbD j=old.getLsbD j := by
   rw [actual_update_bit old x j b hj,if_neg hne]
 unfold wordStep
 simp only [Int.toNat_natCast]
 all_goals try split
 all_goals try split
 all_goals try split
 all_goals try split
 all_goals try split
 all_goals first | rfl | exact hu _

theorem stream_step_pr (sp : Vector (BitVec Out.Functions.superlanes) 30) (x s : Int)
 (keep take : Bool) (e:srf_rep) : (streamStep sp x s keep take e).pr=e.pr := by rfl

theorem stream_pr (e:srf_rep) (prio:Int) (sp:Vector (BitVec Out.Functions.superlanes) 30)
 (x:Int) (keep take:Bool) : (streamLoop e prio sp x keep take).pr=e.pr := by
 unfold streamLoop
 change (IntRange.forIn' (m:=Id) {start:=0,stop:=29,step:=1} e
   (fun s _ a => pure (.yield (streamStep sp x s keep take a)))).pr=e.pr
 rw [actual_indexed_range_fold _ _ _ rfl]
 exact fold_keeps (fun e:srf_rep=>e.pr) _ (fun s a=>stream_step_pr sp x s keep take a) _ _ e

theorem stream_other_bit (e:srf_rep) (prio:Int) (sp:Vector (BitVec Out.Functions.superlanes) 30)
 (x j k:Nat) (keep take:Bool) (hj:j<Out.Functions.superlanes) (hk:k<30) (hne:j≠x) :
 ((streamLoop e prio sp x keep take).sp[k]!).getLsbD j=(e.sp[k]!).getLsbD j := by
 rw [actual_stream_table e prio sp x keep take k hk]
 exact word_other_bit _ _ _ _ _ _ _ hj hne

section
attribute [local irreducible] streamLoop

theorem lane_priority (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (x j:Nat)
 (hx:x<Out.Functions.superlanes) (hj:j<Out.Functions.superlanes) :
 (laneStep prio xs sp x e).pr[j]! =
 (if xs.getLsbD x && decide (j=x) then max e.pr[j]! prio else e.pr[j]!) := by
 have acc:=access_one xs x hx
 unfold laneStep
 dsimp only [Id.run]
 split
 · rename_i selected
   have hbit:xs.getLsbD x=true:=acc.mp (by simpa only [Int.toNat_natCast] using selected)
   split
   · rename_i higher
     change (vectorUpdate (streamLoop e prio sp x _ _).pr x prio)[j]! = _
     rw [stream_pr]
     have hp:prio>e.pr[x]!:=by simpa only [stream_pr,Int.toNat_natCast,decide_eq_true_eq] using higher
     by_cases he:j=x
     · subst j
       simp only [hbit,Bool.true_and,beq_self_eq_true,decide_true,ite_true,
         Int.max_eq_right (by omega:e.pr[x]!≤prio)]
       change (Vector.set! e.pr x prio)[x]! = prio
       change (e.pr.setIfInBounds x prio)[x]! = prio
       rw [getElem!_pos (e.pr.setIfInBounds x prio) x hx]
       exact Vector.getElem_setIfInBounds_self (xs:=e.pr) (x:=prio) hx
     · change (Vector.set! e.pr x prio)[j]! = _
       change (e.pr.setIfInBounds x prio)[j]! = _
       simp only [hbit,Bool.true_and,show decide (j=x)=false from by simp [he],Bool.false_eq_true,ite_false]
       rw [getElem!_pos (e.pr.setIfInBounds x prio) j hj,getElem!_pos e.pr j hj]
       exact Vector.getElem_setIfInBounds_ne (xs:=e.pr) (x:=prio) hj (Ne.symm he)
   · rename_i higher
     change (streamLoop e prio sp x _ _).pr[j]! = _
     rw [stream_pr]
     have hp:¬prio>e.pr[x]!:=by simpa only [stream_pr,Int.toNat_natCast,decide_eq_true_eq] using higher
     by_cases he:j=x
     · subst j
       simp [hbit,Int.max_eq_left (by omega:prio≤e.pr[x]!)]
     · simp [hbit,he]
 · rename_i selected
   have hbit:xs.getLsbD x=false:=by
     cases hv:xs.getLsbD x
     · rfl
     · exact False.elim (selected (by simpa only [Int.toNat_natCast] using (acc.mpr hv)))
   change e.pr[j]! = _
   simp [hbit]

end

theorem actual_merge_priority (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (j:Nat) (hj:j<Out.Functions.superlanes) :
 (srf_rep_merge e prio xs sp).pr[j]! =
 (if xs.getLsbD j then max e.pr[j]! prio else e.pr[j]!) := by
 rw [actual_merge_source]
 unfold mergeFactory
 change (IntRange.forIn' (m:=Id) {start:=0,stop:=(Out.Functions.superlanes:Int)-1,step:=1} e
  (fun x _ a=>pure (.yield (laneStep prio xs sp x a)))).pr[j]! = _
 rw [actual_indexed_range_fold _ _ _ rfl]
 have hr:(Out.Functions.superlanes:Int)-1-0+1=Out.Functions.superlanes:=by omega
 rw [hr]
 have hf:=fold_at (fun (e:srf_rep) j=>e.pr[j]!) (laneStep prio xs sp)
  (fun j old=>if xs.getLsbD j then max old prio else old) Out.Functions.superlanes
  (by
   intro k st j hk hj
   dsimp only
   have hl:=lane_priority st prio xs sp k j hk hj
   rw [hl]
   by_cases he:j=k
   · subst j; simp
   · simp [he])
  Out.Functions.superlanes 0 j e (by omega) hj
 simpa [Nat.zero_add,Int.toNat_natCast,hj] using hf

def parityWord (e:srf_rep) (x:Int) : BitVec Out.Functions.superlanes :=
 let n3:Int := (if (Sail.BitVec.access e.sp[0]! x)==1#1 then 1 else 0)+
  (if (Sail.BitVec.access e.sp[1]! x)==1#1 then 1 else 0)+
  (if (Sail.BitVec.access e.sp[2]! x)==1#1 then 1 else 0)
 Sail.BitVec.update e.sp[3]! x (if Int.emod n3 2==1 then 1#1 else 0#1)

def finishLane (e:srf_rep) (prio x:Int) : srf_rep :=
 let e':={e with sp:=vectorUpdate e.sp 3 (parityWord e x)}
 if (prio >b e'.pr[x]!) then {e' with pr:=vectorUpdate e'.pr x prio} else e'

theorem actual_lane_factory (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (x:Int) :
 laneStep prio xs sp x e=
 if (Sail.BitVec.access xs x)==1#1 then
  finishLane (streamLoop e prio sp x (e.pr[x]!>prio) (e.pr[x]!<prio)) prio x
 else e := by
 unfold laneStep finishLane parityWord
 dsimp only [Id.run]
 split
 · split <;> rfl
 · rfl

section
attribute [local irreducible] streamLoop parityWord

theorem finish_lane_sp (e:srf_rep) (prio x:Int) :
 (finishLane e prio x).sp=vectorUpdate e.sp 3 (parityWord e x) := by
 unfold finishLane
 dsimp only
 split <;> rfl

theorem lane_stream_bit (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (x j k:Nat)
 (hx:x<Out.Functions.superlanes) (hj:j<Out.Functions.superlanes) (hk:k<30) (h3:k≠3) :
 ((laneStep prio xs sp x e).sp[k]!).getLsbD j=
 (if xs.getLsbD x && decide (j=x) then
  bitPolicy k (decide (e.pr[x]!>prio)) (decide (e.pr[x]!<prio))
    (e.sp[k]!.getLsbD j) (sp[k]!.getLsbD j)
 else e.sp[k]!.getLsbD j) := by
 rw [actual_lane_factory]
 have ac:=access_one xs x hx
 by_cases selected:((Sail.BitVec.access xs x)==1#1)=true
 · have hb:=ac.mp selected
   simp only [Int.toNat_natCast,selected,ite_true,finish_lane_sp]
   rw [vector_update_other 30 k 3 _ _ hk (by decide) h3]
   by_cases same:j=x
   · subst j
     rw [actual_stream_table _ _ _ x _ _ k hk]
     simpa [hb,Int.toNat_natCast] using
       word_bit_policy e.sp[k]! sp[k]! x k (e.pr[(x:Int)]!>prio) (e.pr[(x:Int)]!<prio) hx
   · rw [stream_other_bit e prio sp x j k _ _ hj hk same]
     simp [hb,same]
 · have hb:xs.getLsbD x=false:=by cases hv:xs.getLsbD x <;> simp_all
   simp [Int.toNat_natCast,selected,hb]
end

def laneCell (e:srf_rep) (s j:Nat) : Int×Bool := (e.pr[j]!,e.sp[s]!.getLsbD j)
def cellMerge (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (s j:Nat) (old:Int×Bool) : Int×Bool :=
 if xs.getLsbD j then
  (max old.1 prio,bitPolicy s (decide (old.1>prio)) (decide (old.1<prio)) old.2 (sp[s]!.getLsbD j))
 else old

theorem lane_cell_table (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (s k j:Nat)
 (hs:s<30) (h3:s≠3) (hk:k<Out.Functions.superlanes) (hj:j<Out.Functions.superlanes) :
 laneCell (laneStep prio xs sp k e) s j=
 (if j=k then cellMerge prio xs sp s k (laneCell e s j) else laneCell e s j) := by
 apply Prod.ext
 · dsimp only [laneCell,cellMerge]
   rw [lane_priority e prio xs sp k j hk hj]
   by_cases he:j=k
   · subst j; cases xs.getLsbD k <;> simp
   · simp [he]
 · dsimp only [laneCell,cellMerge]
   rw [lane_stream_bit e prio xs sp k j s hk hj hs h3]
   by_cases he:j=k
   · subst j; cases xs.getLsbD k <;> simp
   · simp [he]

theorem actual_merge_cell (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (s j:Nat)
 (hs:s<30) (h3:s≠3) (hj:j<Out.Functions.superlanes) :
 laneCell (srf_rep_merge e prio xs sp) s j=cellMerge prio xs sp s j (laneCell e s j) := by
 rw [actual_merge_source]
 unfold mergeFactory
 change laneCell (IntRange.forIn' (m:=Id) {start:=0,stop:=(Out.Functions.superlanes:Int)-1,step:=1} e
  (fun x _ a=>pure (.yield (laneStep prio xs sp x a)))) s j = _
 rw [actual_indexed_range_fold _ _ _ rfl]
 have hr:(Out.Functions.superlanes:Int)-1-0+1=Out.Functions.superlanes:=by omega
 rw [hr]
 have hf:=fold_at (fun e j=>laneCell e s j) (laneStep prio xs sp)
  (cellMerge prio xs sp s) Out.Functions.superlanes
  (fun k st j hk hj=>lane_cell_table st prio xs sp s k j hs h3 hk hj)
  Out.Functions.superlanes 0 j e (by omega) hj
 simpa [hj] using hf

def odd3 (a b c:Bool) : Bool := Bool.xor (Bool.xor a b) c

theorem access_bool (v:BitVec Out.Functions.superlanes) (x:Nat) (hx:x<Out.Functions.superlanes) :
 ((Sail.BitVec.access v x)==1#1)=v.getLsbD x := by
 have h:=access_one v x hx
 cases hv:v.getLsbD x <;> simp_all

theorem parity_generic (a b c old:BitVec Out.Functions.superlanes) (x j:Nat)
 (hx:x<Out.Functions.superlanes) (hj:j<Out.Functions.superlanes) :
 (Sail.BitVec.update old x (if Int.emod
  ((if (Sail.BitVec.access a x)==1#1 then 1 else 0)+
   (if (Sail.BitVec.access b x)==1#1 then 1 else 0)+
   (if (Sail.BitVec.access c x)==1#1 then 1 else 0)) 2==1 then 1#1 else 0#1)).getLsbD j=
 if j=x then odd3 (a.getLsbD x) (b.getLsbD x) (c.getLsbD x) else old.getLsbD j := by
 rw [actual_update_bit _ x j _ hj]
 by_cases he:j=x
 · subst j
   rw [access_bool a x hx,access_bool b x hx,access_bool c x hx]
   simp only [ite_true]
   cases a.getLsbD x <;> cases b.getLsbD x <;> cases c.getLsbD x <;> rfl
 · simp only [if_neg he]

theorem parity_bit (e:srf_rep) (x j:Nat) (hx:x<Out.Functions.superlanes) (hj:j<Out.Functions.superlanes) :
 (parityWord e x).getLsbD j=
 if j=x then odd3 (e.sp[0]!.getLsbD x) (e.sp[1]!.getLsbD x) (e.sp[2]!.getLsbD x)
 else e.sp[3]!.getLsbD j := by
 exact parity_generic e.sp[0]! e.sp[1]! e.sp[2]! e.sp[3]! x j hx hj

theorem lane_parity_bit (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (x j:Nat)
 (hx:x<Out.Functions.superlanes) (hj:j<Out.Functions.superlanes) :
 ((laneStep prio xs sp x e).sp[3]!).getLsbD j=
 if xs.getLsbD x && decide (j=x) then
  odd3 (e.sp[0]!.getLsbD j || sp[0]!.getLsbD j)
       (e.sp[1]!.getLsbD j || sp[1]!.getLsbD j)
       (e.sp[2]!.getLsbD j || sp[2]!.getLsbD j)
 else e.sp[3]!.getLsbD j := by
 rw [actual_lane_factory]
 by_cases selected:((Sail.BitVec.access xs x)==1#1)=true
 · have hb:xs.getLsbD x=true:=(access_one xs x hx).mp selected
   simp only [Int.toNat_natCast,selected,ite_true,finish_lane_sp]
   change ((Vector.set! (streamLoop e prio sp x (e.pr[(x:Int)]!>prio) (e.pr[(x:Int)]!<prio)).sp 3 (parityWord (streamLoop e prio sp x (e.pr[(x:Int)]!>prio) (e.pr[(x:Int)]!<prio)) x))[3]!).getLsbD j = _
   have hu:(Vector.set! (streamLoop e prio sp x (e.pr[(x:Int)]!>prio) (e.pr[(x:Int)]!<prio)).sp 3 (parityWord (streamLoop e prio sp x (e.pr[(x:Int)]!>prio) (e.pr[(x:Int)]!<prio)) x))[3]!
     =parityWord (streamLoop e prio sp x (e.pr[(x:Int)]!>prio) (e.pr[(x:Int)]!<prio)) x := by simp [Vector.set!]
   rw [hu]
   erw [parity_bit _ x j hx hj]
   by_cases he:j=x
   · subst j
     simp only [ite_true,hb,Bool.true_and,decide_true]
     rw [actual_stream_table _ _ _ x _ _ 0 (by decide),actual_stream_table _ _ _ x _ _ 1 (by decide),actual_stream_table _ _ _ x _ _ 2 (by decide)]
     erw [word_bit_policy _ _ x 0 _ _ hx,word_bit_policy _ _ x 1 _ _ hx,word_bit_policy _ _ x 2 _ _ hx]
     simp only [bitPolicy]
     rfl
   · rw [if_neg he,stream_other_bit _ _ _ x j 3 _ _ hj (by decide) he]
     simp [hb,he]
 · have hb:xs.getLsbD x=false:=by cases hv:xs.getLsbD x <;> simp_all [access_bool xs x hx]
   simp [Int.toNat_natCast,selected,hb]

def parityCell (e:srf_rep) (j:Nat) : Bool×Bool×Bool×Bool :=
 (e.sp[0]!.getLsbD j,e.sp[1]!.getLsbD j,e.sp[2]!.getLsbD j,e.sp[3]!.getLsbD j)
def parityMerge (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (j:Nat) (old:Bool×Bool×Bool×Bool) : Bool×Bool×Bool×Bool :=
 if xs.getLsbD j then
  let a:=old.1 || sp[0]!.getLsbD j
  let b:=old.2.1 || sp[1]!.getLsbD j
  let c:=old.2.2.1 || sp[2]!.getLsbD j
  (a,b,c,odd3 a b c)
 else old

theorem lane_parity_cell (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (k j:Nat)
 (hk:k<Out.Functions.superlanes) (hj:j<Out.Functions.superlanes) :
 parityCell (laneStep prio xs sp k e) j=
 (if j=k then parityMerge xs sp k (parityCell e j) else parityCell e j) := by
 have h0:=lane_stream_bit e prio xs sp k j 0 hk hj (by decide) (by decide)
 have h1:=lane_stream_bit e prio xs sp k j 1 hk hj (by decide) (by decide)
 have h2:=lane_stream_bit e prio xs sp k j 2 hk hj (by decide) (by decide)
 have h3:=lane_parity_bit e prio xs sp k j hk hj
 unfold parityCell parityMerge
 rw [h0,h1,h2,h3]
 by_cases he:j=k
 · subst j
   cases xs.getLsbD k <;> simp [bitPolicy]
 · simp [he]

theorem actual_merge_parity_cell (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (j:Nat) (hj:j<Out.Functions.superlanes) :
 parityCell (srf_rep_merge e prio xs sp) j=parityMerge xs sp j (parityCell e j) := by
 rw [actual_merge_source]
 unfold mergeFactory
 change parityCell (IntRange.forIn' (m:=Id) {start:=0,stop:=(Out.Functions.superlanes:Int)-1,step:=1} e
  (fun x _ a=>pure (.yield (laneStep prio xs sp x a)))) j = _
 rw [actual_indexed_range_fold _ _ _ rfl]
 have hr:(Out.Functions.superlanes:Int)-1-0+1=Out.Functions.superlanes:=by omega
 rw [hr]
 have hf:=fold_at (fun e j=>parityCell e j) (laneStep prio xs sp)
  (parityMerge xs sp) Out.Functions.superlanes
  (fun k st j hk hj=>lane_parity_cell st prio xs sp k j hk hj)
  Out.Functions.superlanes 0 j e (by omega) hj
 simpa [hj] using hf

theorem actual_merge_even_parity (e:srf_rep) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) (j:Nat)
 (hj:j<Out.Functions.superlanes) (hs:xs.getLsbD j=true) :
 (srf_rep_merge e prio xs sp).sp[3]!.getLsbD j=
 odd3 ((srf_rep_merge e prio xs sp).sp[0]!.getLsbD j)
      ((srf_rep_merge e prio xs sp).sp[1]!.getLsbD j)
      ((srf_rep_merge e prio xs sp).sp[2]!.getLsbD j) := by
 have h:=actual_merge_parity_cell e prio xs sp j hj
 have hp:=congrArg (fun t:Bool×Bool×Bool×Bool=>t.2.2.2) h
 have h0:=congrArg (fun t:Bool×Bool×Bool×Bool=>t.1) h
 have h1:=congrArg (fun t:Bool×Bool×Bool×Bool=>t.2.1) h
 have h2:=congrArg (fun t:Bool×Bool×Bool×Bool=>t.2.2.1) h
 simp only [parityCell,parityMerge,hs,ite_true] at hp h0 h1 h2
 rw [hp,h0,h1,h2]

end SrfMergeSemanticProofs
