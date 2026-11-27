-- Exact grouped/ungrouped arithmetic placement for every admitted width.
import Proofs.Wide128Bounds
import Proofs.ReduceSemantics
open Sail Out Out.Functions Out.ExecutionResult
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace LanePlacementProofs
open DotSemanticProofs

def actualParts (grp : Bool) (ws : Nat) : Nat :=
  if ((Out.Functions.not grp):Bool) then 1 else if ((ws==1):Bool) then 4 else if ((ws==2):Bool) then 2 else 1

def rowFactory (op : vxm_lane_op) (xa xb xm : planes) (ws parts : Nat) (l : Int) : SailM (BitVec 32) := do
  let x : (BitVec 32) := 0x00000000#32
  (( do
    let loop_i_lower := 0
    let loop_i_upper := (parts -i 1)
    let mut loop_vars_1 := x
    for i in [loop_i_lower:loop_i_upper:1]i do
      let x := loop_vars_1
      loop_vars_1 ← do
        let o : Int := (i *i ws)
        if (((0 ≤b o) && (o ≤b 3)) : Bool)
        then
          (do
            let r ← do
              (apply op (elem_at xa ws o l) (elem_at xb ws o l) (elem_at xm ws o l))
            let loop_j_lower := 0
            let loop_j_upper := (ws -i 1)
            let mut loop_vars_2 := x
            for j in [loop_j_lower:loop_j_upper:1]i do
              let x := loop_vars_2
              loop_vars_2 :=
                let k := (o +i j)
                if ((k ≤b 3) : Bool)
                then
                  (Sail.BitVec.updateSubrange x ((8 *i k) +i 7) (8 *i k)
                    (Sail.BitVec.extractLsb r ((8 *i j) +i 7) (8 *i j)))
                else x
            (pure loop_vars_2))
        else (pure x)
    (pure loop_vars_1) ) : SailM (BitVec 32) )

def laneFactory (op : vxm_lane_op) (dst : streamref) (a : vsrc) (b : (Option vsrc)) (m : (Option vsrc)) (ws : Nat) (wd : Nat) (g : (BitVec 1)) (chooseParts : Bool → Nat → Nat) : SailM ExecutionResult := do
  let grp : Bool := (g == 1#1)
  let rw : Int :=
    if (grp : Bool)
    then 4
    else ws
  let pa ← do (vxm_read a rw)
  let pb ← (( do
    match b with
    | .some s => (vxm_read s rw)
    | none => (pure (.Ok no_planes)) ) : SailM (Result planes diag) )
  let pm ← (( do
    match m with
    | .some s => (vxm_read s rw)
    | none => (pure (.Ok no_planes)) ) : SailM (Result planes diag) )
  match (pa, pb, pm) with
  | (.Err e, _, _) => (pure (Diagnostic e))
  | (_, .Err e, _) => (pure (Diagnostic e))
  | (_, _, .Err e) => (pure (Diagnostic e))
  | (.Ok xa, .Ok xb, .Ok xm) =>
    (do
      let out : (Vector (BitVec 32) Out.Functions.vlanes) := (vectorInit 0x00000000#32)
      let parts : Nat := chooseParts grp ws
      let out ← (( do
        let loop_l_lower := 0
        let loop_l_upper := (Out.Functions.vlanes -i 1)
        let mut loop_vars := out
        for l in [loop_l_lower:loop_l_upper:1]i do
          let out := loop_vars
          loop_vars ← do
            let x ← rowFactory op xa xb xm ws parts l
            (pure (vectorUpdate out l
                (← do
                  if (grp : Bool)
                  then (pure x)
                  else (apply op (elem xa ws l) (elem xb ws l) (elem xm ws l)))))
        (pure loop_vars) ) : SailM (Vector (BitVec 32) Out.Functions.vlanes) )
      match (← (write_planes dst (words_to_planes out)
          (if (grp : Bool)
          then 4
          else wd) (← (vxm_commit_cycle ())))) with
      | .Ok () => (pure (Issued ()))
      | .Err e => (pure (Diagnostic e)))

section SourceBinding
attribute [local irreducible] IntRange.forIn' Sail.BitVec.updateSubrange Sail.BitVec.extractLsb Out.Functions.apply

theorem actual_lanes_factory_source (op : vxm_lane_op) (dst : streamref) (a : vsrc) (b m : Option vsrc)
    (ws wd : Nat) (g : BitVec 1) :
    vxm_lanes op dst a b m ws wd g=laneFactory op dst a b m ws wd g actualParts := by rfl
end SourceBinding

theorem actual_element_widths (t : eltype) : eltype_width t=1 ∨ eltype_width t=2 ∨ eltype_width t=4 := by
  cases t <;> decide +kernel

theorem complete_parts_table : actualParts true 1=4 ∧ actualParts true 2=2 ∧ actualParts true 4=1 ∧
    (∀ ws, actualParts false ws=1) := by
  exact ⟨rfl,rfl,rfl,fun _ => rfl⟩

theorem actual_byte_geometry (ws : Nat) (hw : ws=1 ∨ ws=2 ∨ ws=4) (i j : Int)
    (hi : 0 ≤ i ∧ i < (actualParts true ws:Int)) (hj : 0 ≤ j ∧ j < (ws:Int)) :
    0 ≤ i*(ws:Int) ∧ i*(ws:Int) ≤ 3 ∧ 0 ≤ i*(ws:Int)+j ∧ i*(ws:Int)+j ≤ 3 ∧
      0 ≤ j ∧ j ≤ 3 := by
  rcases hw with h|h|h <;> subst ws <;> simp [actualParts,Out.Functions.not] at hi hj ⊢ <;> omega

theorem byte_partition (ws : Nat) (hw : ws=1 ∨ ws=2 ∨ ws=4) (k : Nat) (hk : k<4) :
    k/ws < actualParts true ws ∧ k%ws < ws ∧ k/ws*ws+k%ws=k := by
  rcases hw with h|h|h <;> subst ws <;> simp [actualParts,Out.Functions.not] <;> omega

theorem byte_partition_unique (ws : Nat) (hw : ws=1 ∨ ws=2 ∨ ws=4) (i j p q : Nat)
    (hj : j<ws) (hq : q<ws) (he : i*ws+j=p*ws+q) : i=p ∧ j=q := by
  rcases hw with h|h|h <;> subst ws <;> omega

def writeByte (x : BitVec 32) (b : BitVec 8) (slot : Nat) : BitVec 32 :=
  Sail.BitVec.updateSubrange' x (8*slot) 8 b

def packFour (b0 b1 b2 b3 : BitVec 8) : BitVec 32 :=
  writeByte (writeByte (writeByte (writeByte 0 b0 0) b1 1) b2 2) b3 3

theorem append_bit (n m i : Nat) (x : BitVec n) (y : BitVec m) :
    (x.append y).getLsbD i = if i < m then y.getLsbD i else x.getLsbD (i-m) := by
  exact @_root_.BitVec.getLsbD_append n m i x y

theorem pack_four_little_endian (b0 b1 b2 b3 : BitVec 8) :
    packFour b0 b1 b2 b3=((b3.append b2).append b1).append b0 := by
  apply BitVec.eq_of_getLsbD_eq_iff.mpr
  intro i hi
  have hc : i=0 ∨ i=1 ∨ i=2 ∨ i=3 ∨ i=4 ∨ i=5 ∨ i=6 ∨ i=7 ∨ i=8 ∨ i=9 ∨ i=10 ∨ i=11 ∨ i=12 ∨ i=13 ∨ i=14 ∨ i=15 ∨ i=16 ∨ i=17 ∨ i=18 ∨ i=19 ∨ i=20 ∨ i=21 ∨ i=22 ∨ i=23 ∨ i=24 ∨ i=25 ∨ i=26 ∨ i=27 ∨ i=28 ∨ i=29 ∨ i=30 ∨ i=31 := by omega
  simp only [append_bit]
  rcases hc with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> subst i <;>
    simp [packFour,writeByte,Sail.BitVec.updateSubrange',BitVec.zeroExtend]

def getByte (r : BitVec 32) (j : Nat) : BitVec 8 := Sail.BitVec.slice r (8*j) 8

def groupWord (ws : Nat) (R : Nat → BitVec 32) : BitVec 32 :=
  indexedFold (fun i x => indexedFold (fun j xx =>
    writeByte xx (getByte (R i.toNat) j.toNat) (i*(ws:Int)+j).toNat) 0 ws x)
    0 (actualParts true ws) 0

def groupByte (ws : Nat) (R : Nat → BitVec 32) (k : Nat) : BitVec 8 :=
  getByte (R (k/ws)) (k%ws)

theorem complete_group_word (ws : Nat) (hw : ws=1 ∨ ws=2 ∨ ws=4) (R : Nat → BitVec 32) :
    groupWord ws R=packFour (groupByte ws R 0) (groupByte ws R 1)
      (groupByte ws R 2) (groupByte ws R 3) := by
  rcases hw with h|h|h <;> subst ws <;>
    simp [groupWord,groupByte,packFour,indexedFold,actualParts,Out.Functions.not]

theorem group_word_concatenation (ws : Nat) (hw : ws=1 ∨ ws=2 ∨ ws=4) (R : Nat → BitVec 32) :
    groupWord ws R=(((groupByte ws R 3).append (groupByte ws R 2)).append
      (groupByte ws R 1)).append (groupByte ws R 0) := by
  rw [complete_group_word ws hw R,pack_four_little_endian]

theorem actual_byte_update (x r : BitVec 32) (slot j : Nat) (hs : slot<4) (hj : j<4) :
    Sail.BitVec.updateSubrange x (8*(slot:Int)+7) (8*(slot:Int))
      (Sail.BitVec.extractLsb r (8*(j:Int)+7) (8*(j:Int))) = writeByte x (getByte r j) slot := by
  have hc : slot=0 ∨ slot=1 ∨ slot=2 ∨ slot=3 := by omega
  have hd : j=0 ∨ j=1 ∨ j=2 ∨ j=3 := by omega
  rcases hc with h|h|h|h <;> subst slot <;>
    rcases hd with h|h|h|h <;> subst j <;> rfl

def sourceByteStep (o : Int) (r : BitVec 32) (j : Int) (x : BitVec 32) : BitVec 32 :=
  let k := (o +i j)
  if ((k ≤b 3):Bool) then Sail.BitVec.updateSubrange x ((8 *i k) +i 7) (8 *i k)
    (Sail.BitVec.extractLsb r ((8 *i j) +i 7) (8 *i j)) else x

def fillFactoryM (ws : Nat) (o : Int) (r x : BitVec 32) : SailM (BitVec 32) := do
  let mut out := x
  for j in [0:(ws -i 1):1]i do
    out := sourceByteStep o r j out
  pure out

def sourceFill (ws : Nat) (o : Int) (r x : BitVec 32) : BitVec 32 :=
  indexedFold (sourceByteStep o r) 0 ws x

theorem actual_fill_pure (ws : Nat) (o : Int) (r x : BitVec 32) :
    fillFactoryM ws o r x=pure (sourceFill ws o r x) := by
  unfold fillFactoryM
  simp only [bind_pure]
  change IntRange.forIn' (m:=SailM) {start:=0,stop:=(ws:Int)-1,step:=1} x
    (fun j _ out => pure (.yield (sourceByteStep o r j out))) = _
  simpa [sourceFill] using ReduceSemanticProofs.monadic_indexed_range_fold
    (m:=SailM) {start:=0,stop:=(ws:Int)-1,step:=1} (sourceByteStep o r) x rfl

def rowCompact (op : vxm_lane_op) (xa xb xm : planes) (ws parts : Nat) (l : Int) : SailM (BitVec 32) := do
  let mut x : BitVec 32 := 0
  for i in [0:(parts -i 1):1]i do
    let o : Int := (i *i ws)
    if (((0 ≤b o)&&(o ≤b 3)):Bool) then
      let r ← apply op (elem_at xa ws o l) (elem_at xb ws o l) (elem_at xm ws o l)
      x ← fillFactoryM ws o r x
  pure x

theorem actual_row_compact_source (op : vxm_lane_op) (xa xb xm : planes) (ws parts : Nat) (l : Int) :
    rowFactory op xa xb xm ws parts l=rowCompact op xa xb xm ws parts l := by
  simp only [rowFactory,rowCompact,fillFactoryM,sourceByteStep,pure_bind,bind_pure,bind_assoc]
  rfl

def sourceGroup (ws parts : Nat) (R : Nat → BitVec 32) : BitVec 32 :=
  indexedFold (fun i x => let o:=i*(ws:Int)
    if (((0 ≤b o)&&(o ≤b 3)):Bool) then sourceFill ws o (R i.toNat) x else x) 0 parts 0

theorem pure_if {α : Type} (c : Bool) (a b : α) :
    (if c then pure a else pure b : SailM α)=pure (if c then a else b) := by cases c <;> rfl

theorem yield_if {α : Type} (c : Bool) (a b : α) :
    (if c then ForInStep.yield a else ForInStep.yield b)=ForInStep.yield (if c then a else b) := by cases c <;> rfl

theorem actual_row_pure (op : vxm_lane_op) (xa xb xm : planes) (ws parts : Nat) (l : Int)
    (R : Nat → BitVec 32)
    (hR : ∀ i : Int, apply op (elem_at xa ws (i*(ws:Int)).toNat l.toNat)
      (elem_at xb ws (i*(ws:Int)).toNat l.toNat) (elem_at xm ws (i*(ws:Int)).toNat l.toNat)=pure (R i.toNat)) :
    rowFactory op xa xb xm ws parts l=pure (sourceGroup ws parts R) := by
  rw [actual_row_compact_source]
  unfold rowCompact
  simp only [hR,pure_bind,actual_fill_pure,bind_pure]
  simp only [pure_if,yield_if]
  change IntRange.forIn' (m:=SailM) {start:=0,stop:=(parts:Int)-1,step:=1} 0
    (fun i _ x => pure (.yield (if (((0 ≤b i*(ws:Int))&&(i*(ws:Int) ≤b 3)):Bool)
      then sourceFill ws (i*(ws:Int)) (R i.toNat) x else x))) = _
  simpa [sourceGroup] using ReduceSemanticProofs.monadic_indexed_range_fold
    (m:=SailM) {start:=0,stop:=(parts:Int)-1,step:=1}
    (fun i x => if (((0 ≤b i*(ws:Int))&&(i*(ws:Int) ≤b 3)):Bool)
      then sourceFill ws (i*(ws:Int)) (R i.toNat) x else x) (0:BitVec 32) rfl

theorem actual_group_word (ws : Nat) (hw : ws=1 ∨ ws=2 ∨ ws=4) (R : Nat → BitVec 32) :
    sourceGroup ws (actualParts true ws) R=groupWord ws R := by
  rcases hw with h|h|h <;> subst ws <;>
    simp [sourceGroup,sourceFill,sourceByteStep,groupWord,indexedFold,actualParts,Out.Functions.not,
      writeByte,getByte,Sail.BitVec.updateSubrange,Sail.BitVec.extractLsb,Sail.BitVec.slice,BitVec.extractLsb]

/-- Complete actual grouped row: the arithmetic leaf contract is the already
    established total/pure scalar result, not an assumed placement property. -/
theorem actual_grouped_row (op : vxm_lane_op) (xa xb xm : planes) (ws : Nat) (l : Int)
    (hw : ws=1 ∨ ws=2 ∨ ws=4) (R : Nat → BitVec 32)
    (hR : ∀ i : Int, apply op (elem_at xa ws (i*(ws:Int)).toNat l.toNat)
      (elem_at xb ws (i*(ws:Int)).toNat l.toNat) (elem_at xm ws (i*(ws:Int)).toNat l.toNat)=pure (R i.toNat)) :
    rowFactory op xa xb xm ws (actualParts true ws) l=
      pure ((((groupByte ws R 3).append (groupByte ws R 2)).append
        (groupByte ws R 1)).append (groupByte ws R 0)) := by
  rw [actual_row_pure op xa xb xm ws (actualParts true ws) l R hR,
    actual_group_word ws hw R,group_word_concatenation ws hw R]

/-- The ungrouped branch discards the temporary grouped word, then recomputes
    its scalar leaf. Both pure computations leave architectural state intact. -/
theorem discarded_pure_row (x y : BitVec 32) :
    (do let _ ← (pure x : SailM (BitVec 32)); pure y)=pure y := by simp

def planeByte (p : planes) (k l : Nat) : BitVec 8 :=
  Sail.BitVec.slice (GetElem?.getElem! p k) (8*l) 8

def inputWord (p : planes) (ws o l : Nat) : BitVec 32 :=
  if ws=1 then writeByte 0 (planeByte p o l) 0
  else if ws=2 then writeByte (writeByte 0 (planeByte p o l) 0) (planeByte p (o+1) l) 1
  else packFour (planeByte p o l) (planeByte p (o+1) l) (planeByte p (o+2) l) (planeByte p (o+3) l)

def inputStep (p : planes) (o l : Nat) (b : Int) (x : BitVec 32) : BitVec 32 :=
  let k : Int := (o +i b)
  if ((k ≤b 3):Bool) then Sail.BitVec.updateSubrange x ((8 *i b) +i 7) (8 *i b)
    (Sail.BitVec.slice (GetElem?.getElem! p k) (8 *i l) 8) else x

theorem actual_input_fold (p : planes) (ws o l : Nat) :
    elem_at p ws o l=indexedFold (inputStep p o l) 0 ws 0 := by
  unfold elem_at
  change IntRange.forIn' (m:=Id) {start:=0,stop:=(ws:Int)-1,step:=1} 0
    (fun b _ x => pure (.yield (inputStep p o l b x))) = _
  simpa using actual_indexed_range_fold {start:=0,stop:=(ws:Int)-1,step:=1} (inputStep p o l) (0:BitVec 32) rfl

theorem actual_group_input (p : planes) (ws o l : Nat)
    (hw : ws=1 ∨ ws=2 ∨ ws=4) (ho : o+ws<=4) :
    elem_at p ws o l=inputWord p ws o l := by
  have hcast : (8*(l:Int)).toNat=8*l := by omega
  have hcases : o=0 ∨ o=1 ∨ o=2 ∨ o=3 := by omega
  rcases hw with h|h|h <;> subst ws <;>
    rcases hcases with h|h|h|h <;> subst o <;> try omega
  all_goals
    rw [actual_input_fold]
    simp [indexedFold,inputStep,inputWord,packFour,writeByte,planeByte,Sail.BitVec.updateSubrange,hcast] <;> rfl

def plainInputStep (p : planes) (l : Nat) (b : Int) (x : BitVec 32) : BitVec 32 :=
  Sail.BitVec.updateSubrange x ((8 *i b) +i 7) (8 *i b)
    (Sail.BitVec.slice (GetElem?.getElem! p b) (8 *i l) 8)

theorem actual_plain_input_fold (p : planes) (ws l : Nat) :
    elem p ws l=indexedFold (plainInputStep p l) 0 ws 0 := by
  unfold elem
  change IntRange.forIn' (m:=Id) {start:=0,stop:=(ws:Int)-1,step:=1} 0
    (fun b _ x => pure (.yield (plainInputStep p l b x))) = _
  simpa using actual_indexed_range_fold {start:=0,stop:=(ws:Int)-1,step:=1} (plainInputStep p l) (0:BitVec 32) rfl

theorem actual_ungrouped_input (p : planes) (ws l : Nat)
    (hw : ws=1 ∨ ws=2 ∨ ws=4) :
    elem p ws l=inputWord p ws 0 l := by
  have hcast : (8*(l:Int)).toNat=8*l := by omega
  rw [actual_plain_input_fold]
  rcases hw with h|h|h <;> subst ws <;>
    simp [indexedFold,plainInputStep,inputWord,packFour,writeByte,planeByte,Sail.BitVec.updateSubrange,hcast] <;> rfl

end LanePlacementProofs
