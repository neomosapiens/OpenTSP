-- Actual Reduce reads, ordered rounded fold, zero output initialization/write.
import Proofs.DotSemantics
import Out.InstsEnd
open Sail Out Out.Functions Out.ExecutionResult
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace ReduceSemanticProofs
open DotSemanticProofs

theorem monadic_indexed_loop_fold {m : Type → Type} [Monad m] [LawfulMonad m]
    {α : Type} (range : IntRange) (f : Int → α → α) (state : α) (i : Int)
    (hs : (i-range.start)%range.step=0) (hstep : range.step=1) (hstart : range.start ≤ i) :
    IntRange.forIn'.loop (m := m) range (fun k _ s => pure (.yield (f k s))) state i hs=
      pure (indexedFold f i (range.stop - i+1).toNat state) := by
  induction state,i,hs using IntRange.forIn'.loop.induct range with
  | case1 state i hs hi ih =>
    have hmem := hi
    simp [Membership.mem,hstep] at hmem
    rw [IntRange.forIn'.loop.eq_1]
    simp only [hi,dite_true,pure_bind]
    have hn : (range.stop - i+1).toNat=(range.stop - (i+range.step)+1).toNat+1 := by omega
    rw [hn]
    simp only [indexedFold]
    have hh := ih (f i state) (by omega)
    simpa only [hstep] using hh
  | case2 state i hs hi =>
    have hmem := hi
    simp [Membership.mem,hstep] at hmem
    have hn : (range.stop - i+1).toNat=0 := by omega
    rw [IntRange.forIn'.loop.eq_1]
    simp [hi,hn,indexedFold]

theorem monadic_indexed_range_fold {m : Type → Type} [Monad m] [LawfulMonad m]
    {α : Type} (range : IntRange) (f : Int → α → α) (state : α) (hstep : range.step=1) :
    IntRange.forIn' (m := m) range state (fun k _ s => pure (.yield (f k s)))=
      pure (indexedFold f range.start (range.stop-range.start+1).toNat state) := by
  unfold IntRange.forIn'
  exact monadic_indexed_loop_fold range f state range.start (by simp) hstep (by omega)

def reduceGroupM (kop : Int) (t : eltype) (p : planes) (w : Nat) (g upper : Int) : SailM (BitVec 32) := do
  let mut acc : BitVec 32 := elem p w (16 *i g)
  for j in [1:upper:1]i do
    acc := k_binary kop t acc (elem p w ((16 *i g) +i j))
  return acc

def reduceGroup (kop : Int) (t : eltype) (p : planes) (w : Nat) (g upper : Int) : BitVec 32 :=
  indexedFold (fun j acc => k_binary kop t acc (elem p w ((16 *i g) +i j))) 1 upper.toNat (elem p w (16 *i g))

theorem reduce_group_pure (kop : Int) (t : eltype) (p : planes) (w : Nat) (g upper : Int) :
    reduceGroupM kop t p w g upper=pure (reduceGroup kop t p w g upper) := by
  unfold reduceGroupM
  simp only [bind_pure]
  change IntRange.forIn' (m := SailM) {start:=1,stop:=upper,step:=1} (elem p w (16 *i g))
    (fun j _ acc => pure (.yield (k_binary kop t acc (elem p w ((16 *i g) +i j))))) = _
  have h := monadic_indexed_range_fold (m:=SailM) {start:=1,stop:=upper,step:=1}
    (fun j acc => k_binary kop t acc (elem p w ((16 *i g) +i j))) (elem p w (16 *i g)) rfl
  simpa [reduceGroup] using h

def reduceRowsM (kop : Int) (t : eltype) (p : planes) (w : Nat) (gUpper jUpper : Int) : SailM (Vector (BitVec 32) Out.Functions.vlanes) := do
  let mut out : Vector (BitVec 32) Out.Functions.vlanes := vectorInit 0x00000000#32
  for g in [0:gUpper:1]i do
    let acc ← reduceGroupM kop t p w g jUpper
    out := vectorUpdate out (16 *i g) acc
  return out

def reduceRows (kop : Int) (t : eltype) (p : planes) (w : Nat) (gUpper jUpper : Int) : Vector (BitVec 32) Out.Functions.vlanes :=
  indexedFold (fun g out => vectorUpdate out (16 *i g) (reduceGroup kop t p w g jUpper))
    0 (gUpper+1).toNat (vectorInit 0x00000000#32)

theorem reduce_rows_pure (kop : Int) (t : eltype) (p : planes) (w : Nat) (gUpper jUpper : Int) :
    reduceRowsM kop t p w gUpper jUpper=pure (reduceRows kop t p w gUpper jUpper) := by
  unfold reduceRowsM
  simp only [reduce_group_pure,pure_bind,bind_pure]
  change IntRange.forIn' (m := SailM) {start:=0,stop:=gUpper,step:=1} (vectorInit 0x00000000#32)
    (fun g _ out => pure (.yield (vectorUpdate out (16 *i g) (reduceGroup kop t p w g jUpper)))) = _
  simpa [reduceRows] using monadic_indexed_range_fold (m:=SailM) {start:=0,stop:=gUpper,step:=1}
    (fun g out => vectorUpdate out (16 *i g) (reduceGroup kop t p w g jUpper)) (vectorInit 0x00000000#32) rfl

def reduceFactory (op : reduce_op) (t : eltype) (d : streamref) (a : vsrc) (gUpper jUpper : Int) : SailM ExecutionResult := do
  let w := eltype_width t
  let kop : Int := match op with | .RED_ADD => 0 | .RED_MAX => 9 | .RED_MIN => 10
  match ←vxm_read a w with
  | .Err e => pure (Diagnostic e)
  | .Ok p => do
    let out : Vector (BitVec 32) Out.Functions.vlanes := vectorInit 0x00000000#32
    let out ← ((do
      let mut loop_vars := out
      for g in [0:gUpper:1]i do
        let out := loop_vars
        loop_vars ← do
          let acc : BitVec 32 := elem p w (16 *i g)
          let acc ← ((do
            let mut loop_vars_1 := acc
            for j in [1:jUpper:1]i do
              let acc := loop_vars_1
              loop_vars_1 := k_binary kop t acc (elem p w ((16 *i g) +i j))
            pure loop_vars_1) : SailM (BitVec 32))
          pure (vectorUpdate out (16 *i g) acc)
      pure loop_vars) : SailM (Vector (BitVec 32) Out.Functions.vlanes))
    match ←write_planes d (words_to_planes out) w (←vxm_commit_cycle ()) with
    | .Ok () => pure (Issued ())
    | .Err e => pure (Diagnostic e)

theorem actual_reduce_factory_source (op : reduce_op) (t : eltype) (d : streamref) (a : vsrc) :
    execute_VXM_REDUCE op t d a=reduceFactory op t d a ((Out.Functions.superlanes:Int)-1) 15 := by rfl

theorem reduce_factory_decomposition (op : reduce_op) (t : eltype) (d : streamref) (a : vsrc) (gUpper jUpper : Int) :
    reduceFactory op t d a gUpper jUpper=(do
      let w := eltype_width t
      let kop : Int := match op with | .RED_ADD => 0 | .RED_MAX => 9 | .RED_MIN => 10
      match ←vxm_read a w with
      | .Err e => pure (Diagnostic e)
      | .Ok p => do
        let out ←reduceRowsM kop t p w gUpper jUpper
        match ←write_planes d (words_to_planes out) w (←vxm_commit_cycle ()) with
        | .Ok () => pure (Issued ())
        | .Err e => pure (Diagnostic e)) := by rfl

theorem actual_reduce_ordered_fold (op : reduce_op) (t : eltype) (d : streamref) (a : vsrc) :
    execute_VXM_REDUCE op t d a=(do
      let w := eltype_width t
      let kop : Int := match op with | .RED_ADD => 0 | .RED_MAX => 9 | .RED_MIN => 10
      match ←vxm_read a w with
      | .Err e => pure (Diagnostic e)
      | .Ok p => do
        let out := reduceRows kop t p w ((Out.Functions.superlanes:Int)-1) 15
        match ←write_planes d (words_to_planes out) w (←vxm_commit_cycle ()) with
        | .Ok () => pure (Issued ())
        | .Err e => pure (Diagnostic e)) := by
  rw [actual_reduce_factory_source,reduce_factory_decomposition]
  simp only [reduce_rows_pure,pure_bind]

theorem group_slot_domain (g : Int) (hg : 0 ≤ g ∧ g < (Out.Functions.superlanes:Int)) :
    (16*g).toNat < Out.Functions.vlanes ∧ ((16*g).toNat%16=0) := by
  have hgeo : Out.Functions.vlanes=16*Out.Functions.superlanes := by rfl
  omega

theorem update_nonleader (out : Vector (BitVec 32) Out.Functions.vlanes) (g : Int) (value : BitVec 32)
    (hg : 0 ≤ g ∧ g < (Out.Functions.superlanes:Int)) (l : Nat) (hl : l < Out.Functions.vlanes)
    (hn : l%16≠0) : (vectorUpdate out (16*g).toNat value)[l]=out[l] := by
  have hs := group_slot_domain g hg
  have he : (16*g).toNat≠l := by omega
  simp [vectorUpdate,Vector.set!,hl,hs.1,he]

theorem reduce_fold_nonleader (kop : Int) (t : eltype) (p : planes) (w : Nat) (jUpper g : Int)
    (n : Nat) (out : Vector (BitVec 32) Out.Functions.vlanes) (l : Nat)
    (hg : 0 ≤ g ∧ g+(n:Int) ≤ (Out.Functions.superlanes:Int))
    (hl : l < Out.Functions.vlanes) (hn : l%16≠0) :
    (indexedFold (fun g out => vectorUpdate out (16 *i g) (reduceGroup kop t p w g jUpper)) g n out)[l]=out[l] := by
  induction n generalizing g out with
  | zero => rfl
  | succ n ih =>
    simp only [indexedFold]
    rw [ih (g+1) _ (by omega)]
    exact update_nonleader out g _ (by omega) l hl hn

theorem actual_reduce_nonleaders_zero (kop : Int) (t : eltype) (p : planes) (w : Nat)
    (l : Nat) (hl : l < Out.Functions.vlanes) (hn : l%16≠0) :
    (reduceRows kop t p w ((Out.Functions.superlanes:Int)-1) 15)[l]=0 := by
  unfold reduceRows
  rw [reduce_fold_nonleader kop t p w 15 0 _ _ l (by omega) hl hn]
  simp [vectorInit]

end ReduceSemanticProofs
