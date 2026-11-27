import Out
open Sail Sail.ConcurrencyInterfaceV1
set_option autoImplicit false
namespace Proofs.MxmRowClassifier
abbrev Row := Nat × Int × Int
def changedRows : List Row → List Nat
 | [] => []
 | (idx,actual,expected)::rest =>
   if actual=expected then changedRows rest else idx::changedRows rest
def scan : List Row → Nat → Nat → Nat × Nat
 | [],count,loc => (count,loc)
 | (idx,actual,expected)::rest,count,loc =>
   if actual=expected then scan rest count loc else scan rest (count+1) idx
def verdict (state : Nat × Nat) : Nat × Nat :=
 if state.1=0 then (0,0) else if state.1=1 then (1,state.2) else (2,0)

def iterateAt {α : Type} (f : Int → α → α) : Nat → Int → α → α
 | 0,_,value => value
 | n+1,i,value => iterateAt f n (i+1) (f i value)

/-- Exact finite production range semantics, with an index-dependent pure
    body. No opaque whole-loop-success or row-classification premise. -/
theorem range_loop {α : Type} (f : Int → α → α) (range : IntRange)
 (one : range.step=1) (value : α) (i : Int) (hs : (i - range.start)%range.step=0)
 (start : range.start≤i) :
 IntRange.forIn'.loop (m:=Id) range (fun i _ v=>pure (.yield (f i v))) value i hs =
 iterateAt f (range.stop - i + 1).toNat i value := by
 induction value,i,hs using IntRange.forIn'.loop.induct range with
 | case1 value i hs hi ih =>
   have inside:=hi
   simp [Membership.mem,one] at inside
   rw [IntRange.forIn'.loop.eq_1]
   simp only [hi,dite_true,pure_bind]
   have n : (range.stop - i + 1).toNat=(range.stop - (i + range.step) + 1).toNat+1 := by omega
   rw [n]
   simpa only [iterateAt,one] using ih (f i value) (by omega)
 | case2 value i hs hi =>
   have outside:=hi
   simp [Membership.mem,one] at outside
   have n : (range.stop - i + 1).toNat=0 := by omega
   rw [IntRange.forIn'.loop.eq_1]
   simp [hi,n,iterateAt]
   rfl

theorem actual_range {α : Type} (f : Int → α → α) (nr : Nat) (value : α) :
 IntRange.forIn' (m:=Id) {start:=0,stop:=(nr:Int)-1,step:=1} value
   (fun i _ v=>pure (.yield (f i v))) = iterateAt f nr 0 value := by
 unfold IntRange.forIn'
 have h:=range_loop f {start:=0,stop:=(nr:Int)-1,step:=1} rfl value 0 (by simp) (by simp)
 simpa using h

def rows (actual expected : Int → Int) : Nat → Int → List Row
 | 0,_ => []
 | n+1,i => (i.toNat,actual i,expected i)::rows actual expected n (i+1)
def rowStep (actual expected : Int → Int) (i : Int) (state : Nat × Nat) : Nat × Nat :=
 if actual i != expected i then (i.toNat,state.2+1) else state

theorem exact_actual_row_fold (actual expected : Int → Int) (n : Nat) (i : Int) (count loc : Nat) :
 iterateAt (rowStep actual expected) n i (loc,count)=
 ((scan (rows actual expected n i) count loc).2,(scan (rows actual expected n i) count loc).1) := by
 induction n generalizing i count loc with
 | zero => rfl
 | succ n ih =>
   by_cases same : actual i=expected i
   · simpa [iterateAt,rows,scan,rowStep,same] using ih (i+1) count loc
   · simpa [iterateAt,rows,scan,rowStep,same] using ih (i+1) (count+1) i.toNat

/-- Exact source final indexed loop and verdict, abstracting only its two
    already-computed row arrays by their actual read values. -/
def finalRows (actual expected : Int → Int) (nr : Nat) : Nat × Nat := Id.run do
 let (loc,wrong) ← IntRange.forIn' (m:=Id) {start:=0,stop:=(nr:Int)-1,step:=1}
   (0,0) (fun i _ state => pure (.yield (rowStep actual expected i state)))
 if wrong == 0 then return (0,0)
 else if wrong == 1 then return (1,loc)
 else return (2,0)

theorem actual_final_rows (actual expected : Int → Int) (nr : Nat) :
 finalRows actual expected nr=verdict (scan (rows actual expected nr 0) 0 0) := by
 unfold finalRows
 rw [actual_range,exact_actual_row_fold]
 simp only [Id.run,bind,pure]
 generalize scan (rows actual expected nr 0) 0 0 = s
 rcases s with ⟨count,loc⟩
 simp only [verdict,beq_iff_eq]
 rfl

theorem exact_changed_count (xs : List Row) (count loc : Nat) :
 (scan xs count loc).1=count+(changedRows xs).length := by
 induction xs generalizing count loc with
 | nil => simp [scan,changedRows]
 | cons row rest ih =>
   rcases row with ⟨idx,actual,expected⟩
   by_cases same : actual=expected
   · simp [scan,changedRows,same,ih]
   · simp [scan,changedRows,same,ih,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem zero_changed (xs : List Row) (count loc : Nat) (none : changedRows xs=[]) :
 scan xs count loc=(count,loc) := by
 induction xs generalizing count loc with
 | nil => rfl
 | cons row rest ih =>
   rcases row with ⟨idx,actual,expected⟩
   by_cases same : actual=expected
   · simp only [changedRows,same,if_true] at none
     simpa [scan,same] using ih count loc none
   · simp [changedRows,same] at none

theorem one_changed (xs : List Row) (count loc idx : Nat)
 (one : changedRows xs=[idx]) : scan xs count loc=(count+1,idx) := by
 induction xs generalizing count loc with
 | nil => simp [changedRows] at one
 | cons row rest ih =>
   rcases row with ⟨r,actual,expected⟩
   by_cases same : actual=expected
   · simp only [changedRows,same,if_true] at one
     simpa [scan,same] using ih count loc one
   · have parts : r=idx ∧ changedRows rest=[] := by simpa [changedRows,same] using one
     simp only [scan,same,if_false]
     rw [zero_changed rest (count+1) r parts.2,parts.1]

/-- All arbitrary finite error vectors with no final changed row are clean. -/
theorem classify_zero (xs : List Row) (none : changedRows xs=[]) :
 verdict (scan xs 0 0)=(0,0) := by rw [zero_changed xs 0 0 none];rfl

/-- Exactly one final nonzero row residual identifies that precise row. -/
theorem classify_one (xs : List Row) (idx : Nat) (one : changedRows xs=[idx]) :
 verdict (scan xs 0 0)=(1,idx) := by rw [one_changed xs 0 0 idx one];rfl

/-- Every multirow pattern is detected, regardless of aggregate cancellation,
    number of injected flips, row values, numeric format or 96-bit projection. -/
theorem classify_every_multirow (xs : List Row) (multi : 2≤(changedRows xs).length) :
 verdict (scan xs 0 0)=(2,0) := by
 have count : (scan xs 0 0).1=(changedRows xs).length := by simpa using exact_changed_count xs 0 0
 unfold verdict
 rw [count,if_neg (by omega),if_neg (by omega)]
end Proofs.MxmRowClassifier
