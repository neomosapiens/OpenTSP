import Proofs.MxmRowClassifier
import Proofs.SrfCommittedHistory
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace Proofs.MxmNominalRows

def iter {α : Type} (f : Nat → α → α) : Nat → Nat → α → α
 | 0,_,v => v
 | n+1,i,v => iter f n (i+1) (f i v)

theorem range_nat {α : Type} (f : Nat → α → α) (n i : Nat) (v : α) :
 Proofs.MxmRowClassifier.iterateAt (fun j => f j.toNat) n (i:Int) v = iter f n i v := by
 induction n generalizing i v with
 | zero => rfl
 | succ n ih =>
   simp only [Proofs.MxmRowClassifier.iterateAt,iter,Int.toNat_natCast]
   simpa only [Int.natCast_add] using ih (i+1) (f i v)

theorem actual_nat_range {α : Type} (f : Nat → α → α) (n : Nat) (v : α) :
 IntRange.forIn' (m:=Id) {start:=0,stop:=(n:Int)-1,step:=1} v
  (fun j _ v=>pure (.yield (f j.toNat v))) = iter f n 0 v := by
 rw [Proofs.MxmRowClassifier.actual_range]
 exact range_nat f n 0 v

def product (weight input : Option Int) : Int :=
 match weight,input with | some w,some x => w*x | _,_ => 0

def step {N : Nat} (weight : Nat → Nat → Option Int) (input : Nat → Option Int)
 (r k : Nat) (v : Vector Int N) : Vector Int N :=
 match weight r k,input k with
 | some a,some b => vectorUpdate v r (v[r]!+a*b)
 | _,_ => v

def expectedStep {N : Nat} (weight : Nat → Nat → Option Int) (input : Nat → Option Int)
 (r k : Nat) (v : Vector Int N) : Vector Int N :=
 match input k with
 | none => v
 | some b => match weight r k with
   | none => v
   | some a => vectorUpdate v r (v[r]!+a*b)

theorem actual_bodies_equal {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (r k : Nat) (v : Vector Int N) : step w x r k v=expectedStep w x r k v := by
 cases a:w r k <;> cases b:x k <;> simp [step,expectedStep,a,b]

theorem update_get {N : Nat} (v : Vector Int N) (r j : Nat) (value : Int)
 (hr : r<N) (hj : j<N) : (vectorUpdate v j value)[r]! = if r=j then value else v[r]! :=
 SrfCommittedHistoryProofs.vector_updated_get v r j value hr hj

theorem step_get {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (r k j : Nat) (v : Vector Int N) (hr : r<N) (hj : j<N) :
 (step w x r k v)[j]! = v[j]!+(if j=r then product (w r k) (x k) else 0) := by
 cases a:w r k with
 | none => simp [step,a,product]
 | some wa =>
   cases b:x k with
   | none => simp [step,b,product]
   | some xb =>
     simp only [step,a,b,product]
     rw [update_get v j r _ hj hr]
     split <;> simp_all

/-- One column pass changes exactly each visited row, once. -/
theorem column_cell {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (n start k j : Nat) (v : Vector Int N) (bounded : start+n≤N) (hj : j<N) :
 (iter (fun r=>step w x r k) n start v)[j]! =
 v[j]!+(if start≤j ∧ j<start+n then product (w j k) (x k) else 0) := by
 induction n generalizing start v with
 | zero =>
   simp only [iter,Nat.add_zero]
   rw [if_neg (by omega)]
   simp
 | succ n ih =>
   rw [iter,ih (start+1) _ (by omega),step_get w x start k j v (by omega) hj]
   by_cases same:j=start
   · subst j
     have later : ¬(start+1≤start ∧ start<start+1+n) := by omega
     simp [later]
   · by_cases inside:start≤j ∧ j<start+(n+1)
     · have later : start+1≤j ∧ j<start+1+n := by omega
       simp [same,inside,later]
     · have later : ¬(start+1≤j ∧ j<start+1+n) := by omega
       simp [same,inside,later]

def dotFrom (w : Nat → Nat → Option Int) (x : Nat → Option Int) (row n start : Nat) (initial : Int) : Int :=
 iter (fun k value=>value+product (w row k) (x k)) n start initial

/-- Every actual primary inner loop has exactly the dot-product effect on its
    own row and leaves every other physical row cell unchanged. -/
theorem row_cell {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (n start row j : Nat) (v : Vector Int N) (hr : row<N) (hj : j<N) :
 (iter (fun k=>step w x row k) n start v)[j]! =
 if j=row then dotFrom w x row n start v[j]! else v[j]! := by
 induction n generalizing start v with
 | zero => simp [iter,dotFrom]
 | succ n ih =>
   rw [iter,ih (start+1) (step w x row start v),step_get w x row start j v hr hj]
   by_cases same:j=row
   · subst j;simp [dotFrom,iter]
   · simp [same]

/-- An indexed pass whose actual cell transition changes its selected cell only
    applies that transition once to each visited cell. -/
theorem rows_once {N : Nat} (body : Nat → Vector Int N → Vector Int N)
 (transform : Nat → Int → Int)
 (one : ∀ row j v, row<N → j<N →
  (body row v)[j]! = if j=row then transform row v[j]! else v[j]!)
 (n start j : Nat) (v : Vector Int N) (bounded : start+n≤N) (hj : j<N) :
 (iter body n start v)[j]! =
 if start≤j ∧ j<start+n then transform j v[j]! else v[j]! := by
 induction n generalizing start v with
 | zero =>
   simp only [iter,Nat.add_zero]
   rw [if_neg (by omega)]
 | succ n ih =>
   rw [iter,ih (start+1) (body start v) (by omega),one start j v (by omega) hj]
   by_cases same:j=start
   · subst j
     have later : ¬(start+1≤start ∧ start<start+1+n) := by omega
     simp [later]
   · by_cases inside:start≤j ∧ j<start+(n+1)
     · have later : start+1≤j ∧ j<start+1+n := by omega
       simp [same,inside,later]
     · have later : ¬(start+1≤j ∧ j<start+1+n) := by omega
       simp [same,inside,later]

def primary {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (rows cols : Nat) (v : Vector Int N) : Vector Int N :=
 iter (fun r=>iter (fun k=>step w x r k) cols 0) rows 0 v

def expected {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (rows cols : Nat) (v : Vector Int N) : Vector Int N :=
 iter (fun k=>iter (fun r=>step w x r k) rows 0) cols 0 v

theorem primary_cell {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (rows cols j : Nat) (v : Vector Int N) (bounded : rows≤N) (inside : j<rows) :
 (primary w x rows cols v)[j]! = dotFrom w x j cols 0 v[j]! := by
 unfold primary
 rw [rows_once _ (fun r value=>dotFrom w x r cols 0 value)
  (by intro row j v hr hj;exact row_cell w x cols 0 row j v hr hj)
  rows 0 j v (by omega) (by omega)]
 simp [inside]

theorem expected_cell_from {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (rows cols start j : Nat) (v : Vector Int N) (bounded : rows≤N) (inside : j<rows) :
 (iter (fun k=>iter (fun r=>step w x r k) rows 0) cols start v)[j]! =
 dotFrom w x j cols start v[j]! := by
 induction cols generalizing start v with
 | zero => rfl
 | succ n ih =>
   rw [iter,ih (start+1)]
   rw [column_cell w x rows 0 start j v (by omega) (by omega)]
   simp only [Nat.zero_le,Nat.zero_add,inside,and_self,ite_true]
   rfl

/-- Source-connected nominal premise: for arbitrary Option-valued weights and
    captured inputs, both actual indexed update orders produce the same value
    at EVERY installed row, including None elements and partial row counts.
    It assumes no nominal equality or clean checksum conclusion. -/
theorem complete_nominal_per_cell {N : Nat} (w : Nat → Nat → Option Int) (x : Nat → Option Int)
 (rows cols j : Nat) (v : Vector Int N) (bounded : rows≤N) (inside : j<rows) :
 (primary w x rows cols v)[j]! = (expected w x rows cols v)[j]! := by
 rw [primary_cell w x rows cols j v bounded inside]
 exact (expected_cell_from w x rows cols 0 j v bounded inside).symm

end Proofs.MxmNominalRows
