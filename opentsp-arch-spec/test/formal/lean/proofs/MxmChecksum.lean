import Out
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace Proofs.MxmChecksum
def sum : Nat → (Nat → Int) → Int
 | 0,_ => 0
 | n+1,f => sum n f+f n

theorem sum_congr (n : Nat) (f g : Nat → Int) (same : ∀ i,i<n → f i=g i) :
 sum n f=sum n g := by
 induction n with
 | zero => rfl
 | succ n ih =>
   simp only [sum]
   rw [ih (fun i hi=>same i (by omega)),same n (by omega)]

theorem sum_zero (n : Nat) : sum n (fun _=>0)=0 := by
 induction n <;> simp_all [sum]
theorem sum_add (n : Nat) (f g : Nat → Int) :
 sum n (fun i=>f i+g i)=sum n f+sum n g := by
 induction n with
 | zero => rfl
 | succ n ih => simp [sum,ih,Int.add_assoc,Int.add_comm,Int.add_left_comm]

/-- Every finite error pattern has exactly the linear subset syndrome below;
    no restriction to one or two erroneous rows is hidden in this identity. -/
theorem error_syndrome (rows : Nat) (values errors : Nat → Int)
 (selected : Nat → Bool) :
 sum rows (fun r => if selected r then values r + errors r else 0) -
 sum rows (fun r => if selected r then values r else 0) =
 sum rows (fun r => if selected r then errors r else 0) := by
 have splitEq :
  sum rows (fun r => if selected r then values r + errors r else 0) =
  sum rows (fun r => if selected r then values r else 0) +
  sum rows (fun r => if selected r then errors r else 0) := by
   rw [← sum_add]
   apply sum_congr
   intro r hr
   cases selected r <;> simp
 rw [splitEq]
 omega
theorem sum_mul (n : Nat) (f : Nat → Int) (x : Int) :
 sum n (fun i=>f i*x)=sum n f*x := by
 induction n with
 | zero => simp [sum]
 | succ n ih => simp [sum,ih,Int.add_mul]
theorem sum_transpose (rows cols : Nat) (f : Nat → Nat → Int) :
 sum rows (fun r=>sum cols (f r))=sum cols (fun k=>sum rows (fun r=>f r k)) := by
 induction rows with
 | zero => simp [sum,sum_zero]
 | succ rows ih => simp only [sum];rw [sum_add,ih]

/-- Exact checksum identity for any finite matrix, any input and any row subset.
    None-valued nonfinite elements are represented by zero on both paths. -/
theorem clean_checksum (rows cols : Nat) (weight : Nat → Nat → Int)
 (input : Nat → Int) (selected : Nat → Bool) :
 sum rows (fun r=>if selected r then sum cols (fun k=>weight r k * input k) else 0)=
 sum cols (fun k=>sum rows (fun r=>if selected r then weight r k else 0) * input k) := by
 calc
  _=sum rows (fun r=>sum cols (fun k=>(if selected r then weight r k else 0) * input k)) := by
    apply sum_congr
    intro r hr
    cases selected r <;> simp [sum_zero]
  _=sum cols (fun k=>sum rows (fun r=>(if selected r then weight r k else 0) * input k)) :=
    sum_transpose rows cols _
  _=_ := by
    apply sum_congr
    intro k hk
    exact sum_mul rows _ (input k)

theorem spike_sum (n idx : Nat) (delta : Int) (inside : idx<n) :
 sum n (fun r=>if r=idx then delta else 0)=delta := by
 induction n with
 | zero => omega
 | succ n ih =>
   by_cases last : idx=n
   · subst idx
     have earlier : sum n (fun r=>if r=n then delta else 0)=0 := by
       rw [sum_congr n _ (fun _=>0) (by intro i hi;simp [show i≠n from by omega])]
       exact sum_zero n
     simp [sum,earlier]
   · have low : idx<n := by omega
     simp [sum,Ne.symm last,ih low]

/-- Singleton-row specialization: the independent column-major recomputation
    agrees with the nominal row dot for every installed row and finite matrix. -/
theorem independent_nominal_row (rows cols idx : Nat) (inside : idx<rows)
 (weight : Nat → Nat → Int) (input : Nat → Int) :
 sum cols (fun k=>sum rows (fun r=>if r=idx then weight r k else 0) * input k)=
 sum cols (fun k=>weight idx k * input k) := by
 apply sum_congr
 intro k hk
 have same : (fun r=>if r=idx then weight r k else 0)=(fun r=>if r=idx then weight idx k else 0) := by
  funext r
  split <;> simp_all
 rw [same,spike_sum rows idx (weight idx k) inside]

/-- A single changed row changes precisely the checksums whose subset contains it. -/
theorem single_row_delta (rows idx : Nat) (inside : idx<rows) (values : Nat → Int)
 (delta : Int) (selected : Nat → Bool) :
 sum rows (fun r=>if selected r then values r+(if r=idx then delta else 0) else 0) -
 sum rows (fun r=>if selected r then values r else 0) =
 if selected idx then delta else 0 := by
 have splitEq :
   sum rows (fun r=>if selected r then values r+(if r=idx then delta else 0) else 0)=
   sum rows (fun r=>if selected r then values r else 0)+
   sum rows (fun r=>if r=idx then (if selected idx then delta else 0) else 0) := by
   rw [←sum_add]
   apply sum_congr
   intro r hr
   by_cases target : r=idx
   · subst r;cases selected idx <;> simp
   · cases selected r <;> simp [target]
 rw [splitEq,spike_sum rows idx _ inside]
 omega

def nsum : Nat → (Nat → Nat) → Nat
 | 0,_ => 0
 | n+1,f => nsum n f+f n
def rowCode (r : Nat) : Nat :=
 nsum 9 (fun i=>if bit_of r i then (2 ^ i : Nat) else 0)

theorem complete_row_code : ∀ r : Fin 512,rowCode r.val=r.val := by decide +kernel

theorem bounded_row_code (r : Nat) (inside : r<512) : rowCode r=r := complete_row_code ⟨r,inside⟩

/-- Distinct installed rows have a binary subset that separates their errors. -/
theorem distinct_rows_separate (a b : Nat) (ha : a<512) (hb : b<512) (different : a≠b) :
 ∃ i,i<9 ∧ bit_of a i≠bit_of b i := by
 classical
 apply Classical.byContradiction
 intro absent
 have same : ∀ i,i<9 → bit_of a i=bit_of b i := by
   intro i hi
   apply Classical.byContradiction
   intro ne
   exact absent ⟨i,hi,ne⟩
 have aux : ∀ n,n≤9 →
   nsum n (fun i=>if bit_of a i then (2 ^ i : Nat) else 0)=
   nsum n (fun i=>if bit_of b i then (2 ^ i : Nat) else 0) := by
   intro n
   induction n with
   | zero => simp [nsum]
   | succ n ih =>
     intro hn
     simp only [nsum]
     rw [ih (by omega),same n (by omega)]
 have codes : rowCode a=rowCode b := aux 9 (by omega)
 exact different ((bounded_row_code a ha).symm.trans (codes.trans (bounded_row_code b hb)))

/-- A separating subset makes two nonzero row errors fail the single-row rule,
    whether the total error cancels or not. -/
theorem two_row_detection (da db : Int) (ha : da≠0) (hb : db≠0) :
 (da+db=0 → da≠0) ∧
 (da+db≠0 → da≠0 ∧ da≠da+db) := by
 constructor
 · intro h;exact ha
 · intro h;constructor
   · exact ha
   · omega
end Proofs.MxmChecksum
