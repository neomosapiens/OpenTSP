-- Complete source-linked exact-dot loop and magnitude transfer.
import Proofs.CallerBounds
open Sail Out Out.Functions Out.fclass
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
set_option exponentiation.threshold 100000
namespace DotBoundProofs
abbrev DotState := Bool × Bool × Bool × Bool × Int

def stepValues (a b : fval) (u : Int) (state : DotState) : DotState :=
  let (all_negzero,nan,ninf,pinf,sum) := state
  let neg := (neq_bool a.neg b.neg)
  let (all_negzero, nan, ninf, pinf, sum) : (Bool × Bool × Bool × Bool × Int) :=
    if (((a.cls == FNaN) || (b.cls == FNaN)) : Bool)
    then
      (let nan : Bool := true
      (all_negzero, nan, ninf, pinf, sum))
    else
      (let (all_negzero, nan, ninf, pinf, sum) : (Bool × Bool × Bool × Bool × Int) :=
        if ((((a.cls == FInf) && (b.cls == FZero)) || ((a.cls == FZero) && (b.cls == FInf))) : Bool)
        then
          (let nan : Bool := true
          (all_negzero, nan, ninf, pinf, sum))
        else
          (let (all_negzero, ninf, pinf, sum) : (Bool × Bool × Bool × Int) :=
            if (((a.cls == FInf) || (b.cls == FInf)) : Bool)
            then
              (let (ninf, pinf) : (Bool × Bool) :=
                if (neg : Bool)
                then
                  (let ninf : Bool := true
                  (ninf, pinf))
                else
                  (let pinf : Bool := true
                  (ninf, pinf))
              let all_negzero : Bool := false
              (all_negzero, ninf, pinf, sum))
            else
              (let (all_negzero, sum) : (Bool × Int) :=
                if (((a.cls == FZero) || (b.cls == FZero)) : Bool)
                then
                  (let all_negzero : Bool :=
                    if ((Out.Functions.not neg) : Bool)
                    then false
                    else all_negzero
                  (all_negzero, sum))
                else
                  (let all_negzero : Bool := false
                  let m := ((a.mag *i b.mag) *i (pow2i ((a.e +i b.e) +i u)))
                  let sum : Int :=
                    if (neg : Bool)
                    then (sum -i m)
                    else (sum +i m)
                  (all_negzero, sum))
              (all_negzero, ninf, pinf, sum))
          (all_negzero, nan, ninf, pinf, sum))
      (all_negzero, nan, ninf, pinf, sum))
  (all_negzero, nan, ninf, pinf, sum)

def pairStep (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (k : Int) (state : DotState) : DotState :=
  let a := (fdecode f (0x0000#16 +++ ((lane8 whi k) +++ (lane8 wlo k))))
  let b := (fdecode f (0x0000#16 +++ ((lane8 xhi k) +++ (lane8 xlo k))))
  stepValues a b u state

def pairState (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (upper : Int) : DotState := Id.run do
  let mut state : DotState := (true,false,false,false,0)
  for k in [0:upper:1]i do
    state := pairStep f u wlo whi xlo xhi k state
  return state

def pairFactory (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (upper : Int) : dsum := Id.run do
  let nan : Bool := false
  let pinf : Bool := false
  let ninf : Bool := false
  let all_negzero : Bool := true
  let sum : Int := 0
  let (all_negzero, nan, ninf, pinf, sum) ← (( do
    let loop_k_lower := 0
    let loop_k_upper := upper
    let mut loop_vars := (all_negzero, nan, ninf, pinf, sum)
    for k in [loop_k_lower:loop_k_upper:1]i do
      let (all_negzero, nan, ninf, pinf, sum) := loop_vars
      loop_vars :=
        let a := (fdecode f (0x0000#16 +++ ((lane8 whi k) +++ (lane8 wlo k))))
        let b := (fdecode f (0x0000#16 +++ ((lane8 xhi k) +++ (lane8 xlo k))))
        let neg := (neq_bool a.neg b.neg)
        let (all_negzero, nan, ninf, pinf, sum) : (Bool × Bool × Bool × Bool × Int) :=
          if (((a.cls == FNaN) || (b.cls == FNaN)) : Bool)
          then
            (let nan : Bool := true
            (all_negzero, nan, ninf, pinf, sum))
          else
            (let (all_negzero, nan, ninf, pinf, sum) : (Bool × Bool × Bool × Bool × Int) :=
              if ((((a.cls == FInf) && (b.cls == FZero)) || ((a.cls == FZero) && (b.cls == FInf))) : Bool)
              then
                (let nan : Bool := true
                (all_negzero, nan, ninf, pinf, sum))
              else
                (let (all_negzero, ninf, pinf, sum) : (Bool × Bool × Bool × Int) :=
                  if (((a.cls == FInf) || (b.cls == FInf)) : Bool)
                  then
                    (let (ninf, pinf) : (Bool × Bool) :=
                      if (neg : Bool)
                      then
                        (let ninf : Bool := true
                        (ninf, pinf))
                      else
                        (let pinf : Bool := true
                        (ninf, pinf))
                    let all_negzero : Bool := false
                    (all_negzero, ninf, pinf, sum))
                  else
                    (let (all_negzero, sum) : (Bool × Int) :=
                      if (((a.cls == FZero) || (b.cls == FZero)) : Bool)
                      then
                        (let all_negzero : Bool :=
                          if ((Out.Functions.not neg) : Bool)
                          then false
                          else all_negzero
                        (all_negzero, sum))
                      else
                        (let all_negzero : Bool := false
                        let m := ((a.mag *i b.mag) *i (pow2i ((a.e +i b.e) +i u)))
                        let sum : Int :=
                          if (neg : Bool)
                          then (sum -i m)
                          else (sum +i m)
                        (all_negzero, sum))
                    (all_negzero, ninf, pinf, sum))
                (all_negzero, nan, ninf, pinf, sum))
            (all_negzero, nan, ninf, pinf, sum))
        (all_negzero, nan, ninf, pinf, sum)
    (pure loop_vars) ) : Id (Bool × Bool × Bool × Bool × Int) )
  (pure { dnan := nan
          dpinf := pinf
          dninf := ninf
          dnz := all_negzero
          dsum := sum })

theorem pair_exact_factory_source (f : fmt) (u : Int) (wlo whi xlo xhi : vec) :
    pair_exact f u wlo whi xlo xhi=pairFactory f u wlo whi xlo xhi (Out.Functions.vlanes-1) := by rfl

theorem pair_factory_decomposition (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (upper : Int) :
    pairFactory f u wlo whi xlo xhi upper=
      let (dnz,dnan,dninf,dpinf,sum) := pairState f u wlo whi xlo xhi upper
      {dnz:=dnz,dnan:=dnan,dninf:=dninf,dpinf:=dpinf,dsum:=sum} := by rfl

def exactTerm (a b : fval) (u : Int) : Int :=
  if (((a.cls == fclass.FFinite) && (b.cls == fclass.FFinite)) : Bool) then
    let m := a.mag*b.mag*pow2i (a.e+b.e+u)
    if neq_bool a.neg b.neg then -m else m
  else 0

theorem step_sum_exact (a b : fval) (u : Int) (state : DotState) :
    (stepValues a b u state).2.2.2.2=state.2.2.2.2+exactTerm a b u := by
  rcases state with ⟨dnz,dnan,dninf,dpinf,sum⟩
  cases ha : a.cls <;> cases hb : b.cls <;> cases hn : neq_bool a.neg b.neg <;>
    simp [stepValues,exactTerm,ha,hb,hn,Int.sub_eq_add_neg]


def FiniteBounds (a : fval) : Prop :=
  a.cls=FFinite → 0 ≤ a.mag ∧ a.mag < pow2i 24 ∧ a.e ≤ 120

theorem exact_term_bound (a b : fval) (u : Int) (hu : u ≤ 266)
    (hA : FiniteBounds a) (hB : FiniteBounds b) :
    (exactTerm a b u).natAbs ≤ 2^568 := by
  cases ha : a.cls <;> cases hb : b.cls <;> try (solve | simp [exactTerm,ha,hb])
  have hA := hA ha
  have hB := hB hb
  have h := CallerBoundProofs.decoded_product_bound a.mag b.mag a.e b.e u 0
    ⟨hA.1,hA.2.1⟩ ⟨hB.1,hB.2.1⟩ hA.2.2 hB.2.2 hu (by decide)
  simp only [Int.add_zero] at h
  have hp : pow2i 568=((2^568:Nat):Int) := by
    rw [NumericalProofs.pow2i_nonnegative 568 (by decide)]
    simp [Int.natCast_pow]
  have habs : (a.mag*b.mag*pow2i (a.e+b.e+u)).natAbs ≤ 2^568 := by
    have he : ((a.mag*b.mag*pow2i (a.e+b.e+u)).natAbs:Int)=a.mag*b.mag*pow2i (a.e+b.e+u) := by omega
    apply Int.ofNat_le.mp
    rw [he,←hp]
    exact h.2
  cases hn : neq_bool a.neg b.neg <;> simpa [exactTerm,ha,hb,hn,Int.natAbs_neg] using habs

theorem step_sum_growth (a b : fval) (u : Int) (state : DotState)
    (hu : u ≤ 266) (hA : FiniteBounds a) (hB : FiniteBounds b) :
    (stepValues a b u state).2.2.2.2.natAbs ≤ state.2.2.2.2.natAbs+2^568 := by
  rw [step_sum_exact]
  exact Nat.le_trans (Int.natAbs_add_le _ _) (Nat.add_le_add_left (exact_term_bound a b u hu hA hB) _)

/-- Indexed actual IntRange loop: every concrete lane is visited once, and no
    lane-count sampling or constant-index surrogate replaces the program. -/
theorem indexed_loop_bound {α : Type} (range : IntRange) (f : Int → α → α)
    (value : α → Int) (B : Nat)
    (hF : ∀ k state, (value (f k state)).natAbs ≤ (value state).natAbs+B)
    (state : α) (i : Int) (hs : (i - range.start)%range.step=0)
    (hstep : range.step=1) (hstart : range.start ≤ i) :
    (value (IntRange.forIn'.loop (m := Id) range
      (fun k _ x => pure (.yield (f k x))) state i hs)).natAbs ≤
      (value state).natAbs+(range.stop - i + 1).toNat*B := by
  induction state, i, hs using IntRange.forIn'.loop.induct range with
  | case1 state i hs hi ih =>
    have hmem := hi
    simp [Membership.mem,hstep] at hmem
    rw [IntRange.forIn'.loop.eq_1]
    simp only [hi,dite_true,pure_bind]
    have hh := ih (f i state) (by omega)
    have hg := hF i state
    have hn : (range.stop - i + 1).toNat=(range.stop - (i + range.step) + 1).toNat+1 := by omega
    rw [hn,Nat.add_mul]
    simp only [Nat.one_mul]
    omega
  | case2 state i hs hi =>
    have hmem := hi
    simp [Membership.mem,hstep] at hmem
    have hn : (range.stop - i + 1).toNat=0 := by omega
    rw [IntRange.forIn'.loop.eq_1]
    simp [hi,hn]
    exact Nat.le_refl _

theorem indexed_range_bound {α : Type} (range : IntRange) (f : Int → α → α)
    (value : α → Int) (B : Nat)
    (hF : ∀ k state, (value (f k state)).natAbs ≤ (value state).natAbs+B)
    (state : α) (hstep : range.step=1) :
    (value (IntRange.forIn' (m := Id) range state
      (fun k _ x => pure (.yield (f k x))))).natAbs ≤
      (value state).natAbs+(range.stop - range.start + 1).toNat*B := by
  unfold IntRange.forIn'
  exact indexed_loop_bound range f value B hF state range.start (by simp) hstep (by omega)


def pairA (f : fmt) (wlo whi : vec) (k : Int) : fval :=
  fdecode f (0x0000#16 +++ ((lane8 whi k) +++ (lane8 wlo k)))
def pairB (f : fmt) (xlo xhi : vec) (k : Int) : fval :=
  fdecode f (0x0000#16 +++ ((lane8 xhi k) +++ (lane8 xlo k)))

theorem pair_step_source (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (k : Int) (state : DotState) :
    pairStep f u wlo whi xlo xhi k state=
      stepValues (pairA f wlo whi k) (pairB f xlo xhi k) u state := by rfl

theorem pair_state_sum_bound (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (upper : Int)
    (hu : u ≤ 266)
    (hA : ∀ k, FiniteBounds (pairA f wlo whi k))
    (hB : ∀ k, FiniteBounds (pairB f xlo xhi k)) :
    (pairState f u wlo whi xlo xhi upper).2.2.2.2.natAbs ≤ (upper+1).toNat*2^568 := by
  have hF : ∀ k state, (pairStep f u wlo whi xlo xhi k state).2.2.2.2.natAbs ≤
      state.2.2.2.2.natAbs+2^568 := by
    intro k state
    rw [pair_step_source]
    exact step_sum_growth _ _ _ _ hu (hA k) (hB k)
  have h := indexed_range_bound {start:=0,stop:=upper,step:=1}
    (pairStep f u wlo whi xlo xhi) (fun s : DotState => s.2.2.2.2) (2^568) hF
    (true,false,false,false,0) rfl
  change (IntRange.forIn' (m := Id) {start:=0,stop:=upper,step:=1}
    (true,false,false,false,0)
    (fun k _ state => pure (.yield (pairStep f u wlo whi xlo xhi k state)))).2.2.2.2.natAbs ≤ _
  simpa using h

theorem pair_exact_magnitude_bound (f : fmt) (u : Int) (wlo whi xlo xhi : vec)
    (hu : u ≤ 266)
    (hA : ∀ k, FiniteBounds (pairA f wlo whi k))
    (hB : ∀ k, FiniteBounds (pairB f xlo xhi k)) :
    (pair_exact f u wlo whi xlo xhi).dsum.natAbs < 2^600 := by
  have hs := pair_state_sum_bound f u wlo whi xlo xhi (Out.Functions.vlanes-1) hu hA hB
  have hn : (Out.Functions.vlanes-1+1).toNat ≤ 512 := by decide +kernel
  have hb := Nat.mul_le_mul_right (2^568) hn
  have h600 : 512*2^568 < (2:Nat)^600 := by decide +kernel
  rw [pair_exact_factory_source,pair_factory_decomposition]
  change (pairState f u wlo whi xlo xhi (Out.Functions.vlanes-1)).2.2.2.2.natAbs < 2^600
  omega


def fp8A (f : fmt) (wr : vec) (k : Int) : fval := fdecode f (0x000000#24 +++ (lane8 wr k))
def fp8B (f : fmt) (xv : vec) (k : Int) : fval := fdecode f (0x000000#24 +++ (lane8 xv k))
def fp8Step (f : fmt) (wr xv : vec) (k : Int) (state : DotState) : DotState :=
  stepValues (fp8A f wr k) (fp8B f xv k) 48 state

def fp8State (f : fmt) (wr xv : vec) (upper : Int) : DotState := Id.run do
  let mut state : DotState := (true,false,false,false,0)
  for k in [0:upper:1]i do
    state := fp8Step f wr xv k state
  return state

def fp8Factory (f : fmt) (wr xv : vec) (upper : Int) : dsum := Id.run do
  let nan : Bool := false
  let pinf : Bool := false
  let ninf : Bool := false
  let all_negzero : Bool := true
  let sum : Int := 0
  let (all_negzero, nan, ninf, pinf, sum) ← (( do
    let loop_k_lower := 0
    let loop_k_upper := upper
    let mut loop_vars := (all_negzero, nan, ninf, pinf, sum)
    for k in [loop_k_lower:loop_k_upper:1]i do
      let (all_negzero, nan, ninf, pinf, sum) := loop_vars
      loop_vars :=
        let a := (fdecode f (0x000000#24 +++ (lane8 wr k)))
        let b := (fdecode f (0x000000#24 +++ (lane8 xv k)))
        let neg := (neq_bool a.neg b.neg)
        let (all_negzero, nan, ninf, pinf, sum) : (Bool × Bool × Bool × Bool × Int) :=
          if (((a.cls == FNaN) || (b.cls == FNaN)) : Bool)
          then
            (let nan : Bool := true
            (all_negzero, nan, ninf, pinf, sum))
          else
            (let (all_negzero, nan, ninf, pinf, sum) : (Bool × Bool × Bool × Bool × Int) :=
              if ((((a.cls == FInf) && (b.cls == FZero)) || ((a.cls == FZero) && (b.cls == FInf))) : Bool)
              then
                (let nan : Bool := true
                (all_negzero, nan, ninf, pinf, sum))
              else
                (let (all_negzero, ninf, pinf, sum) : (Bool × Bool × Bool × Int) :=
                  if (((a.cls == FInf) || (b.cls == FInf)) : Bool)
                  then
                    (let (ninf, pinf) : (Bool × Bool) :=
                      if (neg : Bool)
                      then
                        (let ninf : Bool := true
                        (ninf, pinf))
                      else
                        (let pinf : Bool := true
                        (ninf, pinf))
                    let all_negzero : Bool := false
                    (all_negzero, ninf, pinf, sum))
                  else
                    (let (all_negzero, sum) : (Bool × Int) :=
                      if (((a.cls == FZero) || (b.cls == FZero)) : Bool)
                      then
                        (let all_negzero : Bool :=
                          if ((Out.Functions.not neg) : Bool)
                          then false
                          else all_negzero
                        (all_negzero, sum))
                      else
                        (let all_negzero : Bool := false
                        let m := ((a.mag *i b.mag) *i (pow2i ((a.e +i b.e) +i 48)))
                        let sum : Int :=
                          if (neg : Bool)
                          then (sum -i m)
                          else (sum +i m)
                        (all_negzero, sum))
                    (all_negzero, ninf, pinf, sum))
                (all_negzero, nan, ninf, pinf, sum))
            (all_negzero, nan, ninf, pinf, sum))
        (all_negzero, nan, ninf, pinf, sum)
    (pure loop_vars) ) : Id (Bool × Bool × Bool × Bool × Int) )
  (pure { dnan := nan
          dpinf := pinf
          dninf := ninf
          dnz := all_negzero
          dsum := sum })

theorem fp8_exact_factory_source (f : fmt) (wr xv : vec) :
    fp8_exact f wr xv=fp8Factory f wr xv (Out.Functions.vlanes-1) := by rfl

theorem fp8_factory_decomposition (f : fmt) (wr xv : vec) (upper : Int) :
    fp8Factory f wr xv upper=
      let (dnz,dnan,dninf,dpinf,sum) := fp8State f wr xv upper
      {dnz:=dnz,dnan:=dnan,dninf:=dninf,dpinf:=dpinf,dsum:=sum} := by rfl

theorem fp8_state_sum_bound (f : fmt) (wr xv : vec) (upper : Int)
    (hA : ∀ k, FiniteBounds (fp8A f wr k)) (hB : ∀ k, FiniteBounds (fp8B f xv k)) :
    (fp8State f wr xv upper).2.2.2.2.natAbs ≤ (upper+1).toNat*2^568 := by
  have hF : ∀ k state, (fp8Step f wr xv k state).2.2.2.2.natAbs ≤
      state.2.2.2.2.natAbs+2^568 := by
    intro k state
    exact step_sum_growth _ _ _ _ (by decide) (hA k) (hB k)
  have h := indexed_range_bound {start:=0,stop:=upper,step:=1}
    (fp8Step f wr xv) (fun s : DotState => s.2.2.2.2) (2^568) hF
    (true,false,false,false,0) rfl
  change (IntRange.forIn' (m := Id) {start:=0,stop:=upper,step:=1}
    (true,false,false,false,0)
    (fun k _ state => pure (.yield (fp8Step f wr xv k state)))).2.2.2.2.natAbs ≤ _
  simpa using h

theorem fp8_exact_magnitude_bound (f : fmt) (wr xv : vec)
    (hA : ∀ k, FiniteBounds (fp8A f wr k)) (hB : ∀ k, FiniteBounds (fp8B f xv k)) :
    (fp8_exact f wr xv).dsum.natAbs < 2^600 := by
  have hs := fp8_state_sum_bound f wr xv (Out.Functions.vlanes-1) hA hB
  have hn : (Out.Functions.vlanes-1+1).toNat ≤ 512 := by decide +kernel
  have hb := Nat.mul_le_mul_right (2^568) hn
  have h600 : 512*2^568 < (2:Nat)^600 := by decide +kernel
  rw [fp8_exact_factory_source,fp8_factory_decomposition]
  change (fp8State f wr xv (Out.Functions.vlanes-1)).2.2.2.2.natAbs < 2^600
  omega


abbrev BfpState := Bool × Bool × Int

def bfpStepValues (a b : fval) (left right : Int) (state : BfpState) : BfpState :=
  let (all_negzero,nan,sum) := state
  let neg := (neq_bool a.neg b.neg)
  let (all_negzero, nan, sum) : (Bool × Bool × Int) :=
    if (((a.cls == FNaN) || (b.cls == FNaN)) : Bool)
    then
      (let nan : Bool := true
      (all_negzero, nan, sum))
    else
      (let (all_negzero, sum) : (Bool × Int) :=
        if (((a.cls == FZero) || (b.cls == FZero)) : Bool)
        then
          (let all_negzero : Bool :=
            if ((Out.Functions.not neg) : Bool)
            then false
            else all_negzero
          (all_negzero, sum))
        else
          (let all_negzero : Bool := false
          let m :=
            ((a.mag *i b.mag) *i (pow2i
                ((((a.e +i b.e) +i left) +i right) +i 48)))
          let sum : Int :=
            if (neg : Bool)
            then (sum -i m)
            else (sum +i m)
          (all_negzero, sum))
      (all_negzero, nan, sum))
  (all_negzero, nan, sum)

def bfpStep (wd ws xd xs : vec) (k : Int) (state : BfpState) : BfpState :=
  bfpStepValues (fp8A FMT_E4M3 wd k) (fp8B FMT_E4M3 xd k)
    (blk_exp ws k) (blk_exp xs k) state

def bfpState (wd ws xd xs : vec) (upper : Int) : BfpState := Id.run do
  let mut state : BfpState := (true,false,0)
  for k in [0:upper:1]i do
    state := bfpStep wd ws xd xs k state
  return state

def bfpFactory (wd ws xd xs : vec) (upper : Int) : dsum := Id.run do
  let nan : Bool := false
  let all_negzero : Bool := true
  let sum : Int := 0
  let (all_negzero, nan, sum) ← (( do
    let loop_k_lower := 0
    let loop_k_upper := upper
    let mut loop_vars := (all_negzero, nan, sum)
    for k in [loop_k_lower:loop_k_upper:1]i do
      let (all_negzero, nan, sum) := loop_vars
      loop_vars :=
        let a := (fdecode FMT_E4M3 (0x000000#24 +++ (lane8 wd k)))
        let b := (fdecode FMT_E4M3 (0x000000#24 +++ (lane8 xd k)))
        let neg := (neq_bool a.neg b.neg)
        let (all_negzero, nan, sum) : (Bool × Bool × Int) :=
          if (((a.cls == FNaN) || (b.cls == FNaN)) : Bool)
          then
            (let nan : Bool := true
            (all_negzero, nan, sum))
          else
            (let (all_negzero, sum) : (Bool × Int) :=
              if (((a.cls == FZero) || (b.cls == FZero)) : Bool)
              then
                (let all_negzero : Bool :=
                  if ((Out.Functions.not neg) : Bool)
                  then false
                  else all_negzero
                (all_negzero, sum))
              else
                (let all_negzero : Bool := false
                let m :=
                  ((a.mag *i b.mag) *i (pow2i
                      ((((a.e +i b.e) +i (blk_exp ws k)) +i (blk_exp xs k)) +i 48)))
                let sum : Int :=
                  if (neg : Bool)
                  then (sum -i m)
                  else (sum +i m)
                (all_negzero, sum))
            (all_negzero, nan, sum))
        (all_negzero, nan, sum)
    (pure loop_vars) ) : Id (Bool × Bool × Int) )
  (pure { dnan := nan
          dpinf := false
          dninf := false
          dnz := all_negzero
          dsum := sum })

theorem bfp_exact_factory_source (wd ws xd xs : vec) :
    bfp8_exact wd ws xd xs=bfpFactory wd ws xd xs (Out.Functions.vlanes-1) := by rfl

theorem bfp_factory_decomposition (wd ws xd xs : vec) (upper : Int) :
    bfpFactory wd ws xd xs upper=
      let (dnz,dnan,sum) := bfpState wd ws xd xs upper
      {dnz:=dnz,dnan:=dnan,dninf:=false,dpinf:=false,dsum:=sum} := by rfl

theorem block_scale_range (v : vec) (k : Nat) : -8 ≤ blk_exp v k ∧ blk_exp v k ≤ 7 := by
  unfold blk_exp
  dsimp only
  split
  · let b : Int := 16*(tdiv_int k 16)
    let x : BitVec 4 := BitVec.slice v (8*b) 4
    have hn : 0 ≤ BitVec.toNatInt x ∧ BitVec.toNatInt x < 16 := by
      have h := x.isLt
      simp [BitVec.toNatInt]
      omega
    change -8 ≤ (if ((BitVec.toNatInt x ≥b 8):Bool) then BitVec.toNatInt x-16 else BitVec.toNatInt x) ∧
      (if ((BitVec.toNatInt x ≥b 8):Bool) then BitVec.toNatInt x-16 else BitVec.toNatInt x) ≤ 7
    by_cases hg : 8 ≤ BitVec.toNatInt x <;> simp [hg] <;> omega
  · decide

theorem e4_decode_no_inf (x : BitVec 32) : (fdecode FMT_E4M3 x).cls≠FInf := by
  simp [fdecode,FMT_E4M3,Out.Functions.not]
  split <;> simp_all <;> split <;> simp_all <;> split <;> simp_all

theorem bfp_step_sum_exact (a b : fval) (left right : Int) (state : BfpState)
    (haI : a.cls≠FInf) (hbI : b.cls≠FInf) :
    (bfpStepValues a b left right state).2.2=state.2.2+exactTerm a b (left+right+48) := by
  rcases state with ⟨dnz,dnan,sum⟩
  have he : a.e+b.e+left+right+48=a.e+b.e+(left+right+48) := by omega
  cases ha : a.cls <;> cases hb : b.cls <;> cases hn : neq_bool a.neg b.neg <;>
    simp_all [bfpStepValues,exactTerm,Int.sub_eq_add_neg,he]

theorem bfp_step_sum_growth (wd ws xd xs : vec) (k : Int) (state : BfpState)
    (hA : FiniteBounds (fp8A FMT_E4M3 wd k)) (hB : FiniteBounds (fp8B FMT_E4M3 xd k)) :
    (bfpStep wd ws xd xs k state).2.2.natAbs ≤ state.2.2.natAbs+2^568 := by
  have hl := block_scale_range ws k.toNat
  have hr := block_scale_range xs k.toNat
  have haI := e4_decode_no_inf (0x000000#24 +++ (lane8 wd k))
  have hbI := e4_decode_no_inf (0x000000#24 +++ (lane8 xd k))
  rw [bfpStep,bfp_step_sum_exact (fp8A FMT_E4M3 wd k) (fp8B FMT_E4M3 xd k)
    (blk_exp ws k) (blk_exp xs k) state haI hbI]
  have hterm := exact_term_bound (fp8A FMT_E4M3 wd k) (fp8B FMT_E4M3 xd k)
    (blk_exp ws k+blk_exp xs k+48) (by omega) hA hB
  exact Nat.le_trans (Int.natAbs_add_le _ _) (Nat.add_le_add_left hterm _)

theorem bfp_state_sum_bound (wd ws xd xs : vec) (upper : Int)
    (hA : ∀ k, FiniteBounds (fp8A FMT_E4M3 wd k))
    (hB : ∀ k, FiniteBounds (fp8B FMT_E4M3 xd k)) :
    (bfpState wd ws xd xs upper).2.2.natAbs ≤ (upper+1).toNat*2^568 := by
  have hF : ∀ k state, (bfpStep wd ws xd xs k state).2.2.natAbs ≤ state.2.2.natAbs+2^568 := by
    intro k state
    exact bfp_step_sum_growth wd ws xd xs k state (hA k) (hB k)
  have h := indexed_range_bound {start:=0,stop:=upper,step:=1}
    (bfpStep wd ws xd xs) (fun s : BfpState => s.2.2) (2^568) hF (true,false,0) rfl
  change (IntRange.forIn' (m := Id) {start:=0,stop:=upper,step:=1} (true,false,0)
    (fun k _ state => pure (.yield (bfpStep wd ws xd xs k state)))).2.2.natAbs ≤ _
  simpa using h

theorem bfp_exact_magnitude_bound (wd ws xd xs : vec)
    (hA : ∀ k, FiniteBounds (fp8A FMT_E4M3 wd k))
    (hB : ∀ k, FiniteBounds (fp8B FMT_E4M3 xd k)) :
    (bfp8_exact wd ws xd xs).dsum.natAbs < 2^600 := by
  have hs := bfp_state_sum_bound wd ws xd xs (Out.Functions.vlanes-1) hA hB
  have hn : (Out.Functions.vlanes-1+1).toNat ≤ 512 := by decide +kernel
  have hb := Nat.mul_le_mul_right (2^568) hn
  have h600 : 512*2^568 < (2:Nat)^600 := by decide +kernel
  rw [bfp_exact_factory_source,bfp_factory_decomposition]
  change (bfpState wd ws xd xs (Out.Functions.vlanes-1)).2.2.natAbs < 2^600
  omega


/-- This is exactly the already separately discharged native decoder contract,
    quantified over every 32-bit carrier rather than only one vector lane. -/
def DecodedBounds (f : fmt) : Prop := ∀ x : BitVec 32, FiniteBounds (fdecode f x)

theorem pair_from_decoder_bound (f : fmt) (u : Int) (wlo whi xlo xhi : vec)
    (hu : u ≤ 266) (h : DecodedBounds f) :
    (pair_exact f u wlo whi xlo xhi).dsum.natAbs < 2^600 := by
  apply pair_exact_magnitude_bound f u wlo whi xlo xhi hu
  · intro k; exact h _
  · intro k; exact h _

theorem fp8_from_decoder_bound (f : fmt) (wr xv : vec) (h : DecodedBounds f) :
    (fp8_exact f wr xv).dsum.natAbs < 2^600 := by
  apply fp8_exact_magnitude_bound f wr xv
  · intro k; exact h _
  · intro k; exact h _

theorem bfp_from_decoder_bound (wd ws xd xs : vec) (h : DecodedBounds FMT_E4M3) :
    (bfp8_exact wd ws xd xs).dsum.natAbs < 2^600 := by
  apply bfp_exact_magnitude_bound wd ws xd xs
  · intro k; exact h _
  · intro k; exact h _

theorem dot_wide_entry_bound (d : dsum) (h : d.dsum.natAbs < 2^600) :
    CallerBoundProofs.WideBound (dsum_wide d) := by
  have hp : pow2i 600=((2^600:Nat):Int) := by
    rw [NumericalProofs.pow2i_nonnegative 600 (by decide)]
    simp [Int.natCast_pow]
  have hc : (d.dsum.natAbs:Int) < ((2^600:Nat):Int) := Int.ofNat_lt.mpr h
  apply CallerBoundProofs.dsum_wide_bound
  rw [←hp] at hc
  omega

end DotBoundProofs
