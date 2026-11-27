-- Actual indexed exact-dot folds, independent product class semantics and sum.
import Proofs.FpSigns
open Sail Out Out.Functions Out.fclass
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace DotSemanticProofs
open DotBoundProofs

inductive Product where
  | invalid | negativeInfinity | positiveInfinity | negativeZero | positiveZero
  | finite (value : Int)

def product (a b : fval) (u : Int) : Product :=
  let neg := a.neg != b.neg
  match a.cls,b.cls with
  | FNaN,_ | _,FNaN => .invalid
  | FInf,FZero | FZero,FInf => .invalid
  | FInf,_ | _,FInf => if neg then .negativeInfinity else .positiveInfinity
  | FZero,_ | _,FZero => if neg then .negativeZero else .positiveZero
  | FFinite,FFinite => .finite (NumericalProofs.signedMag neg (a.mag*b.mag*pow2i (a.e+b.e+u)))

def applyProduct (term : Product) (state : DotState) : DotState :=
  let (nz,nan,ninf,pinf,sum) := state
  match term with
  | .invalid => (nz,true,ninf,pinf,sum)
  | .negativeInfinity => (false,nan,true,pinf,sum)
  | .positiveInfinity => (false,nan,ninf,true,sum)
  | .negativeZero => (nz,nan,ninf,pinf,sum)
  | .positiveZero => (false,nan,ninf,pinf,sum)
  | .finite value => (false,nan,ninf,pinf,sum+value)

theorem complete_product_class_step (a b : fval) (u : Int) (state : DotState) :
    stepValues a b u state=applyProduct (product a b u) state := by
  rcases state with ⟨nz,nan,ninf,pinf,sum⟩
  cases ha : a.cls <;> cases hb : b.cls <;> cases hn : a.neg <;> cases hm : b.neg <;>
    simp [stepValues,product,applyProduct,ha,hb,hn,hm,neq_bool,NumericalProofs.signedMag,Int.sub_eq_add_neg,Out.Functions.not]

def indexedFold {α : Type} (f : Int → α → α) (i : Int) : Nat → α → α
  | 0,state => state
  | n+1,state => indexedFold f (i+1) n (f i state)

def indexedSum (term : Int → Int) (i : Int) : Nat → Int
  | 0 => 0
  | n+1 => term i+ indexedSum term (i+1) n

/-- Actual inclusive indexed Sail loop, with no omitted endpoint or surrogate
    index. The independent fold is ordinary finite mathematical iteration. -/
theorem actual_indexed_loop_fold {α : Type} (range : IntRange) (f : Int → α → α)
    (state : α) (i : Int) (hs : (i-range.start)%range.step=0)
    (hstep : range.step=1) (hstart : range.start ≤ i) :
    IntRange.forIn'.loop (m := Id) range (fun k _ s => pure (.yield (f k s))) state i hs=
      indexedFold f i (range.stop - i+1).toNat state := by
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
    rfl

theorem actual_indexed_range_fold {α : Type} (range : IntRange) (f : Int → α → α)
    (state : α) (hstep : range.step=1) :
    IntRange.forIn' (m := Id) range state (fun k _ s => pure (.yield (f k s)))=
      indexedFold f range.start (range.stop-range.start+1).toNat state := by
  unfold IntRange.forIn'
  exact actual_indexed_loop_fold range f state range.start (by simp) hstep (by omega)

theorem indexed_fold_sum {α : Type} (f : Int → α → α) (value : α → Int) (term : Int → Int)
    (hf : ∀ k state, value (f k state)=value state+term k) (i : Int) (n : Nat) (state : α) :
    value (indexedFold f i n state)=value state+ indexedSum term i n := by
  induction n generalizing i state with
  | zero => simp [indexedFold,indexedSum]
  | succ n ih =>
    simp only [indexedFold,indexedSum]
    rw [ih,hf]
    omega

def pairSemanticStep (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (k : Int) (state : DotState) : DotState :=
  applyProduct (product (pairA f wlo whi k) (pairB f xlo xhi k) u) state

theorem pair_step_complete (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (k : Int) (state : DotState) :
    pairStep f u wlo whi xlo xhi k state=pairSemanticStep f u wlo whi xlo xhi k state :=
  complete_product_class_step _ _ _ _

theorem pair_state_complete (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (upper : Int) :
    pairState f u wlo whi xlo xhi upper=
      indexedFold (pairSemanticStep f u wlo whi xlo xhi) 0 (upper+1).toNat (true,false,false,false,0) := by
  change IntRange.forIn' (m := Id) {start:=0,stop:=upper,step:=1} (true,false,false,false,0)
    (fun k _ state => pure (.yield (pairStep f u wlo whi xlo xhi k state))) = _
  have hf : (pairStep f u wlo whi xlo xhi)=(pairSemanticStep f u wlo whi xlo xhi) := by
    funext k state; exact pair_step_complete f u wlo whi xlo xhi k state
  rw [hf]
  have h := actual_indexed_range_fold {start:=0,stop:=upper,step:=1}
    (pairSemanticStep f u wlo whi xlo xhi) (true,false,false,false,0) rfl
  simpa using h

def packDot (state : DotState) : dsum :=
  let (nz,nan,ninf,pinf,sum) := state
  {dnz:=nz,dnan:=nan,dninf:=ninf,dpinf:=pinf,dsum:=sum}

theorem pack_dot_sum (state : DotState) : (packDot state).dsum=state.2.2.2.2 := by
  rcases state with ⟨nz,nan,ninf,pinf,sum⟩; rfl

theorem pair_factory_pack (f : fmt) (u : Int) (wlo whi xlo xhi : vec) (upper : Int) :
    pairFactory f u wlo whi xlo xhi upper=packDot (pairState f u wlo whi xlo xhi upper) := by
  rw [pair_factory_decomposition]; rfl

theorem pair_exact_complete (f : fmt) (u : Int) (wlo whi xlo xhi : vec) :
    pair_exact f u wlo whi xlo xhi=packDot
      (indexedFold (pairSemanticStep f u wlo whi xlo xhi) 0 Out.Functions.vlanes.toNat
        (true,false,false,false,0)) := by
  rw [pair_exact_factory_source,pair_factory_pack,pair_state_complete]
  have hn : Out.Functions.vlanes-1+1=Out.Functions.vlanes := by omega
  rw [hn]

theorem pair_exact_sum (f : fmt) (u : Int) (wlo whi xlo xhi : vec) :
    (pair_exact f u wlo whi xlo xhi).dsum=
      indexedSum (fun k => exactTerm (pairA f wlo whi k) (pairB f xlo xhi k) u) 0 Out.Functions.vlanes.toNat := by
  rw [pair_exact_complete,pack_dot_sum]
  have h := indexed_fold_sum (pairSemanticStep f u wlo whi xlo xhi) (fun s : DotState => s.2.2.2.2)
    (fun k => exactTerm (pairA f wlo whi k) (pairB f xlo xhi k) u)
    (fun k state => by rw [←pair_step_complete]; exact step_sum_exact _ _ _ _) 0 Out.Functions.vlanes.toNat
    (true,false,false,false,0)
  simpa using h

def fp8SemanticStep (f : fmt) (wr xv : vec) (k : Int) (state : DotState) : DotState :=
  applyProduct (product (fp8A f wr k) (fp8B f xv k) 48) state

theorem fp8_step_complete (f : fmt) (wr xv : vec) (k : Int) (state : DotState) :
    fp8Step f wr xv k state=fp8SemanticStep f wr xv k state := complete_product_class_step _ _ _ _

theorem fp8_state_complete (f : fmt) (wr xv : vec) (upper : Int) :
    fp8State f wr xv upper=indexedFold (fp8SemanticStep f wr xv) 0 (upper+1).toNat (true,false,false,false,0) := by
  change IntRange.forIn' (m := Id) {start:=0,stop:=upper,step:=1} (true,false,false,false,0)
    (fun k _ state => pure (.yield (fp8Step f wr xv k state))) = _
  have hf : fp8Step f wr xv=fp8SemanticStep f wr xv := by funext k state; exact fp8_step_complete f wr xv k state
  rw [hf]
  simpa using actual_indexed_range_fold {start:=0,stop:=upper,step:=1}
    (fp8SemanticStep f wr xv) (true,false,false,false,0) rfl

theorem fp8_factory_pack (f : fmt) (wr xv : vec) (upper : Int) :
    fp8Factory f wr xv upper=packDot (fp8State f wr xv upper) := by
  rw [fp8_factory_decomposition]; rfl

theorem fp8_exact_complete (f : fmt) (wr xv : vec) :
    fp8_exact f wr xv=packDot (indexedFold (fp8SemanticStep f wr xv) 0 Out.Functions.vlanes.toNat
      (true,false,false,false,0)) := by
  rw [fp8_exact_factory_source,fp8_factory_pack,fp8_state_complete]
  rw [show Out.Functions.vlanes-1+1=Out.Functions.vlanes by omega]

theorem fp8_exact_sum (f : fmt) (wr xv : vec) :
    (fp8_exact f wr xv).dsum=indexedSum (fun k => exactTerm (fp8A f wr k) (fp8B f xv k) 48) 0 Out.Functions.vlanes.toNat := by
  rw [fp8_exact_complete,pack_dot_sum]
  have h := indexed_fold_sum (fp8SemanticStep f wr xv) (fun s : DotState => s.2.2.2.2)
    (fun k => exactTerm (fp8A f wr k) (fp8B f xv k) 48)
    (fun k state => by rw [←fp8_step_complete]; exact step_sum_exact _ _ _ _) 0 Out.Functions.vlanes.toNat
    (true,false,false,false,0)
  simpa using h

def applyBfpProduct (term : Product) (state : BfpState) : BfpState :=
  let (nz,nan,sum) := state
  match term with
  | .invalid => (nz,true,sum)
  | .negativeZero => (nz,nan,sum)
  | .positiveZero => (false,nan,sum)
  | .finite value => (false,nan,sum+value)
  | .negativeInfinity | .positiveInfinity => state

theorem bfp_product_class_step (a b : fval) (left right : Int) (state : BfpState)
    (haI : a.cls≠FInf) (hbI : b.cls≠FInf) :
    bfpStepValues a b left right state=applyBfpProduct (product a b (left+right+48)) state := by
  rcases state with ⟨nz,nan,sum⟩
  have he : a.e+b.e+left+right+48=a.e+b.e+(left+right+48) := by omega
  cases ha : a.cls <;> cases hb : b.cls <;> cases hn : a.neg <;> cases hm : b.neg <;>
    simp_all [bfpStepValues,product,applyBfpProduct,neq_bool,NumericalProofs.signedMag,Int.sub_eq_add_neg,Out.Functions.not,he]

def bfpSemanticStep (wd ws xd xs : vec) (k : Int) (state : BfpState) : BfpState :=
  applyBfpProduct (product (fp8A FMT_E4M3 wd k) (fp8B FMT_E4M3 xd k)
    (blk_exp ws k+blk_exp xs k+48)) state

theorem bfp_step_complete (wd ws xd xs : vec) (k : Int) (state : BfpState) :
    bfpStep wd ws xd xs k state=bfpSemanticStep wd ws xd xs k state := by
  exact bfp_product_class_step _ _ _ _ _ (e4_decode_no_inf _) (e4_decode_no_inf _)

theorem bfp_state_complete (wd ws xd xs : vec) (upper : Int) :
    bfpState wd ws xd xs upper=indexedFold (bfpSemanticStep wd ws xd xs) 0 (upper+1).toNat (true,false,0) := by
  change IntRange.forIn' (m := Id) {start:=0,stop:=upper,step:=1} (true,false,0)
    (fun k _ state => pure (.yield (bfpStep wd ws xd xs k state))) = _
  have hf : bfpStep wd ws xd xs=bfpSemanticStep wd ws xd xs := by funext k state; exact bfp_step_complete wd ws xd xs k state
  rw [hf]
  simpa using actual_indexed_range_fold {start:=0,stop:=upper,step:=1}
    (bfpSemanticStep wd ws xd xs) (true,false,0) rfl

def packBfp (state : BfpState) : dsum :=
  let (nz,nan,sum) := state
  {dnz:=nz,dnan:=nan,dninf:=false,dpinf:=false,dsum:=sum}

theorem pack_bfp_sum (state : BfpState) : (packBfp state).dsum=state.2.2 := by
  rcases state with ⟨nz,nan,sum⟩; rfl

theorem bfp_factory_pack (wd ws xd xs : vec) (upper : Int) :
    bfpFactory wd ws xd xs upper=packBfp (bfpState wd ws xd xs upper) := by
  rw [bfp_factory_decomposition]; rfl

theorem bfp_exact_complete (wd ws xd xs : vec) :
    bfp8_exact wd ws xd xs=packBfp (indexedFold (bfpSemanticStep wd ws xd xs) 0 Out.Functions.vlanes.toNat
      (true,false,0)) := by
  rw [bfp_exact_factory_source,bfp_factory_pack,bfp_state_complete]
  rw [show Out.Functions.vlanes-1+1=Out.Functions.vlanes by omega]

theorem bfp_exact_sum (wd ws xd xs : vec) :
    (bfp8_exact wd ws xd xs).dsum=indexedSum (fun k => exactTerm (fp8A FMT_E4M3 wd k) (fp8B FMT_E4M3 xd k)
      (blk_exp ws k+blk_exp xs k+48)) 0 Out.Functions.vlanes.toNat := by
  rw [bfp_exact_complete,pack_bfp_sum]
  have h := indexed_fold_sum (bfpSemanticStep wd ws xd xs) (fun s : BfpState => s.2.2)
    (fun k => exactTerm (fp8A FMT_E4M3 wd k) (fp8B FMT_E4M3 xd k) (blk_exp ws k+blk_exp xs k+48))
    (fun k state => by
      rw [←bfp_step_complete]
      exact bfp_step_sum_exact _ _ _ _ _ (e4_decode_no_inf _) (e4_decode_no_inf _))
    0 Out.Functions.vlanes.toNat (true,false,0)
  simpa using h

def int8Term (wr : vec) (ws : Bool) (xv : vec) (xs : Bool) (k : Int) : Int :=
  (if ws then (lane8 wr k).toInt else BitVec.toNatInt (lane8 wr k))*
    (if xs then (lane8 xv k).toInt else BitVec.toNatInt (lane8 xv k))

def int8State (wr : vec) (ws : Bool) (xv : vec) (xs : Bool) (upper : Int) : Int := Id.run do
  let mut sum : Int := 0
  for k in [0:upper:1]i do
    sum := sum+ int8Term wr ws xv xs k
  return sum

def int8Factory (wr : vec) (ws : Bool) (xv : vec) (xs : Bool) (upper : Int) : BitVec 32 := Id.run do
  let mut sum : Int := 0
  for k in [0:upper:1]i do
    let a : Int := if ws then (lane8 wr k).toInt else BitVec.toNatInt (lane8 wr k)
    let b : Int := if xs then (lane8 xv k).toInt else BitVec.toNatInt (lane8 xv k)
    sum := sum+a*b
  return bits_of_s (wrap_s sum 32) 32

theorem int8_dot_factory_source (wr : vec) (ws : Bool) (xv : vec) (xs : Bool) :
    int8_dot wr ws xv xs=int8Factory wr ws xv xs (Out.Functions.vlanes-1) := by rfl

theorem int8_factory_state (wr : vec) (ws : Bool) (xv : vec) (xs : Bool) (upper : Int) :
    int8Factory wr ws xv xs upper=bits_of_s (wrap_s (int8State wr ws xv xs upper) 32) 32 := by rfl

theorem int8_state_exact_sum (wr : vec) (ws : Bool) (xv : vec) (xs : Bool) (upper : Int) :
    int8State wr ws xv xs upper=indexedSum (int8Term wr ws xv xs) 0 (upper+1).toNat := by
  change IntRange.forIn' (m := Id) {start:=0,stop:=upper,step:=1} 0
    (fun k _ sum => pure (.yield (sum+ int8Term wr ws xv xs k))) = _
  rw [actual_indexed_range_fold _ _ _ rfl]
  have h := indexed_fold_sum (fun k sum => sum+ int8Term wr ws xv xs k) (fun s : Int => s)
    (int8Term wr ws xv xs) (fun _ _ => rfl) 0 (upper+1).toNat 0
  simpa using h

theorem int8_dot_complete (wr : vec) (ws : Bool) (xv : vec) (xs : Bool) :
    int8_dot wr ws xv xs=bits_of_s (wrap_s (indexedSum (int8Term wr ws xv xs) 0 Out.Functions.vlanes.toNat) 32) 32 := by
  rw [int8_dot_factory_source,int8_factory_state,int8_state_exact_sum]
  rw [show Out.Functions.vlanes-1+1=Out.Functions.vlanes by omega]

def dotResultSpec (d : dsum) (u : Int) : BitVec 32 :=
  match d.dnan,d.dpinf,d.dninf with
  | true,_,_ | false,true,true => canon_nan FMT32
  | false,true,false => inf_bits FMT32 false
  | false,false,true => inf_bits FMT32 true
  | false,false,false => if d.dsum=0 then zero_bits FMT32 d.dnz
    else NumericalProofs.roundSigned FMT32 d.dsum (-u)

theorem complete_dot_final_class_table (d : dsum) (u : Int) :
    dsum_round_u d u=dotResultSpec d u := by
  cases hn : d.dnan <;> cases hp : d.dpinf <;> cases hm : d.dninf <;>
    simp [dsum_round_u,dotResultSpec,hn,hp,hm,NumericalProofs.roundSigned]
  all_goals by_cases hz : d.dsum=0 <;> simp [hz]

def WideClass (w : wlane) : Prop := w.wsp=0 ∨ w.wsp=1 ∨ w.wsp=2 ∨ w.wsp=3

theorem wide_zero_class : WideClass wlane_zero := by unfold WideClass; decide +kernel

theorem dot_wide_class (d : dsum) : WideClass (dsum_wide d) := by
  simp [dsum_wide,WideClass]
  repeat first | split | simp_all

def wideAddSpec (a b : wlane) : wlane :=
  match a.wsp,b.wsp with
  | 3,_ | _,3 | 1,2 | 2,1 => {wv:=0,wsp:=3,wnz:=false}
  | 0,0 => {wv:=wrap_s (a.wv+b.wv) 128,wsp:=0,wnz:=a.wnz && b.wnz}
  | 1,_ | 2,_ => a
  | _,_ => b

theorem complete_wide_class_add (a b : wlane) (ha : WideClass a) (hb : WideClass b) :
    wlane_add a b=wideAddSpec a b := by
  rcases ha with ha|ha|ha|ha <;> rcases hb with hb|hb|hb|hb <;>
    simp [wlane_add,wideAddSpec,ha,hb]

theorem wide_add_class_preserved (a b : wlane) (ha : WideClass a) (hb : WideClass b) :
    WideClass (wlane_add a b) := by
  rw [complete_wide_class_add a b ha hb]
  rcases ha with ha|ha|ha|ha <;> rcases hb with hb|hb|hb|hb <;>
    simp [wideAddSpec,WideClass,ha,hb]

def wideReadSpec (w : wlane) : BitVec 32 :=
  match w.wsp with
  | 3 => canon_nan FMT32
  | 1 => inf_bits FMT32 false
  | 2 => inf_bits FMT32 true
  | _ => if w.wv=0 then zero_bits FMT32 w.wnz else NumericalProofs.roundSigned FMT32 w.wv (-48)

theorem complete_wide_read (w : wlane) (hw : WideClass w) : wlane_read w=wideReadSpec w := by
  rcases hw with hw|hw|hw|hw <;> simp [wlane_read,wideReadSpec,hw,NumericalProofs.roundSigned]
  all_goals by_cases hz : w.wv=0 <;> simp [hz]

end DotSemanticProofs
