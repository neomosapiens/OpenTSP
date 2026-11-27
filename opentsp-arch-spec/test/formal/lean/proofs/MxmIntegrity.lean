import Proofs.Common
import Proofs.MxmRowClassifier
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
open Out.mxm_wtype Out.Register Out.mxm_part
set_option autoImplicit false
set_option maxRecDepth 12000
set_option maxHeartbeats 1000000
namespace Proofs.MxmIntegrity
/-- Every admitted installed two-byte row names both actual physical planes,
    including the last pair. This is about the production accessor, Out.Functions.not a codec model. -/
theorem f16_row_valid (ws : Vector wrow Out.mxm_wvals.toNat) (r : Int)
    (hr : 0≤r) (hb : 2*(r:Int)+1<Out.Functions.mxm_wvals) :
    f16_row ws r = (ws[2*r]!,ws[2*r+1]!) := by
  simp [f16_row,hr,hb]
  all_goals first | rfl | constructor <;> rfl

theorem f16_row_invalid (ws : Vector wrow Out.mxm_wvals.toNat) (r : Int)
    (h : ¬(0≤r ∧ 2*(r:Int)+1<Out.Functions.mxm_wvals)) :
    f16_row ws r = (zero_vec,zero_vec) := by
  by_cases hr : 0≤r <;> by_cases hb : 2*r+1<Out.Functions.mxm_wvals
  all_goals simp_all [f16_row] <;> omega

theorem paired_checker_access (p : mxm_plane) (r : Nat) (hr : 2*(r:Int)+1<Out.Functions.mxm_wvals) :
    chk_wrow p W_F16 r = (p.weights[2*(r:Int)]!,p.weights[2*(r:Int)+1]!) ∧
    chk_wrow p W_BFP8 r = (p.weights[2*(r:Int)]!,p.weights[2*(r:Int)+1]!) ∧
    chk_wrow p W_BF16 r = (p.weights[2*(r:Int)]!,p.weights[2*(r:Int)+1]!) := by
  have he := f16_row_valid p.weights r (by omega) (by omega)
  simpa [chk_wrow] using And.intro he (And.intro he he)

/-- The install bound supplies the exact accessor premise for every row; no
    extra last-row exclusion is permitted. Arithmetic is unbounded Int/Nat. -/
theorem installed_pair_access (p : mxm_plane) (r : Nat)
    (install : 2*(p.rows:Int)≤Out.Functions.mxm_wvals) (row : r<p.rows) :
    chk_wrow p W_F16 r = (p.weights[2*(r:Int)]!,p.weights[2*(r:Int)+1]!) ∧
    chk_wrow p W_BFP8 r = (p.weights[2*(r:Int)]!,p.weights[2*(r:Int)+1]!) ∧
    chk_wrow p W_BF16 r = (p.weights[2*(r:Int)]!,p.weights[2*(r:Int)+1]!) := by
  apply paired_checker_access
  omega

theorem paired_bad_access (bs : Vector Bool Out.mxm_wvals.toNat) (r : Int)
    (hr : 0≤r) (hb : 2*(r:Int)+1<Out.Functions.mxm_wvals) :
    f16_row_bad bs r = (bs[2*r]! || bs[2*r+1]!) := by
  simp [f16_row_bad,hr,hb]
  all_goals first | rfl | constructor <;> rfl

open Out.instruction
open Out.ExecutionResult
open Out.mxm_acctype
open Out.streamref
open Out.diag
open Out.eltype
open Out.fclass
open Out.srf_lookup
open Out.acc_mode

/-- Public shape of the paired row selector: exact two-byte planes, otherwise zero.
    This remains defined at every Int index; no admitted input is excluded. -/
def rowSpec (ws : Vector wrow Out.mxm_wvals.toNat) (r : Int) : vec × vec :=
  if 0≤r ∧ 2*r+1<Out.Functions.mxm_wvals then (ws[2*r]!,ws[2*r+1]!) else (zero_vec,zero_vec)
def badSpec (bs : Vector Bool Out.mxm_wvals.toNat) (r : Int) : Bool :=
  if 0≤r ∧ 2*r+1<Out.Functions.mxm_wvals then bs[2*r]! || bs[2*r+1]! else false

theorem actual_row_spec (ws : Vector wrow Out.mxm_wvals.toNat) (r : Int) : f16_row ws r=rowSpec ws r := by
  unfold rowSpec
  by_cases h : 0≤r ∧ 2*r+1<Out.Functions.mxm_wvals
  · rw [if_pos h]
    exact f16_row_valid ws r h.1 h.2
  · rw [if_neg h]
    exact f16_row_invalid ws r h

theorem actual_bad_spec (bs : Vector Bool Out.mxm_wvals.toNat) (r : Int) : f16_row_bad bs r=badSpec bs r := by
  by_cases a : 0≤r <;> by_cases b : 2*r+1<Out.Functions.mxm_wvals
  all_goals simp [f16_row_bad,badSpec,a,b] <;> rfl

/-- Exact full production ABC body, with only its shared row-value/poison selectors
    made explicit parameters. It retains all admission, state, SRF and report effects. -/
def abcWithRows (rowFn : Vector wrow Out.mxm_wvals.toNat → Int → (vec × vec))
    (badFn : Vector Bool Out.mxm_wvals.toNat → Int → Bool) (t : mxm_wtype) (f : (BitVec 1)) (s : streamref) : SailM ExecutionResult := do
  let p ← do (pure (GetElem?.getElem! (← Out.readReg mxm) (← (plane ()))))
  let w : Int :=
    if ((is_pair t) : Bool)
    then 2
    else 1
  if (((← (plane_blocked (some t))) || (← do
         (pure ((p.rows == 0) || (← do
               (pure (((← Out.readReg issue_cycle) <b p.ready_iw) || ((Out.Functions.not (abc_types_ok p.wtype t)) || (← (macc_busy
                         p)))))))))) : Bool)
  then (pure (Diagnostic RES_NOT_READY))
  else
    (do
      match (← (read_planes s w (← (exec_cycle ())))) with
      | .Err e => (pure (Diagnostic e))
      | .Ok x =>
        (do
          let mine : (List (Nat × Nat)) := []
          let others : (List mxm_fault) := []
          let pend ← do Out.readReg mxm_fault_list
          let (mine, others, pend) ← (( do
            let loop__step_lower := 0
            let loop__step_upper ← do (pure ((list_len (← Out.readReg mxm_fault_list)) -i 1))
            let mut loop_vars := (mine, others, pend)
            for _step in [loop__step_lower:loop__step_upper:1]i do
              let (mine, others, pend) := loop_vars
              loop_vars ← do
                let (mine, others, pend) ← (( do
                  match pend with
                  | [] => (pure (mine, others, pend))
                  | (mfe :: rest) =>
                    (do
                      let (mine, others) ← (( do
                        if ((mfe.mf_pl == (← (plane ()))) : Bool)
                        then
                          (let mine : (List (Nat × Nat)) := ((mfe.mf_lane, mfe.mf_bit) :: mine)
                          (pure (mine, others)))
                        else
                          (let others : (List mxm_fault) := (mfe :: others)
                          (pure (mine, others))) ) : SailM ((List (Nat × Nat)) × (List mxm_fault))
                        )
                      let pend : (List mxm_fault) := rest
                      (pure (mine, others, pend))) ) : SailM
                  ((List (Nat × Nat)) × (List mxm_fault) × (List mxm_fault)) )
                (pure (mine, others, pend))
            (pure loop_vars) ) : SailM ((List (Nat × Nat)) × (List mxm_fault) × (List mxm_fault))
            )
          Out.writeReg mxm_fault_list others
          let verdict : (Nat × Nat) := (0, 0)
          let verdict : (Nat × Nat) :=
            match mine with
            | [] => verdict
            | _ =>
              (let x1 : vec :=
                if ((is_pair t) : Bool)
                then (GetElem?.getElem! x 1)
                else zero_vec
              (mxm_check p t f (GetElem?.getElem! x 0) x1 mine))
          let (vstat, vlane) := verdict
          let ec ← (( do
            if (((← (exec_cycle ())) ≥b 0) : Bool)
            then (exec_cycle ())
            else (pure 0) ) : SailM Nat )
          if ((vstat == 2) : Bool)
          then
            (do
              (ras_mxm_event ec (← Out.readReg issue_queue) true (← (plane ())) 65535)
              (pure (Diagnostic ECC_MXM)))
          else
            (do
              if ((vstat == 1) : Bool)
              then
                (do
                  (pure (mxm_corr_callback ec (← Out.readReg issue_queue) (← (plane ())) vlane))
                  (ras_mxm_event ec (← Out.readReg issue_queue) false (← (plane ())) vlane))
              else (pure ())
              let out : (Vector (BitVec 32) Out.Functions.mxm_rows) := (vectorInit 0x00000000#32)
              let wide : (Vector wlane Out.Functions.mxm_rows) := (vectorInit wlane_zero)
              let ws := (p.wtype == W_I8)
              let xs := (t == W_I8)
              let (out, wide) ← (( do
                let loop_r_lower := 0
                let loop_r_upper := (p.rows -i 1)
                let mut loop_vars_1 := (out, wide)
                for r in [loop_r_lower:loop_r_upper:1]i do
                  let (out, wide) := loop_vars_1
                  loop_vars_1 :=
                    let (out, wide) : ((Vector (BitVec 32) Out.mxm_rows.toNat) × (Vector wlane Out.mxm_rows.toNat)) :=
                      match t with
                      | .W_I8 =>
                        (let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r
                            (int8_dot (GetElem?.getElem! p.weights r) ws (GetElem?.getElem! x 0) xs))
                        (out, wide))
                      | .W_U8 =>
                        (let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r
                            (int8_dot (GetElem?.getElem! p.weights r) ws (GetElem?.getElem! x 0) xs))
                        (out, wide))
                      | .W_FP8 =>
                        (let d :=
                          (fp8_exact
                            (if ((f == 0#1) : Bool)
                            then FMT_E4M3
                            else FMT_E5M2) (GetElem?.getElem! p.weights r) (GetElem?.getElem! x 0))
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r (dsum_round d))
                        let wide : (Vector wlane Out.mxm_rows.toNat) :=
                          if (Out.Functions.ext_wide_acc : Bool)
                          then (vectorUpdate wide r (dsum_wide d))
                          else wide
                        (out, wide))
                      | .W_F16 =>
                        (let (wlo, whi) := (rowFn p.weights r)
                        let d := (f16_exact wlo whi (GetElem?.getElem! x 0) (GetElem?.getElem! x 1))
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r (dsum_round d))
                        let wide : (Vector wlane Out.mxm_rows.toNat) :=
                          if (Out.Functions.ext_wide_acc : Bool)
                          then (vectorUpdate wide r (dsum_wide d))
                          else wide
                        (out, wide))
                      | .W_BFP8 =>
                        (let (wd, wsc) := (rowFn p.weights r)
                        let d := (bfp8_exact wd wsc (GetElem?.getElem! x 0) (GetElem?.getElem! x 1))
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r (dsum_round d))
                        let wide : (Vector wlane Out.mxm_rows.toNat) :=
                          if (Out.Functions.ext_wide_acc : Bool)
                          then (vectorUpdate wide r (dsum_wide d))
                          else wide
                        (out, wide))
                      | .W_BF16 =>
                        (let (wlo, whi) := (rowFn p.weights r)
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r
                            (bf16_dot wlo whi (GetElem?.getElem! x 0) (GetElem?.getElem! x 1)))
                        (out, wide))
                    (out, wide)
                (pure loop_vars_1) ) : SailM ((Vector (BitVec 32) Out.mxm_rows.toNat) × (Vector wlane Out.mxm_rows.toNat)) )
              let poison ← (( do
                if (((← Out.readReg ecc_bad) != (BitVec.zero Out.Functions.superlanes)) : Bool)
                then (pure (sail_ones Out.Functions.superlanes))
                else (pure (BitVec.zero Out.Functions.superlanes)) ) : SailM (BitVec Out.Functions.superlanes) )
              let poison ← (( do
                if ((poison == (BitVec.zero Out.Functions.superlanes)) : Bool)
                then
                  (do
                    let loop_r_lower := 0
                    let loop_r_upper := (p.rows -i 1)
                    let mut loop_vars_2 := poison
                    for r in [loop_r_lower:loop_r_upper:1]i do
                      let poison := loop_vars_2
                      loop_vars_2 :=
                        let rb : Bool :=
                          match t with
                          | .W_F16 => (badFn p.w_bad r)
                          | .W_BFP8 => (badFn p.w_bad r)
                          | .W_BF16 => (badFn p.w_bad r)
                          | _ => (GetElem?.getElem! p.w_bad r)
                        if ((rb && ((tdiv_int r 16) <b Out.Functions.superlanes)) : Bool)
                        then (BitVec.update poison (tdiv_int r 16) 1#1)
                        else poison
                    (pure loop_vars_2))
                else (pure poison) ) : SailM (BitVec Out.Functions.superlanes) )
              let e ← (( do
                (pure { dispatch := ← Out.readReg issue_cycle
                        commit := ← (commit_at ())
                        ptype := t
                        product := out
                        wide := wide
                        poison := poison
                        q := ← Out.readReg issue_queue }) ) : SailM abc_entry )
              let q := p
              let q ←
                (pure { q with abc := (e :: (abc_prune p.abc (← Out.readReg issue_cycle))) })
              (mxm_stage (← (plane ())) P_ABC q)
              (pure (Issued ())))))

theorem actual_abc_row_factor (t : mxm_wtype) (f : BitVec 1) (stream : streamref) :
    abc_exec t f stream=abcWithRows f16_row f16_row_bad t f stream := by rfl

/-- The complete arithmetic caller uses the proved accessor at every production
    occurrence, including paired poison selection. Numeric correctness is supplied
    separately by the accepted arithmetic graph for these exact operands. -/
theorem actual_abc_uses_spec_rows (t : mxm_wtype) (f : BitVec 1) (stream : streamref) :
    abc_exec t f stream=abcWithRows rowSpec badSpec t f stream := by
  rw [actual_abc_row_factor]
  have rowEq : f16_row=rowSpec := funext (fun ws => funext (actual_row_spec ws))
  have badEq : f16_row_bad=badSpec := funext (fun bs => funext (actual_bad_spec bs))
  rw [rowEq,badEq]


/-- Complete production checker factor, retaining primary/independent nominal
    recomputation, every actual sequential flip, and the final indexed row check. -/
def checkerWithRows (p : mxm_plane) (t : mxm_wtype) (f : (BitVec 1)) (x0 : vec) (x1 : vec) (fl : (List (Nat × Nat))) : (Nat × Nat) := Id.run do
  let nr := p.rows
  let ws := (p.wtype == W_I8)
  let xs := (t == W_I8)
  let xv : (Vector (Option Int) Out.Functions.vlanes) := (vectorInit none)
  let xv ← (( do
    let loop_k_lower := 0
    let loop_k_upper := (Out.Functions.vlanes -i 1)
    let mut loop_vars := xv
    for k in [loop_k_lower:loop_k_upper:1]i do
      let xv := loop_vars
      loop_vars := (vectorUpdate xv k (chk_elem t f x0 x1 k xs))
    (pure loop_vars) ) : Id (Vector (Option Int) Out.Functions.vlanes) )
  let ps : (Vector Int Out.Functions.mxm_rows) := (vectorInit 0)
  let expected : (Vector Int Out.Functions.mxm_rows) := (vectorInit 0)
  let ps ← (( do
    let loop_r_lower := 0
    let loop_r_upper := (nr -i 1)
    let mut loop_vars_1 := ps
    for r in [loop_r_lower:loop_r_upper:1]i do
      let ps := loop_vars_1
      loop_vars_1 ← do
        let (lo, hi) := (chk_wrow p t r)
        let loop_k_lower := 0
        let loop_k_upper := (Out.Functions.vlanes -i 1)
        let mut loop_vars_2 := ps
        for k in [loop_k_lower:loop_k_upper:1]i do
          let ps := loop_vars_2
          loop_vars_2 :=
            match ((chk_elem t f lo hi k ws), (GetElem?.getElem! xv k)) with
            | (.some a, .some b) => (vectorUpdate ps r ((GetElem?.getElem! ps r) +i (a *i b)))
            | _ => ps
        (pure loop_vars_2)
    (pure loop_vars_1) ) : Id (Vector Int Out.Functions.mxm_rows) )
  let expected ← (( do
    let loop_k_lower := 0
    let loop_k_upper := (Out.Functions.vlanes -i 1)
    let mut loop_vars_3 := expected
    for k in [loop_k_lower:loop_k_upper:1]i do
      let expected := loop_vars_3
      loop_vars_3 ← do
        match (GetElem?.getElem! xv k) with
        | none => (pure expected)
        | .some xk =>
          (do
            let loop_r_lower := 0
            let loop_r_upper := (nr -i 1)
            let mut loop_vars_4 := expected
            for r in [loop_r_lower:loop_r_upper:1]i do
              let expected := loop_vars_4
              loop_vars_4 :=
                let (lo, hi) := (chk_wrow p t r)
                match (chk_elem t f lo hi k ws) with
                | none => expected
                | .some w => (vectorUpdate expected r ((GetElem?.getElem! expected r) +i (w *i xk)))
            (pure loop_vars_4))
    (pure loop_vars_3) ) : Id (Vector Int Out.Functions.mxm_rows) )
  let faulty : (List (Nat × Nat)) := fl
  let (faulty, ps) ← (( do
    let loop__step_lower := 0
    let loop__step_upper := ((list_len fl) -i 1)
    let mut loop_vars_5 := (faulty, ps)
    for _step in [loop__step_lower:loop__step_upper:1]i do
      let (faulty, ps) := loop_vars_5
      loop_vars_5 :=
        let (faulty, ps) : ((List (Nat × Nat)) × (Vector Int Out.Functions.mxm_rows)) :=
          match faulty with
          | [] => (faulty, ps)
          | ((lane, bit) :: rest) =>
            (let ps : (Vector Int Out.Functions.mxm_rows) :=
              if ((lane <b nr) : Bool)
              then (vectorUpdate ps lane (flip96 (GetElem?.getElem! ps lane) bit))
              else ps
            let faulty : (List (Nat × Nat)) := rest
            (faulty, ps))
        (faulty, ps)
    (pure loop_vars_5) ) : Id ((List (Nat × Nat)) × (Vector Int Out.Functions.mxm_rows)) )
  return Proofs.MxmRowClassifier.finalRows
    (fun r => GetElem?.getElem! ps r) (fun r => GetElem?.getElem! expected r) nr

theorem actual_checker_row_factor (p : mxm_plane) (t : mxm_wtype) (f : BitVec 1)
    (x0 x1 : vec) (faults : List (Nat × Nat)) :
    mxm_check p t f x0 x1 faults=checkerWithRows p t f x0 x1 faults := by
  simp only [mxm_check,checkerWithRows,Proofs.MxmRowClassifier.finalRows,Proofs.MxmRowClassifier.rowStep]
  rfl

/-- Every actually installed paired row, including the final one, satisfies the
    full accessor guard. This theorem does Out.Functions.not assume all malformed planes installed. -/
theorem installed_pair_guard (rows r : Nat) (installed : 2*(rows:Int)≤Out.Functions.mxm_wvals) (inside : r<rows) :
    0≤(r:Int) ∧ 2*(r:Int)+1<Out.Functions.mxm_wvals := by
  omega

end Proofs.MxmIntegrity
