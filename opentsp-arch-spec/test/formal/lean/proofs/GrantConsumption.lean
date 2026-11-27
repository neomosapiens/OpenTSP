-- Actual granted/rejected dispatcher calls, retaining real execution and error ordering.
import Proofs.ResourceAdmission
import Proofs.Resources
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions Out.diag Out.bank_use Out.Register
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace GrantConsumptionProofs
def consumeNewRef (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) : SailM (Nat × Option instruction) := do
  let (issued, last) ← (( do
    match (GetElem?.getElem! r.refs k) with
    | .some i =>
      (do
        let (issued, last) ← (( do
          if ((GetElem?.getElem! go k) : Bool)
          then
            (do
              let (issued, last) ← (( do
                if ((GetElem?.getElem! bank k) : Bool)
                then
                  (do
                    (report q c MEM_BANK)
                    let po : mem_path :=
                      if ((k ≥b 2) : Bool)
                      then 0
                      else 1
                    let o := ((2 *i po) +i (tmod_int k 2))
                    let partner : (Option clash_ref) := none
                    let partner : (Option (instruction × Int × Int)) :=
                      match (GetElem?.getElem! loop po) with
                      | .some tg =>
                        (if ((tags_clash (tag_of i 0) tg) : Bool)
                        then (GetElem?.getElem! lref po)
                        else partner)
                      | none => partner
                    let partner : (Option (instruction × Int × Int)) :=
                      match (GetElem?.getElem! r.refs o) with
                      | .some j =>
                        (if (((GetElem?.getElem! go o) && (tags_clash (tag_of i 0)
                               (tag_of j 0))) : Bool)
                        then (some (j, 0, 0))
                        else partner)
                      | none => partner
                    match partner with
                    | .some x => (bank_report q c (i, 0, 0) x)
                    | none => (pure ())
                    (pure (issued, last)))
                else
                  (do
                    let p : mem_path :=
                      if ((k ≥b 2) : Bool)
                      then 1
                      else 0
                    (hold_path q p c)
                    let issued : Nat := (issued +i 1)
                    let last : (Option instruction) := (some i)
                    let alone : Bool :=
                      ((GetElem?.getElem! r.refs
                          ((k -i (tmod_int k 2)) +i (1 -i (tmod_int k 2)))) == none)
                    (run_element q i c 0 0 0)
                    if (((is_loopable i) && alone) : Bool)
                    then
                      (do
                        (agt_set q p (agt_opened i c))
                        writeReg agt_records (vectorUpdate (← readReg agt_records)
                          ((2 *i q) +i p) 0))
                    else (pure ())
                    (pure (issued, last))) ) : SailM (Nat × (Option instruction)) )
              (pure (issued, last)))
          else (pure (issued, last)) ) : SailM (Nat × (Option instruction)) )
        (pure (issued, last)))
    | none => (pure (issued, last)) ) : SailM (Nat × (Option instruction)) )
  (pure (issued, last))

def consumeFetch (q c : Nat) (r : cycle_recs) (fm_ok fm_clash : Bool)
    (issued : Nat) (last : Option instruction) : SailM (Nat × Option instruction) := (( do
    if (fm_ok : Bool)
    then
      (do
        let (issued, last) ← (( do
          if (fm_clash : Bool)
          then
            (do
              (report q c MEM_BANK)
              (pure (issued, last)))
          else
            (do
              let (issued, last) ← (( do
                match r.ifetch with
                | .some i =>
                  (do
                    (run_element q i c 0 0 0)
                    (reserve_local_fetch q i c)
                    let issued : Nat := (issued +i 1)
                    let last : (Option instruction) := (some i)
                    (pure (issued, last)))
                | none => (pure (issued, last)) ) : SailM (Nat × (Option instruction)) )
              (pure (issued, last))) ) : SailM (Nat × (Option instruction)) )
        (pure (issued, last)))
    else (pure (issued, last)) ) : SailM (Nat × (Option instruction)) )

def dispatchFactory (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : SailM Unit := do
  let issued : Nat := 0
  let last : (Option instruction) := none
  let fm_ok : Bool := false
  let fm_bank : bank_use := (BankNone ())
  let fm_prior : Bool := false
  let (fm_bank, fm_prior) ← (( do
    match (GetElem?.getElem! (← readReg local_fetch_bank) q) with
    | .some bn =>
      (let (fm_bank, fm_prior) : (bank_use × Bool) :=
        if ((bn.bn_at == c) : Bool)
        then
          (let fm_prior : Bool := true
          let fm_bank : bank_use := (addr_bank bn.bn_addr)
          (fm_bank, fm_prior))
        else (fm_bank, fm_prior)
      (pure (fm_bank, fm_prior)))
    | none => (pure (fm_bank, fm_prior)) ) : SailM (bank_use × Bool) )
  let (fm_bank, fm_ok, issued, last) ← (( do
    match r.ifetch with
    | none => (pure (fm_bank, fm_ok, issued, last))
    | .some i =>
      (do
        let fpath : (Option mem_path) :=
          match i with
          | .IFETCHM _ => (some 0)
          | _ => none
        let (fm_bank, fm_ok, issued, last) ← (( do
          match (← (dispatch_check q c i fpath)) with
          | .some d =>
            (do
              (report q c d)
              (pure (fm_bank, fm_ok, issued, last)))
          | none =>
            (do
              let (fm_bank, fm_ok, issued, last) ← (( do
                match i with
                | .IFETCHM (.SramAddr a) =>
                  (let fm_ok : Bool := true
                  let fm_bank : bank_use := (addr_bank (BitVec.toNatInt a))
                  (pure (fm_bank, fm_ok, issued, last)))
                | _ =>
                  (do
                    (run_element q i c 0 0 0)
                    let issued : Nat := (issued +i 1)
                    let last : (Option instruction) := (some i)
                    (pure (fm_bank, fm_ok, issued, last))) ) : SailM
                (bank_use × Bool × Nat × (Option instruction)) )
              (pure (fm_bank, fm_ok, issued, last))) ) : SailM
          (bank_use × Bool × Nat × (Option instruction)) )
        (pure (fm_bank, fm_ok, issued, last))) ) : SailM
    (bank_use × Bool × Nat × (Option instruction)) )
  match r.cfg with
  | .some i => (dispatch_cfg q c (← (cfg_path q)) i)
  | none => (pure ())
  let due ← do (due_refs q c parked)
  let loop : (Vector (Option ref_tag) 2) := (vectorInit none)
  let lref : (Vector (Option clash_ref) 2) := (vectorInit none)
  let (loop, lref) ← (( do
    let loop_p_lower := 0
    let loop_p_upper := 1
    let mut loop_vars := (loop, lref)
    for p in [loop_p_lower:loop_p_upper:1]i do
      let (loop, lref) := loop_vars
      loop_vars ← do
        let loop ← (pure (vectorUpdate loop p (← (loop_tag q p (GetElem?.getElem! due p)))))
        let lref ← (pure (vectorUpdate lref p (← (due_ref q p (GetElem?.getElem! due p)))))
        (pure (loop, lref))
    (pure loop_vars) ) : SailM
    ((Vector (Option ref_tag) 2) × (Vector (Option (instruction × Int × Int)) 2)) )
  let go : (Vector Bool 4) := (vectorInit false)
  let go ← (( do
    let loop_k_lower := 0
    let loop_k_upper := 3
    let mut loop_vars_1 := go
    for k in [loop_k_lower:loop_k_upper:1]i do
      let go := loop_vars_1
      loop_vars_1 ← do
        match (GetElem?.getElem! r.refs k) with
        | none => (pure go)
        | .some i =>
          (do
            let p : mem_path :=
              if ((k ≥b 2) : Bool)
              then 1
              else 0
            match (← (dispatch_check q c i (some p))) with
            | .some d =>
              (do
                (report q c d)
                (pure go))
            | none => (pure (vectorUpdate go k true)))
    (pure loop_vars_1) ) : SailM (Vector Bool 4) )
  let rt : (Vector (Option ref_tag) 4) := (vectorInit none)
  let rt ← (( do
    let loop_k_lower := 0
    let loop_k_upper := 3
    let mut loop_vars_2 := rt
    for k in [loop_k_lower:loop_k_upper:1]i do
      let rt := loop_vars_2
      loop_vars_2 :=
        match (GetElem?.getElem! r.refs k) with
        | .some i => (vectorUpdate rt k (some (tag_of i 0)))
        | none => rt
    (pure loop_vars_2) ) : SailM (Vector (Option ref_tag) 4) )
  let bv := (bank_rule rt go loop (fm_ok || fm_prior) fm_bank)
  let bank := bv.vbank
  let fm_clash := bv.vfm_clash
  let (issued,last) ← consumeFetch q c r fm_ok fm_clash issued last
  let lblock := bv.vlblock
  let lpart : (Vector (Option clash_ref) 2) := (vectorInit none)
  let lpart ← (( do
    let loop_p_lower := 0
    let loop_p_upper := 1
    let mut loop_vars_3 := lpart
    for p in [loop_p_lower:loop_p_upper:1]i do
      let lpart := loop_vars_3
      loop_vars_3 :=
        let src := (GetElem?.getElem! bv.vlsrc p)
        if ((src == 4) : Bool)
        then (vectorUpdate lpart p (GetElem?.getElem! lref (1 -i p)))
        else
          (if ((src ≥b 0) : Bool)
          then
            (match (GetElem?.getElem! r.refs (Int.emod src 4)) with
            | .some j => (vectorUpdate lpart p (some (j, 0, 0)))
            | none => lpart)
          else lpart)
    (pure loop_vars_3) ) : SailM (Vector (Option (instruction × Int × Int)) 2) )
  let loop_p_lower := 0
  let loop_p_upper := 1
  let mut loop_vars_4 := ()
  for p in [loop_p_lower:loop_p_upper:1]i do
    let () := loop_vars_4
    loop_vars_4 ← do
      (run_loop q c p (GetElem?.getElem! due p) (GetElem?.getElem! lblock p)
        (GetElem?.getElem! lpart p))
  (pure loop_vars_4)
  let (issued, last) ← (( do
    let loop_k_lower := 0
    let loop_k_upper := 3
    let mut loop_vars_5 := (issued, last)
    for k in [loop_k_lower:loop_k_upper:1]i do
      let (issued, last) := loop_vars_5
      loop_vars_5 ← do
        consumeNewRef q c r loop lref go bank k issued last
    (pure loop_vars_5) ) : SailM (Nat × (Option instruction)) )
  if ((issued >b 1) : Bool)
  then writeReg queue_last (vectorUpdate (← readReg queue_last) q none)
  else
    (do
      if ((issued == 1) : Bool)
      then
        (do
          writeReg queue_last (vectorUpdate (← readReg queue_last) q last)
          writeReg queue_last_at (vectorUpdate (← readReg queue_last_at) q c))
      else (pure ()))


section SourceConnection
attribute [local irreducible] IntRange.forIn' Out.Functions.run_element Out.Functions.dispatch_check Out.Functions.report Out.Functions.cfg_path Out.Functions.bank_rule

theorem actual_dispatch_factory_source (q c : Nat) (r : cycle_recs) (parked : Bool) :
    dispatch_slots q c r parked=dispatchFactory q c r parked := by
  simp only [dispatch_slots,dispatchFactory,consumeNewRef,consumeFetch,pure_bind,bind_pure,bind_assoc]
  rfl
end SourceConnection

def grantedStep (q c : Nat) (r : cycle_recs) (k : Int) (i : instruction) (issued : Nat) : SailM (Nat × Option instruction) := do
  let p : Nat := if ((k ≥b 2):Bool) then 1 else 0
  hold_path q p c
  run_element q i c 0 0 0
  if (((is_loopable i) && ((GetElem?.getElem! r.refs ((k -i (tmod_int k 2)) +i (1 -i (tmod_int k 2))))==none)):Bool) then
    agt_set q p (agt_opened i c)
    writeReg agt_records (vectorUpdate (← readReg agt_records) ((2 *i q) +i p) 0)
  pure (issued+1,some i)

theorem actual_granted_reference (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) (i : instruction)
    (href : GetElem?.getElem! r.refs k=some i) (hgo : GetElem?.getElem! go k=true)
    (hbank : GetElem?.getElem! bank k=false) :
    consumeNewRef q c r loop lref go bank k issued last=grantedStep q c r k i issued := by
  have hc : ((issued:Int)+1).toNat=issued+1 := by omega
  simp [consumeNewRef,grantedStep,href,hgo,hbank,hc,pure_bind,bind_pure,bind_assoc]

theorem unchecked_reference_no_effect (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) (hgo : GetElem?.getElem! go k=false) :
    consumeNewRef q c r loop lref go bank k issued last=pure (issued,last) := by
  unfold consumeNewRef
  cases h : GetElem?.getElem! r.refs k <;> simp [hgo,pure_bind,bind_pure]

theorem absent_reference_no_effect (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) (href : GetElem?.getElem! r.refs k=none) :
    consumeNewRef q c r loop lref go bank k issued last=pure (issued,last) := by
  simp [consumeNewRef,href]

theorem granted_fetch_consumes_reservation (q c : Nat) (r : cycle_recs) (issued : Nat)
    (last : Option instruction) (i : instruction) (href : r.ifetch=some i) :
    consumeFetch q c r true false issued last=(do
      run_element q i c 0 0 0
      reserve_local_fetch q i c
      pure (issued+1,some i)) := by
  have hc : ((issued:Int)+1).toNat=issued+1 := by omega
  simp [consumeFetch,href,hc]

theorem rejected_fetch_only_reports (q c : Nat) (r : cycle_recs) (issued : Nat) (last : Option instruction) :
    consumeFetch q c r true true issued last=(do report q c MEM_BANK; pure (issued,last)) := by
  simp [consumeFetch]

theorem disabled_fetch_no_effect (q c : Nat) (r : cycle_recs) (clash : Bool) (issued : Nat) (last : Option instruction) :
    consumeFetch q c r false clash issued last=pure (issued,last) := by simp [consumeFetch]

def advanceLoop (q c p : Nat) (a : agt) (ls : Vector agt_level 4) (s ss : Int) : SailM Unit := do
  hold_path q p c
  agt_set q p (agt_after_reference a ls s ss c)
  agt_refresh q p

def rejectedLoop (q c p : Nat) (ls : Vector agt_level 4) (s ss : Int)
    (partner : Option clash_ref) : SailM Unit := do
  let a ← agt_get q p
  report q c MEM_BANK
  match partner with
  | some o => bank_report q c (← loop_clash_ref q p s ss) o
  | none => pure ()
  advanceLoop q c p a ls s ss

theorem actual_blocked_loop_reports_then_advances (q c p n : Nat) (ls : Vector agt_level 4)
    (s ss : Int) (partner : Option clash_ref) :
    run_loop q c p (some (ls,n,s,ss)) true partner=rejectedLoop q c p ls s ss partner := by
  cases partner <;> simp [run_loop,rejectedLoop,advanceLoop,bind_assoc]

theorem actual_granted_loop_executes_then_advances (q c p n : Nat) (ls : Vector agt_level 4)
    (s ss : Int) (partner : Option clash_ref) :
    run_loop q c p (some (ls,n,s,ss)) false partner=(do
      let a ← agt_get q p
      run_element q a.instr a.t0 (Max.max (c -i a.t0) 0) (a.offset +i s) (a.soffset +i ss)
      advanceLoop q c p a ls s ss) := by
  simp [run_loop,advanceLoop,bind_assoc]

def rejectPartner (r : cycle_recs) (loop : Vector (Option ref_tag) 2) (lref : Vector (Option clash_ref) 2)
    (go : Vector Bool 4) (k : Int) (i : instruction) : Option clash_ref :=
  let p : Nat := if ((k ≥b 2):Bool) then 0 else 1
  let o : Int := 2*(p:Int)+tmod_int k 2
  let agtPartner := match GetElem?.getElem! loop p with
    | some tg => if tags_clash (tag_of i 0) tg then GetElem?.getElem! lref p else none
    | none => none
  match GetElem?.getElem! r.refs o with
  | some j => if ((GetElem?.getElem! go o)&&(tags_clash (tag_of i 0) (tag_of j 0)):Bool)
      then some (j,0,0) else agtPartner
  | none => agtPartner

theorem actual_rejected_reference_only_reports (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) (i : instruction)
    (href : GetElem?.getElem! r.refs k=some i) (hgo : GetElem?.getElem! go k=true)
    (hbank : GetElem?.getElem! bank k=true) :
    consumeNewRef q c r loop lref go bank k issued last=(do
      report q c MEM_BANK
      match rejectPartner r loop lref go k i with
      | some other => bank_report q c (i,0,0) other
      | none => pure ()
      pure (issued,last)) := by
  simp [consumeNewRef,rejectPartner,href,hgo,hbank,bind_assoc]
  rfl

end GrantConsumptionProofs
