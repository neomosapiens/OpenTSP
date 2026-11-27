-- Actual control-channel frontier-to-quiet transfer. No whole-tick claim.
import Proofs.EventQuiet
import Proofs.EventQueries
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventProofs EventCensusProofs EventQuietProofs
open Register agt_state
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace EventDispatchProofs

theorem control_empty (s : EventProofs.St) (c : Nat) (L : Vector Nat 145)
    (hL : s.regs.get? Register.iq_len = some L) (hz : L[144]! = 0) :
    (Out.Functions.control c).run s = .ok () s := by
  have hz2 : L[144] = 0 := by simpa using hz
  simp [Out.Functions.control,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hL,hz2]

theorem control_incomplete (s : EventProofs.St) (c : Nat) (L : Vector Nat 145) (p : parcel)
    (hL : s.regs.get? Register.iq_len = some L) (hp : p.complete = false)
    (ready : Int) (early : (parcel_earliest 144 p).run s = .ok ready s)
    (scan : (parcel_scan 144).run s = .ok p s) :
    (Out.Functions.control c).run s = .ok () s := by
  have hs : parcel_scan 144 s = .ok p s := scan
  have he : parcel_earliest 144 p s = .ok ready s := early
  simp only [Out.Functions.control,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hL]
  split
  · simp [hs,hp,he,bind,EStateM.bind,pure,EStateM.pure]
  · rfl

theorem control_before_ready (s : EventProofs.St) (c : Nat) (L : Vector Nat 145)
    (p : parcel) (ready : Int)
    (hL : s.regs.get? Register.iq_len = some L)
    (scan : (parcel_scan 144).run s = .ok p s)
    (early : (parcel_earliest 144 p).run s = .ok ready s) (hc : (c:Int) < ready) :
    (Out.Functions.control c).run s = .ok () s := by
  have hs : parcel_scan 144 s = .ok p s := scan
  have he : parcel_earliest 144 p s = .ok ready s := early
  simp only [Out.Functions.control,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hL]
  split
  · simp [hs,he,show ¬ready≤c by omega,bind,EStateM.bind,pure,EStateM.pure]
  · rfl

theorem actual_control_frontier (s : EventProofs.St) (c t : Nat) (L : Vector Nat 145)
    (p : parcel) (ready : Int) (frontier : Int)
    (hL : s.regs.get? Register.iq_len = some L) (hn : 0 < L[144]!)
    (scan : (parcel_scan 144).run s = .ok p s)
    (early : (parcel_earliest 144 p).run s = .ok ready s)
    (event : (control_event c).run s = .ok frontier s)
    (hct : c < t) (ht : (t:Int) < frontier) :
    (Out.Functions.control t).run s = .ok () s := by
  have hs : parcel_scan 144 s = .ok p s := scan
  have he : parcel_earliest 144 p s = .ok ready s := early
  have hn2 : L[144] ≠ 0 := by simpa using (Nat.ne_of_gt hn)
  cases hp : p.complete
  · exact control_incomplete s t L p hL hp ready early scan
  · have ev := event
    simp [control_event,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hL,hn2,hs,he,hp] at ev
    have hr : (t:Int) < ready := by omega
    exact control_before_ready s t L p ready hL scan early hr

-- The actual complete dispatcher suffix; extracted verbatim from dispatch_tail.
def dispatch_scan (c : Nat) (best : Int) : SailM Int := do
  let loop_q_lower := 0
  let loop_q_upper := 143
  let mut loop_vars_5 := best
  for q in [loop_q_lower:loop_q_upper:1]i do
    let best := loop_vars_5
    loop_vars_5 ← do
      let best ← (( do
        let loop_p_lower := 0
        let loop_p_upper := 1
        let mut loop_vars_6 := best
        for p in [loop_p_lower:loop_p_upper:1]i do
          let best := loop_vars_6
          loop_vars_6 ← do
            let a ← do (agt_get q p)
            if ((a.state == AGT_CONFIG) : Bool)
            then (pure (earliest best (c +i 1)))
            else
              (do
                if (((a.state == AGT_RUN) && (Out.Functions.not (← (held q)))) : Bool)
                then
                  (do
                    (pure (earliest best (← (unparked_from q (Max.max a.next (c +i 1)))))))
                else (pure best))
        (pure loop_vars_6) ) : SailM Int )
      let best ← (( do
        match (GetElem?.getElem! (← Out.readReg queue_repeat) q) with
        | .some r =>
          (do
            if ((Out.Functions.not (← (held q))) : Bool)
            then
              (do
                (pure (earliest best (← (unparked_from q (Max.max r.next (c +i 1)))))))
            else (pure best))
        | none => (pure best) ) : SailM Int )
      (pure (earliest best (← (parcel_event q c))))
  (pure loop_vars_5)

theorem actual_tail_split (c : Nat) (best : Int) :
    dispatch_tail c best = (do
      let control ← control_event c
      dispatch_scan c (earliest best control)) := by
  simp only [dispatch_tail,dispatch_scan,bind_assoc,pure_bind,bind_pure]
  rfl

theorem scan_keeps_urgent (c : Nat) (target best : Int) (hb : Urgent target best) :
    Post (Urgent target) (dispatch_scan c best) := by
  simp only [dispatch_scan,bind_assoc,pure_bind,bind_pure]
  apply post_forIn [0:143:1]i (Urgent target) best _
    hb
  intro q best hb
  apply post_bind (Urgent target)
  · apply post_forIn [0:1:1]i (Urgent target) best _ hb
    intro p best hb
    apply post_bind_any
    intro a
    split
    · apply post_pure
      exact earliest_keeps_urgent target _ _ hb
    · apply post_bind_any
      intro held
      apply post_bind (Urgent target)
      · split
        · apply post_bind_any
          intro u
          apply post_pure
          exact earliest_keeps_urgent target _ _ hb
        · exact post_pure _ _ hb
      · intro after ha
        exact post_pure _ _ ha
  · intro after_agt hagt
    apply post_bind_any
    intro repeats
    apply post_bind (Urgent target)
    · split
      · apply post_bind_any
        intro held
        split
        · apply post_bind_any
          intro u
          apply post_pure
          exact earliest_keeps_urgent target _ _ hagt
        · exact post_pure _ _ hagt
      · exact post_pure _ _ hagt
    · intro after_repeat hrep
      apply post_bind_any
      intro event
      apply post_pure
      exact earliest_keeps_urgent target _ _ hrep

theorem earliest_right_urgent (target a b : Int) (hb : Urgent target b) :
    Urgent target (earliest a b) := by
  unfold earliest Urgent at *
  repeat' first | (split <;> simp_all) | omega | constructor

theorem model_captures_control (s after : EventProofs.St) (v : QueuedSnapshot s)
    (c : Nat) (target controlValue result : Int)
    (control_run : (control_event c).run s = .ok controlValue s)
    (hc : Urgent target controlValue)
    (model_run : (model_next_event c).run s = .ok result after) : Urgent target result := by
  rw [production_split] at model_run
  simp only [actual_tail_split] at model_run
  have hp : queued_prefix c s = .ok (snapshotResult s v c) s := queued_snapshot_run s v c
  have hctrl : control_event c s = .ok controlValue s := control_run
  simp only [EStateM.run,bind,EStateM.bind,hp,hctrl] at model_run
  have h := scan_keeps_urgent c target (earliest (snapshotResult s v c) controlValue)
    (earliest_right_urgent target _ _ hc)
  unfold Post at h
  exact h s result after model_run

theorem whole_model_control_quiet (s after : EventProofs.St) (v : QueuedSnapshot s)
    (c t : Nat) (L : Vector Nat 145) (p : parcel) (ready frontier : Int)
    (hL : s.regs.get? Register.iq_len = some L)
    (scan : (parcel_scan 144).run s = .ok p s)
    (early : (parcel_earliest 144 p).run s = .ok ready s)
    (model_run : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) :
    (Out.Functions.control t).run s = .ok () s := by
  by_cases hz : L[144]! = 0
  · exact control_empty s t L hL hz
  · have hn : 0<L[144]! := by omega
    have hn2 : L[144]≠0 := by simpa using hz
    cases hp : p.complete
    · exact control_incomplete s t L p hL hp ready early scan
    · have hs : parcel_scan 144 s = .ok p s := scan
      have he : parcel_earliest 144 p s = .ok ready s := early
      let cv : Int := max ((c:Int)+1) ready
      have control_run : (control_event c).run s = .ok cv s := by
        simp [control_event,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
          pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hL,hn2,hs,he,hp,cv]
      have hcv : 0≤cv := by
        have h := Int.le_max_left ((c:Int)+1) ready
        change 0≤max ((c:Int)+1) ready
        omega
      have htcv : (t:Int)<cv := by
        by_cases h : (t:Int)<cv
        · exact h
        · have urgent : Urgent (t:Int) cv := ⟨hcv,by omega⟩
          have captured := model_captures_control s after v c (t:Int) cv frontier control_run urgent model_run
          unfold Urgent at captured
          omega
      exact actual_control_frontier s c t L p ready cv hL hn scan early control_run hct htcv

theorem actual_earliest_defined (s : EventProofs.St) (complete : MachineInitProofs.Complete s)
    (q : Nat) (p : parcel) : ∃ value, (parcel_earliest q p).run s=.ok value s := by
  obtain ⟨v,hv⟩ := complete Register.iq_last
  refine ⟨((v[q]! : Int)+1)+(p.nops:Int),?_⟩
  simp [parcel_earliest,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hv]
#print axioms actual_earliest_defined

#print axioms whole_model_control_quiet

#print axioms actual_tail_split
#print axioms scan_keeps_urgent
#print axioms earliest_right_urgent
#print axioms model_captures_control

#print axioms control_empty
#print axioms control_incomplete
#print axioms control_before_ready
#print axioms actual_control_frontier
end EventDispatchProofs
