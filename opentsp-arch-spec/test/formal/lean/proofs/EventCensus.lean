-- Actual queued-event census and scheduler frontier, preserving production action order.
-- This does not prove whole-tick stuttering, external callback ordering or global progress.
import Proofs.Events
import Proofs.MachineInit
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventProofs
open Register agt_state
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace EventCensusProofs

def queued_prefix (c : Nat) : SailM Int := do
  let best : Int := (Neg.neg 1)
  let best ← (( do
    let loop_q_lower := 0
    let loop_q_upper := 143
    let mut loop_vars := best
    for q in [loop_q_lower:loop_q_upper:1]i do
      let best := loop_vars
      loop_vars ← do
        match (GetElem?.getElem! (← Out.readReg stream_fetches) q) with
        | .some f => (pure (earliest best (Max.max (c +i 1) f.ready)))
        | none => (pure best)
    (pure loop_vars) ) : SailM Int )
  let best ← (( do
    let loop_q_lower := 0
    let loop_q_upper := 87
    let mut loop_vars_1 := best
    for q in [loop_q_lower:loop_q_upper:1]i do
      let best := loop_vars_1
      loop_vars_1 ← do
        match (GetElem?.getElem! (← Out.readReg local_fetches) q) with
        | .some f => (pure (earliest best (Max.max (c +i 1) f.ready)))
        | none => (pure best)
    (pure loop_vars_1) ) : SailM Int )
  let best ← (( do
    let loop_k_lower := 0
    let loop_k_upper := 127
    let mut loop_vars_2 := best
    for k in [loop_k_lower:loop_k_upper:1]i do
      let best := loop_vars_2
      loop_vars_2 ← do
        match (GetElem?.getElem! (← Out.readReg pending) k) with
        | [] => (pure best)
        | _ => (pure (earliest best ((c +i 1) +i (Int.emod (k -i (c +i 1)) 128))))
    (pure loop_vars_2) ) : SailM Int )
  let best ← (( do
    let loop_k_lower := 0
    let loop_k_upper := 15
    let mut loop_vars_3 := best
    for k in [loop_k_lower:loop_k_upper:1]i do
      let best := loop_vars_3
      loop_vars_3 ← do
        match (GetElem?.getElem! (← Out.readReg execute_ring) k) with
        | [] => (pure best)
        | _ => (pure (earliest best ((c +i 1) +i (Int.emod (k -i (c +i 1)) 16))))
    (pure loop_vars_3) ) : SailM Int )
  let best ← (( do
    let loop_q_lower := 0
    let loop_q_upper := 144
    let mut loop_vars_4 := best
    for q in [loop_q_lower:loop_q_upper:1]i do
      let best := loop_vars_4
      loop_vars_4 ← do
        match (GetElem?.getElem! (← Out.readReg fill_fifo) q) with
        | (seg :: _) => (pure (earliest best (Max.max (c +i 1) seg.ready)))
        | [] => (pure best)
    (pure loop_vars_4) ) : SailM Int )
  pure best

def dispatch_tail (c : Nat) (best : Int) : SailM Int := do
  let best ← (pure (earliest best (← (control_event c))))
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

theorem production_split (c : Nat) : model_next_event c = (do
    let best ← queued_prefix c
    dispatch_tail c best) := by
  simp only [model_next_event,queued_prefix,dispatch_tail,bind_assoc,pure_bind,bind_pure]
  rfl

def Urgent (target e : Int) : Prop := 0 ≤ e ∧ e ≤ target

theorem earliest_keeps_urgent (target a b : Int) (ha : Urgent target a) :
    Urgent target (earliest a b) := by
  unfold earliest Urgent at *
  repeat' first | (split <;> simp_all) | omega | constructor

theorem dispatch_tail_urgent (c : Nat) (target best : Int) (hb : Urgent target best) :
    Post (Urgent target) (dispatch_tail c best) := by
  simp only [dispatch_tail,bind_assoc,pure_bind,bind_pure]
  apply post_bind_any
  intro control
  apply post_forIn [0:143:1]i (Urgent target) (earliest best control) _
    (earliest_keeps_urgent target _ _ hb)
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

theorem model_keeps_queued_urgent (c : Nat) (target queuedValue result : Int)
    (s mid after : St) (hprefix : (queued_prefix c).run s = .ok queuedValue mid)
    (hp : Urgent target queuedValue) (hmodel : (model_next_event c).run s = .ok result after) :
    Urgent target result := by
  rw [production_split] at hmodel
  simp only [EStateM.run,bind,EStateM.bind] at hmodel
  have he : queued_prefix c s = .ok queuedValue mid := hprefix
  simp only [he] at hmodel
  have ht := dispatch_tail_urgent c target queuedValue hp
  unfold Post at ht
  exact ht mid result after hmodel

theorem fixed_state_forIn {β : Type} (range : IntRange) (b : β) (s : St)
    (update : Int → β → β) (body : Int → β → SailM (ForInStep β))
    (hf : ∀ i x, (body i x).run s = .ok (.yield (update i x)) s) :
    (forIn range b body).run s =
      .ok (Id.run (forIn range b (fun i x => pure (.yield (update i x))) : Id β)) s := by
  change (IntRange.forIn' range b (fun i _ x => body i x)).run s =
    .ok (Id.run (IntRange.forIn' range b (fun i _ x => pure (.yield (update i x))) : Id β)) s
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun x i hs =>
      (IntRange.forIn'.loop range (fun i _ x => body i x) x i hs).run s =
        .ok (Id.run (IntRange.forIn'.loop range
          (fun i _ x => pure (.yield (update i x))) x i hs : Id β)) s)
  · intro x i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    conv => rhs; rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    have hb : body i x s = .ok (.yield (update i x)) s := hf i x
    simpa only [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hb,Id.run]
      using ih (update i x)
  · intro x i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    conv => rhs; rw [IntRange.forIn'.loop.eq_def]
    simp [hi,EStateM.run,pure,EStateM.pure,Id.run]

def foldIter {β : Type} (start count : Nat) (update : Nat → β → β) (b : β) : β :=
  match count with
  | 0 => b
  | count+1 => foldIter (start+1) count update (update start b)

def pureBody {β : Type} (n : Nat) (update : Int → β → β)
    (i : Int) (_hi : i ∈ ({stop := (n:Int)-1} : IntRange)) (b : β) :
    Id (ForInStep β) := pure (.yield (update i b))

theorem pure_native_loop {β : Type} (n start count : Nat) (hn : start+count=n)
    (update : Int → β → β) (b : β) :
    Id.run (IntRange.forIn'.loop ({stop := (n:Int)-1} : IntRange)
      (pureBody n update) b (start:Int) (by simp) : Id β) =
      foldIter start count (fun i x => update (i:Int) x) b := by
  induction count generalizing start b with
  | zero =>
    have hnot : ¬(start:Int) ∈ ({stop := (n:Int)-1} : IntRange) := by
      simp only [Membership.mem]
      simp
      omega
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hnot,foldIter,Id.run,pure]
  | succ count ih =>
    have hmem : (start:Int) ∈ ({stop := (n:Int)-1} : IntRange) := by
      simp only [Membership.mem]
      simp
      omega
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hmem,pureBody,foldIter,Id.run,pure,bind]
    have hnext : start+1+count=n := by omega
    have hih := ih (start+1) hnext (update (start:Int) b)
    simpa only [Int.natCast_add,Int.natCast_add_one,Int.toNat_natCast] using hih

theorem earliest_right_urgent (target a b : Int) (hb : Urgent target b) :
    Urgent target (earliest a b) := by
  unfold earliest Urgent at *
  repeat' first | (split <;> simp_all) | omega | constructor

def registerScan (r : Register) (n : Nat)
    (next : RegisterType r → Int → Int → Int) (b : Int) : SailM Int := do
  let mut best := b
  for i in [0:((n:Int)-1):1]i do
    best ← do
      let value ← Out.readReg r
      pure (next value i best)
  pure best

theorem register_scan_run (s : St) (r : Register) (n : Nat)
    (next : RegisterType r → Int → Int → Int) (b : Int) (v : RegisterType r)
    (hr : (Out.readReg r).run s = .ok v s) :
    (registerScan r n next b).run s =
      .ok (foldIter 0 n (fun i best => next v (i:Int) best) b) s := by
  simp only [registerScan,bind_pure,pure_bind]
  have hf : ∀ i x, ((do
      let value ← Out.readReg r
      pure (.yield (next value i x))) : SailM (ForInStep Int)).run s =
      .ok (.yield (next v i x)) s := by
    intro i x
    have hread : Out.readReg r s = .ok v s := hr
    simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hread]
  rw [fixed_state_forIn ({stop := (n:Int)-1} : IntRange) b s
    (fun i x => next v i x) _ hf]
  congr 1
  change Id.run (IntRange.forIn'.loop ({stop := (n:Int)-1} : IntRange)
    (pureBody n (fun i x => next v i x)) b 0 (by simp) : Id Int) = _
  exact pure_native_loop n 0 n (by omega) _ b

def streamCandidate (c : Nat) (v : Vector (Option stream_fetch) 145) (i : Int) : Option Int :=
  match GetElem?.getElem! v i with
  | some f => some (Max.max (c +i 1) f.ready)
  | none => none

def localCandidate (c : Nat) (v : Vector (Option local_fetch) 88) (i : Int) : Option Int :=
  match GetElem?.getElem! v i with
  | some f => some (Max.max (c +i 1) f.ready)
  | none => none

def commitCandidate (c : Nat) (v : Vector (List commit) 128) (i : Int) : Option Int :=
  match GetElem?.getElem! v i with
  | [] => none
  | _ => some ((c +i 1) +i (Int.emod (i -i (c +i 1)) 128))

def executeCandidate (c : Nat) (v : Vector (List exec_element) 16) (i : Int) : Option Int :=
  match GetElem?.getElem! v i with
  | [] => none
  | _ => some ((c +i 1) +i (Int.emod (i -i (c +i 1)) 16))

def fillCandidate (c : Nat) (v : Vector (List fill_seg) 145) (i : Int) : Option Int :=
  match GetElem?.getElem! v i with
  | seg::_ => some (Max.max (c +i 1) seg.ready)
  | [] => none

def offer (b : Int) (candidate : Option Int) : Int :=
  match candidate with | none => b | some e => earliest b e

theorem queued_as_scans (c : Nat) : queued_prefix c = (do
  let b ← registerScan Register.stream_fetches 144
    (fun v i b => offer b (streamCandidate c v i)) (-1)
  let b ← registerScan Register.local_fetches 88
    (fun v i b => offer b (localCandidate c v i)) b
  let b ← registerScan Register.pending 128
    (fun v i b => offer b (commitCandidate c v i)) b
  let b ← registerScan Register.execute_ring 16
    (fun v i b => offer b (executeCandidate c v i)) b
  registerScan Register.fill_fifo 145
    (fun v i b => offer b (fillCandidate c v i)) b) := by
  have hs : ∀ (i b : Int),
      (do let v ← Out.readReg Register.stream_fetches
          match v[i]! with
        | some f => pure (ForInStep.yield (earliest b (Max.max (c +i 1) f.ready)))
        | none => pure (ForInStep.yield b)) =
      (do let v ← Out.readReg Register.stream_fetches
          pure (ForInStep.yield (offer b (streamCandidate c v i)))) := by
    intro i b
    apply congrArg (fun f => Out.readReg Register.stream_fetches >>= f)
    funext v
    cases h : v[i]! <;> simp [streamCandidate,localCandidate,commitCandidate,executeCandidate,fillCandidate,offer,h]
  have hl : ∀ (i b : Int),
      (do let v ← Out.readReg Register.local_fetches
          match v[i]! with
        | some f => pure (ForInStep.yield (earliest b (Max.max (c +i 1) f.ready)))
        | none => pure (ForInStep.yield b)) =
      (do let v ← Out.readReg Register.local_fetches
          pure (ForInStep.yield (offer b (localCandidate c v i)))) := by
    intro i b
    apply congrArg (fun f => Out.readReg Register.local_fetches >>= f)
    funext v
    cases h : v[i]! <;> simp [streamCandidate,localCandidate,commitCandidate,executeCandidate,fillCandidate,offer,h]
  have hp : ∀ (i b : Int),
      (do let v ← Out.readReg Register.pending
          match v[i]! with
        | [] => pure (ForInStep.yield b)
        | _ => pure (ForInStep.yield (earliest b ((c +i 1) +i (Int.emod (i -i (c +i 1)) 128))))) =
      (do let v ← Out.readReg Register.pending
          pure (ForInStep.yield (offer b (commitCandidate c v i)))) := by
    intro i b
    apply congrArg (fun f => Out.readReg Register.pending >>= f)
    funext v
    cases h : v[i]! <;> simp [streamCandidate,localCandidate,commitCandidate,executeCandidate,fillCandidate,offer,h]
  have he : ∀ (i b : Int),
      (do let v ← Out.readReg Register.execute_ring
          match v[i]! with
        | [] => pure (ForInStep.yield b)
        | _ => pure (ForInStep.yield (earliest b ((c +i 1) +i (Int.emod (i -i (c +i 1)) 16))))) =
      (do let v ← Out.readReg Register.execute_ring
          pure (ForInStep.yield (offer b (executeCandidate c v i)))) := by
    intro i b
    apply congrArg (fun f => Out.readReg Register.execute_ring >>= f)
    funext v
    cases h : v[i]! <;> simp [streamCandidate,localCandidate,commitCandidate,executeCandidate,fillCandidate,offer,h]
  have hf : ∀ (i b : Int),
      (do let v ← Out.readReg Register.fill_fifo
          match v[i]! with
        | seg::_ => pure (ForInStep.yield (earliest b (Max.max (c +i 1) seg.ready)))
        | [] => pure (ForInStep.yield b)) =
      (do let v ← Out.readReg Register.fill_fifo
          pure (ForInStep.yield (offer b (fillCandidate c v i)))) := by
    intro i b
    apply congrArg (fun f => Out.readReg Register.fill_fifo >>= f)
    funext v
    cases h : v[i]! <;> simp [streamCandidate,localCandidate,commitCandidate,executeCandidate,fillCandidate,offer,h]
  simp only [queued_prefix,registerScan,bind_assoc,pure_bind,bind_pure,hs,hl,hp,he,hf]
  rfl

structure QueuedSnapshot (s : St) where
  streams : Vector (Option stream_fetch) 145
  locals : Vector (Option local_fetch) 88
  commits : Vector (List commit) 128
  executes : Vector (List exec_element) 16
  fills : Vector (List fill_seg) 145
  stream_read : (Out.readReg Register.stream_fetches).run s = .ok streams s
  local_read : (Out.readReg Register.local_fetches).run s = .ok locals s
  commit_read : (Out.readReg Register.pending).run s = .ok commits s
  execute_read : (Out.readReg Register.execute_ring).run s = .ok executes s
  fill_read : (Out.readReg Register.fill_fifo).run s = .ok fills s

def queueFold (n : Nat) (candidate : Int → Option Int) (b : Int) : Int :=
  foldIter 0 n (fun i best => offer best (candidate (i:Int))) b

def snapshotResult (s : St) (v : QueuedSnapshot s) (c : Nat) : Int :=
  let b := queueFold 144 (streamCandidate c v.streams) (-1)
  let b := queueFold 88 (localCandidate c v.locals) b
  let b := queueFold 128 (commitCandidate c v.commits) b
  let b := queueFold 16 (executeCandidate c v.executes) b
  queueFold 145 (fillCandidate c v.fills) b

theorem queued_snapshot_run (s : St) (v : QueuedSnapshot s) (c : Nat) :
    (queued_prefix c).run s = .ok (snapshotResult s v c) s := by
  rw [queued_as_scans]
  have hs := fun b => register_scan_run s Register.stream_fetches 144
    (fun v i b => offer b (streamCandidate c v i)) b v.streams v.stream_read
  have hl := fun b => register_scan_run s Register.local_fetches 88
    (fun v i b => offer b (localCandidate c v i)) b v.locals v.local_read
  have hp := fun b => register_scan_run s Register.pending 128
    (fun v i b => offer b (commitCandidate c v i)) b v.commits v.commit_read
  have he := fun b => register_scan_run s Register.execute_ring 16
    (fun v i b => offer b (executeCandidate c v i)) b v.executes v.execute_read
  have hf := fun b => register_scan_run s Register.fill_fifo 145
    (fun v i b => offer b (fillCandidate c v i)) b v.fills v.fill_read
  simp only [EStateM.run] at hs hl hp he hf ⊢
  simp only [bind,EStateM.bind,hs,hl,hp,he,hf]
  rfl

theorem offer_urgent (target b : Int) (candidate : Option Int) (hb : Urgent target b) :
    Urgent target (offer b candidate) := by
  cases candidate with
  | none => exact hb
  | some e => exact earliest_keeps_urgent target b e hb

theorem offer_fold_urgent (start count : Nat) (candidate : Nat → Option Int)
    (target b : Int) (hb : Urgent target b) :
    Urgent target (foldIter start count (fun i x => offer x (candidate i)) b) := by
  induction count generalizing start b with
  | zero => exact hb
  | succ count ih => exact ih (start+1) _ (offer_urgent target b _ hb)

theorem offer_fold_capture (start count i : Nat) (candidate : Nat → Option Int)
    (e b : Int) (hi : start ≤ i ∧ i < start+count) (he : 0 ≤ e)
    (hc : candidate i = some e) :
    Urgent e (foldIter start count (fun i x => offer x (candidate i)) b) := by
  induction count generalizing start b with
  | zero => omega
  | succ count ih =>
    simp only [foldIter]
    by_cases hs : i=start
    · subst i
      apply offer_fold_urgent
      rw [hc]
      exact earliest_right_urgent e b e ⟨he,by omega⟩
    · apply ih (start+1) _
      omega

theorem queue_fold_urgent (n : Nat) (candidate : Int → Option Int) (e b : Int)
    (hb : Urgent e b) : Urgent e (queueFold n candidate b) := by
  exact offer_fold_urgent 0 n (fun i => candidate (i:Int)) e b hb

theorem queue_fold_capture (n i : Nat) (candidate : Int → Option Int) (e b : Int)
    (hi : i<n) (he : 0 ≤ e) (hc : candidate (i:Int)=some e) :
    Urgent e (queueFold n candidate b) := by
  exact offer_fold_capture 0 n i (fun i => candidate (i:Int)) e b (by omega) he hc

def QueuedEvent (s : St) (v : QueuedSnapshot s) (c : Nat) (e : Int) : Prop :=
  (∃ i : Nat, i<144 ∧ streamCandidate c v.streams (i:Int)=some e) ∨
  (∃ i : Nat, i<88 ∧ localCandidate c v.locals (i:Int)=some e) ∨
  (∃ i : Nat, i<128 ∧ commitCandidate c v.commits (i:Int)=some e) ∨
  (∃ i : Nat, i<16 ∧ executeCandidate c v.executes (i:Int)=some e) ∨
  (∃ i : Nat, i<145 ∧ fillCandidate c v.fills (i:Int)=some e)

theorem snapshot_captures_event (s : St) (v : QueuedSnapshot s) (c : Nat) (e : Int)
    (he : 0 ≤ e) (hc : QueuedEvent s v c e) : Urgent e (snapshotResult s v c) := by
  unfold snapshotResult
  rcases hc with hs|hl|hp|hx|hf
  · obtain ⟨i,hi,hc⟩ := hs
    apply queue_fold_urgent
    apply queue_fold_urgent
    apply queue_fold_urgent
    apply queue_fold_urgent
    exact queue_fold_capture 144 i _ e (-1) hi he hc
  · obtain ⟨i,hi,hc⟩ := hl
    apply queue_fold_urgent
    apply queue_fold_urgent
    apply queue_fold_urgent
    exact queue_fold_capture 88 i _ e _ hi he hc
  · obtain ⟨i,hi,hc⟩ := hp
    apply queue_fold_urgent
    apply queue_fold_urgent
    exact queue_fold_capture 128 i _ e _ hi he hc
  · obtain ⟨i,hi,hc⟩ := hx
    apply queue_fold_urgent
    exact queue_fold_capture 16 i _ e _ hi he hc
  · obtain ⟨i,hi,hc⟩ := hf
    exact queue_fold_capture 145 i _ e _ hi he hc

theorem model_captures_queued_event (s after : St) (v : QueuedSnapshot s)
    (c : Nat) (e result : Int) (he : 0 ≤ e) (hc : QueuedEvent s v c e)
    (hr : (model_next_event c).run s = .ok result after) : Urgent e result := by
  exact model_keeps_queued_urgent c e (snapshotResult s v c) result s s after
    (queued_snapshot_run s v c) (snapshot_captures_event s v c e he hc) hr

theorem complete_has_snapshot (s : St) (hc : MachineInitProofs.Complete s) :
    Nonempty (QueuedSnapshot s) := by
  obtain ⟨sf,hs⟩ := MachineInitProofs.actual_read_defined s hc Register.stream_fetches
  obtain ⟨lf,hl⟩ := MachineInitProofs.actual_read_defined s hc Register.local_fetches
  obtain ⟨pc,hp⟩ := MachineInitProofs.actual_read_defined s hc Register.pending
  obtain ⟨xe,hx⟩ := MachineInitProofs.actual_read_defined s hc Register.execute_ring
  obtain ⟨ff,hf⟩ := MachineInitProofs.actual_read_defined s hc Register.fill_fifo
  exact ⟨⟨sf,lf,pc,xe,ff,hs,hl,hp,hx,hf⟩⟩

theorem stream_candidate_future (c : Nat) (v : Vector (Option stream_fetch) 145)
    (i e : Int) (hc : streamCandidate c v i = some e) : (c:Int) < e := by
  unfold streamCandidate at hc
  split at hc <;> simp_all
  omega

theorem local_candidate_future (c : Nat) (v : Vector (Option local_fetch) 88)
    (i e : Int) (hc : localCandidate c v i = some e) : (c:Int) < e := by
  unfold localCandidate at hc
  split at hc <;> simp_all
  omega

theorem commit_candidate_future (c : Nat) (v : Vector (List commit) 128)
    (i e : Int) (hc : commitCandidate c v i = some e) : (c:Int) < e := by
  unfold commitCandidate at hc
  split at hc <;> simp_all
  change (c:Int)+1+(i-((c:Int)+1)) % 128 = e at hc
  have hn := Int.emod_nonneg (i - ((c:Int)+1)) (by decide : (128:Int)≠0)
  omega

theorem execute_candidate_future (c : Nat) (v : Vector (List exec_element) 16)
    (i e : Int) (hc : executeCandidate c v i = some e) : (c:Int) < e := by
  unfold executeCandidate at hc
  split at hc <;> simp_all
  change (c:Int)+1+(i-((c:Int)+1)) % 16 = e at hc
  have hn := Int.emod_nonneg (i - ((c:Int)+1)) (by decide : (16:Int)≠0)
  omega

theorem fill_candidate_future (c : Nat) (v : Vector (List fill_seg) 145)
    (i e : Int) (hc : fillCandidate c v i = some e) : (c:Int) < e := by
  unfold fillCandidate at hc
  split at hc <;> simp_all
  omega

theorem queued_event_future (s : St) (v : QueuedSnapshot s) (c : Nat) (e : Int)
    (hc : QueuedEvent s v c e) : (c:Int)<e := by
  rcases hc with ⟨i,hi,h⟩|⟨i,hi,h⟩|⟨i,hi,h⟩|⟨i,hi,h⟩|⟨i,hi,h⟩
  · exact stream_candidate_future c v.streams _ e h
  · exact local_candidate_future c v.locals _ e h
  · exact commit_candidate_future c v.commits _ e h
  · exact execute_candidate_future c v.executes _ e h
  · exact fill_candidate_future c v.fills _ e h

theorem no_queued_event_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c : Nat) (frontier target : Int)
    (hr : (model_next_event c).run s = .ok frontier after) (ht : target<frontier) :
    ∀ e, e≤target → ¬QueuedEvent s v c e := by
  intro e he hc
  have hf := queued_event_future s v c e hc
  have bound := model_captures_queued_event s after v c e frontier (by omega) hc hr
  unfold Urgent at bound
  omega

theorem next_ring128_le (c t : Nat) (ht : c<t) :
    (c:Int)+1+(((t % 128 : Nat):Int)-((c:Int)+1)) % 128 ≤ (t:Int) := by
  omega

theorem next_ring16_le (c t : Nat) (ht : c<t) :
    (c:Int)+1+(((t % 16 : Nat):Int)-((c:Int)+1)) % 16 ≤ (t:Int) := by
  omega

theorem stream_not_due_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t q : Nat) (frontier : Int) (f : stream_fetch)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) (hq : q<144)
    (hcell : v.streams[(q:Int)]! = some f) : t<f.ready := by
  apply Classical.byContradiction
  intro hn
  have hf : f.ready≤t := by omega
  let e : Int := max ((c:Int)+1) (f.ready:Int)
  have he : e≤(t:Int) := by dsimp [e]; omega
  have hc : streamCandidate c v.streams (q:Int)=some e := by
    simp [streamCandidate,hcell,e]
  have event : QueuedEvent s v c e := by
    exact (Or.inl) ⟨q,hq,hc⟩
  exact no_queued_event_before_frontier s after v c frontier (t:Int) hr ht e he event

theorem local_not_due_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t q : Nat) (frontier : Int) (f : local_fetch)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) (hq : q<88)
    (hcell : v.locals[(q:Int)]! = some f) : t<f.ready := by
  apply Classical.byContradiction
  intro hn
  have hf : f.ready≤t := by omega
  let e : Int := max ((c:Int)+1) (f.ready:Int)
  have he : e≤(t:Int) := by dsimp [e]; omega
  have hc : localCandidate c v.locals (q:Int)=some e := by
    simp [localCandidate,hcell,e]
  have event : QueuedEvent s v c e := by
    exact (fun h => Or.inr (Or.inl h)) ⟨q,hq,hc⟩
  exact no_queued_event_before_frontier s after v c frontier (t:Int) hr ht e he event

theorem commit_ring_empty_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) :
    v.commits[((t % 128 : Nat):Int)]! = [] := by
  apply Classical.byContradiction
  intro hn
  let e : Int := (c:Int)+1+(((t % 128 : Nat):Int)-((c:Int)+1)) % 128
  have he : e≤(t:Int) := next_ring128_le c t hct
  have hc : commitCandidate c v.commits ((t % 128 : Nat):Int)=some e := by
    unfold commitCandidate
    split
    · contradiction
    · rfl
  have event : QueuedEvent s v c e := by
    have h : ∃ i : Nat, i<128 ∧ commitCandidate c v.commits (i:Int)=some e :=
      ⟨t % 128,Nat.mod_lt _ (by decide),hc⟩
    exact Or.inr (Or.inr (Or.inl h))
  exact no_queued_event_before_frontier s after v c frontier (t:Int) hr ht e he event

theorem execute_ring_empty_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t : Nat) (frontier : Int)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) :
    v.executes[((t % 16 : Nat):Int)]! = [] := by
  apply Classical.byContradiction
  intro hn
  let e : Int := (c:Int)+1+(((t % 16 : Nat):Int)-((c:Int)+1)) % 16
  have he : e≤(t:Int) := next_ring16_le c t hct
  have hc : executeCandidate c v.executes ((t % 16 : Nat):Int)=some e := by
    unfold executeCandidate
    split
    · contradiction
    · rfl
  have event : QueuedEvent s v c e := by
    have h : ∃ i : Nat, i<16 ∧ executeCandidate c v.executes (i:Int)=some e :=
      ⟨t % 16,Nat.mod_lt _ (by decide),hc⟩
    exact Or.inr (Or.inr (Or.inr (Or.inl h)))
  exact no_queued_event_before_frontier s after v c frontier (t:Int) hr ht e he event

theorem fill_not_due_before_frontier (s after : St) (v : QueuedSnapshot s)
    (c t q : Nat) (frontier : Int) (seg : fill_seg) (rest : List fill_seg)
    (hr : (model_next_event c).run s = .ok frontier after)
    (hct : c<t) (ht : (t:Int)<frontier) (hq : q<145)
    (hcell : v.fills[(q:Int)]! = seg::rest) : t<seg.ready := by
  apply Classical.byContradiction
  intro hn
  have hf : seg.ready≤t := by omega
  let e : Int := max ((c:Int)+1) (seg.ready:Int)
  have he : e≤(t:Int) := by dsimp [e]; omega
  have hc : fillCandidate c v.fills (q:Int)=some e := by simp [fillCandidate,hcell,e]
  have event : QueuedEvent s v c e := Or.inr (Or.inr (Or.inr (Or.inr ⟨q,hq,hc⟩)))
  exact no_queued_event_before_frontier s after v c frontier (t:Int) hr ht e he event

#print axioms stream_not_due_before_frontier
#print axioms local_not_due_before_frontier
#print axioms commit_ring_empty_before_frontier
#print axioms execute_ring_empty_before_frontier
#print axioms fill_not_due_before_frontier

#print axioms next_ring128_le
#print axioms next_ring16_le

#print axioms complete_has_snapshot
#print axioms stream_candidate_future
#print axioms local_candidate_future
#print axioms commit_candidate_future
#print axioms execute_candidate_future
#print axioms fill_candidate_future
#print axioms queued_event_future
#print axioms no_queued_event_before_frontier

#print axioms offer_urgent
#print axioms offer_fold_urgent
#print axioms offer_fold_capture
#print axioms queue_fold_urgent
#print axioms queue_fold_capture
#print axioms snapshot_captures_event
#print axioms model_captures_queued_event

#print axioms queued_snapshot_run

#print axioms queued_as_scans

#print axioms register_scan_run

#print axioms earliest_right_urgent

#print axioms pure_native_loop

#print axioms production_split
#print axioms earliest_keeps_urgent
#print axioms dispatch_tail_urgent
#print axioms model_keeps_queued_urgent
#print axioms fixed_state_forIn
end EventCensusProofs
