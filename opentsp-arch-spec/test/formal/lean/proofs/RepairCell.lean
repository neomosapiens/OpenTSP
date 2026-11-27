import Proofs.HistoryContent
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
open Out.Register Out.ecc_class
open Proofs.TagCoverage Proofs.TaggedCheck Proofs.OrderedImage Proofs.OrderedDelivery
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace Proofs.RepairCell
abbrev State := ResourceFrameProofs.St

/-- Concrete C read values, distinct from read-only state frames. The caller
    proves initialized/live unit domains; tags retain their arbitrary Int value. -/
structure ValuePorts : Prop where
 unit : ∀ q slot (s : State), Proofs.StorageProduct.Good iq_cap s →
   Proofs.Storage.Live iq_cap s q slot →
   (iq_unit_read q slot).run s=.ok (units iq_cap s q slot) s
 tag : ∀ q slot (s : State), q<145 → slot<iq_cap →
   (iq_tag_read q slot).run s=.ok (tags iq_cap s q slot) s

def cell (q : Nat) (want : Int) (m : BitVec 32) (i : Int) : SailM Unit := do
 if (((← (iq_tag_read q
              (ring ((GetElem?.getElem! (← Out.readReg iq_head) q) +i i)))) == want) : Bool)
     then
       (iq_unit_write q
         (ring ((GetElem?.getElem! (← Out.readReg iq_head) q) +i i))
         ((← (iq_peek q i)) ^^^ m))
     else (pure ())
def wordCheckWithCell (step : Nat → Int → BitVec 32 → Int → SailM Unit) (q : Nat) (c : Nat) (w : Int) : SailM Unit := do
  if ((w ≥b 0) : Bool)
  then
    (do
      let wn : Nat := (Max.max w 0)
      let e ← do (iq_word_ov q wn)
      if ((e != 0b0000000000#10) : Bool)
      then
        (do
          let (k, s) := (ecc_classify e)
          match k with
          | .ECC_OK => (pure ())
          | .ECC_UNCORRECTABLE => (ras_ecc_event c q 2 true false 0 (Int.emod wn 16777216) 0 s)
          | .ECC_CORRECTED =>
            (do
              (ras_ecc_event c q 2 false false 0 (Int.emod wn 16777216) 0 s)
              let (_, fixed, _) := (ecc_correct (BitVec.zero 128) e)
              let loop_k_lower := 0
              let loop_k_upper := 3
              let mut loop_vars := ()
              for k in [loop_k_lower:loop_k_upper:1]i do
                let () := loop_vars
                loop_vars ← do
                  let m : (BitVec 32) := (BitVec.slice fixed (32 *i k) 32)
                  if ((m != 0x00000000#32) : Bool)
                  then
                    (do
                      let want := ((4 *i wn) +i k)
                      let loop_i_lower := 0
                      let loop_i_upper ← do
                        (pure ((GetElem?.getElem! (← Out.readReg iq_len) q) -i 1))
                      let mut loop_vars_1 := ()
                      for i in [loop_i_lower:loop_i_upper:1]i do
                        let () := loop_vars_1
                        loop_vars_1 ← do
                          (step q want m i)
                      (pure loop_vars_1))
                  else (pure ())
              (pure loop_vars)
              (iq_word_ov_set q wn 0b0000000000#10)))
      else (pure ()))
  else (pure ())

/-- Type quantifiers: units : Nat, c : Nat, q : Nat, 0 ≤ q ∧ q ≤ 144, 0 ≤ c, 0 ≤ units -/
theorem actual_word_cell_factor (q c : Nat) (w : Int) :
 iq_word_check q c w=wordCheckWithCell cell q c w := by
 simp only [iq_word_check,wordCheckWithCell,cell,bind_assoc,pure_bind,bind_pure,Proofs.OrderedImage.bind_if]
 rfl

theorem actual_cell_exact (reads : ValuePorts) (writes : Proofs.Initialized.WritePorts iq_cap)
 (A : Nat → Prop) (H L : Vector Nat 145) (s : State) (keep : Keep A H L s)
 (view : CView) (rep : Represents iq_cap view s) (q : Nat) (want : Int) (m : BitVec 32) (i : Int)
 (hq : q<145) (hi : 0≤i) (inside : i<(L[q]! : Int)) :
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]! + i.toNat)
 ∃ after,(cell q want m i).run s=.ok () after ∧ Keep A H L after ∧
 Represents iq_cap (if view.tag q slot=want
   then cUnitWrite view q slot (view.unit q slot ^^^ m) else view) after := by
 dsimp only
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]! + i.toNat)
 have bounds:=Proofs.HistoryComposition.captured_bounds A H L s keep q hq
 have hslot : slot<iq_cap := by
  dsimp [slot]
  rw [Proofs.Capacity.ring_spec]
  split <;> omega
 have live : Proofs.Storage.Live iq_cap s q slot :=
  ⟨H,L,keep.1.2.1,keep.1.2.2,hq,bounds.1,bounds.2,i.toNat,by omega,rfl⟩
 obtain ⟨old,oldrep,H0,L0,hH0,hL0,bound⟩:=keep.1.1
 have eh : H0=H := by rw [keep.1.2.1] at hH0;cases hH0;rfl
 have el : L0=L := by rw [keep.1.2.2] at hL0;cases hL0;rfl
 subst H0;subst L0
 have initialized : view.initialized q slot := (rep.2 q slot hq hslot).1.mpr
   ((bound q hq).2.2 i.toNat (by omega))
 have ur : iq_unit_read q slot s=.ok (view.unit q slot) s := by
  have r:=reads.unit q slot s keep.1.1 live
  simpa [EStateM.run,represented_unit iq_cap view s rep q slot hq hslot initialized] using r
 have tr : iq_tag_read q slot s=.ok (view.tag q slot) s := by
  have r:=reads.tag q slot s hq hslot
  simpa [EStateM.run,represented_tag iq_cap view s rep q slot hq hslot] using r
 have hr : Out.readReg Register.iq_head s=.ok H s := by
  simp [Out.readReg,PreSail.readReg,keep.1.2.1,EStateM.run,bind,EStateM.bind,
   pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
 have cast : ((H[q]! : Int) + i).toNat=H[q]! + i.toNat := by omega
 have imax : max i 0=i := by omega
 have ringNat : ring (H[q]! + i.toNat)=slot := by rfl
 have ringInt : ring ((H[q]! : Int) + i).toNat=slot := by rw [cast];exact ringNat
 have peekCast : ((H[q]! : Int) + (i.toNat : Int)).toNat=H[q]! + i.toNat := by omega
 have peek : iq_peek q i.toNat s=.ok (view.unit q slot) s := by
  simp only [iq_peek,EStateM.run,bind,EStateM.bind,hr,peekCast,ringNat,ur]

 by_cases eq : view.tag q slot=want
 · let value:=view.unit q slot ^^^ m
   let after:=writeCell s (145 * iq_cap) 0 (q * iq_cap+slot) value.toNat
   have wr:=writes.unit q slot value s hq hslot
   have good:=unit_write_safe A (H:=H) (L:=L) q slot value hq hslot writes s keep
   have good' : Keep A H L after := by simpa [wr,after,value] using good
   refine ⟨after,?_,good',?_⟩
   · have wr' : iq_unit_write q slot (view.unit q slot ^^^ m) s=.ok () after := wr
     simpa [cell,EStateM.run,bind,EStateM.bind,hr,ringInt,tr,eq,peek] using wr'
   · rw [if_pos eq]
     exact unit_represents iq_cap view s rep q slot value hq hslot
 · refine ⟨s,?_,keep,?_⟩
   · simp [cell,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hr,ringInt,tr,eq]
   · rw [if_neg eq]
     exact rep


def updateView (q : Nat) (want : Int) (m : BitVec 32) (H : Vector Nat 145)
 (view : CView) (i : Int) : CView :=
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]! + i.toNat)
 if view.tag q slot=want then cUnitWrite view q slot (view.unit q slot ^^^ m) else view

def cellsFrom (q : Nat) (want : Int) (m : BitVec 32) (i : Int) : Nat → SailM Unit
 | 0 => pure ()
 | n+1 => cell q want m i >>= fun _=>cellsFrom q want m (i+1) n

def viewFrom (q : Nat) (want : Int) (m : BitVec 32) (H : Vector Nat 145)
 (i : Int) : Nat → CView → CView
 | 0,view => view
 | n+1,view => viewFrom q want m H (i+1) n (updateView q want m H view i)

/-- Exact computed unit/tag history over every actual bounded repair-cell step.
    The input view is retained and transformed by the real tag comparison/XOR,
    not replaced by a fresh existential witness after a repair. -/
theorem actual_cells_content (reads : ValuePorts) (writes : Proofs.Initialized.WritePorts iq_cap)
 (A : Nat → Prop) (H L : Vector Nat 145) (q : Nat) (want : Int) (m : BitVec 32) (hq : q<145)
 (n : Nat) (i : Int) (nonneg : 0≤i) (endbound : i+(n:Int)≤(L[q]! : Int))
 (s : State) (keep : Keep A H L s) (view : CView) (rep : Represents iq_cap view s) :
 ∃ after,(cellsFrom q want m i n).run s=.ok () after ∧ Keep A H L after ∧
   Represents iq_cap (viewFrom q want m H i n view) after := by
 induction n generalizing i s view with
 | zero => exact ⟨s,rfl,keep,rep⟩
 | succ n ih =>
   obtain ⟨middle,run,good,represented⟩:=actual_cell_exact reads writes A H L s keep view rep q want m i hq nonneg (by omega)
   have r2 : Represents iq_cap (updateView q want m H view i) middle := by
     simpa only [updateView] using represented
   obtain ⟨after,rest,kept,result⟩:=ih (i+1) (by omega) (by omega) middle good (updateView q want m H view i) r2
   refine ⟨after,?_,kept,result⟩
   change cell q want m i s=.ok () middle at run
   change cellsFrom q want m (i+1) n middle=.ok () after at rest
   simp only [cellsFrom,EStateM.run,bind,EStateM.bind,run,rest]

/-- Actual inclusive Sail range, retaining the effectful cell operation. -/
theorem actual_range_cells (q : Nat) (want : Int) (m : BitVec 32) (range : IntRange)
 (hstep : range.step=1) (i : Int) (hs : (i-range.start)%range.step=0) (start : range.start≤i) :
 IntRange.forIn'.loop (m:=SailM) range
   (fun k _ (_ : Unit)=>cell q want m k >>= fun u=>pure (.yield u)) () i hs =
 cellsFrom q want m i (range.stop - i+1).toNat := by
 induction (),i,hs using IntRange.forIn'.loop.induct range with
 | case1 u i hs hi ih =>
   cases u
   have hmem:=hi
   simp [Membership.mem,hstep] at hmem
   rw [IntRange.forIn'.loop.eq_1]
   simp only [hi,dite_true,bind_assoc,pure_bind]
   have hn : (range.stop - i+1).toNat=(range.stop-(i+range.step)+1).toNat+1 := by omega
   rw [hn]
   simp only [cellsFrom]
   congr 1
   funext unit
   cases unit
   simpa only [hstep] using ih () (by omega)
 | case2 u i hs hi =>
   cases u
   have hmem:=hi
   simp [Membership.mem,hstep] at hmem
   have hn : (range.stop - i+1).toNat=0 := by omega
   rw [IntRange.forIn'.loop.eq_1]
   simp [hi,hn,cellsFrom]



theorem actual_range_cells_start (q : Nat) (want : Int) (m : BitVec 32) (range : IntRange)
 (hstep : range.step=1) :
 IntRange.forIn' (m:=SailM) range ()
   (fun k _ (_ : Unit)=>cell q want m k >>= fun u=>pure (.yield u)) =
 cellsFrom q want m range.start (range.stop-range.start+1).toNat := by
 unfold IntRange.forIn'
 exact actual_range_cells q want m range hstep range.start (by simp) (by omega)


theorem cell_yield_map (q : Nat) (want : Int) (m : BitVec 32) (i : Int) :
 (ForInStep.yield <$> cell q want m i)=(cell q want m i >>= fun u=>pure (.yield u)) := by rfl

theorem actual_range_cells_map (q : Nat) (want : Int) (m : BitVec 32) (range : IntRange)
 (hstep : range.step=1) :
 IntRange.forIn' (m:=SailM) range ()
   (fun k _ (_ : Unit)=>ForInStep.yield <$> cell q want m k) =
 cellsFrom q want m range.start (range.stop-range.start+1).toNat := by
 simp only [cell_yield_map]
 exact actual_range_cells_start q want m range hstep

def laneActual (q wn : Nat) (fixed : BitVec 128) (k : Int) : SailM Unit := do
 let m : (BitVec 32) := (BitVec.slice fixed (32 *i k) 32)
 if ((m != 0x00000000#32) : Bool)
 then
   (do
     let want := ((4 *i wn) +i k)
     let loop_i_lower := 0
     let loop_i_upper ← do
       (pure ((GetElem?.getElem! (← Out.readReg iq_len) q) -i 1))
     let mut loop_vars_1 := ()
     for i in [loop_i_lower:loop_i_upper:1]i do
       let () := loop_vars_1
       loop_vars_1 ← do
         (cell q want m i)
     (pure loop_vars_1))
 else (pure ())
def wordCheckWithLane (lane : Nat → Nat → BitVec 128 → Int → SailM Unit) (q : Nat) (c : Nat) (w : Int) : SailM Unit := do
  if ((w ≥b 0) : Bool)
  then
    (do
      let wn : Nat := (Max.max w 0)
      let e ← do (iq_word_ov q wn)
      if ((e != 0b0000000000#10) : Bool)
      then
        (do
          let (k, s) := (ecc_classify e)
          match k with
          | .ECC_OK => (pure ())
          | .ECC_UNCORRECTABLE => (ras_ecc_event c q 2 true false 0 (Int.emod wn 16777216) 0 s)
          | .ECC_CORRECTED =>
            (do
              (ras_ecc_event c q 2 false false 0 (Int.emod wn 16777216) 0 s)
              let (_, fixed, _) := (ecc_correct (BitVec.zero 128) e)
              let loop_k_lower := 0
              let loop_k_upper := 3
              let mut loop_vars := ()
              for k in [loop_k_lower:loop_k_upper:1]i do
                let () := loop_vars
                loop_vars ← do
                  (lane q wn fixed k)
              (pure loop_vars)
              (iq_word_ov_set q wn 0b0000000000#10)))
      else (pure ()))
  else (pure ())

/-- Type quantifiers: units : Nat, c : Nat, q : Nat, 0 ≤ q ∧ q ≤ 144, 0 ≤ c, 0 ≤ units -/
theorem actual_word_lane_factor (q c : Nat) (w : Int) :
 iq_word_check q c w=wordCheckWithLane laneActual q c w := by
 simp only [iq_word_check,wordCheckWithLane,laneActual,cell,bind_assoc,pure_bind,bind_pure,Proofs.OrderedImage.bind_if]
 rfl

def laneView (q wn : Nat) (fixed : BitVec 128) (k : Int) (H L : Vector Nat 145)
 (view : CView) : CView :=
 let mask:=BitVec.slice fixed (32 * k).toNat 32
 if mask=0#32 then view else viewFrom q (4 * (wn:Int) + k) mask H 0 L[q]! view

theorem actual_lane_content (reads : ValuePorts) (writes : Proofs.Initialized.WritePorts iq_cap)
 (A : Nat → Prop) (H L : Vector Nat 145) (s : State) (keep : Keep A H L s)
 (view : CView) (rep : Represents iq_cap view s) (q wn : Nat) (fixed : BitVec 128) (k : Int)
 (hq : q<145) :
 ∃ after,(laneActual q wn fixed k).run s=.ok () after ∧ Keep A H L after ∧
 Represents iq_cap (laneView q wn fixed k H L view) after := by
 let mask:=BitVec.slice fixed (32 * k).toNat 32
 have hr : Out.readReg Register.iq_len s=.ok L s := by
  simp [Out.readReg,PreSail.readReg,keep.1.2.2,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
 by_cases zero : mask=0#32
 · refine ⟨s,?_,keep,?_⟩
   · simp [laneActual,mask,zero,EStateM.run,pure,EStateM.pure]
   · simpa [laneView,mask,zero] using rep
 · obtain ⟨after,run,good,represented⟩:=actual_cells_content reads writes A H L q (4 * (wn:Int) + k)
     mask hq L[q]! 0 (by omega) (by omega) s keep view rep
   refine ⟨after,?_,good,?_⟩
   · have laneEq : laneActual q wn fixed k=
       (do let lengths ← Out.readReg Register.iq_len
           cellsFrom q (4 * (wn:Int) + k) mask 0 lengths[q]!) := by
       simp [laneActual,mask,zero,ForIn.forIn,ForIn'.forIn',instForInOfForIn',
         bind_assoc,pure_bind,bind_pure,actual_range_cells_map,actual_range_cells_start]
     rw [laneEq]
     simpa [EStateM.run,bind,EStateM.bind,hr] using run
   · simpa [laneView,mask,zero] using represented


def lanesFrom (q wn : Nat) (fixed : BitVec 128) (i : Int) : Nat → SailM Unit
 | 0 => pure ()
 | n+1 => laneActual q wn fixed i >>= fun _=>lanesFrom q wn fixed (i+1) n

def lanesView (q wn : Nat) (fixed : BitVec 128) (H L : Vector Nat 145)
 (i : Int) : Nat → CView → CView
 | 0,view => view
 | n+1,view => lanesView q wn fixed H L (i+1) n (laneView q wn fixed i H L view)

theorem actual_lanes_content (reads : ValuePorts) (writes : Proofs.Initialized.WritePorts iq_cap)
 (A : Nat → Prop) (H L : Vector Nat 145) (q wn : Nat) (fixed : BitVec 128) (hq : q<145)
 (n : Nat) (i : Int) (s : State) (keep : Keep A H L s) (view : CView) (rep : Represents iq_cap view s) :
 ∃ after,(lanesFrom q wn fixed i n).run s=.ok () after ∧ Keep A H L after ∧
   Represents iq_cap (lanesView q wn fixed H L i n view) after := by
 induction n generalizing i s view with
 | zero => exact ⟨s,rfl,keep,rep⟩
 | succ n ih =>
   obtain ⟨middle,run,good,represented⟩:=actual_lane_content reads writes A H L s keep view rep q wn fixed i hq
   obtain ⟨after,rest,kept,result⟩:=ih (i+1) middle good (laneView q wn fixed i H L view) represented
   refine ⟨after,?_,kept,result⟩
   change laneActual q wn fixed i s=.ok () middle at run
   change lanesFrom q wn fixed (i+1) n middle=.ok () after at rest
   simp only [lanesFrom,EStateM.run,bind,EStateM.bind,run,rest]

theorem actual_range_lanes (q wn : Nat) (fixed : BitVec 128) (range : IntRange)
 (hstep : range.step=1) (i : Int) (hs : (i-range.start)%range.step=0) (start : range.start≤i) :
 IntRange.forIn'.loop (m:=SailM) range
   (fun k _ (_ : Unit)=>laneActual q wn fixed k >>= fun u=>pure (.yield u)) () i hs =
 lanesFrom q wn fixed i (range.stop - i+1).toNat := by
 induction (),i,hs using IntRange.forIn'.loop.induct range with
 | case1 u i hs hi ih =>
   cases u
   have hmem:=hi
   simp [Membership.mem,hstep] at hmem
   rw [IntRange.forIn'.loop.eq_1]
   simp only [hi,dite_true,bind_assoc,pure_bind]
   have hn : (range.stop - i+1).toNat=(range.stop-(i+range.step)+1).toNat+1 := by omega
   rw [hn]
   simp only [lanesFrom]
   congr 1
   funext unit
   cases unit
   simpa only [hstep] using ih () (by omega)
 | case2 u i hs hi =>
   cases u
   have hmem:=hi
   simp [Membership.mem,hstep] at hmem
   have hn : (range.stop - i+1).toNat=0 := by omega
   rw [IntRange.forIn'.loop.eq_1]
   simp [hi,hn,lanesFrom]



theorem actual_range_lanes_start (q wn : Nat) (fixed : BitVec 128) (range : IntRange)
 (hstep : range.step=1) :
 IntRange.forIn' (m:=SailM) range ()
   (fun k _ (_ : Unit)=>laneActual q wn fixed k >>= fun u=>pure (.yield u)) =
 lanesFrom q wn fixed range.start (range.stop-range.start+1).toNat := by
 unfold IntRange.forIn'
 exact actual_range_lanes q wn fixed range hstep range.start (by simp) (by omega)


theorem lane_yield_map (q wn : Nat) (fixed : BitVec 128) (i : Int) :
 (ForInStep.yield <$> laneActual q wn fixed i)=(laneActual q wn fixed i >>= fun u=>pure (.yield u)) := by rfl

theorem actual_range_lanes_map (q wn : Nat) (fixed : BitVec 128) (range : IntRange)
 (hstep : range.step=1) :
 IntRange.forIn' (m:=SailM) range ()
   (fun k _ (_ : Unit)=>ForInStep.yield <$> laneActual q wn fixed k) =
 lanesFrom q wn fixed range.start (range.stop-range.start+1).toNat := by
 simp only [lane_yield_map]
 exact actual_range_lanes_start q wn fixed range hstep



theorem overlay_same_content (A : Nat → Prop) (H L : Vector Nat 145) (s : State)
 (keep : Keep A H L s) (view : CView) (rep : Represents iq_cap view s) (q wn : Nat) :
 ∃ after,(iq_word_ov_set q wn 0#10).run s=.ok () after ∧ Keep A H L after ∧ Represents iq_cap view after := by
 have start : Proofs.OrderedDelivery.OverlayFrame s s :=
   ⟨rfl,rfl,(by intros;rfl),keep.2.2.2 Register.iq_wov⟩
 obtain ⟨u,after,run,frame⟩:=Proofs.OrderedDelivery.overlay_total s q wn 0#10 s start
 cases u
 have safe:=Proofs.TaggedCheck.overlay_write_safe A (H:=H) (L:=L) q wn 0#10 s keep
 refine ⟨after,run,?_,Proofs.IqReach.represents_memory iq_cap view s after frame.1 rep⟩
 simpa [run] using safe

def correctionActual (q wn : Nat) (fixed : BitVec 128) : SailM Unit := do
 for k in [0:3:1]i do
   laneActual q wn fixed k
 iq_word_ov_set q wn 0#10

theorem correction_sequence (q wn : Nat) (fixed : BitVec 128) : correctionActual q wn fixed=
 (do lanesFrom q wn fixed 0 4; iq_word_ov_set q wn 0#10) := by
 simp [correctionActual,ForIn.forIn,ForIn'.forIn',instForInOfForIn',
   bind_assoc,pure_bind,bind_pure,actual_range_lanes_map,actual_range_lanes_start]

theorem actual_correction_content (reads : ValuePorts) (writes : Proofs.Initialized.WritePorts iq_cap)
 (A : Nat → Prop) (H L : Vector Nat 145) (s : State) (keep : Keep A H L s)
 (view : CView) (rep : Represents iq_cap view s) (q wn : Nat) (fixed : BitVec 128) (hq : q<145) :
 ∃ after,(correctionActual q wn fixed).run s=.ok () after ∧ Keep A H L after ∧
   Represents iq_cap (lanesView q wn fixed H L 0 4 view) after := by
 obtain ⟨middle,run,good,represented⟩:=actual_lanes_content reads writes A H L q wn fixed hq 4 0 s keep view rep
 obtain ⟨after,rest,kept,result⟩:=overlay_same_content A H L middle good (lanesView q wn fixed H L 0 4 view) represented q wn
 refine ⟨after,?_,kept,result⟩
 rw [correction_sequence]
 change lanesFrom q wn fixed 0 4 s=.ok () middle at run
 change iq_word_ov_set q wn 0#10 middle=.ok () after at rest
 simp only [EStateM.run,bind,EStateM.bind,run,rest]

def wordCheckWithCorrection (finish : Nat → Nat → BitVec 128 → SailM Unit) (q : Nat) (c : Nat) (w : Int) : SailM Unit := do
  if ((w ≥b 0) : Bool)
  then
    (do
      let wn : Nat := (Max.max w 0)
      let e ← do (iq_word_ov q wn)
      if ((e != 0b0000000000#10) : Bool)
      then
        (do
          let (k, s) := (ecc_classify e)
          match k with
          | .ECC_OK => (pure ())
          | .ECC_UNCORRECTABLE => (ras_ecc_event c q 2 true false 0 (Int.emod wn 16777216) 0 s)
          | .ECC_CORRECTED =>
            (do
              (ras_ecc_event c q 2 false false 0 (Int.emod wn 16777216) 0 s)
              let (_, fixed, _) := (ecc_correct (BitVec.zero 128) e)
              (finish q wn fixed)))
      else (pure ()))
  else (pure ())

/-- Type quantifiers: units : Nat, c : Nat, q : Nat, 0 ≤ q ∧ q ≤ 144, 0 ≤ c, 0 ≤ units -/


theorem actual_word_correction_factor (q c : Nat) (w : Int) :
 iq_word_check q c w=wordCheckWithCorrection correctionActual q c w := by
 simp only [iq_word_check,wordCheckWithCorrection,correctionActual,laneActual,cell,
   bind_assoc,pure_bind,bind_pure,Proofs.OrderedImage.bind_if]
 rfl

/-- The real RAS/callback phase retains the SAME IQ witness on both normal and
    partial-error exits. This does not erase IRQ or other external side effects. -/
theorem ras_content (ports : Proofs.HistoryComposition.Ports) (A : Nat → Prop)
 (H L : Vector Nat 145) (s : State) (keep : Keep A H L s) (view : CView)
 (rep : Represents iq_cap view s) (c q : Nat) (unc : Bool) (addr pos : Nat) :
 match (ras_ecc_event c q 2 unc false 0 addr 0 pos).run s with
 | .ok _ after => Keep A H L after ∧ Represents iq_cap view after
 | .error _ after => Keep A H L after ∧ Represents iq_cap view after := by
 have good:=Proofs.TaggedCheck.ras_ecc_safe A (H:=H) (L:=L) c q 2 unc false 0 addr 0 pos ports.irq s keep
 have same:=Proofs.HistoryContent.actual_phase_same_history _
   (Proofs.HistoryFrames.project_frame _ (ResourceExecuteFrameProofs.ras_ecc_event_frame ports.resource c q 2 unc false 0 addr 0 pos)) s view rep
 cases run:(ras_ecc_event c q 2 unc false 0 addr 0 pos).run s <;>
   simpa only [run] using And.intro good same

/-- Complete actual correctable-word path: real RAS first, exact per-tag XOR
    repair of all four lanes next, overlay clearing last. Any RAS error retains
    the incoming content, while success has the calculated SAME-view history. -/
theorem actual_corrected_word_content (ports : Proofs.HistoryComposition.Ports)
 (reads : ValuePorts) (A : Nat → Prop) (H L : Vector Nat 145) (s : State)
 (keep : Keep A H L s) (view : CView) (rep : Represents iq_cap view s)
 (q c : Nat) (w : Int) (hq : q<145) (hw : 0≤w) (e : BitVec 10)
 (overlay : (iq_word_ov q w.toNat).run s=.ok e s) (nonzero : e≠0#10)
 (corrected : (ecc_classify e).1=ECC_CORRECTED) :
 match (iq_word_check q c w).run s with
 | .ok _ after => Keep A H L after ∧
     Represents iq_cap (lanesView q w.toNat (ecc_correct (BitVec.zero 128) e).2.1 H L 0 4 view) after
 | .error _ after => Keep A H L after ∧ Represents iq_cap view after := by
 let pos : Nat := (ecc_classify e).2
 have cls : ecc_classify e=(ECC_CORRECTED,pos) := by
   apply Prod.ext
   · exact corrected
   · rfl
 let addr : Nat := ((max w 0).emod 16777216).toNat
 change iq_word_ov q w.toNat s=.ok e s at overlay
 have source : (iq_word_check q c w).run s=
   (ras_ecc_event c q 2 false false 0 addr 0 pos >>= fun _=>
      correctionActual q w.toNat (ecc_correct (BitVec.zero 128) e).2.1).run s := by
   rw [actual_word_correction_factor]
   simp [wordCheckWithCorrection,EStateM.run,bind,EStateM.bind,hw,overlay,nonzero,cls,addr]
 rw [source]
 have framed:=ras_content ports A H L s keep view rep c q false addr pos
 cases run:(ras_ecc_event c q 2 false false 0 addr 0 pos).run s with
 | error err after =>
   have good : Keep A H L after ∧ Represents iq_cap view after := by simpa [run] using framed
   change ras_ecc_event c q 2 false false 0 addr 0 pos s=.error err after at run
   simpa [EStateM.run,bind,EStateM.bind,run] using good
 | ok u middle =>
   cases u
   have good : Keep A H L middle ∧ Represents iq_cap view middle := by simpa [run] using framed
   obtain ⟨after,rest,kept,result⟩:=actual_correction_content reads ports.writes A H L middle good.1 view good.2 q w.toNat (ecc_correct (BitVec.zero 128) e).2.1 hq
   change ras_ecc_event c q 2 false false 0 addr 0 pos s=.ok () middle at run
   change correctionActual q w.toNat (ecc_correct (BitVec.zero 128) e).2.1 middle=.ok () after at rest
   simp only [EStateM.run,bind,EStateM.bind,run]
   rw [rest]
   exact And.intro kept result


def wordView (q : Nat) (w : Int) (e : BitVec 10) (H L : Vector Nat 145) (view : CView) : CView :=
 if w<0 then view else if e=0#10 then view else
 match (ecc_classify e).1 with
 | ECC_CORRECTED => lanesView q w.toNat (ecc_correct (BitVec.zero 128) e).2.1 H L 0 4 view
 | _ => view

/-- All cases of the actual word checker, including absent overlays, negative
    host tags, uncorrectable RAS and partial RAS failure, share one content law. -/
theorem actual_word_content (ports : Proofs.HistoryComposition.Ports) (reads : ValuePorts)
 (A : Nat → Prop) (H L : Vector Nat 145) (s : State) (keep : Keep A H L s)
 (view : CView) (rep : Represents iq_cap view s) (q c : Nat) (w : Int) (hq : q<145)
 (e : BitVec 10) (overlay : (iq_word_ov q w.toNat).run s=.ok e s) :
 match (iq_word_check q c w).run s with
 | .ok _ after => Keep A H L after ∧ Represents iq_cap (wordView q w e H L view) after
 | .error _ after => Keep A H L after ∧ Represents iq_cap view after := by
 have start:=And.intro keep rep
 by_cases hw : 0≤w
 · by_cases zero : e=0#10
   · subst e
     have run : (iq_word_check q c w).run s=.ok () s := by
       change iq_word_ov q w.toNat s=.ok 0#10 s at overlay
       rw [actual_word_correction_factor]
       simp [wordCheckWithCorrection,EStateM.run,bind,EStateM.bind,hw,overlay,pure,EStateM.pure]
     simpa [run,wordView,show ¬w<0 from by omega] using start
   · cases cls : (ecc_classify e).1 with
     | ECC_CORRECTED =>
       have result:=actual_corrected_word_content ports reads A H L s keep view rep q c w hq hw e overlay zero cls
       simpa [wordView,show ¬w<0 from by omega,zero,cls] using result
     | ECC_OK =>
       let pos : Nat := (ecc_classify e).2
       have pair : ecc_classify e=(ECC_OK,pos) := Prod.ext cls rfl
       have run : (iq_word_check q c w).run s=.ok () s := by
         change iq_word_ov q w.toNat s=.ok e s at overlay
         rw [actual_word_correction_factor]
         simp [wordCheckWithCorrection,EStateM.run,bind,EStateM.bind,hw,overlay,zero,pair,pure,EStateM.pure]
       simpa [run,wordView,show ¬w<0 from by omega,zero,cls] using start
     | ECC_UNCORRECTABLE =>
       let pos : Nat := (ecc_classify e).2
       have pair : ecc_classify e=(ECC_UNCORRECTABLE,pos) := Prod.ext cls rfl
       let addr : Nat := ((max w 0).emod 16777216).toNat
       have run : (iq_word_check q c w).run s=
         (ras_ecc_event c q 2 true false 0 addr 0 pos).run s := by
         change iq_word_ov q w.toNat s=.ok e s at overlay
         rw [actual_word_correction_factor]
         simp [wordCheckWithCorrection,EStateM.run,bind,EStateM.bind,hw,overlay,zero,pair,addr]
       have result:=ras_content ports A H L s keep view rep c q true addr pos
       simpa [run,wordView,show ¬w<0 from by omega,zero,cls] using result
 · have run : (iq_word_check q c w).run s=.ok () s := by
     simp [iq_word_check,EStateM.run,hw,pure,EStateM.pure]
   simpa [run,wordView,show w<0 from by omega] using start
end Proofs.RepairCell
