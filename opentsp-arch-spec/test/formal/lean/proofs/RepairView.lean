import Proofs.RepairCell
import Proofs.OverlayHistory
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
open Out.Register Out.ecc_class
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace Proofs.RepairView
/-- The actual SECDED correction is affine in the raw data: the overlay
    determines one XOR mask independent of every payload, bundle and width bit. -/
theorem actual_ecc_affine (data : BitVec 128) (overlay : BitVec 10) :
 (ecc_correct data overlay).2.1 = data ^^^ (ecc_correct (BitVec.zero 128) overlay).2.1 := by
 simp only [ecc_correct]
 split <;> simp_all

/-- Every live IQ lane observes the same corrected value before the raw XOR
    repair and after clearing that word overlay, including bits 28..31. -/
theorem actual_lane_view (unit : BitVec 32) (overlay : BitVec 10) (lane : Nat)
 (bound : lane<4) :
 BitVec.slice (ecc_correct ((Sail.BitVec.zeroExtend unit 128) <<< (32*(lane:Int))) overlay).2.1
     (32*lane) 32 =
 unit ^^^ BitVec.slice (ecc_correct (BitVec.zero 128) overlay).2.1 (32*lane) 32 := by
 rw [actual_ecc_affine]
 have cases : lane=0 ∨ lane=1 ∨ lane=2 ∨ lane=3 := by omega
 rcases cases with h|h|h|h <;> subst lane <;> simp only [Sail.BitVec.slice,Sail.BitVec.zeroExtend,HShiftLeft.hShiftLeft]
 all_goals
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp [BitVec.getLsbD_extractLsb', BitVec.getLsbD_shiftLeft]
  simp [hi, show i < 128 from by omega,
    show 32 + i < 128 from by omega, show ¬32 + i < 32 from by omega,
    show 64 + i < 128 from by omega, show ¬64 + i < 64 from by omega,
    show 96 + i < 128 from by omega, show ¬96 + i < 96 from by omega]

open Proofs.RepairCell
 theorem ring_live_injective (h len a b : Nat) (hh : h<iq_cap) (hl : len≤iq_cap)
 (ha : a<len) (hb : b<len) :
 Proofs.Capacity.ringC iq_cap (h+a)=Proofs.Capacity.ringC iq_cap (h+b) → a=b := by
 rw [Proofs.Capacity.ring_spec,Proofs.Capacity.ring_spec]
 split <;> split <;> omega

 theorem update_tag (q r slot : Nat) (want : Int) (m : BitVec 32) (H : Vector Nat 145)
 (v : CView) (i : Int) : (updateView q want m H v i).tag r slot=v.tag r slot := by
 dsimp only [updateView]
 split <;> rfl
 theorem cells_tag (q r slot : Nat) (want : Int) (m : BitVec 32) (H : Vector Nat 145)
 (i : Int) (n : Nat) (v : CView) :
 (viewFrom q want m H i n v).tag r slot=v.tag r slot := by
 induction n generalizing i v with
 | zero => rfl
 | succ n ih => simp only [viewFrom,ih,update_tag]

 theorem cells_unit_away (q r slot : Nat) (want : Int) (m : BitVec 32) (H : Vector Nat 145)
 (start n : Nat) (v : CView)
 (away : ∀ j, j<n → ¬(q=r ∧ Proofs.Capacity.ringC iq_cap (H[q]!+start+j)=slot)) :
 (viewFrom q want m H (start:Int) n v).unit r slot=v.unit r slot := by
 induction n generalizing start v with
 | zero => rfl
 | succ n ih =>
   simp only [viewFrom]
   have next : (start:Int)+1=((start+1:Nat):Int) := by omega
   rw [next,ih (start+1) (updateView q want m H v start) (by
     intro j hj
     simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using away (j+1) (by omega))]
   have first:=away 0 (by omega)
   simpa [updateView,cUnitWrite,Nat.add_assoc] using
    (show (updateView q want m H v start).unit r slot=v.unit r slot from by
      dsimp only [updateView]
      split <;> simp_all [cUnitWrite])

 theorem cells_unit_live (q : Nat) (want : Int) (m : BitVec 32) (H : Vector Nat 145)
 (len start n pos : Nat) (v : CView) (hh : H[q]!<iq_cap) (hl : len≤iq_cap)
 (hp : pos<len) (endbound : start+n≤len) :
 (viewFrom q want m H (start:Int) n v).unit q (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) =
 if start≤pos ∧ pos<start+n ∧ v.tag q (Proofs.Capacity.ringC iq_cap (H[q]!+pos))=want
 then v.unit q (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) ^^^ m
 else v.unit q (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) := by
 induction n generalizing start v with
 | zero =>
   simp only [viewFrom]
   rw [if_neg (by omega)]
 | succ n ih =>
   have firstbound : start<len := by omega
   have indexeq : Proofs.Capacity.ringC iq_cap (H[q]!+start)=
       Proofs.Capacity.ringC iq_cap (H[q]!+pos) ↔ start=pos :=
     ⟨ring_live_injective H[q]! len start pos hh hl firstbound hp,fun e=>by rw [e]⟩
   simp only [viewFrom]
   have next : (start:Int)+1=((start+1:Nat):Int) := by omega
   rw [next,ih (start+1) (updateView q want m H v start) (by omega)]
   simp only [update_tag]
   by_cases eq : start=pos
   · subst start
     have bad : ¬pos+1≤pos := by omega
     simp only [bad,false_and,if_false,Nat.le_refl,true_and]
     dsimp only [updateView]
     split <;> simp_all [cUnitWrite]
   · have ne : Proofs.Capacity.ringC iq_cap (H[q]!+start)≠
         Proofs.Capacity.ringC iq_cap (H[q]!+pos) := fun h=>eq (indexeq.mp h)
     have same : (updateView q want m H v start).unit q
       (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) = v.unit q
       (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) := by
         dsimp only [updateView]
         split <;> simp_all [cUnitWrite]
     rw [same]
     have arith : (start+1≤pos ∧ pos<start+1+n) ↔ (start≤pos ∧ pos<start+(n+1)) := by omega
     simp only [and_assoc] at arith ⊢
     have low : start+1≤pos ↔ start≤pos := by omega
     simp only [low,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

 theorem lane_tag (q r slot wn : Nat) (fixed : BitVec 128) (k : Int)
 (H L : Vector Nat 145) (v : CView) :
 (laneView q wn fixed k H L v).tag r slot=v.tag r slot := by
 dsimp only [laneView]
 split <;> simp only [cells_tag]

 theorem lane_unit (q wn : Nat) (fixed : BitVec 128) (k : Int)
 (H L : Vector Nat 145) (v : CView) (pos : Nat)
 (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hp : pos<L[q]!) :
 (laneView q wn fixed k H L v).unit q (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) =
 if v.tag q (Proofs.Capacity.ringC iq_cap (H[q]!+pos))=4*(wn:Int)+k
 then v.unit q (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) ^^^ BitVec.slice fixed (32*k).toNat 32
 else v.unit q (Proofs.Capacity.ringC iq_cap (H[q]!+pos)) := by
 dsimp only [laneView]
 split
 · rename_i zero
   split <;> simp_all
 · simpa only [Nat.zero_add,Nat.zero_le,true_and,hp] using
     cells_unit_live q (4*(wn:Int)+k) (BitVec.slice fixed (32*k).toNat 32) H L[q]! 0 L[q]! pos v hh hl hp (by omega)

 theorem quartet_tag (q r slot wn : Nat) (fixed : BitVec 128)
 (H L : Vector Nat 145) (v : CView) :
 (lanesView q wn fixed H L 0 4 v).tag r slot=v.tag r slot := by
 simp only [lanesView,lane_tag]

 /-- Every live unit of the word receives exactly its own one-of-four mask;
     all other words and host-tagged units retain their original raw value. -/
 theorem quartet_unit (q wn : Nat) (fixed : BitVec 128)
 (H L : Vector Nat 145) (v : CView) (pos : Nat)
 (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hp : pos<L[q]!) :
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]!+pos)
 (lanesView q wn fixed H L 0 4 v).unit q slot =
 if v.tag q slot / 4=(wn:Int) then
    v.unit q slot ^^^ BitVec.slice fixed (32*(v.tag q slot % 4).toNat) 32
 else v.unit q slot := by
 dsimp only
 simp only [lanesView,lane_unit q wn fixed _ H L _ pos hh hl hp,lane_tag]
 simp only [Int.add_zero,Int.reduceAdd]
 generalize tdef : v.tag q (Proofs.Capacity.ringC iq_cap (H[q]!+pos))=t
 by_cases h0 : t=4*(wn:Int)
 · have h1 : t≠4*(wn:Int)+1 := by omega
   have h2 : t≠4*(wn:Int)+2 := by omega
   have h3 : t≠4*(wn:Int)+3 := by omega
   have div : t/4=(wn:Int) := by omega
   have rem : t%4=0 := by omega
   simp only [if_pos h0,if_neg h1,if_neg h2,if_neg h3,if_pos div,rem];rfl
 · by_cases h1 : t=4*(wn:Int)+1
   · have h2 : t≠4*(wn:Int)+2 := by omega
     have h3 : t≠4*(wn:Int)+3 := by omega
     have div : t/4=(wn:Int) := by omega
     have rem : t%4=1 := by omega
     simp only [if_neg h0,if_pos h1,if_neg h2,if_neg h3,if_pos div,rem];rfl
   · by_cases h2 : t=4*(wn:Int)+2
     · have h3 : t≠4*(wn:Int)+3 := by omega
       have div : t/4=(wn:Int) := by omega
       have rem : t%4=2 := by omega
       simp only [if_neg h0,if_neg h1,if_pos h2,if_neg h3,if_pos div,rem];rfl
     · by_cases h3 : t=4*(wn:Int)+3
       · have div : t/4=(wn:Int) := by omega
         have rem : t%4=3 := by omega
         simp only [if_neg h0,if_neg h1,if_neg h2,if_pos h3,if_pos div,rem];rfl
       · have div : t/4≠(wn:Int) := by omega
         simp only [if_neg h0,if_neg h1,if_neg h2,if_neg h3,if_neg div]

 def observedUnit (raw : BitVec 32) (tag : Int) (overlay : Int → BitVec 10) : BitVec 32 :=
 if tag=iq_no_word ∨ tag/4<0 then raw else
 let e:=overlay (tag/4)
 if e=0#10 then raw else
 BitVec.slice (ecc_correct ((Sail.BitVec.zeroExtend raw 128) <<< (32*(tag%4))) e).2.1
   (32*(tag%4).toNat) 32
 theorem observed_xor (raw : BitVec 32) (tag : Int) (overlay : Int → BitVec 10)
 (ordinary : tag≠iq_no_word) (nonneg : 0≤tag/4) :
 observedUnit raw tag overlay = raw ^^^ BitVec.slice
   (ecc_correct (BitVec.zero 128) (overlay (tag/4))).2.1 (32*(tag%4).toNat) 32 := by
 have lane : (tag%4).toNat<4 := by omega
 have cast : ((tag%4).toNat:Int)=tag%4 := by omega
 unfold observedUnit
 rw [if_neg (by omega)]
 by_cases zero : overlay (tag/4)=0#10
 · rw [if_pos zero]
   have ez : (ecc_correct (BitVec.zero 128) (0#10)).2.1=0#128 := by decide +kernel
   rw [zero,ez]
   simp [Sail.BitVec.slice]
 · rw [if_neg zero]
   simpa only [cast] using actual_lane_view raw (overlay (tag/4)) (tag%4).toNat lane

 /-- Complete live-position observation law for the actual four-lane repair
     schedule plus the specified overlay clear. Other-word overlays are retained.
     Width, opcode, bundle and operand bits are all covered by 32-bit equality. -/
 theorem repaired_observation (q wn : Nat) (e : BitVec 10)
 (H L : Vector Nat 145) (v : CView) (pos : Nat)
 (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hp : pos<L[q]!)
 (overlay : Int → BitVec 10) (word : overlay (wn:Int)=e) :
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]!+pos)
 let repaired:=lanesView q wn (ecc_correct (BitVec.zero 128) e).2.1 H L 0 4 v
 observedUnit (repaired.unit q slot) (repaired.tag q slot)
   (fun w=>if w=(wn:Int) then 0#10 else overlay w) =
 observedUnit (v.unit q slot) (v.tag q slot) overlay := by
 dsimp only
 rw [quartet_tag,quartet_unit q wn _ H L v pos hh hl hp]
 generalize tdef : v.tag q (Proofs.Capacity.ringC iq_cap (H[q]!+pos))=t
 by_cases same : t/4=(wn:Int)
 · have ordinary : t≠iq_no_word := by
     have sentinel : iq_no_word = (-1:Int) := by rfl
     rw [sentinel]
     omega
   have nonneg : 0≤t/4 := by omega
   rw [if_pos same]
   have before:=observed_xor (v.unit q (Proofs.Capacity.ringC iq_cap (H[q]!+pos))) t overlay ordinary nonneg
   rw [before]
   simp only [same,word]
   simp [observedUnit,ordinary,nonneg,same]
 · rw [if_neg same]
   unfold observedUnit
   split
   · rfl
   · simp [same]

theorem actual_live_reads (reads : ValuePorts) (A : Nat → Prop) (H L : Vector Nat 145)
 (s : RepairCell.State) (keep : Proofs.TaggedCheck.Keep A H L s)
 (view : CView) (rep : Represents iq_cap view s) (q pos : Nat)
 (hq : q<145) (hp : pos<L[q]!) :
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]!+pos)
 (iq_peek q pos).run s=.ok (view.unit q slot) s ∧
 (iq_unit_word q pos).run s=.ok
   (if view.tag q slot=iq_no_word then -1 else Int.ediv (view.tag q slot) 4) s ∧
 (iq_unit_lane q pos).run s=.ok (Int.emod (view.tag q slot) 4).toNat s := by
 dsimp only
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]!+pos)
 have bounds:=Proofs.HistoryComposition.captured_bounds A H L s keep q hq
 have live : Proofs.Storage.Live iq_cap s q slot :=
   ⟨H,L,keep.1.2.1,keep.1.2.2,hq,bounds.1,bounds.2,pos,hp,rfl⟩
 obtain ⟨old,oldrep,initialized⟩:=keep.1.1
 have observed:=Proofs.Initialized.represented_live_read iq_cap view s rep initialized q slot live
 have ur : iq_unit_read q slot s=.ok (view.unit q slot) s := by
   simpa [EStateM.run,observed.2.2.2] using reads.unit q slot s keep.1.1 live
 have tr : iq_tag_read q slot s=.ok (view.tag q slot) s := by
   simpa [EStateM.run,represented_tag iq_cap view s rep q slot hq observed.2.1] using
     reads.tag q slot s hq observed.2.1
 have hr : Out.readReg Register.iq_head s=.ok H s := by
   simp [Out.readReg,PreSail.readReg,keep.1.2.1,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
 have cast : ((H[q]! :Int)+(pos:Int)).toNat=H[q]!+pos := by omega
 have ring : Out.Functions.ring (H[q]!+pos)=slot := by rfl
 constructor
 · change iq_peek q pos s=.ok (view.unit q slot) s
   simp only [iq_peek,EStateM.run,bind,EStateM.bind,hr,cast,ring,ur]
 · constructor
   · change iq_unit_word q pos s=.ok (if view.tag q slot=iq_no_word then -1 else Int.ediv (view.tag q slot) 4) s
     by_cases sentinel : view.tag q slot=iq_no_word
     · simp [iq_unit_word,EStateM.run,bind,EStateM.bind,hr,cast,ring,tr,sentinel,pure,EStateM.pure]
     · simp [iq_unit_word,EStateM.run,bind,EStateM.bind,hr,cast,ring,tr,sentinel,pure,EStateM.pure]
   · change iq_unit_lane q pos s=.ok (Int.emod (view.tag q slot) 4).toNat s
     simp [iq_unit_lane,EStateM.run,bind,EStateM.bind,hr,cast,ring,tr,pure,EStateM.pure]

theorem actual_view_value (reads : ValuePorts) (A : Nat → Prop) (H L : Vector Nat 145)
 (s : RepairCell.State) (keep : Proofs.TaggedCheck.Keep A H L s)
 (view : CView) (rep : Represents iq_cap view s) (q pos : Nat)
 (hq : q<145) (hp : pos<L[q]!)
 (W : Vector (List ov_ent) 145) (present : s.regs.get? Register.iq_wov=some W)
 (eccLive : s.regs.get? Register.iq_ecc_live=some true) :
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]!+pos)
 (iq_view q pos).run s=.ok
   (observedUnit (view.unit q slot) (view.tag q slot)
     (fun key=>Proofs.OverlayHistory.lookup key.toNat W[q]!)) s := by
 dsimp only
 let slot:=Proofs.Capacity.ringC iq_cap (H[q]!+pos)
 let raw:=view.unit q slot
 let tag:=view.tag q slot
 obtain ⟨peek,word,lane⟩:=actual_live_reads reads A H L s keep view rep q pos hq hp
 change iq_peek q pos s=.ok raw s at peek
 change iq_unit_word q pos s=.ok (if tag=iq_no_word then -1 else Int.ediv tag 4) s at word
 change iq_unit_lane q pos s=.ok (Int.emod tag 4).toNat s at lane
 have flag : Out.readReg Register.iq_ecc_live s=.ok true s := by
   simp [Out.readReg,PreSail.readReg,eccLive,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
 change iq_view q pos s=.ok (observedUnit raw tag
   (fun key=>Proofs.OverlayHistory.lookup key.toNat W[q]!)) s
 by_cases sentinel : tag=iq_no_word
 · simp [iq_view,EStateM.run,bind,EStateM.bind,peek,flag,word,lane,sentinel,iq_no_word,
     observedUnit,Out.Functions.not,pure,EStateM.pure]
 · by_cases negative : Int.ediv tag 4<0
   · have negativeN : tag/4<0 := negative
     simp [iq_view,EStateM.run,bind,EStateM.bind,peek,flag,word,lane,sentinel,negative,negativeN,
       observedUnit,Out.Functions.not,pure,EStateM.pure]
   · have negativeN : ¬tag/4<0 := negative
     have overlay:=Proofs.OverlayHistory.actual_getter s W present q (tag/4).toNat
     change iq_word_ov q (Int.ediv tag 4).toNat s=.ok
       (Proofs.OverlayHistory.lookup (tag/4).toNat W[q]!) s at overlay
     have rem : ((Int.emod tag 4).toNat:Int)=tag%4 := by
       change ((tag%4).toNat:Int)=tag%4
       omega
     simp [iq_view,EStateM.run,bind,EStateM.bind,peek,flag,word,lane,sentinel,negative,negativeN,overlay,
       observedUnit,Out.Functions.not,pure,EStateM.pure,rem]
     have offset : (32*(tag%4)).toNat=32*(tag%4).toNat := by omega
     split <;> simp_all only [pure,EStateM.pure,bind,EStateM.bind,lane,rem,offset,if_pos,if_neg]

end Proofs.RepairView






