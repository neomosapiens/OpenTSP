import Proofs.HistoryComposition
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
open Proofs.TagCoverage Proofs.TaggedCheck Proofs.OrderedImage Proofs.OrderedDelivery
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace Proofs.HistoryContent
abbrev State := ResourceFrameProofs.St

/-- A represented live cell has a unique concrete pair. Existential witnesses in
    the structural invariant cannot choose a different unit or tag at a live slot. -/
theorem represented_pair_unique (cap : Nat) (left right : CView) (s : State)
 (hl : Represents cap left s) (hr : Represents cap right s)
 (q slot : Nat) (hq : q<145) (hs : slot<cap) (live : Initialized cap s q slot) :
 (left.unit q slot,left.tag q slot)=(right.unit q slot,right.tag q slot) := by
 have li : left.initialized q slot := (hl.2 q slot hq hs).1.mpr live
 have ri : right.initialized q slot := (hr.2 q slot hq hs).1.mpr live
 exact Prod.ext ((represented_unit cap left s hl q slot hq hs li).symm.trans
   (represented_unit cap right s hr q slot hq hs ri))
   ((represented_tag cap left s hl q slot hq hs).symm.trans
   (represented_tag cap right s hr q slot hq hs))

theorem represented_content_unique (cap : Nat) (left right : CView) (s : State)
 (hl : Represents cap left s) (hr : Represents cap right s)
 (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
 (hL : s.regs.get? Register.iq_len=some L) (bounds : Proofs.Initialized.Bounds cap s)
 (q : Nat) (hq : q<145) : content cap left q H[q]! L[q]! = content cap right q H[q]! L[q]! := by
 obtain ⟨H0,L0,hH0,hL0,good⟩:=bounds
 have eH : H0=H := by rw [hH] at hH0;cases hH0;rfl
 have eL : L0=L := by rw [hL] at hL0;cases hL0;rfl
 subst H0;subst L0
 unfold content Proofs.Storage.contents
 apply List.map_congr_left
 intro i member
 have hi : i<L[q]! := List.mem_range.mp member
 have slot : Proofs.Capacity.ringC cap (H[q]! + i)<cap := by
  rw [Proofs.Capacity.ring_spec]
  have hh: H[q]!<cap := (good q hq).1
  have ll: L[q]!≤cap := (good q hq).2.1
  split <;> omega
 exact represented_pair_unique cap left right s hl hr q _ hq slot ((good q hq).2.2 i hi)

/-- Rebind an arbitrary incoming concrete history witness, not a newly chosen
    existential store. Complete supplies only the untouched metadata registers. -/
theorem ready_seen (s : State) (ready : Proofs.HistoryComposition.Ready s)
 (view : CView) (rep : Represents iq_cap view s) :
 ∃ (H L : Vector Nat 145) (F : Vector (List fill_seg) 145),
   Seen iq_cap view H L F s := by
 obtain ⟨A,H,L,keep⟩:=ready
 obtain ⟨old,oldrep,bounds⟩:=keep.1.1
 obtain ⟨F,hF⟩:=keep.2.2.2 Register.fill_fifo
 exact ⟨H,L,F,rep,bounds,keep.1.2.1,keep.1.2.2,hF,
   keep.2.2.2 Register.iq_active,keep.2.2.2 Register.iq_wov⟩

/-- At every reachable producer boundary, the SAME incoming content history is
    appended by exactly the offered nonpadding prefix that fits. Allocator and
    initialized/tag invariants hold for the SAME actual resulting state. -/
theorem actual_delivery_content_product (ports : Proofs.HistoryComposition.Ports)
 (s : State) (ready : Proofs.HistoryComposition.Ready s) (view : CView)
 (rep : Represents iq_cap view s) (q c : Nat) (hq : q<145) :
 ∃ (H L : Vector Nat 145) (F : Vector (List fill_seg) 145),
   Seen iq_cap view H L F s ∧
   let result:=finishPlan (planLoop iq_cap L[q]! c (iq_rate+list_len F[q]! + 1) (startPlan iq_rate F[q]!))
   ∃ after,(iq_deliver q c).run s=.ok (result.pushed,result.lost) after ∧
     Proofs.HistoryComposition.Ready after ∧
     Seen iq_cap (historyStore iq_cap view q H[q]! L[q]! result.accepted) H
       (vectorUpdate L q (L[q]! + result.accepted.length)) (vectorUpdate F q result.fifo) after ∧
     Prefix (iq_cap-L[q]!) result ∧
     content iq_cap (historyStore iq_cap view q H[q]! L[q]! result.accepted)
       q H[q]! (L[q]! + result.accepted.length)=
       content iq_cap view q H[q]! L[q]! ++ result.offered.take (iq_cap-L[q]!) := by
 obtain ⟨H,L,F,seen⟩:=ready_seen s ready view rep
 refine ⟨H,L,F,seen,?_⟩
 dsimp only
 obtain ⟨after,run,ordered,pre,contents⟩:=actual_delivery_prefix_exact q c hq ports.writes view H L F s seen
 obtain ⟨v,other,otherRun,good⟩:=Proofs.HistoryComposition.actual_delivery_boundary ports.writes q c hq s ready
 rw [run] at otherRun
 cases otherRun
 exact ⟨after,run,good,ordered,pre,contents⟩

/-- A frame phase retains the incoming concrete store/history witness exactly.
    This applies to all non-IQ-store phases of the actual whole tick. -/
theorem actual_phase_same_history {α : Type} (a : SailM α)
 (frame : ResourceFrameProofs.Keeps Proofs.HistoryFrames.View a)
 (s : State) (view : CView) (rep : Represents iq_cap view s) :
 match a.run s with
 | .ok _ after => Represents iq_cap view after
 | .error _ after => Represents iq_cap view after := by
 have h:=frame s
 cases run:a.run s with
 | ok v after =>
   have eq : Proofs.HistoryFrames.View after=Proofs.HistoryFrames.View s := by simpa [run] using h
   exact Proofs.QueueConsumption.represents_projection iq_cap view s after (congrArg (fun v=>v.2.2) eq) rep
 | error e after =>
   have eq : Proofs.HistoryFrames.View after=Proofs.HistoryFrames.View s := by simpa [run] using h
   exact Proofs.QueueConsumption.represents_projection iq_cap view s after (congrArg (fun v=>v.2.2) eq) rep

theorem content_pop_tail (cap : Nat) (view : CView) (q h len count : Nat)
 (hh : h<cap) (hl : len≤cap) (hc : count≤len) :
 content cap view q (Proofs.Capacity.ringC cap (h+count)) (len-count)=
   (content cap view q h len).drop count := by
 apply List.ext_getElem
 · simp [content,Proofs.Storage.contents]
 · intro i hi hj
   have inside : i<len-count := by simpa [content,Proofs.Storage.contents] using hi
   simp only [content,Proofs.Storage.contents,List.getElem_map,List.getElem_range,List.getElem_drop]
   rw [Proofs.Storage.ring_shift cap h len count i hh hl hc inside]
   simp [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

/-- Actual pop retains the SAME repaired store and removes exactly its prefix.
    It does not rebuild tags, free the physical allocator or choose new values. -/
theorem actual_pop_content_product (A : Nat → Prop) (H L : Vector Nat 145)
 (s : State) (keep : Keep A H L s) (view : CView) (rep : Represents iq_cap view s)
 (q count : Nat) (hq : q<145) (hc : count≤L[q]!) :
 ∃ after,(iq_pop q count).run s=.ok () after ∧
   Keep A (popHead H q count) (popLength L q count) after ∧ Represents iq_cap view after ∧
   content iq_cap view q (Proofs.Capacity.ringC iq_cap (H[q]!+count)) (L[q]!-count)=
     (content iq_cap view q H[q]! L[q]!).drop count := by
 let after : State := {s with regs := (s.regs.insert Register.iq_head (popHead H q count)).insert Register.iq_len (popLength L q count)}
 have run : (iq_pop q count).run s=.ok () after := by
   simpa [Proofs.Capacity.pop_production,after,popHead,popLength] using
     Proofs.Capacity.iq_pop_run q count s H L keep.1.2.1 keep.1.2.2
 obtain ⟨other,otherRun,good⟩:=Proofs.TaggedCheck.pop_product A q count hq hc s keep
 rw [run] at otherRun
 cases otherRun
 have bounds:=Proofs.HistoryComposition.captured_bounds A H L s keep q hq
 exact ⟨after,run,good,rep,content_pop_tail iq_cap view q H[q]! L[q]! count bounds.1 bounds.2 hc⟩
end Proofs.HistoryContent

