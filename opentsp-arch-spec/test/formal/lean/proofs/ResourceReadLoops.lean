-- Exact repeated-register read loop transfer, preserving all state and errors.
import Proofs.ResourceScan
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
namespace ResourceReadLoopProofs
abbrev St := ResourceFrameProofs.St

theorem read_present (s : St) (r : Register) (v : RegisterType r) (h:s.regs.get? r=some v) :
 (Out.readReg r).run s=.ok v s := by
 simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
 get,getThe,MonadStateOf.get,EStateM.get,h,pure,EStateM.pure]

theorem read_loop_run {α : Type} (range : IntRange) (r : Register) (v : RegisterType r)
 (step : RegisterType r → Int → α → α) (state : α) (i : Int) (hs:(i-range.start)%range.step=0)
 (s : St) (hr:s.regs.get? r=some v) :
 (IntRange.forIn'.loop (m:=SailM) range
  (fun k _ a => do let x ← Out.readReg r; pure (.yield (step x k a))) state i hs).run s=
 .ok (IntRange.forIn'.loop (m:=Id) range (fun k _ a => pure (.yield (step v k a))) state i hs) s := by
 induction state,i,hs using IntRange.forIn'.loop.induct range with
 | case1 state i hs hi ih =>
   rw [IntRange.forIn'.loop.eq_1,IntRange.forIn'.loop.eq_1]
   simp only [hi,dite_true,pure_bind]
   have hh:=ih (step v i state)
   have rd:=read_present s r v hr
   simpa only [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,show Out.readReg r s=.ok v s from rd] using hh
 | case2 state i hs hi =>
   rw [IntRange.forIn'.loop.eq_1,IntRange.forIn'.loop.eq_1]
   simp [hi,EStateM.run,pure,EStateM.pure]

theorem read_range_run {α : Type} (range : IntRange) (r : Register) (v : RegisterType r)
 (step : RegisterType r → Int → α → α) (state : α) (s : St) (hr:s.regs.get? r=some v) :
 (IntRange.forIn' (m:=SailM) range state
  (fun k _ a => do let x ← Out.readReg r; pure (.yield (step x k a)))).run s=
 .ok (IntRange.forIn' (m:=Id) range state (fun k _ a => pure (.yield (step v k a)))) s := by
 unfold IntRange.forIn'
 exact read_loop_run _ _ _ _ _ _ _ _ hr
end ResourceReadLoopProofs
