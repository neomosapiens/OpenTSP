-- Concrete AGT register origin invariants and actual producer transition composition.
import Proofs.AgtOrigins
import Proofs.AgtExecutionFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace AgtOriginStateProofs
open ResourceFrameProofs ResourceInvariantProofs AgtOriginProofs
abbrev St:=ResourceFrameProofs.St
abbrev Arr:=Vector agt 144
def AllTimes (xs:Arr) : Prop:=∀i:Nat,i<144→TimeOrigin xs[i]!
def ValidOpt (o:Option Arr) : Prop:=∀xs,o=some xs→AllTimes xs
def OriginView (v:Option Arr×Option Arr) : Prop:=ValidOpt v.1 ∧ ValidOpt v.2
def Origin (s:St) : Prop:=OriginView (AgtExecutionFrameProofs.RView s)

theorem update_times (xs:Arr) (q:Nat) (a:agt) (hq:q<144) (h:AllTimes xs) (ha:TimeOrigin a) :
 AllTimes (vectorUpdate xs q a) := by
 intro i hi
 by_cases he:i=q
 · subst i
   simpa [vectorUpdate,Vector.set!,hq] using ha
 · have hg:(vectorUpdate xs q a)[i]! =xs[i]! :=
    vector_update_other 144 i q xs a hi hq he
   rw [hg]
   exact h i hi

theorem other_write (r:Register) (v:RegisterType r) (hr:r≠Register.agts) (hw:r≠Register.agts_w) :
 Inv Origin (Out.writeReg r v) :=
 keeps_pred AgtExecutionFrameProofs.RView OriginView _
  (AgtExecutionFrameProofs.write_frame r v hr hw)

theorem r_write (xs:Arr) (hx:AllTimes xs) : Inv Origin (Out.writeReg Register.agts xs) := by
 intro st hs
 change Origin {st with regs:=st.regs.insert Register.agts xs}
 unfold Origin OriginView ValidOpt AgtExecutionFrameProofs.RView at *
 simpa [Std.ExtDHashMap.get?_insert] using And.intro hx hs.2

theorem w_write (xs:Arr) (hx:AllTimes xs) : Inv Origin (Out.writeReg Register.agts_w xs) := by
 intro st hs
 change Origin {st with regs:=st.regs.insert Register.agts_w xs}
 unfold Origin OriginView ValidOpt AgtExecutionFrameProofs.RView at *
 simpa [Std.ExtDHashMap.get?_insert] using And.intro hs.1 hx

theorem actual_agt_set_origin (q p:Nat) (a:agt) (hq:q<144) (ha:TimeOrigin a) :
 Inv Origin (agt_set q p a) := by
 unfold agt_set
 split
 · apply read_bind_bounded Origin Register.agts AllTimes
   · intro st hs xs hx
     exact hs.1 xs hx
   · intro xs hx
     exact r_write _ (update_times xs q a hq hx ha)
 · apply read_bind_bounded Origin Register.agts_w AllTimes
   · intro st hs xs hx
     exact hs.2 xs hx
   · intro xs hx
     exact w_write _ (update_times xs q a hq hx ha)

theorem get_bind_origin {α:Type} (q p:Nat) (hq:q<144) (f:agt→SailM α)
 (hf:∀a,TimeOrigin a→Inv Origin (f a)) :
 Inv Origin (agt_get q p >>= f) := by
 unfold agt_get
 split
 · simp only [pure_bind,bind_assoc]
   apply read_bind_bounded Origin Register.agts AllTimes
   · intro st hs xs hx; exact hs.1 xs hx
   · intro xs hx; exact hf xs[q]! (hx q hq)
 · simp only [pure_bind,bind_assoc]
   apply read_bind_bounded Origin Register.agts_w AllTimes
   · intro st hs xs hx; exact hs.2 xs hx
   · intro xs hx; exact hf xs[q]! (hx q hq)

theorem actual_configure_origin (q p:Nat) (hq:q<144) (f:agt_field) (v:BitVec 16) :
 Inv Origin (agt_configure q p f v) := by
 unfold agt_configure
 simp only [pure_bind]
 apply get_bind_origin q p hq
 intro a ha
 split
 · apply bind_inv
   · exact actual_agt_set_origin q p _ hq (configure_origin a f v ha)
   · intro; exact pure_inv _ _
 · exact pure_inv _ _

theorem actual_refresh_origin (q p:Nat) (hq:q<144) : Inv Origin (agt_refresh q p) := by
 unfold agt_refresh
 simp only [pure_bind]
 apply get_bind_origin q p hq
 intro a ha
 repeat' first
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact other_write _ _ (by decide) (by decide)
  | exact actual_agt_set_origin q p _ hq (refresh_origin a ha)
  | apply bind_inv
  | split
  | intro

theorem actual_finish_origin (q p:Nat) (hq:q<144) : Inv Origin (agt_finish_config q p) := by
 unfold agt_finish_config
 simp only [pure_bind]
 apply get_bind_origin q p hq
 intro a ha
 repeat' first
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact other_write _ _ (by decide) (by decide)
  | exact actual_agt_set_origin q p _ hq (finish_origin a ha)
  | apply bind_inv
  | split
  | intro

theorem array_idle_origin : AllTimes (vectorInit agt_idle) := by
 intro i hi
 simpa [vectorInit,hi] using idle_origin

theorem execution_origin {α:Type} (action:SailM α) (hf:Keeps AgtExecutionFrameProofs.RView action) :
 Inv Origin action := keeps_pred AgtExecutionFrameProofs.RView OriginView action hf

theorem actual_get_origin (s:St) (hs:Origin s) (q p:Nat) (hq:q<144) (a:agt)
 (hget:(agt_get q p).run s=.ok a s) : TimeOrigin a := by
 by_cases hp:p=0
 · cases hr:s.regs.get? Register.agts with
   | none =>
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
       pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     cases hget
   | some arr =>
     have ha:=hs.1 arr hr q hq
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
       pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     simpa [hget] using ha
 · cases hr:s.regs.get? Register.agts_w with
   | none =>
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
       pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     cases hget
   | some arr =>
     have ha:=hs.2 arr hr q hq
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
       pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     simpa [hget] using ha

theorem actual_hold_path_origin (q p c:Nat) : Inv Origin (hold_path q p c) := by
 unfold hold_path
 exact bind_inv _ _ _ (read_inv _ _) (fun _=>other_write _ _ (by decide) (by decide))

theorem actual_bank_report_origin (ports:AgtExecutionFrameProofs.Ports) (q c:Nat)
 (a b:instruction×Int×Int) : Inv Origin (bank_report q c a b) := by
 unfold bank_report
 repeat' first
  | assumption
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact execution_origin _ (AgtExecutionFrameProofs.srf_report_frame ports _ _ _ _ _ _)
  | reduce_resource_beta
  | apply forIn_inv
  | apply bind_inv
  | cases_resource_pair
  | split
  | intro

def runTail (q c p:Nat) (ls:Vector agt_level 4) (off soff:Int) (a:agt)
 (blocked:Bool) (partner:Option (instruction×Int×Int)) : SailM Unit := do
 if blocked then
  report q c .MEM_BANK
  match partner with
  | some other=>bank_report q c (←loop_clash_ref q p off soff) other
  | none=>pure ()
 else run_element q a.instr a.t0 (max ((c:Int)-a.t0) 0).toNat (a.offset+off) (a.soffset+soff)
 hold_path q p c
 agt_set q p (agt_after_reference a ls off soff c)
 agt_refresh q p

theorem max_to_nat_zero (z:Int) : (max z 0).toNat=z.toNat := by omega

theorem actual_run_loop_source (q c p:Nat) (step:Option (Vector agt_level 4×Nat×Int×Int))
 (blocked:Bool) (partner:Option (instruction×Int×Int)) :
 run_loop q c p step blocked partner=
 (match step with
 | none=>pure ()
 | some (ls,_,off,soff)=>do let a←agt_get q p; runTail q c p ls off soff a blocked partner) := by
 unfold run_loop runTail
 cases step with
 | none => rfl
 | some step =>
   rcases step with ⟨ls,delay,off,soff⟩
   cases partner <;> cases blocked <;> simp only [Bool.false_eq_true,if_false,if_true,pure_bind,bind_assoc,Nat.max_zero,max_to_nat_zero] <;> rfl

theorem tail_origin (ports:AgtExecutionFrameProofs.Ports) (q c p:Nat) (hq:q<144)
 (ls:Vector agt_level 4) (delay:Nat) (off soff:Int) (a:agt) (parked blocked:Bool)
 (partner:Option (instruction×Int×Int)) (ha:TimeOrigin a)
 (hd:agt_due_pure a c parked=some (ls,delay,off,soff)) :
 Inv Origin (runTail q c p ls off soff a blocked partner) := by
 unfold runTail loop_clash_ref
 repeat' first
  | assumption
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact execution_origin _ (AgtExecutionFrameProofs.report_frame ports _ _ _)
  | exact execution_origin _ (AgtExecutionFrameProofs.agt_get_frame ports _ _)
  | exact execution_origin _ (AgtExecutionFrameProofs.run_element_frame ports _ _ _ _ _ _)
  | exact actual_bank_report_origin ports _ _ _ _
  | exact actual_hold_path_origin _ _ _
  | exact actual_agt_set_origin q p _ hq (after_reference_origin a c parked ls delay off soff ha hd)
  | exact actual_refresh_origin q p hq
  | apply bind_inv
  | split
  | intro

theorem actual_run_loop_snapshot_origin (ports:AgtExecutionFrameProofs.Ports)
 (s:St) (hs:Origin s) (q c p:Nat) (hq:q<144) (a:agt)
 (hget:(agt_get q p).run s=.ok a s) (ls:Vector agt_level 4) (delay:Nat) (off soff:Int)
 (parked blocked:Bool) (partner:Option (instruction×Int×Int))
 (hd:agt_due_pure a c parked=some (ls,delay,off,soff)) :
 match (run_loop q c p (some (ls,delay,off,soff)) blocked partner).run s with
 | .ok _ after=>Origin after
 | .error _ after=>Origin after := by
 have ha:=actual_get_origin s hs q p hq a hget
 have ht:=tail_origin ports q c p hq ls delay off soff a parked blocked partner ha hd s hs
 rw [actual_run_loop_source]
 simp only [EStateM.run,bind,EStateM.bind,show agt_get q p s=.ok a s from hget]
 dsimp only [EStateM.run] at ht
 cases htail:runTail q c p ls off soff a blocked partner s
 all_goals simp only [htail] at ht ⊢
 all_goals exact ht

end AgtOriginStateProofs
