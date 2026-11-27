-- Concrete AGT register origin invariants and actual producer transition composition.
import Proofs.AgtOrigins
import Proofs.AgtOriginReset
import Proofs.AgtExecutionFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace AgtTimeBoundProofs
variable {cycle:Nat}
open ResourceFrameProofs ResourceInvariantProofs AgtOriginProofs
abbrev St:=ResourceFrameProofs.St
abbrev Arr:=Vector agt 144
def At (cycle:Nat) (a:agt):Prop:=a.t0≤a.last ∧ a.last≤cycle

theorem idle_at : At cycle agt_idle:=by constructor <;> simp [At,agt_idle]

theorem opened_at (i:instruction) : At cycle (agt_opened i cycle):=by simp [At,agt_opened]

theorem configure_at (a:agt) (f:agt_field) (v:BitVec 16) (ha:At cycle a):
 At cycle (agt_configure_pure a f v).1:=by
 have h:=configure_fields a f v
 simpa only [At,h.1,h.2] using ha

theorem refresh_at (a:agt) (ha:At cycle a):At cycle (agt_refresh_pure a).1:=by
 unfold agt_refresh_pure
 dsimp only [Id.run]
 split
 · split
   · exact idle_at
   · exact ha
 · exact ha

theorem finish_at (a:agt) (ha:At cycle a):At cycle (agt_finish_config_pure a).1:=
 refresh_at {a with state:=.AGT_RUN} ha

def AllTimes (cycle:Nat) (xs:Arr) : Prop:=∀i:Nat,i<144→At cycle xs[i]!
def ValidOpt (cycle:Nat) (o:Option Arr) : Prop:=∀xs,o=some xs→(AllTimes cycle) xs
def BoundView (cycle:Nat) (v:Option Arr×Option Arr) : Prop:=(ValidOpt cycle) v.1 ∧ (ValidOpt cycle) v.2
def Bound (cycle:Nat) (s:St) : Prop:=(BoundView cycle) (AgtExecutionFrameProofs.RView s)

theorem update_times (xs:Arr) (q:Nat) (a:agt) (hq:q<144) (h:(AllTimes cycle) xs) (ha:At cycle a) :
 (AllTimes cycle) (vectorUpdate xs q a) := by
 intro i hi
 by_cases he:i=q
 · subst i
   simpa [vectorUpdate,Vector.set!,hq] using ha
 · have hg:(vectorUpdate xs q a)[i]! =xs[i]! :=
    vector_update_other 144 i q xs a hi hq he
   rw [hg]
   exact h i hi

theorem other_write (r:Register) (v:RegisterType r) (hr:r≠Register.agts) (hw:r≠Register.agts_w) :
 Inv (Bound cycle) (Out.writeReg r v) :=
 keeps_pred AgtExecutionFrameProofs.RView (BoundView cycle) _
  (AgtExecutionFrameProofs.write_frame r v hr hw)

theorem r_write (xs:Arr) (hx:(AllTimes cycle) xs) : Inv (Bound cycle) (Out.writeReg Register.agts xs) := by
 intro st hs
 change (Bound cycle) {st with regs:=st.regs.insert Register.agts xs}
 unfold Bound BoundView ValidOpt AgtExecutionFrameProofs.RView at *
 simpa [Std.ExtDHashMap.get?_insert] using And.intro hx hs.2

theorem w_write (xs:Arr) (hx:(AllTimes cycle) xs) : Inv (Bound cycle) (Out.writeReg Register.agts_w xs) := by
 intro st hs
 change (Bound cycle) {st with regs:=st.regs.insert Register.agts_w xs}
 unfold Bound BoundView ValidOpt AgtExecutionFrameProofs.RView at *
 simpa [Std.ExtDHashMap.get?_insert] using And.intro hs.1 hx

theorem actual_agt_set_origin (q p:Nat) (a:agt) (hq:q<144) (ha:At cycle a) :
 Inv (Bound cycle) (agt_set q p a) := by
 unfold agt_set
 split
 · apply read_bind_bounded (Bound cycle) Register.agts (AllTimes cycle)
   · intro st hs xs hx
     exact hs.1 xs hx
   · intro xs hx
     exact r_write _ (update_times xs q a hq hx ha)
 · apply read_bind_bounded (Bound cycle) Register.agts_w (AllTimes cycle)
   · intro st hs xs hx
     exact hs.2 xs hx
   · intro xs hx
     exact w_write _ (update_times xs q a hq hx ha)

theorem get_bind_origin {α:Type} (q p:Nat) (hq:q<144) (f:agt→SailM α)
 (hf:∀a,At cycle a→Inv (Bound cycle) (f a)) :
 Inv (Bound cycle) (agt_get q p >>= f) := by
 unfold agt_get
 split
 · simp only [pure_bind,bind_assoc]
   apply read_bind_bounded (Bound cycle) Register.agts (AllTimes cycle)
   · intro st hs xs hx; exact hs.1 xs hx
   · intro xs hx; exact hf xs[q]! (hx q hq)
 · simp only [pure_bind,bind_assoc]
   apply read_bind_bounded (Bound cycle) Register.agts_w (AllTimes cycle)
   · intro st hs xs hx; exact hs.2 xs hx
   · intro xs hx; exact hf xs[q]! (hx q hq)

theorem actual_configure_origin (q p:Nat) (hq:q<144) (f:agt_field) (v:BitVec 16) :
 Inv (Bound cycle) (agt_configure q p f v) := by
 unfold agt_configure
 simp only [pure_bind]
 apply get_bind_origin q p hq
 intro a ha
 split
 · apply bind_inv
   · exact actual_agt_set_origin q p _ hq (configure_at a f v ha)
   · intro; exact pure_inv _ _
 · exact pure_inv _ _

theorem actual_refresh_origin (q p:Nat) (hq:q<144) : Inv (Bound cycle) (agt_refresh q p) := by
 unfold agt_refresh
 simp only [pure_bind]
 apply get_bind_origin q p hq
 intro a ha
 repeat' first
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact other_write _ _ (by decide) (by decide)
  | exact actual_agt_set_origin q p _ hq (refresh_at a ha)
  | apply bind_inv
  | split
  | intro

theorem actual_finish_origin (q p:Nat) (hq:q<144) : Inv (Bound cycle) (agt_finish_config q p) := by
 unfold agt_finish_config
 simp only [pure_bind]
 apply get_bind_origin q p hq
 intro a ha
 repeat' first
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact other_write _ _ (by decide) (by decide)
  | exact actual_agt_set_origin q p _ hq (finish_at a ha)
  | apply bind_inv
  | split
  | intro

theorem array_idle_at : (AllTimes cycle) (vectorInit agt_idle) := by
 intro i hi
 simpa [vectorInit,hi] using idle_at

theorem execution_origin {α:Type} (action:SailM α) (hf:Keeps AgtExecutionFrameProofs.RView action) :
 Inv (Bound cycle) action := keeps_pred AgtExecutionFrameProofs.RView (BoundView cycle) action hf

theorem actual_get_origin (s:St) (hs:(Bound cycle) s) (q p:Nat) (hq:q<144) (a:agt)
 (hget:(agt_get q p).run s=.ok a s) : At cycle a := by
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

theorem actual_hold_path_origin (q p c:Nat) : Inv (Bound cycle) (hold_path q p c) := by
 unfold hold_path
 exact bind_inv _ _ _ (read_inv _ _) (fun _=>other_write _ _ (by decide) (by decide))

theorem actual_bank_report_origin (ports:AgtExecutionFrameProofs.Ports) (q c:Nat)
 (a b:instruction×Int×Int) : Inv (Bound cycle) (bank_report q c a b) := by
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



theorem default_at : At cycle (default:agt):=by change (0:Nat)≤0 ∧ (0:Nat)≤cycle;omega

theorem array_get_at (xs:Arr) (hx:AllTimes cycle xs) (q:Nat):At cycle xs[q]!:=by
 by_cases hq:q<144
 · exact hx q hq
 · rw [getElem!_neg xs q hq]
   exact default_at

theorem update_times_any (xs:Arr) (q:Nat) (a:agt) (hx:AllTimes cycle xs) (ha:At cycle a):
 AllTimes cycle (vectorUpdate xs q a):=by
 by_cases hq:q<144
 · exact update_times xs q a hq hx ha
 · intro i hi
   change At cycle ((xs.setIfInBounds q a)[i]!)
   rw [getElem!_pos _ i hi]
   rw [Vector.getElem_setIfInBounds_ne (xs:=xs) (x:=a) hi (by omega)]
   have hxi:=hx i hi
   simpa only [getElem!_pos xs i hi] using hxi

theorem actual_agt_set_bound_any (q p:Nat) (a:agt) (ha:At cycle a):
 Inv (Bound cycle) (agt_set q p a):=by
 unfold agt_set
 split
 · apply read_bind_bounded (Bound cycle) Register.agts (AllTimes cycle)
   · intro st hs xs hx;exact hs.1 xs hx
   · intro xs hx;exact r_write _ (update_times_any xs q a hx ha)
 · apply read_bind_bounded (Bound cycle) Register.agts_w (AllTimes cycle)
   · intro st hs xs hx;exact hs.2 xs hx
   · intro xs hx;exact w_write _ (update_times_any xs q a hx ha)

theorem get_bind_bound_any {α:Type} (q p:Nat) (f:agt→SailM α)
 (hf:∀a,At cycle a→Inv (Bound cycle) (f a)):
 Inv (Bound cycle) (agt_get q p >>= f):=by
 unfold agt_get
 split
 · simp only [pure_bind,bind_assoc]
   apply read_bind_bounded (Bound cycle) Register.agts (AllTimes cycle)
   · intro st hs xs hx;exact hs.1 xs hx
   · intro xs hx;exact hf xs[q]! (array_get_at xs hx q)
 · simp only [pure_bind,bind_assoc]
   apply read_bind_bounded (Bound cycle) Register.agts_w (AllTimes cycle)
   · intro st hs xs hx;exact hs.2 xs hx
   · intro xs hx;exact hf xs[q]! (array_get_at xs hx q)

theorem actual_configure_bound_any (q p:Nat) (f:agt_field) (v:BitVec 16) :
 Inv (Bound cycle) (agt_configure q p f v) := by
 unfold agt_configure
 simp only [pure_bind]
 apply get_bind_bound_any q p
 intro a ha
 split
 · apply bind_inv
   · exact actual_agt_set_bound_any q p _ (configure_at a f v ha)
   · intro; exact pure_inv _ _
 · exact pure_inv _ _

theorem actual_refresh_bound_any (q p:Nat) : Inv (Bound cycle) (agt_refresh q p) := by
 unfold agt_refresh
 simp only [pure_bind]
 apply get_bind_bound_any q p
 intro a ha
 repeat' first
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact other_write _ _ (by decide) (by decide)
  | exact actual_agt_set_bound_any q p _ (refresh_at a ha)
  | apply bind_inv
  | split
  | intro

theorem actual_finish_bound_any (q p:Nat) : Inv (Bound cycle) (agt_finish_config q p) := by
 unfold agt_finish_config
 simp only [pure_bind]
 apply get_bind_bound_any q p
 intro a ha
 repeat' first
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact other_write _ _ (by decide) (by decide)
  | exact actual_agt_set_bound_any q p _ (finish_at a ha)
  | apply bind_inv
  | split
  | intro


theorem after_reference_at (a:agt) (ls:Vector agt_level 4) (off soff:Int)
 (ha:At cycle a):At cycle (agt_after_reference a ls off soff cycle):=by
 have ht:=ha.1
 have hl:=ha.2
 simp only [At,agt_after_reference]
 constructor <;> omega

theorem actual_run_loop_bound (ports:AgtExecutionFrameProofs.Ports)
 (q p:Nat) (step:Option (Vector agt_level 4×Nat×Int×Int))
 (blocked:Bool) (partner:Option (instruction×Int×Int)):
 Inv (Bound cycle) (run_loop q cycle p step blocked partner):=by
 rw [AgtOriginStateProofs.actual_run_loop_source]
 cases step with
 | none=>exact pure_inv _ _
 | some step=>
  rcases step with ⟨ls,delay,off,soff⟩
  apply get_bind_bound_any q p
  intro a ha
  unfold AgtOriginStateProofs.runTail loop_clash_ref
  repeat' first
   | exact pure_inv _ _
   | exact read_inv _ _
   | exact execution_origin _ (AgtExecutionFrameProofs.report_frame ports _ _ _)
   | exact execution_origin _ (AgtExecutionFrameProofs.agt_get_frame ports _ _)
   | exact execution_origin _ (AgtExecutionFrameProofs.run_element_frame ports _ _ _ _ _ _)
   | exact actual_bank_report_origin ports _ _ _ _
   | exact actual_hold_path_origin _ _ _
   | exact actual_agt_set_bound_any q p _ (after_reference_at a ls off soff ha)
   | exact actual_refresh_bound_any q p
   | apply bind_inv
   | split
   | intro

end AgtTimeBoundProofs
