-- Actual report search, retention and architectural eviction-warning flow.
import Proofs.ResourceReadLoops
import Proofs.SrfNearest
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions Out.Register Out.row
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfReportFlowProofs
open ResourceReadLoopProofs ResourceScanProofs
def findReport (rw : Bool) (g : Nat) (d : Int) : SailM Int := do
  let at' : Int := -1
  (( do
    let loop_i_lower := 0
    let loop_i_upper := 255
    let mut loop_vars := at'
    for i in [loop_i_lower:loop_i_upper:1]i do
      let at' := loop_vars
      loop_vars ← do
        let e ← do (pure (GetElem?.getElem! (← readReg srf_reps) i))
        if ((e.used && ((e.rw == rw) && ((e.diag == d) && ((e.wgap == g) : Bool)))) : Bool)
        then (pure i)
        else (pure at')
    (pure loop_vars) ) : SailM Int )

def newReport (c : Nat) (e : srf_rep) : SailM Bool := (do
      let slot ← do (pure (Int.emod (← readReg srf_rep_n) 256))
      let old ← do (pure (GetElem?.getElem! (← readReg srf_reps) slot))
      writeReg srf_reps (vectorUpdate (← readReg srf_reps) slot e)
      writeReg srf_rep_n ((← readReg srf_rep_n) +i 1)
      (pure (old.used && ((old.diag +i c) ≤b 44))))

def reportFactory (r : row) (g : Nat) (c : Nat) (prio : Int) (xs : (BitVec Out.Functions.superlanes)) (sp : (Vector (BitVec Out.Functions.superlanes) 30)) : SailM Bool := do
  let d := (g -i c)
  let rw := (r == RowW)
  let at' : Int := (Neg.neg 1)
  let at' ← findReport rw g d
  let e0 ← (( do
    if ((at' ≥b 0) : Bool)
    then (pure (GetElem?.getElem! (← readReg srf_reps) (Int.emod at' 256)))
    else
      (pure { used := true
              rw := rw
              diag := d
              wgap := g
              sp := (vectorInit (BitVec.zero Out.Functions.superlanes))
              pr := (vectorInit 0) }) ) : SailM srf_rep )
  let e := (srf_rep_merge e0 prio xs sp)
  if ((at' ≥b 0) : Bool)
  then
    (do
      writeReg srf_reps (vectorUpdate (← readReg srf_reps) (Int.emod at' 256) e)
      (pure false))
  else
    newReport c e
section
attribute [local irreducible] IntRange.forIn'
theorem actual_report_factory (r : row) (g c : Nat) (prio : Int) (xs : BitVec Out.Functions.superlanes) (sp : Vector (BitVec Out.Functions.superlanes) 30) :
 srf_report r g c prio xs sp=reportFactory r g c prio xs sp := by
 unfold srf_report reportFactory findReport newReport
 rfl
end

def keyMatch (rs : Vector srf_rep 256) (rw : Bool) (g : Nat) (d i : Int) : Bool :=
 let e:=rs[i]!
 e.used && ((e.rw==rw) && ((e.diag==d) && (e.wgap==g)))
def findStep (rs : Vector srf_rep 256) (rw : Bool) (g : Nat) (d i best : Int) : Int :=
 if keyMatch rs rw g d i then i else best

def findValue (rs : Vector srf_rep 256) (rw : Bool) (g : Nat) (d : Int) : Int :=
 IntRange.forIn' (m:=Id) {start:=0,stop:=255,step:=1} (-1)
  (fun i _ best => pure (.yield (findStep rs rw g d i best)))

theorem if_yield (p : Bool) (i best : Int) :
 (if p then (pure (.yield i) : SailM (ForInStep Int)) else pure (.yield best))=
 pure (.yield (if p then i else best)) := by cases p <;> rfl

theorem actual_find_value (s : ResourceFrameProofs.St) (rs : Vector srf_rep 256)
 (hr:s.regs.get? Register.srf_reps=some rs) (rw : Bool) (g : Nat) (d : Int) :
 (findReport rw g d).run s=.ok (findValue rs rw g d) s := by
 simp only [findReport,pure_bind,bind_pure,if_yield]
 change (IntRange.forIn' (m:=SailM) {start:=0,stop:=255,step:=1} (-1)
  (fun i _ best => do let x ← Out.readReg Register.srf_reps; pure (.yield (findStep x rw g d i best)))).run s=_
 exact read_range_run _ _ _ _ _ _ hr

theorem find_next_winner (rs : Vector srf_rep 256) (rw : Bool) (g : Nat) (d n best : Int)
 (hn:0≤n) (h:SrfNearestProofs.Winner (keyMatch rs rw g d) n best) :
 SrfNearestProofs.Winner (keyMatch rs rw g d) (n+1) (findStep rs rw g d n best) := by
 have he:findStep rs rw g d n best=SrfNearestProofs.chooseStep (keyMatch rs rw g d) n best := by
   unfold findStep SrfNearestProofs.chooseStep
   simp [show n > best from h.2.1]
 rw [he]
 exact SrfNearestProofs.next_winner _ _ _ hn h

theorem find_fold_winner (rs : Vector srf_rep 256) (rw : Bool) (g : Nat) (d : Int)
 (n : Nat) (start best : Int) (hs:0≤start) (hb:SrfNearestProofs.Winner (keyMatch rs rw g d) start best) :
 SrfNearestProofs.Winner (keyMatch rs rw g d) (start+n) (indexedFold (findStep rs rw g d) start n best) := by
 induction n generalizing start best with
 | zero => simpa [indexedFold] using hb
 | succ n ih =>
   have hp:=find_next_winner rs rw g d start best hs hb
   have hf:=ih (start+1) _ (by omega) hp
   simpa [indexedFold,Int.natCast_add,Int.natCast_one,Int.add_assoc,Int.add_comm,Int.add_left_comm] using hf

theorem actual_find_winner (rs : Vector srf_rep 256) (rw : Bool) (g : Nat) (d : Int) :
 SrfNearestProofs.Winner (keyMatch rs rw g d) 256 (findValue rs rw g d) := by
 unfold findValue
 rw [actual_indexed_range_fold _ _ _ rfl]
 exact find_fold_winner rs rw g d 256 0 (-1) (by decide) (SrfNearestProofs.initial_winner _)

def newState (s : ResourceFrameProofs.St) (rs : Vector srf_rep 256) (n : Nat) (e : srf_rep) : ResourceFrameProofs.St :=
 {s with regs:=(s.regs.insert Register.srf_reps (vectorUpdate rs (Int.emod n 256) e)).insert Register.srf_rep_n (n+1)}

theorem actual_new_retention (s : ResourceFrameProofs.St) (rs : Vector srf_rep 256) (n : Nat)
 (hr:s.regs.get? Register.srf_reps=some rs) (hn:s.regs.get? Register.srf_rep_n=some n)
 (c : Nat) (e : srf_rep) :
 (newReport c e).run s=.ok (rs[Int.emod n 256]!.used && decide (rs[Int.emod n 256]!.diag+c≤44)) (newState s rs n e) := by
 simp [newReport,newState,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
 EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
 modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hr,hn]

def selectedReport (rs:Vector srf_rep 256) (r:row) (g c:Nat) (prio:Int)
 (xs:BitVec Out.Functions.superlanes) (sp:Vector (BitVec Out.Functions.superlanes) 30) : srf_rep :=
 let chosen:=findValue rs (r==RowW) g ((g:Int)-c)
 let base:=if chosen>=0 then rs[Int.emod chosen 256]!
  else {used:=true,rw:=(r==RowW),diag:=(g:Int)-c,wgap:=g,sp:=vectorInit (BitVec.zero _),pr:=vectorInit 0}
 srf_rep_merge base prio xs sp

theorem actual_report_run (s:ResourceFrameProofs.St) (rs:Vector srf_rep 256) (n:Nat)
 (hr:s.regs.get? Register.srf_reps=some rs) (hn:s.regs.get? Register.srf_rep_n=some n)
 (r:row) (g c:Nat) (prio:Int) (xs:BitVec Out.Functions.superlanes)
 (sp:Vector (BitVec Out.Functions.superlanes) 30) :
 (srf_report r g c prio xs sp).run s=
 let chosen:=findValue rs (r==RowW) g ((g:Int)-c)
 let e:=selectedReport rs r g c prio xs sp
 if chosen>=0 then .ok false {s with regs:=s.regs.insert Register.srf_reps (vectorUpdate rs (Int.emod chosen 256) e)}
 else .ok (rs[Int.emod n 256]!.used && decide (rs[Int.emod n 256]!.diag+c<=44)) (newState s rs n e) := by
 have hf:findReport (r==RowW) g ((g:Int)-c) s=.ok (findValue rs (r==RowW) g ((g:Int)-c)) s :=
  actual_find_value s rs hr (r==RowW) g ((g:Int)-c)
 rw [actual_report_factory]
 by_cases hchosen:0<=findValue rs (r==RowW) g ((g:Int)-c)
 all_goals simp [reportFactory,selectedReport,newReport,newState,hchosen,Std.ExtDHashMap.get?_insert,
  Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
  EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
  modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,hf,hr,hn]
 all_goals rfl

end SrfReportFlowProofs
