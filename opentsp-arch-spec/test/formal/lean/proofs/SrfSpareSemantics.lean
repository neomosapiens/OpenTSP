-- Actual report visibility/replacement loops and source-connected query result.
import Proofs.SrfReportFlow
import Proofs.SrfUpstream
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions Out.Register Out.row
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 100000
namespace SrfSpareSemanticProofs
open ResourceReadLoopProofs ResourceScanProofs

theorem pure_loop_run {α:Type} (range:IntRange) (f:Int→α→α)
 (state:α) (i:Int) (hs:(i-range.start)%range.step=0) (s:ResourceFrameProofs.St) :
 (IntRange.forIn'.loop (m:=SailM) range (fun k _ a=>pure (.yield (f k a))) state i hs).run s=
 .ok (IntRange.forIn'.loop (m:=Id) range (fun k _ a=>pure (.yield (f k a))) state i hs) s := by
 induction state,i,hs using IntRange.forIn'.loop.induct range with
 | case1 state i hs hi ih =>
   rw [IntRange.forIn'.loop.eq_1,IntRange.forIn'.loop.eq_1]
   simp only [hi,dite_true,pure_bind]
   simpa only [EStateM.run,bind,EStateM.bind,pure,EStateM.pure] using ih (f i state)
 | case2 state i hs hi =>
   rw [IntRange.forIn'.loop.eq_1,IntRange.forIn'.loop.eq_1]
   simp [hi,EStateM.run,pure,EStateM.pure]

theorem pure_range_eq {α:Type} (range:IntRange) (f:Int→α→α) (state:α) :
 IntRange.forIn' (m:=SailM) range state (fun k _ a=>pure (.yield (f k a)))=
 pure (IntRange.forIn' (m:=Id) range state (fun k _ a=>pure (.yield (f k a)))) := by
 funext s
 exact pure_loop_run range f state range.start (by simp) s

def replaceStep (slot:srf_slot) (g:Nat) (d:Int) (e:srf_rep) (w:Int) (old:Bool) : Bool :=
 if (w>b e.wgap) && ((w<b g) && (slot[w]!.valid && (slot[w]!.diag==d))) then true else old

def replaced (slot:srf_slot) (g:Nat) (d:Int) (e:srf_rep) : Bool :=
 IntRange.forIn' (m:=Id) {start:=0,stop:=43,step:=1} false
  (fun w _ old=>pure (.yield (replaceStep slot g d e w old)))

def replaceAction (slot:srf_slot) (g:Nat) (d:Int) (e:srf_rep) : SailM Bool := do
 let mut old:=false
 for w in [0:43:1]i do
  old:=replaceStep slot g d e w old
 pure old

theorem actual_replace_pure (slot:srf_slot) (g:Nat) (d:Int) (e:srf_rep) :
 replaceAction slot g d e=pure (replaced slot g d e) := by
 unfold replaceAction replaced
 simp only [pure_bind,bind_pure]
 exact pure_range_eq _ _ _

def visible (rw:Bool) (g:Nat) (d:Int) (e:srf_rep) : Bool :=
 e.used && ((e.rw==rw) && ((e.diag==d) && (e.wgap<b g)))

def spareStep (rw:Bool) (slot:srf_slot) (s g:Nat) (d:Int)
 (rs:Vector srf_rep 256) (i:Int) (m:BitVec Out.Functions.superlanes) : BitVec Out.Functions.superlanes :=
 let e:=rs[i]!
 if visible rw g d e then (if !(replaced slot g d e) then m ||| e.sp[s]! else m) else m

def spareLoop (rw:Bool) (slot:srf_slot) (s g:Nat) (d:Int) : SailM (BitVec Out.Functions.superlanes) := do
 let mut m:=BitVec.zero Out.Functions.superlanes
 for i in [0:255:1]i do
  let e:=(← Out.readReg Register.srf_reps)[i]!
  if visible rw g d e then
   let skip←replaceAction slot g d e
   if !skip then m:=m ||| e.sp[s]!
 pure m

def spareValue (rw:Bool) (slot:srf_slot) (s g:Nat) (d:Int) (rs:Vector srf_rep 256) : BitVec Out.Functions.superlanes :=
 IntRange.forIn' (m:=Id) {start:=0,stop:=255,step:=1} (BitVec.zero Out.Functions.superlanes)
  (fun i _ m=>pure (.yield (spareStep rw slot s g d rs i m)))

theorem loop_value (st:ResourceFrameProofs.St) (rs:Vector srf_rep 256)
 (hr:st.regs.get? Register.srf_reps=some rs)
 (rw:Bool) (slot:srf_slot) (s g:Nat) (d:Int) :
 (spareLoop rw slot s g d).run st=.ok (spareValue rw slot s g d rs) st := by
 unfold spareLoop spareValue
 simp only [actual_replace_pure,pure_bind,bind_pure]
 simp only [show ∀(b:Bool) (u v:BitVec Out.Functions.superlanes), (if b then pure (.yield u) else pure (.yield v) : SailM (ForInStep (BitVec Out.Functions.superlanes)))=pure (.yield (if b then u else v)) from by intro b u v; cases b <;> rfl]
 change (IntRange.forIn' (m:=SailM) {start:=0,stop:=255,step:=1} (BitVec.zero Out.Functions.superlanes)
  (fun i _ m=>do let a←Out.readReg Register.srf_reps; pure (.yield (spareStep rw slot s g d a i m)))).run st=_
 exact read_range_run _ _ _ _ _ _ hr

def spareFactory (r:row) (s g:Nat) (t:Int) : SailM (BitVec Out.Functions.superlanes) := do
 let n←Out.readReg Register.srf_rep_n
 if (n>b 0) && (s<b 30) then
  let d:=(g:Int)-t
  let slot:=(←get_stream r s)[slot_of d]!
  spareLoop (r==RowW) slot s g d
 else pure (BitVec.zero Out.Functions.superlanes)

theorem state_bind_pure {α:Type} (a:SailM α) : (a >>= fun x=>pure x)=a := by
 funext st
 cases ha:a st <;> simp [bind,EStateM.bind,pure,EStateM.pure,ha]

theorem state_pure_bind {α β:Type} (x:α) (f:α→SailM β) : (pure x >>= f)=f x := by rfl

theorem state_bind_assoc {α β γ:Type} (a:SailM α) (f:α→SailM β) (g:β→SailM γ) :
 ((a >>= f) >>= g)=(a >>= fun x=>f x >>= g) := by
 funext st
 cases ha:a st <;> simp [bind,EStateM.bind,ha]

theorem prop_if_bind {α β:Type} (p:Prop) [Decidable p] (a b:SailM α) (f:α→SailM β) :
 ((if p then a else b) >>= f)=(if p then a >>= f else b >>= f) := by
 split <;> rfl

theorem bool_if_bind {α β:Type} (p:Bool) (a b:SailM α) (f:α→SailM β) :
 ((if p then a else b) >>= f)=(if p then a >>= f else b >>= f) := by
 cases p <;> rfl

section
attribute [local irreducible] IntRange.forIn'
theorem actual_spare_source (r:row) (s g:Nat) (t:Int) :
 srf_spare r s g t=spareFactory r s g t := by
 unfold srf_spare spareFactory spareLoop replaceAction replaceStep visible
 simp only [pure_bind,bind_pure,bind_assoc,bool_if_bind,prop_if_bind,state_bind_pure,state_pure_bind,state_bind_assoc,Out.Functions.not]
 congr 3
 funext n
 split
 · congr 1
   funext stream
   exact state_bind_pure _
 · rfl
end

def rowStream (east west:Vector srf_stream 32) (r:row) (s:Nat) : srf_stream :=
 match r with | .RowE=>east[s]! | .RowW=>west[s]!

theorem actual_get_stream_run (st:ResourceFrameProofs.St) (east west:Vector srf_stream 32)
 (he:st.regs.get? Register.srf_e=some east) (hw:st.regs.get? Register.srf_w=some west)
 (r:row) (s:Nat) : (get_stream r s).run st=.ok (rowStream east west r s) st := by
 cases r <;> simp [get_stream,rowStream,Out.readReg,PreSail.readReg,
 EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,he,hw]

theorem actual_spare_run (st:ResourceFrameProofs.St) (east west:Vector srf_stream 32)
 (rs:Vector srf_rep 256) (n:Nat)
 (he:st.regs.get? Register.srf_e=some east) (hw:st.regs.get? Register.srf_w=some west)
 (hr:st.regs.get? Register.srf_reps=some rs) (hn:st.regs.get? Register.srf_rep_n=some n)
 (r:row) (s g:Nat) (t:Int) :
 (srf_spare r s g t).run st=.ok
 (if 0<n ∧ s<30 then
  spareValue (r==RowW) (rowStream east west r s)[slot_of ((g:Int)-t)]! s g ((g:Int)-t) rs
  else BitVec.zero Out.Functions.superlanes) st := by
 have hread:Out.readReg Register.srf_rep_n st=.ok n st := read_present st _ _ hn
 have hstream:get_stream r s st=.ok (rowStream east west r s) st := actual_get_stream_run st east west he hw r s
 have hloop:spareLoop (r==RowW) (rowStream east west r s)[slot_of ((g:Int)-t)]! s g ((g:Int)-t) st=
  .ok (spareValue (r==RowW) (rowStream east west r s)[slot_of ((g:Int)-t)]! s g ((g:Int)-t) rs) st :=
  loop_value st rs hr _ _ _ _ _
 rw [actual_spare_source]
 by_cases hp:0<n ∧ s<30
 all_goals simp [spareFactory,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hread,hstream,hloop,hp]
 all_goals rfl

theorem actual_replaced_iff (slot:srf_slot) (g:Nat) (d:Int) (e:srf_rep) :
 replaced slot g d e=true ↔
 ∃w:Int,0<=w ∧ w<44 ∧ (e.wgap:Int)<w ∧ w<g ∧ slot[w]!.valid=true ∧ slot[w]!.diag=d := by
 unfold replaced
 rw [actual_indexed_range_fold _ _ _ rfl]
 have h:=SrfUpstreamProofs.scan_spec (replaceStep slot g d e) (fun b=>b)
  (fun w=>(e.wgap:Int)<w ∧ w<g ∧ slot[w]!.valid=true ∧ slot[w]!.diag=d)
  (by intro w old; cases old <;> simp [replaceStep, Bool.and_eq_true, Bool.or_eq_true])
  0 44 false
 simpa using h

def visibleBit (rw:Bool) (slot:srf_slot) (s g:Nat) (d:Int) (rs:Vector srf_rep 256) (x:Nat) (i:Int) : Prop :=
 visible rw g d rs[i]! =true ∧ replaced slot g d rs[i]! =false ∧ rs[i]!.sp[s]!.getLsbD x=true

theorem spare_bit_step (rw:Bool) (slot:srf_slot) (s g:Nat) (d:Int) (rs:Vector srf_rep 256)
 (x:Nat) (i:Int) (m:BitVec Out.Functions.superlanes) :
 (spareStep rw slot s g d rs i m).getLsbD x=true ↔ m.getLsbD x=true ∨ visibleBit rw slot s g d rs x i := by
 unfold spareStep visibleBit
 by_cases hv:visible rw g d rs[i]! =true
 · by_cases hr:replaced slot g d rs[i]! =true
   · simp [hv,hr]
   · have hf:replaced slot g d rs[i]! =false:=by cases h:replaced slot g d rs[i]! <;> simp_all
     simp only [hv,hf,ite_true,Bool.not_false,Bool.false_eq_true,and_self,and_true]
     erw [BitVec.getLsbD_or]
     simp
     rfl
 · have hf:visible rw g d rs[i]! =false:=by cases h:visible rw g d rs[i]! <;> simp_all
   simp [hf]

theorem actual_spare_bit_iff (rw:Bool) (slot:srf_slot) (s g:Nat) (d:Int) (rs:Vector srf_rep 256) (x:Nat) :
 (spareValue rw slot s g d rs).getLsbD x=true ↔
 ∃i:Int,0<=i ∧ i<256 ∧ visibleBit rw slot s g d rs x i := by
 unfold spareValue
 rw [actual_indexed_range_fold _ _ _ rfl]
 have h:=SrfUpstreamProofs.scan_spec (spareStep rw slot s g d rs) (fun m=>m.getLsbD x)
  (visibleBit rw slot s g d rs x)
  (fun i m=>spare_bit_step rw slot s g d rs x i m) 0 256 (BitVec.zero Out.Functions.superlanes)
 simpa using h

def spareQuery (east west:Vector srf_stream 32) (rs:Vector srf_rep 256) (n:Nat)
 (r:row) (s g:Nat) (t:Int) : BitVec Out.Functions.superlanes :=
 if 0<n ∧ s<30 then
  spareValue (r==RowW) (rowStream east west r s)[slot_of ((g:Int)-t)]! s g ((g:Int)-t) rs
 else BitVec.zero Out.Functions.superlanes

def lookupValue (east west:Vector srf_stream 32) (rs:Vector srf_rep 256) (n:Nat)
 (r:row) (s g:Nat) (t:Int) : srf_lookup × vec × ecc_ov :=
 let stream:=rowStream east west r s
 let (found,value,ov):=srf_slot_read_ov stream[slot_of ((g:Int)-t)]! g ((g:Int)-t)
 if found then (.SRF_FOUND,value,srf_ov_spare ov (spareQuery east west rs n r s g t))
 else if srf_stream_only_downstream stream g t then (.SRF_UPSTREAM,zero_vec,ecc_ov_zero)
 else (.SRF_NONE,zero_vec,ecc_ov_zero)

theorem actual_read_run (st:ResourceFrameProofs.St) (east west:Vector srf_stream 32)
 (rs:Vector srf_rep 256) (n:Nat)
 (he:st.regs.get? Register.srf_e=some east) (hw:st.regs.get? Register.srf_w=some west)
 (hr:st.regs.get? Register.srf_reps=some rs) (hn:st.regs.get? Register.srf_rep_n=some n)
 (r:row) (s g:Nat) (t:Int) :
 (srf_read_ov r s g t).run st=.ok (lookupValue east west rs n r s g t) st := by
 have hstream:get_stream r s st=.ok (rowStream east west r s) st := actual_get_stream_run st east west he hw r s
 have hspare:srf_spare r s g t st=.ok (spareQuery east west rs n r s g t) st :=
  actual_spare_run st east west rs n he hw hr hn r s g t
 rcases hx:srf_slot_read_ov (rowStream east west r s)[slot_of ((g:Int)-t)]! g ((g:Int)-t) with ⟨found,value,ov⟩
 cases found
 all_goals simp [lookupValue,srf_read_ov,srf_only_downstream,
  EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hstream,hspare,hx]
 all_goals first | rfl | (cases hd:srf_stream_only_downstream (rowStream east west r s) g t <;> simp [hd] <;> rfl)

theorem actual_chain_run (st:ResourceFrameProofs.St) (east west:Vector srf_stream 32)
 (he:st.regs.get? Register.srf_e=some east) (hw:st.regs.get? Register.srf_w=some west)
 (r:row) (s g:Nat) (t:Int) :
 (srf_read_chain_ov r s g t).run st=.ok
 (let (found,value,ov):=srf_slot_chain_ov (rowStream east west r s)[slot_of ((g:Int)-t+1)]! g ((g:Int)-t+1)
  if found then (.SRF_FOUND,value,ov) else (.SRF_NONE,zero_vec,ecc_ov_zero)) st := by
 have hstream:get_stream r s st=.ok (rowStream east west r s) st := actual_get_stream_run st east west he hw r s
 rcases hx:srf_slot_chain_ov (rowStream east west r s)[slot_of ((g:Int)-t+1)]! g ((g:Int)-t+1) with ⟨found,value,ov⟩
 cases found
 all_goals simp [srf_read_chain_ov,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hstream,hx]
 all_goals rfl

end SrfSpareSemanticProofs
