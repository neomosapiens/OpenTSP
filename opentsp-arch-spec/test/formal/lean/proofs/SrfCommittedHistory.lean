-- Retained committed writers, connected to actual SRF storage and upstream selection.
import Proofs.SrfSpareSemantics
import Proofs.SrfLifetime
import Proofs.ResourceFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfCommittedHistoryProofs
open ResourceFrameProofs SrfLifetimeProofs
abbrev St:=ResourceFrameProofs.St
abbrev Bank:=Vector srf_stream 32

structure Write where
 stream:Nat
 gap:Nat
 cycle:Nat
 value:vec
 ov:ecc_ov
 queue:Nat:=0
 plane:Nat:=0
 baseValue:vec:=value
 baseOv:ecc_ov:=ov

def diagonal (e:Write):Int:=(e.gap:Int)-e.cycle

def word (e:Write):Out.srf_write:={valid:=true,diag:=diagonal e,wgap:=e.gap,value:=e.value,ov:=e.ov}

def store (bank:Bank) (e:Write):Bank:=
 let stream:=bank[e.stream]!
 let k:=slot_of (diagonal e)
 vectorUpdate bank e.stream (vectorUpdate stream k (srf_slot_write stream[k]! e.gap (diagonal e) e.value e.ov))

def emptyBank:Bank:=vectorInit (vectorInit (vectorInit empty_write))

def bankHistory:List Write→Bank
 | []=>emptyBank
 | e::history=>store (bankHistory history) e

def keyMatches (e:Write) (stream slot gap:Nat):Prop:=e.stream=stream ∧ slot_of (diagonal e)=slot ∧ e.gap=gap
instance (e:Write) (s k g:Nat):Decidable (keyMatches e s k g):=inferInstanceAs (Decidable (_∧_∧_))

def retained:List Write→Nat→Nat→Nat→Option Write
 | [],_,_,_=>none
 | e::history,s,k,g=>if keyMatches e s k g then some e else retained history s k g

def Valid (e:Write):Prop:=e.stream<32 ∧ e.gap<44

theorem vector_updated_get {α:Type} [Inhabited α] {N:Nat} (xs:Vector α N) (i j:Nat) (v:α)
 (hi:i<N) (hj:j<N): (vectorUpdate xs j v)[i]! = if i=j then v else xs[i]!:=by
 by_cases same:i=j
 · subst i;simp [vectorUpdate,Vector.set!,hj]
 · rw [vector_update_other N i j xs v hi hj same,if_neg same]

theorem actual_store_cell (bank:Bank) (e:Write) (he:Valid e)
 (s k g:Nat) (hs:s<32) (hk:k<64) (hg:g<44):
 (store bank e)[s]![k]![g]! = if keyMatches e s k g then word e else bank[s]![k]![g]!:=by
 unfold store
 rw [vector_updated_get bank s e.stream _ hs he.1]
 by_cases hstream:s=e.stream
 · simp only [hstream,if_true]
   rw [vector_updated_get _ k (slot_of (diagonal e)) _ hk (srf_slot_range _).2]
   by_cases hslot:k=slot_of (diagonal e)
   · simp only [hslot,if_true,srf_slot_write]
     rw [vector_updated_get _ g e.gap _ hg he.2]
     by_cases hgap:g=e.gap
     · simp [keyMatches,word,hstream,hslot,hgap]
     · simp [keyMatches,hstream,hslot,Ne.symm hgap,hgap]
   · simp [keyMatches,hstream,hslot,Ne.symm hslot]
 · simp [keyMatches,hstream,Ne.symm hstream]

theorem actual_history_cell (history:List Write) (hv:∀e,e∈history→Valid e)
 (s k g:Nat) (hs:s<32) (hk:k<64) (hg:g<44):
 (bankHistory history)[s]![k]![g]! =
 match retained history s k g with | some e=>word e | none=>empty_write:=by
 induction history with
 | nil=>simp [bankHistory,emptyBank,vectorInit,hs,hk,hg,retained]
 | cons e history ih=>
   have he:=hv e (by simp)
   rw [bankHistory,actual_store_cell _ e he s k g hs hk hg]
   simp only [retained]
   split
   · rfl
   · exact ih (fun e he=>hv e (List.mem_cons_of_mem _ he))

theorem retained_member (history:List Write) (s k g:Nat) (e:Write)
 (hr:retained history s k g=some e):e∈history ∧ keyMatches e s k g:=by
 induction history with
 | nil=>simp [retained] at hr
 | cons h tail ih=>
   unfold retained at hr
   split at hr
   · have he:h=e:=Option.some.inj hr
     subst e
     exact ⟨by simp,by assumption⟩
   · obtain ⟨hm,hk⟩:=ih hr
     exact ⟨List.mem_cons_of_mem _ hm,hk⟩

theorem actual_selected_retained_writer (history:List Write) (hv:∀e,e∈history→Valid e)
 (s:Nat) (hs:s<32) (reader:Nat) (d:Int)
 (found:0≤srf_slot_best (bankHistory history)[s]![slot_of d]! reader d):
 ∃e,e∈history ∧ e.stream=s ∧ e.gap<reader ∧ diagonal e=d ∧
  srf_slot_read_ov (bankHistory history)[s]![slot_of d]! reader d=(true,e.value,e.ov) ∧
  ∀w:Int,0≤w→w<44→SrfNearestProofs.eligible (bankHistory history)[s]![slot_of d]! reader d w=true→w≤e.gap:=by
 let best:=srf_slot_best (bankHistory history)[s]![slot_of d]! reader d
 have hb:=SrfNearestProofs.actual_found_eligible (bankHistory history)[s]![slot_of d]! reader d found
 have hcell:=actual_history_cell history hv s (slot_of d) best.toNat hs (srf_slot_range d).2 (by omega)
 have hidx:(GetElem?.getElem! (bankHistory history)[s]![slot_of d]! best)=
   (bankHistory history)[s]![slot_of d]![best.toNat]!:=rfl
 have heleg:=hb.2.2
 unfold SrfNearestProofs.eligible at heleg
 rw [hidx,hcell] at heleg
 cases hr:retained history s (slot_of d) best.toNat with
 | none=>simp [hr,empty_write] at heleg
 | some e=>
   simp only [hr,word,Bool.true_and,Bool.and_eq_true,beq_iff_eq,decide_eq_true_eq] at heleg
   obtain ⟨hm,hmatch⟩:=retained_member history s (slot_of d) best.toNat e hr
   have hgap:e.gap=best.toNat:=hmatch.2.2
   refine ⟨e,hm,hmatch.1,?_,heleg.1,?_,?_⟩
   · omega
   · rw [SrfNearestProofs.actual_slot_read_found _ _ _ found,hidx,hcell,hr]
     rfl
   · intro w hw hl hel
     have hmax:=(SrfNearestProofs.actual_nearest _ reader d).2.2.2 w hw hl hel
     omega

def rowValue (r:row) (st:St):Option Bank:=match r with
 | .RowE=>st.regs.get? Register.srf_e
 | .RowW=>st.regs.get? Register.srf_w

def rowState (r:row) (st:St) (bank:Bank):St:=match r with
 | .RowE=>{st with regs:=st.regs.insert Register.srf_e bank}
 | .RowW=>{st with regs:=st.regs.insert Register.srf_w bank}

theorem actual_write_stores (r:row) (st:St) (bank:Bank) (e:Write)
 (hr:rowValue r st=some bank):
 (srf_write_ov r e.stream e.gap e.cycle e.value e.ov).run st=
 .ok (srf_slot_collides bank[e.stream]![slot_of (diagonal e)]! e.gap (diagonal e))
  (rowState r st (store bank e)):=by
 cases r
 all_goals simp [rowValue,rowState,store,diagonal,srf_write_ov,get_stream,set_stream,
  Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
  get,getThe,MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet] at hr ⊢
 all_goals simp [hr,EStateM.pure]
 all_goals rfl

theorem actual_committed_write_preserves_history (r:row) (st:St) (history:List Write) (e:Write)
 (hr:rowValue r st=some (bankHistory history)):
 ∃collision after,(srf_write_ov r e.stream e.gap e.cycle e.value e.ov).run st=.ok collision after ∧
 rowValue r after=some (bankHistory (e::history)):=by
 refine ⟨_,rowState r st (store (bankHistory history) e),actual_write_stores r st (bankHistory history) e hr,?_⟩
 cases r <;> simp [rowValue,rowState,bankHistory,Std.ExtDHashMap.get?_insert]

end SrfCommittedHistoryProofs
