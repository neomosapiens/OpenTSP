-- Fault injection transforms the retained writer, preserving its origin and eligibility.
import Proofs.SrfCommitJournal
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfFaultHistoryProofs
open ResourceFrameProofs SrfCommittedHistoryProofs SrfLifetimeProofs
abbrev St:=ResourceFrameProofs.St

def corrupt (e:Write) (sl n:Nat):Write:=
 let v:vec:=if n < 128 then e.value ^^^ (get_slice_int (8*Out.Functions.vlanes).toNat 1 0 <<< ((128:Int)*sl+n)) else e.value
 let ov:ecc_ov:=ecc_ov_set e.ov sl (ecc_ov_get e.ov sl ^^^ ecc_flip_delta n)
 {e with value:=v,ov:=ov}

def faultBank (bank:Bank) (s gp:Nat) (d:Int) (sl n:Nat):Bank:=
 vectorUpdate bank s (vectorUpdate bank[s]! (slot_of d) (srf_slot_flip bank[s]![slot_of d]! gp d sl n))

theorem fault_origin (e:Write) (sl n:Nat):
 (corrupt e sl n).queue=e.queue ∧ (corrupt e sl n).plane=e.plane ∧
 (corrupt e sl n).cycle=e.cycle ∧ (corrupt e sl n).baseValue=e.baseValue ∧
 (corrupt e sl n).baseOv=e.baseOv ∧ diagonal (corrupt e sl n)=diagonal e:=by
 exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem retained_when_valid (history:List Write) (hv:∀e,e∈history→Valid e)
 (s k g:Nat) (hs:s<32) (hk:k<64) (hg:g<44)
 (valid:(bankHistory history)[s]![k]![g]!.valid=true):
 ∃e,retained history s k g=some e ∧ word e=(bankHistory history)[s]![k]![g]! ∧
 e∈history ∧ keyMatches e s k g:=by
 have hc:=actual_history_cell history hv s k g hs hk hg
 cases hr:retained history s k g with
 | none=>rw [hr] at hc;rw [hc] at valid;cases valid
 | some e=>
   rw [hr] at hc
   obtain ⟨hm,hmatch⟩:=retained_member history s k g e hr
   exact ⟨e,rfl,hc.symm,hm,hmatch⟩

theorem actual_fault_retained (history:List Write) (hv:∀e,e∈history→Valid e)
 (s gp sl n:Nat) (d:Int) (hs:s<32) (hsl:sl<Out.Functions.superlanes)
 (found:0≤srf_slot_best (bankHistory history)[s]![slot_of d]! gp d):
 ∃e,e∈history ∧ e.stream=s ∧ diagonal e=d ∧
 faultBank (bankHistory history) s gp d sl n=bankHistory (corrupt e sl n::history):=by
 let best:=srf_slot_best (bankHistory history)[s]![slot_of d]! gp d
 have hb:=SrfNearestProofs.actual_found_eligible (bankHistory history)[s]![slot_of d]! gp d found
 have heleg:=hb.2.2
 unfold SrfNearestProofs.eligible at heleg
 have hi:best.toNat<44:=by omega
 have hic:0 ≤ best:=found
 have hmod:(Int.emod best 44).toNat=best.toNat:=by
   change (best%44).toNat=best.toNat
   rw [Int.emod_eq_of_lt hic hb.2.1]
 have hidx:(GetElem?.getElem! (bankHistory history)[s]![slot_of d]! best)=
  (bankHistory history)[s]![slot_of d]![best.toNat]!:=rfl
 rw [hidx] at heleg
 simp only [Bool.and_eq_true,beq_iff_eq,decide_eq_true_eq] at heleg
 have hvcell:(bankHistory history)[s]![slot_of d]![best.toNat]!.valid=true:=by
  exact heleg.1
 obtain ⟨e,hr,hw,hm,hkey⟩:=retained_when_valid history hv s (slot_of d) best.toNat hs (srf_slot_range d).2 hi hvcell
 have hgap:e.gap=best.toNat:=hkey.2.2
 have hd:diagonal e=d:=by
  have hdiag:=heleg.2.1
  rw [←hw] at hdiag
  exact hdiag
 refine ⟨e,hm,hkey.1,hd,?_⟩
 change faultBank (bankHistory history) s gp d sl n=store (bankHistory history) (corrupt e sl n)
 unfold faultBank store
 have hcs:(corrupt e sl n).stream=s:=hkey.1
 have hcd:diagonal (corrupt e sl n)=d:=hd
 rw [hcs,hcd]
 congr 2
 unfold srf_slot_flip
 simp only [show srf_slot_best (bankHistory history)[s]![slot_of d]! gp d=best from rfl,
  show decide (best ≥ 0)=true from by exact decide_eq_true found,show decide (sl<Out.Functions.superlanes)=true from by simp [hsl],Bool.true_and,if_true]
 rw [hmod,←hw,←hgap]
 unfold srf_slot_write word corrupt
 by_cases hn:n<128
 all_goals simp only [hn,decide_true,decide_false,if_true,if_false,hd]
 all_goals rfl

theorem vector_updated_self {α:Type} [Inhabited α] {N:Nat} (xs:Vector α N) (k:Nat) (hk:k<N):
 vectorUpdate xs k xs[k]! = xs:=by
 apply Vector.ext
 intro i hi
 have hh:=vector_updated_get xs i k xs[k]! hi hk
 rw [getElem!_pos _ i hi,getElem!_pos xs i hi] at hh
 by_cases he:i=k
 · subst i;simpa [getElem!_pos xs k hk] using hh
 · simpa [he] using hh

theorem actual_fault_no_value (bank:Bank) (s gp sl n:Nat) (d:Int) (hs:s<32)
 (nonefound:srf_slot_best bank[s]![slot_of d]! gp d<0):
 faultBank bank s gp d sl n=bank:=by
 unfold faultBank srf_slot_flip
 simp only [show decide (srf_slot_best bank[s]![slot_of d]! gp d≥0)=false from by simp;omega,Bool.false_and,Bool.false_eq_true,if_false]
 rw [vector_updated_self _ _ (srf_slot_range d).2,vector_updated_self _ _ hs]

theorem actual_fault_outside_lanes (bank:Bank) (s gp sl n:Nat) (d:Int) (hs:s<32)
 (hsl:Out.Functions.superlanes≤sl):faultBank bank s gp d sl n=bank:=by
 unfold faultBank srf_slot_flip
 simp only [show decide (sl<Out.Functions.superlanes)=false from by simp;omega,
  Bool.and_false,Bool.false_eq_true,if_false]
 rw [vector_updated_self _ _ (srf_slot_range d).2,vector_updated_self _ _ hs]

theorem actual_fault_bank_run (r:row) (st:St) (bank:Bank) (s gp c sl n:Nat)
 (hr:rowValue r st=some bank):
 (srf_flip c r s gp sl n).run st=.ok () (rowState r st (faultBank bank s gp ((gp:Int)-c) sl n)):=by
 cases r
 all_goals simp [rowValue,rowState,faultBank,srf_flip,get_stream,set_stream,
  Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
  get,getThe,MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet] at hr ⊢
 all_goals simp [hr,EStateM.pure]
 all_goals rfl

end SrfFaultHistoryProofs
