-- Entire actual commit arbitration/application appends exactly the successful stored writers.
import Proofs.SrfCommittedHistory
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfCommitJournalProofs
open SrfCommittedHistoryProofs
abbrev St:=ResourceFrameProofs.St
abbrev Histories:=List Write×List Write

def HasRows (st:St) (h:Histories):Prop:=rowValue .RowE st=some (bankHistory h.1) ∧ rowValue .RowW st=some (bankHistory h.2)

def selectedWrite (c:Nat) (e:commit) (v:vec) (ov:ecc_ov):Write:={stream:=e.id,gap:=e.g,cycle:=c,value:=v,ov:=ov,queue:=e.q,plane:=e.b}

def record (r:row) (w:Write) (h:Histories):Histories:=match r with
 | .RowE=>(w::h.1,h.2)
 | .RowW=>(h.1,w::h.2)

def journal (c:Nat) (all:List commit):List commit→Nat→Histories→Histories
 | [],_,h=>h
 | e::rest,i,h=>
   let (_,win,v,ov):=commit_wins e i all
   journal c all rest (i+1) (if win then record e.r (selectedWrite c e v ov) h else h)

theorem actual_successful_write_journal (st:St) (h:Histories) (hh:HasRows st h)
 (r:row) (w:Write):
 ∃collision after,(srf_write_ov r w.stream w.gap w.cycle w.value w.ov).run st=.ok collision after ∧
 HasRows after (record r w h):=by
 cases r
 · have hw:=actual_write_stores .RowE st (bankHistory h.1) w hh.1
   refine ⟨_,_,hw,?_⟩
   constructor
   · simp [rowValue,rowState,record,bankHistory,Std.ExtDHashMap.get?_insert]
   · simpa [rowValue,rowState,record,Std.ExtDHashMap.get?_insert] using hh.2
 · have hw:=actual_write_stores .RowW st (bankHistory h.2) w hh.2
   refine ⟨_,_,hw,?_⟩
   constructor
   · simpa [rowValue,rowState,record,Std.ExtDHashMap.get?_insert] using hh.1
   · simp [rowValue,rowState,record,bankHistory,Std.ExtDHashMap.get?_insert]

theorem actual_commit_apply_journal (c:Nat) (all xs:List commit) (i:Nat)
 (hits:Vector Nat 145) (st:St) (h:Histories) (hh:HasRows st h):
 ∃counts after,(commit_apply c all xs i hits).run st=.ok counts after ∧
 HasRows after (journal c all xs i h):=by
 induction xs generalizing i hits st h with
 | nil=>exact ⟨hits,st,rfl,hh⟩
 | cons e rest ih=>
   rcases hw:commit_wins e i all with ⟨collide,win,v,ov⟩
   let newHits:Vector Nat 145:=if collide then vectorUpdate hits e.q (hits[e.q]!+1) else hits
   cases hwin:win
   · obtain ⟨counts,after,hr,hs⟩:=ih (i+1) newHits st h hh
     refine ⟨counts,after,?_,?_⟩
     · have hhreal:commit_apply c all rest (i+1) newHits st=.ok counts after:=hr
       simp [commit_apply,hw,hwin,newHits,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hhreal]
     · simpa [journal,hw,hwin] using hs
   · obtain ⟨collision,written,hrw,hhw⟩:=actual_successful_write_journal st h hh e.r (selectedWrite c e v ov)
     obtain ⟨counts,after,hr,hs⟩:=ih (i+1) newHits written (record e.r (selectedWrite c e v ov) h) hhw
     refine ⟨counts,after,?_,?_⟩
     · have hhreal:commit_apply c all rest (i+1) newHits written=.ok counts after:=hr
       have hww:srf_write_ov e.r e.id e.g c v ov st=.ok collision written:=hrw
       simp [commit_apply,hw,hwin,newHits,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hww,hhreal]
     · simpa [journal,hw,hwin] using hs

end SrfCommitJournalProofs
