import Proofs.ScrubCatchup
import Proofs.PatrolCycleFrames
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions ScrubCatchupProofs
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace EventScrubPrefixProofs
abbrev St := SequentialState RegisterType trivialChoiceSource

def count (n p : Nat) (u : Int) := (scrub_due_count n p u).toNat

theorem count_last (n p : Nat) (u : Int) (hk:0<count n p u) :
    (n:Int)+((count n p u-1:Nat):Int)*(p:Int)≤u := by
  by_cases hu:0≤u
  · exact due_count_last n p u hu hk
  · have hz:=negative_count_zero n p u (by omega)
    change count n p u=0 at hz
    omega

theorem count_unique (start p : Nat) (u : Int) (n k : Nat)
    (fn:u<(start:Int)+(n:Int)*(p:Int))
    (fk:u<(start:Int)+(k:Int)*(p:Int))
    (ln:0<n→(start:Int)+((n-1:Nat):Int)*(p:Int)≤u)
    (lk:0<k→(start:Int)+((k-1:Nat):Int)*(p:Int)≤u) : n=k := by
  apply Classical.byContradiction
  intro hn
  have hcases:n<k ∨ k<n:=by omega
  rcases hcases with hnk|hkn
  · have hk:0<k:=by omega
    have hm:=Int.mul_le_mul_of_nonneg_right
      (show (n:Int)≤((k-1:Nat):Int) by omega) (Int.natCast_nonneg p)
    have hl:=lk hk
    omega
  · have hn:0<n:=by omega
    have hm:=Int.mul_le_mul_of_nonneg_right
      (show (k:Int)≤((n-1:Nat):Int) by omega) (Int.natCast_nonneg p)
    have hl:=ln hn
    omega

theorem count_split (start p : Nat) (a b : Int) (hp:0<p) (hab:a ≤ b) :
    count start p b=count start p a+count (start+count start p a*p) p b := by
  let na:=count start p a
  let nb:=count (start+na*p) p b
  have hfront:=due_count_frontier start p b hp
  have hshift:=due_count_frontier (start+na*p) p b hp
  have hsum:((start+na*p:Nat):Int)+(nb:Int)*(p:Int)=
      (start:Int)+((na+nb:Nat):Int)*(p:Int):=by
    simp only [Int.natCast_add,Int.natCast_mul,Int.add_mul,Int.add_assoc]
  apply count_unique start p b (count start p b) (na+nb)
  · exact hfront
  · change b<((start+na*p:Nat):Int)+(nb:Int)*(p:Int) at hshift
    rw [hsum] at hshift
    exact hshift
  · exact count_last start p b
  · intro hn
    by_cases hb:0<nb
    · have hl:=count_last (start+na*p) p b hb
      have he:na+nb-1=na+(nb-1):=by omega
      rw [he]
      simp only [Int.natCast_add,Int.add_mul]
      simpa only [Int.natCast_add,Int.natCast_mul,Int.add_assoc] using hl
    · have hz:nb=0:=by omega
      have ha:0<na:=by omega
      have hl:=count_last start p a ha
      simp only [hz,Nat.add_zero]
      exact Int.le_trans hl hab

theorem actual_step_cursor
    (ports : PatrolCycleFrameProofs.RawRegisterPorts Register.scrub_next)
    (s after : St) (n : Nat) (hr:s.regs.get? Register.scrub_next=some n)
    (hs:patrol_step.run s=.ok () after) :
    after.regs.get? Register.scrub_next=some (n+scrub_period) := by
  have hread:Out.readReg Register.scrub_next s=.ok n s:=by
    simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
      get,getThe,MonadStateOf.get,EStateM.get,hr]
  have hframe:=PatrolCycleFrameProofs.actual_patrol_cycle_scrub_next ports n s
  unfold patrol_step at hs
  simp only [EStateM.run,bind,EStateM.bind,hread] at hs
  cases hc:scrub_cycle n s with
  | error e mid=>simp [hc] at hs
  | ok x mid=>
    cases x
    have hm:mid.regs.get? Register.scrub_next=some n:=by
      simpa only [ResourceFrameProofs.RegKeeps,ResourceFrameProofs.Keeps,EStateM.run,hc,hr] using hframe
    simp [hc,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,EStateM.run,
      bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hm] at hs
    change EStateM.Result.ok () {mid with regs:=mid.regs.insert Register.scrub_next (n+scrub_period)}=
      EStateM.Result.ok () after at hs
    have he: {mid with regs:=mid.regs.insert Register.scrub_next (n+scrub_period)}=after := (EStateM.Result.ok.inj hs).2
    rw [←he]
    simp [Std.ExtDHashMap.get?_insert]

theorem repeated_cursor
    (ports : PatrolCycleFrameProofs.RawRegisterPorts Register.scrub_next)
    (s after : St) (n k : Nat) (hr:s.regs.get? Register.scrub_next=some n)
    (hs:(repeated patrol_step k).run s=.ok () after) :
    after.regs.get? Register.scrub_next=some (n+k*scrub_period) := by
  induction k generalizing s n with
  | zero=>
    have he:s=after:=by simpa [repeated,EStateM.run,pure,EStateM.pure] using hs
    simpa [he] using hr
  | succ k ih=>
    simp only [repeated,EStateM.run,bind,EStateM.bind] at hs
    cases hfirst:patrol_step s with
    | error e mid=>simp [hfirst] at hs
    | ok x mid=>
      cases x
      simp only [hfirst] at hs
      have hm:=actual_step_cursor ports s mid n hr hfirst
      have hlast:=ih mid (n+scrub_period) hm hs
      simpa [Nat.succ_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hlast

theorem actual_run_from_cursor (s : St) (n : Nat) (u : Int)
    (hr:s.regs.get? Register.scrub_next=some n) :
    (scrub_run u).run s=(repeated patrol_step (count n scrub_period u)).run s := by
  rw [scrub_run_production]
  simp [count,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
    get,getThe,MonadStateOf.get,EStateM.get,hr]

-- Equality includes actual effect order, error identity and retained partial state.
theorem actual_scrub_composes
    (ports : PatrolCycleFrameProofs.RawRegisterPorts Register.scrub_next)
    (s : St) (n : Nat) (a b : Int) (hr:s.regs.get? Register.scrub_next=some n)
    (hab:a ≤ b) :
    ((do scrub_run a; scrub_run b) : SailM Unit).run s=(scrub_run b).run s := by
  by_cases hp:0<scrub_period
  · rw [actual_run_from_cursor s n b hr,count_split n scrub_period a b hp hab,repeated_add]
    have hfirst:scrub_run a s=(repeated patrol_step (count n scrub_period a)) s:=actual_run_from_cursor s n a hr
    simp only [EStateM.run,bind,EStateM.bind,hfirst]
    cases hx:(repeated patrol_step (count n scrub_period a)) s with
    | error e mid=>rfl
    | ok x mid=>
      cases x
      have hm:=repeated_cursor ports s mid n (count n scrub_period a) hr hx
      exact actual_run_from_cursor mid (n+count n scrub_period a*scrub_period) b hm
  · have hz:scrub_period=0:=by omega
    have ha:count n scrub_period a=0:=by simp [count,scrub_due_count,hz]
    have hb:count n scrub_period b=0:=by simp [count,scrub_due_count,hz]
    have h1:scrub_run a s=.ok () s:=by
      simpa only [ha,repeated,EStateM.run,pure,EStateM.pure] using actual_run_from_cursor s n a hr
    have h2:scrub_run b s=.ok () s:=by
      simpa only [hb,repeated,EStateM.run,pure,EStateM.pure] using actual_run_from_cursor s n b hr
    simp [EStateM.run,bind,EStateM.bind,h1]

#print axioms actual_run_from_cursor
#print axioms actual_scrub_composes

#print axioms actual_step_cursor
#print axioms repeated_cursor

#print axioms count_last
#print axioms count_unique
#print axioms count_split
end EventScrubPrefixProofs
