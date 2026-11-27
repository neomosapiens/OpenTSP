-- Presence of every typed model register after the actual generated initializer.
import Out
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 100000
namespace MachineInitProofs
abbrev St := SequentialState RegisterType trivialChoiceSource

def Complete (s : St) : Prop := ∀ r : Register, ∃ v, s.regs.get? r = some v

theorem actual_init_succeeds (s : St) :
    ∃ after, (sail_model_init ()).run s = EStateM.Result.ok () after := by
  exact ⟨_, rfl⟩

theorem actual_init_complete (s : St) :
    ∃ after, (sail_model_init ()).run s = EStateM.Result.ok () after ∧ Complete after := by
  refine ⟨_, rfl, ?_⟩
  intro r
  cases r <;> simp [Std.ExtDHashMap.get?_insert]

theorem complete_insert (s : St) (r : Register) (v : RegisterType r)
    (hc : Complete s) : Complete {s with regs := s.regs.insert r v} := by
  intro k
  by_cases he : k = r
  · subst k
    exact ⟨v, by simp [Std.ExtDHashMap.get?_insert]⟩
  · obtain ⟨w,hw⟩ := hc k
    exact ⟨w, by simpa [Std.ExtDHashMap.get?_insert,he,Ne.symm he] using hw⟩

theorem actual_read_defined (s : St) (hc : Complete s) (r : Register) :
    ∃ v, (Out.readReg r).run s = EStateM.Result.ok v s := by
  obtain ⟨v,hv⟩ := hc r
  exact ⟨v, by simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hv]⟩

theorem actual_write_complete (s : St) (hc : Complete s) (r : Register) (v : RegisterType r) :
    ∃ after, (Out.writeReg r v).run s = EStateM.Result.ok () after ∧ Complete after := by
  exact ⟨_,rfl,complete_insert s r v hc⟩

theorem actual_init_scrub (s : St) :
    ∃ after, (sail_model_init ()).run s = EStateM.Result.ok () after ∧
      after.regs.get? Register.scrub_next = some 0 ∧
      after.regs.get? Register.scrub_ptr = some (vectorInit 0) ∧
      after.regs.get? Register.scrub_pending = some (vectorInit none) := by
  refine ⟨_,rfl,?_,?_,?_⟩
  all_goals simp [Std.ExtDHashMap.get?_insert]

#print axioms actual_init_succeeds
#print axioms actual_init_complete
#print axioms complete_insert
#print axioms actual_read_defined
#print axioms actual_write_complete
#print axioms actual_init_scrub
end MachineInitProofs
