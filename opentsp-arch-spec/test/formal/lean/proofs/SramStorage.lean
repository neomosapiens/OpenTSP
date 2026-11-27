-- Shared IQ/SRAM external storage witness; exact disjoint namespaces/cells.
import Proofs.StorageRep
import Proofs.ResourceGeometry
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
namespace SramStorageProofs
open Proofs.StorageRep ResourceFrameProofs ResourceGeometryProofs

theorem key_field4 (count field slot value : Nat) (hf : field<4) :
    cellKey count field slot value%4=field := by unfold cellKey; omega

theorem key_slot4 (count field slot value : Nat) (hf : field<4) (hs : slot<count) :
    (cellKey count field slot value/4)%count=slot := by
  have hb : cellKey count field slot value/4=value*count+slot := by unfold cellKey; omega
  rw [hb]
  simp [Nat.add_mod,Nat.mod_eq_of_lt hs]

theorem write_foreign_key (state : ResourceFrameProofs.St) (count field slot value key : Nat)
    (hf : field<4) (hk : key%4≠field) :
    (writeCell state count field slot value).mem.get? key=state.mem.get? key := by
  have hne : cellKey count field slot value≠key := by
    intro he
    have h:=key_field4 count field slot value hf
    rw [he] at h
    exact hk h
  simp only [writeCell,Std.ExtHashMap.get?_eq_getElem?,Std.ExtHashMap.getElem?_insert,
    Std.ExtHashMap.getElem?_filter']
  simp [hne,Ne.symm hne,hk]
  cases state.mem[key]? <;> rfl

theorem write_other_slot_key (state : ResourceFrameProofs.St) (count field slot value key : Nat)
    (hf : field<4) (hs : slot<count) (hk : (key/4)%count≠slot) :
    (writeCell state count field slot value).mem.get? key=state.mem.get? key := by
  have hne : cellKey count field slot value≠key := by
    intro he
    have h:=key_slot4 count field slot value hf hs
    rw [he] at h
    exact hk h
  simp only [writeCell,Std.ExtHashMap.get?_eq_getElem?,Std.ExtHashMap.getElem?_insert,
    Std.ExtHashMap.getElem?_filter']
  simp [hne,Ne.symm hne,hk]
  cases state.mem[key]? <;> rfl

def SliceView (words q : Nat) (state : ResourceFrameProofs.St) (field : Fin 2) (addr : Fin words) (value : Nat) :=
  state.mem.get? (cellKey (88*words) (2+field.val) (q*words+addr.val) value)

theorem iq_write_preserves_sram (state : ResourceFrameProofs.St) (words q count field slot value : Nat)
    (hf : field<2) : SliceView words q (writeCell state count field slot value)=SliceView words q state := by
  funext f a v
  apply write_foreign_key
  · omega
  · have h:=key_field4 (88*words) (2+f.val) (q*words+a.val) v (by omega)
    omega

theorem sram_write_preserves_iq_key (state : ResourceFrameProofs.St) (count field slot value key : Nat)
    (hf : field=2 ∨ field=3) (hk : key%4<2) :
    (writeCell state count field slot value).mem.get? key=state.mem.get? key := by
  apply write_foreign_key <;> omega

theorem other_slice_preserved (state : ResourceFrameProofs.St) (words q slice addr field value : Nat)
    (hw : 0<words) (hq : q<88) (hs : slice<88) (ha : addr<words) (hne : slice≠q)
    (hf : field=2 ∨ field=3) :
    SliceView words q (writeCell state (88*words) field (slice*words+addr) value)=SliceView words q state := by
  funext f a v
  apply write_other_slot_key
  · omega
  · exact flat_index_bound words slice addr hw hs ha
  · rw [key_slot4 _ _ _ _ (by omega) (flat_index_bound words q a.val hw hq a.isLt)]
    intro he
    have h:=flat_index_injective words q slice a.val addr hw a.isLt ha he
    exact hne h.1.symm

theorem write_register_preserves_memory (words q : Nat) (r : Register) (v : RegisterType r) :
    Keeps (SliceView words q) (Out.writeReg r v) := by
  apply ResourceFrameProofs.write_keeps
  intro state
  rfl
end SramStorageProofs
