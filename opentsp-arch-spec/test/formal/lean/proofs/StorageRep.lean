-- Proof-only external IQ storage embedding in the native sequential state's unused mem field.
import Proofs.Storage
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Storage
namespace Proofs.StorageRep

def zigzag : Int → Nat
  | .ofNat n => 2*n
  | .negSucc n => 2*n+1

def unzigzag (n : Nat) : Int :=
  if n%2=0 then .ofNat (n/2) else .negSucc (n/2)

theorem unzigzag_zigzag (x : Int) : unzigzag (zigzag x) = x := by
  cases x <;> simp only [zigzag,unzigzag]
  all_goals
    split
    all_goals congr 1 <;> omega

def cellKey (count field slot value : Nat) : Nat := (value*count+slot)*4+field

theorem key_bucket (count field slot value : Nat) (hf : field < 2) :
    cellKey count field slot value / 4 = value*count+slot := by
  unfold cellKey
  omega

theorem key_field (count field slot value : Nat) (hf : field < 2) :
    cellKey count field slot value % 4 = field := by
  unfold cellKey
  omega

theorem key_slot (count field slot value : Nat) (hf : field < 2) (hs : slot < count) :
    (cellKey count field slot value / 4) % count = slot := by
  rw [key_bucket count field slot value hf]
  simp [Nat.add_mod,Nat.mod_eq_of_lt hs]

theorem key_value (count field slot value : Nat) (hf : field < 2) (hs : slot < count) :
    (cellKey count field slot value / 4) / count = value := by
  rw [key_bucket count field slot value hf]
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega)]
  simp [Nat.div_eq_of_lt hs]

theorem key_injective (count f g i j a b : Nat)
    (hf : f<2) (hg : g<2) (hi : i<count) (hj : j<count)
    (h : cellKey count f i a = cellKey count g j b) :
    f=g ∧ i=j ∧ a=b := by
  constructor
  · have h1 := key_field count f i a hf
    rw [h] at h1
    rw [key_field count g j b hg] at h1
    exact h1.symm
  constructor
  · have h1 := key_slot count f i a hf hi
    rw [h] at h1
    rw [key_slot count g j b hg hj] at h1
    exact h1.symm
  · have h1 := key_value count f i a hf hi
    rw [h] at h1
    rw [key_value count g j b hg hj] at h1
    exact h1.symm

def Present (s : St) (count field slot value : Nat) : Prop :=
  s.mem.get? (cellKey count field slot value) = some (1#8)

theorem present_blind (s : St) (regs) (count field slot value : Nat) :
    Present {s with regs := regs} count field slot value ↔ Present s count field slot value := by rfl

def Consistent (s : St) (count : Nat) : Prop :=
  ∀ field slot, field < 2 → slot < count → ∀ a b,
    Present s count field slot a → Present s count field slot b → a=b

noncomputable def chooseValue (P : Nat → Prop) : Option Nat := by
  classical
  exact if h : ∃ n, P n then some (Classical.choose h) else none

noncomputable def readValue (s : St) (count field slot : Nat) : Option Nat :=
  chooseValue (fun n => Present s count field slot n)

theorem readValue_unique (s : St) (count field slot value : Nat)
    (hc : Consistent s count) (hf : field < 2) (hs : slot < count)
    (hp : Present s count field slot value) :
    readValue s count field slot = some value := by
  classical
  unfold readValue chooseValue
  split
  · rename_i h
    rw [hc field slot hf hs _ value (Classical.choose_spec h) hp]
  · rename_i h
    exact False.elim (h ⟨value,hp⟩)

theorem readValue_blind (s : St) (regs) (count field slot : Nat) :
    readValue {s with regs := regs} count field slot = readValue s count field slot := by rfl

noncomputable def units (cap : Nat) (s : St) (q slot : Nat) : BitVec 32 :=
  BitVec.ofNat 32 ((readValue s (145*cap) 0 (q*cap+slot)).getD 0)

noncomputable def tags (cap : Nat) (s : St) (q slot : Nat) : Int :=
  ((readValue s (145*cap) 1 (q*cap+slot)).map unzigzag).getD (-1)

def Initialized (cap : Nat) (s : St) (q slot : Nat) : Prop :=
  ∃ value, Present s (145*cap) 0 (q*cap+slot) value

theorem views_blind (cap : Nat) (s : St) regs :
    units cap {s with regs := regs} = units cap s ∧ tags cap {s with regs := regs} = tags cap s := by
  exact ⟨rfl,rfl⟩

/-- The C side supplies these actual array/GMP values and a ghost initialized-cell mask.
    The mask records successful unit stores, not the contents of uninitialized malloc bytes. -/
structure CView where
  initialized : Nat → Nat → Prop
  unit : Nat → Nat → BitVec 32
  tag : Nat → Nat → Int

/-- Explicit external-store relation; model register correspondence is a separate product
    component. The generated opaque callbacks do not prove this port contract themselves. -/
def Represents (cap : Nat) (c : CView) (s : St) : Prop :=
  Consistent s (145*cap) ∧ ∀ q slot, q<145 → slot<cap →
    (c.initialized q slot ↔ Initialized cap s q slot) ∧
    (c.initialized q slot → readValue s (145*cap) 0 (q*cap+slot) = some (c.unit q slot).toNat) ∧
    tags cap s q slot = c.tag q slot

theorem represents_blind (cap : Nat) (c : CView) (s : St) regs :
    Represents cap c {s with regs := regs} ↔ Represents cap c s := by rfl

theorem represented_unit (cap : Nat) (c : CView) (s : St) (hr : Represents cap c s)
    (q slot : Nat) (hq : q<145) (hs : slot<cap) (hi : c.initialized q slot) :
    units cap s q slot = c.unit q slot := by
  have he := (hr.2 q slot hq hs).2.1 hi
  simp [units,he]

theorem represented_tag (cap : Nat) (c : CView) (s : St) (hr : Represents cap c s)
    (q slot : Nat) (hq : q<145) (hs : slot<cap) :
    tags cap s q slot = c.tag q slot :=
  (hr.2 q slot hq hs).2.2

theorem readValue_ext (s t : St) (count field slot : Nat)
    (h : ∀ n, Present s count field slot n ↔ Present t count field slot n) :
    readValue s count field slot = readValue t count field slot := by
  apply congrArg chooseValue
  funext n
  exact propext (h n)

theorem empty_read (s : St) (count field slot : Nat) :
    readValue {s with mem := ∅} count field slot = none := by
  classical
  simp [readValue,chooseValue,Present]

def emptyC : CView := ⟨fun _ _ => False, fun _ _ => 0#32, fun _ _ => -1⟩

theorem empty_represents (cap : Nat) (s : St) :
    Represents cap emptyC {s with mem := ∅} := by
  constructor
  · intro field slot hf hs a b ha
    simp [Present] at ha
  · intro q slot hq hs
    constructor
    · simp [emptyC,Initialized,Present]
    constructor
    · simp [emptyC]
    · simp [tags,empty_read,emptyC]

def writeCell (s : St) (count field slot value : Nat) : St :=
  {s with mem := (s.mem.filter (fun key _ =>
    !(decide (key%4=field) && decide ((key/4)%count=slot)))).insert (cellKey count field slot value) (1#8)}

theorem present_write (s : St) (count f g i j a b : Nat)
    (hf : f<2) (hg : g<2) (hi : i<count) (hj : j<count) :
    Present (writeCell s count f i a) count g j b ↔
      if f=g ∧ i=j then a=b else Present s count g j b := by
  have ke : cellKey count f i a = cellKey count g j b ↔ f=g ∧ i=j ∧ a=b := by
    constructor
    · exact key_injective count f g i j a b hf hg hi hj
    · rintro ⟨rfl,rfl,rfl⟩; rfl
  simp only [Present,writeCell,Std.ExtHashMap.get?_eq_getElem?,
    Std.ExtHashMap.getElem?_insert,Std.ExtHashMap.getElem?_filter']
  rw [key_field count g j b hg,key_slot count g j b hg hj]
  by_cases hcell : f=g ∧ i=j
  · obtain ⟨rfl,rfl⟩ := hcell
    simp [ke]
  · by_cases hfg : f=g <;> by_cases hij : i=j <;> simp_all [beq_iff_eq]
    all_goals first
      | exact fun _ => Ne.symm hij
      | exact fun _ => Ne.symm hfg
      | exact fun _ => Or.inl (Ne.symm hfg)

theorem write_consistent (s : St) (count f i a : Nat)
    (hc : Consistent s count) (hf : f<2) (hi : i<count) :
    Consistent (writeCell s count f i a) count := by
  intro g j hg hj x y hx hy
  rw [present_write s count f g i j a x hf hg hi hj] at hx
  rw [present_write s count f g i j a y hf hg hi hj] at hy
  by_cases hcell : f=g ∧ i=j
  · simp only [if_pos hcell] at hx hy
    exact hx.symm.trans hy
  · simp only [if_neg hcell] at hx hy
    exact hc g j hg hj x y hx hy

theorem read_written (s : St) (count f i a : Nat)
    (hc : Consistent s count) (hf : f<2) (hi : i<count) :
    readValue (writeCell s count f i a) count f i = some a := by
  apply readValue_unique _ _ _ _ _ (write_consistent s count f i a hc hf hi) hf hi
  rw [present_write s count f f i i a a hf hf hi hi]
  simp

theorem read_other (s : St) (count f g i j a : Nat)
    (hf : f<2) (hg : g<2) (hi : i<count) (hj : j<count)
    (hne : ¬(f=g ∧ i=j)) :
    readValue (writeCell s count f i a) count g j = readValue s count g j := by
  apply readValue_ext
  intro b
  rw [present_write s count f g i j a b hf hg hi hj]
  simp [hne]

theorem present_write_cross (s : St) (count f g i j a b : Nat)
    (hf : f<2) (hg : g<2) (hne : f≠g) :
    Present (writeCell s count f i a) count g j b ↔ Present s count g j b := by
  have hk : cellKey count f i a ≠ cellKey count g j b := by
    intro he
    have h1 := key_field count f i a hf
    rw [he,key_field count g j b hg] at h1
    exact hne h1.symm
  simp only [Present,writeCell,Std.ExtHashMap.get?_eq_getElem?,
    Std.ExtHashMap.getElem?_insert,Std.ExtHashMap.getElem?_filter']
  rw [key_field count g j b hg]
  simp [hk,Ne.symm hne]

theorem read_cross (s : St) (count f g i j a : Nat)
    (hf : f<2) (hg : g<2) (hne : f≠g) :
    readValue (writeCell s count f i a) count g j = readValue s count g j := by
  apply readValue_ext
  intro b
  exact present_write_cross s count f g i j a b hf hg hne

theorem readValue_exact (s : St) (count field slot value : Nat)
    (h : ∀ n, Present s count field slot n ↔ value=n) :
    readValue s count field slot = some value := by
  classical
  unfold readValue chooseValue
  split
  · rename_i hp
    have he := (h (Classical.choose hp)).1 (Classical.choose_spec hp)
    exact congrArg some he.symm
  · rename_i hp
    exact False.elim (hp ⟨value,(h value).2 rfl⟩)

theorem read_written_any (s : St) (count f i a : Nat) (hf : f<2) (hi : i<count) :
    readValue (writeCell s count f i a) count f i = some a := by
  apply readValue_exact
  intro n
  rw [present_write s count f f i i a n hf hf hi hi]
  simp

theorem slot_bound (cap q slot : Nat) (hq : q<145) (hs : slot<cap) :
    q*cap+slot < 145*cap := by
  calc
    q*cap+slot < q*cap+cap := Nat.add_lt_add_left hs _
    _ = (q+1)*cap := by rw [Nat.add_mul]; simp
    _ ≤ 145*cap := Nat.mul_le_mul_right cap (by omega)

theorem initialized_blind (cap : Nat) (s : St) regs (q slot : Nat) :
    Initialized cap {s with regs := regs} q slot ↔ Initialized cap s q slot := by rfl

theorem initialized_cross (cap : Nat) (s : St) (slot value q j : Nat) :
    Initialized cap (writeCell s (145*cap) 1 slot value) q j ↔ Initialized cap s q j := by
  unfold Initialized
  constructor
  · rintro ⟨v,hv⟩
    exact ⟨v,(present_write_cross s (145*cap) 1 0 slot (q*cap+j) value v (by decide) (by decide) (by decide)).1 hv⟩
  · rintro ⟨v,hv⟩
    exact ⟨v,(present_write_cross s (145*cap) 1 0 slot (q*cap+j) value v (by decide) (by decide) (by decide)).2 hv⟩

theorem initialized_written (cap : Nat) (s : St) (q slot value : Nat)
    (hq : q<145) (hs : slot<cap) :
    Initialized cap (writeCell s (145*cap) 0 (q*cap+slot) value) q slot := by
  refine ⟨value,?_⟩
  rw [present_write s (145*cap) 0 0 (q*cap+slot) (q*cap+slot) value value
    (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap q slot hq hs)]
  simp

theorem initialized_preserved (cap : Nat) (s : St) (q slot value r j : Nat)
    (hq : q<145) (hs : slot<cap) (hr : r<145) (hj : j<cap)
    (old : Initialized cap s r j) :
    Initialized cap (writeCell s (145*cap) 0 (q*cap+slot) value) r j := by
  rcases old with ⟨v,hv⟩
  by_cases he : q*cap+slot = r*cap+j
  · refine ⟨value,?_⟩
    rw [present_write s (145*cap) 0 0 (q*cap+slot) (r*cap+j) value value
      (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj)]
    simp [he]
  · refine ⟨v,?_⟩
    rw [present_write s (145*cap) 0 0 (q*cap+slot) (r*cap+j) value v
      (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj)]
    simpa [he] using hv

theorem flat_injective (cap q r i j : Nat) (hi : i<cap) (hj : j<cap)
    (he : q*cap + i=r*cap+j) : q=r ∧ i=j := by
  have hcap : 0<cap := by omega
  have qi : (q*cap + i)/cap=q := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hcap]
    simp [Nat.div_eq_of_lt hi]
  have rj : (r*cap+j)/cap=r := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hcap]
    simp [Nat.div_eq_of_lt hj]
  have mi : (q*cap + i)%cap=i := by simp [Nat.add_mod,Nat.mod_eq_of_lt hi]
  have mj : (r*cap+j)%cap=j := by simp [Nat.add_mod,Nat.mod_eq_of_lt hj]
  constructor
  · rw [he,rj] at qi; exact qi.symm
  · rw [he,mj] at mi; exact mi.symm

theorem initialized_write_iff (cap : Nat) (s : St) (q slot value r j : Nat)
    (hq : q<145) (hs : slot<cap) (hr : r<145) (hj : j<cap) :
    Initialized cap (writeCell s (145*cap) 0 (q*cap+slot) value) r j ↔
      if q=r ∧ slot=j then True else Initialized cap s r j := by
  have he : q*cap+slot=r*cap+j ↔ q=r ∧ slot=j := by
    constructor
    · exact flat_injective cap q r slot j hs hj
    · rintro ⟨rfl,rfl⟩; rfl
  unfold Initialized
  simp only [present_write s (145*cap) 0 0 (q*cap+slot) (r*cap+j) value _
    (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj)]
  by_cases target : q=r ∧ slot=j
  · simp [he,target]
  · simp [he,target]

def cUnitWrite (c : CView) (q slot : Nat) (value : BitVec 32) : CView :=
  ⟨fun r j => if q=r ∧ slot=j then True else c.initialized r j,
   fun r j => if q=r ∧ slot=j then value else c.unit r j,
   c.tag⟩

def cTagWrite (c : CView) (q slot : Nat) (value : Int) : CView :=
  ⟨c.initialized,c.unit,fun r j => if q=r ∧ slot=j then value else c.tag r j⟩

/-- Kernel-checkable projection step for the C indexed unit assignment plus its ghost
    initialization mark. Matching this tiny step to C/allocator semantics is trusted. -/
theorem unit_represents (cap : Nat) (c : CView) (s : St) (hrep : Represents cap c s)
    (q slot : Nat) (value : BitVec 32) (hq : q<145) (hs : slot<cap) :
    Represents cap (cUnitWrite c q slot value)
      (writeCell s (145*cap) 0 (q*cap+slot) value.toNat) := by
  constructor
  · exact write_consistent s _ _ _ _ hrep.1 (by decide) (slot_bound cap q slot hq hs)
  · intro r j hr hj
    have hi := (hrep.2 r j hr hj).1
    have hu := (hrep.2 r j hr hj).2.1
    have ht := (hrep.2 r j hr hj).2.2
    constructor
    · rw [initialized_write_iff cap s q slot value.toNat r j hq hs hr hj]
      simpa [cUnitWrite] using (by split <;> simp_all : (if q=r ∧ slot=j then True else c.initialized r j) ↔ (if q=r ∧ slot=j then True else Initialized cap s r j))
    constructor
    · intro live
      by_cases target : q=r ∧ slot=j
      · obtain ⟨rfl,rfl⟩ := target
        simpa [cUnitWrite] using read_written_any s (145*cap) 0 (q*cap+slot) value.toNat
          (by decide) (slot_bound cap q slot hq hs)
      · have diff : q*cap+slot≠r*cap+j := by intro he; exact target (flat_injective cap q r slot j hs hj he)
        rw [read_other s (145*cap) 0 0 (q*cap+slot) (r*cap+j) value.toNat
          (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj) (by simpa using diff)]
        have old : c.initialized r j := by simpa only [cUnitWrite,if_neg target] using live
        simpa [cUnitWrite,target] using hu old
    · unfold tags
      rw [read_cross s (145*cap) 0 1 (q*cap+slot) (r*cap+j) value.toNat (by decide) (by decide) (by decide)]
      exact ht

theorem tag_represents (cap : Nat) (c : CView) (s : St) (hrep : Represents cap c s)
    (q slot : Nat) (value : Int) (hq : q<145) (hs : slot<cap) :
    Represents cap (cTagWrite c q slot value)
      (writeCell s (145*cap) 1 (q*cap+slot) (zigzag value)) := by
  constructor
  · exact write_consistent s _ _ _ _ hrep.1 (by decide) (slot_bound cap q slot hq hs)
  · intro r j hr hj
    have hi := (hrep.2 r j hr hj).1
    have hu := (hrep.2 r j hr hj).2.1
    have ht := (hrep.2 r j hr hj).2.2
    constructor
    · rw [initialized_cross]
      exact hi
    constructor
    · intro live
      rw [read_cross s (145*cap) 1 0 (q*cap+slot) (r*cap+j) (zigzag value) (by decide) (by decide) (by decide)]
      exact hu live
    · unfold tags
      by_cases target : q=r ∧ slot=j
      · obtain ⟨rfl,rfl⟩ := target
        rw [read_written_any s (145*cap) 1 (q*cap+slot) (zigzag value)
          (by decide) (slot_bound cap q slot hq hs)]
        simp [unzigzag_zigzag,cTagWrite]
      · have diff : q*cap+slot≠r*cap+j := by intro he; exact target (flat_injective cap q r slot j hs hj he)
        rw [read_other s (145*cap) 1 1 (q*cap+slot) (r*cap+j) (zigzag value)
          (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj) (by simpa using diff)]
        simpa [cTagWrite,target] using ht

end Proofs.StorageRep
