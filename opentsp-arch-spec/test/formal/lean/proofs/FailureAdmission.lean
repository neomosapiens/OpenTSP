import Proofs.ResourceAdmission
import Out.TimingCheck
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions Out.Register Out.diag Out.timing_class
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace FailureAdmissionProofs

def KnownConfig (i : instruction) : Prop := ∃ f v, agt_config i=some (f,v)
def ConfigRecords (r : cycle_recs) : Prop := ∀ i, r.cfg=some i → KnownConfig i

/-- Actual placement retains a valid configuration descriptor even on collision/refusal. -/
theorem place_preserves_config (r : cycle_recs) (i : instruction) (h : ConfigRecords r) :
    ConfigRecords (place r i).1 := by
  cases hc : agt_config i with
  | some config =>
    rcases config with ⟨f,v⟩
    simp only [place,hc]
    split
    · exact h
    · intro j hj
      cases hj
      exact ⟨f,v,hc⟩
  | none =>
    cases hp : ref_path i with
    | some p =>
      simp only [place,hc,hp]
      split <;> simpa only [ConfigRecords] using h
    | none =>
      cases i <;> simp only [place,hc,hp]
      all_goals first
        | simpa only [ConfigRecords] using h
        | split <;> simpa only [ConfigRecords] using h

theorem place_all_preserves_config (r : cycle_recs) (xs : List (Result instruction diag)) (ok : Bool)
    (h : ConfigRecords r) : ConfigRecords (place_all r xs ok).1 := by
  induction xs generalizing r ok with
  | nil => simpa [place_all] using h
  | cons item rest ih =>
    cases item with
    | Err d => simpa [place_all] using ih r false h
    | Ok i =>
      have hp := place_preserves_config r i h
      simpa [place_all] using ih (place r i).1 (ok && (place r i).2) hp

theorem empty_config (n : Nat) : ConfigRecords {no_recs with n:=n} := by
  simp [ConfigRecords,no_recs]

/-- The complete production classifier establishes the descriptor premise for
    arbitrary item lists, including collisions, invalid words and all lengths. -/
theorem actual_classify_config (q : Nat) (items : List (Result instruction diag)) :
    ConfigRecords (classify_items q items) := by
  cases items with
  | nil => simp [classify_items,ConfigRecords,no_recs]
  | cons item rest =>
    cases rest with
    | nil =>
      cases item with
      | Err d => simp [classify_items,ConfigRecords,no_recs]
      | Ok i =>
        simp only [classify_items]
        split
        · exact empty_config 0
        · have hp := place_preserves_config {no_recs with n:=items_len [.Ok i]} i (empty_config 0)
          dsimp only at *
          split
          · exact hp
          · exact empty_config 0
    | cons second rest =>
      simp only [classify_items]
      split
      · exact empty_config 0
      · have hp := place_all_preserves_config {no_recs with n:=items_len (item::second::rest)}
            (item::second::rest) true (empty_config 0)
        simpa only [ConfigRecords] using hp

theorem actual_classifier_configuration_resolves (q : Nat) (items : List (Result instruction diag))
    (i : instruction) (selected : (classify_items q items).cfg=some i) :
    ∃ f v, agt_config i=some (f,v) :=
  actual_classify_config q items i selected

/-- Exact entry lookup admission. The noncatalogue internal_error branch cannot
    run under a catalogue-name premise; no malformed helper name is declared safe. -/
theorem entry_row_admitted (name : String) (t : timing_class)
    (admit : class_of_entry name catalogue_entries=some t) :
    entry_row name=pure (entry_latency name t) := by
  simp [entry_row,admit]

theorem class_lookup_member (name : String) (xs : List (String × timing_class))
    (member : ∃ t, (name,t)∈xs) : ∃ t, class_of_entry name xs=some t := by
  induction xs with
  | nil => simpa using member
  | cons pair rest ih =>
    rcases pair with ⟨n,t⟩
    by_cases eq : n=name
    · subst n
      exact ⟨t,by simp [class_of_entry]⟩
    · obtain ⟨u,hu⟩ := member
      have tail : (name,u)∈rest := by simpa [List.mem_cons,eq,Ne.symm eq] using hu
      obtain ⟨v,hv⟩ := ih ⟨u,tail⟩
      exact ⟨v,by simp [class_of_entry,eq,hv]⟩

/-- Every SRAM timing-loop name is from the actual catalogue. -/
theorem actual_sram_entry_admission (name : String) (member : name∈sram_entries) :
    ∃ t, class_of_entry name catalogue_entries=some t := by
  simp only [sram_entries,List.mem_cons,List.mem_nil_iff,or_false] at member
  rcases member with h|h|h|h|h <;> subst name
  · exact ⟨T_MEM,by rfl⟩
  · exact ⟨T_MEM,by rfl⟩
  · exact ⟨T_MEM,by rfl⟩
  · exact ⟨T_MEM,by rfl⟩
  · exact ⟨T_CU,by rfl⟩

theorem actual_fixed_entry_admission (j : Int) :
    ∃ t, class_of_entry (if j=0 then "Shift" else if j=1 then "Permute" else if j=2 then "ABC" else "ABC8")
      catalogue_entries=some t := by
  by_cases a : j=0
  · simp only [a,if_true]
    exact ⟨T_SXM_CROSS,by rfl⟩
  · by_cases b : j=1
    · simp only [a,b,if_false,if_true]
      exact ⟨T_SXM_CROSS,by rfl⟩
    · by_cases c : j=2
      · simp only [a,b,c,if_false,if_true]
        exact ⟨T_MXM_ABC,by rfl⟩
      · simp only [a,b,c,if_false]
        exact ⟨T_MXM_ABC,by rfl⟩

end FailureAdmissionProofs
