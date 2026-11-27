-- Native indexed-loop bridge reused from the accepted numerical whole-fold proof.
import Proofs.ResourceGeometry
open Sail Out Out.Functions
set_option autoImplicit false
namespace ResourceScanProofs
def indexedFold {α : Type} (f : Int → α → α) (i : Int) : Nat → α → α
  | 0,state => state
  | n+1,state => indexedFold f (i+1) n (f i state)

def indexedSum (term : Int → Int) (i : Int) : Nat → Int
  | 0 => 0
  | n+1 => term i+ indexedSum term (i+1) n

/-- Actual inclusive indexed Sail loop, with no omitted endpoint or surrogate
    index. The independent fold is ordinary finite mathematical iteration. -/
theorem actual_indexed_loop_fold {α : Type} (range : IntRange) (f : Int → α → α)
    (state : α) (i : Int) (hs : (i-range.start)%range.step=0)
    (hstep : range.step=1) (hstart : range.start ≤ i) :
    IntRange.forIn'.loop (m := Id) range (fun k _ s => pure (.yield (f k s))) state i hs=
      indexedFold f i (range.stop - i+1).toNat state := by
  induction state,i,hs using IntRange.forIn'.loop.induct range with
  | case1 state i hs hi ih =>
    have hmem := hi
    simp [Membership.mem,hstep] at hmem
    rw [IntRange.forIn'.loop.eq_1]
    simp only [hi,dite_true,pure_bind]
    have hn : (range.stop - i+1).toNat=(range.stop - (i+range.step)+1).toNat+1 := by omega
    rw [hn]
    simp only [indexedFold]
    have hh := ih (f i state) (by omega)
    simpa only [hstep] using hh
  | case2 state i hs hi =>
    have hmem := hi
    simp [Membership.mem,hstep] at hmem
    have hn : (range.stop - i+1).toNat=0 := by omega
    rw [IntRange.forIn'.loop.eq_1]
    simp [hi,hn,indexedFold]
    rfl

theorem actual_indexed_range_fold {α : Type} (range : IntRange) (f : Int → α → α)
    (state : α) (hstep : range.step=1) :
    IntRange.forIn' (m := Id) range state (fun k _ s => pure (.yield (f k s)))=
      indexedFold f range.start (range.stop-range.start+1).toNat state := by
  unfold IntRange.forIn'
  exact actual_indexed_loop_fold range f state range.start (by simp) hstep (by omega)

end ResourceScanProofs
