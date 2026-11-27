-- Native partial-correctness invariants: success and error poststates both count.
import Proofs.ResourceFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace ResourceInvariantProofs
open ResourceFrameProofs

def Inv {α : Type} (predicate : St → Prop) (action : SailM α) : Prop :=
  ∀ s, predicate s → match action.run s with
    | .ok _ after => predicate after
    | .error _ after => predicate after

theorem pure_inv {α : Type} (predicate : St → Prop) (x : α) : Inv predicate (pure x) := by
  intro s hs
  exact hs

theorem bind_inv {α β : Type} (predicate : St → Prop) (a : SailM α) (f : α → SailM β)
    (ha : Inv predicate a) (hf : ∀ x, Inv predicate (f x)) : Inv predicate (a >>= f) := by
  intro s hs
  have h1 := ha s hs
  cases h : a.run s with
  | ok x after =>
    simp only [h] at h1
    simpa [EStateM.run,bind,EStateM.bind,show a s = .ok x after from h] using hf x after h1
  | error err after =>
    simpa [h,EStateM.run,bind,EStateM.bind,show a s = .error err after from h] using h1

theorem keeps_inv {α β : Type} (view : St → β) (base : β) (a : SailM α) (h : Keeps view a) :
    Inv (fun s => view s=base) a := by
  intro s hs
  have h1 := h s
  cases hx : a.run s <;> simp only [hx] at h1 <;> simpa [hx] using h1.trans hs

theorem read_inv (predicate : St → Prop) (r : Register) : Inv predicate (Out.readReg r) := by
  intro s hs
  have hf := read_keeps (fun s : St => s) r s
  cases hr : (Out.readReg r).run s <;> simp only [hr] at hf <;> simpa [hr,hf] using hs

theorem forIn_inv {α : Type} (predicate : St → Prop) (range : IntRange) (value : α)
    (f : (i : Int) → i ∈ range → α → SailM (ForInStep α))
    (hf : ∀ i hi x, Inv predicate (f i hi x)) : Inv predicate (IntRange.forIn' range value f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => Inv predicate (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply bind_inv _ _ _ (hf i hi value)
    intro step
    cases step with
    | done value => exact pure_inv predicate value
    | yield value => exact ih value
  · intro value i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact pure_inv predicate value

theorem pair_keeps {α β γ : Type} (a : St → β) (b : St → γ) (action : SailM α)
    (ha : Keeps a action) (hb : Keeps b action) : Keeps (fun s => (a s,b s)) action := by
  intro s
  have h1 := ha s
  have h2 := hb s
  cases hr : action.run s <;> simp only [hr] at h1 h2 <;> simp only [hr] <;>
    exact Prod.ext h1 h2
theorem read_bind_bounded {α : Type} (predicate : St → Prop) (r : Register)
    (valueBound : RegisterType r → Prop) (f : RegisterType r → SailM α)
    (bound : ∀ s,predicate s → ∀ v,s.regs.get? r=some v → valueBound v)
    (hf : ∀ v,valueBound v → Inv predicate (f v)) :
    Inv predicate (Out.readReg r >>= f) := by
  intro s hs
  cases hr : s.regs.get? r with
  | none =>
    have hm:=read_missing_run s r hr
    change Out.readReg r s=.error Error.Unreachable s at hm
    simpa only [EStateM.run,bind,EStateM.bind,hm] using hs
  | some v =>
    have hv:=hf v (bound s hs v hr) s hs
    have hm : Out.readReg r s=.ok v s := by
      simp [Out.readReg,PreSail.readReg,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,hr,pure,EStateM.pure]
    simpa only [EStateM.run,bind,EStateM.bind,hm] using hv
theorem keeps_pred {α β : Type} (view : St → β) (predicate : β → Prop) (action : SailM α)
    (h : Keeps view action) : Inv (fun s => predicate (view s)) action := by
  intro s hs
  have hh:=h s
  cases hr : action.run s <;> simp only [hr] at hh <;> simpa only [hr,hh] using hs
end ResourceInvariantProofs
