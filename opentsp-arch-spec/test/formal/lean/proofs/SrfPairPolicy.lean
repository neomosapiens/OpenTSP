import Proofs.SrfCommitPolicy
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace SrfPairPolicyProofs
/-- Architectural queue classes: west/east SEL and the two TSP queues of each hemisphere. -/
def ORClass (q : Nat) : Nat × Nat := match q with
 | 125=>(1,0) |139=>(1,1) |120|121=>(2,0) |134|135=>(2,1) |_ =>(0,0)

def ORPair (a b : Nat) : Bool :=
 let ka:=ORClass a
 let kb:=ORClass b
 ka.1!=0 && kb.1!=0 && ka.1!=kb.1 && ka.2==kb.2

theorem actual_queue_class (q : Nat) (hq : q≤144) : sxm_or_kind q=ORClass q := by
 have finite : ∀ k : Fin 145,sxm_or_kind k.val=ORClass k.val := by decide +kernel
 exact finite ⟨q,by omega⟩

theorem actual_or_pair (a b : commit) (ha : a.q≤144) (hb : b.q≤144) :
 or_pair a b=ORPair a.q b.q := by
 unfold or_pair ORPair
 rw [actual_queue_class a.q ha,actual_queue_class b.q hb]
 simp only [Bool.and_assoc]

#print axioms actual_queue_class
#print axioms actual_or_pair
end SrfPairPolicyProofs
