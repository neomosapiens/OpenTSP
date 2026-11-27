-- Coverage arithmetic for the exact native bank certificate decoder.
import Proofs.BankEvaluator
set_option autoImplicit false
namespace BankDomainProofs
open BankCertificate Out

def encodeBank : bank_use → Nat
 | .BankNone _ => 0
 | .BankOf t => 1+t.toNat
 | .BankAny _ => 3

theorem thread_cases (t : BitVec 1) : t=0#1 ∨ t=1#1 := by
 have h:=t.isLt
 have h0 : t.toNat=0 ∨ t.toNat=1 := by omega
 rcases h0 with h0|h1
 · exact Or.inl (BitVec.eq_of_toNat_eq h0)
 · exact Or.inr (BitVec.eq_of_toNat_eq h1)

theorem bank_roundtrip (b : bank_use) : bank (encodeBank b)=b := by
 cases b with
 | BankNone u => cases u; rfl
 | BankAny u => cases u; rfl
 | BankOf t => rcases thread_cases t with h|h <;> subst t <;> rfl

theorem bank_code_bound (b : bank_use) : encodeBank b<4 := by
 cases b with
 | BankNone u => cases u; decide
 | BankAny u => cases u; decide
 | BankOf t => have h:=t.isLt; simp only [encodeBank]; omega

def encodeTag (t : Option ref_tag) : Nat := match t with
 | none => 0
 | some t => 1+4*t.thread.toNat+encodeBank t.bank

theorem tag_roundtrip (t : Option ref_tag) : tag (encodeTag t)=t := by
 cases t with
 | none => rfl
 | some t =>
   cases t with
   | mk thread b =>
     rcases thread_cases thread with h|h <;> subst thread <;> cases b with
     | BankNone u => cases u; rfl
     | BankAny u => cases u; rfl
     | BankOf bt => rcases thread_cases bt with h|h <;> subst bt <;> rfl

theorem tag_code_bound (t : Option ref_tag) : encodeTag t<9 := by
 cases t with
 | none => decide
 | some t => have ht:=t.thread.isLt; have hb:=bank_code_bound t.bank; simp only [encodeTag]; omega

def pack (a b c d e f flags : Nat) := a+9*b+81*c+729*d+6561*e+59049*f+531441*flags

theorem radix_decode (a b c d e f flags : Nat)
 (ha:a<9) (hb:b<9) (hc:c<9) (hd:d<9) (he:e<9) (hf:f<9) (hg:flags<128) :
 let n:=pack a b c d e f flags
 n<domain ∧ digit n 0=a ∧ digit n 1=b ∧ digit n 2=c ∧ digit n 3=d ∧ digit n 4=e ∧ digit n 5=f ∧ n/531441=flags := by
 dsimp [pack,domain,digit]
 omega

theorem radix_encode (n : Nat) (hn:n<domain) :
 pack (digit n 0) (digit n 1) (digit n 2) (digit n 3) (digit n 4) (digit n 5) (n/531441)=n := by
 dsimp [pack,digit]
 omega

theorem flags_decode (a b c d active fb : Nat)
 (ha:a<2) (hb:b<2) (hc:c<2) (hd:d<2) (he:active<2) (hf:fb<4) :
 let flags:=a+2*b+4*c+8*d+16*active+32*fb
 flags<128 ∧ flags%2=a ∧ flags/2%2=b ∧ flags/4%2=c ∧ flags/8%2=d ∧ flags/16%2=active ∧ flags/32%4=fb := by
 dsimp
 omega

theorem full_count : domain=68024448 := by decide

def boolCode (b : Bool) : Nat := if b then 1 else 0
theorem bool_code_bound (b : Bool) : boolCode b<2 := by cases b <;> decide
theorem bool_roundtrip (b : Bool) : (boolCode b==1)=b := by cases b <;> rfl

def encodeInput (rt : Vector (Option ref_tag) 4) (go : Vector Bool 4)
 (lp : Vector (Option ref_tag) 2) (active : Bool) (fb : bank_use) : Nat :=
 pack (encodeTag rt[0]) (encodeTag rt[1]) (encodeTag rt[2]) (encodeTag rt[3])
 (encodeTag lp[0]) (encodeTag lp[1])
 (boolCode go[0]+2*boolCode go[1]+4*boolCode go[2]+8*boolCode go[3]+16*boolCode active+32*encodeBank fb)

theorem entire_input_coverage (rt : Vector (Option ref_tag) 4) (go : Vector Bool 4)
 (lp : Vector (Option ref_tag) 2) (active : Bool) (fb : bank_use) :
 let code:=encodeInput rt go lp active fb
 code<domain ∧
 (Vector.ofFn (fun k : Fin 4 => tag (digit code k.val)))=rt ∧
 (Vector.ofFn (fun k : Fin 4 => code/9^6/2^k.val%2==1))=go ∧
 (Vector.ofFn (fun k : Fin 2 => tag (digit code (4+k.val))))=lp ∧
 (code/9^6/16%2==1)=active ∧ bank (code/9^6/32%4)=fb := by
 let flags:=boolCode go[0]+2*boolCode go[1]+4*boolCode go[2]+8*boolCode go[3]+16*boolCode active+32*encodeBank fb
 have hf:=flags_decode (boolCode go[0]) (boolCode go[1]) (boolCode go[2]) (boolCode go[3]) (boolCode active) (encodeBank fb)
   (bool_code_bound _) (bool_code_bound _) (bool_code_bound _) (bool_code_bound _) (bool_code_bound _) (bank_code_bound _)
 change flags<128 ∧ flags%2=boolCode go[0] ∧ flags/2%2=boolCode go[1] ∧ flags/4%2=boolCode go[2] ∧ flags/8%2=boolCode go[3] ∧ flags/16%2=boolCode active ∧ flags/32%4=encodeBank fb at hf
 have hd:=radix_decode (encodeTag rt[0]) (encodeTag rt[1]) (encodeTag rt[2]) (encodeTag rt[3]) (encodeTag lp[0]) (encodeTag lp[1]) flags
   (tag_code_bound _) (tag_code_bound _) (tag_code_bound _) (tag_code_bound _) (tag_code_bound _) (tag_code_bound _) hf.1
 have hcode : encodeInput rt go lp active fb=pack (encodeTag rt[0]) (encodeTag rt[1]) (encodeTag rt[2]) (encodeTag rt[3]) (encodeTag lp[0]) (encodeTag lp[1]) flags := rfl
 simp only [←hcode] at hd
 dsimp only
 refine ⟨hd.1,?_,?_,?_,?_,?_⟩
 · apply Vector.ext
   intro i hi
   have hcase : i=0 ∨ i=1 ∨ i=2 ∨ i=3 := by omega
   rcases hcase with h|h|h|h <;> subst i <;>
    simp [Vector.getElem_ofFn,hd.2.1,hd.2.2.1,hd.2.2.2.1,hd.2.2.2.2.1,tag_roundtrip]
 · apply Vector.ext
   intro i hi
   have hcase : i=0 ∨ i=1 ∨ i=2 ∨ i=3 := by omega
   rcases hcase with h|h|h|h <;> subst i <;>
    simp [Vector.getElem_ofFn,show 9^6=531441 from rfl,hd.2.2.2.2.2.2.2,hf.2.1,hf.2.2.1,hf.2.2.2.1,hf.2.2.2.2.1,bool_roundtrip]
 · apply Vector.ext
   intro i hi
   have hcase : i=0 ∨ i=1 := by omega
   rcases hcase with h|h <;> subst i <;>
    simp [Vector.getElem_ofFn,hd.2.2.2.2.2.1,hd.2.2.2.2.2.2.1,tag_roundtrip]
 · simpa only [show 9^6=531441 from rfl,hd.2.2.2.2.2.2.2,hf.2.2.2.2.2.1,bool_roundtrip]
 · simpa only [show 9^6=531441 from rfl,hd.2.2.2.2.2.2.2,hf.2.2.2.2.2.2,bank_roundtrip]
theorem production_covers_typed_inputs (rt : Vector (Option ref_tag) 4) (go : Vector Bool 4)
 (lp : Vector (Option ref_tag) 2) (active : Bool) (fb : bank_use) :
 production (encodeInput rt go lp active fb)=Out.Functions.bank_rule rt go lp active fb := by
 have h:=entire_input_coverage rt go lp active fb
 unfold production
 dsimp only
 rw [h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2]
end BankDomainProofs
