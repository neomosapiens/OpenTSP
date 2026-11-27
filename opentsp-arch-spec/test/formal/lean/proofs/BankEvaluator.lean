-- Positive evaluations call the actual imported production function.
import Out
open Out Out.Functions
namespace BankCertificate

def domain : Nat := 9^6*16*2*4

def bank (b : Nat) : bank_use :=
  if b=0 then .BankNone () else if b=1 then .BankOf 0#1
  else if b=2 then .BankOf 1#1 else .BankAny ()
def tag (n : Nat) : Option ref_tag :=
  if n=0 then none else some {thread:=BitVec.ofNat 1 ((n-1)/4),bank:=bank ((n-1)%4)}
def digit (code k : Nat) := (code/9^k)%9

def meet (a b : Nat) : Bool := decide (a≠0 ∧ b≠0 ∧ (a=3 ∨ b=3 ∨ a=b))
def conflict (a b : Nat) : Bool :=
  decide (a≠0 ∧ b≠0 ∧ (a-1)/4=(b-1)/4) && meet ((a-1)%4) ((b-1)%4)
def fetchConflict (a b : Nat) : Bool := decide (a≠0) && meet ((a-1)%4) b

def reference (code : Nat) : bank_verdict := Id.run do
  let rt := fun k => digit code k
  let lp := fun p => digit code (4+p)
  let flags := code/9^6
  let go := fun k => flags/2^k%2==1
  let active := flags/16%2==1
  let fb := flags/32%4
  let fl := active && fetchConflict (lp 1) fb
  let fc := fl || (active && ((go 2 && fetchConflict (rt 2) fb) || (go 3 && fetchConflict (rt 3) fb)))
  let blocked := fun (k : Nat) =>
    let p:=k/2
    let other:=2*(1-p)+k%2
    decide (rt k≠0) && go k &&
      ((go other && conflict (rt k) (rt other)) || conflict (rt k) (lp (1-p)) ||
       (decide (p=1) && active && fetchConflict (rt k) fb))
  let loopHit := fun (p : Nat) =>
    conflict (lp p) (lp (1-p)) ||
    (go (2*(1-p)) && conflict (lp p) (rt (2*(1-p)))) ||
    (go (2*(1-p)+1) && conflict (lp p) (rt (2*(1-p)+1)))
  let source := fun (p : Nat) =>
    let lo:=2*(1-p)
    if lp p=0 then (-1:Int)
    else if go (lo+1) && conflict (lp p) (rt (lo+1)) then (lo+1:Int)
    else if go lo && conflict (lp p) (rt lo) then (lo:Int)
    else if conflict (lp p) (lp (1-p)) then (4:Int) else -1
  pure {
    vbank := Vector.ofFn (fun k => blocked k.val)
    vfm_clash := fc
    vfm_loop := fl
    vlblock := Vector.ofFn (fun p => decide (lp p.val ≠ 0) && (loopHit p.val || (decide (p.val=1) && fl)))
    vlsrc := Vector.ofFn (fun p => source p.val) }

def production (code : Nat) : bank_verdict :=
  let flags:=code/9^6
  Out.Functions.bank_rule (Vector.ofFn (fun k => tag (digit code k.val)))
    (Vector.ofFn (fun k => flags/2^k.val%2==1))
    (Vector.ofFn (fun p => tag (digit code (4+p.val)))) (flags/16%2==1) (bank (flags/32%4))

def main (args : List String) : IO UInt32 := do
  if args.length≠3 then IO.eprintln "usage: BankEvaluator pilot|range START END";return 2
  let mode:=args[0]!
  let some start:=args[1]!.toNat? | return 2
  let some stop:=args[2]!.toNat? | return 2
  if start≥stop || stop>domain || (mode !="pilot" && mode !="range") then
    IO.eprintln "invalid range or mode; expected0 <= START < END <= 68024448"
    return 2
  let t0 ← IO.monoMsNow
  let mut checked:=0
  for i in [start:stop] do
    let code:=if mode="pilot" then (i*65537)%domain else i
    let got:=production code
    let expected:=reference code
    if got !=expected then
      IO.println s!"MISMATCH code={code} checked={checked} actual={repr got} expected={repr expected}"
      return 1
    checked:=checked+1
    if checked%4096=0 then
      let now ← IO.monoMsNow
      IO.println s!"PROGRESS checked={checked} milliseconds={now-t0}"
  let t1 ← IO.monoMsNow
  IO.println s!"COMPLETE start={start} end={stop} checked={checked} milliseconds={t1-t0} mode={mode}"
  return 0
end BankCertificate

def main := BankCertificate.main
