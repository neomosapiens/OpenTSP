default Order dec
$include <prelude.sail>
val nonempty_list : forall ('a : Type). list('a) -> bool
function nonempty_list(xs) = match xs { [||] => false, _ => true }
val f : list(nat) -> nat
function f(fl) = {
  var faulty : list(nat) = fl;
  var n : nat = 0;
  while nonempty_list(faulty) do { match faulty { [||] => (), x :: rest => { n = n + x; faulty = rest } } };
  n
}
