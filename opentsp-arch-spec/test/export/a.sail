default Order dec
$include <prelude.sail>
val f : list(nat) -> nat
function f(fl) = {
  var faulty : list(nat) = fl;
  var n : nat = 0;
  while (match faulty { [||] => false, _ => true }) do { match faulty { [||] => (), x :: rest => { n = n + x; faulty = rest } } };
  n
}
