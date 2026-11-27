// The specification of ilog2 on 20-bit inputs: 2^ilog2(n) <= n < 2^(ilog2(n) + 1) for every n >= 1. The negation must be
// unsatisfiable. export_check.sh generates it with sail -smt from the model's own ilog2 and solves it with z3.
$property
function p_ilog2(x : bits(20)) -> bool = {
  let n = unsigned(x);
  n == 0 | (pow2(ilog2(n)) <= n & n < pow2(ilog2(n) + 1))
}
