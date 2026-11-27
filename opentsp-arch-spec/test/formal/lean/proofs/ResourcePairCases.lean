-- Restrict tuple decomposition to actual Prod locals; never destruct arbitrary vectors.
import Lean
open Lean Meta Elab Tactic
elab "cases_resource_pair" : tactic => do
  let goal ← getMainGoal
  goal.withContext do
    for decl in (← getLCtx) do
      if decl.isImplementationDetail || decl.isLet then continue
      let type ← whnf decl.type
      if type.isAppOfArity ``Prod 2 then
        let next ← goal.cases decl.fvarId
        replaceMainGoal (next.toList.map (·.mvarId))
        return
    throwError "no product local"

elab "reduce_resource_beta" : tactic => do
  let before ← (← getMainGoal).getType
  evalTactic (← `(tactic| dsimp only))
  let goals ← getGoals
  if goals.isEmpty then return
  let after ← (← getMainGoal).getType
  if before == after then throwError "no beta/let reduction"
