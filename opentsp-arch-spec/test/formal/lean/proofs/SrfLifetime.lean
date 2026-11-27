-- Unbounded flow/lifetime and deferred-commit ring arithmetic.
import Proofs.SrfNearest
open Sail Out Out.Functions
set_option autoImplicit false
namespace SrfLifetimeProofs

theorem flow_diagonal (writer reader commit readAt : Int) :
    writer-commit=reader-readAt ↔ readAt-commit=reader-writer := by omega

theorem flow_after_commit (writer reader commit readAt : Int)
    (hdiag : writer-commit=reader-readAt) (hup : writer < reader) : commit < readAt := by omega

theorem chain_previous_commit (gap commit readAt : Int) :
    gap-commit=gap-readAt+1 ↔ commit=readAt-1 := by omega

theorem committed_live_position (gap commit now : Int) (hg : 0 ≤ gap) (hp : commit ≤ now)
    (hl : gap-commit+now ≤ 44) : -now ≤ gap-commit ∧ gap-commit ≤ 44-now := by omega

theorem live_diagonals_distinct (a b now : Int)
    (ha : -now ≤ a ∧ a ≤ 44-now) (hb : -now ≤ b ∧ b ≤ 44-now)
    (hslot : a%64=b%64) : a=b := by
  have h1:=Int.ediv_mul_add_emod a 64
  have h2:=Int.ediv_mul_add_emod b 64
  omega

theorem committed_live_no_alias (ga gb ca cb now : Int)
    (hga : 0 ≤ ga) (hgb : 0 ≤ gb) (hca : ca ≤ now) (hcb : cb ≤ now)
    (hla : ga-ca+now ≤ 44) (hlb : gb-cb+now ≤ 44)
    (hslot : (ga-ca)%64=(gb-cb)%64) : ga-ca=gb-cb := by
  exact live_diagonals_distinct _ _ now (committed_live_position ga ca now hga hca hla)
    (committed_live_position gb cb now hgb hcb hlb) hslot

theorem overwritten_diagonal_expired (gap older newer : Int) (hg : 0 ≤ gap) (hc : older < newer)
    (hslot : (gap-older)%64=(gap-newer)%64) : 44 < gap-older+newer := by
  have h1:=Int.ediv_mul_add_emod (gap-older) 64
  have h2:=Int.ediv_mul_add_emod (gap-newer) 64
  omega

theorem srf_slot_range (d : Int) : 0 ≤ slot_of d ∧ slot_of d < 64 := by
  simp only [slot_of]
  constructor
  · omega
  · change (d%64).toNat<64
    exact (Int.toNat_lt (Int.emod_nonneg d (by decide))).mpr (Int.emod_lt_of_pos d (by decide))

theorem deferred_commit_horizon (dispatch element skew func : Int)
    (hskew : 0 ≤ skew) (hlat : skew < func) (hfunc : func ≤ 63) :
    0 < (dispatch+element+func)-(dispatch+element+skew) ∧
    (dispatch+element+func)-(dispatch+element+skew) ≤ 63 := by omega

theorem pending_no_128_alias (earlier later horizon : Int)
    (horizon0 : 0 < horizon) (horizon128 : horizon < 128)
    (he : earlier < later) (hl : later-earlier ≤ horizon) : earlier%128≠later%128 := by
  intro heq
  have h1:=Int.ediv_mul_add_emod earlier 128
  have h2:=Int.ediv_mul_add_emod later 128
  omega

theorem future_commit_not_current_bucket (now commit : Int)
    (hf : 0 < commit-now) (hh : commit-now ≤ 63) : now%128≠commit%128 :=
  pending_no_128_alias now commit 63 (by decide) (by decide) (by omega) hh

theorem future_commit_late_bucket (now firstAt later : Int)
    (hat : now < firstAt ∧ firstAt ≤ now+63) (hlater : firstAt < later ∧ later ≤ now+63)
    (hslot : firstAt%128=later%128) : False := by
  have h:=pending_no_128_alias firstAt later 63 (by decide) (by decide) hlater.1 (by omega)
  exact h hslot
end SrfLifetimeProofs
