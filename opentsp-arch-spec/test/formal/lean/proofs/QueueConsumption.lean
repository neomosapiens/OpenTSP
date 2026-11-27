import Proofs.ControlConsumption
import Proofs.ResourceExecuteFrames
import Proofs.GrantConsumption
import Proofs.ResourceComplete
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
open Proofs.TagCoverage Proofs.TaggedCheck Proofs.ParcelFrame
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace Proofs.QueueConsumption
abbrev IQMem := ResourceExecuteFrameProofs.IQMem

theorem present_projection (s t : ResourceFrameProofs.St) (same : IQMem t=IQMem s)
    (count field slot value : Nat) (hf : field<2) :
    Present t count field slot value ↔ Present s count field slot value := by
  have h := congrFun same (cellKey count field slot value)
  simp only [IQMem,ResourceExecuteFrameProofs.IQMem,key_field count field slot value hf,
    hf,ite_true] at h
  unfold Present
  rw [h]

theorem read_projection (s t : ResourceFrameProofs.St) (same : IQMem t=IQMem s)
    (count field slot : Nat) (hf : field<2) :
    readValue t count field slot=readValue s count field slot :=
  readValue_ext t s count field slot (fun v => present_projection s t same count field slot v hf)

theorem represents_projection (cap : Nat) (view : CView) (s t : ResourceFrameProofs.St)
    (same : IQMem t=IQMem s) (rep : Represents cap view s) : Represents cap view t := by
  constructor
  · intro f slot hf hs a b ha hb
    exact rep.1 f slot hf hs a b
      ((present_projection s t same _ _ _ _ hf).mp ha)
      ((present_projection s t same _ _ _ _ hf).mp hb)
  · intro q slot hq hs
    have init : Initialized cap t q slot ↔ Initialized cap s q slot := by
      unfold Initialized
      constructor <;> rintro ⟨v,hv⟩
      · exact ⟨v,(present_projection s t same _ _ _ _ (by decide)).mp hv⟩
      · exact ⟨v,(present_projection s t same _ _ _ _ (by decide)).mpr hv⟩
    refine ⟨(rep.2 q slot hq hs).1.trans init.symm,?_,?_⟩
    · intro h
      rw [read_projection s t same _ _ _ (by decide)]
      exact (rep.2 q slot hq hs).2.1 h
    · unfold tags
      rw [read_projection s t same _ _ _ (by decide)]
      exact (rep.2 q slot hq hs).2.2

theorem keep_projection (A : Nat → Prop) (H L : Vector Nat 145) (s t : ResourceFrameProofs.St)
    (keep : Keep A H L s)
    (hh : t.regs.get? Register.iq_head=s.regs.get? Register.iq_head)
    (hl : t.regs.get? Register.iq_len=s.regs.get? Register.iq_len)
    (memory : IQMem t=IQMem s) (complete : MachineInitProofs.Complete t) : Keep A H L t := by
  obtain ⟨⟨⟨view,rep,⟨H0,L0,hH0,hL0,bounds⟩⟩,hH,hL⟩,⟨Ht,Lt,ht,lt,tags⟩,allocated,_⟩ := keep
  refine ⟨⟨⟨view,represents_projection iq_cap view s t memory rep,?_
    ⟩,hh.trans hH,hl.trans hL⟩,?_,?_,complete⟩
  · refine ⟨H0,L0,hh.trans hH0,hl.trans hL0,?_⟩
    intro q hq
    obtain ⟨head,len,init⟩ := bounds q hq
    refine ⟨head,len,?_⟩
    intro i hi
    obtain ⟨v,hv⟩ := init i hi
    exact ⟨v,(present_projection s t memory _ _ _ _ (by decide)).mpr hv⟩
  · refine ⟨Ht,Lt,hh.trans ht,hl.trans lt,?_⟩
    intro q hq
    obtain ⟨head,len,live⟩ := tags q hq
    refine ⟨head,len,?_⟩
    intro i hi
    obtain ⟨v,hv⟩ := live i hi
    exact ⟨v,(present_projection s t memory _ _ _ _ (by decide)).mpr hv⟩
  · intro q hq
    rw [allocated q hq]
    constructor <;> rintro ⟨slot,hs,v,hv⟩
    · exact ⟨slot,hs,v,(present_projection s t memory _ _ _ _ (by decide)).mpr hv⟩
    · exact ⟨slot,hs,v,(present_projection s t memory _ _ _ _ (by decide)).mp hv⟩

theorem frame_complete_safe {α : Type} (A : Nat → Prop) (H L : Vector Nat 145) (action : SailM α)
    (frame : ResourceFrameProofs.Keeps ResourceExecuteFrameProofs.RView action)
    (full : ResourceInvariantProofs.Inv MachineInitProofs.Complete action) : Safe A (H:=H) (L:=L) action := by
  intro s keep
  have hh := ResourceExecuteFrameProofs.iq_head_projection action frame s
  have hl := ResourceExecuteFrameProofs.iq_len_projection action frame s
  have hm := ResourceExecuteFrameProofs.iq_memory_projection action frame s
  have hc := full s keep.2.2.2
  cases run : action.run s with
  | ok value after =>
    have hh' : after.regs.get? Register.iq_head=s.regs.get? Register.iq_head := by simpa [ResourceFrameProofs.RegKeeps,ResourceFrameProofs.Keeps,run] using hh
    have hl' : after.regs.get? Register.iq_len=s.regs.get? Register.iq_len := by simpa [ResourceFrameProofs.RegKeeps,ResourceFrameProofs.Keeps,run] using hl
    have hm' : IQMem after=IQMem s := by simpa [ResourceFrameProofs.Keeps,run] using hm
    have hc' : MachineInitProofs.Complete after := by simpa [run] using hc
    simpa [run] using keep_projection A H L s after keep hh' hl' hm' hc'
  | error error after =>
    have hh' : after.regs.get? Register.iq_head=s.regs.get? Register.iq_head := by simpa [ResourceFrameProofs.RegKeeps,ResourceFrameProofs.Keeps,run] using hh
    have hl' : after.regs.get? Register.iq_len=s.regs.get? Register.iq_len := by simpa [ResourceFrameProofs.RegKeeps,ResourceFrameProofs.Keeps,run] using hl
    have hm' : IQMem after=IQMem s := by simpa [ResourceFrameProofs.Keeps,run] using hm
    have hc' : MachineInitProofs.Complete after := by simpa [run] using hc
    simpa [run] using keep_projection A H L s after keep hh' hl' hm' hc'

open vxm_unop
open vxm_lane_op
open vxm_conv
open vxm_binop
open vsrc
open unit_kind
open timing_class
open sxm_mapop
open streamref
open srf_lookup
open sram_addr
open shift_dir
open scrub_action
open row
open reduce_op
open option
open mxm_wtype
open mxm_part
open mxm_acctype
open mem_dir
open instruction
open fclass
open exception
open eltype
open ecc_class
open diag
open cmp_mode
open c2c_ptype
open c2c_op
open bank_use
open agt_state
open agt_field
open acc_mode
open Register
open ExecutionResult


def scanHead (q c : Nat) : SailM (Bool × parcel) := do
  if (((GetElem?.getElem! (← Out.readReg iq_len) q) >b 0) : Bool) then
    let p ← parcel_scan q
    pure (p.complete && (c ≥b (← parcel_earliest q p)),p)
  else pure (false,{ complete := false, units := 0, nops := 0, n := 0, items := [] })

def prepareK (q c : Nat) (p : parcel) (due : Bool) (finish : Bool → cycle_recs → Bool → SailM Unit) : SailM Unit := do
  let recs : cycle_recs :=
    if (due : Bool)
    then (classify_items q p.items)
    else no_recs
  let parked ← do (queue_parked q c)
  match (GetElem?.getElem! (← Out.readReg queue_poll) q) with
  | .some p =>
    (do
      if ((← do if ((poll_epoch p.since c c2c_hac_period parked) : Bool)
                then deskewrecv_ready c p.link else pure false) : Bool)
      then
        (do
          Out.writeReg queue_poll (vectorUpdate (← Out.readReg queue_poll) q none)
          let rcv := (C2C_RECEIVE (p.chip, p.link, p.s))
          Out.writeReg queue_busy_until (vectorUpdate (← Out.readReg queue_busy_until) q
            (c +i (← (occupancy rcv c))))
          (run_element q rcv c 0 0 0))
      else (pure ()))
  | none => (pure ())
  match (GetElem?.getElem! (← Out.readReg queue_repeat) q) with
  | .some r =>
    (do
      if (((r.next ≤b c) && (Out.Functions.not parked)) : Bool)
      then
        (do
          (observe_site "repeat" c q 0 0 (GetElem?.getElem! (← Out.readReg queue_last_at) q))
          (repeat_issue q r.instr c)
          Out.writeReg queue_last_at (vectorUpdate (← Out.readReg queue_last_at) q c)
          Out.writeReg queue_busy_until (vectorUpdate (← Out.readReg queue_busy_until) q (c +i 1))
          Out.writeReg queue_repeat (vectorUpdate (← Out.readReg queue_repeat) q (repeat_after r c)))
      else (pure ()))
  | none => (pure ())
  let go ← (( do (pure (due && (Out.Functions.not (← (parcel_held q c recs p.items))))) ) : SailM Bool )
  let ended : Bool := false
  let ended ← (( do
    let loop_path_lower := 0
    let loop_path_upper := 1
    let mut loop_vars := ended
    for path in [loop_path_lower:loop_path_upper:1]i do
      let ended := loop_vars
      loop_vars ← do
        if ((config_ends (← (agt_get q path)).state go (config_for recs)) : Bool)
        then
          (do
            (agt_finish_config q path)
            (pure true))
        else (pure ended)
    (pure loop_vars) ) : SailM Bool )
  let go ← (( do
    if ((ended && (due && (Out.Functions.not go))) : Bool)
    then
      (do
        (pure (Out.Functions.not (← (parcel_held q c recs p.items)))))
    else (pure go) ) : SailM Bool )
  if ((due && (Out.Functions.not go)) : Bool)
  then
    (observe_site
      (wait_reason (← (host_parked q c)) parked (← (polling q))
        (bne (GetElem?.getElem! (← Out.readReg queue_repeat) q) none)) c q
      (GetElem?.getElem! (← Out.readReg iq_head) q) p.units (← (parcel_earliest q p)))
  else (pure ())
  finish go recs parked

def consume (q c : Nat) (p : parcel) (go : Bool) (recs : cycle_recs) (parked : Bool) : SailM Unit := do
  let r2 : cycle_recs := no_recs
  let r2 ← (( do
    if (go : Bool)
    then
      (do
        let r2 : cycle_recs := recs
        (observe_site "dispatch" c q (GetElem?.getElem! (← Out.readReg iq_head) q) p.units
          (← (parcel_earliest q p)))
        (Proofs.TextCheckProduct.checkedPop q c p)
        Out.writeReg iq_last (vectorUpdate (← Out.readReg iq_last) q c)
        Out.writeReg iq_active (vectorUpdate (← Out.readReg iq_active) q c)
        if ((GetElem?.getElem! (← Out.readReg iq_dry) q) : Bool)
        then
          (do
            Out.writeReg iq_dry (vectorUpdate (← Out.readReg iq_dry) q false)
            (pure (lint q c LINT_IQ_UNDERFLOW)))
        else (pure ())
        (pure r2))
    else (pure r2) ) : SailM cycle_recs )
  (dispatch q c r2 parked)


theorem actual_tick_queue_factor (q c : Nat) : tick_queue q c =
    (do let (due,p) ← scanHead q c
        prepareK q c p due (consume q c p)) := by
  rw [Proofs.TextCheckProduct.tick_queue_checked_pop_bridge]
  simp only [Proofs.TextCheckProduct.tickQueueUsingCheckedPop,scanHead,prepareK,consume,
    Proofs.OrderedImage.bind_if,bind_assoc,pure_bind]
  all_goals rfl


open ResourceFrameProofs ResourceExecuteFrameProofs
section MetadataFrames
attribute [local irreducible] ResourceFrameProofs.Keeps
attribute [local irreducible] GetElem?.getElem! Inhabited.default Bind.bind Pure.pure EStateM.bind EStateM.pure PreSail.readReg PreSail.writeReg IntRange.forIn'
attribute [local irreducible] Out.Functions.__id Out.Functions.abc_after Out.Functions.abc_at Out.Functions.abc_fp8 Out.Functions.abc_others Out.Functions.abc_others_fp8 Out.Functions.abc_pair_pick Out.Functions.abc_prune Out.Functions.abc_types_ok Out.Functions.acc_mode_name_backwards_matches Out.Functions.acc_mode_name_forwards Out.Functions.acc_mode_name_forwards_matches Out.Functions.acc_mode_of_num Out.Functions.acctype_declared Out.Functions.acctype_name_backwards_matches Out.Functions.acctype_name_forwards Out.Functions.acctype_name_forwards_matches Out.Functions.addr_bank Out.Functions.agt_advance Out.Functions.agt_advance_test Out.Functions.agt_after_reference Out.Functions.agt_config Out.Functions.agt_configure_pure Out.Functions.agt_due_pure Out.Functions.agt_field_of_num Out.Functions.agt_field_pos Out.Functions.agt_finish_config_pure Out.Functions.agt_nlevels Out.Functions.agt_opened Out.Functions.agt_refresh_pure Out.Functions.agt_state_of_num Out.Functions.aligned Out.Functions.all_ones Out.Functions.append_element Out.Functions.assembly_forwards_matches Out.Functions.bank_rule Out.Functions.banks_meet Out.Functions.bf16_dot Out.Functions.bfp8_exact Out.Functions.binop_entry_backwards_matches Out.Functions.binop_entry_forwards Out.Functions.binop_entry_forwards_matches Out.Functions.binop_legal Out.Functions.bit_index Out.Functions.bit_of Out.Functions.bits_of_s Out.Functions.bits_t Out.Functions.blk_exp Out.Functions.bool_bits Out.Functions.c2c_op_of_num Out.Functions.c2c_packet_flits Out.Functions.c2c_poison_int Out.Functions.c2c_ptype_code Out.Functions.c2c_ptype_of_num Out.Functions.c2c_ptype_queued Out.Functions.c2c_rx Out.Functions.c2c_tx Out.Functions.canon_nan Out.Functions.cfg_path_pick Out.Functions.chain_suffix_backwards_matches Out.Functions.chain_suffix_forwards Out.Functions.chain_suffix_forwards_matches Out.Functions.chk_elem Out.Functions.chk_wrow Out.Functions.clash_addr Out.Functions.clash_stream Out.Functions.classify_items Out.Functions.clears_repeat Out.Functions.cmp_legal Out.Functions.cmp_mnemonic_backwards_matches Out.Functions.cmp_mnemonic_forwards Out.Functions.cmp_mnemonic_forwards_matches Out.Functions.commit_scan Out.Functions.commit_wins Out.Functions.config_ends Out.Functions.config_for Out.Functions.conv_dst Out.Functions.conv_entry_backwards_matches Out.Functions.conv_entry_forwards Out.Functions.conv_entry_forwards_matches Out.Functions.conv_mnemonic_backwards_matches Out.Functions.conv_mnemonic_forwards Out.Functions.conv_mnemonic_forwards_matches Out.Functions.conv_src Out.Functions.conv_src_type Out.Functions.conv_types Out.Functions.csr_value Out.Functions.db_idx Out.Functions.dec_bits_10_backwards Out.Functions.dec_bits_10_backwards_matches Out.Functions.dec_bits_1_backwards Out.Functions.dec_bits_1_backwards_matches Out.Functions.dec_bits_2_backwards Out.Functions.dec_bits_2_backwards_matches Out.Functions.dec_bits_32_backwards Out.Functions.dec_bits_32_backwards_matches Out.Functions.dec_bits_3_backwards Out.Functions.dec_bits_3_backwards_matches Out.Functions.dec_bits_4_backwards Out.Functions.dec_bits_4_backwards_matches Out.Functions.dec_bits_5_backwards Out.Functions.dec_bits_5_backwards_matches Out.Functions.dec_bits_6_backwards Out.Functions.dec_bits_6_backwards_matches Out.Functions.dec_bits_7_backwards Out.Functions.dec_bits_7_backwards_matches Out.Functions.dec_bits_8_backwards Out.Functions.dec_bits_8_backwards_matches Out.Functions.dec_bits_9_backwards Out.Functions.dec_bits_9_backwards_matches Out.Functions.dec_bits_backwards Out.Functions.dec_bits_backwards_matches Out.Functions.dec_bits_forwards Out.Functions.dec_bits_forwards_matches Out.Functions.def_spc_backwards Out.Functions.def_spc_backwards_matches Out.Functions.def_spc_forwards Out.Functions.def_spc_forwards_matches Out.Functions.diag_code_backwards_matches Out.Functions.diag_code_forwards Out.Functions.diag_code_forwards_matches Out.Functions.diag_idx Out.Functions.diag_is_data Out.Functions.diag_number Out.Functions.diag_of_num Out.Functions.drop_commits Out.Functions.dst_split_backwards_matches Out.Functions.dst_split_forwards Out.Functions.dst_split_forwards_matches Out.Functions.dsum_round Out.Functions.dsum_round_u Out.Functions.dsum_wide Out.Functions.earliest Out.Functions.earliest_int Out.Functions.ecc_class_of_num Out.Functions.ecc_classify Out.Functions.ecc_correct Out.Functions.ecc_data_index Out.Functions.ecc_encode Out.Functions.ecc_flip_delta Out.Functions.ecc_is_pow2 Out.Functions.ecc_log2 Out.Functions.ecc_ov_get Out.Functions.ecc_ov_set Out.Functions.ecc_par10 Out.Functions.ecc_poison_ov Out.Functions.ecc_poison_sl Out.Functions.ecc_pos Out.Functions.elem Out.Functions.elem_at Out.Functions.eltype_name_backwards_matches Out.Functions.eltype_name_forwards Out.Functions.eltype_name_forwards_matches Out.Functions.eltype_of_num Out.Functions.eltype_width Out.Functions.empty_plane Out.Functions.encdec_acc_mode_backwards Out.Functions.encdec_acc_mode_backwards_matches Out.Functions.encdec_acc_mode_forwards Out.Functions.encdec_acc_mode_forwards_matches Out.Functions.encdec_acctype_backwards_matches Out.Functions.encdec_acctype_forwards Out.Functions.encdec_acctype_forwards_matches Out.Functions.encdec_c2c_op_backwards_matches Out.Functions.encdec_c2c_op_forwards Out.Functions.encdec_c2c_op_forwards_matches Out.Functions.encdec_cmp_mode_backwards_matches Out.Functions.encdec_cmp_mode_forwards Out.Functions.encdec_cmp_mode_forwards_matches Out.Functions.encdec_eltype_backwards_matches Out.Functions.encdec_eltype_forwards Out.Functions.encdec_eltype_forwards_matches Out.Functions.encdec_mapop_backwards_matches Out.Functions.encdec_mapop_forwards Out.Functions.encdec_mapop_forwards_matches Out.Functions.encdec_mem_dir_backwards_matches Out.Functions.encdec_mem_dir_forwards Out.Functions.encdec_mem_dir_forwards_matches Out.Functions.encdec_reduce_op_backwards_matches Out.Functions.encdec_reduce_op_forwards Out.Functions.encdec_reduce_op_forwards_matches Out.Functions.encdec_shift_dir_backwards Out.Functions.encdec_shift_dir_backwards_matches Out.Functions.encdec_shift_dir_forwards Out.Functions.encdec_shift_dir_forwards_matches Out.Functions.encdec_vxm_binop_backwards_matches Out.Functions.encdec_vxm_binop_forwards Out.Functions.encdec_vxm_binop_forwards_matches Out.Functions.encdec_vxm_conv_backwards_matches Out.Functions.encdec_vxm_conv_forwards Out.Functions.encdec_vxm_conv_forwards_matches Out.Functions.encdec_vxm_unop_backwards_matches Out.Functions.encdec_vxm_unop_forwards Out.Functions.encdec_vxm_unop_forwards_matches Out.Functions.entry_latency Out.Functions.entry_latency_typed Out.Functions.entry_of Out.Functions.execute_C2C_DESKEW Out.Functions.execute_C2C_DESKEWRECV Out.Functions.execute_C2C_HOSTWAIT Out.Functions.execute_C2C_RTDESKEW Out.Functions.execute_ILLEGAL32 Out.Functions.execute_ILLEGAL64 Out.Functions.execute_MEM_COUNTDOWN Out.Functions.execute_MEM_DEEPSLEEP Out.Functions.execute_MEM_ITERATIONS Out.Functions.execute_MEM_POWERCONFIG Out.Functions.execute_MEM_STEP Out.Functions.execute_MEM_STREAMSTEP Out.Functions.execute_NOP Out.Functions.execute_NOTIFY Out.Functions.execute_REPEAT Out.Functions.execute_SYNC Out.Functions.exp_iv Out.Functions.exp_small Out.Functions.extensions Out.Functions.f16_dot Out.Functions.f16_exact Out.Functions.f16_row Out.Functions.f16_row_bad Out.Functions.f16_stream_values Out.Functions.fadd_core Out.Functions.fclass_eq Out.Functions.fclass_of_num Out.Functions.fcmp Out.Functions.fdecode Out.Functions.fetch_profile_causal Out.Functions.fetch_tile_available Out.Functions.fetchm_bank_note Out.Functions.fetchm_hold_until Out.Functions.field Out.Functions.fill_append Out.Functions.fill_len Out.Functions.fill_push_unit Out.Functions.fill_units_append Out.Functions.find_override Out.Functions.flip96 Out.Functions.fmt_of Out.Functions.fp8_dot Out.Functions.fp8_exact Out.Functions.fp8_format_name_backwards_matches Out.Functions.fp8_format_name_forwards Out.Functions.fp8_format_name_forwards_matches Out.Functions.fp_abs Out.Functions.fp_add Out.Functions.fp_convert Out.Functions.fp_max Out.Functions.fp_min Out.Functions.fp_mul Out.Functions.fp_neg Out.Functions.fp_of_int Out.Functions.fp_sub Out.Functions.fp_tr_value Out.Functions.from_bytes_le Out.Functions.fround Out.Functions.fval_int Out.Functions.fx_of Out.Functions.gap_or_none Out.Functions.has_extension Out.Functions.hex_bits_10_backwards_matches Out.Functions.hex_bits_11_backwards_matches Out.Functions.hex_bits_12_backwards_matches Out.Functions.hex_bits_13_backwards_matches Out.Functions.hex_bits_14_backwards_matches Out.Functions.hex_bits_15_backwards_matches Out.Functions.hex_bits_16_backwards_matches Out.Functions.hex_bits_17_backwards_matches Out.Functions.hex_bits_18_backwards_matches Out.Functions.hex_bits_19_backwards_matches Out.Functions.hex_bits_1_backwards_matches Out.Functions.hex_bits_20_backwards_matches Out.Functions.hex_bits_21_backwards_matches Out.Functions.hex_bits_22_backwards_matches Out.Functions.hex_bits_23_backwards_matches Out.Functions.hex_bits_24_backwards_matches Out.Functions.hex_bits_25_backwards_matches Out.Functions.hex_bits_26_backwards_matches Out.Functions.hex_bits_27_backwards_matches Out.Functions.hex_bits_28_backwards_matches Out.Functions.hex_bits_29_backwards_matches Out.Functions.hex_bits_2_backwards_matches Out.Functions.hex_bits_30_backwards_matches Out.Functions.hex_bits_31_backwards_matches Out.Functions.hex_bits_32_backwards_matches Out.Functions.hex_bits_33_backwards_matches Out.Functions.hex_bits_34_backwards_matches Out.Functions.hex_bits_35_backwards_matches Out.Functions.hex_bits_36_backwards_matches Out.Functions.hex_bits_37_backwards_matches Out.Functions.hex_bits_38_backwards_matches Out.Functions.hex_bits_39_backwards_matches Out.Functions.hex_bits_3_backwards_matches Out.Functions.hex_bits_40_backwards_matches Out.Functions.hex_bits_41_backwards_matches Out.Functions.hex_bits_42_backwards_matches Out.Functions.hex_bits_43_backwards_matches Out.Functions.hex_bits_44_backwards_matches Out.Functions.hex_bits_45_backwards_matches Out.Functions.hex_bits_46_backwards_matches Out.Functions.hex_bits_47_backwards_matches Out.Functions.hex_bits_48_backwards_matches Out.Functions.hex_bits_49_backwards_matches Out.Functions.hex_bits_4_backwards_matches Out.Functions.hex_bits_50_backwards_matches Out.Functions.hex_bits_51_backwards_matches Out.Functions.hex_bits_52_backwards_matches Out.Functions.hex_bits_53_backwards_matches Out.Functions.hex_bits_54_backwards_matches Out.Functions.hex_bits_55_backwards_matches Out.Functions.hex_bits_56_backwards_matches Out.Functions.hex_bits_57_backwards_matches Out.Functions.hex_bits_58_backwards_matches Out.Functions.hex_bits_59_backwards_matches Out.Functions.hex_bits_5_backwards_matches Out.Functions.hex_bits_60_backwards_matches Out.Functions.hex_bits_61_backwards_matches Out.Functions.hex_bits_62_backwards_matches Out.Functions.hex_bits_63_backwards_matches Out.Functions.hex_bits_64_backwards_matches Out.Functions.hex_bits_6_backwards_matches Out.Functions.hex_bits_7_backwards_matches Out.Functions.hex_bits_8_backwards_matches Out.Functions.hex_bits_9_backwards_matches Out.Functions.hex_bits_backwards Out.Functions.hex_bits_backwards_matches Out.Functions.hex_bits_forwards Out.Functions.hex_bits_forwards_matches Out.Functions.hold_decide Out.Functions.holds_f16 Out.Functions.i8_tr_value Out.Functions.identity_map Out.Functions.idiv_ceil Out.Functions.idiv_floor Out.Functions.ilog2 Out.Functions.in_barrier_window Out.Functions.inf_bits Out.Functions.int8_dot Out.Functions.int_of Out.Functions.int_of_fp Out.Functions.inward_row Out.Functions.iq_padding Out.Functions.is_bool Out.Functions.is_dst Out.Functions.is_float Out.Functions.is_int Out.Functions.is_loopable Out.Functions.is_mxm_queue Out.Functions.is_nan Out.Functions.is_none Out.Functions.is_pair Out.Functions.is_prm Out.Functions.is_sel Out.Functions.is_shf Out.Functions.is_some Out.Functions.is_tsp Out.Functions.is_unsigned Out.Functions.isqrt Out.Functions.issued_by Out.Functions.issuer_of Out.Functions.items_len Out.Functions.k_binary Out.Functions.k_cmp Out.Functions.k_convert Out.Functions.k_convert_any Out.Functions.k_nonzero Out.Functions.k_unary Out.Functions.lane8 Out.Functions.lane_byte Out.Functions.latency_of Out.Functions.latency_of_instr Out.Functions.link_flit_mask Out.Functions.link_pow2 Out.Functions.lint Out.Functions.list_len Out.Functions.ln2_iv Out.Functions.log_iv Out.Functions.mapop_name_backwards_matches Out.Functions.mapop_name_forwards Out.Functions.mapop_name_forwards_matches Out.Functions.mem_dir_mnemonic_backwards_matches Out.Functions.mem_dir_mnemonic_forwards Out.Functions.mem_dir_mnemonic_forwards_matches Out.Functions.mem_quad Out.Functions.mem_slice_present Out.Functions.mem_unit Out.Functions.merge_tiles Out.Functions.model_link_flit_mask Out.Functions.model_link_flits Out.Functions.model_ptype_queued Out.Functions.mxm_acctype_of_num Out.Functions.mxm_bad Out.Functions.mxm_cap Out.Functions.mxm_check Out.Functions.mxm_clash Out.Functions.mxm_part_of_num Out.Functions.mxm_queue_present Out.Functions.mxm_unit Out.Functions.mxm_wtype_of_num Out.Functions.nbits_of Out.Functions.neq_bool Out.Functions.nonempty_list Out.Functions.not Out.Functions.not_bit Out.Functions.num_of_acc_mode Out.Functions.num_of_agt_field Out.Functions.num_of_agt_state Out.Functions.num_of_c2c_op Out.Functions.num_of_c2c_ptype Out.Functions.num_of_diag Out.Functions.num_of_ecc_class Out.Functions.num_of_eltype Out.Functions.num_of_fclass Out.Functions.num_of_mxm_acctype Out.Functions.num_of_mxm_part Out.Functions.num_of_mxm_wtype Out.Functions.num_of_row Out.Functions.num_of_scrub_action Out.Functions.num_of_srf_lookup Out.Functions.num_of_timing_class Out.Functions.num_of_unit_kind Out.Functions.one_plane Out.Functions.opt_spc_backwards Out.Functions.opt_spc_backwards_matches Out.Functions.opt_spc_forwards Out.Functions.opt_spc_forwards_matches Out.Functions.or_pair Out.Functions.outward_row Out.Functions.ov_nth Out.Functions.pair_exact Out.Functions.path_hold_until Out.Functions.path_views Out.Functions.place Out.Functions.place_all Out.Functions.poll_epoch Out.Functions.pow2i Out.Functions.q_in Out.Functions.queue_present Out.Functions.ras_idx Out.Functions.ras_loc Out.Functions.ras_sat_inc Out.Functions.read_gap Out.Functions.reduce_mnemonic_backwards_matches Out.Functions.reduce_mnemonic_forwards Out.Functions.reduce_mnemonic_forwards_matches Out.Functions.ref_bank Out.Functions.ref_path Out.Functions.ref_thread Out.Functions.rep_sp_set Out.Functions.repeat_after Out.Functions.repeat_plan Out.Functions.repeatable Out.Functions.rev_items Out.Functions.ring Out.Functions.rne_int Out.Functions.round_iv Out.Functions.row_of_num Out.Functions.rows_of Out.Functions.s_of_bits Out.Functions.sail_mask Out.Functions.sail_ones Out.Functions.same_target Out.Functions.sat_s Out.Functions.sat_t Out.Functions.scrub_action_of_num Out.Functions.scrub_arbitrate Out.Functions.scrub_due_count Out.Functions.scrub_slot_taken Out.Functions.sep_forwards Out.Functions.sep_forwards_matches Out.Functions.set_lane_byte Out.Functions.sg4_suffix_backwards_matches Out.Functions.sg4_suffix_forwards Out.Functions.sg4_suffix_forwards_matches Out.Functions.shift_dir_name_backwards_matches Out.Functions.shift_dir_name_forwards Out.Functions.shift_dir_name_forwards_matches Out.Functions.slice_mask Out.Functions.slot_of Out.Functions.spc_backwards Out.Functions.spc_backwards_matches Out.Functions.spc_forwards Out.Functions.spc_forwards_matches Out.Functions.sram_addr_of Out.Functions.sram_bank Out.Functions.sram_field_backwards Out.Functions.sram_field_backwards_matches Out.Functions.sram_field_forwards Out.Functions.sram_field_forwards_matches Out.Functions.sram_index Out.Functions.sram_text_backwards Out.Functions.sram_text_backwards_matches Out.Functions.sram_text_forwards Out.Functions.sram_text_forwards_matches Out.Functions.srf_lookup_of_num Out.Functions.srf_ov_spare Out.Functions.srf_rep_merge Out.Functions.srf_slot_best Out.Functions.srf_slot_chain_ov Out.Functions.srf_slot_collides Out.Functions.srf_slot_flip Out.Functions.srf_slot_read Out.Functions.srf_slot_read_ov Out.Functions.srf_slot_write Out.Functions.srf_stream_only_downstream Out.Functions.stream_id Out.Functions.stream_name_forwards_matches Out.Functions.stream_row Out.Functions.string_of_exception Out.Functions.sxm_bad Out.Functions.sxm_clash Out.Functions.sxm_lane Out.Functions.sxm_map_distinct Out.Functions.sxm_or_kind Out.Functions.sxm_set_lane Out.Functions.sxm_stream_plus Out.Functions.sxm_unit_at Out.Functions.tag_of Out.Functions.tags_clash Out.Functions.text_width Out.Functions.tile_active Out.Functions.tile_addr Out.Functions.timing_class_name Out.Functions.timing_class_of_num Out.Functions.timing_of Out.Functions.timing_overrides Out.Functions.timing_type_of Out.Functions.to_bytes_le Out.Functions.tr_eval Out.Functions.tr_eval_int Out.Functions.tr_interval Out.Functions.ty_split_backwards_matches Out.Functions.ty_split_forwards Out.Functions.ty_split_forwards_matches Out.Functions.ty_valid Out.Functions.unit_kind_of_num Out.Functions.unop_entry_backwards_matches Out.Functions.unop_entry_forwards Out.Functions.unop_entry_forwards_matches Out.Functions.unop_legal Out.Functions.vec_units Out.Functions.view_free Out.Functions.view_paths_busy Out.Functions.vsrc_name_forwards_matches Out.Functions.vxm_binop_mnemonic_backwards_matches Out.Functions.vxm_binop_mnemonic_forwards Out.Functions.vxm_binop_mnemonic_forwards_matches Out.Functions.vxm_unit Out.Functions.vxm_unop_mnemonic_backwards_matches Out.Functions.vxm_unop_mnemonic_forwards Out.Functions.vxm_unop_mnemonic_forwards_matches Out.Functions.wait_reason Out.Functions.wants_pair Out.Functions.wlane_add Out.Functions.wlane_read Out.Functions.words_to_planes Out.Functions.wrap_s Out.Functions.wrap_t Out.Functions.write_gap Out.Functions.writer_before Out.Functions.wtype_name_backwards_matches Out.Functions.wtype_name_forwards Out.Functions.wtype_name_forwards_matches Out.Functions.wtype_ok Out.Functions.wtype_split_backwards_matches Out.Functions.wtype_split_forwards Out.Functions.wtype_split_forwards_matches Out.Functions.wtype_valid Out.Functions.zero_bits Out.abs_real Out.add_real Out.diag_callback Out.div_real Out.ecc_event_callback Out.eq_real Out.gt_real Out.gteq_real Out.lint_callback Out.lt_real Out.lteq_real Out.mult_real Out.mxm_corr_callback Out.neg_real Out.prerr_real Out.print_real Out.random_real Out.real_power Out.round_down Out.round_up Out.scrub_event_callback Out.spare_event_callback Out.sqrt_real Out.sram_write_callback Out.srf_write_callback Out.sub_real Out.tdiv_int Out.tmod_int Out.to_real ecc_begin exec_cycle host_deliver host_msix execute_C2C_HOSTNOTIFY commit_at ecc_poison_mask ecc_result_ov plane_stream write_planes_fwd write_planes execute_C2C_PCIRECV pci_busy ras_record ras_ecc_event ras_spare_event get_stream srf_spare srf_s03 ecc_check_value srf_read_chain_ov srf_only_downstream srf_read_ov read_planes_full read_planes_core read_planes execute_C2C_PCISEND c2c_link c2c_peer_ok execute_C2C_RECEIVE c2c_busy c2c_put execute_C2C_SEND execute_C2C_TRANSMIT execute_CONFIG stream_fetch_arm stream_fetch_start execute_IFETCH execute_IFETCHM commit_cycle eff_addr eff_stream read_cycle read_plane_raw sram_read sram_read_ov write_sram_ecc write_sram_word sram_write_ov tile_on execute_MEM_DIRECT ecc_check_sl srf_report mem_report execute_MEM_GATHER execute_MEM_SCATTER execute_MEM_SETSEGMENT macc_busy_in macc_busy mxm_stage_acc mxm_stage plane mxm_plane_at plane_blocked ras_mxm_event abc_exec execute_MXM_ABC execute_MXM_ABC8 execute_MXM_ACC execute_MXM_IW read_planes16 execute_MXM_IWS execute_MXM_LW execute_SETVL sx sxm_load_map sxm_rd_cycle execute_SXM_LOADMAP execute_SXM_LOADMAPWIDE sxm_cm_cycle sxm_emit execute_SXM_MAP execute_SXM_ROTATE execute_SXM_SELECT execute_SXM_SHIFT execute_SXM_TRANSPOSE Out.sailThrow internal_error fp_transcendental i8_transcendental apply vxm_commit_cycle vxm_read_cycle vxm_read vxm_lanes execute_VXM_BIN execute_VXM_CMP execute_VXM_CONV execute_VXM_CVT execute_VXM_MASK execute_VXM_REDUCE execute_VXM_SELECT execute_VXM_UN execute diag_record diag_report report execute_element run_element agt_get agt_set agt_configure host_parked path_busy path_tail path_view_of polling parked_one queue_parked hold_view_of hold_check dispatch_check dispatch_cfg agt_due due_refs agt_refresh bank_report hold_path loop_clash_ref run_loop
attribute [local irreducible] Out.Functions.deskewrecv_ready Out.Functions.occupancy Out.Functions.agt_finish_config Out.Functions.reserve_local_fetch Out.Functions.repeat_issue Out.Functions.due_ref Out.Functions.loop_tag Out.Functions.dispatch_body Out.Functions.dispatch_decoded Out.Functions.dispatch_slots Out.Functions.dispatch Out.Functions.strict_held Out.Functions.parcel_held
theorem deskewrecv_ready_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) (link : (BitVec 7)) : Keeps RView (deskewrecv_ready c link : SailM Bool) := by
  unfold deskewrecv_ready
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.c2c_link_frame ports
    | apply ports.c2c_recv_ready
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem occupancy_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (arg0 : instruction) (arg1 : Nat) : Keeps RView (occupancy arg0 arg1 : SailM Nat) := by
  unfold occupancy
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem agt_finish_config_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (p : Nat) : Keeps RView (agt_finish_config q p : SailM Unit) := by
  unfold agt_finish_config
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.agt_get_frame ports
    | apply ResourceExecuteFrameProofs.agt_set_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem reserve_local_fetch_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (i : instruction) (c : Nat) : Keeps RView (reserve_local_fetch q i c : SailM Unit) := by
  unfold reserve_local_fetch
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem repeat_issue_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (i : instruction) (c : Nat) : Keeps RView (repeat_issue q i c : SailM Unit) := by
  unfold repeat_issue
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.run_element_frame ports
    | apply reserve_local_fetch_iq_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem c2c_link_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (link : (BitVec 7)) : Keeps RView (c2c_link link : SailM (Option Nat)) := by
  unfold c2c_link
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem c2c_peer_ok_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (chip : (BitVec 9)) (l : Nat) : Keeps RView (c2c_peer_ok chip l : SailM Bool) := by
  unfold c2c_peer_ok
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ports.c2c_peer
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem due_ref_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Keeps RView (due_ref q p st : SailM (Option (instruction × Int × Int))) := by
  unfold due_ref
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.loop_clash_ref_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem loop_tag_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Keeps RView (loop_tag q p st : SailM (Option ref_tag)) := by
  unfold loop_tag
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.agt_get_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem deskewrecv_diag_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (iss : issuer) (chip : BitVec 9) (link : BitVec 7) (stream : streamref) :
    Keeps RView (deskewrecv_diag iss chip link stream) := by
  unfold deskewrecv_diag
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | apply ResourceExecuteFrameProofs.c2c_link_frame ports
    | apply ResourceExecuteFrameProofs.c2c_peer_ok_frame ports
    | apply bind_keeps
    | split
    | intro

theorem dispatch_body_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) (i : instruction) : Keeps RView (dispatch_body q c i : SailM Unit) := by
  unfold dispatch_body
  repeat' first
    | assumption
    | apply deskewrecv_diag_iq_frame ports
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.report_frame ports
    | apply ResourceExecuteFrameProofs.run_element_frame ports
    | apply occupancy_iq_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem dispatch_decoded_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) (i : instruction) : Keeps RView (dispatch_decoded q c i : SailM Unit) := by
  unfold dispatch_decoded
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.report_frame ports
    | apply ResourceExecuteFrameProofs.dispatch_check_frame ports
    | apply repeat_issue_iq_frame ports
    | apply dispatch_body_iq_frame ports
    | apply ports.observe_site
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem cfg_path_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) : Keeps RView (cfg_path q : SailM Nat) := by
  unfold cfg_path
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem consumeNewRef_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) : Keeps RView (GrantConsumptionProofs.consumeNewRef q c r loop lref go bank k issued last) := by
  unfold GrantConsumptionProofs.consumeNewRef
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ResourceExecuteFrameProofs.agt_set_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.bank_report_frame ports _ _ _ _
    | exact cfg_path_iq_frame ports _
    | exact ResourceExecuteFrameProofs.dispatch_cfg_frame ports _ _ _ _
    | exact ResourceExecuteFrameProofs.dispatch_check_frame ports _ _ _ _
    | exact due_ref_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.due_refs_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.hold_path_frame ports _ _ _
    | exact loop_tag_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.report_frame ports _ _ _
    | exact reserve_local_fetch_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.run_element_frame ports _ _ _ _ _ _
    | exact ResourceExecuteFrameProofs.run_loop_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem consumeFetch_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q c : Nat) (r : cycle_recs) (fm_ok fm_clash : Bool)
    (issued : Nat) (last : Option instruction) : Keeps RView (GrantConsumptionProofs.consumeFetch q c r fm_ok fm_clash issued last) := by
  unfold GrantConsumptionProofs.consumeFetch
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ResourceExecuteFrameProofs.agt_set_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.bank_report_frame ports _ _ _ _
    | exact cfg_path_iq_frame ports _
    | exact ResourceExecuteFrameProofs.dispatch_cfg_frame ports _ _ _ _
    | exact ResourceExecuteFrameProofs.dispatch_check_frame ports _ _ _ _
    | exact due_ref_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.due_refs_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.hold_path_frame ports _ _ _
    | exact loop_tag_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.report_frame ports _ _ _
    | exact reserve_local_fetch_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.run_element_frame ports _ _ _ _ _ _
    | exact ResourceExecuteFrameProofs.run_loop_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem dispatch_slots_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : Keeps RView (dispatch_slots q c r parked : SailM Unit) := by
  rw [GrantConsumptionProofs.actual_dispatch_factory_source]
  unfold GrantConsumptionProofs.dispatchFactory
  repeat' first
    | exact consumeNewRef_iq_frame ports _ _ _ _ _ _ _ _ _ _
    | exact consumeFetch_iq_frame ports _ _ _ _ _ _ _
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ResourceExecuteFrameProofs.agt_set_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.bank_report_frame ports _ _ _ _
    | exact cfg_path_iq_frame ports _
    | exact ResourceExecuteFrameProofs.dispatch_cfg_frame ports _ _ _ _
    | exact ResourceExecuteFrameProofs.dispatch_check_frame ports _ _ _ _
    | exact due_ref_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.due_refs_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.hold_path_frame ports _ _ _
    | exact loop_tag_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.report_frame ports _ _ _
    | exact reserve_local_fetch_iq_frame ports _ _ _
    | exact ResourceExecuteFrameProofs.run_element_frame ports _ _ _ _ _ _
    | exact ResourceExecuteFrameProofs.run_loop_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem dispatch_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : Keeps RView (dispatch q c r parked : SailM Unit) := by
  unfold dispatch
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.report_frame ports
    | apply ResourceExecuteFrameProofs.due_refs_frame ports
    | apply ResourceExecuteFrameProofs.run_loop_frame ports
    | apply due_ref_iq_frame ports
    | apply loop_tag_iq_frame ports
    | apply dispatch_decoded_iq_frame ports
    | apply dispatch_slots_iq_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem strict_held_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) (xs : (List (Result instruction diag))) (held : Bool) : Keeps RView (strict_held q c xs held : SailM Bool) := by
  induction xs generalizing held with
  | nil => unfold strict_held; exact pure_keeps _ _
  | cons y ys ih =>
    unfold strict_held
    repeat' first
      | assumption
      | apply ih
      | exact pure_keeps _ _
      | exact ResourceExecuteFrameProofs.throw_frame _
      | exact read_keeps _ _
      | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
      | apply ResourceExecuteFrameProofs.hold_check_frame ports
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro


theorem parcel_held_iq_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) (r : cycle_recs) (items : (List (Result instruction diag))) : Keeps RView (parcel_held q c r items : SailM Bool) := by
  unfold parcel_held
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact ResourceExecuteFrameProofs.throw_frame _
    | exact read_keeps _ _
    | exact ResourceExecuteFrameProofs.write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | apply ResourceExecuteFrameProofs.hold_check_frame ports
    | apply strict_held_iq_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

end MetadataFrames

theorem queue_parked_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (c : Nat) : Safe A (H:=H) (L:=L) (queue_parked q c : SailM Bool) :=
  frame_complete_safe A H L _ (ResourceExecuteFrameProofs.queue_parked_frame rp q c) (ResourceCompleteProofs.queue_parked_complete cp q c)

theorem deskewrecv_ready_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (c : Nat) (link : (BitVec 7)) : Safe A (H:=H) (L:=L) (deskewrecv_ready c link : SailM Bool) :=
  frame_complete_safe A H L _ (deskewrecv_ready_iq_frame rp c link) (ResourceCompleteProofs.deskewrecv_ready_complete cp c link)

theorem occupancy_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (arg0 : instruction) (arg1 : Nat) : Safe A (H:=H) (L:=L) (occupancy arg0 arg1 : SailM Nat) :=
  frame_complete_safe A H L _ (occupancy_iq_frame rp arg0 arg1) (ResourceCompleteProofs.occupancy_complete cp arg0 arg1)

theorem run_element_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (i : instruction) (c0 : Nat) (k : Nat) (off : Int) (soff : Int) : Safe A (H:=H) (L:=L) (run_element q i c0 k off soff : SailM Unit) :=
  frame_complete_safe A H L _ (ResourceExecuteFrameProofs.run_element_frame rp q i c0 k off soff) (ResourceCompleteProofs.run_element_complete cp q i c0 k off soff)

theorem repeat_issue_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (i : instruction) (c : Nat) : Safe A (H:=H) (L:=L) (repeat_issue q i c : SailM Unit) :=
  frame_complete_safe A H L _ (repeat_issue_iq_frame rp q i c) (ResourceCompleteProofs.repeat_issue_complete cp q i c)

theorem agt_get_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (p : Nat) : Safe A (H:=H) (L:=L) (agt_get q p : SailM agt) :=
  frame_complete_safe A H L _ (ResourceExecuteFrameProofs.agt_get_frame rp q p) (ResourceCompleteProofs.agt_get_complete cp q p)

theorem agt_finish_config_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (p : Nat) : Safe A (H:=H) (L:=L) (agt_finish_config q p : SailM Unit) :=
  frame_complete_safe A H L _ (agt_finish_config_iq_frame rp q p) (ResourceCompleteProofs.agt_finish_config_complete cp q p)

theorem parcel_held_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (c : Nat) (r : cycle_recs) (items : (List (Result instruction diag))) : Safe A (H:=H) (L:=L) (parcel_held q c r items : SailM Bool) :=
  frame_complete_safe A H L _ (parcel_held_iq_frame rp q c r items) (ResourceCompleteProofs.parcel_held_complete cp q c r items)

theorem host_parked_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (c : Nat) : Safe A (H:=H) (L:=L) (host_parked q c : SailM Bool) :=
  frame_complete_safe A H L _ (ResourceExecuteFrameProofs.host_parked_frame rp q c) (ResourceCompleteProofs.host_parked_complete cp q c)

theorem polling_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) : Safe A (H:=H) (L:=L) (polling q : SailM Bool) :=
  frame_complete_safe A H L _ (ResourceExecuteFrameProofs.polling_frame rp q) (ResourceCompleteProofs.polling_complete cp q)

theorem dispatch_safe (A : Nat → Prop) (H L : Vector Nat 145) (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports) (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : Safe A (H:=H) (L:=L) (dispatch q c r parked : SailM Unit) :=
  frame_complete_safe A H L _ (dispatch_iq_frame rp q c r parked) (ResourceCompleteProofs.dispatch_complete cp q c r parked)


theorem earliest_safe (A : Nat → Prop) (H L : Vector Nat 145) (q : Nat) (p : parcel) :
    Safe A (H:=H) (L:=L) (parcel_earliest q p) :=
  frame_safe A _ (fun s _ => Proofs.ControlConsumption.earliest_frame s q p)

theorem observer_zero_safe (A : Nat → Prop) (H L : Vector Nat 145)
    (observer : Proofs.ControlConsumption.DirectObserver A)
    (event : String) (c q head : Nat) (detail : Int) (hq : q<145) :
    Safe A (H:=H) (L:=L) (observe_site event c q head 0 detail) := by
  intro s keep
  obtain ⟨after,run,hr,hm⟩ := observer.frame event c q head 0 detail s keep.2.2.1 hq (by omega) (by omega)
  have kept := Proofs.ControlConsumption.keep_same_parts A H L s after keep hr hm
  simpa [run] using kept

theorem observer_head_safe (A : Nat → Prop) (H L : Vector Nat 145)
    (observer : Proofs.ControlConsumption.DirectObserver A)
    (event : String) (c q units : Nat) (detail : Int)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hu : units≤L[q]!) :
    Safe A (H:=H) (L:=L) (observe_site event c q H[q]! units detail) := by
  intro s keep
  obtain ⟨after,run,hr,hm⟩ := observer.frame event c q H[q]! units detail s keep.2.2.1 hq (by omega)
    (Proofs.ControlConsumption.direct_dispatch_domain A H L s keep q units hq hh hu)
  have kept := Proofs.ControlConsumption.keep_same_parts A H L s after keep hr hm
  simpa [run] using kept

def Satisfies {α : Type} (A : Nat → Prop) (H L : Vector Nat 145)
    (post : ResourceFrameProofs.St → Prop) (action : SailM α) : Prop :=
  ∀ s, Keep A H L s → match action.run s with
    | .ok _ after => post after
    | .error _ after => post after

theorem satisfies_bind {α β : Type} (A : Nat → Prop) (H L : Vector Nat 145)
    (post : ResourceFrameProofs.St → Prop) (a : SailM α) (f : α → SailM β)
    (ha : Safe A (H:=H) (L:=L) a)
    (hf : ∀ x, Satisfies A H L post (f x))
    (old : ∀ s, Keep A H L s → post s) : Satisfies A H L post (a >>= f) := by
  intro s keep
  have ia := ha s keep
  cases run : a.run s with
  | error error after =>
    have kept : Keep A H L after := by simpa [run] using ia
    change a s=.error error after at run
    simpa [EStateM.run,bind,EStateM.bind,run] using old after kept
  | ok x middle =>
    have kept : Keep A H L middle := by simpa [run] using ia
    change a s=.ok x middle at run
    simpa [EStateM.run,bind,EStateM.bind,run] using hf x middle kept

theorem satisfies_head_bind {α : Type} (A : Nat → Prop) (H L : Vector Nat 145)
    (post : ResourceFrameProofs.St → Prop) (f : Vector Nat 145 → SailM α)
    (hf : Satisfies A H L post (f H)) : Satisfies A H L post (Out.readReg iq_head >>= f) := by
  intro s keep
  have read : Out.readReg iq_head s=.ok H s := by
    simp [Out.readReg,PreSail.readReg,keep.1.2.1,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,read] using hf s keep

theorem satisfies_safe {α : Type} (A : Nat → Prop) (H L : Vector Nat 145)
    (post : ResourceFrameProofs.St → Prop) (a : SailM α) (ha : Safe A (H:=H) (L:=L) a)
    (old : ∀ s, Keep A H L s → post s) : Satisfies A H L post a := by
  intro s keep
  have h := ha s keep
  cases run : a.run s <;> simpa [run] using old _ (by simpa [run] using h)

attribute [local irreducible] Out.Functions.__id Out.Functions.abc_after Out.Functions.abc_at Out.Functions.abc_fp8 Out.Functions.abc_others Out.Functions.abc_others_fp8 Out.Functions.abc_pair_pick Out.Functions.abc_prune Out.Functions.abc_types_ok Out.Functions.acc_mode_name_backwards_matches Out.Functions.acc_mode_name_forwards Out.Functions.acc_mode_name_forwards_matches Out.Functions.acc_mode_of_num Out.Functions.acctype_declared Out.Functions.acctype_name_backwards_matches Out.Functions.acctype_name_forwards Out.Functions.acctype_name_forwards_matches Out.Functions.addr_bank Out.Functions.agt_advance Out.Functions.agt_advance_test Out.Functions.agt_after_reference Out.Functions.agt_config Out.Functions.agt_configure_pure Out.Functions.agt_due_pure Out.Functions.agt_field_of_num Out.Functions.agt_field_pos Out.Functions.agt_finish_config_pure Out.Functions.agt_nlevels Out.Functions.agt_opened Out.Functions.agt_refresh_pure Out.Functions.agt_state_of_num Out.Functions.aligned Out.Functions.all_ones Out.Functions.append_element Out.Functions.assembly_forwards_matches Out.Functions.bank_rule Out.Functions.banks_meet Out.Functions.bf16_dot Out.Functions.bfp8_exact Out.Functions.binop_entry_backwards_matches Out.Functions.binop_entry_forwards Out.Functions.binop_entry_forwards_matches Out.Functions.binop_legal Out.Functions.bit_index Out.Functions.bit_of Out.Functions.bits_of_s Out.Functions.bits_t Out.Functions.blk_exp Out.Functions.bool_bits Out.Functions.c2c_op_of_num Out.Functions.c2c_packet_flits Out.Functions.c2c_poison_int Out.Functions.c2c_ptype_code Out.Functions.c2c_ptype_of_num Out.Functions.c2c_ptype_queued Out.Functions.c2c_rx Out.Functions.c2c_tx Out.Functions.canon_nan Out.Functions.cfg_path_pick Out.Functions.chain_suffix_backwards_matches Out.Functions.chain_suffix_forwards Out.Functions.chain_suffix_forwards_matches Out.Functions.chk_elem Out.Functions.chk_wrow Out.Functions.clash_addr Out.Functions.clash_stream Out.Functions.classify_items Out.Functions.clears_repeat Out.Functions.cmp_legal Out.Functions.cmp_mnemonic_backwards_matches Out.Functions.cmp_mnemonic_forwards Out.Functions.cmp_mnemonic_forwards_matches Out.Functions.commit_scan Out.Functions.commit_wins Out.Functions.config_ends Out.Functions.config_for Out.Functions.conv_dst Out.Functions.conv_entry_backwards_matches Out.Functions.conv_entry_forwards Out.Functions.conv_entry_forwards_matches Out.Functions.conv_mnemonic_backwards_matches Out.Functions.conv_mnemonic_forwards Out.Functions.conv_mnemonic_forwards_matches Out.Functions.conv_src Out.Functions.conv_src_type Out.Functions.conv_types Out.Functions.csr_value Out.Functions.db_idx Out.Functions.dec_bits_10_backwards Out.Functions.dec_bits_10_backwards_matches Out.Functions.dec_bits_1_backwards Out.Functions.dec_bits_1_backwards_matches Out.Functions.dec_bits_2_backwards Out.Functions.dec_bits_2_backwards_matches Out.Functions.dec_bits_32_backwards Out.Functions.dec_bits_32_backwards_matches Out.Functions.dec_bits_3_backwards Out.Functions.dec_bits_3_backwards_matches Out.Functions.dec_bits_4_backwards Out.Functions.dec_bits_4_backwards_matches Out.Functions.dec_bits_5_backwards Out.Functions.dec_bits_5_backwards_matches Out.Functions.dec_bits_6_backwards Out.Functions.dec_bits_6_backwards_matches Out.Functions.dec_bits_7_backwards Out.Functions.dec_bits_7_backwards_matches Out.Functions.dec_bits_8_backwards Out.Functions.dec_bits_8_backwards_matches Out.Functions.dec_bits_9_backwards Out.Functions.dec_bits_9_backwards_matches Out.Functions.dec_bits_backwards Out.Functions.dec_bits_backwards_matches Out.Functions.dec_bits_forwards Out.Functions.dec_bits_forwards_matches Out.Functions.def_spc_backwards Out.Functions.def_spc_backwards_matches Out.Functions.def_spc_forwards Out.Functions.def_spc_forwards_matches Out.Functions.diag_code_backwards_matches Out.Functions.diag_code_forwards Out.Functions.diag_code_forwards_matches Out.Functions.diag_idx Out.Functions.diag_is_data Out.Functions.diag_number Out.Functions.diag_of_num Out.Functions.drop_commits Out.Functions.dst_split_backwards_matches Out.Functions.dst_split_forwards Out.Functions.dst_split_forwards_matches Out.Functions.dsum_round Out.Functions.dsum_round_u Out.Functions.dsum_wide Out.Functions.earliest Out.Functions.earliest_int Out.Functions.ecc_class_of_num Out.Functions.ecc_classify Out.Functions.ecc_correct Out.Functions.ecc_data_index Out.Functions.ecc_encode Out.Functions.ecc_flip_delta Out.Functions.ecc_is_pow2 Out.Functions.ecc_log2 Out.Functions.ecc_ov_get Out.Functions.ecc_ov_set Out.Functions.ecc_par10 Out.Functions.ecc_poison_ov Out.Functions.ecc_poison_sl Out.Functions.ecc_pos Out.Functions.elem Out.Functions.elem_at Out.Functions.eltype_name_backwards_matches Out.Functions.eltype_name_forwards Out.Functions.eltype_name_forwards_matches Out.Functions.eltype_of_num Out.Functions.eltype_width Out.Functions.empty_plane Out.Functions.encdec_acc_mode_backwards Out.Functions.encdec_acc_mode_backwards_matches Out.Functions.encdec_acc_mode_forwards Out.Functions.encdec_acc_mode_forwards_matches Out.Functions.encdec_acctype_backwards_matches Out.Functions.encdec_acctype_forwards Out.Functions.encdec_acctype_forwards_matches Out.Functions.encdec_c2c_op_backwards_matches Out.Functions.encdec_c2c_op_forwards Out.Functions.encdec_c2c_op_forwards_matches Out.Functions.encdec_cmp_mode_backwards_matches Out.Functions.encdec_cmp_mode_forwards Out.Functions.encdec_cmp_mode_forwards_matches Out.Functions.encdec_eltype_backwards_matches Out.Functions.encdec_eltype_forwards Out.Functions.encdec_eltype_forwards_matches Out.Functions.encdec_mapop_backwards_matches Out.Functions.encdec_mapop_forwards Out.Functions.encdec_mapop_forwards_matches Out.Functions.encdec_mem_dir_backwards_matches Out.Functions.encdec_mem_dir_forwards Out.Functions.encdec_mem_dir_forwards_matches Out.Functions.encdec_reduce_op_backwards_matches Out.Functions.encdec_reduce_op_forwards Out.Functions.encdec_reduce_op_forwards_matches Out.Functions.encdec_shift_dir_backwards Out.Functions.encdec_shift_dir_backwards_matches Out.Functions.encdec_shift_dir_forwards Out.Functions.encdec_shift_dir_forwards_matches Out.Functions.encdec_vxm_binop_backwards_matches Out.Functions.encdec_vxm_binop_forwards Out.Functions.encdec_vxm_binop_forwards_matches Out.Functions.encdec_vxm_conv_backwards_matches Out.Functions.encdec_vxm_conv_forwards Out.Functions.encdec_vxm_conv_forwards_matches Out.Functions.encdec_vxm_unop_backwards_matches Out.Functions.encdec_vxm_unop_forwards Out.Functions.encdec_vxm_unop_forwards_matches Out.Functions.entry_latency Out.Functions.entry_latency_typed Out.Functions.entry_of Out.Functions.execute_C2C_DESKEW Out.Functions.execute_C2C_DESKEWRECV Out.Functions.execute_C2C_HOSTWAIT Out.Functions.execute_C2C_RTDESKEW Out.Functions.execute_ILLEGAL32 Out.Functions.execute_ILLEGAL64 Out.Functions.execute_MEM_COUNTDOWN Out.Functions.execute_MEM_DEEPSLEEP Out.Functions.execute_MEM_ITERATIONS Out.Functions.execute_MEM_POWERCONFIG Out.Functions.execute_MEM_STEP Out.Functions.execute_MEM_STREAMSTEP Out.Functions.execute_NOP Out.Functions.execute_NOTIFY Out.Functions.execute_REPEAT Out.Functions.execute_SYNC Out.Functions.exp_iv Out.Functions.exp_small Out.Functions.extensions Out.Functions.f16_dot Out.Functions.f16_exact Out.Functions.f16_row Out.Functions.f16_row_bad Out.Functions.f16_stream_values Out.Functions.fadd_core Out.Functions.fclass_eq Out.Functions.fclass_of_num Out.Functions.fcmp Out.Functions.fdecode Out.Functions.fetch_profile_causal Out.Functions.fetch_tile_available Out.Functions.fetchm_bank_note Out.Functions.fetchm_hold_until Out.Functions.field Out.Functions.fill_append Out.Functions.fill_len Out.Functions.fill_push_unit Out.Functions.fill_units_append Out.Functions.find_override Out.Functions.flip96 Out.Functions.fmt_of Out.Functions.fp8_dot Out.Functions.fp8_exact Out.Functions.fp8_format_name_backwards_matches Out.Functions.fp8_format_name_forwards Out.Functions.fp8_format_name_forwards_matches Out.Functions.fp_abs Out.Functions.fp_add Out.Functions.fp_convert Out.Functions.fp_max Out.Functions.fp_min Out.Functions.fp_mul Out.Functions.fp_neg Out.Functions.fp_of_int Out.Functions.fp_sub Out.Functions.fp_tr_value Out.Functions.from_bytes_le Out.Functions.fround Out.Functions.fval_int Out.Functions.fx_of Out.Functions.gap_or_none Out.Functions.has_extension Out.Functions.hex_bits_10_backwards_matches Out.Functions.hex_bits_11_backwards_matches Out.Functions.hex_bits_12_backwards_matches Out.Functions.hex_bits_13_backwards_matches Out.Functions.hex_bits_14_backwards_matches Out.Functions.hex_bits_15_backwards_matches Out.Functions.hex_bits_16_backwards_matches Out.Functions.hex_bits_17_backwards_matches Out.Functions.hex_bits_18_backwards_matches Out.Functions.hex_bits_19_backwards_matches Out.Functions.hex_bits_1_backwards_matches Out.Functions.hex_bits_20_backwards_matches Out.Functions.hex_bits_21_backwards_matches Out.Functions.hex_bits_22_backwards_matches Out.Functions.hex_bits_23_backwards_matches Out.Functions.hex_bits_24_backwards_matches Out.Functions.hex_bits_25_backwards_matches Out.Functions.hex_bits_26_backwards_matches Out.Functions.hex_bits_27_backwards_matches Out.Functions.hex_bits_28_backwards_matches Out.Functions.hex_bits_29_backwards_matches Out.Functions.hex_bits_2_backwards_matches Out.Functions.hex_bits_30_backwards_matches Out.Functions.hex_bits_31_backwards_matches Out.Functions.hex_bits_32_backwards_matches Out.Functions.hex_bits_33_backwards_matches Out.Functions.hex_bits_34_backwards_matches Out.Functions.hex_bits_35_backwards_matches Out.Functions.hex_bits_36_backwards_matches Out.Functions.hex_bits_37_backwards_matches Out.Functions.hex_bits_38_backwards_matches Out.Functions.hex_bits_39_backwards_matches Out.Functions.hex_bits_3_backwards_matches Out.Functions.hex_bits_40_backwards_matches Out.Functions.hex_bits_41_backwards_matches Out.Functions.hex_bits_42_backwards_matches Out.Functions.hex_bits_43_backwards_matches Out.Functions.hex_bits_44_backwards_matches Out.Functions.hex_bits_45_backwards_matches Out.Functions.hex_bits_46_backwards_matches Out.Functions.hex_bits_47_backwards_matches Out.Functions.hex_bits_48_backwards_matches Out.Functions.hex_bits_49_backwards_matches Out.Functions.hex_bits_4_backwards_matches Out.Functions.hex_bits_50_backwards_matches Out.Functions.hex_bits_51_backwards_matches Out.Functions.hex_bits_52_backwards_matches Out.Functions.hex_bits_53_backwards_matches Out.Functions.hex_bits_54_backwards_matches Out.Functions.hex_bits_55_backwards_matches Out.Functions.hex_bits_56_backwards_matches Out.Functions.hex_bits_57_backwards_matches Out.Functions.hex_bits_58_backwards_matches Out.Functions.hex_bits_59_backwards_matches Out.Functions.hex_bits_5_backwards_matches Out.Functions.hex_bits_60_backwards_matches Out.Functions.hex_bits_61_backwards_matches Out.Functions.hex_bits_62_backwards_matches Out.Functions.hex_bits_63_backwards_matches Out.Functions.hex_bits_64_backwards_matches Out.Functions.hex_bits_6_backwards_matches Out.Functions.hex_bits_7_backwards_matches Out.Functions.hex_bits_8_backwards_matches Out.Functions.hex_bits_9_backwards_matches Out.Functions.hex_bits_backwards Out.Functions.hex_bits_backwards_matches Out.Functions.hex_bits_forwards Out.Functions.hex_bits_forwards_matches Out.Functions.hold_decide Out.Functions.holds_f16 Out.Functions.i8_tr_value Out.Functions.identity_map Out.Functions.idiv_ceil Out.Functions.idiv_floor Out.Functions.ilog2 Out.Functions.in_barrier_window Out.Functions.inf_bits Out.Functions.int8_dot Out.Functions.int_of Out.Functions.int_of_fp Out.Functions.inward_row Out.Functions.iq_padding Out.Functions.is_bool Out.Functions.is_dst Out.Functions.is_float Out.Functions.is_int Out.Functions.is_loopable Out.Functions.is_mxm_queue Out.Functions.is_nan Out.Functions.is_none Out.Functions.is_pair Out.Functions.is_prm Out.Functions.is_sel Out.Functions.is_shf Out.Functions.is_some Out.Functions.is_tsp Out.Functions.is_unsigned Out.Functions.isqrt Out.Functions.issued_by Out.Functions.issuer_of Out.Functions.items_len Out.Functions.k_binary Out.Functions.k_cmp Out.Functions.k_convert Out.Functions.k_convert_any Out.Functions.k_nonzero Out.Functions.k_unary Out.Functions.lane8 Out.Functions.lane_byte Out.Functions.latency_of Out.Functions.latency_of_instr Out.Functions.link_flit_mask Out.Functions.link_pow2 Out.Functions.lint Out.Functions.list_len Out.Functions.ln2_iv Out.Functions.log_iv Out.Functions.mapop_name_backwards_matches Out.Functions.mapop_name_forwards Out.Functions.mapop_name_forwards_matches Out.Functions.mem_dir_mnemonic_backwards_matches Out.Functions.mem_dir_mnemonic_forwards Out.Functions.mem_dir_mnemonic_forwards_matches Out.Functions.mem_quad Out.Functions.mem_slice_present Out.Functions.mem_unit Out.Functions.merge_tiles Out.Functions.model_link_flit_mask Out.Functions.model_link_flits Out.Functions.model_ptype_queued Out.Functions.mxm_acctype_of_num Out.Functions.mxm_bad Out.Functions.mxm_cap Out.Functions.mxm_check Out.Functions.mxm_clash Out.Functions.mxm_part_of_num Out.Functions.mxm_queue_present Out.Functions.mxm_unit Out.Functions.mxm_wtype_of_num Out.Functions.nbits_of Out.Functions.neq_bool Out.Functions.nonempty_list Out.Functions.not Out.Functions.not_bit Out.Functions.num_of_acc_mode Out.Functions.num_of_agt_field Out.Functions.num_of_agt_state Out.Functions.num_of_c2c_op Out.Functions.num_of_c2c_ptype Out.Functions.num_of_diag Out.Functions.num_of_ecc_class Out.Functions.num_of_eltype Out.Functions.num_of_fclass Out.Functions.num_of_mxm_acctype Out.Functions.num_of_mxm_part Out.Functions.num_of_mxm_wtype Out.Functions.num_of_row Out.Functions.num_of_scrub_action Out.Functions.num_of_srf_lookup Out.Functions.num_of_timing_class Out.Functions.num_of_unit_kind Out.Functions.one_plane Out.Functions.opt_spc_backwards Out.Functions.opt_spc_backwards_matches Out.Functions.opt_spc_forwards Out.Functions.opt_spc_forwards_matches Out.Functions.or_pair Out.Functions.outward_row Out.Functions.ov_nth Out.Functions.pair_exact Out.Functions.path_hold_until Out.Functions.path_views Out.Functions.place Out.Functions.place_all Out.Functions.poll_epoch Out.Functions.pow2i Out.Functions.q_in Out.Functions.queue_present Out.Functions.ras_idx Out.Functions.ras_loc Out.Functions.ras_sat_inc Out.Functions.read_gap Out.Functions.reduce_mnemonic_backwards_matches Out.Functions.reduce_mnemonic_forwards Out.Functions.reduce_mnemonic_forwards_matches Out.Functions.ref_bank Out.Functions.ref_path Out.Functions.ref_thread Out.Functions.rep_sp_set Out.Functions.repeat_after Out.Functions.repeat_plan Out.Functions.repeatable Out.Functions.rev_items Out.Functions.ring Out.Functions.rne_int Out.Functions.round_iv Out.Functions.row_of_num Out.Functions.rows_of Out.Functions.s_of_bits Out.Functions.sail_mask Out.Functions.sail_ones Out.Functions.same_target Out.Functions.sat_s Out.Functions.sat_t Out.Functions.scrub_action_of_num Out.Functions.scrub_arbitrate Out.Functions.scrub_due_count Out.Functions.scrub_slot_taken Out.Functions.sep_forwards Out.Functions.sep_forwards_matches Out.Functions.set_lane_byte Out.Functions.sg4_suffix_backwards_matches Out.Functions.sg4_suffix_forwards Out.Functions.sg4_suffix_forwards_matches Out.Functions.shift_dir_name_backwards_matches Out.Functions.shift_dir_name_forwards Out.Functions.shift_dir_name_forwards_matches Out.Functions.slice_mask Out.Functions.slot_of Out.Functions.spc_backwards Out.Functions.spc_backwards_matches Out.Functions.spc_forwards Out.Functions.spc_forwards_matches Out.Functions.sram_addr_of Out.Functions.sram_bank Out.Functions.sram_field_backwards Out.Functions.sram_field_backwards_matches Out.Functions.sram_field_forwards Out.Functions.sram_field_forwards_matches Out.Functions.sram_index Out.Functions.sram_text_backwards Out.Functions.sram_text_backwards_matches Out.Functions.sram_text_forwards Out.Functions.sram_text_forwards_matches Out.Functions.srf_lookup_of_num Out.Functions.srf_ov_spare Out.Functions.srf_rep_merge Out.Functions.srf_slot_best Out.Functions.srf_slot_chain_ov Out.Functions.srf_slot_collides Out.Functions.srf_slot_flip Out.Functions.srf_slot_read Out.Functions.srf_slot_read_ov Out.Functions.srf_slot_write Out.Functions.srf_stream_only_downstream Out.Functions.stream_id Out.Functions.stream_name_forwards_matches Out.Functions.stream_row Out.Functions.string_of_exception Out.Functions.sxm_bad Out.Functions.sxm_clash Out.Functions.sxm_lane Out.Functions.sxm_map_distinct Out.Functions.sxm_or_kind Out.Functions.sxm_set_lane Out.Functions.sxm_stream_plus Out.Functions.sxm_unit_at Out.Functions.tag_of Out.Functions.tags_clash Out.Functions.text_width Out.Functions.tile_active Out.Functions.tile_addr Out.Functions.timing_class_name Out.Functions.timing_class_of_num Out.Functions.timing_of Out.Functions.timing_overrides Out.Functions.timing_type_of Out.Functions.to_bytes_le Out.Functions.tr_eval Out.Functions.tr_eval_int Out.Functions.tr_interval Out.Functions.ty_split_backwards_matches Out.Functions.ty_split_forwards Out.Functions.ty_split_forwards_matches Out.Functions.ty_valid Out.Functions.unit_kind_of_num Out.Functions.unop_entry_backwards_matches Out.Functions.unop_entry_forwards Out.Functions.unop_entry_forwards_matches Out.Functions.unop_legal Out.Functions.vec_units Out.Functions.view_free Out.Functions.view_paths_busy Out.Functions.vsrc_name_forwards_matches Out.Functions.vxm_binop_mnemonic_backwards_matches Out.Functions.vxm_binop_mnemonic_forwards Out.Functions.vxm_binop_mnemonic_forwards_matches Out.Functions.vxm_unit Out.Functions.vxm_unop_mnemonic_backwards_matches Out.Functions.vxm_unop_mnemonic_forwards Out.Functions.vxm_unop_mnemonic_forwards_matches Out.Functions.wait_reason Out.Functions.wants_pair Out.Functions.wlane_add Out.Functions.wlane_read Out.Functions.words_to_planes Out.Functions.wrap_s Out.Functions.wrap_t Out.Functions.write_gap Out.Functions.writer_before Out.Functions.wtype_name_backwards_matches Out.Functions.wtype_name_forwards Out.Functions.wtype_name_forwards_matches Out.Functions.wtype_ok Out.Functions.wtype_split_backwards_matches Out.Functions.wtype_split_forwards Out.Functions.wtype_split_forwards_matches Out.Functions.wtype_valid Out.Functions.zero_bits Out.abs_real Out.add_real Out.diag_callback Out.div_real Out.ecc_event_callback Out.eq_real Out.gt_real Out.gteq_real Out.lint_callback Out.lt_real Out.lteq_real Out.mult_real Out.mxm_corr_callback Out.neg_real Out.prerr_real Out.print_real Out.random_real Out.real_power Out.round_down Out.round_up Out.scrub_event_callback Out.spare_event_callback Out.sqrt_real Out.sram_write_callback Out.srf_write_callback Out.sub_real Out.tdiv_int Out.tmod_int Out.to_real ecc_begin exec_cycle host_deliver host_msix execute_C2C_HOSTNOTIFY commit_at ecc_poison_mask ecc_result_ov plane_stream write_planes_fwd write_planes execute_C2C_PCIRECV pci_busy ras_record ras_ecc_event ras_spare_event get_stream srf_spare srf_s03 ecc_check_value srf_read_chain_ov srf_only_downstream srf_read_ov read_planes_full read_planes_core read_planes execute_C2C_PCISEND c2c_link c2c_peer_ok execute_C2C_RECEIVE c2c_busy c2c_put execute_C2C_SEND execute_C2C_TRANSMIT execute_CONFIG stream_fetch_arm stream_fetch_start execute_IFETCH execute_IFETCHM commit_cycle eff_addr eff_stream read_cycle read_plane_raw sram_read sram_read_ov write_sram_ecc write_sram_word sram_write_ov tile_on execute_MEM_DIRECT ecc_check_sl srf_report mem_report execute_MEM_GATHER execute_MEM_SCATTER execute_MEM_SETSEGMENT macc_busy_in macc_busy mxm_stage_acc mxm_stage plane mxm_plane_at plane_blocked ras_mxm_event abc_exec execute_MXM_ABC execute_MXM_ABC8 execute_MXM_ACC execute_MXM_IW read_planes16 execute_MXM_IWS execute_MXM_LW execute_SETVL sx sxm_load_map sxm_rd_cycle execute_SXM_LOADMAP execute_SXM_LOADMAPWIDE sxm_cm_cycle sxm_emit execute_SXM_MAP execute_SXM_ROTATE execute_SXM_SELECT execute_SXM_SHIFT execute_SXM_TRANSPOSE Out.sailThrow internal_error fp_transcendental i8_transcendental apply vxm_commit_cycle vxm_read_cycle vxm_read vxm_lanes execute_VXM_BIN execute_VXM_CMP execute_VXM_CONV execute_VXM_CVT execute_VXM_MASK execute_VXM_REDUCE execute_VXM_SELECT execute_VXM_UN execute diag_record diag_report report execute_element run_element agt_get agt_set agt_configure host_parked path_busy path_tail path_view_of polling parked_one queue_parked hold_view_of hold_check dispatch_check dispatch_cfg agt_due due_refs agt_refresh bank_report hold_path loop_clash_ref run_loop
attribute [local irreducible] Out.Functions.deskewrecv_ready Out.Functions.occupancy Out.Functions.agt_finish_config Out.Functions.reserve_local_fetch Out.Functions.repeat_issue Out.Functions.due_ref Out.Functions.loop_tag Out.Functions.dispatch_body Out.Functions.dispatch_decoded Out.Functions.dispatch_slots Out.Functions.dispatch Out.Functions.strict_held Out.Functions.parcel_held
attribute [local irreducible] Proofs.TaggedCheck.Safe Satisfies


/-- All actual receive/Repeat/AGT work between scan and consumption retains the
    same captured IQ and allocation projection, while keeping every real effect. -/
theorem prepareK_satisfies (A : Nat → Prop) (H L : Vector Nat 145)
    (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports)
    (observer : Proofs.ControlConsumption.DirectObserver A)
    (q c : Nat) (p : parcel) (due : Bool)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hu : p.units≤L[q]!)
    (finish : Bool → cycle_recs → Bool → SailM Unit) (post : ResourceFrameProofs.St → Prop)
    (old : ∀ s, Keep A H L s → post s)
    (next : ∀ go recs parked, Satisfies A H L post (finish go recs parked)) :
    Satisfies A H L post (prepareK q c p due finish) := by
  unfold prepareK
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | apply next
    | assumption
    | exact preserves_pure A _
    | apply queue_parked_safe A H L rp cp
    | apply deskewrecv_ready_safe A H L rp cp
    | apply occupancy_safe A H L rp cp
    | apply run_element_safe A H L rp cp
    | apply repeat_issue_safe A H L rp cp
    | apply agt_get_safe A H L rp cp
    | apply agt_finish_config_safe A H L rp cp
    | apply parcel_held_safe A H L rp cp
    | apply host_parked_safe A H L rp cp
    | apply polling_safe A H L rp cp
    | apply dispatch_safe A H L rp cp
    | apply earliest_safe A H L
    | apply observer_zero_safe A H L observer <;> omega
    | apply observer_head_safe A H L observer <;> assumption
    | apply head_bind_safe A
    | apply preserves_read A
    | apply preserves_other_write A <;> decide
    | apply preserves_forIn A
    | apply preserves_bind A
    | apply satisfies_head_bind A H L
    | apply satisfies_bind A H L
    | reduce_resource_beta
    | split
    | intro



def consumedTail (q c : Nat) (recs : cycle_recs) (parked : Bool) : SailM Unit := do
  Out.writeReg iq_last (vectorUpdate (← Out.readReg iq_last) q c)
  Out.writeReg iq_active (vectorUpdate (← Out.readReg iq_active) q c)
  if ((GetElem?.getElem! (← Out.readReg iq_dry) q) : Bool)
  then
    (do
      Out.writeReg iq_dry (vectorUpdate (← Out.readReg iq_dry) q false)
      (pure (lint q c LINT_IQ_UNDERFLOW)))
  else (pure ())
  dispatch q c recs parked

theorem consume_factor (q c : Nat) (p : parcel) (go : Bool) (recs : cycle_recs) (parked : Bool) :
    consume q c p go recs parked =
      (if go then
        (do observe_site "dispatch" c q (GetElem?.getElem! (← Out.readReg iq_head) q) p.units
              (← parcel_earliest q p)
            Proofs.TextCheckProduct.checkedPop q c p
            consumedTail q c recs parked)
       else dispatch q c no_recs parked) := by
  cases go <;> simp only [consume,consumedTail,Bool.false_eq_true,if_false,if_true,
    Proofs.OrderedImage.bind_if,bind_assoc,pure_bind] <;> rfl

theorem consumedTail_safe (A : Nat → Prop) (H L : Vector Nat 145)
    (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports)
    (q c : Nat) (recs : cycle_recs) (parked : Bool) :
    Safe A (H:=H) (L:=L) (consumedTail q c recs parked) := by
  unfold consumedTail
  repeat' first
    | apply dispatch_safe A H L rp cp
    | exact preserves_pure A _
    | apply preserves_read A
    | apply preserves_other_write A <;> decide
    | apply preserves_bind A
    | split
    | intro

def QueuePost (A : Nat → Prop) (H L : Vector Nat 145) (q : Nat) (after : ResourceFrameProofs.St) : Prop :=
  Keep A H L after ∨ ∃ p : parcel, p.units≤L[q]! ∧
    Keep A (popHead H q p.units) (popLength L q p.units) after

theorem observe_check_tail (A : Nat → Prop) (H L : Vector Nat 145)
    (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports)
    (observer : Proofs.ControlConsumption.DirectObserver A)
    (reads : ReadPorts iq_cap) (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame)
    (q c : Nat) (p : parcel) (recs : cycle_recs) (parked : Bool) (detail : Int)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hu : p.units≤L[q]!) :
    Satisfies A H L (QueuePost A H L q)
      (do observe_site "dispatch" c q H[q]! p.units detail
          Proofs.TextCheckProduct.checkedPop q c p
          consumedTail q c recs parked) := by
  unfold Satisfies
  intro s keep
  obtain ⟨observed,he,hr,hm⟩ := observer.frame "dispatch" c q H[q]! p.units detail s keep.2.2.1 hq (by omega)
    (Proofs.ControlConsumption.direct_dispatch_domain A H L s keep q p.units hq hh hu)
  have kept := Proofs.ControlConsumption.keep_same_parts A H L s observed keep hr hm
  have checked := checked_pop_product A q c p.units hq hh hl hu reads writes irq observed kept
  change (match (Proofs.TextCheckProduct.checkedPop q c p).run observed with
    | .ok _ after => Keep A (popHead H q p.units) (popLength L q p.units) after
    | .error _ after => Keep A H L after) at checked
  change observe_site "dispatch" c q H[q]! p.units detail s=.ok () observed at he
  try simp at he
  cases run : (Proofs.TextCheckProduct.checkedPop q c p).run observed with
  | error error after =>
    have old : Keep A H L after := by simpa [run] using checked
    change Proofs.TextCheckProduct.checkedPop q c p observed=.error error after at run
    simpa [EStateM.run,bind,EStateM.bind,he,run] using Or.inl old
  | ok u middle =>
    cases u
    have changed : Keep A (popHead H q p.units) (popLength L q p.units) middle := by simpa [run] using checked
    have tail := consumedTail_safe A (popHead H q p.units) (popLength L q p.units) rp cp q c recs parked
    unfold Proofs.TaggedCheck.Safe at tail
    have final := tail middle changed
    change Proofs.TextCheckProduct.checkedPop q c p observed=.ok () middle at run
    cases rt : (consumedTail q c recs parked).run middle with
    | ok value after =>
      have result : Keep A (popHead H q p.units) (popLength L q p.units) after := by simpa [rt] using final
      change consumedTail q c recs parked middle=.ok value after at rt
      simpa [EStateM.run,bind,EStateM.bind,he,run,rt] using Or.inr ⟨p,hu,result⟩
    | error error after =>
      have result : Keep A (popHead H q p.units) (popLength L q p.units) after := by simpa [rt] using final
      change consumedTail q c recs parked middle=.error error after at rt
      simpa [EStateM.run,bind,EStateM.bind,he,run,rt] using Or.inr ⟨p,hu,result⟩



theorem consume_satisfies (A : Nat → Prop) (H L : Vector Nat 145)
    (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports)
    (observer : Proofs.ControlConsumption.DirectObserver A)
    (reads : ReadPorts iq_cap) (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame)
    (q c : Nat) (p : parcel) (go : Bool) (recs : cycle_recs) (parked : Bool)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (hu : p.units≤L[q]!) :
    Satisfies A H L (QueuePost A H L q) (consume q c p go recs parked) := by
  rw [consume_factor]
  cases go with
  | false =>
    simp only [Bool.false_eq_true,if_false]
    exact satisfies_safe A H L _ _ (dispatch_safe A H L rp cp q c no_recs parked)
      (fun _ kept => Or.inl kept)
  | true =>
    simp only [if_true]
    apply satisfies_head_bind A H L
    apply satisfies_bind A H L
    · exact earliest_safe A H L q p
    · intro detail
      exact observe_check_tail A H L rp cp observer reads writes irq q c p recs parked detail hq hh hl hu
    · exact fun _ kept => Or.inl kept

theorem scanHead_spec (A : Nat → Prop) (H L : Vector Nat 145)
    (s : ResourceFrameProofs.St) (keep : Keep A H L s) (q c : Nat)
    (hq : q<145) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) (reads : ReadPorts iq_cap) :
    match (scanHead q c).run s with
    | .ok pair after => after=s ∧ pair.2.units≤L[q]!
    | .error _ after => after=s := by
  have read : Out.readReg iq_len s=.ok L s := by
    simp [Out.readReg,PreSail.readReg,keep.1.2.2,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  have parsed := Proofs.ParcelBound.scan_bound s q reads H L keep.1.2.1 keep.1.2.2 keep.1.1 hq hh hl
  by_cases positive : 0<L[q]!
  · cases scan : (parcel_scan q).run s with
    | error error after =>
      have eq : after=s := by simpa [Proofs.ParcelBound.SpecAt,scan] using parsed
      change parcel_scan q s=.error error after at scan
      simpa [scanHead,EStateM.run,bind,EStateM.bind,read,positive,scan] using eq
    | ok p after =>
      have good : after=s ∧ p.units≤L[q]! := by simpa [Proofs.ParcelBound.SpecAt,scan] using parsed
      obtain ⟨eq,bound⟩ := good
      subst after
      change parcel_scan q s=.ok p s at scan
      have frame := Proofs.ControlConsumption.earliest_frame s q p
      cases early : (parcel_earliest q p).run s with
      | error error after =>
        have eq : after=s := by simpa [FrameAt,early] using frame
        change parcel_earliest q p s=.error error after at early
        simpa [scanHead,EStateM.run,bind,EStateM.bind,read,positive,scan,early] using eq
      | ok detail after =>
        have eq : after=s := by simpa [FrameAt,early] using frame
        subst after
        change parcel_earliest q p s=.ok detail s at early
        simpa [scanHead,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,read,positive,scan,early] using (show s=s ∧ p.units≤L[q]! from ⟨rfl,bound⟩)
  · simp [scanHead,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,read,positive]

/-- The actual entire data-queue transition composes its SAME scan with every
    intervening receive, Repeat, AGT and grant/execute operation, then bounded
    text-repair/pop and the real dispatch tail. No effectful callee is a logger. -/
theorem actual_tick_queue_consumption (A : Nat → Prop) (H L : Vector Nat 145)
    (rp : ResourceExecuteFrameProofs.Ports) (cp : ResourceCompleteProofs.Ports)
    (observer : Proofs.ControlConsumption.DirectObserver A)
    (reads : ReadPorts iq_cap) (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame)
    (s : ResourceFrameProofs.St) (keep : Keep A H L s) (q c : Nat)
    (hq : q<144) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) :
    match (tick_queue q c).run s with
    | .ok _ after => QueuePost A H L q after
    | .error _ after => QueuePost A H L q after := by
  have parsed := scanHead_spec A H L s keep q c (by omega) hh hl reads
  rw [actual_tick_queue_factor]
  cases scan : (scanHead q c).run s with
  | error error after =>
    have eq : after=s := by simpa [scan] using parsed
    subst after
    change scanHead q c s=.error error s at scan
    simpa [EStateM.run,bind,EStateM.bind,scan] using Or.inl keep
  | ok pair middle =>
    rcases pair with ⟨due,p⟩
    have good : middle=s ∧ p.units≤L[q]! := by simpa [scan] using parsed
    obtain ⟨eq,bound⟩ := good
    subst middle
    have prepare := prepareK_satisfies A H L rp cp observer q c p due (by omega) hh hl bound
      (consume q c p) (QueuePost A H L q) (fun _ kept => Or.inl kept)
      (fun go recs parked => consume_satisfies A H L rp cp observer reads writes irq q c p go recs parked (by omega) hh hl bound)
    unfold Satisfies at prepare
    have final := prepare s keep
    change scanHead q c s=.ok (due,p) s at scan
    simp only [EStateM.run,bind,EStateM.bind,scan]
    cases executed : prepareK q c p due (consume q c p) s <;> simpa [EStateM.run,executed] using final

end Proofs.QueueConsumption
