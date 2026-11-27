-- Native whole-function resource register frames, actual bodies and primitive-only contracts.
import Proofs.PatrolValidity
import Proofs.ResourcePairCases
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 100000
namespace ResourceExecuteFrameProofs
open ResourceFrameProofs ResourceInvariantProofs
def IQMem (s : ResourceFrameProofs.St) := fun key : Nat => if key%4<2 then s.mem.get? key else none
def RView (s : ResourceFrameProofs.St) := (PatrolValidityProofs.View s,(s.regs.get? Register.iq_head,s.regs.get? Register.iq_len,IQMem s))
attribute [local irreducible] ResourceFrameProofs.Keeps
structure Ports : Prop where
  c2c_packet_busy : ∀ (a0 : Nat) (a1 : Nat), Keeps RView (c2c_packet_busy a0 a1)
  c2c_peer : ∀ (a0 : Nat), Keeps RView (c2c_peer a0)
  c2c_recv : ∀ (a0 : Nat) (a1 : Nat), Keeps RView (c2c_recv a0 a1)
  c2c_recv_bad : ∀ (a0 : Nat), Keeps RView (c2c_recv_bad a0)
  c2c_recv_poison : ∀ (a0 : Nat) (a1 : Nat), Keeps RView (c2c_recv_poison a0 a1)
  c2c_recv_ready : ∀ (a0 : Nat) (a1 : Nat), Keeps RView (c2c_recv_ready a0 a1)
  c2c_send : ∀ (a0 : Nat) (a1 : Nat) (a2 : Nat) (a3 : vec) (a4 : Nat), Keeps RView (c2c_send a0 a1 a2 a3 a4)
  host_irq_callback : ∀ (a0 : Nat) (a1 : Nat), Keeps RView (host_irq_callback a0 a1)
  host_recv : ∀ (a0 : Nat), Keeps RView (host_recv a0)
  host_recv_ready : ∀ (a0 : Nat), Keeps RView (host_recv_ready a0)
  host_send : ∀ (a0 : Nat) (a1 : vec), Keeps RView (host_send a0 a1)
  observe_site : ∀ (a0 : String) (a1 : Nat) (a2 : Nat) (a3 : Nat) (a4 : Nat) (a5 : Int), Keeps RView (observe_site a0 a1 a2 a3 a4 a5)
  raw_write_sram_ecc : ∀ (a0 : mem_slice) (a1 : Nat) (a2 : ecc_ov), Keeps RView (raw_write_sram_ecc a0 a1 a2)
  raw_write_sram_word : ∀ (a0 : mem_slice) (a1 : Nat) (a2 : vec), Keeps RView (raw_write_sram_word a0 a1 a2)
  read_sram_ecc : ∀ (a0 : mem_slice) (a1 : Nat), Keeps RView (read_sram_ecc a0 a1)
  read_sram_word : ∀ (a0 : mem_slice) (a1 : Nat), Keeps RView (read_sram_word a0 a1)
theorem throw_frame {α : Type} (e : Error exception) : Keeps RView (throw e : SailM α) := by
  unfold ResourceFrameProofs.Keeps
  intro state
  rfl
theorem write_frame (r : Register) (v : RegisterType r) (hr : r≠Register.scrub_ptr) (hp : r≠Register.scrub_pending) (hh : r≠Register.iq_head) (hl : r≠Register.iq_len) : Keeps RView (Out.writeReg r v) := by
  apply pair_keeps
  · exact pair_keeps _ _ _ (unrelated_register_write _ _ _ (Ne.symm hr)) (unrelated_register_write _ _ _ (Ne.symm hp))
  · apply pair_keeps
    · exact unrelated_register_write _ _ _ (Ne.symm hh)
    · apply pair_keeps
      · exact unrelated_register_write _ _ _ (Ne.symm hl)
      · apply write_keeps; intro state; rfl
attribute [local irreducible] GetElem?.getElem! Inhabited.default Bind.bind Pure.pure EStateM.bind EStateM.pure PreSail.readReg PreSail.writeReg IntRange.forIn'
attribute [local irreducible] Out.Functions.__id Out.Functions.abc_after Out.Functions.abc_at Out.Functions.abc_fp8 Out.Functions.abc_others Out.Functions.abc_others_fp8 Out.Functions.abc_pair_pick Out.Functions.abc_prune Out.Functions.abc_types_ok Out.Functions.acc_mode_name_backwards_matches Out.Functions.acc_mode_name_forwards Out.Functions.acc_mode_name_forwards_matches Out.Functions.acc_mode_of_num Out.Functions.acctype_declared Out.Functions.acctype_name_backwards_matches Out.Functions.acctype_name_forwards Out.Functions.acctype_name_forwards_matches Out.Functions.addr_bank Out.Functions.agt_advance Out.Functions.agt_advance_test Out.Functions.agt_after_reference Out.Functions.agt_config Out.Functions.agt_configure_pure Out.Functions.agt_due_pure Out.Functions.agt_field_of_num Out.Functions.agt_field_pos Out.Functions.agt_finish_config_pure Out.Functions.agt_nlevels Out.Functions.agt_opened Out.Functions.agt_refresh_pure Out.Functions.agt_state_of_num Out.Functions.aligned Out.Functions.all_ones Out.Functions.append_element Out.Functions.assembly_forwards_matches Out.Functions.bank_rule Out.Functions.banks_meet Out.Functions.bf16_dot Out.Functions.bfp8_exact Out.Functions.binop_entry_backwards_matches Out.Functions.binop_entry_forwards Out.Functions.binop_entry_forwards_matches Out.Functions.binop_legal Out.Functions.bit_index Out.Functions.bit_of Out.Functions.bits_of_s Out.Functions.bits_t Out.Functions.blk_exp Out.Functions.bool_bits Out.Functions.c2c_op_of_num Out.Functions.c2c_packet_flits Out.Functions.c2c_poison_int Out.Functions.c2c_ptype_code Out.Functions.c2c_ptype_of_num Out.Functions.c2c_ptype_queued Out.Functions.c2c_rx Out.Functions.c2c_tx Out.Functions.canon_nan Out.Functions.cfg_path_pick Out.Functions.chain_suffix_backwards_matches Out.Functions.chain_suffix_forwards Out.Functions.chain_suffix_forwards_matches Out.Functions.chk_elem Out.Functions.chk_wrow Out.Functions.clash_addr Out.Functions.clash_stream Out.Functions.classify_items Out.Functions.clears_repeat Out.Functions.cmp_legal Out.Functions.cmp_mnemonic_backwards_matches Out.Functions.cmp_mnemonic_forwards Out.Functions.cmp_mnemonic_forwards_matches Out.Functions.commit_scan Out.Functions.commit_wins Out.Functions.config_ends Out.Functions.config_for Out.Functions.conv_dst Out.Functions.conv_entry_backwards_matches Out.Functions.conv_entry_forwards Out.Functions.conv_entry_forwards_matches Out.Functions.conv_mnemonic_backwards_matches Out.Functions.conv_mnemonic_forwards Out.Functions.conv_mnemonic_forwards_matches Out.Functions.conv_src Out.Functions.conv_src_type Out.Functions.conv_types Out.Functions.csr_value Out.Functions.db_idx Out.Functions.dec_bits_10_backwards Out.Functions.dec_bits_10_backwards_matches Out.Functions.dec_bits_1_backwards Out.Functions.dec_bits_1_backwards_matches Out.Functions.dec_bits_2_backwards Out.Functions.dec_bits_2_backwards_matches Out.Functions.dec_bits_32_backwards Out.Functions.dec_bits_32_backwards_matches Out.Functions.dec_bits_3_backwards Out.Functions.dec_bits_3_backwards_matches Out.Functions.dec_bits_4_backwards Out.Functions.dec_bits_4_backwards_matches Out.Functions.dec_bits_5_backwards Out.Functions.dec_bits_5_backwards_matches Out.Functions.dec_bits_6_backwards Out.Functions.dec_bits_6_backwards_matches Out.Functions.dec_bits_7_backwards Out.Functions.dec_bits_7_backwards_matches Out.Functions.dec_bits_8_backwards Out.Functions.dec_bits_8_backwards_matches Out.Functions.dec_bits_9_backwards Out.Functions.dec_bits_9_backwards_matches Out.Functions.dec_bits_backwards Out.Functions.dec_bits_backwards_matches Out.Functions.dec_bits_forwards Out.Functions.dec_bits_forwards_matches Out.Functions.def_spc_backwards Out.Functions.def_spc_backwards_matches Out.Functions.def_spc_forwards Out.Functions.def_spc_forwards_matches Out.Functions.diag_code_backwards_matches Out.Functions.diag_code_forwards Out.Functions.diag_code_forwards_matches Out.Functions.diag_idx Out.Functions.diag_is_data Out.Functions.diag_number Out.Functions.diag_of_num Out.Functions.drop_commits Out.Functions.dst_split_backwards_matches Out.Functions.dst_split_forwards Out.Functions.dst_split_forwards_matches Out.Functions.dsum_round Out.Functions.dsum_round_u Out.Functions.dsum_wide Out.Functions.earliest Out.Functions.earliest_int Out.Functions.ecc_class_of_num Out.Functions.ecc_classify Out.Functions.ecc_correct Out.Functions.ecc_data_index Out.Functions.ecc_encode Out.Functions.ecc_flip_delta Out.Functions.ecc_is_pow2 Out.Functions.ecc_log2 Out.Functions.ecc_ov_get Out.Functions.ecc_ov_set Out.Functions.ecc_par10 Out.Functions.ecc_poison_ov Out.Functions.ecc_poison_sl Out.Functions.ecc_pos Out.Functions.elem Out.Functions.elem_at Out.Functions.eltype_name_backwards_matches Out.Functions.eltype_name_forwards Out.Functions.eltype_name_forwards_matches Out.Functions.eltype_of_num Out.Functions.eltype_width Out.Functions.empty_plane Out.Functions.encdec_acc_mode_backwards Out.Functions.encdec_acc_mode_backwards_matches Out.Functions.encdec_acc_mode_forwards Out.Functions.encdec_acc_mode_forwards_matches Out.Functions.encdec_acctype_backwards_matches Out.Functions.encdec_acctype_forwards Out.Functions.encdec_acctype_forwards_matches Out.Functions.encdec_c2c_op_backwards_matches Out.Functions.encdec_c2c_op_forwards Out.Functions.encdec_c2c_op_forwards_matches Out.Functions.encdec_cmp_mode_backwards_matches Out.Functions.encdec_cmp_mode_forwards Out.Functions.encdec_cmp_mode_forwards_matches Out.Functions.encdec_eltype_backwards_matches Out.Functions.encdec_eltype_forwards Out.Functions.encdec_eltype_forwards_matches Out.Functions.encdec_mapop_backwards_matches Out.Functions.encdec_mapop_forwards Out.Functions.encdec_mapop_forwards_matches Out.Functions.encdec_mem_dir_backwards_matches Out.Functions.encdec_mem_dir_forwards Out.Functions.encdec_mem_dir_forwards_matches Out.Functions.encdec_reduce_op_backwards_matches Out.Functions.encdec_reduce_op_forwards Out.Functions.encdec_reduce_op_forwards_matches Out.Functions.encdec_shift_dir_backwards Out.Functions.encdec_shift_dir_backwards_matches Out.Functions.encdec_shift_dir_forwards Out.Functions.encdec_shift_dir_forwards_matches Out.Functions.encdec_vxm_binop_backwards_matches Out.Functions.encdec_vxm_binop_forwards Out.Functions.encdec_vxm_binop_forwards_matches Out.Functions.encdec_vxm_conv_backwards_matches Out.Functions.encdec_vxm_conv_forwards Out.Functions.encdec_vxm_conv_forwards_matches Out.Functions.encdec_vxm_unop_backwards_matches Out.Functions.encdec_vxm_unop_forwards Out.Functions.encdec_vxm_unop_forwards_matches Out.Functions.entry_latency Out.Functions.entry_latency_typed Out.Functions.entry_of Out.Functions.execute_C2C_DESKEW Out.Functions.execute_C2C_DESKEWRECV Out.Functions.execute_C2C_HOSTWAIT Out.Functions.execute_C2C_RTDESKEW Out.Functions.execute_ILLEGAL32 Out.Functions.execute_ILLEGAL64 Out.Functions.execute_MEM_COUNTDOWN Out.Functions.execute_MEM_DEEPSLEEP Out.Functions.execute_MEM_ITERATIONS Out.Functions.execute_MEM_POWERCONFIG Out.Functions.execute_MEM_STEP Out.Functions.execute_MEM_STREAMSTEP Out.Functions.execute_NOP Out.Functions.execute_NOTIFY Out.Functions.execute_REPEAT Out.Functions.execute_SYNC Out.Functions.exp_iv Out.Functions.exp_small Out.Functions.extensions Out.Functions.f16_dot Out.Functions.f16_exact Out.Functions.f16_row Out.Functions.f16_row_bad Out.Functions.f16_stream_values Out.Functions.fadd_core Out.Functions.fclass_eq Out.Functions.fclass_of_num Out.Functions.fcmp Out.Functions.fdecode Out.Functions.fetch_profile_causal Out.Functions.fetch_tile_available Out.Functions.fetchm_bank_note Out.Functions.fetchm_hold_until Out.Functions.field Out.Functions.fill_append Out.Functions.fill_len Out.Functions.fill_push_unit Out.Functions.fill_units_append Out.Functions.find_override Out.Functions.flip96 Out.Functions.fmt_of Out.Functions.fp8_dot Out.Functions.fp8_exact Out.Functions.fp8_format_name_backwards_matches Out.Functions.fp8_format_name_forwards Out.Functions.fp8_format_name_forwards_matches Out.Functions.fp_abs Out.Functions.fp_add Out.Functions.fp_convert Out.Functions.fp_max Out.Functions.fp_min Out.Functions.fp_mul Out.Functions.fp_neg Out.Functions.fp_of_int Out.Functions.fp_sub Out.Functions.fp_tr_value Out.Functions.from_bytes_le Out.Functions.fround Out.Functions.fval_int Out.Functions.fx_of Out.Functions.gap_or_none Out.Functions.has_extension Out.Functions.hex_bits_10_backwards_matches Out.Functions.hex_bits_11_backwards_matches Out.Functions.hex_bits_12_backwards_matches Out.Functions.hex_bits_13_backwards_matches Out.Functions.hex_bits_14_backwards_matches Out.Functions.hex_bits_15_backwards_matches Out.Functions.hex_bits_16_backwards_matches Out.Functions.hex_bits_17_backwards_matches Out.Functions.hex_bits_18_backwards_matches Out.Functions.hex_bits_19_backwards_matches Out.Functions.hex_bits_1_backwards_matches Out.Functions.hex_bits_20_backwards_matches Out.Functions.hex_bits_21_backwards_matches Out.Functions.hex_bits_22_backwards_matches Out.Functions.hex_bits_23_backwards_matches Out.Functions.hex_bits_24_backwards_matches Out.Functions.hex_bits_25_backwards_matches Out.Functions.hex_bits_26_backwards_matches Out.Functions.hex_bits_27_backwards_matches Out.Functions.hex_bits_28_backwards_matches Out.Functions.hex_bits_29_backwards_matches Out.Functions.hex_bits_2_backwards_matches Out.Functions.hex_bits_30_backwards_matches Out.Functions.hex_bits_31_backwards_matches Out.Functions.hex_bits_32_backwards_matches Out.Functions.hex_bits_33_backwards_matches Out.Functions.hex_bits_34_backwards_matches Out.Functions.hex_bits_35_backwards_matches Out.Functions.hex_bits_36_backwards_matches Out.Functions.hex_bits_37_backwards_matches Out.Functions.hex_bits_38_backwards_matches Out.Functions.hex_bits_39_backwards_matches Out.Functions.hex_bits_3_backwards_matches Out.Functions.hex_bits_40_backwards_matches Out.Functions.hex_bits_41_backwards_matches Out.Functions.hex_bits_42_backwards_matches Out.Functions.hex_bits_43_backwards_matches Out.Functions.hex_bits_44_backwards_matches Out.Functions.hex_bits_45_backwards_matches Out.Functions.hex_bits_46_backwards_matches Out.Functions.hex_bits_47_backwards_matches Out.Functions.hex_bits_48_backwards_matches Out.Functions.hex_bits_49_backwards_matches Out.Functions.hex_bits_4_backwards_matches Out.Functions.hex_bits_50_backwards_matches Out.Functions.hex_bits_51_backwards_matches Out.Functions.hex_bits_52_backwards_matches Out.Functions.hex_bits_53_backwards_matches Out.Functions.hex_bits_54_backwards_matches Out.Functions.hex_bits_55_backwards_matches Out.Functions.hex_bits_56_backwards_matches Out.Functions.hex_bits_57_backwards_matches Out.Functions.hex_bits_58_backwards_matches Out.Functions.hex_bits_59_backwards_matches Out.Functions.hex_bits_5_backwards_matches Out.Functions.hex_bits_60_backwards_matches Out.Functions.hex_bits_61_backwards_matches Out.Functions.hex_bits_62_backwards_matches Out.Functions.hex_bits_63_backwards_matches Out.Functions.hex_bits_64_backwards_matches Out.Functions.hex_bits_6_backwards_matches Out.Functions.hex_bits_7_backwards_matches Out.Functions.hex_bits_8_backwards_matches Out.Functions.hex_bits_9_backwards_matches Out.Functions.hex_bits_backwards Out.Functions.hex_bits_backwards_matches Out.Functions.hex_bits_forwards Out.Functions.hex_bits_forwards_matches Out.Functions.hold_decide Out.Functions.holds_f16 Out.Functions.i8_tr_value Out.Functions.identity_map Out.Functions.idiv_ceil Out.Functions.idiv_floor Out.Functions.ilog2 Out.Functions.in_barrier_window Out.Functions.inf_bits Out.Functions.int8_dot Out.Functions.int_of Out.Functions.int_of_fp Out.Functions.inward_row Out.Functions.iq_padding Out.Functions.is_bool Out.Functions.is_dst Out.Functions.is_float Out.Functions.is_int Out.Functions.is_loopable Out.Functions.is_mxm_queue Out.Functions.is_nan Out.Functions.is_none Out.Functions.is_pair Out.Functions.is_prm Out.Functions.is_sel Out.Functions.is_shf Out.Functions.is_some Out.Functions.is_tsp Out.Functions.is_unsigned Out.Functions.isqrt Out.Functions.issued_by Out.Functions.issuer_of Out.Functions.items_len Out.Functions.k_binary Out.Functions.k_cmp Out.Functions.k_convert Out.Functions.k_convert_any Out.Functions.k_nonzero Out.Functions.k_unary Out.Functions.lane8 Out.Functions.lane_byte Out.Functions.latency_of Out.Functions.latency_of_instr Out.Functions.link_flit_mask Out.Functions.link_pow2 Out.Functions.lint Out.Functions.list_len Out.Functions.ln2_iv Out.Functions.log_iv Out.Functions.mapop_name_backwards_matches Out.Functions.mapop_name_forwards Out.Functions.mapop_name_forwards_matches Out.Functions.mem_dir_mnemonic_backwards_matches Out.Functions.mem_dir_mnemonic_forwards Out.Functions.mem_dir_mnemonic_forwards_matches Out.Functions.mem_quad Out.Functions.mem_slice_present Out.Functions.mem_unit Out.Functions.merge_tiles Out.Functions.model_link_flit_mask Out.Functions.model_link_flits Out.Functions.model_ptype_queued Out.Functions.mxm_acctype_of_num Out.Functions.mxm_bad Out.Functions.mxm_cap Out.Functions.mxm_check Out.Functions.mxm_clash Out.Functions.mxm_part_of_num Out.Functions.mxm_queue_present Out.Functions.mxm_unit Out.Functions.mxm_wtype_of_num Out.Functions.nbits_of Out.Functions.neq_bool Out.Functions.nonempty_list Out.Functions.not Out.Functions.not_bit Out.Functions.num_of_acc_mode Out.Functions.num_of_agt_field Out.Functions.num_of_agt_state Out.Functions.num_of_c2c_op Out.Functions.num_of_c2c_ptype Out.Functions.num_of_diag Out.Functions.num_of_ecc_class Out.Functions.num_of_eltype Out.Functions.num_of_fclass Out.Functions.num_of_mxm_acctype Out.Functions.num_of_mxm_part Out.Functions.num_of_mxm_wtype Out.Functions.num_of_row Out.Functions.num_of_scrub_action Out.Functions.num_of_srf_lookup Out.Functions.num_of_timing_class Out.Functions.num_of_unit_kind Out.Functions.one_plane Out.Functions.opt_spc_backwards Out.Functions.opt_spc_backwards_matches Out.Functions.opt_spc_forwards Out.Functions.opt_spc_forwards_matches Out.Functions.or_pair Out.Functions.outward_row Out.Functions.ov_nth Out.Functions.pair_exact Out.Functions.path_hold_until Out.Functions.path_views Out.Functions.place Out.Functions.place_all Out.Functions.poll_epoch Out.Functions.pow2i Out.Functions.q_in Out.Functions.queue_present Out.Functions.ras_idx Out.Functions.ras_loc Out.Functions.ras_sat_inc Out.Functions.read_gap Out.Functions.reduce_mnemonic_backwards_matches Out.Functions.reduce_mnemonic_forwards Out.Functions.reduce_mnemonic_forwards_matches Out.Functions.ref_bank Out.Functions.ref_path Out.Functions.ref_thread Out.Functions.rep_sp_set Out.Functions.repeat_after Out.Functions.repeat_plan Out.Functions.repeatable Out.Functions.rev_items Out.Functions.ring Out.Functions.rne_int Out.Functions.round_iv Out.Functions.row_of_num Out.Functions.rows_of Out.Functions.s_of_bits Out.Functions.sail_mask Out.Functions.sail_ones Out.Functions.same_target Out.Functions.sat_s Out.Functions.sat_t Out.Functions.scrub_action_of_num Out.Functions.scrub_arbitrate Out.Functions.scrub_due_count Out.Functions.scrub_slot_taken Out.Functions.sep_forwards Out.Functions.sep_forwards_matches Out.Functions.set_lane_byte Out.Functions.sg4_suffix_backwards_matches Out.Functions.sg4_suffix_forwards Out.Functions.sg4_suffix_forwards_matches Out.Functions.shift_dir_name_backwards_matches Out.Functions.shift_dir_name_forwards Out.Functions.shift_dir_name_forwards_matches Out.Functions.slice_mask Out.Functions.slot_of Out.Functions.spc_backwards Out.Functions.spc_backwards_matches Out.Functions.spc_forwards Out.Functions.spc_forwards_matches Out.Functions.sram_addr_of Out.Functions.sram_bank Out.Functions.sram_field_backwards Out.Functions.sram_field_backwards_matches Out.Functions.sram_field_forwards Out.Functions.sram_field_forwards_matches Out.Functions.sram_index Out.Functions.sram_text_backwards Out.Functions.sram_text_backwards_matches Out.Functions.sram_text_forwards Out.Functions.sram_text_forwards_matches Out.Functions.srf_lookup_of_num Out.Functions.srf_ov_spare Out.Functions.srf_rep_merge Out.Functions.srf_slot_best Out.Functions.srf_slot_chain_ov Out.Functions.srf_slot_collides Out.Functions.srf_slot_flip Out.Functions.srf_slot_read Out.Functions.srf_slot_read_ov Out.Functions.srf_slot_write Out.Functions.srf_stream_only_downstream Out.Functions.stream_id Out.Functions.stream_name_forwards_matches Out.Functions.stream_row Out.Functions.string_of_exception Out.Functions.sxm_bad Out.Functions.sxm_clash Out.Functions.sxm_lane Out.Functions.sxm_map_distinct Out.Functions.sxm_or_kind Out.Functions.sxm_set_lane Out.Functions.sxm_stream_plus Out.Functions.sxm_unit_at Out.Functions.tag_of Out.Functions.tags_clash Out.Functions.text_width Out.Functions.tile_active Out.Functions.tile_addr Out.Functions.timing_class_name Out.Functions.timing_class_of_num Out.Functions.timing_of Out.Functions.timing_overrides Out.Functions.timing_type_of Out.Functions.to_bytes_le Out.Functions.tr_eval Out.Functions.tr_eval_int Out.Functions.tr_interval Out.Functions.ty_split_backwards_matches Out.Functions.ty_split_forwards Out.Functions.ty_split_forwards_matches Out.Functions.ty_valid Out.Functions.unit_kind_of_num Out.Functions.unop_entry_backwards_matches Out.Functions.unop_entry_forwards Out.Functions.unop_entry_forwards_matches Out.Functions.unop_legal Out.Functions.vec_units Out.Functions.view_free Out.Functions.view_paths_busy Out.Functions.vsrc_name_forwards_matches Out.Functions.vxm_binop_mnemonic_backwards_matches Out.Functions.vxm_binop_mnemonic_forwards Out.Functions.vxm_binop_mnemonic_forwards_matches Out.Functions.vxm_unit Out.Functions.vxm_unop_mnemonic_backwards_matches Out.Functions.vxm_unop_mnemonic_forwards Out.Functions.vxm_unop_mnemonic_forwards_matches Out.Functions.wait_reason Out.Functions.wants_pair Out.Functions.wlane_add Out.Functions.wlane_read Out.Functions.words_to_planes Out.Functions.wrap_s Out.Functions.wrap_t Out.Functions.write_gap Out.Functions.writer_before Out.Functions.wtype_name_backwards_matches Out.Functions.wtype_name_forwards Out.Functions.wtype_name_forwards_matches Out.Functions.wtype_ok Out.Functions.wtype_split_backwards_matches Out.Functions.wtype_split_forwards Out.Functions.wtype_split_forwards_matches Out.Functions.wtype_valid Out.Functions.zero_bits Out.abs_real Out.add_real Out.diag_callback Out.div_real Out.ecc_event_callback Out.eq_real Out.gt_real Out.gteq_real Out.lint_callback Out.lt_real Out.lteq_real Out.mult_real Out.mxm_corr_callback Out.neg_real Out.prerr_real Out.print_real Out.random_real Out.real_power Out.round_down Out.round_up Out.scrub_event_callback Out.spare_event_callback Out.sqrt_real Out.sram_write_callback Out.srf_write_callback Out.sub_real Out.tdiv_int Out.tmod_int Out.to_real ecc_begin exec_cycle host_deliver host_msix execute_C2C_HOSTNOTIFY commit_at ecc_poison_mask ecc_result_ov plane_stream write_planes_fwd write_planes execute_C2C_PCIRECV pci_busy ras_record ras_ecc_event ras_spare_event get_stream srf_spare srf_s03 ecc_check_value srf_read_chain_ov srf_only_downstream srf_read_ov read_planes_full read_planes_core read_planes execute_C2C_PCISEND c2c_link c2c_peer_ok execute_C2C_RECEIVE c2c_busy c2c_put execute_C2C_SEND execute_C2C_TRANSMIT execute_CONFIG stream_fetch_arm stream_fetch_start execute_IFETCH execute_IFETCHM commit_cycle eff_addr eff_stream read_cycle read_plane_raw sram_read sram_read_ov write_sram_ecc write_sram_word sram_write_ov tile_on execute_MEM_DIRECT ecc_check_sl srf_report mem_report execute_MEM_GATHER execute_MEM_SCATTER execute_MEM_SETSEGMENT macc_busy_in macc_busy mxm_stage_acc mxm_stage plane mxm_plane_at plane_blocked ras_mxm_event abc_exec execute_MXM_ABC execute_MXM_ABC8 execute_MXM_ACC execute_MXM_IW read_planes16 execute_MXM_IWS execute_MXM_LW execute_SETVL sx sxm_load_map sxm_rd_cycle execute_SXM_LOADMAP execute_SXM_LOADMAPWIDE sxm_cm_cycle sxm_emit execute_SXM_MAP execute_SXM_ROTATE execute_SXM_SELECT execute_SXM_SHIFT execute_SXM_TRANSPOSE sailThrow internal_error fp_transcendental i8_transcendental apply vxm_commit_cycle vxm_read_cycle vxm_read vxm_lanes execute_VXM_BIN execute_VXM_CMP execute_VXM_CONV execute_VXM_CVT execute_VXM_MASK execute_VXM_REDUCE execute_VXM_SELECT execute_VXM_UN execute diag_record diag_report report execute_element run_element agt_get agt_set agt_configure host_parked path_busy path_tail path_view_of polling parked_one queue_parked hold_view_of hold_check dispatch_check dispatch_cfg agt_due due_refs agt_refresh bank_report hold_path loop_clash_ref run_loop
theorem ecc_begin_frame (ports : Ports) (unitArg : Unit) : Keeps RView (ecc_begin unitArg : SailM Unit) := by
  unfold ecc_begin
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem exec_cycle_frame (ports : Ports) (unitArg : Unit) : Keeps RView (exec_cycle unitArg : SailM Nat) := by
  unfold exec_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem host_deliver_frame (ports : Ports) (c : Nat) (n : Nat) : Keeps RView (host_deliver c n : SailM Unit) := by
  unfold host_deliver
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ports.host_irq_callback _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem host_msix_frame (ports : Ports) (c : Nat) (n : Nat) : Keeps RView (host_msix c n : SailM Unit) := by
  unfold host_msix
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact host_deliver_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_HOSTNOTIFY_frame (ports : Ports) (n : (BitVec 4)) : Keeps RView (execute_C2C_HOSTNOTIFY n : SailM ExecutionResult) := by
  unfold execute_C2C_HOSTNOTIFY
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact host_msix_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem commit_at_frame (ports : Ports) (unitArg : Unit) : Keeps RView (commit_at unitArg : SailM Nat) := by
  unfold commit_at
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ecc_poison_mask_frame (ports : Ports) (unitArg : Unit) : Keeps RView (ecc_poison_mask unitArg : SailM (BitVec Out.Functions.superlanes)) := by
  unfold ecc_poison_mask
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ecc_result_ov_frame (ports : Ports) (unitArg : Unit) : Keeps RView (ecc_result_ov unitArg : SailM ecc_ov) := by
  unfold ecc_result_ov
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ecc_poison_mask_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem plane_stream_frame (ports : Ports) (s : streamref) (b : Int) : Keeps RView (plane_stream s b : SailM Nat) := by
  unfold plane_stream
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem write_planes_fwd_frame (ports : Ports) (s : streamref) (p : (Vector vec 4)) (w : Nat) (c : Nat) (fwd : (Option ecc_ov)) : Keeps RView (write_planes_fwd s p w c fwd : SailM (Result Unit diag)) := by
  unfold write_planes_fwd
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ecc_result_ov_frame ports _
    | exact plane_stream_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem write_planes_frame (ports : Ports) (s : streamref) (p : (Vector vec 4)) (w : Nat) (c : Nat) : Keeps RView (write_planes s p w c : SailM (Result Unit diag)) := by
  unfold write_planes
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact write_planes_fwd_frame ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_PCIRECV_frame (ports : Ports) (s : streamref) : Keeps RView (execute_C2C_PCIRECV s : SailM ExecutionResult) := by
  unfold execute_C2C_PCIRECV
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | exact exec_cycle_frame ports _
    | exact write_planes_frame ports _ _ _ _
    | exact ports.host_recv _
    | exact ports.host_recv_ready _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem pci_busy_frame (ports : Ports) (c : Nat) : Keeps RView (pci_busy c : SailM Bool) := by
  unfold pci_busy
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ras_record_frame (ports : Ports) (c : Nat) (q : Int) (kind : Nat) (loc : (BitVec 43)) (sl : Nat) (pos : Nat) : Keeps RView (ras_record c q kind loc sl pos : SailM Unit) := by
  unfold ras_record
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact host_msix_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ras_ecc_event_frame (ports : Ports) (c : Nat) (q : Nat) (kind : Nat) (unc : Bool) (isw : Bool) (id : Nat) (addr : Nat) (sl : Nat) (pos : Nat) : Keeps RView (ras_ecc_event c q kind unc isw id addr sl pos : SailM Unit) := by
  unfold ras_ecc_event
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ras_record_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ras_spare_event_frame (ports : Ports) (c : Nat) (q : Nat) (isw : Bool) (id : Nat) (gap : Nat) (sl : Nat) (s03 : (BitVec 4)) : Keeps RView (ras_spare_event c q isw id gap sl s03 : SailM Unit) := by
  unfold ras_spare_event
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ras_record_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem get_stream_frame (ports : Ports) (r : row) (s : Nat) : Keeps RView (get_stream r s : SailM (Vector (Vector Out.srf_write 44) 64)) := by
  unfold get_stream
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem srf_spare_frame (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Keeps RView (srf_spare r s g t : SailM (BitVec Out.Functions.superlanes)) := by
  unfold srf_spare
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact get_stream_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem srf_s03_frame (ports : Ports) (rw : Bool) (g : Int) (t : Int) (x : Int) : Keeps RView (srf_s03 rw g t x : SailM (BitVec 4)) := by
  unfold srf_s03
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact srf_spare_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ecc_check_value_frame (ports : Ports) (ci : Int) (q : Nat) (kind : Nat) (isw : Bool) (id : Nat) (addr : Nat) (v : vec) (ov : ecc_ov) : Keeps RView (ecc_check_value ci q kind isw id addr v ov : SailM (vec × (BitVec Out.Functions.superlanes))) := by
  unfold ecc_check_value
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ras_ecc_event_frame ports _ _ _ _ _ _ _ _ _
    | exact ras_spare_event_frame ports _ _ _ _ _ _ _
    | exact srf_s03_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem srf_read_chain_ov_frame (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Keeps RView (srf_read_chain_ov r s g t : SailM (srf_lookup × vec × ecc_ov)) := by
  unfold srf_read_chain_ov
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact get_stream_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem srf_only_downstream_frame (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Keeps RView (srf_only_downstream r s g t : SailM Bool) := by
  unfold srf_only_downstream
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact get_stream_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem srf_read_ov_frame (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Keeps RView (srf_read_ov r s g t : SailM (srf_lookup × vec × ecc_ov)) := by
  unfold srf_read_ov
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact get_stream_frame ports _ _
    | exact srf_only_downstream_frame ports _ _ _ _
    | exact srf_spare_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem read_planes_full_frame (ports : Ports) (s : streamref) (w : Nat) (t : Int) (chain : Bool) (chk : Bool) : Keeps RView (read_planes_full s w t chain chk : SailM (Result ((Vector vec 4) × (Vector ecc_ov 4)) diag)) := by
  unfold read_planes_full
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ecc_check_value_frame ports _ _ _ _ _ _ _ _
    | exact plane_stream_frame ports _ _
    | exact srf_read_chain_ov_frame ports _ _ _ _
    | exact srf_read_ov_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem read_planes_core_frame (ports : Ports) (s : streamref) (w : Nat) (t : Int) (chain : Bool) : Keeps RView (read_planes_core s w t chain : SailM (Result (Vector vec 4) diag)) := by
  unfold read_planes_core
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_full_frame ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem read_planes_frame (ports : Ports) (s : streamref) (w : Nat) (t : Int) : Keeps RView (read_planes s w t : SailM (Result (Vector vec 4) diag)) := by
  unfold read_planes
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_core_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_PCISEND_frame (ports : Ports) (s : streamref) : Keeps RView (execute_C2C_PCISEND s : SailM ExecutionResult) := by
  unfold execute_C2C_PCISEND
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact pci_busy_frame ports _
    | exact read_planes_frame ports _ _ _
    | exact ports.host_send _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem c2c_link_frame (ports : Ports) (link : (BitVec 7)) : Keeps RView (c2c_link link : SailM (Option Nat)) := by
  unfold c2c_link
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem c2c_peer_ok_frame (ports : Ports) (chip : (BitVec 9)) (l : Nat) : Keeps RView (c2c_peer_ok chip l : SailM Bool) := by
  unfold c2c_peer_ok
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ports.c2c_peer _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_RECEIVE_frame (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Keeps RView (execute_C2C_RECEIVE chip link s : SailM ExecutionResult) := by
  unfold execute_C2C_RECEIVE
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact c2c_link_frame ports _
    | exact c2c_peer_ok_frame ports _ _
    | exact commit_at_frame ports _
    | exact exec_cycle_frame ports _
    | exact write_planes_frame ports _ _ _ _
    | exact ports.c2c_recv _ _
    | exact ports.c2c_recv_bad _
    | exact ports.c2c_recv_poison _ _
    | exact ports.c2c_recv_ready _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem c2c_busy_frame (ports : Ports) (l : Nat) (c : Nat) : Keeps RView (c2c_busy l c : SailM Bool) := by
  unfold c2c_busy
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem c2c_put_frame (ports : Ports) (l : Nat) (t : c2c_ptype) (v : vec) (poison : Nat) : Keeps RView (c2c_put l t v poison : SailM Unit) := by
  unfold c2c_put
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact ports.c2c_send _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_SEND_frame (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Keeps RView (execute_C2C_SEND chip link s : SailM ExecutionResult) := by
  unfold execute_C2C_SEND
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact c2c_busy_frame ports _ _
    | exact c2c_link_frame ports _
    | exact c2c_peer_ok_frame ports _ _
    | exact c2c_put_frame ports _ _ _ _
    | exact exec_cycle_frame ports _
    | exact read_planes_frame ports _ _ _
    | exact ports.c2c_packet_busy _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_TRANSMIT_frame (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) : Keeps RView (execute_C2C_TRANSMIT chip link : SailM ExecutionResult) := by
  unfold execute_C2C_TRANSMIT
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact c2c_busy_frame ports _ _
    | exact c2c_link_frame ports _
    | exact c2c_peer_ok_frame ports _ _
    | exact c2c_put_frame ports _ _ _ _
    | exact exec_cycle_frame ports _
    | exact ports.c2c_packet_busy _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_CONFIG_frame (ports : Ports) (v : (BitVec 16)) : Keeps RView (execute_CONFIG v : SailM ExecutionResult) := by
  unfold execute_CONFIG
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_arm_frame (ports : Ports) (q : Nat) (st : streamref) (t : Nat) (data : vec) (ov : ecc_ov) : Keeps RView (stream_fetch_arm q st t data ov : SailM Unit) := by
  unfold stream_fetch_arm
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_start_frame (ports : Ports) (st : streamref) : Keeps RView (stream_fetch_start st : SailM ExecutionResult) := by
  unfold stream_fetch_start
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact read_planes_frame ports _ _ _
    | exact stream_fetch_arm_frame ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_IFETCH_frame (ports : Ports) (st : streamref) : Keeps RView (execute_IFETCH st : SailM ExecutionResult) := by
  unfold execute_IFETCH
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact stream_fetch_start_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_IFETCHM_frame (ports : Ports) (app_0 : sram_addr) : Keeps RView (execute_IFETCHM app_0 : SailM ExecutionResult) := by
  unfold execute_IFETCHM
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact ports.read_sram_ecc _ _
    | exact ports.read_sram_word _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem commit_cycle_frame (ports : Ports) (unitArg : Unit) : Keeps RView (commit_cycle unitArg : SailM Nat) := by
  unfold commit_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem eff_addr_frame (ports : Ports) (app_0 : sram_addr) : Keeps RView (eff_addr app_0 : SailM (Option sram_addr)) := by
  unfold eff_addr
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem eff_stream_frame (ports : Ports) (app_0 : streamref) : Keeps RView (eff_stream app_0 : SailM (Option streamref)) := by
  unfold eff_stream
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem read_cycle_frame (ports : Ports) (unitArg : Unit) : Keeps RView (read_cycle unitArg : SailM Int) := by
  unfold read_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem read_plane_raw_frame (ports : Ports) (s : streamref) (t : Int) : Keeps RView (read_plane_raw s t : SailM (Result (vec × ecc_ov) diag)) := by
  unfold read_plane_raw
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_full_frame ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sram_read_frame (ports : Ports) (s : Nat) (a : sram_addr) : Keeps RView (sram_read s a : SailM (Option vec)) := by
  unfold sram_read
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ports.read_sram_word _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sram_read_ov_frame (ports : Ports) (s : Nat) (a : sram_addr) : Keeps RView (sram_read_ov s a : SailM ecc_ov) := by
  unfold sram_read_ov
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ports.read_sram_ecc _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem write_sram_ecc_frame (ports : Ports) (s : Nat) (a : Nat) (v : ecc_ov) : Keeps RView (write_sram_ecc s a v : SailM Unit) := by
  unfold write_sram_ecc
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ports.raw_write_sram_ecc _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem write_sram_word_frame (ports : Ports) (s : Nat) (a : Nat) (v : vec) : Keeps RView (write_sram_word s a v : SailM Unit) := by
  unfold write_sram_word
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ports.raw_write_sram_word _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sram_write_ov_frame (ports : Ports) (c : Nat) (s : Nat) (typ_2 : sram_addr) (v : vec) (ov : ecc_ov) : Keeps RView (sram_write_ov c s typ_2 v ov : SailM Bool) := by
  unfold sram_write_ov
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact write_sram_ecc_frame ports _ _ _
    | exact write_sram_word_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem tile_on_frame (ports : Ports) (g : Int) : Keeps RView (tile_on g : SailM Bool) := by
  unfold tile_on
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_DIRECT_frame (ports : Ports) (op : mem_dir) (a0 : sram_addr) (s0 : streamref) (t : (BitVec 1)) : Keeps RView (execute_MEM_DIRECT op a0 s0 t : SailM ExecutionResult) := by
  unfold execute_MEM_DIRECT
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_cycle_frame ports _
    | exact eff_addr_frame ports _
    | exact eff_stream_frame ports _
    | exact exec_cycle_frame ports _
    | exact read_cycle_frame ports _
    | exact read_plane_raw_frame ports _ _
    | exact sram_read_frame ports _ _
    | exact sram_read_ov_frame ports _ _
    | exact sram_write_ov_frame ports _ _ _ _ _
    | exact tile_on_frame ports _
    | exact write_planes_fwd_frame ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ecc_check_sl_frame (ports : Ports) (ci : Int) (q : Nat) (slice_ : Nat) (addr : Nat) (g : Nat) (d : (BitVec 128)) (e : (BitVec 10)) : Keeps RView (ecc_check_sl ci q slice_ addr g d e : SailM ((BitVec 128) × Bool)) := by
  unfold ecc_check_sl
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ras_ecc_event_frame ports _ _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem srf_report_frame (ports : Ports) (r : row) (g : Nat) (c : Nat) (prio : Int) (xs : (BitVec Out.Functions.superlanes)) (sp : (Vector (BitVec Out.Functions.superlanes) 30)) : Keeps RView (srf_report r g c prio xs sp : SailM Bool) := by
  unfold srf_report
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem mem_report_frame (ports : Ports) (prio : Int) (bounds : (BitVec Out.Functions.superlanes)) (addrs : (Vector Int Out.Functions.superlanes)) (data : streamref) (map : streamref) (c : Nat) : Keeps RView (mem_report prio bounds addrs data map c : SailM Unit) := by
  unfold mem_report
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact srf_report_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_GATHER_frame (ports : Ports) (d0 : streamref) (m : streamref) (kb : (BitVec 4)) (t : (BitVec 1)) : Keeps RView (execute_MEM_GATHER d0 m kb t : SailM ExecutionResult) := by
  unfold execute_MEM_GATHER
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_cycle_frame ports _
    | exact ecc_check_sl_frame ports _ _ _ _ _ _ _
    | exact eff_addr_frame ports _
    | exact eff_stream_frame ports _
    | exact mem_report_frame ports _ _ _ _ _ _
    | exact read_cycle_frame ports _
    | exact read_planes_frame ports _ _ _
    | exact sram_read_frame ports _ _
    | exact sram_read_ov_frame ports _ _
    | exact tile_on_frame ports _
    | exact write_planes_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_SCATTER_frame (ports : Ports) (s0 : streamref) (m : streamref) (kb : (BitVec 4)) (t : (BitVec 1)) : Keeps RView (execute_MEM_SCATTER s0 m kb t : SailM ExecutionResult) := by
  unfold execute_MEM_SCATTER
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_cycle_frame ports _
    | exact ecc_check_sl_frame ports _ _ _ _ _ _ _
    | exact eff_addr_frame ports _
    | exact eff_stream_frame ports _
    | exact exec_cycle_frame ports _
    | exact mem_report_frame ports _ _ _ _ _ _
    | exact read_cycle_frame ports _
    | exact read_planes_frame ports _ _ _
    | exact sram_read_frame ports _ _
    | exact sram_read_ov_frame ports _ _
    | exact sram_write_ov_frame ports _ _ _ _ _
    | exact tile_on_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_SETSEGMENT_frame (ports : Ports) (n : (BitVec 17)) : Keeps RView (execute_MEM_SETSEGMENT n : SailM ExecutionResult) := by
  unfold execute_MEM_SETSEGMENT
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem macc_busy_in_frame (ports : Ports) (xs : (List abc_entry)) : Keeps RView (macc_busy_in xs : SailM Bool) := by
  induction xs
  case nil =>
    unfold macc_busy_in
    repeat' first
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold macc_busy_in
    repeat' first
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
theorem macc_busy_frame (ports : Ports) (p : mxm_plane) : Keeps RView (macc_busy p : SailM Bool) := by
  unfold macc_busy
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact macc_busy_in_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem mxm_stage_acc_frame (ports : Ports) (pl : Nat) (part : mxm_part) (a : Nat) (v : mxm_plane) : Keeps RView (mxm_stage_acc pl part a v : SailM Unit) := by
  unfold mxm_stage_acc
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem mxm_stage_frame (ports : Ports) (pl : Nat) (part : mxm_part) (v : mxm_plane) : Keeps RView (mxm_stage pl part v : SailM Unit) := by
  unfold mxm_stage
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact mxm_stage_acc_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem plane_frame (ports : Ports) (unitArg : Unit) : Keeps RView (plane unitArg : SailM Nat) := by
  unfold plane
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem mxm_plane_at_frame (ports : Ports) (i : Int) : Keeps RView (mxm_plane_at i : SailM mxm_plane) := by
  unfold mxm_plane_at
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem plane_blocked_frame (ports : Ports) (t : (Option mxm_wtype)) : Keeps RView (plane_blocked t : SailM Bool) := by
  unfold plane_blocked
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact mxm_plane_at_frame ports _
    | exact plane_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ras_mxm_event_frame (ports : Ports) (c : Nat) (q : Int) (unc : Bool) (plane : Nat) (lane : Nat) : Keeps RView (ras_mxm_event c q unc plane lane : SailM Unit) := by
  unfold ras_mxm_event
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact plane_frame ports _
    | exact ras_record_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
open Out.Register Out.ExecutionResult Out.mxm_wtype Out.mxm_part Out.diag
def abcFaultPhase (mine : List (Nat × Nat)) (others pend : List mxm_fault) :
    SailM ((List (Nat × Nat)) × List mxm_fault × List mxm_fault) :=
  (( do
            let loop__step_lower := 0
            let loop__step_upper ← do (pure ((list_len (← readReg mxm_fault_list)) -i 1))
            let mut loop_vars := (mine, others, pend)
            for _step in [loop__step_lower:loop__step_upper:1]i do
              let (mine, others, pend) := loop_vars
              loop_vars ← do
                let (mine, others, pend) ← (( do
                  match pend with
                  | [] => (pure (mine, others, pend))
                  | (mfe :: rest) =>
                    (do
                      let (mine, others) ← (( do
                        if ((mfe.mf_pl == (← (plane ()))) : Bool)
                        then
                          (let mine : (List (Nat × Nat)) := ((mfe.mf_lane, mfe.mf_bit) :: mine)
                          (pure (mine, others)))
                        else
                          (let others : (List mxm_fault) := (mfe :: others)
                          (pure (mine, others))) ) : SailM ((List (Nat × Nat)) × (List mxm_fault))
                        )
                      let pend : (List mxm_fault) := rest
                      (pure (mine, others, pend))) ) : SailM
                  ((List (Nat × Nat)) × (List mxm_fault) × (List mxm_fault)) )
                (pure (mine, others, pend))
            (pure loop_vars) ) : SailM ((List (Nat × Nat)) × (List mxm_fault) × (List mxm_fault))
            )

def abcProductPhase (t : mxm_wtype) (f : BitVec 1) (p : mxm_plane) (x : planes)
    (ws xs : Bool) (out : Vector (BitVec 32) Out.Functions.mxm_rows) (wide : Vector wlane Out.Functions.mxm_rows) :
    SailM ((Vector (BitVec 32) Out.Functions.mxm_rows) × Vector wlane Out.Functions.mxm_rows) :=
  (( do
                let loop_r_lower := 0
                let loop_r_upper := (p.rows -i 1)
                let mut loop_vars_1 := (out, wide)
                for r in [loop_r_lower:loop_r_upper:1]i do
                  let (out, wide) := loop_vars_1
                  loop_vars_1 :=
                    let (out, wide) : ((Vector (BitVec 32) Out.Functions.mxm_rows) × (Vector wlane Out.Functions.mxm_rows)) :=
                      match t with
                      | .W_I8 =>
                        (let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r
                            (int8_dot (GetElem?.getElem! p.weights r) ws (GetElem?.getElem! x 0) xs))
                        (out, wide))
                      | .W_U8 =>
                        (let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r
                            (int8_dot (GetElem?.getElem! p.weights r) ws (GetElem?.getElem! x 0) xs))
                        (out, wide))
                      | .W_FP8 =>
                        (let d :=
                          (fp8_exact
                            (if ((f == 0#1) : Bool)
                            then FMT_E4M3
                            else FMT_E5M2) (GetElem?.getElem! p.weights r) (GetElem?.getElem! x 0))
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r (dsum_round d))
                        let wide : (Vector wlane Out.Functions.mxm_rows) :=
                          if (ext_wide_acc : Bool)
                          then (vectorUpdate wide r (dsum_wide d))
                          else wide
                        (out, wide))
                      | .W_F16 =>
                        (let (wlo, whi) := (f16_row p.weights r)
                        let d := (f16_exact wlo whi (GetElem?.getElem! x 0) (GetElem?.getElem! x 1))
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r (dsum_round d))
                        let wide : (Vector wlane Out.Functions.mxm_rows) :=
                          if (ext_wide_acc : Bool)
                          then (vectorUpdate wide r (dsum_wide d))
                          else wide
                        (out, wide))
                      | .W_BFP8 =>
                        (let (wd, wsc) := (f16_row p.weights r)
                        let d := (bfp8_exact wd wsc (GetElem?.getElem! x 0) (GetElem?.getElem! x 1))
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r (dsum_round d))
                        let wide : (Vector wlane Out.Functions.mxm_rows) :=
                          if (ext_wide_acc : Bool)
                          then (vectorUpdate wide r (dsum_wide d))
                          else wide
                        (out, wide))
                      | .W_BF16 =>
                        (let (wlo, whi) := (f16_row p.weights r)
                        let out : (Vector (BitVec 32) Out.Functions.mxm_rows) :=
                          (vectorUpdate out r
                            (bf16_dot wlo whi (GetElem?.getElem! x 0) (GetElem?.getElem! x 1)))
                        (out, wide))
                    (out, wide)
                (pure loop_vars_1) ) : SailM ((Vector (BitVec 32) Out.Functions.mxm_rows) × (Vector wlane Out.Functions.mxm_rows)) )

def abcPoisonPhase (t : mxm_wtype) (p : mxm_plane) (poison : BitVec Out.Functions.superlanes) : SailM (BitVec Out.Functions.superlanes) :=
  (( do
                if ((poison == (BitVec.zero Out.Functions.superlanes)) : Bool)
                then
                  (do
                    let loop_r_lower := 0
                    let loop_r_upper := (p.rows -i 1)
                    let mut loop_vars_2 := poison
                    for r in [loop_r_lower:loop_r_upper:1]i do
                      let poison := loop_vars_2
                      loop_vars_2 :=
                        let rb : Bool :=
                          match t with
                          | .W_F16 => (f16_row_bad p.w_bad r)
                          | .W_BFP8 => (f16_row_bad p.w_bad r)
                          | .W_BF16 => (f16_row_bad p.w_bad r)
                          | _ => (GetElem?.getElem! p.w_bad r)
                        if ((rb && ((tdiv_int r 16) <b Out.Functions.superlanes)) : Bool)
                        then (BitVec.update poison (tdiv_int r 16) 1#1)
                        else poison
                    (pure loop_vars_2))
                else (pure poison) ) : SailM (BitVec Out.Functions.superlanes) )

def abcExecFactory (t : mxm_wtype) (f : (BitVec 1)) (s : streamref) : SailM ExecutionResult := do
  let p ← do (pure (GetElem?.getElem! (← readReg mxm) (← (plane ()))))
  let w : Int :=
    if ((is_pair t) : Bool)
    then 2
    else 1
  if (((← (plane_blocked (some t))) || (← do
         (pure ((p.rows == 0) || (← do
               (pure (((← readReg issue_cycle) <b p.ready_iw) || ((Out.Functions.not (abc_types_ok p.wtype t)) || (← (macc_busy
                         p)))))))))) : Bool)
  then (pure (Diagnostic RES_NOT_READY))
  else
    (do
      match (← (read_planes s w (← (exec_cycle ())))) with
      | .Err e => (pure (Diagnostic e))
      | .Ok x =>
        (do
          let mine : (List (Nat × Nat)) := []
          let others : (List mxm_fault) := []
          let pend ← do readReg mxm_fault_list
          let (mine, others, pend) ← abcFaultPhase mine others pend
          writeReg mxm_fault_list others
          let verdict : (Nat × Nat) := (0, 0)
          let verdict : (Nat × Nat) :=
            match mine with
            | [] => verdict
            | _ =>
              (let x1 : vec :=
                if ((is_pair t) : Bool)
                then (GetElem?.getElem! x 1)
                else zero_vec
              (mxm_check p t f (GetElem?.getElem! x 0) x1 mine))
          let (vstat, vlane) := verdict
          let ec ← (( do
            if (((← (exec_cycle ())) ≥b 0) : Bool)
            then (exec_cycle ())
            else (pure 0) ) : SailM Nat )
          if ((vstat == 2) : Bool)
          then
            (do
              (ras_mxm_event ec (← readReg issue_queue) true (← (plane ())) 65535)
              (pure (Diagnostic ECC_MXM)))
          else
            (do
              if ((vstat == 1) : Bool)
              then
                (do
                  (pure (mxm_corr_callback ec (← readReg issue_queue) (← (plane ())) vlane))
                  (ras_mxm_event ec (← readReg issue_queue) false (← (plane ())) vlane))
              else (pure ())
              let out : (Vector (BitVec 32) Out.Functions.mxm_rows) := (vectorInit 0x00000000#32)
              let wide : (Vector wlane Out.Functions.mxm_rows) := (vectorInit wlane_zero)
              let ws := (p.wtype == W_I8)
              let xs := (t == W_I8)
              let (out, wide) ← abcProductPhase t f p x ws xs out wide
              let poison ← (( do
                if (((← readReg ecc_bad) != (BitVec.zero Out.Functions.superlanes)) : Bool)
                then (pure (sail_ones Out.Functions.superlanes))
                else (pure (BitVec.zero Out.Functions.superlanes)) ) : SailM (BitVec Out.Functions.superlanes) )
              let poison ← abcPoisonPhase t p poison
              let e ← (( do
                (pure { dispatch := ← readReg issue_cycle
                        commit := ← (commit_at ())
                        ptype := t
                        product := out
                        wide := wide
                        poison := poison
                        q := ← readReg issue_queue }) ) : SailM abc_entry )
              let q := p
              let q ←
                (pure { q with abc := (e :: (abc_prune p.abc (← readReg issue_cycle))) })
              (mxm_stage (← (plane ())) P_ABC q)
              (pure (Issued ())))))

theorem actual_abc_factory_source (t : mxm_wtype) (f : BitVec 1) (s : streamref) :
    abc_exec t f s=abcExecFactory t f s := by
  unfold abc_exec abcExecFactory abcFaultPhase abcProductPhase abcPoisonPhase
  rfl

theorem abcFaultPhase_frame (ports : Ports) (mine : List (Nat × Nat)) (others pend : List mxm_fault) : Keeps RView (abcFaultPhase mine others pend) := by
  unfold abcFaultPhase
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact plane_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem abcProductPhase_frame (ports : Ports) (t : mxm_wtype) (f : BitVec 1) (p : mxm_plane) (x : planes) (ws xs : Bool) (out : Vector (BitVec 32) Out.Functions.mxm_rows) (wide : Vector wlane Out.Functions.mxm_rows) : Keeps RView (abcProductPhase t f p x ws xs out wide) := by
  unfold abcProductPhase
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact plane_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

theorem abcPoisonPhase_frame (ports : Ports) (t : mxm_wtype) (p : mxm_plane) (poison : BitVec Out.Functions.superlanes) : Keeps RView (abcPoisonPhase t p poison) := by
  unfold abcPoisonPhase
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact plane_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro

attribute [local irreducible] abcFaultPhase abcProductPhase abcPoisonPhase

theorem abc_exec_frame (ports : Ports) (t : mxm_wtype) (f : (BitVec 1)) (s : streamref) : Keeps RView (abc_exec t f s : SailM ExecutionResult) := by
  rw [actual_abc_factory_source]
  unfold abcExecFactory
  repeat' first
    | exact abcFaultPhase_frame ports _ _ _
    | exact abcProductPhase_frame ports _ _ _ _ _ _ _ _
    | exact abcPoisonPhase_frame ports _ _ _
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | exact exec_cycle_frame ports _
    | exact macc_busy_frame ports _
    | exact mxm_stage_frame ports _ _ _
    | exact plane_frame ports _
    | exact plane_blocked_frame ports _
    | exact ras_mxm_event_frame ports _ _ _ _ _
    | exact read_planes_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_ABC_frame (ports : Ports) (t : mxm_wtype) (s : streamref) : Keeps RView (execute_MXM_ABC t s : SailM ExecutionResult) := by
  unfold execute_MXM_ABC
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact abc_exec_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_ABC8_frame (ports : Ports) (f : (BitVec 1)) (s : streamref) : Keeps RView (execute_MXM_ABC8 f s : SailM ExecutionResult) := by
  unfold execute_MXM_ABC8
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact abc_exec_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_ACC_frame (ports : Ports) (t : mxm_acctype) (m : acc_mode) (d : streamref) (ab : (BitVec 5)) : Keeps RView (execute_MXM_ACC t m d ab : SailM ExecutionResult) := by
  unfold execute_MXM_ACC
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | exact mxm_stage_acc_frame ports _ _ _ _
    | exact plane_frame ports _
    | exact plane_blocked_frame ports _
    | exact write_planes_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_IW_frame (ports : Ports) (t : mxm_wtype) : Keeps RView (execute_MXM_IW t : SailM ExecutionResult) := by
  unfold execute_MXM_IW
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | exact mxm_stage_frame ports _ _ _
    | exact plane_frame ports _
    | exact plane_blocked_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem read_planes16_frame (ports : Ports) (s : streamref) (t : Int) : Keeps RView (read_planes16 s t : SailM (Result (Vector vec 16) diag)) := by
  unfold read_planes16
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ecc_check_value_frame ports _ _ _ _ _ _ _ _
    | exact plane_stream_frame ports _ _
    | exact srf_read_ov_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_IWS_frame (ports : Ports) (t : mxm_wtype) (s : streamref) : Keeps RView (execute_MXM_IWS t s : SailM ExecutionResult) := by
  unfold execute_MXM_IWS
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | exact exec_cycle_frame ports _
    | exact mxm_stage_frame ports _ _ _
    | exact plane_frame ports _
    | exact plane_blocked_frame ports _
    | exact read_planes16_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_LW_frame (ports : Ports) (t : mxm_wtype) (s : streamref) : Keeps RView (execute_MXM_LW t s : SailM ExecutionResult) := by
  unfold execute_MXM_LW
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | exact exec_cycle_frame ports _
    | exact mxm_stage_frame ports _ _ _
    | exact plane_frame ports _
    | exact plane_blocked_frame ports _
    | exact read_planes16_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SETVL_frame (ports : Ports) (n : (BitVec 6)) : Keeps RView (execute_SETVL n : SailM ExecutionResult) := by
  unfold execute_SETVL
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sx_frame (ports : Ports) (unitArg : Unit) : Keeps RView (sx unitArg : SailM Nat) := by
  unfold sx
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sxm_load_map_frame (ports : Ports) (mi : (BitVec 4)) (lo : vec) (hi : vec) : Keeps RView (sxm_load_map mi lo hi : SailM Unit) := by
  unfold sxm_load_map
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact sx_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sxm_rd_cycle_frame (ports : Ports) (unitArg : Unit) : Keeps RView (sxm_rd_cycle unitArg : SailM Int) := by
  unfold sxm_rd_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_LOADMAP_frame (ports : Ports) (mi : (BitVec 4)) (s : streamref) : Keeps RView (execute_SXM_LOADMAP mi s : SailM ExecutionResult) := by
  unfold execute_SXM_LOADMAP
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_frame ports _ _ _
    | exact sxm_load_map_frame ports _ _ _
    | exact sxm_rd_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_LOADMAPWIDE_frame (ports : Ports) (mi : (BitVec 4)) (s : streamref) (s2 : streamref) : Keeps RView (execute_SXM_LOADMAPWIDE mi s s2 : SailM ExecutionResult) := by
  unfold execute_SXM_LOADMAPWIDE
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_frame ports _ _ _
    | exact sxm_load_map_frame ports _ _ _
    | exact sxm_rd_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sxm_cm_cycle_frame (ports : Ports) (unitArg : Unit) : Keeps RView (sxm_cm_cycle unitArg : SailM Nat) := by
  unfold sxm_cm_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sxm_emit_frame (ports : Ports) (d : streamref) (v : vec) : Keeps RView (sxm_emit d v : SailM ExecutionResult) := by
  unfold sxm_emit
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact sxm_cm_cycle_frame ports _
    | exact write_planes_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_MAP_frame (ports : Ports) (op : sxm_mapop) (mi : (BitVec 4)) (s : streamref) (d : streamref) : Keeps RView (execute_SXM_MAP op mi s d : SailM ExecutionResult) := by
  unfold execute_SXM_MAP
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | exact read_planes_frame ports _ _ _
    | exact sx_frame ports _
    | exact sxm_emit_frame ports _ _
    | exact sxm_rd_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_ROTATE_frame (ports : Ports) (nb : (BitVec 10)) (s : streamref) (d : streamref) : Keeps RView (execute_SXM_ROTATE nb s d : SailM ExecutionResult) := by
  unfold execute_SXM_ROTATE
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_frame ports _ _ _
    | exact sxm_cm_cycle_frame ports _
    | exact sxm_rd_cycle_frame ports _
    | exact write_planes_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_SELECT_frame (ports : Ports) (a : streamref) (b : streamref) (c : streamref) (m : streamref) (d : streamref) : Keeps RView (execute_SXM_SELECT a b c m d : SailM ExecutionResult) := by
  unfold execute_SXM_SELECT
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_frame ports _ _ _
    | exact sxm_emit_frame ports _ _
    | exact sxm_rd_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_SHIFT_frame (ports : Ports) (dir : shift_dir) (n : (BitVec 9)) (s : streamref) (d : streamref) : Keeps RView (execute_SXM_SHIFT dir n s d : SailM ExecutionResult) := by
  unfold execute_SXM_SHIFT
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_frame ports _ _ _
    | exact sxm_emit_frame ports _ _
    | exact sxm_rd_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_TRANSPOSE_frame (ports : Ports) (s : streamref) (d : streamref) : Keeps RView (execute_SXM_TRANSPOSE s d : SailM ExecutionResult) := by
  unfold execute_SXM_TRANSPOSE
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes16_frame ports _ _
    | exact sxm_cm_cycle_frame ports _
    | exact sxm_rd_cycle_frame ports _
    | exact write_planes_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem sailThrow_frame {α : Type} (ports : Ports) (e : exception) : Keeps RView (sailThrow e : SailM α) := by
  unfold sailThrow PreSail.sailThrow
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem internal_error_frame {α : Type} (ports : Ports) (file : String) (line : Int) (s : String) : Keeps RView (internal_error file line s : SailM α) := by
  unfold internal_error
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact sailThrow_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem fp_transcendental_frame (ports : Ports) (op : Int) (f : fmt) (x : (BitVec 32)) : Keeps RView (fp_transcendental op f x : SailM (BitVec 32)) := by
  unfold fp_transcendental
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact internal_error_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem i8_transcendental_frame (ports : Ports) (op : Int) (x : (BitVec 32)) : Keeps RView (i8_transcendental op x : SailM (BitVec 32)) := by
  unfold i8_transcendental
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact internal_error_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem apply_frame (ports : Ports) (op : vxm_lane_op) (a : (BitVec 32)) (b : (BitVec 32)) (m : (BitVec 32)) : Keeps RView (apply op a b m : SailM (BitVec 32)) := by
  unfold apply
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact fp_transcendental_frame ports _ _ _
    | exact i8_transcendental_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem vxm_commit_cycle_frame (ports : Ports) (unitArg : Unit) : Keeps RView (vxm_commit_cycle unitArg : SailM Nat) := by
  unfold vxm_commit_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact commit_at_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem vxm_read_cycle_frame (ports : Ports) (unitArg : Unit) : Keeps RView (vxm_read_cycle unitArg : SailM Int) := by
  unfold vxm_read_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact exec_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem vxm_read_frame (ports : Ports) (a : vsrc) (w : Nat) : Keeps RView (vxm_read a w : SailM (Result (Vector vec 4) diag)) := by
  unfold vxm_read
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact read_planes_core_frame ports _ _ _ _
    | exact vxm_read_cycle_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem vxm_lanes_frame (ports : Ports) (op : vxm_lane_op) (dst : streamref) (a : vsrc) (b : (Option vsrc)) (m : (Option vsrc)) (ws : Nat) (wd : Nat) (g : (BitVec 1)) : Keeps RView (vxm_lanes op dst a b m ws wd g : SailM ExecutionResult) := by
  unfold vxm_lanes
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact apply_frame ports _ _ _ _
    | exact vxm_commit_cycle_frame ports _
    | exact vxm_read_frame ports _ _
    | exact write_planes_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_BIN_frame (ports : Ports) (op : vxm_binop) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (g : (BitVec 1)) : Keeps RView (execute_VXM_BIN op t d a b g : SailM ExecutionResult) := by
  unfold execute_VXM_BIN
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_lanes_frame ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_CMP_frame (ports : Ports) (mode : cmp_mode) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (g : (BitVec 1)) : Keeps RView (execute_VXM_CMP mode t d a b g : SailM ExecutionResult) := by
  unfold execute_VXM_CMP
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_lanes_frame ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_CONV_frame (ports : Ports) (c : vxm_conv) (g__241 : eltype) (d : streamref) (a : vsrc) : Keeps RView (execute_VXM_CONV c g__241 d a : SailM ExecutionResult) := by
  unfold execute_VXM_CONV
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_lanes_frame ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_CVT_frame (ports : Ports) (st : eltype) (dt : eltype) (d : streamref) (a : vsrc) : Keeps RView (execute_VXM_CVT st dt d a : SailM ExecutionResult) := by
  unfold execute_VXM_CVT
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_lanes_frame ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_MASK_frame (ports : Ports) (t : eltype) (d : streamref) (a : vsrc) (m : vsrc) (g : (BitVec 1)) : Keeps RView (execute_VXM_MASK t d a m g : SailM ExecutionResult) := by
  unfold execute_VXM_MASK
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_lanes_frame ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_REDUCE_frame (ports : Ports) (op : reduce_op) (t : eltype) (d : streamref) (a : vsrc) : Keeps RView (execute_VXM_REDUCE op t d a : SailM ExecutionResult) := by
  unfold execute_VXM_REDUCE
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_commit_cycle_frame ports _
    | exact vxm_read_frame ports _ _
    | exact write_planes_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_SELECT_frame (ports : Ports) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (m : vsrc) (g : (BitVec 1)) : Keeps RView (execute_VXM_SELECT t d a b m g : SailM ExecutionResult) := by
  unfold execute_VXM_SELECT
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_lanes_frame ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_UN_frame (ports : Ports) (op : vxm_unop) (t : eltype) (d : streamref) (a : vsrc) (g : (BitVec 1)) : Keeps RView (execute_VXM_UN op t d a g : SailM ExecutionResult) := by
  unfold execute_VXM_UN
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact vxm_lanes_frame ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_frame (ports : Ports) (merge_var : instruction) : Keeps RView (execute merge_var : SailM ExecutionResult) := by
  unfold execute
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact execute_C2C_HOSTNOTIFY_frame ports _
    | exact execute_C2C_PCIRECV_frame ports _
    | exact execute_C2C_PCISEND_frame ports _
    | exact execute_C2C_RECEIVE_frame ports _ _ _
    | exact execute_C2C_SEND_frame ports _ _ _
    | exact execute_C2C_TRANSMIT_frame ports _ _
    | exact execute_CONFIG_frame ports _
    | exact execute_IFETCH_frame ports _
    | exact execute_IFETCHM_frame ports _
    | exact execute_MEM_DIRECT_frame ports _ _ _ _
    | exact execute_MEM_GATHER_frame ports _ _ _ _
    | exact execute_MEM_SCATTER_frame ports _ _ _ _
    | exact execute_MEM_SETSEGMENT_frame ports _
    | exact execute_MXM_ABC_frame ports _ _
    | exact execute_MXM_ABC8_frame ports _ _
    | exact execute_MXM_ACC_frame ports _ _ _ _
    | exact execute_MXM_IW_frame ports _
    | exact execute_MXM_IWS_frame ports _ _
    | exact execute_MXM_LW_frame ports _ _
    | exact execute_SETVL_frame ports _
    | exact execute_SXM_LOADMAP_frame ports _ _
    | exact execute_SXM_LOADMAPWIDE_frame ports _ _ _
    | exact execute_SXM_MAP_frame ports _ _ _ _
    | exact execute_SXM_ROTATE_frame ports _ _ _
    | exact execute_SXM_SELECT_frame ports _ _ _ _ _
    | exact execute_SXM_SHIFT_frame ports _ _ _ _
    | exact execute_SXM_TRANSPOSE_frame ports _ _
    | exact execute_VXM_BIN_frame ports _ _ _ _ _ _
    | exact execute_VXM_CMP_frame ports _ _ _ _ _ _
    | exact execute_VXM_CONV_frame ports _ _ _ _
    | exact execute_VXM_CVT_frame ports _ _ _ _
    | exact execute_VXM_MASK_frame ports _ _ _ _ _
    | exact execute_VXM_REDUCE_frame ports _ _ _ _
    | exact execute_VXM_SELECT_frame ports _ _ _ _ _ _
    | exact execute_VXM_UN_frame ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem diag_record_frame (ports : Ports) (c : Nat) (q : Nat) (code : Nat) : Keeps RView (diag_record c q code : SailM Unit) := by
  unfold diag_record
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem diag_report_frame (ports : Ports) (c : Nat) (q : Nat) (d : diag) : Keeps RView (diag_report c q d : SailM Unit) := by
  unfold diag_report
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact diag_record_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem report_frame (ports : Ports) (q : Nat) (c : Nat) (d : diag) : Keeps RView (report q c d : SailM Unit) := by
  unfold report
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact diag_report_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_element_frame (ports : Ports) (e : exec_element) : Keeps RView (execute_element e : SailM Unit) := by
  unfold execute_element
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact ecc_begin_frame ports _
    | exact exec_cycle_frame ports _
    | exact execute_frame ports _
    | exact report_frame ports _ _ _
    | exact ports.observe_site _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem run_element_frame (ports : Ports) (q : Nat) (i : instruction) (c0 : Nat) (k : Nat) (off : Int) (soff : Int) : Keeps RView (run_element q i c0 k off soff : SailM Unit) := by
  unfold run_element
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact execute_element_frame ports _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem agt_get_frame (ports : Ports) (q : Nat) (p : Nat) : Keeps RView (agt_get q p : SailM agt) := by
  unfold agt_get
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem agt_set_frame (ports : Ports) (q : Nat) (p : Nat) (a : agt) : Keeps RView (agt_set q p a : SailM Unit) := by
  unfold agt_set
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem agt_configure_frame (ports : Ports) (q : Nat) (p : Nat) (f : agt_field) (v : (BitVec 16)) : Keeps RView (agt_configure q p f v : SailM Bool) := by
  unfold agt_configure
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_get_frame ports _ _
    | exact agt_set_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem host_parked_frame (ports : Ports) (q : Nat) (c : Nat) : Keeps RView (host_parked q c : SailM Bool) := by
  unfold host_parked
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem path_busy_frame (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Keeps RView (path_busy q p c : SailM Bool) := by
  unfold path_busy
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem path_tail_frame (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Keeps RView (path_tail q p c : SailM Bool) := by
  unfold path_tail
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem path_view_of_frame (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Keeps RView (path_view_of q p c : SailM path_view) := by
  unfold path_view_of
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_get_frame ports _ _
    | exact path_busy_frame ports _ _ _
    | exact path_tail_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem polling_frame (ports : Ports) (q : Nat) : Keeps RView (polling q : SailM Bool) := by
  unfold polling
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem parked_one_frame (ports : Ports) (k : Nat) (c : Nat) : Keeps RView (parked_one k c : SailM Bool) := by
  unfold parked_one
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem queue_parked_frame (ports : Ports) (q : Nat) (c : Nat) : Keeps RView (queue_parked q c : SailM Bool) := by
  unfold queue_parked
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact parked_one_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem hold_view_of_frame (ports : Ports) (q : Nat) (c : Nat) : Keeps RView (hold_view_of q c : SailM hold_view) := by
  unfold hold_view_of
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact host_parked_frame ports _ _
    | exact path_view_of_frame ports _ _ _
    | exact polling_frame ports _
    | exact queue_parked_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem hold_check_frame (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) (strict : Bool) : Keeps RView (hold_check q c i path strict : SailM Bool) := by
  unfold hold_check
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact hold_view_of_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem dispatch_check_frame (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) : Keeps RView (dispatch_check q c i path : SailM (Option diag)) := by
  unfold dispatch_check
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact hold_check_frame ports _ _ _ _ _
    | exact internal_error_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem dispatch_cfg_frame (ports : Ports) (q : Nat) (c : Nat) (p : Nat) (i : instruction) : Keeps RView (dispatch_cfg q c p i : SailM Unit) := by
  unfold dispatch_cfg
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_configure_frame ports _ _ _ _
    | exact agt_get_frame ports _ _
    | exact agt_set_frame ports _ _ _
    | exact dispatch_check_frame ports _ _ _ _
    | exact internal_error_frame ports _ _ _
    | exact report_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem agt_due_frame (ports : Ports) (q : Nat) (p : Nat) (c : Nat) (parked : Bool) : Keeps RView (agt_due q p c parked : SailM (Option ((Vector agt_level 4) × Nat × Int × Int))) := by
  unfold agt_due
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_get_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem due_refs_frame (ports : Ports) (q : Nat) (c : Nat) (parked : Bool) : Keeps RView (due_refs q c parked : SailM (Vector (Option ((Vector agt_level 4) × Nat × Int × Int)) 2)) := by
  unfold due_refs
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_due_frame ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem agt_refresh_frame (ports : Ports) (q : Nat) (p : Nat) : Keeps RView (agt_refresh q p : SailM Unit) := by
  unfold agt_refresh
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_get_frame ports _ _
    | exact agt_set_frame ports _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem bank_report_frame (ports : Ports) (q : Nat) (c : Nat) (a : (instruction × Int × Int)) (b : (instruction × Int × Int)) : Keeps RView (bank_report q c a b : SailM Unit) := by
  unfold bank_report
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact srf_report_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem hold_path_frame (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Keeps RView (hold_path q p c : SailM Unit) := by
  unfold hold_path
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem loop_clash_ref_frame (ports : Ports) (q : Nat) (p : Nat) (s : Int) (ss : Int) : Keeps RView (loop_clash_ref q p s ss : SailM (instruction × Int × Int)) := by
  unfold loop_clash_ref
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_get_frame ports _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem run_loop_frame (ports : Ports) (q : Nat) (c : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) (blocked : Bool) (partner : (Option (instruction × Int × Int))) : Keeps RView (run_loop q c p st blocked partner : SailM Unit) := by
  unfold run_loop
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide) (by decide) (by decide)
    | exact agt_get_frame ports _ _
    | exact agt_refresh_frame ports _ _
    | exact agt_set_frame ports _ _ _
    | exact bank_report_frame ports _ _ _ _
    | exact hold_path_frame ports _ _ _
    | exact loop_clash_ref_frame ports _ _ _ _
    | exact report_frame ports _ _ _
    | exact run_element_frame ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem resource_projection {α : Type} (action : SailM α) (h : Keeps RView action) : Keeps PatrolValidityProofs.View action :=
  PatrolProtectedProofs.map_keeps RView Prod.fst action h
theorem iq_head_projection {α : Type} (action : SailM α) (h : Keeps RView action) : RegKeeps Register.iq_head action :=
  PatrolProtectedProofs.map_keeps RView (fun x => x.2.1) action h
theorem iq_len_projection {α : Type} (action : SailM α) (h : Keeps RView action) : RegKeeps Register.iq_len action :=
  PatrolProtectedProofs.map_keeps RView (fun x => x.2.2.1) action h
theorem iq_memory_projection {α : Type} (action : SailM α) (h : Keeps RView action) : Keeps IQMem action :=
  PatrolProtectedProofs.map_keeps RView (fun x => x.2.2.2) action h
theorem actual_run_element_iq_head (ports : Ports) (q : Nat) (i : instruction) (c k : Nat) (off soff : Int) :
    RegKeeps Register.iq_head (run_element q i c k off soff) := iq_head_projection _ (run_element_frame ports q i c k off soff)
theorem actual_run_element_iq_len (ports : Ports) (q : Nat) (i : instruction) (c k : Nat) (off soff : Int) :
    RegKeeps Register.iq_len (run_element q i c k off soff) := iq_len_projection _ (run_element_frame ports q i c k off soff)
theorem actual_run_element_iq_memory (ports : Ports) (q : Nat) (i : instruction) (c k : Nat) (off soff : Int) :
    Keeps IQMem (run_element q i c k off soff) := iq_memory_projection _ (run_element_frame ports q i c k off soff)
theorem actual_run_element_resource (ports : Ports) (q : Nat) (i : instruction) (c k : Nat) (off soff : Int) :
    Keeps PatrolValidityProofs.View (run_element q i c k off soff) := resource_projection _ (run_element_frame ports q i c k off soff)
end ResourceExecuteFrameProofs
