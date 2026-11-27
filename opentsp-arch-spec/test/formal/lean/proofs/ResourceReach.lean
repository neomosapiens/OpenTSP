-- Whole actual reset/tick patrol validity; no whole-callee frame premise.
import Proofs.ResourceExecuteFrames
import Proofs.ParcelFrame
import Proofs.GrantConsumption
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 100000
namespace ResourceReachProofs
open ResourceFrameProofs ResourceInvariantProofs PatrolValidityProofs
structure Ports : Prop where
  execution : ResourceExecuteFrameProofs.Ports
  patrol : ∀m, PatrolValidityProofs.Ports m
  c2c_recv_ready : ∀ (a0 : Nat) (a1 : Nat), Keeps View (c2c_recv_ready a0 a1)
  iq_image_len : ∀ (a0 : Nat), Keeps View (iq_image_len a0)
  iq_image_unit : ∀ (a0 : Nat) (a1 : Nat), Keeps View (iq_image_unit a0 a1)
  iq_tag_read : ∀ (a0 : Nat) (a1 : Nat), Keeps View (iq_tag_read a0 a1)
  iq_tag_write : ∀ (a0 : Nat) (a1 : Nat) (a2 : Int), Keeps View (iq_tag_write a0 a1 a2)
  iq_unit_read : ∀ (a0 : Nat) (a1 : Nat), Keeps View (iq_unit_read a0 a1)
  iq_unit_write : ∀ (a0 : Nat) (a1 : Nat) (a2 : (BitVec 32)), Keeps View (iq_unit_write a0 a1 a2)
  observe_site : ∀ (a0 : String) (a1 : Nat) (a2 : Nat) (a3 : Nat) (a4 : Nat) (a5 : Int), Keeps View (observe_site a0 a1 a2 a3 a4 a5)
  read_sram_ecc : ∀ (a0 : mem_slice) (a1 : Nat), Keeps View (read_sram_ecc a0 a1)
  read_sram_word : ∀ (a0 : mem_slice) (a1 : Nat), Keeps View (read_sram_word a0 a1)
theorem throw_valid {α : Type} (e : Error exception) : Inv Valid (throw e : SailM α) := by
  intro s hs
  exact hs
theorem write_valid (r : Register) (v : RegisterType r) (hr:r≠Register.scrub_ptr) (hp:r≠Register.scrub_pending) : Inv Valid (Out.writeReg r v) := unrelated_valid r v hr hp
theorem ptr_zero : Inv Valid (Out.writeReg Register.scrub_ptr (vectorInit 0)) := by
  apply write_ptr_valid
  intro q hq
  simp [vectorInit,hq,Out.Functions.words_per_slice]
theorem repair_none : Inv Valid (Out.writeReg Register.scrub_pending (vectorInit none)) := by
  apply write_repair_valid
  intro q hq w hw
  simp [vectorInit,hq] at hw
attribute [local irreducible] ResourceInvariantProofs.Inv
attribute [local irreducible] GetElem?.getElem! Inhabited.default Bind.bind Pure.pure EStateM.bind EStateM.pure PreSail.readReg PreSail.writeReg IntRange.forIn'
attribute [local irreducible] Out.Functions.__id Out.Functions.abc_after Out.Functions.abc_at Out.Functions.abc_fp8 Out.Functions.abc_others Out.Functions.abc_others_fp8 Out.Functions.abc_pair_pick Out.Functions.abc_prune Out.Functions.abc_types_ok Out.Functions.acc_mode_name_backwards_matches Out.Functions.acc_mode_name_forwards Out.Functions.acc_mode_name_forwards_matches Out.Functions.acc_mode_of_num Out.Functions.acctype_declared Out.Functions.acctype_name_backwards_matches Out.Functions.acctype_name_forwards Out.Functions.acctype_name_forwards_matches Out.Functions.addr_bank Out.Functions.agt_advance Out.Functions.agt_advance_test Out.Functions.agt_after_reference Out.Functions.agt_config Out.Functions.agt_configure_pure Out.Functions.agt_due_pure Out.Functions.agt_field_of_num Out.Functions.agt_field_pos Out.Functions.agt_finish_config_pure Out.Functions.agt_nlevels Out.Functions.agt_opened Out.Functions.agt_refresh_pure Out.Functions.agt_state_of_num Out.Functions.aligned Out.Functions.all_ones Out.Functions.append_element Out.Functions.assembly_forwards_matches Out.Functions.bank_rule Out.Functions.banks_meet Out.Functions.bf16_dot Out.Functions.bfp8_exact Out.Functions.binop_entry_backwards_matches Out.Functions.binop_entry_forwards Out.Functions.binop_entry_forwards_matches Out.Functions.binop_legal Out.Functions.bit_index Out.Functions.bit_of Out.Functions.bits_of_s Out.Functions.bits_t Out.Functions.blk_exp Out.Functions.bool_bits Out.Functions.c2c_op_of_num Out.Functions.c2c_packet_flits Out.Functions.c2c_poison_int Out.Functions.c2c_ptype_code Out.Functions.c2c_ptype_of_num Out.Functions.c2c_ptype_queued Out.Functions.c2c_rx Out.Functions.c2c_tx Out.Functions.canon_nan Out.Functions.cfg_path_pick Out.Functions.chain_suffix_backwards_matches Out.Functions.chain_suffix_forwards Out.Functions.chain_suffix_forwards_matches Out.Functions.chk_elem Out.Functions.chk_wrow Out.Functions.clash_addr Out.Functions.clash_stream Out.Functions.classify_items Out.Functions.clears_repeat Out.Functions.cmp_legal Out.Functions.cmp_mnemonic_backwards_matches Out.Functions.cmp_mnemonic_forwards Out.Functions.cmp_mnemonic_forwards_matches Out.Functions.commit_scan Out.Functions.commit_wins Out.Functions.config_ends Out.Functions.config_for Out.Functions.conv_dst Out.Functions.conv_entry_backwards_matches Out.Functions.conv_entry_forwards Out.Functions.conv_entry_forwards_matches Out.Functions.conv_mnemonic_backwards_matches Out.Functions.conv_mnemonic_forwards Out.Functions.conv_mnemonic_forwards_matches Out.Functions.conv_src Out.Functions.conv_src_type Out.Functions.conv_types Out.Functions.csr_value Out.Functions.db_idx Out.Functions.dec_bits_10_backwards Out.Functions.dec_bits_10_backwards_matches Out.Functions.dec_bits_1_backwards Out.Functions.dec_bits_1_backwards_matches Out.Functions.dec_bits_2_backwards Out.Functions.dec_bits_2_backwards_matches Out.Functions.dec_bits_32_backwards Out.Functions.dec_bits_32_backwards_matches Out.Functions.dec_bits_3_backwards Out.Functions.dec_bits_3_backwards_matches Out.Functions.dec_bits_4_backwards Out.Functions.dec_bits_4_backwards_matches Out.Functions.dec_bits_5_backwards Out.Functions.dec_bits_5_backwards_matches Out.Functions.dec_bits_6_backwards Out.Functions.dec_bits_6_backwards_matches Out.Functions.dec_bits_7_backwards Out.Functions.dec_bits_7_backwards_matches Out.Functions.dec_bits_8_backwards Out.Functions.dec_bits_8_backwards_matches Out.Functions.dec_bits_9_backwards Out.Functions.dec_bits_9_backwards_matches Out.Functions.dec_bits_backwards Out.Functions.dec_bits_backwards_matches Out.Functions.dec_bits_forwards Out.Functions.dec_bits_forwards_matches Out.Functions.def_spc_backwards Out.Functions.def_spc_backwards_matches Out.Functions.def_spc_forwards Out.Functions.def_spc_forwards_matches Out.Functions.diag_code_backwards_matches Out.Functions.diag_code_forwards Out.Functions.diag_code_forwards_matches Out.Functions.diag_idx Out.Functions.diag_is_data Out.Functions.diag_number Out.Functions.diag_of_num Out.Functions.drop_commits Out.Functions.dst_split_backwards_matches Out.Functions.dst_split_forwards Out.Functions.dst_split_forwards_matches Out.Functions.dsum_round Out.Functions.dsum_round_u Out.Functions.dsum_wide Out.Functions.earliest Out.Functions.earliest_int Out.Functions.ecc_class_of_num Out.Functions.ecc_classify Out.Functions.ecc_correct Out.Functions.ecc_data_index Out.Functions.ecc_encode Out.Functions.ecc_flip_delta Out.Functions.ecc_is_pow2 Out.Functions.ecc_log2 Out.Functions.ecc_ov_get Out.Functions.ecc_ov_set Out.Functions.ecc_par10 Out.Functions.ecc_poison_ov Out.Functions.ecc_poison_sl Out.Functions.ecc_pos Out.Functions.elem Out.Functions.elem_at Out.Functions.eltype_name_backwards_matches Out.Functions.eltype_name_forwards Out.Functions.eltype_name_forwards_matches Out.Functions.eltype_of_num Out.Functions.eltype_width Out.Functions.empty_plane Out.Functions.encdec_acc_mode_backwards Out.Functions.encdec_acc_mode_backwards_matches Out.Functions.encdec_acc_mode_forwards Out.Functions.encdec_acc_mode_forwards_matches Out.Functions.encdec_acctype_backwards_matches Out.Functions.encdec_acctype_forwards Out.Functions.encdec_acctype_forwards_matches Out.Functions.encdec_c2c_op_backwards_matches Out.Functions.encdec_c2c_op_forwards Out.Functions.encdec_c2c_op_forwards_matches Out.Functions.encdec_cmp_mode_backwards_matches Out.Functions.encdec_cmp_mode_forwards Out.Functions.encdec_cmp_mode_forwards_matches Out.Functions.encdec_eltype_backwards_matches Out.Functions.encdec_eltype_forwards Out.Functions.encdec_eltype_forwards_matches Out.Functions.encdec_mapop_backwards_matches Out.Functions.encdec_mapop_forwards Out.Functions.encdec_mapop_forwards_matches Out.Functions.encdec_mem_dir_backwards_matches Out.Functions.encdec_mem_dir_forwards Out.Functions.encdec_mem_dir_forwards_matches Out.Functions.encdec_reduce_op_backwards_matches Out.Functions.encdec_reduce_op_forwards Out.Functions.encdec_reduce_op_forwards_matches Out.Functions.encdec_shift_dir_backwards Out.Functions.encdec_shift_dir_backwards_matches Out.Functions.encdec_shift_dir_forwards Out.Functions.encdec_shift_dir_forwards_matches Out.Functions.encdec_vxm_binop_backwards_matches Out.Functions.encdec_vxm_binop_forwards Out.Functions.encdec_vxm_binop_forwards_matches Out.Functions.encdec_vxm_conv_backwards_matches Out.Functions.encdec_vxm_conv_forwards Out.Functions.encdec_vxm_conv_forwards_matches Out.Functions.encdec_vxm_unop_backwards_matches Out.Functions.encdec_vxm_unop_forwards Out.Functions.encdec_vxm_unop_forwards_matches Out.Functions.entry_latency Out.Functions.entry_latency_typed Out.Functions.entry_of Out.Functions.execute_C2C_DESKEW Out.Functions.execute_C2C_DESKEWRECV Out.Functions.execute_C2C_HOSTWAIT Out.Functions.execute_C2C_RTDESKEW Out.Functions.execute_ILLEGAL32 Out.Functions.execute_ILLEGAL64 Out.Functions.execute_MEM_COUNTDOWN Out.Functions.execute_MEM_DEEPSLEEP Out.Functions.execute_MEM_ITERATIONS Out.Functions.execute_MEM_POWERCONFIG Out.Functions.execute_MEM_STEP Out.Functions.execute_MEM_STREAMSTEP Out.Functions.execute_NOP Out.Functions.execute_NOTIFY Out.Functions.execute_REPEAT Out.Functions.execute_SYNC Out.Functions.exp_iv Out.Functions.exp_small Out.Functions.extensions Out.Functions.f16_dot Out.Functions.f16_exact Out.Functions.f16_row Out.Functions.f16_row_bad Out.Functions.f16_stream_values Out.Functions.fadd_core Out.Functions.fclass_eq Out.Functions.fclass_of_num Out.Functions.fcmp Out.Functions.fdecode Out.Functions.fetch_profile_causal Out.Functions.fetch_tile_available Out.Functions.fetchm_bank_note Out.Functions.fetchm_hold_until Out.Functions.field Out.Functions.fill_append Out.Functions.fill_len Out.Functions.fill_push_unit Out.Functions.fill_units_append Out.Functions.find_override Out.Functions.flip96 Out.Functions.fmt_of Out.Functions.fp8_dot Out.Functions.fp8_exact Out.Functions.fp8_format_name_backwards_matches Out.Functions.fp8_format_name_forwards Out.Functions.fp8_format_name_forwards_matches Out.Functions.fp_abs Out.Functions.fp_add Out.Functions.fp_convert Out.Functions.fp_max Out.Functions.fp_min Out.Functions.fp_mul Out.Functions.fp_neg Out.Functions.fp_of_int Out.Functions.fp_sub Out.Functions.fp_tr_value Out.Functions.from_bytes_le Out.Functions.fround Out.Functions.fval_int Out.Functions.fx_of Out.Functions.gap_or_none Out.Functions.has_extension Out.Functions.hex_bits_10_backwards_matches Out.Functions.hex_bits_11_backwards_matches Out.Functions.hex_bits_12_backwards_matches Out.Functions.hex_bits_13_backwards_matches Out.Functions.hex_bits_14_backwards_matches Out.Functions.hex_bits_15_backwards_matches Out.Functions.hex_bits_16_backwards_matches Out.Functions.hex_bits_17_backwards_matches Out.Functions.hex_bits_18_backwards_matches Out.Functions.hex_bits_19_backwards_matches Out.Functions.hex_bits_1_backwards_matches Out.Functions.hex_bits_20_backwards_matches Out.Functions.hex_bits_21_backwards_matches Out.Functions.hex_bits_22_backwards_matches Out.Functions.hex_bits_23_backwards_matches Out.Functions.hex_bits_24_backwards_matches Out.Functions.hex_bits_25_backwards_matches Out.Functions.hex_bits_26_backwards_matches Out.Functions.hex_bits_27_backwards_matches Out.Functions.hex_bits_28_backwards_matches Out.Functions.hex_bits_29_backwards_matches Out.Functions.hex_bits_2_backwards_matches Out.Functions.hex_bits_30_backwards_matches Out.Functions.hex_bits_31_backwards_matches Out.Functions.hex_bits_32_backwards_matches Out.Functions.hex_bits_33_backwards_matches Out.Functions.hex_bits_34_backwards_matches Out.Functions.hex_bits_35_backwards_matches Out.Functions.hex_bits_36_backwards_matches Out.Functions.hex_bits_37_backwards_matches Out.Functions.hex_bits_38_backwards_matches Out.Functions.hex_bits_39_backwards_matches Out.Functions.hex_bits_3_backwards_matches Out.Functions.hex_bits_40_backwards_matches Out.Functions.hex_bits_41_backwards_matches Out.Functions.hex_bits_42_backwards_matches Out.Functions.hex_bits_43_backwards_matches Out.Functions.hex_bits_44_backwards_matches Out.Functions.hex_bits_45_backwards_matches Out.Functions.hex_bits_46_backwards_matches Out.Functions.hex_bits_47_backwards_matches Out.Functions.hex_bits_48_backwards_matches Out.Functions.hex_bits_49_backwards_matches Out.Functions.hex_bits_4_backwards_matches Out.Functions.hex_bits_50_backwards_matches Out.Functions.hex_bits_51_backwards_matches Out.Functions.hex_bits_52_backwards_matches Out.Functions.hex_bits_53_backwards_matches Out.Functions.hex_bits_54_backwards_matches Out.Functions.hex_bits_55_backwards_matches Out.Functions.hex_bits_56_backwards_matches Out.Functions.hex_bits_57_backwards_matches Out.Functions.hex_bits_58_backwards_matches Out.Functions.hex_bits_59_backwards_matches Out.Functions.hex_bits_5_backwards_matches Out.Functions.hex_bits_60_backwards_matches Out.Functions.hex_bits_61_backwards_matches Out.Functions.hex_bits_62_backwards_matches Out.Functions.hex_bits_63_backwards_matches Out.Functions.hex_bits_64_backwards_matches Out.Functions.hex_bits_6_backwards_matches Out.Functions.hex_bits_7_backwards_matches Out.Functions.hex_bits_8_backwards_matches Out.Functions.hex_bits_9_backwards_matches Out.Functions.hex_bits_backwards Out.Functions.hex_bits_backwards_matches Out.Functions.hex_bits_forwards Out.Functions.hex_bits_forwards_matches Out.Functions.hold_decide Out.Functions.holds_f16 Out.Functions.i8_tr_value Out.Functions.identity_map Out.Functions.idiv_ceil Out.Functions.idiv_floor Out.Functions.ilog2 Out.Functions.in_barrier_window Out.Functions.inf_bits Out.Functions.int8_dot Out.Functions.int_of Out.Functions.int_of_fp Out.Functions.inward_row Out.Functions.iq_padding Out.Functions.is_bool Out.Functions.is_dst Out.Functions.is_float Out.Functions.is_int Out.Functions.is_loopable Out.Functions.is_mxm_queue Out.Functions.is_nan Out.Functions.is_none Out.Functions.is_pair Out.Functions.is_prm Out.Functions.is_sel Out.Functions.is_shf Out.Functions.is_some Out.Functions.is_tsp Out.Functions.is_unsigned Out.Functions.isqrt Out.Functions.issued_by Out.Functions.issuer_of Out.Functions.items_len Out.Functions.k_binary Out.Functions.k_cmp Out.Functions.k_convert Out.Functions.k_convert_any Out.Functions.k_nonzero Out.Functions.k_unary Out.Functions.lane8 Out.Functions.lane_byte Out.Functions.latency_of Out.Functions.latency_of_instr Out.Functions.link_flit_mask Out.Functions.link_pow2 Out.Functions.lint Out.Functions.list_len Out.Functions.ln2_iv Out.Functions.log_iv Out.Functions.mapop_name_backwards_matches Out.Functions.mapop_name_forwards Out.Functions.mapop_name_forwards_matches Out.Functions.mem_dir_mnemonic_backwards_matches Out.Functions.mem_dir_mnemonic_forwards Out.Functions.mem_dir_mnemonic_forwards_matches Out.Functions.mem_quad Out.Functions.mem_slice_present Out.Functions.mem_unit Out.Functions.merge_tiles Out.Functions.model_link_flit_mask Out.Functions.model_link_flits Out.Functions.model_ptype_queued Out.Functions.mxm_acctype_of_num Out.Functions.mxm_bad Out.Functions.mxm_cap Out.Functions.mxm_check Out.Functions.mxm_clash Out.Functions.mxm_part_of_num Out.Functions.mxm_queue_present Out.Functions.mxm_unit Out.Functions.mxm_wtype_of_num Out.Functions.nbits_of Out.Functions.neq_bool Out.Functions.nonempty_list Out.Functions.not Out.Functions.not_bit Out.Functions.num_of_acc_mode Out.Functions.num_of_agt_field Out.Functions.num_of_agt_state Out.Functions.num_of_c2c_op Out.Functions.num_of_c2c_ptype Out.Functions.num_of_diag Out.Functions.num_of_ecc_class Out.Functions.num_of_eltype Out.Functions.num_of_fclass Out.Functions.num_of_mxm_acctype Out.Functions.num_of_mxm_part Out.Functions.num_of_mxm_wtype Out.Functions.num_of_row Out.Functions.num_of_scrub_action Out.Functions.num_of_srf_lookup Out.Functions.num_of_timing_class Out.Functions.num_of_unit_kind Out.Functions.one_plane Out.Functions.opt_spc_backwards Out.Functions.opt_spc_backwards_matches Out.Functions.opt_spc_forwards Out.Functions.opt_spc_forwards_matches Out.Functions.or_pair Out.Functions.outward_row Out.Functions.ov_nth Out.Functions.pair_exact Out.Functions.path_hold_until Out.Functions.path_views Out.Functions.place Out.Functions.place_all Out.Functions.poll_epoch Out.Functions.pow2i Out.Functions.q_in Out.Functions.queue_present Out.Functions.ras_idx Out.Functions.ras_loc Out.Functions.ras_sat_inc Out.Functions.read_gap Out.Functions.reduce_mnemonic_backwards_matches Out.Functions.reduce_mnemonic_forwards Out.Functions.reduce_mnemonic_forwards_matches Out.Functions.ref_bank Out.Functions.ref_path Out.Functions.ref_thread Out.Functions.rep_sp_set Out.Functions.repeat_after Out.Functions.repeat_plan Out.Functions.repeatable Out.Functions.rev_items Out.Functions.ring Out.Functions.rne_int Out.Functions.round_iv Out.Functions.row_of_num Out.Functions.rows_of Out.Functions.s_of_bits Out.Functions.sail_mask Out.Functions.sail_ones Out.Functions.same_target Out.Functions.sat_s Out.Functions.sat_t Out.Functions.scrub_action_of_num Out.Functions.scrub_arbitrate Out.Functions.scrub_due_count Out.Functions.scrub_slot_taken Out.Functions.sep_forwards Out.Functions.sep_forwards_matches Out.Functions.set_lane_byte Out.Functions.sg4_suffix_backwards_matches Out.Functions.sg4_suffix_forwards Out.Functions.sg4_suffix_forwards_matches Out.Functions.shift_dir_name_backwards_matches Out.Functions.shift_dir_name_forwards Out.Functions.shift_dir_name_forwards_matches Out.Functions.slice_mask Out.Functions.slot_of Out.Functions.spc_backwards Out.Functions.spc_backwards_matches Out.Functions.spc_forwards Out.Functions.spc_forwards_matches Out.Functions.sram_addr_of Out.Functions.sram_bank Out.Functions.sram_field_backwards Out.Functions.sram_field_backwards_matches Out.Functions.sram_field_forwards Out.Functions.sram_field_forwards_matches Out.Functions.sram_index Out.Functions.sram_text_backwards Out.Functions.sram_text_backwards_matches Out.Functions.sram_text_forwards Out.Functions.sram_text_forwards_matches Out.Functions.srf_lookup_of_num Out.Functions.srf_ov_spare Out.Functions.srf_rep_merge Out.Functions.srf_slot_best Out.Functions.srf_slot_chain_ov Out.Functions.srf_slot_collides Out.Functions.srf_slot_flip Out.Functions.srf_slot_read Out.Functions.srf_slot_read_ov Out.Functions.srf_slot_write Out.Functions.srf_stream_only_downstream Out.Functions.stream_id Out.Functions.stream_name_forwards_matches Out.Functions.stream_row Out.Functions.string_of_exception Out.Functions.sxm_bad Out.Functions.sxm_clash Out.Functions.sxm_lane Out.Functions.sxm_map_distinct Out.Functions.sxm_or_kind Out.Functions.sxm_set_lane Out.Functions.sxm_stream_plus Out.Functions.sxm_unit_at Out.Functions.tag_of Out.Functions.tags_clash Out.Functions.text_width Out.Functions.tile_active Out.Functions.tile_addr Out.Functions.timing_class_name Out.Functions.timing_class_of_num Out.Functions.timing_of Out.Functions.timing_overrides Out.Functions.timing_type_of Out.Functions.to_bytes_le Out.Functions.tr_eval Out.Functions.tr_eval_int Out.Functions.tr_interval Out.Functions.ty_split_backwards_matches Out.Functions.ty_split_forwards Out.Functions.ty_split_forwards_matches Out.Functions.ty_valid Out.Functions.unit_kind_of_num Out.Functions.unop_entry_backwards_matches Out.Functions.unop_entry_forwards Out.Functions.unop_entry_forwards_matches Out.Functions.unop_legal Out.Functions.vec_units Out.Functions.view_free Out.Functions.view_paths_busy Out.Functions.vsrc_name_forwards_matches Out.Functions.vxm_binop_mnemonic_backwards_matches Out.Functions.vxm_binop_mnemonic_forwards Out.Functions.vxm_binop_mnemonic_forwards_matches Out.Functions.vxm_unit Out.Functions.vxm_unop_mnemonic_backwards_matches Out.Functions.vxm_unop_mnemonic_forwards Out.Functions.vxm_unop_mnemonic_forwards_matches Out.Functions.wait_reason Out.Functions.wants_pair Out.Functions.wlane_add Out.Functions.wlane_read Out.Functions.words_to_planes Out.Functions.wrap_s Out.Functions.wrap_t Out.Functions.write_gap Out.Functions.writer_before Out.Functions.wtype_name_backwards_matches Out.Functions.wtype_name_forwards Out.Functions.wtype_name_forwards_matches Out.Functions.wtype_ok Out.Functions.wtype_split_backwards_matches Out.Functions.wtype_split_forwards Out.Functions.wtype_split_forwards_matches Out.Functions.wtype_valid Out.Functions.zero_bits Out.abs_real Out.add_real Out.diag_callback Out.div_real Out.ecc_event_callback Out.eq_real Out.gt_real Out.gteq_real Out.lint_callback Out.lt_real Out.lteq_real Out.mult_real Out.mxm_corr_callback Out.neg_real Out.prerr_real Out.print_real Out.random_real Out.real_power Out.round_down Out.round_up Out.scrub_event_callback Out.spare_event_callback Out.sqrt_real Out.sram_write_callback Out.srf_write_callback Out.sub_real Out.tdiv_int Out.tmod_int Out.to_real c2c_reset host_reset iq_push_tagged iq_reset masks_reset mem_reset mxm_reset srf_reset sxm_reset reset barrier_retire get_stream set_stream srf_write_ov commit_apply report commit_due barrier_notify barrier_sync iq_pop iq_unit_word iq_peek iq_unit_lane iq_word_ov iq_view iq_word_ov_set ras_ecc_event iq_word_check iq_text_check parcel_earliest decode_text iq_word_unc iq_text_unc iq_decode_at parcel_scan Out.Functions.control hold_view_of free_at iq_deliver enqueue execute_element execute_all execute_due diag_report iq_fetch_ov local_fetch_complete scrub_cycle scrub_run ecc_begin read_planes stream_fetch_take stream_fetch_service stream_fetch_complete agt_get agt_set agt_finish_config c2c_link deskewrecv_ready c2c_peer_ok deskewrecv_diag host_wait occupancy run_element dispatch_body dispatch_check reserve_local_fetch repeat_issue dispatch_decoded bank_report cfg_path dispatch_cfg loop_clash_ref due_ref due_refs hold_path loop_tag run_loop dispatch_slots dispatch host_parked held hold_check strict_held parcel_held polling queue_parked tick_queue mxm_apply mxm_apply_ok sxm_apply_ok unit_state_flush tick
theorem c2c_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (c2c_reset unitArg : SailM Unit) := by
  unfold c2c_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (host_reset unitArg : SailM Unit) := by
  unfold host_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_push_tagged_valid (ports : Ports) (q : Nat) (u : (BitVec 32)) (tag : Int) : Inv Valid (iq_push_tagged q u tag : SailM Bool) := by
  unfold iq_push_tagged
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact keeps_pred View ValidView _ (ports.iq_tag_write _ _ _)
    | exact keeps_pred View ValidView _ (ports.iq_unit_write _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (iq_reset unitArg : SailM Unit) := by
  unfold iq_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_push_tagged_valid ports _ _ _
    | exact keeps_pred View ValidView _ (ports.iq_image_len _)
    | exact keeps_pred View ValidView _ (ports.iq_image_unit _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem masks_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (masks_reset unitArg : SailM Unit) := by
  unfold masks_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mem_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (mem_reset unitArg : SailM Unit) := by
  unfold mem_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (mxm_reset unitArg : SailM Unit) := by
  unfold mxm_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (srf_reset unitArg : SailM Unit) := by
  unfold srf_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sxm_reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (sxm_reset unitArg : SailM Unit) := by
  unfold sxm_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem reset_valid (ports : Ports) (unitArg : Unit) : Inv Valid (reset unitArg : SailM Unit) := by
  unfold reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact c2c_reset_valid ports _
    | exact host_reset_valid ports _
    | exact iq_reset_valid ports _
    | exact masks_reset_valid ports _
    | exact mem_reset_valid ports _
    | exact mxm_reset_valid ports _
    | exact srf_reset_valid ports _
    | exact sxm_reset_valid ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_retire_valid (ports : Ports) (c : Nat) : Inv Valid (barrier_retire c : SailM Unit) := by
  unfold barrier_retire
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem get_stream_valid (ports : Ports) (r : row) (s : Nat) : Inv Valid (get_stream r s : SailM (Vector (Vector Out.srf_write 44) 64)) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.get_stream_frame ports.execution r s))
theorem set_stream_valid (ports : Ports) (r : row) (s : Nat) (v : (Vector (Vector Out.srf_write 44) 64)) : Inv Valid (set_stream r s v : SailM Unit) := by
  unfold set_stream
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_write_ov_valid (ports : Ports) (r : row) (s : Nat) (g : Nat) (c : Nat) (v : vec) (ov : ecc_ov) : Inv Valid (srf_write_ov r s g c v ov : SailM Bool) := by
  unfold srf_write_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact get_stream_valid ports _ _
    | exact set_stream_valid ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem commit_apply_valid (ports : Ports) (c : Nat) (all : (List commit)) (xs : (List commit)) (i : Nat) (hits : (Vector Nat 145)) : Inv Valid (commit_apply c all xs i hits : SailM (Vector Nat 145)) := by
  induction xs generalizing c all i hits
  case nil =>
    unfold commit_apply
    repeat' first
      | exact ih _ _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact srf_write_ov_valid ports _ _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold commit_apply
    repeat' first
      | exact ih _ _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact srf_write_ov_valid ports _ _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem report_valid (ports : Ports) (q : Nat) (c : Nat) (d : diag) : Inv Valid (report q c d : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.report_frame ports.execution q c d))
theorem commit_due_valid (ports : Ports) (c : Nat) : Inv Valid (commit_due c : SailM Unit) := by
  unfold commit_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact commit_apply_valid ports _ _ _ _ _
    | exact report_valid ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_notify_valid (ports : Ports) (n : Nat) : Inv Valid (barrier_notify n : SailM Unit) := by
  unfold barrier_notify
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_sync_valid (ports : Ports) (k : Nat) (c : Nat) : Inv Valid (barrier_sync k c : SailM Unit) := by
  unfold barrier_sync
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_pop_valid (ports : Ports) (q : Nat) (n : Nat) : Inv Valid (iq_pop q n : SailM Unit) := by
  unfold iq_pop
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_unit_word_valid (ports : Ports) (q : Nat) (i : Nat) : Inv Valid (iq_unit_word q i : SailM Int) := by
  unfold iq_unit_word
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact keeps_pred View ValidView _ (ports.iq_tag_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_peek_valid (ports : Ports) (q : Nat) (i : Nat) : Inv Valid (iq_peek q i : SailM (BitVec 32)) := by
  unfold iq_peek
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact keeps_pred View ValidView _ (ports.iq_unit_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_unit_lane_valid (ports : Ports) (q : Nat) (i : Nat) : Inv Valid (iq_unit_lane q i : SailM Nat) := by
  unfold iq_unit_lane
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact keeps_pred View ValidView _ (ports.iq_tag_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_word_ov_valid (ports : Ports) (q : Nat) (w : Nat) : Inv Valid (iq_word_ov q w : SailM (BitVec 10)) := by
  unfold iq_word_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_view_valid (ports : Ports) (q : Nat) (i : Nat) : Inv Valid (iq_view q i : SailM (BitVec 32)) := by
  unfold iq_view
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_peek_valid ports _ _
    | exact iq_unit_lane_valid ports _ _
    | exact iq_unit_word_valid ports _ _
    | exact iq_word_ov_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_word_ov_set_valid (ports : Ports) (q : Nat) (w : Nat) (e : (BitVec 10)) : Inv Valid (iq_word_ov_set q w e : SailM Unit) := by
  unfold iq_word_ov_set
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ras_ecc_event_valid (ports : Ports) (c : Nat) (q : Nat) (kind : Nat) (unc : Bool) (isw : Bool) (id : Nat) (addr : Nat) (sl : Nat) (pos : Nat) : Inv Valid (ras_ecc_event c q kind unc isw id addr sl pos : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.ras_ecc_event_frame ports.execution c q kind unc isw id addr sl pos))
theorem iq_word_check_valid (ports : Ports) (q : Nat) (c : Nat) (w : Int) : Inv Valid (iq_word_check q c w : SailM Unit) := by
  unfold iq_word_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_peek_valid ports _ _
    | exact iq_word_ov_valid ports _ _
    | exact iq_word_ov_set_valid ports _ _ _
    | exact ras_ecc_event_valid ports _ _ _ _ _ _ _ _ _
    | exact keeps_pred View ValidView _ (ports.iq_tag_read _ _)
    | exact keeps_pred View ValidView _ (ports.iq_unit_write _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_text_check_valid (ports : Ports) (q : Nat) (c : Nat) (units : Nat) : Inv Valid (iq_text_check q c units : SailM Unit) := by
  unfold iq_text_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_unit_word_valid ports _ _
    | exact iq_view_valid ports _ _
    | exact iq_word_check_valid ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parcel_earliest_valid (ports : Ports) (q : Nat) (p : parcel) : Inv Valid (parcel_earliest q p : SailM Int) := by
  unfold parcel_earliest
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem decode_text_valid (ports : Ports) (u0 : (BitVec 32)) (u1 : (BitVec 32)) : Inv Valid (decode_text u0 u1 : SailM (Result instruction diag)) := by
  unfold ResourceInvariantProofs.Inv
  intro state hs
  have h:=Proofs.ParcelFrame.decode_text_frame state u0 u1
  cases hx : (decode_text u0 u1).run state <;> simp only [Proofs.ParcelFrame.FrameAt,hx] at h <;> simpa [hx,h] using hs
theorem iq_word_unc_valid (ports : Ports) (q : Nat) (w : Int) : Inv Valid (iq_word_unc q w : SailM Bool) := by
  unfold iq_word_unc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_word_ov_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_text_unc_valid (ports : Ports) (q : Nat) (pos : Nat) (w : Nat) : Inv Valid (iq_text_unc q pos w : SailM Bool) := by
  unfold iq_text_unc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_unit_word_valid ports _ _
    | exact iq_word_unc_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_decode_at_valid (ports : Ports) (q : Nat) (pos : Nat) (w : Nat) : Inv Valid (iq_decode_at q pos w : SailM (Result instruction diag)) := by
  unfold iq_decode_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact decode_text_valid ports _ _
    | exact iq_text_unc_valid ports _ _ _
    | exact iq_view_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parcel_scan_valid (ports : Ports) (q : Nat) : Inv Valid (parcel_scan q : SailM parcel) := by
  unfold parcel_scan
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_decode_at_valid ports _ _ _
    | exact iq_view_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem control_valid (ports : Ports) (c : Nat) : Inv Valid (Out.Functions.control c : SailM Unit) := by
  unfold Out.Functions.control
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact barrier_notify_valid ports _
    | exact barrier_sync_valid ports _ _
    | exact iq_pop_valid ports _ _
    | exact iq_text_check_valid ports _ _ _
    | exact parcel_earliest_valid ports _ _
    | exact parcel_scan_valid ports _
    | exact report_valid ports _ _ _
    | exact keeps_pred View ValidView _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_view_of_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (hold_view_of q c : SailM hold_view) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.hold_view_of_frame ports.execution q c))
theorem free_at_valid (ports : Ports) (q : Nat) (t : Nat) : Inv Valid (free_at q t : SailM Bool) := by
  unfold free_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact hold_view_of_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_deliver_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (iq_deliver q c : SailM (Bool × Bool)) := by
  unfold iq_deliver
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact iq_push_tagged_valid ports _ _ _
    | exact iq_word_ov_set_valid ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem enqueue_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (enqueue q c : SailM Unit) := by
  unfold enqueue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact free_at_valid ports _ _
    | exact iq_deliver_valid ports _ _
    | exact parcel_earliest_valid ports _ _
    | exact parcel_scan_valid ports _
    | exact report_valid ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_element_valid (ports : Ports) (e : exec_element) : Inv Valid (execute_element e : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.execute_element_frame ports.execution e))
theorem execute_all_valid (ports : Ports) (xs : (List exec_element)) : Inv Valid (execute_all xs : SailM Unit) := by
  induction xs
  case nil =>
    unfold execute_all
    repeat' first
      | exact ih 
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact execute_element_valid ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold execute_all
    repeat' first
      | exact ih 
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact execute_element_valid ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem execute_due_valid (ports : Ports) (c : Nat) : Inv Valid (execute_due c : SailM Unit) := by
  unfold execute_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact execute_all_valid ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem diag_report_valid (ports : Ports) (c : Nat) (q : Nat) (d : diag) : Inv Valid (diag_report c q d : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.diag_report_frame ports.execution c q d))
theorem iq_fetch_ov_valid (ports : Ports) (q : Nat) (t : Nat) (a : vec) (b : vec) (ova : ecc_ov) (ovb : ecc_ov) : Inv Valid (iq_fetch_ov q t a b ova ovb : SailM Bool) := by
  unfold iq_fetch_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact keeps_pred View ValidView _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem local_fetch_complete_valid (ports : Ports) (c : Nat) : Inv Valid (local_fetch_complete c : SailM Unit) := by
  unfold local_fetch_complete
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact diag_report_valid ports _ _ _
    | exact iq_fetch_ov_valid ports _ _ _ _ _ _
    | exact keeps_pred View ValidView _ (ports.read_sram_ecc _ _)
    | exact keeps_pred View ValidView _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_cycle_valid (ports : Ports) (m : Nat) : Inv Valid (scrub_cycle m : SailM Unit) := by
  exact actual_cycle_valid m (ports.patrol m)
theorem scrub_run_valid (ports : Ports) (upto : Int) : Inv Valid (scrub_run upto : SailM Unit) := by
  unfold scrub_run
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact scrub_cycle_valid ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ecc_begin_valid (ports : Ports) (unitArg : Unit) : Inv Valid (ecc_begin unitArg : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.ecc_begin_frame ports.execution unitArg))
theorem read_planes_valid (ports : Ports) (s : streamref) (w : Nat) (t : Int) : Inv Valid (read_planes s w t : SailM (Result (Vector vec 4) diag)) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.read_planes_frame ports.execution s w t))
theorem stream_fetch_take_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (stream_fetch_take q c : SailM (Option stream_fetch)) := by
  unfold stream_fetch_take
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_service_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (stream_fetch_service q c : SailM Unit) := by
  unfold stream_fetch_service
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact diag_report_valid ports _ _ _
    | exact ecc_begin_valid ports _
    | exact iq_fetch_ov_valid ports _ _ _ _ _ _
    | exact read_planes_valid ports _ _ _
    | exact stream_fetch_take_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_complete_valid (ports : Ports) (c : Nat) : Inv Valid (stream_fetch_complete c : SailM Unit) := by
  unfold stream_fetch_complete
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact stream_fetch_service_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_get_valid (ports : Ports) (q : Nat) (p : Nat) : Inv Valid (agt_get q p : SailM agt) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.agt_get_frame ports.execution q p))
theorem agt_set_valid (ports : Ports) (q : Nat) (p : Nat) (a : agt) : Inv Valid (agt_set q p a : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.agt_set_frame ports.execution q p a))
theorem agt_finish_config_valid (ports : Ports) (q : Nat) (p : Nat) : Inv Valid (agt_finish_config q p : SailM Unit) := by
  unfold agt_finish_config
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact agt_get_valid ports _ _
    | exact agt_set_valid ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem c2c_link_valid (ports : Ports) (link : (BitVec 7)) : Inv Valid (c2c_link link : SailM (Option Nat)) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.c2c_link_frame ports.execution link))
theorem deskewrecv_ready_valid (ports : Ports) (c : Nat) (link : (BitVec 7)) : Inv Valid (deskewrecv_ready c link : SailM Bool) := by
  unfold deskewrecv_ready
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact c2c_link_valid ports _
    | exact keeps_pred View ValidView _ (ports.c2c_recv_ready _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem c2c_peer_ok_valid (ports : Ports) (chip : (BitVec 9)) (l : Nat) : Inv Valid (c2c_peer_ok chip l : SailM Bool) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.c2c_peer_ok_frame ports.execution chip l))
theorem deskewrecv_diag_valid (ports : Ports) (iu : issuer) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Inv Valid (deskewrecv_diag iu chip link s : SailM (Option diag)) := by
  unfold deskewrecv_diag
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact c2c_link_valid ports _
    | exact c2c_peer_ok_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_wait_valid (ports : Ports) (q : Nat) (c : Nat) (i : Nat) : Inv Valid (host_wait q c i : SailM Unit) := by
  unfold host_wait
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem occupancy_valid (ports : Ports) (arg0 : instruction) (arg1 : Nat) : Inv Valid (occupancy arg0 arg1 : SailM Nat) := by
  unfold occupancy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem run_element_valid (ports : Ports) (q : Nat) (i : instruction) (c0 : Nat) (k : Nat) (off : Int) (soff : Int) : Inv Valid (run_element q i c0 k off soff : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.run_element_frame ports.execution q i c0 k off soff))
theorem dispatch_body_valid (ports : Ports) (q : Nat) (c : Nat) (i : instruction) : Inv Valid (dispatch_body q c i : SailM Unit) := by
  unfold dispatch_body
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact barrier_notify_valid ports _
    | exact barrier_sync_valid ports _ _
    | exact deskewrecv_diag_valid ports _ _ _ _
    | exact host_wait_valid ports _ _ _
    | exact occupancy_valid ports _ _
    | exact report_valid ports _ _ _
    | exact run_element_valid ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_check_valid (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) : Inv Valid (dispatch_check q c i path : SailM (Option diag)) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.dispatch_check_frame ports.execution q c i path))
theorem reserve_local_fetch_valid (ports : Ports) (q : Nat) (i : instruction) (c : Nat) : Inv Valid (reserve_local_fetch q i c : SailM Unit) := by
  unfold reserve_local_fetch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem repeat_issue_valid (ports : Ports) (q : Nat) (i : instruction) (c : Nat) : Inv Valid (repeat_issue q i c : SailM Unit) := by
  unfold repeat_issue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact reserve_local_fetch_valid ports _ _ _
    | exact run_element_valid ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_decoded_valid (ports : Ports) (q : Nat) (c : Nat) (i : instruction) : Inv Valid (dispatch_decoded q c i : SailM Unit) := by
  unfold dispatch_decoded
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact dispatch_body_valid ports _ _ _
    | exact dispatch_check_valid ports _ _ _ _
    | exact repeat_issue_valid ports _ _ _
    | exact report_valid ports _ _ _
    | exact keeps_pred View ValidView _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem bank_report_valid (ports : Ports) (q : Nat) (c : Nat) (a : (instruction × Int × Int)) (b : (instruction × Int × Int)) : Inv Valid (bank_report q c a b : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.bank_report_frame ports.execution q c a b))
theorem cfg_path_valid (ports : Ports) (q : Nat) : Inv Valid (cfg_path q : SailM Nat) := by
  unfold cfg_path
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_cfg_valid (ports : Ports) (q : Nat) (c : Nat) (p : Nat) (i : instruction) : Inv Valid (dispatch_cfg q c p i : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.dispatch_cfg_frame ports.execution q c p i))
theorem loop_clash_ref_valid (ports : Ports) (q : Nat) (p : Nat) (s : Int) (ss : Int) : Inv Valid (loop_clash_ref q p s ss : SailM (instruction × Int × Int)) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.loop_clash_ref_frame ports.execution q p s ss))
theorem due_ref_valid (ports : Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Inv Valid (due_ref q p st : SailM (Option (instruction × Int × Int))) := by
  unfold due_ref
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact loop_clash_ref_valid ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem due_refs_valid (ports : Ports) (q : Nat) (c : Nat) (parked : Bool) : Inv Valid (due_refs q c parked : SailM (Vector (Option ((Vector agt_level 4) × Nat × Int × Int)) 2)) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.due_refs_frame ports.execution q c parked))
theorem hold_path_valid (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv Valid (hold_path q p c : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.hold_path_frame ports.execution q p c))
theorem loop_tag_valid (ports : Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Inv Valid (loop_tag q p st : SailM (Option ref_tag)) := by
  unfold loop_tag
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact agt_get_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem run_loop_valid (ports : Ports) (q : Nat) (c : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) (blocked : Bool) (partner : (Option (instruction × Int × Int))) : Inv Valid (run_loop q c p st blocked partner : SailM Unit) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.run_loop_frame ports.execution q c p st blocked partner))
theorem consumeNewRef_valid (ports : Ports) (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) : Inv Valid (GrantConsumptionProofs.consumeNewRef q c r loop lref go bank k issued last) := by
  unfold GrantConsumptionProofs.consumeNewRef
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact agt_set_valid ports _ _ _
    | exact bank_report_valid ports _ _ _ _
    | exact cfg_path_valid ports _
    | exact dispatch_cfg_valid ports _ _ _ _
    | exact dispatch_check_valid ports _ _ _ _
    | exact due_ref_valid ports _ _ _
    | exact due_refs_valid ports _ _ _
    | exact hold_path_valid ports _ _ _
    | exact loop_tag_valid ports _ _ _
    | exact report_valid ports _ _ _
    | exact reserve_local_fetch_valid ports _ _ _
    | exact run_element_valid ports _ _ _ _ _ _
    | exact run_loop_valid ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem consumeFetch_valid (ports : Ports) (q c : Nat) (r : cycle_recs) (fm_ok fm_clash : Bool)
    (issued : Nat) (last : Option instruction) : Inv Valid (GrantConsumptionProofs.consumeFetch q c r fm_ok fm_clash issued last) := by
  unfold GrantConsumptionProofs.consumeFetch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact agt_set_valid ports _ _ _
    | exact bank_report_valid ports _ _ _ _
    | exact cfg_path_valid ports _
    | exact dispatch_cfg_valid ports _ _ _ _
    | exact dispatch_check_valid ports _ _ _ _
    | exact due_ref_valid ports _ _ _
    | exact due_refs_valid ports _ _ _
    | exact hold_path_valid ports _ _ _
    | exact loop_tag_valid ports _ _ _
    | exact report_valid ports _ _ _
    | exact reserve_local_fetch_valid ports _ _ _
    | exact run_element_valid ports _ _ _ _ _ _
    | exact run_loop_valid ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro

theorem dispatch_slots_valid (ports : Ports) (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : Inv Valid (dispatch_slots q c r parked : SailM Unit) := by
  rw [GrantConsumptionProofs.actual_dispatch_factory_source]
  unfold GrantConsumptionProofs.dispatchFactory
  repeat' first
    | exact consumeNewRef_valid ports _ _ _ _ _ _ _ _ _ _
    | exact consumeFetch_valid ports _ _ _ _ _ _ _
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact agt_set_valid ports _ _ _
    | exact bank_report_valid ports _ _ _ _
    | exact cfg_path_valid ports _
    | exact dispatch_cfg_valid ports _ _ _ _
    | exact dispatch_check_valid ports _ _ _ _
    | exact due_ref_valid ports _ _ _
    | exact due_refs_valid ports _ _ _
    | exact hold_path_valid ports _ _ _
    | exact loop_tag_valid ports _ _ _
    | exact report_valid ports _ _ _
    | exact reserve_local_fetch_valid ports _ _ _
    | exact run_element_valid ports _ _ _ _ _ _
    | exact run_loop_valid ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_valid (ports : Ports) (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : Inv Valid (dispatch q c r parked : SailM Unit) := by
  unfold dispatch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact dispatch_decoded_valid ports _ _ _
    | exact dispatch_slots_valid ports _ _ _ _
    | exact due_ref_valid ports _ _ _
    | exact due_refs_valid ports _ _ _
    | exact loop_tag_valid ports _ _ _
    | exact report_valid ports _ _ _
    | exact run_loop_valid ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_parked_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (host_parked q c : SailM Bool) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.host_parked_frame ports.execution q c))
theorem held_valid (ports : Ports) (q : Nat) : Inv Valid (held q : SailM Bool) := by
  unfold held
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_check_valid (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) (strict : Bool) : Inv Valid (hold_check q c i path strict : SailM Bool) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.hold_check_frame ports.execution q c i path strict))
theorem strict_held_valid (ports : Ports) (q : Nat) (c : Nat) (xs : (List (Result instruction diag))) (held : Bool) : Inv Valid (strict_held q c xs held : SailM Bool) := by
  induction xs generalizing q c held
  case nil =>
    unfold strict_held
    repeat' first
      | exact ih _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact hold_check_valid ports _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold strict_held
    repeat' first
      | exact ih _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact hold_check_valid ports _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem parcel_held_valid (ports : Ports) (q : Nat) (c : Nat) (r : cycle_recs) (items : (List (Result instruction diag))) : Inv Valid (parcel_held q c r items : SailM Bool) := by
  unfold parcel_held
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact cfg_path_valid ports _
    | exact held_valid ports _
    | exact hold_check_valid ports _ _ _ _ _
    | exact strict_held_valid ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem polling_valid (ports : Ports) (q : Nat) : Inv Valid (polling q : SailM Bool) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.polling_frame ports.execution q))
theorem queue_parked_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (queue_parked q c : SailM Bool) := by
  exact keeps_pred View ValidView _ (ResourceExecuteFrameProofs.resource_projection _ (ResourceExecuteFrameProofs.queue_parked_frame ports.execution q c))
theorem tick_queue_valid (ports : Ports) (q : Nat) (c : Nat) : Inv Valid (tick_queue q c : SailM Unit) := by
  unfold tick_queue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact agt_finish_config_valid ports _ _
    | exact agt_get_valid ports _ _
    | exact deskewrecv_ready_valid ports _ _
    | exact dispatch_valid ports _ _ _ _
    | exact host_parked_valid ports _ _
    | exact iq_pop_valid ports _ _
    | exact iq_text_check_valid ports _ _ _
    | exact occupancy_valid ports _ _
    | exact parcel_earliest_valid ports _ _
    | exact parcel_held_valid ports _ _ _ _
    | exact parcel_scan_valid ports _
    | exact polling_valid ports _
    | exact queue_parked_valid ports _ _
    | exact repeat_issue_valid ports _ _ _
    | exact run_element_valid ports _ _ _ _ _ _
    | exact keeps_pred View ValidView _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_valid (ports : Ports) (e : mxm_staged) : Inv Valid (mxm_apply e : SailM Unit) := by
  unfold mxm_apply
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_ok_valid (ports : Ports) (xs : (List mxm_staged)) (bad : (List Nat)) : Inv Valid (mxm_apply_ok xs bad : SailM Unit) := by
  induction xs generalizing bad
  case nil =>
    unfold mxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact mxm_apply_valid ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold mxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | exact mxm_apply_valid ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem sxm_apply_ok_valid (ports : Ports) (xs : (List sxm_staged)) (bad : (List Nat)) : Inv Valid (sxm_apply_ok xs bad : SailM Unit) := by
  induction xs generalizing bad
  case nil =>
    unfold sxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold sxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_valid _
      | exact read_inv _ _
      | exact write_valid _ _ (by decide) (by decide)
      | exact ptr_zero
      | exact repair_none
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem unit_state_flush_valid (ports : Ports) (c : Nat) : Inv Valid (unit_state_flush c : SailM Unit) := by
  unfold unit_state_flush
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact mxm_apply_ok_valid ports _ _
    | exact report_valid ports _ _ _
    | exact sxm_apply_ok_valid ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem tick_valid (ports : Ports) (c : Nat) : Inv Valid (tick c : SailM Unit) := by
  unfold tick
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_valid _
    | exact read_inv _ _
    | exact write_valid _ _ (by decide) (by decide)
    | exact ptr_zero
    | exact repair_none
    | exact barrier_retire_valid ports _
    | exact commit_due_valid ports _
    | exact control_valid ports _
    | exact enqueue_valid ports _ _
    | exact execute_due_valid ports _
    | exact local_fetch_complete_valid ports _
    | exact scrub_run_valid ports _
    | exact stream_fetch_complete_valid ports _
    | exact tick_queue_valid ports _ _
    | exact unit_state_flush_valid ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
end ResourceReachProofs
