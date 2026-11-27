-- Actual full reset/tick caller composition for AGT time bounds.
import Proofs.AgtTimeBound
import Proofs.GrantConsumption
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 100000
namespace AgtTimeClosureProofs
open ResourceFrameProofs ResourceInvariantProofs AgtTimeBoundProofs
variable {cycle:Nat}
abbrev Ports:=ResourceCompleteProofs.Ports
theorem registers_bound {α:Type} (action:SailM α)
 (h:Keeps (fun st:ResourceFrameProofs.St=>st.regs) action):Inv (Bound cycle) action:=
 keeps_pred (fun st:ResourceFrameProofs.St=>st.regs)
  (fun regs=>BoundView cycle (regs.get? Register.agts,regs.get? Register.agts_w)) action h

def execution_ports (ports:Ports):AgtExecutionFrameProofs.Ports where
 c2c_packet_busy a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.c2c_packet_busy a0 a1)
 c2c_peer a0:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.c2c_peer a0)
 c2c_recv a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.c2c_recv a0 a1)
 c2c_recv_bad a0:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.c2c_recv_bad a0)
 c2c_recv_poison a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.c2c_recv_poison a0 a1)
 c2c_recv_ready a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.c2c_recv_ready a0 a1)
 c2c_send a0 a1 a2 a3 a4:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.c2c_send a0 a1 a2 a3 a4)
 host_irq_callback a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.host_irq_callback a0 a1)
 host_recv a0:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.host_recv a0)
 host_recv_ready a0:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.host_recv_ready a0)
 host_send a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.host_send a0 a1)
 observe_site a0 a1 a2 a3 a4 a5:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.observe_site a0 a1 a2 a3 a4 a5)
 raw_write_sram_ecc a0 a1 a2:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.raw_write_sram_ecc a0 a1 a2)
 raw_write_sram_word a0 a1 a2:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.raw_write_sram_word a0 a1 a2)
 read_sram_ecc a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.read_sram_ecc a0 a1)
 read_sram_word a0 a1:=PatrolProtectedProofs.map_keeps (fun st:ResourceFrameProofs.St=>st.regs) (fun regs=>(regs.get? Register.agts,regs.get? Register.agts_w)) _ (ports.read_sram_word a0 a1)
theorem throw_bound {α:Type} (e:Error exception):Inv (Bound cycle) (throw e:SailM α):=by intro st hs;exact hs

theorem assert_bound (p:Bool) (message:String):Inv (Bound cycle) (Sail.ConcurrencyInterfaceV1.PreSail.assert p message:SailM Unit):=by cases p <;> intro st hs <;> exact hs

theorem actual_initialized_bound (st:ResourceFrameProofs.St):
 ∃after,(sail_model_init ()).run st=.ok () after ∧ Bound cycle after:=by
 refine ⟨_,rfl,?_⟩
 constructor
 all_goals intro xs hx
 all_goals simp [AgtExecutionFrameProofs.RView,Std.ExtDHashMap.get?_insert] at hx
 all_goals cases hx
 all_goals exact array_idle_at

attribute [local irreducible] ResourceInvariantProofs.Inv
attribute [local irreducible] GetElem?.getElem! Inhabited.default Bind.bind Pure.pure EStateM.bind EStateM.pure PreSail.readReg PreSail.writeReg IntRange.forIn'
attribute [local irreducible] Out.Functions.__id Out.Functions.abc_after Out.Functions.abc_at Out.Functions.abc_fp8 Out.Functions.abc_others Out.Functions.abc_others_fp8 Out.Functions.abc_pair_pick Out.Functions.abc_prune Out.Functions.abc_types_ok Out.Functions.acc_mode_name_backwards_matches Out.Functions.acc_mode_name_forwards Out.Functions.acc_mode_name_forwards_matches Out.Functions.acc_mode_of_num Out.Functions.acctype_declared Out.Functions.acctype_name_backwards_matches Out.Functions.acctype_name_forwards Out.Functions.acctype_name_forwards_matches Out.Functions.addr_bank Out.Functions.agt_advance Out.Functions.agt_advance_test Out.Functions.agt_after_reference Out.Functions.agt_config Out.Functions.agt_configure_pure Out.Functions.agt_due_pure Out.Functions.agt_field_of_num Out.Functions.agt_field_pos Out.Functions.agt_finish_config_pure Out.Functions.agt_nlevels Out.Functions.agt_opened Out.Functions.agt_refresh_pure Out.Functions.agt_state_of_num Out.Functions.aligned Out.Functions.all_ones Out.Functions.append_element Out.Functions.assembly_forwards_matches Out.Functions.bank_rule Out.Functions.banks_meet Out.Functions.bf16_dot Out.Functions.bfp8_exact Out.Functions.binop_entry_backwards_matches Out.Functions.binop_entry_forwards Out.Functions.binop_entry_forwards_matches Out.Functions.binop_legal Out.Functions.bit_index Out.Functions.bit_of Out.Functions.bits_of_s Out.Functions.bits_t Out.Functions.blk_exp Out.Functions.bool_bits Out.Functions.c2c_op_of_num Out.Functions.c2c_packet_flits Out.Functions.c2c_poison_int Out.Functions.c2c_ptype_code Out.Functions.c2c_ptype_of_num Out.Functions.c2c_ptype_queued Out.Functions.c2c_rx Out.Functions.c2c_tx Out.Functions.canon_nan Out.Functions.cfg_path_pick Out.Functions.chain_suffix_backwards_matches Out.Functions.chain_suffix_forwards Out.Functions.chain_suffix_forwards_matches Out.Functions.chk_elem Out.Functions.chk_wrow Out.Functions.clash_addr Out.Functions.clash_stream Out.Functions.classify_items Out.Functions.clears_repeat Out.Functions.cmp_legal Out.Functions.cmp_mnemonic_backwards_matches Out.Functions.cmp_mnemonic_forwards Out.Functions.cmp_mnemonic_forwards_matches Out.Functions.commit_scan Out.Functions.commit_wins Out.Functions.config_ends Out.Functions.config_for Out.Functions.conv_dst Out.Functions.conv_entry_backwards_matches Out.Functions.conv_entry_forwards Out.Functions.conv_entry_forwards_matches Out.Functions.conv_mnemonic_backwards_matches Out.Functions.conv_mnemonic_forwards Out.Functions.conv_mnemonic_forwards_matches Out.Functions.conv_src Out.Functions.conv_src_type Out.Functions.conv_types Out.Functions.csr_value Out.Functions.db_idx Out.Functions.dec_bits_10_backwards Out.Functions.dec_bits_10_backwards_matches Out.Functions.dec_bits_1_backwards Out.Functions.dec_bits_1_backwards_matches Out.Functions.dec_bits_2_backwards Out.Functions.dec_bits_2_backwards_matches Out.Functions.dec_bits_32_backwards Out.Functions.dec_bits_32_backwards_matches Out.Functions.dec_bits_3_backwards Out.Functions.dec_bits_3_backwards_matches Out.Functions.dec_bits_4_backwards Out.Functions.dec_bits_4_backwards_matches Out.Functions.dec_bits_5_backwards Out.Functions.dec_bits_5_backwards_matches Out.Functions.dec_bits_6_backwards Out.Functions.dec_bits_6_backwards_matches Out.Functions.dec_bits_7_backwards Out.Functions.dec_bits_7_backwards_matches Out.Functions.dec_bits_8_backwards Out.Functions.dec_bits_8_backwards_matches Out.Functions.dec_bits_9_backwards Out.Functions.dec_bits_9_backwards_matches Out.Functions.dec_bits_backwards Out.Functions.dec_bits_backwards_matches Out.Functions.dec_bits_forwards Out.Functions.dec_bits_forwards_matches Out.Functions.def_spc_backwards Out.Functions.def_spc_backwards_matches Out.Functions.def_spc_forwards Out.Functions.def_spc_forwards_matches Out.Functions.diag_code_backwards_matches Out.Functions.diag_code_forwards Out.Functions.diag_code_forwards_matches Out.Functions.diag_idx Out.Functions.diag_is_data Out.Functions.diag_number Out.Functions.diag_of_num Out.Functions.drop_commits Out.Functions.dst_split_backwards_matches Out.Functions.dst_split_forwards Out.Functions.dst_split_forwards_matches Out.Functions.dsum_round Out.Functions.dsum_round_u Out.Functions.dsum_wide Out.Functions.earliest Out.Functions.earliest_int Out.Functions.ecc_class_of_num Out.Functions.ecc_classify Out.Functions.ecc_correct Out.Functions.ecc_data_index Out.Functions.ecc_encode Out.Functions.ecc_flip_delta Out.Functions.ecc_is_pow2 Out.Functions.ecc_log2 Out.Functions.ecc_ov_get Out.Functions.ecc_ov_set Out.Functions.ecc_par10 Out.Functions.ecc_poison_ov Out.Functions.ecc_poison_sl Out.Functions.ecc_pos Out.Functions.elem Out.Functions.elem_at Out.Functions.eltype_name_backwards_matches Out.Functions.eltype_name_forwards Out.Functions.eltype_name_forwards_matches Out.Functions.eltype_of_num Out.Functions.eltype_width Out.Functions.empty_plane Out.Functions.encdec_acc_mode_backwards Out.Functions.encdec_acc_mode_backwards_matches Out.Functions.encdec_acc_mode_forwards Out.Functions.encdec_acc_mode_forwards_matches Out.Functions.encdec_acctype_backwards_matches Out.Functions.encdec_acctype_forwards Out.Functions.encdec_acctype_forwards_matches Out.Functions.encdec_c2c_op_backwards_matches Out.Functions.encdec_c2c_op_forwards Out.Functions.encdec_c2c_op_forwards_matches Out.Functions.encdec_cmp_mode_backwards_matches Out.Functions.encdec_cmp_mode_forwards Out.Functions.encdec_cmp_mode_forwards_matches Out.Functions.encdec_eltype_backwards_matches Out.Functions.encdec_eltype_forwards Out.Functions.encdec_eltype_forwards_matches Out.Functions.encdec_mapop_backwards_matches Out.Functions.encdec_mapop_forwards Out.Functions.encdec_mapop_forwards_matches Out.Functions.encdec_mem_dir_backwards_matches Out.Functions.encdec_mem_dir_forwards Out.Functions.encdec_mem_dir_forwards_matches Out.Functions.encdec_reduce_op_backwards_matches Out.Functions.encdec_reduce_op_forwards Out.Functions.encdec_reduce_op_forwards_matches Out.Functions.encdec_shift_dir_backwards Out.Functions.encdec_shift_dir_backwards_matches Out.Functions.encdec_shift_dir_forwards Out.Functions.encdec_shift_dir_forwards_matches Out.Functions.encdec_vxm_binop_backwards_matches Out.Functions.encdec_vxm_binop_forwards Out.Functions.encdec_vxm_binop_forwards_matches Out.Functions.encdec_vxm_conv_backwards_matches Out.Functions.encdec_vxm_conv_forwards Out.Functions.encdec_vxm_conv_forwards_matches Out.Functions.encdec_vxm_unop_backwards_matches Out.Functions.encdec_vxm_unop_forwards Out.Functions.encdec_vxm_unop_forwards_matches Out.Functions.entry_latency Out.Functions.entry_latency_typed Out.Functions.entry_of Out.Functions.execute_C2C_DESKEW Out.Functions.execute_C2C_DESKEWRECV Out.Functions.execute_C2C_HOSTWAIT Out.Functions.execute_C2C_RTDESKEW Out.Functions.execute_ILLEGAL32 Out.Functions.execute_ILLEGAL64 Out.Functions.execute_MEM_COUNTDOWN Out.Functions.execute_MEM_DEEPSLEEP Out.Functions.execute_MEM_ITERATIONS Out.Functions.execute_MEM_POWERCONFIG Out.Functions.execute_MEM_STEP Out.Functions.execute_MEM_STREAMSTEP Out.Functions.execute_NOP Out.Functions.execute_NOTIFY Out.Functions.execute_REPEAT Out.Functions.execute_SYNC Out.Functions.exp_iv Out.Functions.exp_small Out.Functions.extensions Out.Functions.f16_dot Out.Functions.f16_exact Out.Functions.f16_row Out.Functions.f16_row_bad Out.Functions.f16_stream_values Out.Functions.fadd_core Out.Functions.fclass_eq Out.Functions.fclass_of_num Out.Functions.fcmp Out.Functions.fdecode Out.Functions.fetch_profile_causal Out.Functions.fetch_tile_available Out.Functions.fetchm_bank_note Out.Functions.fetchm_hold_until Out.Functions.field Out.Functions.fill_append Out.Functions.fill_len Out.Functions.fill_push_unit Out.Functions.fill_units_append Out.Functions.find_override Out.Functions.flip96 Out.Functions.fmt_of Out.Functions.fp8_dot Out.Functions.fp8_exact Out.Functions.fp8_format_name_backwards_matches Out.Functions.fp8_format_name_forwards Out.Functions.fp8_format_name_forwards_matches Out.Functions.fp_abs Out.Functions.fp_add Out.Functions.fp_convert Out.Functions.fp_max Out.Functions.fp_min Out.Functions.fp_mul Out.Functions.fp_neg Out.Functions.fp_of_int Out.Functions.fp_sub Out.Functions.fp_tr_value Out.Functions.from_bytes_le Out.Functions.fround Out.Functions.fval_int Out.Functions.fx_of Out.Functions.gap_or_none Out.Functions.has_extension Out.Functions.hex_bits_10_backwards_matches Out.Functions.hex_bits_11_backwards_matches Out.Functions.hex_bits_12_backwards_matches Out.Functions.hex_bits_13_backwards_matches Out.Functions.hex_bits_14_backwards_matches Out.Functions.hex_bits_15_backwards_matches Out.Functions.hex_bits_16_backwards_matches Out.Functions.hex_bits_17_backwards_matches Out.Functions.hex_bits_18_backwards_matches Out.Functions.hex_bits_19_backwards_matches Out.Functions.hex_bits_1_backwards_matches Out.Functions.hex_bits_20_backwards_matches Out.Functions.hex_bits_21_backwards_matches Out.Functions.hex_bits_22_backwards_matches Out.Functions.hex_bits_23_backwards_matches Out.Functions.hex_bits_24_backwards_matches Out.Functions.hex_bits_25_backwards_matches Out.Functions.hex_bits_26_backwards_matches Out.Functions.hex_bits_27_backwards_matches Out.Functions.hex_bits_28_backwards_matches Out.Functions.hex_bits_29_backwards_matches Out.Functions.hex_bits_2_backwards_matches Out.Functions.hex_bits_30_backwards_matches Out.Functions.hex_bits_31_backwards_matches Out.Functions.hex_bits_32_backwards_matches Out.Functions.hex_bits_33_backwards_matches Out.Functions.hex_bits_34_backwards_matches Out.Functions.hex_bits_35_backwards_matches Out.Functions.hex_bits_36_backwards_matches Out.Functions.hex_bits_37_backwards_matches Out.Functions.hex_bits_38_backwards_matches Out.Functions.hex_bits_39_backwards_matches Out.Functions.hex_bits_3_backwards_matches Out.Functions.hex_bits_40_backwards_matches Out.Functions.hex_bits_41_backwards_matches Out.Functions.hex_bits_42_backwards_matches Out.Functions.hex_bits_43_backwards_matches Out.Functions.hex_bits_44_backwards_matches Out.Functions.hex_bits_45_backwards_matches Out.Functions.hex_bits_46_backwards_matches Out.Functions.hex_bits_47_backwards_matches Out.Functions.hex_bits_48_backwards_matches Out.Functions.hex_bits_49_backwards_matches Out.Functions.hex_bits_4_backwards_matches Out.Functions.hex_bits_50_backwards_matches Out.Functions.hex_bits_51_backwards_matches Out.Functions.hex_bits_52_backwards_matches Out.Functions.hex_bits_53_backwards_matches Out.Functions.hex_bits_54_backwards_matches Out.Functions.hex_bits_55_backwards_matches Out.Functions.hex_bits_56_backwards_matches Out.Functions.hex_bits_57_backwards_matches Out.Functions.hex_bits_58_backwards_matches Out.Functions.hex_bits_59_backwards_matches Out.Functions.hex_bits_5_backwards_matches Out.Functions.hex_bits_60_backwards_matches Out.Functions.hex_bits_61_backwards_matches Out.Functions.hex_bits_62_backwards_matches Out.Functions.hex_bits_63_backwards_matches Out.Functions.hex_bits_64_backwards_matches Out.Functions.hex_bits_6_backwards_matches Out.Functions.hex_bits_7_backwards_matches Out.Functions.hex_bits_8_backwards_matches Out.Functions.hex_bits_9_backwards_matches Out.Functions.hex_bits_backwards Out.Functions.hex_bits_backwards_matches Out.Functions.hex_bits_forwards Out.Functions.hex_bits_forwards_matches Out.Functions.hold_decide Out.Functions.holds_f16 Out.Functions.i8_tr_value Out.Functions.identity_map Out.Functions.idiv_ceil Out.Functions.idiv_floor Out.Functions.ilog2 Out.Functions.in_barrier_window Out.Functions.inf_bits Out.Functions.int8_dot Out.Functions.int_of Out.Functions.int_of_fp Out.Functions.inward_row Out.Functions.iq_padding Out.Functions.is_bool Out.Functions.is_dst Out.Functions.is_float Out.Functions.is_int Out.Functions.is_loopable Out.Functions.is_mxm_queue Out.Functions.is_nan Out.Functions.is_none Out.Functions.is_pair Out.Functions.is_prm Out.Functions.is_sel Out.Functions.is_shf Out.Functions.is_some Out.Functions.is_tsp Out.Functions.is_unsigned Out.Functions.isqrt Out.Functions.issued_by Out.Functions.issuer_of Out.Functions.items_len Out.Functions.k_binary Out.Functions.k_cmp Out.Functions.k_convert Out.Functions.k_convert_any Out.Functions.k_nonzero Out.Functions.k_unary Out.Functions.lane8 Out.Functions.lane_byte Out.Functions.latency_of Out.Functions.latency_of_instr Out.Functions.link_flit_mask Out.Functions.link_pow2 Out.Functions.lint Out.Functions.list_len Out.Functions.ln2_iv Out.Functions.log_iv Out.Functions.mapop_name_backwards_matches Out.Functions.mapop_name_forwards Out.Functions.mapop_name_forwards_matches Out.Functions.mem_dir_mnemonic_backwards_matches Out.Functions.mem_dir_mnemonic_forwards Out.Functions.mem_dir_mnemonic_forwards_matches Out.Functions.mem_quad Out.Functions.mem_slice_present Out.Functions.mem_unit Out.Functions.merge_tiles Out.Functions.model_link_flit_mask Out.Functions.model_link_flits Out.Functions.model_ptype_queued Out.Functions.mxm_acctype_of_num Out.Functions.mxm_bad Out.Functions.mxm_cap Out.Functions.mxm_check Out.Functions.mxm_clash Out.Functions.mxm_part_of_num Out.Functions.mxm_queue_present Out.Functions.mxm_unit Out.Functions.mxm_wtype_of_num Out.Functions.nbits_of Out.Functions.neq_bool Out.Functions.nonempty_list Out.Functions.not Out.Functions.not_bit Out.Functions.num_of_acc_mode Out.Functions.num_of_agt_field Out.Functions.num_of_agt_state Out.Functions.num_of_c2c_op Out.Functions.num_of_c2c_ptype Out.Functions.num_of_diag Out.Functions.num_of_ecc_class Out.Functions.num_of_eltype Out.Functions.num_of_fclass Out.Functions.num_of_mxm_acctype Out.Functions.num_of_mxm_part Out.Functions.num_of_mxm_wtype Out.Functions.num_of_row Out.Functions.num_of_scrub_action Out.Functions.num_of_srf_lookup Out.Functions.num_of_timing_class Out.Functions.num_of_unit_kind Out.Functions.one_plane Out.Functions.opt_spc_backwards Out.Functions.opt_spc_backwards_matches Out.Functions.opt_spc_forwards Out.Functions.opt_spc_forwards_matches Out.Functions.or_pair Out.Functions.outward_row Out.Functions.ov_nth Out.Functions.pair_exact Out.Functions.path_hold_until Out.Functions.path_views Out.Functions.place Out.Functions.place_all Out.Functions.poll_epoch Out.Functions.pow2i Out.Functions.q_in Out.Functions.queue_present Out.Functions.ras_idx Out.Functions.ras_loc Out.Functions.ras_sat_inc Out.Functions.read_gap Out.Functions.reduce_mnemonic_backwards_matches Out.Functions.reduce_mnemonic_forwards Out.Functions.reduce_mnemonic_forwards_matches Out.Functions.ref_bank Out.Functions.ref_path Out.Functions.ref_thread Out.Functions.rep_sp_set Out.Functions.repeat_after Out.Functions.repeat_plan Out.Functions.repeatable Out.Functions.rev_items Out.Functions.ring Out.Functions.rne_int Out.Functions.round_iv Out.Functions.row_of_num Out.Functions.rows_of Out.Functions.s_of_bits Out.Functions.sail_mask Out.Functions.sail_ones Out.Functions.same_target Out.Functions.sat_s Out.Functions.sat_t Out.Functions.scrub_action_of_num Out.Functions.scrub_arbitrate Out.Functions.scrub_due_count Out.Functions.scrub_slot_taken Out.Functions.sep_forwards Out.Functions.sep_forwards_matches Out.Functions.set_lane_byte Out.Functions.sg4_suffix_backwards_matches Out.Functions.sg4_suffix_forwards Out.Functions.sg4_suffix_forwards_matches Out.Functions.shift_dir_name_backwards_matches Out.Functions.shift_dir_name_forwards Out.Functions.shift_dir_name_forwards_matches Out.Functions.slice_mask Out.Functions.slot_of Out.Functions.spc_backwards Out.Functions.spc_backwards_matches Out.Functions.spc_forwards Out.Functions.spc_forwards_matches Out.Functions.sram_addr_of Out.Functions.sram_bank Out.Functions.sram_field_backwards Out.Functions.sram_field_backwards_matches Out.Functions.sram_field_forwards Out.Functions.sram_field_forwards_matches Out.Functions.sram_index Out.Functions.sram_text_backwards Out.Functions.sram_text_backwards_matches Out.Functions.sram_text_forwards Out.Functions.sram_text_forwards_matches Out.Functions.srf_lookup_of_num Out.Functions.srf_ov_spare Out.Functions.srf_rep_merge Out.Functions.srf_slot_best Out.Functions.srf_slot_chain_ov Out.Functions.srf_slot_collides Out.Functions.srf_slot_flip Out.Functions.srf_slot_read Out.Functions.srf_slot_read_ov Out.Functions.srf_slot_write Out.Functions.srf_stream_only_downstream Out.Functions.stream_id Out.Functions.stream_name_forwards_matches Out.Functions.stream_row Out.Functions.string_of_exception Out.Functions.sxm_bad Out.Functions.sxm_clash Out.Functions.sxm_lane Out.Functions.sxm_map_distinct Out.Functions.sxm_or_kind Out.Functions.sxm_set_lane Out.Functions.sxm_stream_plus Out.Functions.sxm_unit_at Out.Functions.tag_of Out.Functions.tags_clash Out.Functions.text_width Out.Functions.tile_active Out.Functions.tile_addr Out.Functions.timing_class_name Out.Functions.timing_class_of_num Out.Functions.timing_of Out.Functions.timing_overrides Out.Functions.timing_type_of Out.Functions.to_bytes_le Out.Functions.tr_eval Out.Functions.tr_eval_int Out.Functions.tr_interval Out.Functions.ty_split_backwards_matches Out.Functions.ty_split_forwards Out.Functions.ty_split_forwards_matches Out.Functions.ty_valid Out.Functions.unit_kind_of_num Out.Functions.unop_entry_backwards_matches Out.Functions.unop_entry_forwards Out.Functions.unop_entry_forwards_matches Out.Functions.unop_legal Out.Functions.vec_units Out.Functions.view_free Out.Functions.view_paths_busy Out.Functions.vsrc_name_forwards_matches Out.Functions.vxm_binop_mnemonic_backwards_matches Out.Functions.vxm_binop_mnemonic_forwards Out.Functions.vxm_binop_mnemonic_forwards_matches Out.Functions.vxm_unit Out.Functions.vxm_unop_mnemonic_backwards_matches Out.Functions.vxm_unop_mnemonic_forwards Out.Functions.vxm_unop_mnemonic_forwards_matches Out.Functions.wait_reason Out.Functions.wants_pair Out.Functions.wlane_add Out.Functions.wlane_read Out.Functions.words_to_planes Out.Functions.wrap_s Out.Functions.wrap_t Out.Functions.write_gap Out.Functions.writer_before Out.Functions.wtype_name_backwards_matches Out.Functions.wtype_name_forwards Out.Functions.wtype_name_forwards_matches Out.Functions.wtype_ok Out.Functions.wtype_split_backwards_matches Out.Functions.wtype_split_forwards Out.Functions.wtype_split_forwards_matches Out.Functions.wtype_valid Out.Functions.zero_bits Out.abs_real Out.add_real Out.diag_callback Out.div_real Out.ecc_event_callback Out.eq_real Out.gt_real Out.gteq_real Out.lint_callback Out.lt_real Out.lteq_real Out.mult_real Out.mxm_corr_callback Out.neg_real Out.prerr_real Out.print_real Out.random_real Out.real_power Out.round_down Out.round_up Out.scrub_event_callback Out.spare_event_callback Out.sqrt_real Out.sram_write_callback Out.srf_write_callback Out.sub_real Out.tdiv_int Out.tmod_int Out.to_real c2c_reset host_reset iq_push_tagged iq_reset masks_reset mem_reset mxm_reset srf_reset sxm_reset reset barrier_retire get_stream set_stream srf_write_ov commit_apply diag_record diag_report report commit_due barrier_notify barrier_sync iq_pop iq_unit_word iq_peek iq_unit_lane iq_word_ov iq_view iq_word_ov_set host_deliver host_msix ras_record ras_ecc_event iq_word_check iq_text_check parcel_earliest decode_text iq_word_unc iq_text_unc iq_decode_at parcel_scan Out.Functions.control host_parked agt_get path_busy path_tail path_view_of polling parked_one queue_parked hold_view_of free_at iq_deliver enqueue ecc_begin exec_cycle execute_C2C_HOSTNOTIFY commit_at ecc_poison_mask ecc_result_ov plane_stream write_planes_fwd write_planes execute_C2C_PCIRECV pci_busy ras_spare_event srf_spare srf_s03 ecc_check_value srf_read_chain_ov srf_only_downstream srf_read_ov read_planes_full read_planes_core read_planes execute_C2C_PCISEND c2c_link c2c_peer_ok execute_C2C_RECEIVE c2c_busy c2c_put execute_C2C_SEND execute_C2C_TRANSMIT execute_CONFIG stream_fetch_arm stream_fetch_start execute_IFETCH execute_IFETCHM commit_cycle eff_addr eff_stream read_cycle read_plane_raw sram_read sram_read_ov write_sram_ecc write_sram_word sram_write_ov tile_on execute_MEM_DIRECT ecc_check_sl srf_report mem_report execute_MEM_GATHER execute_MEM_SCATTER execute_MEM_SETSEGMENT macc_busy_in macc_busy mxm_stage_acc mxm_stage plane mxm_plane_at plane_blocked ras_mxm_event abc_exec execute_MXM_ABC execute_MXM_ABC8 execute_MXM_ACC execute_MXM_IW read_planes16 execute_MXM_IWS execute_MXM_LW execute_SETVL sx sxm_load_map sxm_rd_cycle execute_SXM_LOADMAP execute_SXM_LOADMAPWIDE sxm_cm_cycle sxm_emit execute_SXM_MAP execute_SXM_ROTATE execute_SXM_SELECT execute_SXM_SHIFT execute_SXM_TRANSPOSE sailThrow internal_error fp_transcendental i8_transcendental apply vxm_commit_cycle vxm_read_cycle vxm_read vxm_lanes execute_VXM_BIN execute_VXM_CMP execute_VXM_CONV execute_VXM_CVT execute_VXM_MASK execute_VXM_REDUCE execute_VXM_SELECT execute_VXM_UN execute execute_element execute_all execute_due iq_fetch_ov local_fetch_complete held scrub_issued ras_scrub_event scrub_slice scrub_take scrub_cycle scrub_run stream_fetch_take stream_fetch_service stream_fetch_complete agt_set agt_finish_config deskewrecv_ready deskewrecv_diag host_wait occupancy run_element dispatch_body hold_check dispatch_check reserve_local_fetch repeat_issue dispatch_decoded bank_report cfg_path agt_configure dispatch_cfg loop_clash_ref due_ref agt_due due_refs hold_path loop_tag agt_refresh run_loop dispatch_slots dispatch strict_held parcel_held tick_queue mxm_apply mxm_apply_ok sxm_apply_ok unit_state_flush tick
theorem c2c_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (c2c_reset unitArg : SailM Unit) := by
  unfold c2c_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (host_reset unitArg : SailM Unit) := by
  unfold host_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_push_tagged_bound (ports : Ports) (q : Nat) (u : (BitVec 32)) (tag : Int) : Inv (Bound cycle) (iq_push_tagged q u tag : SailM Bool) := by
  unfold iq_push_tagged
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact registers_bound _ (ports.iq_tag_write _ _ _)
    | exact registers_bound _ (ports.iq_unit_write _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (iq_reset unitArg : SailM Unit) := by
  unfold iq_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_push_tagged_bound ports _ _ _
    | exact registers_bound _ (ports.iq_image_len _)
    | exact registers_bound _ (ports.iq_image_unit _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem masks_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (masks_reset unitArg : SailM Unit) := by
  unfold masks_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mem_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (mem_reset unitArg : SailM Unit) := by
  unfold mem_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (mxm_reset unitArg : SailM Unit) := by
  unfold mxm_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (srf_reset unitArg : SailM Unit) := by
  unfold srf_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sxm_reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (sxm_reset unitArg : SailM Unit) := by
  unfold sxm_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem reset_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (reset unitArg : SailM Unit) := by
  unfold reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact c2c_reset_bound ports _
    | exact host_reset_bound ports _
    | exact iq_reset_bound ports _
    | exact masks_reset_bound ports _
    | exact mem_reset_bound ports _
    | exact mxm_reset_bound ports _
    | exact srf_reset_bound ports _
    | exact sxm_reset_bound ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_retire_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (barrier_retire c : SailM Unit) := by
  unfold barrier_retire
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem get_stream_bound (ports : Ports) (r : row) (s : Nat) : Inv (Bound cycle) (get_stream r s : SailM (Vector (Vector Out.srf_write 44) 64)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.get_stream_frame (execution_ports ports) r s)
theorem set_stream_bound (ports : Ports) (r : row) (s : Nat) (v : (Vector (Vector Out.srf_write 44) 64)) : Inv (Bound cycle) (set_stream r s v : SailM Unit) := by
  unfold set_stream
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_write_ov_bound (ports : Ports) (r : row) (s : Nat) (g : Nat) (c : Nat) (v : vec) (ov : ecc_ov) : Inv (Bound cycle) (srf_write_ov r s g c v ov : SailM Bool) := by
  unfold srf_write_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact get_stream_bound ports _ _
    | exact set_stream_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem commit_apply_bound (ports : Ports) (c : Nat) (all : (List commit)) (xs : (List commit)) (i : Nat) (hits : (Vector Nat 145)) : Inv (Bound cycle) (commit_apply c all xs i hits : SailM (Vector Nat 145)) := by
  induction xs generalizing c all i hits
  case nil =>
    unfold commit_apply
    repeat' first
      | exact ih _ _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact srf_write_ov_bound ports _ _ _ _ _ _
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
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact srf_write_ov_bound ports _ _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem diag_record_bound (ports : Ports) (c : Nat) (q : Nat) (code : Nat) : Inv (Bound cycle) (diag_record c q code : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.diag_record_frame (execution_ports ports) c q code)
theorem diag_report_bound (ports : Ports) (c : Nat) (q : Nat) (d : diag) : Inv (Bound cycle) (diag_report c q d : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.diag_report_frame (execution_ports ports) c q d)
theorem report_bound (ports : Ports) (q : Nat) (c : Nat) (d : diag) : Inv (Bound cycle) (report q c d : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.report_frame (execution_ports ports) q c d)
theorem commit_due_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (commit_due c : SailM Unit) := by
  unfold commit_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact commit_apply_bound ports _ _ _ _ _
    | exact report_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_notify_bound (ports : Ports) (n : Nat) : Inv (Bound cycle) (barrier_notify n : SailM Unit) := by
  unfold barrier_notify
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_sync_bound (ports : Ports) (k : Nat) (c : Nat) : Inv (Bound cycle) (barrier_sync k c : SailM Unit) := by
  unfold barrier_sync
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_pop_bound (ports : Ports) (q : Nat) (n : Nat) : Inv (Bound cycle) (iq_pop q n : SailM Unit) := by
  unfold iq_pop
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_unit_word_bound (ports : Ports) (q : Nat) (i : Nat) : Inv (Bound cycle) (iq_unit_word q i : SailM Int) := by
  unfold iq_unit_word
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact registers_bound _ (ports.iq_tag_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_peek_bound (ports : Ports) (q : Nat) (i : Nat) : Inv (Bound cycle) (iq_peek q i : SailM (BitVec 32)) := by
  unfold iq_peek
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact registers_bound _ (ports.iq_unit_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_unit_lane_bound (ports : Ports) (q : Nat) (i : Nat) : Inv (Bound cycle) (iq_unit_lane q i : SailM Nat) := by
  unfold iq_unit_lane
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact registers_bound _ (ports.iq_tag_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_word_ov_bound (ports : Ports) (q : Nat) (w : Nat) : Inv (Bound cycle) (iq_word_ov q w : SailM (BitVec 10)) := by
  unfold iq_word_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_view_bound (ports : Ports) (q : Nat) (i : Nat) : Inv (Bound cycle) (iq_view q i : SailM (BitVec 32)) := by
  unfold iq_view
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_peek_bound ports _ _
    | exact iq_unit_lane_bound ports _ _
    | exact iq_unit_word_bound ports _ _
    | exact iq_word_ov_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_word_ov_set_bound (ports : Ports) (q : Nat) (w : Nat) (e : (BitVec 10)) : Inv (Bound cycle) (iq_word_ov_set q w e : SailM Unit) := by
  unfold iq_word_ov_set
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_deliver_bound (ports : Ports) (c : Nat) (n : Nat) : Inv (Bound cycle) (host_deliver c n : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.host_deliver_frame (execution_ports ports) c n)
theorem host_msix_bound (ports : Ports) (c : Nat) (n : Nat) : Inv (Bound cycle) (host_msix c n : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.host_msix_frame (execution_ports ports) c n)
theorem ras_record_bound (ports : Ports) (c : Nat) (q : Int) (kind : Nat) (loc : (BitVec 43)) (sl : Nat) (pos : Nat) : Inv (Bound cycle) (ras_record c q kind loc sl pos : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ras_record_frame (execution_ports ports) c q kind loc sl pos)
theorem ras_ecc_event_bound (ports : Ports) (c : Nat) (q : Nat) (kind : Nat) (unc : Bool) (isw : Bool) (id : Nat) (addr : Nat) (sl : Nat) (pos : Nat) : Inv (Bound cycle) (ras_ecc_event c q kind unc isw id addr sl pos : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ras_ecc_event_frame (execution_ports ports) c q kind unc isw id addr sl pos)
theorem iq_word_check_bound (ports : Ports) (q : Nat) (c : Nat) (w : Int) : Inv (Bound cycle) (iq_word_check q c w : SailM Unit) := by
  unfold iq_word_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_peek_bound ports _ _
    | exact iq_word_ov_bound ports _ _
    | exact iq_word_ov_set_bound ports _ _ _
    | exact ras_ecc_event_bound ports _ _ _ _ _ _ _ _ _
    | exact registers_bound _ (ports.iq_tag_read _ _)
    | exact registers_bound _ (ports.iq_unit_write _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_text_check_bound (ports : Ports) (q : Nat) (c : Nat) (units : Nat) : Inv (Bound cycle) (iq_text_check q c units : SailM Unit) := by
  unfold iq_text_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_unit_word_bound ports _ _
    | exact iq_view_bound ports _ _
    | exact iq_word_check_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parcel_earliest_bound (ports : Ports) (q : Nat) (p : parcel) : Inv (Bound cycle) (parcel_earliest q p : SailM Int) := by
  unfold parcel_earliest
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem decode_text_bound (ports : Ports) (u0 : (BitVec 32)) (u1 : (BitVec 32)) : Inv (Bound cycle) (decode_text u0 u1 : SailM (Result instruction diag)) := by
  unfold ResourceInvariantProofs.Inv
  intro state hs
  have h:=Proofs.ParcelFrame.decode_text_frame state u0 u1
  cases hx : (decode_text u0 u1).run state <;> simp only [Proofs.ParcelFrame.FrameAt,hx] at h <;> simpa [hx,h] using hs
theorem iq_word_unc_bound (ports : Ports) (q : Nat) (w : Int) : Inv (Bound cycle) (iq_word_unc q w : SailM Bool) := by
  unfold iq_word_unc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_word_ov_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_text_unc_bound (ports : Ports) (q : Nat) (pos : Nat) (w : Nat) : Inv (Bound cycle) (iq_text_unc q pos w : SailM Bool) := by
  unfold iq_text_unc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_unit_word_bound ports _ _
    | exact iq_word_unc_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_decode_at_bound (ports : Ports) (q : Nat) (pos : Nat) (w : Nat) : Inv (Bound cycle) (iq_decode_at q pos w : SailM (Result instruction diag)) := by
  unfold iq_decode_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact decode_text_bound ports _ _
    | exact iq_text_unc_bound ports _ _ _
    | exact iq_view_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parcel_scan_bound (ports : Ports) (q : Nat) : Inv (Bound cycle) (parcel_scan q : SailM parcel) := by
  unfold parcel_scan
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_decode_at_bound ports _ _ _
    | exact iq_view_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem control_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (Out.Functions.control c : SailM Unit) := by
  unfold Out.Functions.control
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact barrier_notify_bound ports _
    | exact barrier_sync_bound ports _ _
    | exact iq_pop_bound ports _ _
    | exact iq_text_check_bound ports _ _ _
    | exact parcel_earliest_bound ports _ _
    | exact parcel_scan_bound ports _
    | exact report_bound ports _ _ _
    | exact registers_bound _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_parked_bound (ports : Ports) (q : Nat) (c : Nat) : Inv (Bound cycle) (host_parked q c : SailM Bool) := by
  unfold host_parked
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_get_bound (ports : Ports) (q : Nat) (p : Nat) : Inv (Bound cycle) (agt_get q p : SailM agt) := by
  exact execution_origin _ (AgtExecutionFrameProofs.agt_get_frame (execution_ports ports) q p)
theorem path_busy_bound (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv (Bound cycle) (path_busy q p c : SailM Bool) := by
  unfold path_busy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem path_tail_bound (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv (Bound cycle) (path_tail q p c : SailM Bool) := by
  unfold path_tail
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem path_view_of_bound (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv (Bound cycle) (path_view_of q p c : SailM path_view) := by
  unfold path_view_of
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact agt_get_bound ports _ _
    | exact path_busy_bound ports _ _ _
    | exact path_tail_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem polling_bound (ports : Ports) (q : Nat) : Inv (Bound cycle) (polling q : SailM Bool) := by
  unfold polling
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parked_one_bound (ports : Ports) (k : Nat) (c : Nat) : Inv (Bound cycle) (parked_one k c : SailM Bool) := by
  unfold parked_one
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem queue_parked_bound (ports : Ports) (q : Nat) (c : Nat) : Inv (Bound cycle) (queue_parked q c : SailM Bool) := by
  unfold queue_parked
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact parked_one_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_view_of_bound (ports : Ports) (q : Nat) (c : Nat) : Inv (Bound cycle) (hold_view_of q c : SailM hold_view) := by
  unfold hold_view_of
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact host_parked_bound ports _ _
    | exact path_view_of_bound ports _ _ _
    | exact polling_bound ports _
    | exact queue_parked_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem free_at_bound (ports : Ports) (q : Nat) (t : Nat) : Inv (Bound cycle) (free_at q t : SailM Bool) := by
  unfold free_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact hold_view_of_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_deliver_bound (ports : Ports) (q : Nat) (c : Nat) : Inv (Bound cycle) (iq_deliver q c : SailM (Bool × Bool)) := by
  unfold iq_deliver
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact iq_push_tagged_bound ports _ _ _
    | exact iq_word_ov_set_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem enqueue_bound (ports : Ports) (q : Nat) (c : Nat) : Inv (Bound cycle) (enqueue q c : SailM Unit) := by
  unfold enqueue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact free_at_bound ports _ _
    | exact iq_deliver_bound ports _ _
    | exact parcel_earliest_bound ports _ _
    | exact parcel_scan_bound ports _
    | exact report_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ecc_begin_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (ecc_begin unitArg : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ecc_begin_frame (execution_ports ports) unitArg)
theorem exec_cycle_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (exec_cycle unitArg : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.exec_cycle_frame (execution_ports ports) unitArg)
theorem execute_C2C_HOSTNOTIFY_bound (ports : Ports) (n : (BitVec 4)) : Inv (Bound cycle) (execute_C2C_HOSTNOTIFY n : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_C2C_HOSTNOTIFY_frame (execution_ports ports) n)
theorem commit_at_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (commit_at unitArg : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.commit_at_frame (execution_ports ports) unitArg)
theorem ecc_poison_mask_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (ecc_poison_mask unitArg : SailM (BitVec Out.Functions.superlanes)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ecc_poison_mask_frame (execution_ports ports) unitArg)
theorem ecc_result_ov_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (ecc_result_ov unitArg : SailM ecc_ov) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ecc_result_ov_frame (execution_ports ports) unitArg)
theorem plane_stream_bound (ports : Ports) (s : streamref) (b : Int) : Inv (Bound cycle) (plane_stream s b : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.plane_stream_frame (execution_ports ports) s b)
theorem write_planes_fwd_bound (ports : Ports) (s : streamref) (p : (Vector vec 4)) (w : Nat) (c : Nat) (fwd : (Option ecc_ov)) : Inv (Bound cycle) (write_planes_fwd s p w c fwd : SailM (Result Unit diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.write_planes_fwd_frame (execution_ports ports) s p w c fwd)
theorem write_planes_bound (ports : Ports) (s : streamref) (p : (Vector vec 4)) (w : Nat) (c : Nat) : Inv (Bound cycle) (write_planes s p w c : SailM (Result Unit diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.write_planes_frame (execution_ports ports) s p w c)
theorem execute_C2C_PCIRECV_bound (ports : Ports) (s : streamref) : Inv (Bound cycle) (execute_C2C_PCIRECV s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_C2C_PCIRECV_frame (execution_ports ports) s)
theorem pci_busy_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (pci_busy c : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.pci_busy_frame (execution_ports ports) c)
theorem ras_spare_event_bound (ports : Ports) (c : Nat) (q : Nat) (isw : Bool) (id : Nat) (gap : Nat) (sl : Nat) (s03 : (BitVec 4)) : Inv (Bound cycle) (ras_spare_event c q isw id gap sl s03 : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ras_spare_event_frame (execution_ports ports) c q isw id gap sl s03)
theorem srf_spare_bound (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv (Bound cycle) (srf_spare r s g t : SailM (BitVec Out.Functions.superlanes)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.srf_spare_frame (execution_ports ports) r s g t)
theorem srf_s03_bound (ports : Ports) (rw : Bool) (g : Int) (t : Int) (x : Int) : Inv (Bound cycle) (srf_s03 rw g t x : SailM (BitVec 4)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.srf_s03_frame (execution_ports ports) rw g t x)
theorem ecc_check_value_bound (ports : Ports) (ci : Int) (q : Nat) (kind : Nat) (isw : Bool) (id : Nat) (addr : Nat) (v : vec) (ov : ecc_ov) : Inv (Bound cycle) (ecc_check_value ci q kind isw id addr v ov : SailM (vec × (BitVec Out.Functions.superlanes))) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ecc_check_value_frame (execution_ports ports) ci q kind isw id addr v ov)
theorem srf_read_chain_ov_bound (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv (Bound cycle) (srf_read_chain_ov r s g t : SailM (srf_lookup × vec × ecc_ov)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.srf_read_chain_ov_frame (execution_ports ports) r s g t)
theorem srf_only_downstream_bound (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv (Bound cycle) (srf_only_downstream r s g t : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.srf_only_downstream_frame (execution_ports ports) r s g t)
theorem srf_read_ov_bound (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv (Bound cycle) (srf_read_ov r s g t : SailM (srf_lookup × vec × ecc_ov)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.srf_read_ov_frame (execution_ports ports) r s g t)
theorem read_planes_full_bound (ports : Ports) (s : streamref) (w : Nat) (t : Int) (chain : Bool) (chk : Bool) : Inv (Bound cycle) (read_planes_full s w t chain chk : SailM (Result ((Vector vec 4) × (Vector ecc_ov 4)) diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.read_planes_full_frame (execution_ports ports) s w t chain chk)
theorem read_planes_core_bound (ports : Ports) (s : streamref) (w : Nat) (t : Int) (chain : Bool) : Inv (Bound cycle) (read_planes_core s w t chain : SailM (Result (Vector vec 4) diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.read_planes_core_frame (execution_ports ports) s w t chain)
theorem read_planes_bound (ports : Ports) (s : streamref) (w : Nat) (t : Int) : Inv (Bound cycle) (read_planes s w t : SailM (Result (Vector vec 4) diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.read_planes_frame (execution_ports ports) s w t)
theorem execute_C2C_PCISEND_bound (ports : Ports) (s : streamref) : Inv (Bound cycle) (execute_C2C_PCISEND s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_C2C_PCISEND_frame (execution_ports ports) s)
theorem c2c_link_bound (ports : Ports) (link : (BitVec 7)) : Inv (Bound cycle) (c2c_link link : SailM (Option Nat)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.c2c_link_frame (execution_ports ports) link)
theorem c2c_peer_ok_bound (ports : Ports) (chip : (BitVec 9)) (l : Nat) : Inv (Bound cycle) (c2c_peer_ok chip l : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.c2c_peer_ok_frame (execution_ports ports) chip l)
theorem execute_C2C_RECEIVE_bound (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Inv (Bound cycle) (execute_C2C_RECEIVE chip link s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_C2C_RECEIVE_frame (execution_ports ports) chip link s)
theorem c2c_busy_bound (ports : Ports) (l : Nat) (c : Nat) : Inv (Bound cycle) (c2c_busy l c : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.c2c_busy_frame (execution_ports ports) l c)
theorem c2c_put_bound (ports : Ports) (l : Nat) (t : c2c_ptype) (v : vec) (poison : Nat) : Inv (Bound cycle) (c2c_put l t v poison : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.c2c_put_frame (execution_ports ports) l t v poison)
theorem execute_C2C_SEND_bound (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Inv (Bound cycle) (execute_C2C_SEND chip link s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_C2C_SEND_frame (execution_ports ports) chip link s)
theorem execute_C2C_TRANSMIT_bound (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) : Inv (Bound cycle) (execute_C2C_TRANSMIT chip link : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_C2C_TRANSMIT_frame (execution_ports ports) chip link)
theorem execute_CONFIG_bound (ports : Ports) (v : (BitVec 16)) : Inv (Bound cycle) (execute_CONFIG v : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_CONFIG_frame (execution_ports ports) v)
theorem stream_fetch_arm_bound (ports : Ports) (q : Nat) (st : streamref) (t : Nat) (data : vec) (ov : ecc_ov) : Inv (Bound cycle) (stream_fetch_arm q st t data ov : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.stream_fetch_arm_frame (execution_ports ports) q st t data ov)
theorem stream_fetch_start_bound (ports : Ports) (st : streamref) : Inv (Bound cycle) (stream_fetch_start st : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.stream_fetch_start_frame (execution_ports ports) st)
theorem execute_IFETCH_bound (ports : Ports) (st : streamref) : Inv (Bound cycle) (execute_IFETCH st : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_IFETCH_frame (execution_ports ports) st)
theorem execute_IFETCHM_bound (ports : Ports) (app_0 : sram_addr) : Inv (Bound cycle) (execute_IFETCHM app_0 : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_IFETCHM_frame (execution_ports ports) app_0)
theorem commit_cycle_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (commit_cycle unitArg : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.commit_cycle_frame (execution_ports ports) unitArg)
theorem eff_addr_bound (ports : Ports) (app_0 : sram_addr) : Inv (Bound cycle) (eff_addr app_0 : SailM (Option sram_addr)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.eff_addr_frame (execution_ports ports) app_0)
theorem eff_stream_bound (ports : Ports) (app_0 : streamref) : Inv (Bound cycle) (eff_stream app_0 : SailM (Option streamref)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.eff_stream_frame (execution_ports ports) app_0)
theorem read_cycle_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (read_cycle unitArg : SailM Int) := by
  exact execution_origin _ (AgtExecutionFrameProofs.read_cycle_frame (execution_ports ports) unitArg)
theorem read_plane_raw_bound (ports : Ports) (s : streamref) (t : Int) : Inv (Bound cycle) (read_plane_raw s t : SailM (Result (vec × ecc_ov) diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.read_plane_raw_frame (execution_ports ports) s t)
theorem sram_read_bound (ports : Ports) (s : Nat) (a : sram_addr) : Inv (Bound cycle) (sram_read s a : SailM (Option vec)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sram_read_frame (execution_ports ports) s a)
theorem sram_read_ov_bound (ports : Ports) (s : Nat) (a : sram_addr) : Inv (Bound cycle) (sram_read_ov s a : SailM ecc_ov) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sram_read_ov_frame (execution_ports ports) s a)
theorem write_sram_ecc_bound (ports : Ports) (s : Nat) (a : Nat) (v : ecc_ov) : Inv (Bound cycle) (write_sram_ecc s a v : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.write_sram_ecc_frame (execution_ports ports) s a v)
theorem write_sram_word_bound (ports : Ports) (s : Nat) (a : Nat) (v : vec) : Inv (Bound cycle) (write_sram_word s a v : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.write_sram_word_frame (execution_ports ports) s a v)
theorem sram_write_ov_bound (ports : Ports) (c : Nat) (s : Nat) (typ_2 : sram_addr) (v : vec) (ov : ecc_ov) : Inv (Bound cycle) (sram_write_ov c s typ_2 v ov : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sram_write_ov_frame (execution_ports ports) c s typ_2 v ov)
theorem tile_on_bound (ports : Ports) (g : Int) : Inv (Bound cycle) (tile_on g : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.tile_on_frame (execution_ports ports) g)
theorem execute_MEM_DIRECT_bound (ports : Ports) (op : mem_dir) (a0 : sram_addr) (s0 : streamref) (t : (BitVec 1)) : Inv (Bound cycle) (execute_MEM_DIRECT op a0 s0 t : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MEM_DIRECT_frame (execution_ports ports) op a0 s0 t)
theorem ecc_check_sl_bound (ports : Ports) (ci : Int) (q : Nat) (slice_ : Nat) (addr : Nat) (g : Nat) (d : (BitVec 128)) (e : (BitVec 10)) : Inv (Bound cycle) (ecc_check_sl ci q slice_ addr g d e : SailM ((BitVec 128) × Bool)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ecc_check_sl_frame (execution_ports ports) ci q slice_ addr g d e)
theorem srf_report_bound (ports : Ports) (r : row) (g : Nat) (c : Nat) (prio : Int) (xs : (BitVec Out.Functions.superlanes)) (sp : (Vector (BitVec Out.Functions.superlanes) 30)) : Inv (Bound cycle) (srf_report r g c prio xs sp : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.srf_report_frame (execution_ports ports) r g c prio xs sp)
theorem mem_report_bound (ports : Ports) (prio : Int) (bounds : (BitVec Out.Functions.superlanes)) (addrs : (Vector Int Out.Functions.superlanes)) (data : streamref) (map : streamref) (c : Nat) : Inv (Bound cycle) (mem_report prio bounds addrs data map c : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.mem_report_frame (execution_ports ports) prio bounds addrs data map c)
theorem execute_MEM_GATHER_bound (ports : Ports) (d0 : streamref) (m : streamref) (kb : (BitVec 4)) (t : (BitVec 1)) : Inv (Bound cycle) (execute_MEM_GATHER d0 m kb t : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MEM_GATHER_frame (execution_ports ports) d0 m kb t)
theorem execute_MEM_SCATTER_bound (ports : Ports) (s0 : streamref) (m : streamref) (kb : (BitVec 4)) (t : (BitVec 1)) : Inv (Bound cycle) (execute_MEM_SCATTER s0 m kb t : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MEM_SCATTER_frame (execution_ports ports) s0 m kb t)
theorem execute_MEM_SETSEGMENT_bound (ports : Ports) (n : (BitVec 17)) : Inv (Bound cycle) (execute_MEM_SETSEGMENT n : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MEM_SETSEGMENT_frame (execution_ports ports) n)
theorem macc_busy_in_bound (ports : Ports) (xs : (List abc_entry)) : Inv (Bound cycle) (macc_busy_in xs : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.macc_busy_in_frame (execution_ports ports) xs)
theorem macc_busy_bound (ports : Ports) (p : mxm_plane) : Inv (Bound cycle) (macc_busy p : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.macc_busy_frame (execution_ports ports) p)
theorem mxm_stage_acc_bound (ports : Ports) (pl : Nat) (part : mxm_part) (a : Nat) (v : mxm_plane) : Inv (Bound cycle) (mxm_stage_acc pl part a v : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.mxm_stage_acc_frame (execution_ports ports) pl part a v)
theorem mxm_stage_bound (ports : Ports) (pl : Nat) (part : mxm_part) (v : mxm_plane) : Inv (Bound cycle) (mxm_stage pl part v : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.mxm_stage_frame (execution_ports ports) pl part v)
theorem plane_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (plane unitArg : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.plane_frame (execution_ports ports) unitArg)
theorem mxm_plane_at_bound (ports : Ports) (i : Int) : Inv (Bound cycle) (mxm_plane_at i : SailM mxm_plane) := by
  exact execution_origin _ (AgtExecutionFrameProofs.mxm_plane_at_frame (execution_ports ports) i)
theorem plane_blocked_bound (ports : Ports) (t : (Option mxm_wtype)) : Inv (Bound cycle) (plane_blocked t : SailM Bool) := by
  exact execution_origin _ (AgtExecutionFrameProofs.plane_blocked_frame (execution_ports ports) t)
theorem ras_mxm_event_bound (ports : Ports) (c : Nat) (q : Int) (unc : Bool) (plane : Nat) (lane : Nat) : Inv (Bound cycle) (ras_mxm_event c q unc plane lane : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.ras_mxm_event_frame (execution_ports ports) c q unc plane lane)
theorem abc_exec_bound (ports : Ports) (t : mxm_wtype) (f : (BitVec 1)) (s : streamref) : Inv (Bound cycle) (abc_exec t f s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.abc_exec_frame (execution_ports ports) t f s)
theorem execute_MXM_ABC_bound (ports : Ports) (t : mxm_wtype) (s : streamref) : Inv (Bound cycle) (execute_MXM_ABC t s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MXM_ABC_frame (execution_ports ports) t s)
theorem execute_MXM_ABC8_bound (ports : Ports) (f : (BitVec 1)) (s : streamref) : Inv (Bound cycle) (execute_MXM_ABC8 f s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MXM_ABC8_frame (execution_ports ports) f s)
theorem execute_MXM_ACC_bound (ports : Ports) (t : mxm_acctype) (m : acc_mode) (d : streamref) (ab : (BitVec 5)) : Inv (Bound cycle) (execute_MXM_ACC t m d ab : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MXM_ACC_frame (execution_ports ports) t m d ab)
theorem execute_MXM_IW_bound (ports : Ports) (t : mxm_wtype) : Inv (Bound cycle) (execute_MXM_IW t : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MXM_IW_frame (execution_ports ports) t)
theorem read_planes16_bound (ports : Ports) (s : streamref) (t : Int) : Inv (Bound cycle) (read_planes16 s t : SailM (Result (Vector vec 16) diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.read_planes16_frame (execution_ports ports) s t)
theorem execute_MXM_IWS_bound (ports : Ports) (t : mxm_wtype) (s : streamref) : Inv (Bound cycle) (execute_MXM_IWS t s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MXM_IWS_frame (execution_ports ports) t s)
theorem execute_MXM_LW_bound (ports : Ports) (t : mxm_wtype) (s : streamref) : Inv (Bound cycle) (execute_MXM_LW t s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_MXM_LW_frame (execution_ports ports) t s)
theorem execute_SETVL_bound (ports : Ports) (n : (BitVec 6)) : Inv (Bound cycle) (execute_SETVL n : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SETVL_frame (execution_ports ports) n)
theorem sx_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (sx unitArg : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sx_frame (execution_ports ports) unitArg)
theorem sxm_load_map_bound (ports : Ports) (mi : (BitVec 4)) (lo : vec) (hi : vec) : Inv (Bound cycle) (sxm_load_map mi lo hi : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sxm_load_map_frame (execution_ports ports) mi lo hi)
theorem sxm_rd_cycle_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (sxm_rd_cycle unitArg : SailM Int) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sxm_rd_cycle_frame (execution_ports ports) unitArg)
theorem execute_SXM_LOADMAP_bound (ports : Ports) (mi : (BitVec 4)) (s : streamref) : Inv (Bound cycle) (execute_SXM_LOADMAP mi s : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SXM_LOADMAP_frame (execution_ports ports) mi s)
theorem execute_SXM_LOADMAPWIDE_bound (ports : Ports) (mi : (BitVec 4)) (s : streamref) (s2 : streamref) : Inv (Bound cycle) (execute_SXM_LOADMAPWIDE mi s s2 : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SXM_LOADMAPWIDE_frame (execution_ports ports) mi s s2)
theorem sxm_cm_cycle_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (sxm_cm_cycle unitArg : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sxm_cm_cycle_frame (execution_ports ports) unitArg)
theorem sxm_emit_bound (ports : Ports) (d : streamref) (v : vec) : Inv (Bound cycle) (sxm_emit d v : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sxm_emit_frame (execution_ports ports) d v)
theorem execute_SXM_MAP_bound (ports : Ports) (op : sxm_mapop) (mi : (BitVec 4)) (s : streamref) (d : streamref) : Inv (Bound cycle) (execute_SXM_MAP op mi s d : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SXM_MAP_frame (execution_ports ports) op mi s d)
theorem execute_SXM_ROTATE_bound (ports : Ports) (nb : (BitVec 10)) (s : streamref) (d : streamref) : Inv (Bound cycle) (execute_SXM_ROTATE nb s d : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SXM_ROTATE_frame (execution_ports ports) nb s d)
theorem execute_SXM_SELECT_bound (ports : Ports) (a : streamref) (b : streamref) (c : streamref) (m : streamref) (d : streamref) : Inv (Bound cycle) (execute_SXM_SELECT a b c m d : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SXM_SELECT_frame (execution_ports ports) a b c m d)
theorem execute_SXM_SHIFT_bound (ports : Ports) (dir : shift_dir) (n : (BitVec 9)) (s : streamref) (d : streamref) : Inv (Bound cycle) (execute_SXM_SHIFT dir n s d : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SXM_SHIFT_frame (execution_ports ports) dir n s d)
theorem execute_SXM_TRANSPOSE_bound (ports : Ports) (s : streamref) (d : streamref) : Inv (Bound cycle) (execute_SXM_TRANSPOSE s d : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_SXM_TRANSPOSE_frame (execution_ports ports) s d)
theorem sailThrow_bound {α : Type} (ports : Ports) (e : exception) : Inv (Bound cycle) (sailThrow e : SailM α) := by
  exact execution_origin _ (AgtExecutionFrameProofs.sailThrow_frame (execution_ports ports) e)
theorem internal_error_bound {α : Type} (ports : Ports) (file : String) (line : Int) (s : String) : Inv (Bound cycle) (internal_error file line s : SailM α) := by
  exact execution_origin _ (AgtExecutionFrameProofs.internal_error_frame (execution_ports ports) file line s)
theorem fp_transcendental_bound (ports : Ports) (op : Int) (f : fmt) (x : (BitVec 32)) : Inv (Bound cycle) (fp_transcendental op f x : SailM (BitVec 32)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.fp_transcendental_frame (execution_ports ports) op f x)
theorem i8_transcendental_bound (ports : Ports) (op : Int) (x : (BitVec 32)) : Inv (Bound cycle) (i8_transcendental op x : SailM (BitVec 32)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.i8_transcendental_frame (execution_ports ports) op x)
theorem apply_bound (ports : Ports) (op : vxm_lane_op) (a : (BitVec 32)) (b : (BitVec 32)) (m : (BitVec 32)) : Inv (Bound cycle) (apply op a b m : SailM (BitVec 32)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.apply_frame (execution_ports ports) op a b m)
theorem vxm_commit_cycle_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (vxm_commit_cycle unitArg : SailM Nat) := by
  exact execution_origin _ (AgtExecutionFrameProofs.vxm_commit_cycle_frame (execution_ports ports) unitArg)
theorem vxm_read_cycle_bound (ports : Ports) (unitArg : Unit) : Inv (Bound cycle) (vxm_read_cycle unitArg : SailM Int) := by
  exact execution_origin _ (AgtExecutionFrameProofs.vxm_read_cycle_frame (execution_ports ports) unitArg)
theorem vxm_read_bound (ports : Ports) (a : vsrc) (w : Nat) : Inv (Bound cycle) (vxm_read a w : SailM (Result (Vector vec 4) diag)) := by
  exact execution_origin _ (AgtExecutionFrameProofs.vxm_read_frame (execution_ports ports) a w)
theorem vxm_lanes_bound (ports : Ports) (op : vxm_lane_op) (dst : streamref) (a : vsrc) (b : (Option vsrc)) (m : (Option vsrc)) (ws : Nat) (wd : Nat) (g : (BitVec 1)) : Inv (Bound cycle) (vxm_lanes op dst a b m ws wd g : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.vxm_lanes_frame (execution_ports ports) op dst a b m ws wd g)
theorem execute_VXM_BIN_bound (ports : Ports) (op : vxm_binop) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (g : (BitVec 1)) : Inv (Bound cycle) (execute_VXM_BIN op t d a b g : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_BIN_frame (execution_ports ports) op t d a b g)
theorem execute_VXM_CMP_bound (ports : Ports) (mode : cmp_mode) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (g : (BitVec 1)) : Inv (Bound cycle) (execute_VXM_CMP mode t d a b g : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_CMP_frame (execution_ports ports) mode t d a b g)
theorem execute_VXM_CONV_bound (ports : Ports) (c : vxm_conv) (g__241 : eltype) (d : streamref) (a : vsrc) : Inv (Bound cycle) (execute_VXM_CONV c g__241 d a : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_CONV_frame (execution_ports ports) c g__241 d a)
theorem execute_VXM_CVT_bound (ports : Ports) (st : eltype) (dt : eltype) (d : streamref) (a : vsrc) : Inv (Bound cycle) (execute_VXM_CVT st dt d a : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_CVT_frame (execution_ports ports) st dt d a)
theorem execute_VXM_MASK_bound (ports : Ports) (t : eltype) (d : streamref) (a : vsrc) (m : vsrc) (g : (BitVec 1)) : Inv (Bound cycle) (execute_VXM_MASK t d a m g : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_MASK_frame (execution_ports ports) t d a m g)
theorem execute_VXM_REDUCE_bound (ports : Ports) (op : reduce_op) (t : eltype) (d : streamref) (a : vsrc) : Inv (Bound cycle) (execute_VXM_REDUCE op t d a : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_REDUCE_frame (execution_ports ports) op t d a)
theorem execute_VXM_SELECT_bound (ports : Ports) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (m : vsrc) (g : (BitVec 1)) : Inv (Bound cycle) (execute_VXM_SELECT t d a b m g : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_SELECT_frame (execution_ports ports) t d a b m g)
theorem execute_VXM_UN_bound (ports : Ports) (op : vxm_unop) (t : eltype) (d : streamref) (a : vsrc) (g : (BitVec 1)) : Inv (Bound cycle) (execute_VXM_UN op t d a g : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_VXM_UN_frame (execution_ports ports) op t d a g)
theorem execute_bound (ports : Ports) (merge_var : instruction) : Inv (Bound cycle) (execute merge_var : SailM ExecutionResult) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_frame (execution_ports ports) merge_var)
theorem execute_element_bound (ports : Ports) (e : exec_element) : Inv (Bound cycle) (execute_element e : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.execute_element_frame (execution_ports ports) e)
theorem execute_all_bound (ports : Ports) (xs : (List exec_element)) : Inv (Bound cycle) (execute_all xs : SailM Unit) := by
  induction xs
  case nil =>
    unfold execute_all
    repeat' first
      | exact ih 
      | assumption
      | exact pure_inv _ _
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact execute_element_bound ports _
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
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact execute_element_bound ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem execute_due_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (execute_due c : SailM Unit) := by
  unfold execute_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact execute_all_bound ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_fetch_ov_bound (ports : Ports) (q : Nat) (t : Nat) (a : vec) (b : vec) (ova : ecc_ov) (ovb : ecc_ov) : Inv (Bound cycle) (iq_fetch_ov q t a b ova ovb : SailM Bool) := by
  unfold iq_fetch_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact registers_bound _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem local_fetch_complete_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (local_fetch_complete c : SailM Unit) := by
  unfold local_fetch_complete
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact diag_report_bound ports _ _ _
    | exact iq_fetch_ov_bound ports _ _ _ _ _ _
    | exact registers_bound _ (ports.read_sram_ecc _ _)
    | exact registers_bound _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem held_bound (ports : Ports) (q : Nat) : Inv (Bound cycle) (held q : SailM Bool) := by
  unfold held
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_issued_bound (ports : Ports) (q : Nat) (m : Nat) : Inv (Bound cycle) (scrub_issued q m : SailM Bool) := by
  unfold scrub_issued
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ras_scrub_event_bound (ports : Ports) (c : Nat) (s : Nat) (unc : Bool) (a : Nat) (sl : Nat) (pos : Nat) : Inv (Bound cycle) (ras_scrub_event c s unc a sl pos : SailM Unit) := by
  unfold ras_scrub_event
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact ras_record_bound ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_slice_bound (ports : Ports) (s : Nat) (m : Nat) : Inv (Bound cycle) (scrub_slice s m : SailM Unit) := by
  unfold scrub_slice
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact ras_scrub_event_bound ports _ _ _ _ _ _
    | exact registers_bound _ (ports.read_sram_ecc _ _)
    | exact registers_bound _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_take_bound (ports : Ports) (s : Nat) : Inv (Bound cycle) (scrub_take s : SailM (Option scrub_write)) := by
  unfold scrub_take
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_cycle_bound (ports : Ports) (m : Nat) : Inv (Bound cycle) (scrub_cycle m : SailM Unit) := by
  unfold scrub_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact held_bound ports _
    | exact scrub_issued_bound ports _ _
    | exact scrub_slice_bound ports _ _
    | exact scrub_take_bound ports _
    | exact write_sram_ecc_bound ports _ _ _
    | exact write_sram_word_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_run_bound (ports : Ports) (upto : Int) : Inv (Bound cycle) (scrub_run upto : SailM Unit) := by
  unfold scrub_run
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact scrub_cycle_bound ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_take_bound (ports : Ports) (q : Nat) (c : Nat) : Inv (Bound cycle) (stream_fetch_take q c : SailM (Option stream_fetch)) := by
  unfold stream_fetch_take
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_service_bound (ports : Ports) (q : Nat) (c : Nat) : Inv (Bound cycle) (stream_fetch_service q c : SailM Unit) := by
  unfold stream_fetch_service
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact diag_report_bound ports _ _ _
    | exact ecc_begin_bound ports _
    | exact iq_fetch_ov_bound ports _ _ _ _ _ _
    | exact read_planes_bound ports _ _ _
    | exact stream_fetch_take_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_complete_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (stream_fetch_complete c : SailM Unit) := by
  unfold stream_fetch_complete
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact stream_fetch_service_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_finish_config_bound (ports : Ports) (q : Nat) (p : Nat) : Inv (Bound cycle) (agt_finish_config q p : SailM Unit) :=by
 exact actual_finish_bound_any q p
theorem deskewrecv_ready_bound (ports : Ports) (c : Nat) (link : (BitVec 7)) : Inv (Bound cycle) (deskewrecv_ready c link : SailM Bool) := by
  unfold deskewrecv_ready
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact c2c_link_bound ports _
    | exact registers_bound _ (ports.c2c_recv_ready _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem deskewrecv_diag_bound (ports : Ports) (iu : issuer) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Inv (Bound cycle) (deskewrecv_diag iu chip link s : SailM (Option diag)) := by
  unfold deskewrecv_diag
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact c2c_link_bound ports _
    | exact c2c_peer_ok_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_wait_bound (ports : Ports) (q : Nat) (c : Nat) (i : Nat) : Inv (Bound cycle) (host_wait q c i : SailM Unit) := by
  unfold host_wait
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem occupancy_bound (ports : Ports) (arg0 : instruction) (arg1 : Nat) : Inv (Bound cycle) (occupancy arg0 arg1 : SailM Nat) := by
  unfold occupancy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem run_element_bound (ports : Ports) (q : Nat) (i : instruction) (c0 : Nat) (k : Nat) (off : Int) (soff : Int) : Inv (Bound cycle) (run_element q i c0 k off soff : SailM Unit) := by
  exact execution_origin _ (AgtExecutionFrameProofs.run_element_frame (execution_ports ports) q i c0 k off soff)
theorem dispatch_body_bound (ports : Ports) (q : Nat) (c : Nat) (i : instruction) : Inv (Bound cycle) (dispatch_body q c i : SailM Unit) := by
  unfold dispatch_body
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact barrier_notify_bound ports _
    | exact barrier_sync_bound ports _ _
    | exact deskewrecv_diag_bound ports _ _ _ _
    | exact host_wait_bound ports _ _ _
    | exact occupancy_bound ports _ _
    | exact report_bound ports _ _ _
    | exact run_element_bound ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_check_bound (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) (strict : Bool) : Inv (Bound cycle) (hold_check q c i path strict : SailM Bool) := by
  unfold hold_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact hold_view_of_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_check_bound (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) : Inv (Bound cycle) (dispatch_check q c i path : SailM (Option diag)) := by
  unfold dispatch_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact hold_check_bound ports _ _ _ _ _
    | exact internal_error_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem reserve_local_fetch_bound (ports : Ports) (q : Nat) (i : instruction) (c : Nat) : Inv (Bound cycle) (reserve_local_fetch q i c : SailM Unit) := by
  unfold reserve_local_fetch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem repeat_issue_bound (ports : Ports) (q : Nat) (i : instruction) (c : Nat) : Inv (Bound cycle) (repeat_issue q i c : SailM Unit) := by
  unfold repeat_issue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact reserve_local_fetch_bound ports _ _ _
    | exact run_element_bound ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_decoded_bound (ports : Ports) (q : Nat) (c : Nat) (i : instruction) : Inv (Bound cycle) (dispatch_decoded q c i : SailM Unit) := by
  unfold dispatch_decoded
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact dispatch_body_bound ports _ _ _
    | exact dispatch_check_bound ports _ _ _ _
    | exact repeat_issue_bound ports _ _ _
    | exact report_bound ports _ _ _
    | exact registers_bound _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem bank_report_bound (ports : Ports) (q : Nat) (c : Nat) (a : (instruction × Int × Int)) (b : (instruction × Int × Int)) : Inv (Bound cycle) (bank_report q c a b : SailM Unit) := by
  unfold bank_report
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact srf_report_bound ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem cfg_path_bound (ports : Ports) (q : Nat) : Inv (Bound cycle) (cfg_path q : SailM Nat) := by
  unfold cfg_path
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_configure_bound (ports : Ports) (q : Nat) (p : Nat) (f : agt_field) (v : (BitVec 16)) : Inv (Bound cycle) (agt_configure q p f v : SailM Bool) :=by
 exact actual_configure_bound_any q p f v
theorem dispatch_cfg_bound (ports : Ports) (q : Nat) (c : Nat) (p : Nat) (i : instruction) : Inv (Bound cycle) (dispatch_cfg q c p i : SailM Unit) := by
  unfold dispatch_cfg
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact agt_configure_bound ports _ _ _ _
    | exact agt_get_bound ports _ _
    | exact actual_agt_set_bound_any _ _ _ idle_at
    | exact actual_agt_set_bound_any _ _ _ (opened_at _)
    | exact dispatch_check_bound ports _ _ _ _
    | exact internal_error_bound ports _ _ _
    | exact report_bound ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem loop_clash_ref_bound (ports : Ports) (q : Nat) (p : Nat) (s : Int) (ss : Int) : Inv (Bound cycle) (loop_clash_ref q p s ss : SailM (instruction × Int × Int)) := by
  unfold loop_clash_ref
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact agt_get_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem due_ref_bound (ports : Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Inv (Bound cycle) (due_ref q p st : SailM (Option (instruction × Int × Int))) := by
  unfold due_ref
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact loop_clash_ref_bound ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_due_bound (ports : Ports) (q : Nat) (p : Nat) (c : Nat) (parked : Bool) : Inv (Bound cycle) (agt_due q p c parked : SailM (Option ((Vector agt_level 4) × Nat × Int × Int))) := by
  unfold agt_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact agt_get_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem due_refs_bound (ports : Ports) (q : Nat) (c : Nat) (parked : Bool) : Inv (Bound cycle) (due_refs q c parked : SailM (Vector (Option ((Vector agt_level 4) × Nat × Int × Int)) 2)) := by
  unfold due_refs
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact agt_due_bound ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_path_bound (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv (Bound cycle) (hold_path q p c : SailM Unit) := by
  unfold hold_path
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem loop_tag_bound (ports : Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Inv (Bound cycle) (loop_tag q p st : SailM (Option ref_tag)) := by
  unfold loop_tag
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact agt_get_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_refresh_bound (ports : Ports) (q : Nat) (p : Nat) : Inv (Bound cycle) (agt_refresh q p : SailM Unit) :=by
 exact actual_refresh_bound_any q p
theorem run_loop_bound (ports : Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) (blocked : Bool) (partner : (Option (instruction × Int × Int))) : Inv (Bound cycle) (run_loop q cycle p st blocked partner : SailM Unit) :=by
 exact actual_run_loop_bound (execution_ports ports) q p st blocked partner
theorem consumeNewRef_bound (ports : Ports) (q : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) : Inv (Bound cycle) (GrantConsumptionProofs.consumeNewRef q cycle r loop lref go bank k issued last) := by
  unfold GrantConsumptionProofs.consumeNewRef
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact actual_agt_set_bound_any _ _ _ idle_at
    | exact actual_agt_set_bound_any _ _ _ (opened_at _)
    | exact bank_report_bound ports _ _ _ _
    | exact cfg_path_bound ports _
    | exact dispatch_cfg_bound ports _ _ _ _
    | exact dispatch_check_bound ports _ _ _ _
    | exact due_ref_bound ports _ _ _
    | exact due_refs_bound ports _ _ _
    | exact hold_path_bound ports _ _ _
    | exact loop_tag_bound ports _ _ _
    | exact report_bound ports _ _ _
    | exact reserve_local_fetch_bound ports _ _ _
    | exact run_element_bound ports _ _ _ _ _ _
    | exact run_loop_bound ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem consumeFetch_bound (ports : Ports) (q : Nat) (r : cycle_recs) (fm_ok fm_clash : Bool)
    (issued : Nat) (last : Option instruction) : Inv (Bound cycle) (GrantConsumptionProofs.consumeFetch q cycle r fm_ok fm_clash issued last) := by
  unfold GrantConsumptionProofs.consumeFetch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact actual_agt_set_bound_any _ _ _ idle_at
    | exact actual_agt_set_bound_any _ _ _ (opened_at _)
    | exact bank_report_bound ports _ _ _ _
    | exact cfg_path_bound ports _
    | exact dispatch_cfg_bound ports _ _ _ _
    | exact dispatch_check_bound ports _ _ _ _
    | exact due_ref_bound ports _ _ _
    | exact due_refs_bound ports _ _ _
    | exact hold_path_bound ports _ _ _
    | exact loop_tag_bound ports _ _ _
    | exact report_bound ports _ _ _
    | exact reserve_local_fetch_bound ports _ _ _
    | exact run_element_bound ports _ _ _ _ _ _
    | exact run_loop_bound ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_slots_bound (ports : Ports) (q : Nat) (r : cycle_recs) (parked : Bool) : Inv (Bound cycle) (dispatch_slots q cycle r parked : SailM Unit) := by
  rw [GrantConsumptionProofs.actual_dispatch_factory_source]
  unfold GrantConsumptionProofs.dispatchFactory
  repeat' first
    | exact consumeNewRef_bound ports _ _ _ _ _ _ _ _ _
    | exact consumeFetch_bound ports _ _ _ _ _ _
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact actual_agt_set_bound_any _ _ _ idle_at
    | exact actual_agt_set_bound_any _ _ _ (opened_at _)
    | exact bank_report_bound ports _ _ _ _
    | exact cfg_path_bound ports _
    | exact dispatch_cfg_bound ports _ _ _ _
    | exact dispatch_check_bound ports _ _ _ _
    | exact due_ref_bound ports _ _ _
    | exact due_refs_bound ports _ _ _
    | exact hold_path_bound ports _ _ _
    | exact loop_tag_bound ports _ _ _
    | exact report_bound ports _ _ _
    | exact reserve_local_fetch_bound ports _ _ _
    | exact run_element_bound ports _ _ _ _ _ _
    | exact run_loop_bound ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_bound (ports : Ports) (q : Nat) (r : cycle_recs) (parked : Bool) : Inv (Bound cycle) (dispatch q cycle r parked : SailM Unit) := by
  unfold dispatch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact dispatch_decoded_bound ports _ _ _
    | exact dispatch_slots_bound ports _ _ _
    | exact due_ref_bound ports _ _ _
    | exact due_refs_bound ports _ _ _
    | exact loop_tag_bound ports _ _ _
    | exact report_bound ports _ _ _
    | exact run_loop_bound ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem strict_held_bound (ports : Ports) (q : Nat) (c : Nat) (xs : (List (Result instruction diag))) (held : Bool) : Inv (Bound cycle) (strict_held q c xs held : SailM Bool) := by
  induction xs generalizing q c held
  case nil =>
    unfold strict_held
    repeat' first
      | exact ih _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact hold_check_bound ports _ _ _ _ _
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
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact hold_check_bound ports _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem parcel_held_bound (ports : Ports) (q : Nat) (c : Nat) (r : cycle_recs) (items : (List (Result instruction diag))) : Inv (Bound cycle) (parcel_held q c r items : SailM Bool) := by
  unfold parcel_held
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact cfg_path_bound ports _
    | exact held_bound ports _
    | exact hold_check_bound ports _ _ _ _ _
    | exact strict_held_bound ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem tick_queue_bound (ports : Ports) (q : Nat) : Inv (Bound cycle) (tick_queue q cycle : SailM Unit) := by
  unfold tick_queue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact agt_finish_config_bound ports _ _
    | exact agt_get_bound ports _ _
    | exact deskewrecv_ready_bound ports _ _
    | exact dispatch_bound ports _ _ _
    | exact host_parked_bound ports _ _
    | exact iq_pop_bound ports _ _
    | exact iq_text_check_bound ports _ _ _
    | exact occupancy_bound ports _ _
    | exact parcel_earliest_bound ports _ _
    | exact parcel_held_bound ports _ _ _ _
    | exact parcel_scan_bound ports _
    | exact polling_bound ports _
    | exact queue_parked_bound ports _ _
    | exact repeat_issue_bound ports _ _ _
    | exact run_element_bound ports _ _ _ _ _ _
    | exact registers_bound _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_bound (ports : Ports) (e : mxm_staged) : Inv (Bound cycle) (mxm_apply e : SailM Unit) := by
  unfold mxm_apply
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_ok_bound (ports : Ports) (xs : (List mxm_staged)) (bad : (List Nat)) : Inv (Bound cycle) (mxm_apply_ok xs bad : SailM Unit) := by
  induction xs generalizing bad
  case nil =>
    unfold mxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact mxm_apply_bound ports _
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
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | exact mxm_apply_bound ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem sxm_apply_ok_bound (ports : Ports) (xs : (List sxm_staged)) (bad : (List Nat)) : Inv (Bound cycle) (sxm_apply_ok xs bad : SailM Unit) := by
  induction xs generalizing bad
  case nil =>
    unfold sxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
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
      | exact throw_bound _
      | exact read_inv _ _
      | exact other_write _ _ (by decide) (by decide)
      | exact r_write _ array_idle_at
      | exact w_write _ array_idle_at
      | exact assert_bound _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem unit_state_flush_bound (ports : Ports) (c : Nat) : Inv (Bound cycle) (unit_state_flush c : SailM Unit) := by
  unfold unit_state_flush
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact mxm_apply_ok_bound ports _ _
    | exact report_bound ports _ _ _
    | exact sxm_apply_ok_bound ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem tick_bound (ports : Ports) : Inv (Bound cycle) (tick cycle : SailM Unit) := by
  unfold tick
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_bound _
    | exact read_inv _ _
    | exact other_write _ _ (by decide) (by decide)
    | exact r_write _ array_idle_at
    | exact w_write _ array_idle_at
    | exact assert_bound _ _
    | exact barrier_retire_bound ports _
    | exact commit_due_bound ports _
    | exact control_bound ports _
    | exact enqueue_bound ports _ _
    | exact execute_due_bound ports _
    | exact local_fetch_complete_bound ports _
    | exact scrub_run_bound ports _
    | exact stream_fetch_complete_bound ports _
    | exact tick_queue_bound ports _
    | exact unit_state_flush_bound ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
end AgtTimeClosureProofs
