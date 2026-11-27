import Proofs.QueueConsumption
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
open ResourceFrameProofs ResourceInvariantProofs
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000
namespace Proofs.HistoryFrames
abbrev State := ResourceFrameProofs.St
def View (s : State) := (s.regs.get? Register.iq_head,s.regs.get? Register.iq_len,ResourceExecuteFrameProofs.IQMem s)
theorem project_frame {α : Type} (a : SailM α)
 (h : Keeps ResourceExecuteFrameProofs.RView a) : Keeps View a := by
 intro s
 have hp:=h s
 cases run:a.run s <;> simp only [run] at hp ⊢
 all_goals exact congrArg Prod.snd hp

theorem write_frame (r : Register) (v : RegisterType r)
 (hh : r≠Register.iq_head) (hl : r≠Register.iq_len) : Keeps View (Out.writeReg r v) := by
 apply pair_keeps
 · exact unrelated_register_write _ _ _ (Ne.symm hh)
 · apply pair_keeps
   · exact unrelated_register_write _ _ _ (Ne.symm hl)
   · apply write_keeps
     intro s
     rfl

theorem throw_frame {α : Type} (e : Error exception) : Keeps View (throw e : SailM α) := by
 intro s
 rfl
attribute [local irreducible] ResourceFrameProofs.Keeps
attribute [local irreducible] GetElem?.getElem! Inhabited.default Bind.bind Pure.pure EStateM.bind EStateM.pure PreSail.readReg PreSail.writeReg IntRange.forIn'
attribute [local irreducible] Out.Functions.__id Out.Functions.abc_after Out.Functions.abc_at Out.Functions.abc_fp8 Out.Functions.abc_others Out.Functions.abc_others_fp8 Out.Functions.abc_pair_pick Out.Functions.abc_prune Out.Functions.abc_types_ok Out.Functions.acc_mode_name_backwards_matches Out.Functions.acc_mode_name_forwards Out.Functions.acc_mode_name_forwards_matches Out.Functions.acc_mode_of_num Out.Functions.acctype_declared Out.Functions.acctype_name_backwards_matches Out.Functions.acctype_name_forwards Out.Functions.acctype_name_forwards_matches Out.Functions.addr_bank Out.Functions.agt_advance Out.Functions.agt_advance_test Out.Functions.agt_after_reference Out.Functions.agt_config Out.Functions.agt_configure_pure Out.Functions.agt_due_pure Out.Functions.agt_field_of_num Out.Functions.agt_field_pos Out.Functions.agt_finish_config_pure Out.Functions.agt_nlevels Out.Functions.agt_opened Out.Functions.agt_refresh_pure Out.Functions.agt_state_of_num Out.Functions.aligned Out.Functions.all_ones Out.Functions.append_element Out.Functions.assembly_forwards_matches Out.Functions.bank_rule Out.Functions.banks_meet Out.Functions.bf16_dot Out.Functions.bfp8_exact Out.Functions.binop_entry_backwards_matches Out.Functions.binop_entry_forwards Out.Functions.binop_entry_forwards_matches Out.Functions.binop_legal Out.Functions.bit_index Out.Functions.bit_of Out.Functions.bits_of_s Out.Functions.bits_t Out.Functions.blk_exp Out.Functions.bool_bits Out.Functions.c2c_op_of_num Out.Functions.c2c_packet_flits Out.Functions.c2c_poison_int Out.Functions.c2c_ptype_code Out.Functions.c2c_ptype_of_num Out.Functions.c2c_ptype_queued Out.Functions.c2c_rx Out.Functions.c2c_tx Out.Functions.canon_nan Out.Functions.cfg_path_pick Out.Functions.chain_suffix_backwards_matches Out.Functions.chain_suffix_forwards Out.Functions.chain_suffix_forwards_matches Out.Functions.chk_elem Out.Functions.chk_wrow Out.Functions.clash_addr Out.Functions.clash_stream Out.Functions.classify_items Out.Functions.clears_repeat Out.Functions.cmp_legal Out.Functions.cmp_mnemonic_backwards_matches Out.Functions.cmp_mnemonic_forwards Out.Functions.cmp_mnemonic_forwards_matches Out.Functions.commit_scan Out.Functions.commit_wins Out.Functions.config_ends Out.Functions.config_for Out.Functions.conv_dst Out.Functions.conv_entry_backwards_matches Out.Functions.conv_entry_forwards Out.Functions.conv_entry_forwards_matches Out.Functions.conv_mnemonic_backwards_matches Out.Functions.conv_mnemonic_forwards Out.Functions.conv_mnemonic_forwards_matches Out.Functions.conv_src Out.Functions.conv_src_type Out.Functions.conv_types Out.Functions.csr_value Out.Functions.db_idx Out.Functions.dec_bits_10_backwards Out.Functions.dec_bits_10_backwards_matches Out.Functions.dec_bits_1_backwards Out.Functions.dec_bits_1_backwards_matches Out.Functions.dec_bits_2_backwards Out.Functions.dec_bits_2_backwards_matches Out.Functions.dec_bits_32_backwards Out.Functions.dec_bits_32_backwards_matches Out.Functions.dec_bits_3_backwards Out.Functions.dec_bits_3_backwards_matches Out.Functions.dec_bits_4_backwards Out.Functions.dec_bits_4_backwards_matches Out.Functions.dec_bits_5_backwards Out.Functions.dec_bits_5_backwards_matches Out.Functions.dec_bits_6_backwards Out.Functions.dec_bits_6_backwards_matches Out.Functions.dec_bits_7_backwards Out.Functions.dec_bits_7_backwards_matches Out.Functions.dec_bits_8_backwards Out.Functions.dec_bits_8_backwards_matches Out.Functions.dec_bits_9_backwards Out.Functions.dec_bits_9_backwards_matches Out.Functions.dec_bits_backwards Out.Functions.dec_bits_backwards_matches Out.Functions.dec_bits_forwards Out.Functions.dec_bits_forwards_matches Out.Functions.def_spc_backwards Out.Functions.def_spc_backwards_matches Out.Functions.def_spc_forwards Out.Functions.def_spc_forwards_matches Out.Functions.diag_code_backwards_matches Out.Functions.diag_code_forwards Out.Functions.diag_code_forwards_matches Out.Functions.diag_idx Out.Functions.diag_is_data Out.Functions.diag_number Out.Functions.diag_of_num Out.Functions.drop_commits Out.Functions.dst_split_backwards_matches Out.Functions.dst_split_forwards Out.Functions.dst_split_forwards_matches Out.Functions.dsum_round Out.Functions.dsum_round_u Out.Functions.dsum_wide Out.Functions.earliest Out.Functions.earliest_int Out.Functions.ecc_class_of_num Out.Functions.ecc_classify Out.Functions.ecc_correct Out.Functions.ecc_data_index Out.Functions.ecc_encode Out.Functions.ecc_flip_delta Out.Functions.ecc_is_pow2 Out.Functions.ecc_log2 Out.Functions.ecc_ov_get Out.Functions.ecc_ov_set Out.Functions.ecc_par10 Out.Functions.ecc_poison_ov Out.Functions.ecc_poison_sl Out.Functions.ecc_pos Out.Functions.elem Out.Functions.elem_at Out.Functions.eltype_name_backwards_matches Out.Functions.eltype_name_forwards Out.Functions.eltype_name_forwards_matches Out.Functions.eltype_of_num Out.Functions.eltype_width Out.Functions.empty_plane Out.Functions.encdec_acc_mode_backwards Out.Functions.encdec_acc_mode_backwards_matches Out.Functions.encdec_acc_mode_forwards Out.Functions.encdec_acc_mode_forwards_matches Out.Functions.encdec_acctype_backwards_matches Out.Functions.encdec_acctype_forwards Out.Functions.encdec_acctype_forwards_matches Out.Functions.encdec_c2c_op_backwards_matches Out.Functions.encdec_c2c_op_forwards Out.Functions.encdec_c2c_op_forwards_matches Out.Functions.encdec_cmp_mode_backwards_matches Out.Functions.encdec_cmp_mode_forwards Out.Functions.encdec_cmp_mode_forwards_matches Out.Functions.encdec_eltype_backwards_matches Out.Functions.encdec_eltype_forwards Out.Functions.encdec_eltype_forwards_matches Out.Functions.encdec_mapop_backwards_matches Out.Functions.encdec_mapop_forwards Out.Functions.encdec_mapop_forwards_matches Out.Functions.encdec_mem_dir_backwards_matches Out.Functions.encdec_mem_dir_forwards Out.Functions.encdec_mem_dir_forwards_matches Out.Functions.encdec_reduce_op_backwards_matches Out.Functions.encdec_reduce_op_forwards Out.Functions.encdec_reduce_op_forwards_matches Out.Functions.encdec_shift_dir_backwards Out.Functions.encdec_shift_dir_backwards_matches Out.Functions.encdec_shift_dir_forwards Out.Functions.encdec_shift_dir_forwards_matches Out.Functions.encdec_vxm_binop_backwards_matches Out.Functions.encdec_vxm_binop_forwards Out.Functions.encdec_vxm_binop_forwards_matches Out.Functions.encdec_vxm_conv_backwards_matches Out.Functions.encdec_vxm_conv_forwards Out.Functions.encdec_vxm_conv_forwards_matches Out.Functions.encdec_vxm_unop_backwards_matches Out.Functions.encdec_vxm_unop_forwards Out.Functions.encdec_vxm_unop_forwards_matches Out.Functions.entry_latency Out.Functions.entry_latency_typed Out.Functions.entry_of Out.Functions.execute_C2C_DESKEW Out.Functions.execute_C2C_DESKEWRECV Out.Functions.execute_C2C_HOSTWAIT Out.Functions.execute_C2C_RTDESKEW Out.Functions.execute_ILLEGAL32 Out.Functions.execute_ILLEGAL64 Out.Functions.execute_MEM_COUNTDOWN Out.Functions.execute_MEM_DEEPSLEEP Out.Functions.execute_MEM_ITERATIONS Out.Functions.execute_MEM_POWERCONFIG Out.Functions.execute_MEM_STEP Out.Functions.execute_MEM_STREAMSTEP Out.Functions.execute_NOP Out.Functions.execute_NOTIFY Out.Functions.execute_REPEAT Out.Functions.execute_SYNC Out.Functions.exp_iv Out.Functions.exp_small Out.Functions.extensions Out.Functions.f16_dot Out.Functions.f16_exact Out.Functions.f16_row Out.Functions.f16_row_bad Out.Functions.f16_stream_values Out.Functions.fadd_core Out.Functions.fclass_eq Out.Functions.fclass_of_num Out.Functions.fcmp Out.Functions.fdecode Out.Functions.fetch_profile_causal Out.Functions.fetch_tile_available Out.Functions.fetchm_bank_note Out.Functions.fetchm_hold_until Out.Functions.field Out.Functions.fill_append Out.Functions.fill_len Out.Functions.fill_push_unit Out.Functions.fill_units_append Out.Functions.find_override Out.Functions.flip96 Out.Functions.fmt_of Out.Functions.fp8_dot Out.Functions.fp8_exact Out.Functions.fp8_format_name_backwards_matches Out.Functions.fp8_format_name_forwards Out.Functions.fp8_format_name_forwards_matches Out.Functions.fp_abs Out.Functions.fp_add Out.Functions.fp_convert Out.Functions.fp_max Out.Functions.fp_min Out.Functions.fp_mul Out.Functions.fp_neg Out.Functions.fp_of_int Out.Functions.fp_sub Out.Functions.fp_tr_value Out.Functions.from_bytes_le Out.Functions.fround Out.Functions.fval_int Out.Functions.fx_of Out.Functions.gap_or_none Out.Functions.has_extension Out.Functions.hex_bits_10_backwards_matches Out.Functions.hex_bits_11_backwards_matches Out.Functions.hex_bits_12_backwards_matches Out.Functions.hex_bits_13_backwards_matches Out.Functions.hex_bits_14_backwards_matches Out.Functions.hex_bits_15_backwards_matches Out.Functions.hex_bits_16_backwards_matches Out.Functions.hex_bits_17_backwards_matches Out.Functions.hex_bits_18_backwards_matches Out.Functions.hex_bits_19_backwards_matches Out.Functions.hex_bits_1_backwards_matches Out.Functions.hex_bits_20_backwards_matches Out.Functions.hex_bits_21_backwards_matches Out.Functions.hex_bits_22_backwards_matches Out.Functions.hex_bits_23_backwards_matches Out.Functions.hex_bits_24_backwards_matches Out.Functions.hex_bits_25_backwards_matches Out.Functions.hex_bits_26_backwards_matches Out.Functions.hex_bits_27_backwards_matches Out.Functions.hex_bits_28_backwards_matches Out.Functions.hex_bits_29_backwards_matches Out.Functions.hex_bits_2_backwards_matches Out.Functions.hex_bits_30_backwards_matches Out.Functions.hex_bits_31_backwards_matches Out.Functions.hex_bits_32_backwards_matches Out.Functions.hex_bits_33_backwards_matches Out.Functions.hex_bits_34_backwards_matches Out.Functions.hex_bits_35_backwards_matches Out.Functions.hex_bits_36_backwards_matches Out.Functions.hex_bits_37_backwards_matches Out.Functions.hex_bits_38_backwards_matches Out.Functions.hex_bits_39_backwards_matches Out.Functions.hex_bits_3_backwards_matches Out.Functions.hex_bits_40_backwards_matches Out.Functions.hex_bits_41_backwards_matches Out.Functions.hex_bits_42_backwards_matches Out.Functions.hex_bits_43_backwards_matches Out.Functions.hex_bits_44_backwards_matches Out.Functions.hex_bits_45_backwards_matches Out.Functions.hex_bits_46_backwards_matches Out.Functions.hex_bits_47_backwards_matches Out.Functions.hex_bits_48_backwards_matches Out.Functions.hex_bits_49_backwards_matches Out.Functions.hex_bits_4_backwards_matches Out.Functions.hex_bits_50_backwards_matches Out.Functions.hex_bits_51_backwards_matches Out.Functions.hex_bits_52_backwards_matches Out.Functions.hex_bits_53_backwards_matches Out.Functions.hex_bits_54_backwards_matches Out.Functions.hex_bits_55_backwards_matches Out.Functions.hex_bits_56_backwards_matches Out.Functions.hex_bits_57_backwards_matches Out.Functions.hex_bits_58_backwards_matches Out.Functions.hex_bits_59_backwards_matches Out.Functions.hex_bits_5_backwards_matches Out.Functions.hex_bits_60_backwards_matches Out.Functions.hex_bits_61_backwards_matches Out.Functions.hex_bits_62_backwards_matches Out.Functions.hex_bits_63_backwards_matches Out.Functions.hex_bits_64_backwards_matches Out.Functions.hex_bits_6_backwards_matches Out.Functions.hex_bits_7_backwards_matches Out.Functions.hex_bits_8_backwards_matches Out.Functions.hex_bits_9_backwards_matches Out.Functions.hex_bits_backwards Out.Functions.hex_bits_backwards_matches Out.Functions.hex_bits_forwards Out.Functions.hex_bits_forwards_matches Out.Functions.hold_decide Out.Functions.holds_f16 Out.Functions.i8_tr_value Out.Functions.identity_map Out.Functions.idiv_ceil Out.Functions.idiv_floor Out.Functions.ilog2 Out.Functions.in_barrier_window Out.Functions.inf_bits Out.Functions.int8_dot Out.Functions.int_of Out.Functions.int_of_fp Out.Functions.inward_row Out.Functions.iq_padding Out.Functions.is_bool Out.Functions.is_dst Out.Functions.is_float Out.Functions.is_int Out.Functions.is_loopable Out.Functions.is_mxm_queue Out.Functions.is_nan Out.Functions.is_none Out.Functions.is_pair Out.Functions.is_prm Out.Functions.is_sel Out.Functions.is_shf Out.Functions.is_some Out.Functions.is_tsp Out.Functions.is_unsigned Out.Functions.isqrt Out.Functions.issued_by Out.Functions.issuer_of Out.Functions.items_len Out.Functions.k_binary Out.Functions.k_cmp Out.Functions.k_convert Out.Functions.k_convert_any Out.Functions.k_nonzero Out.Functions.k_unary Out.Functions.lane8 Out.Functions.lane_byte Out.Functions.latency_of Out.Functions.latency_of_instr Out.Functions.link_flit_mask Out.Functions.link_pow2 Out.Functions.lint Out.Functions.list_len Out.Functions.ln2_iv Out.Functions.log_iv Out.Functions.mapop_name_backwards_matches Out.Functions.mapop_name_forwards Out.Functions.mapop_name_forwards_matches Out.Functions.mem_dir_mnemonic_backwards_matches Out.Functions.mem_dir_mnemonic_forwards Out.Functions.mem_dir_mnemonic_forwards_matches Out.Functions.mem_quad Out.Functions.mem_slice_present Out.Functions.mem_unit Out.Functions.merge_tiles Out.Functions.model_link_flit_mask Out.Functions.model_link_flits Out.Functions.model_ptype_queued Out.Functions.mxm_acctype_of_num Out.Functions.mxm_bad Out.Functions.mxm_cap Out.Functions.mxm_check Out.Functions.mxm_clash Out.Functions.mxm_part_of_num Out.Functions.mxm_queue_present Out.Functions.mxm_unit Out.Functions.mxm_wtype_of_num Out.Functions.nbits_of Out.Functions.neq_bool Out.Functions.nonempty_list Out.Functions.not Out.Functions.not_bit Out.Functions.num_of_acc_mode Out.Functions.num_of_agt_field Out.Functions.num_of_agt_state Out.Functions.num_of_c2c_op Out.Functions.num_of_c2c_ptype Out.Functions.num_of_diag Out.Functions.num_of_ecc_class Out.Functions.num_of_eltype Out.Functions.num_of_fclass Out.Functions.num_of_mxm_acctype Out.Functions.num_of_mxm_part Out.Functions.num_of_mxm_wtype Out.Functions.num_of_row Out.Functions.num_of_scrub_action Out.Functions.num_of_srf_lookup Out.Functions.num_of_timing_class Out.Functions.num_of_unit_kind Out.Functions.one_plane Out.Functions.opt_spc_backwards Out.Functions.opt_spc_backwards_matches Out.Functions.opt_spc_forwards Out.Functions.opt_spc_forwards_matches Out.Functions.or_pair Out.Functions.outward_row Out.Functions.ov_nth Out.Functions.pair_exact Out.Functions.path_hold_until Out.Functions.path_views Out.Functions.place Out.Functions.place_all Out.Functions.poll_epoch Out.Functions.pow2i Out.Functions.q_in Out.Functions.queue_present Out.Functions.ras_idx Out.Functions.ras_loc Out.Functions.ras_sat_inc Out.Functions.read_gap Out.Functions.reduce_mnemonic_backwards_matches Out.Functions.reduce_mnemonic_forwards Out.Functions.reduce_mnemonic_forwards_matches Out.Functions.ref_bank Out.Functions.ref_path Out.Functions.ref_thread Out.Functions.rep_sp_set Out.Functions.repeat_after Out.Functions.repeat_plan Out.Functions.repeatable Out.Functions.rev_items Out.Functions.ring Out.Functions.rne_int Out.Functions.round_iv Out.Functions.row_of_num Out.Functions.rows_of Out.Functions.s_of_bits Out.Functions.sail_mask Out.Functions.sail_ones Out.Functions.same_target Out.Functions.sat_s Out.Functions.sat_t Out.Functions.scrub_action_of_num Out.Functions.scrub_arbitrate Out.Functions.scrub_due_count Out.Functions.scrub_slot_taken Out.Functions.sep_forwards Out.Functions.sep_forwards_matches Out.Functions.set_lane_byte Out.Functions.sg4_suffix_backwards_matches Out.Functions.sg4_suffix_forwards Out.Functions.sg4_suffix_forwards_matches Out.Functions.shift_dir_name_backwards_matches Out.Functions.shift_dir_name_forwards Out.Functions.shift_dir_name_forwards_matches Out.Functions.slice_mask Out.Functions.slot_of Out.Functions.spc_backwards Out.Functions.spc_backwards_matches Out.Functions.spc_forwards Out.Functions.spc_forwards_matches Out.Functions.sram_addr_of Out.Functions.sram_bank Out.Functions.sram_field_backwards Out.Functions.sram_field_backwards_matches Out.Functions.sram_field_forwards Out.Functions.sram_field_forwards_matches Out.Functions.sram_index Out.Functions.sram_text_backwards Out.Functions.sram_text_backwards_matches Out.Functions.sram_text_forwards Out.Functions.sram_text_forwards_matches Out.Functions.srf_lookup_of_num Out.Functions.srf_ov_spare Out.Functions.srf_rep_merge Out.Functions.srf_slot_best Out.Functions.srf_slot_chain_ov Out.Functions.srf_slot_collides Out.Functions.srf_slot_flip Out.Functions.srf_slot_read Out.Functions.srf_slot_read_ov Out.Functions.srf_slot_write Out.Functions.srf_stream_only_downstream Out.Functions.stream_id Out.Functions.stream_name_forwards_matches Out.Functions.stream_row Out.Functions.string_of_exception Out.Functions.sxm_bad Out.Functions.sxm_clash Out.Functions.sxm_lane Out.Functions.sxm_map_distinct Out.Functions.sxm_or_kind Out.Functions.sxm_set_lane Out.Functions.sxm_stream_plus Out.Functions.sxm_unit_at Out.Functions.tag_of Out.Functions.tags_clash Out.Functions.text_width Out.Functions.tile_active Out.Functions.tile_addr Out.Functions.timing_class_name Out.Functions.timing_class_of_num Out.Functions.timing_of Out.Functions.timing_overrides Out.Functions.timing_type_of Out.Functions.to_bytes_le Out.Functions.tr_eval Out.Functions.tr_eval_int Out.Functions.tr_interval Out.Functions.ty_split_backwards_matches Out.Functions.ty_split_forwards Out.Functions.ty_split_forwards_matches Out.Functions.ty_valid Out.Functions.unit_kind_of_num Out.Functions.unop_entry_backwards_matches Out.Functions.unop_entry_forwards Out.Functions.unop_entry_forwards_matches Out.Functions.unop_legal Out.Functions.vec_units Out.Functions.view_free Out.Functions.view_paths_busy Out.Functions.vsrc_name_forwards_matches Out.Functions.vxm_binop_mnemonic_backwards_matches Out.Functions.vxm_binop_mnemonic_forwards Out.Functions.vxm_binop_mnemonic_forwards_matches Out.Functions.vxm_unit Out.Functions.vxm_unop_mnemonic_backwards_matches Out.Functions.vxm_unop_mnemonic_forwards Out.Functions.vxm_unop_mnemonic_forwards_matches Out.Functions.wait_reason Out.Functions.wants_pair Out.Functions.wlane_add Out.Functions.wlane_read Out.Functions.words_to_planes Out.Functions.wrap_s Out.Functions.wrap_t Out.Functions.write_gap Out.Functions.writer_before Out.Functions.wtype_name_backwards_matches Out.Functions.wtype_name_forwards Out.Functions.wtype_name_forwards_matches Out.Functions.wtype_ok Out.Functions.wtype_split_backwards_matches Out.Functions.wtype_split_forwards Out.Functions.wtype_split_forwards_matches Out.Functions.wtype_valid Out.Functions.zero_bits Out.abs_real Out.add_real Out.diag_callback Out.div_real Out.ecc_event_callback Out.eq_real Out.gt_real Out.gteq_real Out.lint_callback Out.lt_real Out.lteq_real Out.mult_real Out.mxm_corr_callback Out.neg_real Out.prerr_real Out.print_real Out.random_real Out.real_power Out.round_down Out.round_up Out.scrub_event_callback Out.spare_event_callback Out.sqrt_real Out.sram_write_callback Out.srf_write_callback Out.sub_real Out.tdiv_int Out.tmod_int Out.to_real ecc_begin exec_cycle host_deliver host_msix execute_C2C_HOSTNOTIFY commit_at ecc_poison_mask ecc_result_ov plane_stream write_planes_fwd write_planes execute_C2C_PCIRECV pci_busy ras_record ras_ecc_event ras_spare_event get_stream srf_spare srf_s03 ecc_check_value srf_read_chain_ov srf_only_downstream srf_read_ov read_planes_full read_planes_core read_planes execute_C2C_PCISEND c2c_link c2c_peer_ok execute_C2C_RECEIVE c2c_busy c2c_put execute_C2C_SEND execute_C2C_TRANSMIT execute_CONFIG stream_fetch_arm stream_fetch_start execute_IFETCH execute_IFETCHM commit_cycle eff_addr eff_stream read_cycle read_plane_raw sram_read sram_read_ov write_sram_ecc write_sram_word sram_write_ov tile_on execute_MEM_DIRECT ecc_check_sl srf_report mem_report execute_MEM_GATHER execute_MEM_SCATTER execute_MEM_SETSEGMENT macc_busy_in macc_busy mxm_stage_acc mxm_stage plane mxm_plane_at plane_blocked ras_mxm_event abc_exec execute_MXM_ABC execute_MXM_ABC8 execute_MXM_ACC execute_MXM_IW read_planes16 execute_MXM_IWS execute_MXM_LW execute_SETVL sx sxm_load_map sxm_rd_cycle execute_SXM_LOADMAP execute_SXM_LOADMAPWIDE sxm_cm_cycle sxm_emit execute_SXM_MAP execute_SXM_ROTATE execute_SXM_SELECT execute_SXM_SHIFT execute_SXM_TRANSPOSE Out.sailThrow internal_error fp_transcendental i8_transcendental apply vxm_commit_cycle vxm_read_cycle vxm_read vxm_lanes execute_VXM_BIN execute_VXM_CMP execute_VXM_CONV execute_VXM_CVT execute_VXM_MASK execute_VXM_REDUCE execute_VXM_SELECT execute_VXM_UN execute diag_record diag_report report execute_element run_element agt_get agt_set agt_configure host_parked path_busy path_tail path_view_of polling parked_one queue_parked hold_view_of hold_check dispatch_check dispatch_cfg agt_due due_refs agt_refresh bank_report hold_path loop_clash_ref run_loop
theorem iq_fetch_ov_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (t : Nat) (a : vec) (b : vec) (ova : ecc_ov) (ovb : ecc_ov) : Keeps View (iq_fetch_ov q t a b ova ovb : SailM Bool) := by
  unfold iq_fetch_ov
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem local_fetch_complete_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) : Keeps View (local_fetch_complete c : SailM Unit) := by
  unfold local_fetch_complete
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.diag_report_frame ports _ _ _)
    | apply iq_fetch_ov_frame ports
    | exact project_frame _ (ports.read_sram_ecc _ _)
    | exact project_frame _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem barrier_retire_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) : Keeps View (barrier_retire c : SailM Unit) := by
  unfold barrier_retire
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem set_stream_frame (ports : ResourceExecuteFrameProofs.Ports) (r : row) (s : Nat) (v : (Vector (Vector Out.srf_write 44) 64)) : Keeps View (set_stream r s v : SailM Unit) := by
  unfold set_stream
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem srf_write_ov_frame (ports : ResourceExecuteFrameProofs.Ports) (r : row) (s : Nat) (g : Nat) (c : Nat) (v : vec) (ov : ecc_ov) : Keeps View (srf_write_ov r s g c v ov : SailM Bool) := by
  unfold srf_write_ov
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.get_stream_frame ports _ _)
    | apply set_stream_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem commit_apply_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) (all : (List commit)) (xs : (List commit)) (i : Nat) (hits : (Vector Nat 145)) : Keeps View (commit_apply c all xs i hits : SailM (Vector Nat 145)) := by
  induction xs generalizing i hits
  case nil =>
    unfold commit_apply
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | apply srf_write_ov_frame ports
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold commit_apply
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | apply srf_write_ov_frame ports
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
theorem commit_due_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) : Keeps View (commit_due c : SailM Unit) := by
  unfold commit_due
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.report_frame ports _ _ _)
    | apply commit_apply_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem execute_all_frame (ports : ResourceExecuteFrameProofs.Ports) (xs : (List exec_element)) : Keeps View (execute_all xs : SailM Unit) := by
  induction xs
  case nil =>
    unfold execute_all
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | exact project_frame _ (ResourceExecuteFrameProofs.execute_element_frame ports _)
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold execute_all
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | exact project_frame _ (ResourceExecuteFrameProofs.execute_element_frame ports _)
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
theorem execute_due_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) : Keeps View (execute_due c : SailM Unit) := by
  unfold execute_due
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | apply execute_all_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_take_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) : Keeps View (stream_fetch_take q c : SailM (Option stream_fetch)) := by
  unfold stream_fetch_take
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_service_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (c : Nat) : Keeps View (stream_fetch_service q c : SailM Unit) := by
  unfold stream_fetch_service
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.ecc_begin_frame ports _)
    | exact project_frame _ (ResourceExecuteFrameProofs.read_planes_frame ports _ _ _)
    | exact project_frame _ (ResourceExecuteFrameProofs.diag_report_frame ports _ _ _)
    | apply iq_fetch_ov_frame ports
    | apply stream_fetch_take_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_complete_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) : Keeps View (stream_fetch_complete c : SailM Unit) := by
  unfold stream_fetch_complete
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | apply stream_fetch_service_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_frame (ports : ResourceExecuteFrameProofs.Ports) (e : mxm_staged) : Keeps View (mxm_apply e : SailM Unit) := by
  unfold mxm_apply
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_ok_frame (ports : ResourceExecuteFrameProofs.Ports) (xs : (List mxm_staged)) (bad : (List Nat)) : Keeps View (mxm_apply_ok xs bad : SailM Unit) := by
  induction xs
  case nil =>
    unfold mxm_apply_ok
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | apply mxm_apply_frame ports
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold mxm_apply_ok
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | apply mxm_apply_frame ports
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
theorem sxm_apply_ok_frame (ports : ResourceExecuteFrameProofs.Ports) (xs : (List sxm_staged)) (bad : (List Nat)) : Keeps View (sxm_apply_ok xs bad : SailM Unit) := by
  induction xs
  case nil =>
    unfold sxm_apply_ok
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold sxm_apply_ok
    repeat' first
      | apply ih
      | assumption
      | exact pure_keeps _ _
      | exact throw_frame _
      | exact read_keeps _ _
      | exact write_frame _ _ (by decide) (by decide)
      | reduce_resource_beta
      | apply forIn_keeps
      | apply ite_keeps
      | apply bind_keeps
      | cases_resource_pair
      | split
      | intro
theorem unit_state_flush_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) : Keeps View (unit_state_flush c : SailM Unit) := by
  unfold unit_state_flush
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.report_frame ports _ _ _)
    | apply mxm_apply_ok_frame ports
    | apply sxm_apply_ok_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem held_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) : Keeps View (held q : SailM Bool) := by
  unfold held
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem scrub_issued_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (m : Nat) : Keeps View (scrub_issued q m : SailM Bool) := by
  unfold scrub_issued
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem ras_scrub_event_frame (ports : ResourceExecuteFrameProofs.Ports) (c : Nat) (s : Nat) (unc : Bool) (a : Nat) (sl : Nat) (pos : Nat) : Keeps View (ras_scrub_event c s unc a sl pos : SailM Unit) := by
  unfold ras_scrub_event
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.ras_record_frame ports _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem scrub_slice_frame (ports : ResourceExecuteFrameProofs.Ports) (s : Nat) (m : Nat) : Keeps View (scrub_slice s m : SailM Unit) := by
  unfold scrub_slice
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | apply ras_scrub_event_frame ports
    | exact project_frame _ (ports.read_sram_ecc _ _)
    | exact project_frame _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem scrub_take_frame (ports : ResourceExecuteFrameProofs.Ports) (s : Nat) : Keeps View (scrub_take s : SailM (Option scrub_write)) := by
  unfold scrub_take
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem scrub_cycle_frame (ports : ResourceExecuteFrameProofs.Ports) (m : Nat) : Keeps View (scrub_cycle m : SailM Unit) := by
  unfold scrub_cycle
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.write_sram_ecc_frame ports _ _ _)
    | exact project_frame _ (ResourceExecuteFrameProofs.write_sram_word_frame ports _ _ _)
    | apply held_frame ports
    | apply scrub_issued_frame ports
    | apply scrub_slice_frame ports
    | apply scrub_take_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem scrub_run_frame (ports : ResourceExecuteFrameProofs.Ports) (upto : Int) : Keeps View (scrub_run upto : SailM Unit) := by
  unfold scrub_run
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | apply scrub_cycle_frame ports
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
theorem free_at_frame (ports : ResourceExecuteFrameProofs.Ports) (q : Nat) (t : Nat) : Keeps View (free_at q t : SailM Bool) := by
  unfold free_at
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact throw_frame _
    | exact read_keeps _ _
    | exact write_frame _ _ (by decide) (by decide)
    | exact project_frame _ (ResourceExecuteFrameProofs.hold_view_of_frame ports _ _)
    | reduce_resource_beta
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | cases_resource_pair
    | split
    | intro
end Proofs.HistoryFrames
