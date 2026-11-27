-- Whole actual reset/tick patrol validity; no whole-callee frame premise.
import Proofs.PatrolValidity
import Proofs.ResourcePairCases
import Proofs.MachineInit
import Proofs.ParcelFrame
import Proofs.GrantConsumption
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 100000
namespace ResourceCompleteProofs
open ResourceFrameProofs ResourceInvariantProofs PatrolValidityProofs
structure Ports : Prop where
  c2c_packet_busy : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (c2c_packet_busy a0 a1)
  c2c_peer : ∀ (a0 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (c2c_peer a0)
  c2c_recv : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (c2c_recv a0 a1)
  c2c_recv_bad : ∀ (a0 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (c2c_recv_bad a0)
  c2c_recv_poison : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (c2c_recv_poison a0 a1)
  c2c_recv_ready : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (c2c_recv_ready a0 a1)
  c2c_send : ∀ (a0 : Nat) (a1 : Nat) (a2 : Nat) (a3 : vec) (a4 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (c2c_send a0 a1 a2 a3 a4)
  host_irq_callback : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (host_irq_callback a0 a1)
  host_recv : ∀ (a0 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (host_recv a0)
  host_recv_ready : ∀ (a0 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (host_recv_ready a0)
  host_send : ∀ (a0 : Nat) (a1 : vec), Keeps (fun s : ResourceFrameProofs.St => s.regs) (host_send a0 a1)
  iq_image_len : ∀ (a0 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (iq_image_len a0)
  iq_image_unit : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (iq_image_unit a0 a1)
  iq_tag_read : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (iq_tag_read a0 a1)
  iq_tag_write : ∀ (a0 : Nat) (a1 : Nat) (a2 : Int), Keeps (fun s : ResourceFrameProofs.St => s.regs) (iq_tag_write a0 a1 a2)
  iq_unit_read : ∀ (a0 : Nat) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (iq_unit_read a0 a1)
  iq_unit_write : ∀ (a0 : Nat) (a1 : Nat) (a2 : (BitVec 32)), Keeps (fun s : ResourceFrameProofs.St => s.regs) (iq_unit_write a0 a1 a2)
  observe_site : ∀ (a0 : String) (a1 : Nat) (a2 : Nat) (a3 : Nat) (a4 : Nat) (a5 : Int), Keeps (fun s : ResourceFrameProofs.St => s.regs) (observe_site a0 a1 a2 a3 a4 a5)
  raw_write_sram_ecc : ∀ (a0 : mem_slice) (a1 : Nat) (a2 : ecc_ov), Keeps (fun s : ResourceFrameProofs.St => s.regs) (raw_write_sram_ecc a0 a1 a2)
  raw_write_sram_word : ∀ (a0 : mem_slice) (a1 : Nat) (a2 : vec), Keeps (fun s : ResourceFrameProofs.St => s.regs) (raw_write_sram_word a0 a1 a2)
  read_sram_ecc : ∀ (a0 : mem_slice) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (read_sram_ecc a0 a1)
  read_sram_word : ∀ (a0 : mem_slice) (a1 : Nat), Keeps (fun s : ResourceFrameProofs.St => s.regs) (read_sram_word a0 a1)
open MachineInitProofs
theorem throw_complete {α : Type} (e : Error exception) : Inv Complete (throw e : SailM α) := by
 intro s hs
 exact hs
theorem write_complete (r : Register) (v : RegisterType r) : Inv Complete (Out.writeReg r v) := by
 intro s hs
 change Complete {s with regs:=s.regs.insert r v}
 exact MachineInitProofs.complete_insert s r v hs
theorem assert_complete (p : Bool) (message : String) : Inv Complete (Sail.ConcurrencyInterfaceV1.PreSail.assert p message : SailM Unit) := by
 cases p <;> intro state hs <;> exact hs

attribute [local irreducible] ResourceInvariantProofs.Inv
attribute [local irreducible] GetElem?.getElem! Inhabited.default Bind.bind Pure.pure EStateM.bind EStateM.pure PreSail.readReg PreSail.writeReg IntRange.forIn'
attribute [local irreducible] Out.Functions.__id Out.Functions.abc_after Out.Functions.abc_at Out.Functions.abc_fp8 Out.Functions.abc_others Out.Functions.abc_others_fp8 Out.Functions.abc_pair_pick Out.Functions.abc_prune Out.Functions.abc_types_ok Out.Functions.acc_mode_name_backwards_matches Out.Functions.acc_mode_name_forwards Out.Functions.acc_mode_name_forwards_matches Out.Functions.acc_mode_of_num Out.Functions.acctype_declared Out.Functions.acctype_name_backwards_matches Out.Functions.acctype_name_forwards Out.Functions.acctype_name_forwards_matches Out.Functions.addr_bank Out.Functions.agt_advance Out.Functions.agt_advance_test Out.Functions.agt_after_reference Out.Functions.agt_config Out.Functions.agt_configure_pure Out.Functions.agt_due_pure Out.Functions.agt_field_of_num Out.Functions.agt_field_pos Out.Functions.agt_finish_config_pure Out.Functions.agt_nlevels Out.Functions.agt_opened Out.Functions.agt_refresh_pure Out.Functions.agt_state_of_num Out.Functions.aligned Out.Functions.all_ones Out.Functions.append_element Out.Functions.assembly_forwards_matches Out.Functions.bank_rule Out.Functions.banks_meet Out.Functions.bf16_dot Out.Functions.bfp8_exact Out.Functions.binop_entry_backwards_matches Out.Functions.binop_entry_forwards Out.Functions.binop_entry_forwards_matches Out.Functions.binop_legal Out.Functions.bit_index Out.Functions.bit_of Out.Functions.bits_of_s Out.Functions.bits_t Out.Functions.blk_exp Out.Functions.bool_bits Out.Functions.c2c_op_of_num Out.Functions.c2c_packet_flits Out.Functions.c2c_poison_int Out.Functions.c2c_ptype_code Out.Functions.c2c_ptype_of_num Out.Functions.c2c_ptype_queued Out.Functions.c2c_rx Out.Functions.c2c_tx Out.Functions.canon_nan Out.Functions.cfg_path_pick Out.Functions.chain_suffix_backwards_matches Out.Functions.chain_suffix_forwards Out.Functions.chain_suffix_forwards_matches Out.Functions.chk_elem Out.Functions.chk_wrow Out.Functions.clash_addr Out.Functions.clash_stream Out.Functions.classify_items Out.Functions.clears_repeat Out.Functions.cmp_legal Out.Functions.cmp_mnemonic_backwards_matches Out.Functions.cmp_mnemonic_forwards Out.Functions.cmp_mnemonic_forwards_matches Out.Functions.commit_scan Out.Functions.commit_wins Out.Functions.config_ends Out.Functions.config_for Out.Functions.conv_dst Out.Functions.conv_entry_backwards_matches Out.Functions.conv_entry_forwards Out.Functions.conv_entry_forwards_matches Out.Functions.conv_mnemonic_backwards_matches Out.Functions.conv_mnemonic_forwards Out.Functions.conv_mnemonic_forwards_matches Out.Functions.conv_src Out.Functions.conv_src_type Out.Functions.conv_types Out.Functions.csr_value Out.Functions.db_idx Out.Functions.dec_bits_10_backwards Out.Functions.dec_bits_10_backwards_matches Out.Functions.dec_bits_1_backwards Out.Functions.dec_bits_1_backwards_matches Out.Functions.dec_bits_2_backwards Out.Functions.dec_bits_2_backwards_matches Out.Functions.dec_bits_32_backwards Out.Functions.dec_bits_32_backwards_matches Out.Functions.dec_bits_3_backwards Out.Functions.dec_bits_3_backwards_matches Out.Functions.dec_bits_4_backwards Out.Functions.dec_bits_4_backwards_matches Out.Functions.dec_bits_5_backwards Out.Functions.dec_bits_5_backwards_matches Out.Functions.dec_bits_6_backwards Out.Functions.dec_bits_6_backwards_matches Out.Functions.dec_bits_7_backwards Out.Functions.dec_bits_7_backwards_matches Out.Functions.dec_bits_8_backwards Out.Functions.dec_bits_8_backwards_matches Out.Functions.dec_bits_9_backwards Out.Functions.dec_bits_9_backwards_matches Out.Functions.dec_bits_backwards Out.Functions.dec_bits_backwards_matches Out.Functions.dec_bits_forwards Out.Functions.dec_bits_forwards_matches Out.Functions.def_spc_backwards Out.Functions.def_spc_backwards_matches Out.Functions.def_spc_forwards Out.Functions.def_spc_forwards_matches Out.Functions.diag_code_backwards_matches Out.Functions.diag_code_forwards Out.Functions.diag_code_forwards_matches Out.Functions.diag_idx Out.Functions.diag_is_data Out.Functions.diag_number Out.Functions.diag_of_num Out.Functions.drop_commits Out.Functions.dst_split_backwards_matches Out.Functions.dst_split_forwards Out.Functions.dst_split_forwards_matches Out.Functions.dsum_round Out.Functions.dsum_round_u Out.Functions.dsum_wide Out.Functions.earliest Out.Functions.earliest_int Out.Functions.ecc_class_of_num Out.Functions.ecc_classify Out.Functions.ecc_correct Out.Functions.ecc_data_index Out.Functions.ecc_encode Out.Functions.ecc_flip_delta Out.Functions.ecc_is_pow2 Out.Functions.ecc_log2 Out.Functions.ecc_ov_get Out.Functions.ecc_ov_set Out.Functions.ecc_par10 Out.Functions.ecc_poison_ov Out.Functions.ecc_poison_sl Out.Functions.ecc_pos Out.Functions.elem Out.Functions.elem_at Out.Functions.eltype_name_backwards_matches Out.Functions.eltype_name_forwards Out.Functions.eltype_name_forwards_matches Out.Functions.eltype_of_num Out.Functions.eltype_width Out.Functions.empty_plane Out.Functions.encdec_acc_mode_backwards Out.Functions.encdec_acc_mode_backwards_matches Out.Functions.encdec_acc_mode_forwards Out.Functions.encdec_acc_mode_forwards_matches Out.Functions.encdec_acctype_backwards_matches Out.Functions.encdec_acctype_forwards Out.Functions.encdec_acctype_forwards_matches Out.Functions.encdec_c2c_op_backwards_matches Out.Functions.encdec_c2c_op_forwards Out.Functions.encdec_c2c_op_forwards_matches Out.Functions.encdec_cmp_mode_backwards_matches Out.Functions.encdec_cmp_mode_forwards Out.Functions.encdec_cmp_mode_forwards_matches Out.Functions.encdec_eltype_backwards_matches Out.Functions.encdec_eltype_forwards Out.Functions.encdec_eltype_forwards_matches Out.Functions.encdec_mapop_backwards_matches Out.Functions.encdec_mapop_forwards Out.Functions.encdec_mapop_forwards_matches Out.Functions.encdec_mem_dir_backwards_matches Out.Functions.encdec_mem_dir_forwards Out.Functions.encdec_mem_dir_forwards_matches Out.Functions.encdec_reduce_op_backwards_matches Out.Functions.encdec_reduce_op_forwards Out.Functions.encdec_reduce_op_forwards_matches Out.Functions.encdec_shift_dir_backwards Out.Functions.encdec_shift_dir_backwards_matches Out.Functions.encdec_shift_dir_forwards Out.Functions.encdec_shift_dir_forwards_matches Out.Functions.encdec_vxm_binop_backwards_matches Out.Functions.encdec_vxm_binop_forwards Out.Functions.encdec_vxm_binop_forwards_matches Out.Functions.encdec_vxm_conv_backwards_matches Out.Functions.encdec_vxm_conv_forwards Out.Functions.encdec_vxm_conv_forwards_matches Out.Functions.encdec_vxm_unop_backwards_matches Out.Functions.encdec_vxm_unop_forwards Out.Functions.encdec_vxm_unop_forwards_matches Out.Functions.entry_latency Out.Functions.entry_latency_typed Out.Functions.entry_of Out.Functions.execute_C2C_DESKEW Out.Functions.execute_C2C_DESKEWRECV Out.Functions.execute_C2C_HOSTWAIT Out.Functions.execute_C2C_RTDESKEW Out.Functions.execute_ILLEGAL32 Out.Functions.execute_ILLEGAL64 Out.Functions.execute_MEM_COUNTDOWN Out.Functions.execute_MEM_DEEPSLEEP Out.Functions.execute_MEM_ITERATIONS Out.Functions.execute_MEM_POWERCONFIG Out.Functions.execute_MEM_STEP Out.Functions.execute_MEM_STREAMSTEP Out.Functions.execute_NOP Out.Functions.execute_NOTIFY Out.Functions.execute_REPEAT Out.Functions.execute_SYNC Out.Functions.exp_iv Out.Functions.exp_small Out.Functions.extensions Out.Functions.f16_dot Out.Functions.f16_exact Out.Functions.f16_row Out.Functions.f16_row_bad Out.Functions.f16_stream_values Out.Functions.fadd_core Out.Functions.fclass_eq Out.Functions.fclass_of_num Out.Functions.fcmp Out.Functions.fdecode Out.Functions.fetch_profile_causal Out.Functions.fetch_tile_available Out.Functions.fetchm_bank_note Out.Functions.fetchm_hold_until Out.Functions.field Out.Functions.fill_append Out.Functions.fill_len Out.Functions.fill_push_unit Out.Functions.fill_units_append Out.Functions.find_override Out.Functions.flip96 Out.Functions.fmt_of Out.Functions.fp8_dot Out.Functions.fp8_exact Out.Functions.fp8_format_name_backwards_matches Out.Functions.fp8_format_name_forwards Out.Functions.fp8_format_name_forwards_matches Out.Functions.fp_abs Out.Functions.fp_add Out.Functions.fp_convert Out.Functions.fp_max Out.Functions.fp_min Out.Functions.fp_mul Out.Functions.fp_neg Out.Functions.fp_of_int Out.Functions.fp_sub Out.Functions.fp_tr_value Out.Functions.from_bytes_le Out.Functions.fround Out.Functions.fval_int Out.Functions.fx_of Out.Functions.gap_or_none Out.Functions.has_extension Out.Functions.hex_bits_10_backwards_matches Out.Functions.hex_bits_11_backwards_matches Out.Functions.hex_bits_12_backwards_matches Out.Functions.hex_bits_13_backwards_matches Out.Functions.hex_bits_14_backwards_matches Out.Functions.hex_bits_15_backwards_matches Out.Functions.hex_bits_16_backwards_matches Out.Functions.hex_bits_17_backwards_matches Out.Functions.hex_bits_18_backwards_matches Out.Functions.hex_bits_19_backwards_matches Out.Functions.hex_bits_1_backwards_matches Out.Functions.hex_bits_20_backwards_matches Out.Functions.hex_bits_21_backwards_matches Out.Functions.hex_bits_22_backwards_matches Out.Functions.hex_bits_23_backwards_matches Out.Functions.hex_bits_24_backwards_matches Out.Functions.hex_bits_25_backwards_matches Out.Functions.hex_bits_26_backwards_matches Out.Functions.hex_bits_27_backwards_matches Out.Functions.hex_bits_28_backwards_matches Out.Functions.hex_bits_29_backwards_matches Out.Functions.hex_bits_2_backwards_matches Out.Functions.hex_bits_30_backwards_matches Out.Functions.hex_bits_31_backwards_matches Out.Functions.hex_bits_32_backwards_matches Out.Functions.hex_bits_33_backwards_matches Out.Functions.hex_bits_34_backwards_matches Out.Functions.hex_bits_35_backwards_matches Out.Functions.hex_bits_36_backwards_matches Out.Functions.hex_bits_37_backwards_matches Out.Functions.hex_bits_38_backwards_matches Out.Functions.hex_bits_39_backwards_matches Out.Functions.hex_bits_3_backwards_matches Out.Functions.hex_bits_40_backwards_matches Out.Functions.hex_bits_41_backwards_matches Out.Functions.hex_bits_42_backwards_matches Out.Functions.hex_bits_43_backwards_matches Out.Functions.hex_bits_44_backwards_matches Out.Functions.hex_bits_45_backwards_matches Out.Functions.hex_bits_46_backwards_matches Out.Functions.hex_bits_47_backwards_matches Out.Functions.hex_bits_48_backwards_matches Out.Functions.hex_bits_49_backwards_matches Out.Functions.hex_bits_4_backwards_matches Out.Functions.hex_bits_50_backwards_matches Out.Functions.hex_bits_51_backwards_matches Out.Functions.hex_bits_52_backwards_matches Out.Functions.hex_bits_53_backwards_matches Out.Functions.hex_bits_54_backwards_matches Out.Functions.hex_bits_55_backwards_matches Out.Functions.hex_bits_56_backwards_matches Out.Functions.hex_bits_57_backwards_matches Out.Functions.hex_bits_58_backwards_matches Out.Functions.hex_bits_59_backwards_matches Out.Functions.hex_bits_5_backwards_matches Out.Functions.hex_bits_60_backwards_matches Out.Functions.hex_bits_61_backwards_matches Out.Functions.hex_bits_62_backwards_matches Out.Functions.hex_bits_63_backwards_matches Out.Functions.hex_bits_64_backwards_matches Out.Functions.hex_bits_6_backwards_matches Out.Functions.hex_bits_7_backwards_matches Out.Functions.hex_bits_8_backwards_matches Out.Functions.hex_bits_9_backwards_matches Out.Functions.hex_bits_backwards Out.Functions.hex_bits_backwards_matches Out.Functions.hex_bits_forwards Out.Functions.hex_bits_forwards_matches Out.Functions.hold_decide Out.Functions.holds_f16 Out.Functions.i8_tr_value Out.Functions.identity_map Out.Functions.idiv_ceil Out.Functions.idiv_floor Out.Functions.ilog2 Out.Functions.in_barrier_window Out.Functions.inf_bits Out.Functions.int8_dot Out.Functions.int_of Out.Functions.int_of_fp Out.Functions.inward_row Out.Functions.iq_padding Out.Functions.is_bool Out.Functions.is_dst Out.Functions.is_float Out.Functions.is_int Out.Functions.is_loopable Out.Functions.is_mxm_queue Out.Functions.is_nan Out.Functions.is_none Out.Functions.is_pair Out.Functions.is_prm Out.Functions.is_sel Out.Functions.is_shf Out.Functions.is_some Out.Functions.is_tsp Out.Functions.is_unsigned Out.Functions.isqrt Out.Functions.issued_by Out.Functions.issuer_of Out.Functions.items_len Out.Functions.k_binary Out.Functions.k_cmp Out.Functions.k_convert Out.Functions.k_convert_any Out.Functions.k_nonzero Out.Functions.k_unary Out.Functions.lane8 Out.Functions.lane_byte Out.Functions.latency_of Out.Functions.latency_of_instr Out.Functions.link_flit_mask Out.Functions.link_pow2 Out.Functions.lint Out.Functions.list_len Out.Functions.ln2_iv Out.Functions.log_iv Out.Functions.mapop_name_backwards_matches Out.Functions.mapop_name_forwards Out.Functions.mapop_name_forwards_matches Out.Functions.mem_dir_mnemonic_backwards_matches Out.Functions.mem_dir_mnemonic_forwards Out.Functions.mem_dir_mnemonic_forwards_matches Out.Functions.mem_quad Out.Functions.mem_slice_present Out.Functions.mem_unit Out.Functions.merge_tiles Out.Functions.model_link_flit_mask Out.Functions.model_link_flits Out.Functions.model_ptype_queued Out.Functions.mxm_acctype_of_num Out.Functions.mxm_bad Out.Functions.mxm_cap Out.Functions.mxm_check Out.Functions.mxm_clash Out.Functions.mxm_part_of_num Out.Functions.mxm_queue_present Out.Functions.mxm_unit Out.Functions.mxm_wtype_of_num Out.Functions.nbits_of Out.Functions.neq_bool Out.Functions.nonempty_list Out.Functions.not Out.Functions.not_bit Out.Functions.num_of_acc_mode Out.Functions.num_of_agt_field Out.Functions.num_of_agt_state Out.Functions.num_of_c2c_op Out.Functions.num_of_c2c_ptype Out.Functions.num_of_diag Out.Functions.num_of_ecc_class Out.Functions.num_of_eltype Out.Functions.num_of_fclass Out.Functions.num_of_mxm_acctype Out.Functions.num_of_mxm_part Out.Functions.num_of_mxm_wtype Out.Functions.num_of_row Out.Functions.num_of_scrub_action Out.Functions.num_of_srf_lookup Out.Functions.num_of_timing_class Out.Functions.num_of_unit_kind Out.Functions.one_plane Out.Functions.opt_spc_backwards Out.Functions.opt_spc_backwards_matches Out.Functions.opt_spc_forwards Out.Functions.opt_spc_forwards_matches Out.Functions.or_pair Out.Functions.outward_row Out.Functions.ov_nth Out.Functions.pair_exact Out.Functions.path_hold_until Out.Functions.path_views Out.Functions.place Out.Functions.place_all Out.Functions.poll_epoch Out.Functions.pow2i Out.Functions.q_in Out.Functions.queue_present Out.Functions.ras_idx Out.Functions.ras_loc Out.Functions.ras_sat_inc Out.Functions.read_gap Out.Functions.reduce_mnemonic_backwards_matches Out.Functions.reduce_mnemonic_forwards Out.Functions.reduce_mnemonic_forwards_matches Out.Functions.ref_bank Out.Functions.ref_path Out.Functions.ref_thread Out.Functions.rep_sp_set Out.Functions.repeat_after Out.Functions.repeat_plan Out.Functions.repeatable Out.Functions.rev_items Out.Functions.ring Out.Functions.rne_int Out.Functions.round_iv Out.Functions.row_of_num Out.Functions.rows_of Out.Functions.s_of_bits Out.Functions.sail_mask Out.Functions.sail_ones Out.Functions.same_target Out.Functions.sat_s Out.Functions.sat_t Out.Functions.scrub_action_of_num Out.Functions.scrub_arbitrate Out.Functions.scrub_due_count Out.Functions.scrub_slot_taken Out.Functions.sep_forwards Out.Functions.sep_forwards_matches Out.Functions.set_lane_byte Out.Functions.sg4_suffix_backwards_matches Out.Functions.sg4_suffix_forwards Out.Functions.sg4_suffix_forwards_matches Out.Functions.shift_dir_name_backwards_matches Out.Functions.shift_dir_name_forwards Out.Functions.shift_dir_name_forwards_matches Out.Functions.slice_mask Out.Functions.slot_of Out.Functions.spc_backwards Out.Functions.spc_backwards_matches Out.Functions.spc_forwards Out.Functions.spc_forwards_matches Out.Functions.sram_addr_of Out.Functions.sram_bank Out.Functions.sram_field_backwards Out.Functions.sram_field_backwards_matches Out.Functions.sram_field_forwards Out.Functions.sram_field_forwards_matches Out.Functions.sram_index Out.Functions.sram_text_backwards Out.Functions.sram_text_backwards_matches Out.Functions.sram_text_forwards Out.Functions.sram_text_forwards_matches Out.Functions.srf_lookup_of_num Out.Functions.srf_ov_spare Out.Functions.srf_rep_merge Out.Functions.srf_slot_best Out.Functions.srf_slot_chain_ov Out.Functions.srf_slot_collides Out.Functions.srf_slot_flip Out.Functions.srf_slot_read Out.Functions.srf_slot_read_ov Out.Functions.srf_slot_write Out.Functions.srf_stream_only_downstream Out.Functions.stream_id Out.Functions.stream_name_forwards_matches Out.Functions.stream_row Out.Functions.string_of_exception Out.Functions.sxm_bad Out.Functions.sxm_clash Out.Functions.sxm_lane Out.Functions.sxm_map_distinct Out.Functions.sxm_or_kind Out.Functions.sxm_set_lane Out.Functions.sxm_stream_plus Out.Functions.sxm_unit_at Out.Functions.tag_of Out.Functions.tags_clash Out.Functions.text_width Out.Functions.tile_active Out.Functions.tile_addr Out.Functions.timing_class_name Out.Functions.timing_class_of_num Out.Functions.timing_of Out.Functions.timing_overrides Out.Functions.timing_type_of Out.Functions.to_bytes_le Out.Functions.tr_eval Out.Functions.tr_eval_int Out.Functions.tr_interval Out.Functions.ty_split_backwards_matches Out.Functions.ty_split_forwards Out.Functions.ty_split_forwards_matches Out.Functions.ty_valid Out.Functions.unit_kind_of_num Out.Functions.unop_entry_backwards_matches Out.Functions.unop_entry_forwards Out.Functions.unop_entry_forwards_matches Out.Functions.unop_legal Out.Functions.vec_units Out.Functions.view_free Out.Functions.view_paths_busy Out.Functions.vsrc_name_forwards_matches Out.Functions.vxm_binop_mnemonic_backwards_matches Out.Functions.vxm_binop_mnemonic_forwards Out.Functions.vxm_binop_mnemonic_forwards_matches Out.Functions.vxm_unit Out.Functions.vxm_unop_mnemonic_backwards_matches Out.Functions.vxm_unop_mnemonic_forwards Out.Functions.vxm_unop_mnemonic_forwards_matches Out.Functions.wait_reason Out.Functions.wants_pair Out.Functions.wlane_add Out.Functions.wlane_read Out.Functions.words_to_planes Out.Functions.wrap_s Out.Functions.wrap_t Out.Functions.write_gap Out.Functions.writer_before Out.Functions.wtype_name_backwards_matches Out.Functions.wtype_name_forwards Out.Functions.wtype_name_forwards_matches Out.Functions.wtype_ok Out.Functions.wtype_split_backwards_matches Out.Functions.wtype_split_forwards Out.Functions.wtype_split_forwards_matches Out.Functions.wtype_valid Out.Functions.zero_bits Out.abs_real Out.add_real Out.diag_callback Out.div_real Out.ecc_event_callback Out.eq_real Out.gt_real Out.gteq_real Out.lint_callback Out.lt_real Out.lteq_real Out.mult_real Out.mxm_corr_callback Out.neg_real Out.prerr_real Out.print_real Out.random_real Out.real_power Out.round_down Out.round_up Out.scrub_event_callback Out.spare_event_callback Out.sqrt_real Out.sram_write_callback Out.srf_write_callback Out.sub_real Out.tdiv_int Out.tmod_int Out.to_real c2c_reset host_reset iq_push_tagged iq_reset masks_reset mem_reset mxm_reset srf_reset sxm_reset reset barrier_retire get_stream set_stream srf_write_ov commit_apply diag_record diag_report report commit_due barrier_notify barrier_sync iq_pop iq_unit_word iq_peek iq_unit_lane iq_word_ov iq_view iq_word_ov_set host_deliver host_msix ras_record ras_ecc_event iq_word_check iq_text_check parcel_earliest decode_text iq_word_unc iq_text_unc iq_decode_at parcel_scan Out.Functions.control host_parked agt_get path_busy path_tail path_view_of polling parked_one queue_parked hold_view_of free_at iq_deliver enqueue ecc_begin exec_cycle execute_C2C_HOSTNOTIFY commit_at ecc_poison_mask ecc_result_ov plane_stream write_planes_fwd write_planes execute_C2C_PCIRECV pci_busy ras_spare_event srf_spare srf_s03 ecc_check_value srf_read_chain_ov srf_only_downstream srf_read_ov read_planes_full read_planes_core read_planes execute_C2C_PCISEND c2c_link c2c_peer_ok execute_C2C_RECEIVE c2c_busy c2c_put execute_C2C_SEND execute_C2C_TRANSMIT execute_CONFIG stream_fetch_arm stream_fetch_start execute_IFETCH execute_IFETCHM commit_cycle eff_addr eff_stream read_cycle read_plane_raw sram_read sram_read_ov write_sram_ecc write_sram_word sram_write_ov tile_on execute_MEM_DIRECT ecc_check_sl srf_report mem_report execute_MEM_GATHER execute_MEM_SCATTER execute_MEM_SETSEGMENT macc_busy_in macc_busy mxm_stage_acc mxm_stage plane mxm_plane_at plane_blocked ras_mxm_event abc_exec execute_MXM_ABC execute_MXM_ABC8 execute_MXM_ACC execute_MXM_IW read_planes16 execute_MXM_IWS execute_MXM_LW execute_SETVL sx sxm_load_map sxm_rd_cycle execute_SXM_LOADMAP execute_SXM_LOADMAPWIDE sxm_cm_cycle sxm_emit execute_SXM_MAP execute_SXM_ROTATE execute_SXM_SELECT execute_SXM_SHIFT execute_SXM_TRANSPOSE sailThrow internal_error fp_transcendental i8_transcendental apply vxm_commit_cycle vxm_read_cycle vxm_read vxm_lanes execute_VXM_BIN execute_VXM_CMP execute_VXM_CONV execute_VXM_CVT execute_VXM_MASK execute_VXM_REDUCE execute_VXM_SELECT execute_VXM_UN execute execute_element execute_all execute_due iq_fetch_ov local_fetch_complete held scrub_issued ras_scrub_event scrub_slice scrub_take scrub_cycle scrub_run stream_fetch_take stream_fetch_service stream_fetch_complete agt_set agt_finish_config deskewrecv_ready deskewrecv_diag host_wait occupancy run_element dispatch_body hold_check dispatch_check reserve_local_fetch repeat_issue dispatch_decoded bank_report cfg_path agt_configure dispatch_cfg loop_clash_ref due_ref agt_due due_refs hold_path loop_tag agt_refresh run_loop dispatch_slots dispatch strict_held parcel_held tick_queue mxm_apply mxm_apply_ok sxm_apply_ok unit_state_flush tick
theorem c2c_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (c2c_reset unitArg : SailM Unit) := by
  unfold c2c_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (host_reset unitArg : SailM Unit) := by
  unfold host_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_push_tagged_complete (ports : Ports) (q : Nat) (u : (BitVec 32)) (tag : Int) : Inv Complete (iq_push_tagged q u tag : SailM Bool) := by
  unfold iq_push_tagged
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_tag_write _ _ _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_unit_write _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (iq_reset unitArg : SailM Unit) := by
  unfold iq_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_push_tagged_complete ports _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_image_len _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_image_unit _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem masks_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (masks_reset unitArg : SailM Unit) := by
  unfold masks_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mem_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (mem_reset unitArg : SailM Unit) := by
  unfold mem_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (mxm_reset unitArg : SailM Unit) := by
  unfold mxm_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (srf_reset unitArg : SailM Unit) := by
  unfold srf_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sxm_reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (sxm_reset unitArg : SailM Unit) := by
  unfold sxm_reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem reset_complete (ports : Ports) (unitArg : Unit) : Inv Complete (reset unitArg : SailM Unit) := by
  unfold reset
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact c2c_reset_complete ports _
    | exact host_reset_complete ports _
    | exact iq_reset_complete ports _
    | exact masks_reset_complete ports _
    | exact mem_reset_complete ports _
    | exact mxm_reset_complete ports _
    | exact srf_reset_complete ports _
    | exact sxm_reset_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_retire_complete (ports : Ports) (c : Nat) : Inv Complete (barrier_retire c : SailM Unit) := by
  unfold barrier_retire
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem get_stream_complete (ports : Ports) (r : row) (s : Nat) : Inv Complete (get_stream r s : SailM (Vector (Vector Out.srf_write 44) 64)) := by
  unfold get_stream
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem set_stream_complete (ports : Ports) (r : row) (s : Nat) (v : (Vector (Vector Out.srf_write 44) 64)) : Inv Complete (set_stream r s v : SailM Unit) := by
  unfold set_stream
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_write_ov_complete (ports : Ports) (r : row) (s : Nat) (g : Nat) (c : Nat) (v : vec) (ov : ecc_ov) : Inv Complete (srf_write_ov r s g c v ov : SailM Bool) := by
  unfold srf_write_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact get_stream_complete ports _ _
    | exact set_stream_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem commit_apply_complete (ports : Ports) (c : Nat) (all : (List commit)) (xs : (List commit)) (i : Nat) (hits : (Vector Nat 145)) : Inv Complete (commit_apply c all xs i hits : SailM (Vector Nat 145)) := by
  induction xs generalizing c all i hits
  case nil =>
    unfold commit_apply
    repeat' first
      | exact ih _ _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact srf_write_ov_complete ports _ _ _ _ _ _
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
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact srf_write_ov_complete ports _ _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem diag_record_complete (ports : Ports) (c : Nat) (q : Nat) (code : Nat) : Inv Complete (diag_record c q code : SailM Unit) := by
  unfold diag_record
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem diag_report_complete (ports : Ports) (c : Nat) (q : Nat) (d : diag) : Inv Complete (diag_report c q d : SailM Unit) := by
  unfold diag_report
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact diag_record_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem report_complete (ports : Ports) (q : Nat) (c : Nat) (d : diag) : Inv Complete (report q c d : SailM Unit) := by
  unfold report
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact diag_report_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem commit_due_complete (ports : Ports) (c : Nat) : Inv Complete (commit_due c : SailM Unit) := by
  unfold commit_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_apply_complete ports _ _ _ _ _
    | exact report_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_notify_complete (ports : Ports) (n : Nat) : Inv Complete (barrier_notify n : SailM Unit) := by
  unfold barrier_notify
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem barrier_sync_complete (ports : Ports) (k : Nat) (c : Nat) : Inv Complete (barrier_sync k c : SailM Unit) := by
  unfold barrier_sync
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_pop_complete (ports : Ports) (q : Nat) (n : Nat) : Inv Complete (iq_pop q n : SailM Unit) := by
  unfold iq_pop
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_unit_word_complete (ports : Ports) (q : Nat) (i : Nat) : Inv Complete (iq_unit_word q i : SailM Int) := by
  unfold iq_unit_word
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_tag_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_peek_complete (ports : Ports) (q : Nat) (i : Nat) : Inv Complete (iq_peek q i : SailM (BitVec 32)) := by
  unfold iq_peek
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_unit_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_unit_lane_complete (ports : Ports) (q : Nat) (i : Nat) : Inv Complete (iq_unit_lane q i : SailM Nat) := by
  unfold iq_unit_lane
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_tag_read _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_word_ov_complete (ports : Ports) (q : Nat) (w : Nat) : Inv Complete (iq_word_ov q w : SailM (BitVec 10)) := by
  unfold iq_word_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_view_complete (ports : Ports) (q : Nat) (i : Nat) : Inv Complete (iq_view q i : SailM (BitVec 32)) := by
  unfold iq_view
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_peek_complete ports _ _
    | exact iq_unit_lane_complete ports _ _
    | exact iq_unit_word_complete ports _ _
    | exact iq_word_ov_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_word_ov_set_complete (ports : Ports) (q : Nat) (w : Nat) (e : (BitVec 10)) : Inv Complete (iq_word_ov_set q w e : SailM Unit) := by
  unfold iq_word_ov_set
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_deliver_complete (ports : Ports) (c : Nat) (n : Nat) : Inv Complete (host_deliver c n : SailM Unit) := by
  unfold host_deliver
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.host_irq_callback _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_msix_complete (ports : Ports) (c : Nat) (n : Nat) : Inv Complete (host_msix c n : SailM Unit) := by
  unfold host_msix
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact host_deliver_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ras_record_complete (ports : Ports) (c : Nat) (q : Int) (kind : Nat) (loc : (BitVec 43)) (sl : Nat) (pos : Nat) : Inv Complete (ras_record c q kind loc sl pos : SailM Unit) := by
  unfold ras_record
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact host_msix_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ras_ecc_event_complete (ports : Ports) (c : Nat) (q : Nat) (kind : Nat) (unc : Bool) (isw : Bool) (id : Nat) (addr : Nat) (sl : Nat) (pos : Nat) : Inv Complete (ras_ecc_event c q kind unc isw id addr sl pos : SailM Unit) := by
  unfold ras_ecc_event
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ras_record_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_word_check_complete (ports : Ports) (q : Nat) (c : Nat) (w : Int) : Inv Complete (iq_word_check q c w : SailM Unit) := by
  unfold iq_word_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_peek_complete ports _ _
    | exact iq_word_ov_complete ports _ _
    | exact iq_word_ov_set_complete ports _ _ _
    | exact ras_ecc_event_complete ports _ _ _ _ _ _ _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_tag_read _ _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.iq_unit_write _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_text_check_complete (ports : Ports) (q : Nat) (c : Nat) (units : Nat) : Inv Complete (iq_text_check q c units : SailM Unit) := by
  unfold iq_text_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_unit_word_complete ports _ _
    | exact iq_view_complete ports _ _
    | exact iq_word_check_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parcel_earliest_complete (ports : Ports) (q : Nat) (p : parcel) : Inv Complete (parcel_earliest q p : SailM Int) := by
  unfold parcel_earliest
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem decode_text_complete (ports : Ports) (u0 : (BitVec 32)) (u1 : (BitVec 32)) : Inv Complete (decode_text u0 u1 : SailM (Result instruction diag)) := by
  unfold ResourceInvariantProofs.Inv
  intro state hs
  have h:=Proofs.ParcelFrame.decode_text_frame state u0 u1
  cases hx : (decode_text u0 u1).run state <;> simp only [Proofs.ParcelFrame.FrameAt,hx] at h <;> simpa [hx,h] using hs
theorem iq_word_unc_complete (ports : Ports) (q : Nat) (w : Int) : Inv Complete (iq_word_unc q w : SailM Bool) := by
  unfold iq_word_unc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_word_ov_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_text_unc_complete (ports : Ports) (q : Nat) (pos : Nat) (w : Nat) : Inv Complete (iq_text_unc q pos w : SailM Bool) := by
  unfold iq_text_unc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_unit_word_complete ports _ _
    | exact iq_word_unc_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_decode_at_complete (ports : Ports) (q : Nat) (pos : Nat) (w : Nat) : Inv Complete (iq_decode_at q pos w : SailM (Result instruction diag)) := by
  unfold iq_decode_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact decode_text_complete ports _ _
    | exact iq_text_unc_complete ports _ _ _
    | exact iq_view_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parcel_scan_complete (ports : Ports) (q : Nat) : Inv Complete (parcel_scan q : SailM parcel) := by
  unfold parcel_scan
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_decode_at_complete ports _ _ _
    | exact iq_view_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem control_complete (ports : Ports) (c : Nat) : Inv Complete (Out.Functions.control c : SailM Unit) := by
  unfold Out.Functions.control
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact barrier_notify_complete ports _
    | exact barrier_sync_complete ports _ _
    | exact iq_pop_complete ports _ _
    | exact iq_text_check_complete ports _ _ _
    | exact parcel_earliest_complete ports _ _
    | exact parcel_scan_complete ports _
    | exact report_complete ports _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_parked_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (host_parked q c : SailM Bool) := by
  unfold host_parked
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_get_complete (ports : Ports) (q : Nat) (p : Nat) : Inv Complete (agt_get q p : SailM agt) := by
  unfold agt_get
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem path_busy_complete (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv Complete (path_busy q p c : SailM Bool) := by
  unfold path_busy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem path_tail_complete (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv Complete (path_tail q p c : SailM Bool) := by
  unfold path_tail
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem path_view_of_complete (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv Complete (path_view_of q p c : SailM path_view) := by
  unfold path_view_of
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | exact path_busy_complete ports _ _ _
    | exact path_tail_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem polling_complete (ports : Ports) (q : Nat) : Inv Complete (polling q : SailM Bool) := by
  unfold polling
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem parked_one_complete (ports : Ports) (k : Nat) (c : Nat) : Inv Complete (parked_one k c : SailM Bool) := by
  unfold parked_one
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem queue_parked_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (queue_parked q c : SailM Bool) := by
  unfold queue_parked
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact parked_one_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_view_of_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (hold_view_of q c : SailM hold_view) := by
  unfold hold_view_of
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact host_parked_complete ports _ _
    | exact path_view_of_complete ports _ _ _
    | exact polling_complete ports _
    | exact queue_parked_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem free_at_complete (ports : Ports) (q : Nat) (t : Nat) : Inv Complete (free_at q t : SailM Bool) := by
  unfold free_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact hold_view_of_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_deliver_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (iq_deliver q c : SailM (Bool × Bool)) := by
  unfold iq_deliver
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact iq_push_tagged_complete ports _ _ _
    | exact iq_word_ov_set_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem enqueue_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (enqueue q c : SailM Unit) := by
  unfold enqueue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact free_at_complete ports _ _
    | exact iq_deliver_complete ports _ _
    | exact parcel_earliest_complete ports _ _
    | exact parcel_scan_complete ports _
    | exact report_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ecc_begin_complete (ports : Ports) (unitArg : Unit) : Inv Complete (ecc_begin unitArg : SailM Unit) := by
  unfold ecc_begin
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem exec_cycle_complete (ports : Ports) (unitArg : Unit) : Inv Complete (exec_cycle unitArg : SailM Nat) := by
  unfold exec_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_HOSTNOTIFY_complete (ports : Ports) (n : (BitVec 4)) : Inv Complete (execute_C2C_HOSTNOTIFY n : SailM ExecutionResult) := by
  unfold execute_C2C_HOSTNOTIFY
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact host_msix_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem commit_at_complete (ports : Ports) (unitArg : Unit) : Inv Complete (commit_at unitArg : SailM Nat) := by
  unfold commit_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ecc_poison_mask_complete (ports : Ports) (unitArg : Unit) : Inv Complete (ecc_poison_mask unitArg : SailM (BitVec Out.Functions.superlanes)) := by
  unfold ecc_poison_mask
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ecc_result_ov_complete (ports : Ports) (unitArg : Unit) : Inv Complete (ecc_result_ov unitArg : SailM ecc_ov) := by
  unfold ecc_result_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ecc_poison_mask_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem plane_stream_complete (ports : Ports) (s : streamref) (b : Int) : Inv Complete (plane_stream s b : SailM Nat) := by
  unfold plane_stream
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem write_planes_fwd_complete (ports : Ports) (s : streamref) (p : (Vector vec 4)) (w : Nat) (c : Nat) (fwd : (Option ecc_ov)) : Inv Complete (write_planes_fwd s p w c fwd : SailM (Result Unit diag)) := by
  unfold write_planes_fwd
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ecc_result_ov_complete ports _
    | exact plane_stream_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem write_planes_complete (ports : Ports) (s : streamref) (p : (Vector vec 4)) (w : Nat) (c : Nat) : Inv Complete (write_planes s p w c : SailM (Result Unit diag)) := by
  unfold write_planes
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact write_planes_fwd_complete ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_PCIRECV_complete (ports : Ports) (s : streamref) : Inv Complete (execute_C2C_PCIRECV s : SailM ExecutionResult) := by
  unfold execute_C2C_PCIRECV
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | exact exec_cycle_complete ports _
    | exact write_planes_complete ports _ _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.host_recv _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.host_recv_ready _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem pci_busy_complete (ports : Ports) (c : Nat) : Inv Complete (pci_busy c : SailM Bool) := by
  unfold pci_busy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ras_spare_event_complete (ports : Ports) (c : Nat) (q : Nat) (isw : Bool) (id : Nat) (gap : Nat) (sl : Nat) (s03 : (BitVec 4)) : Inv Complete (ras_spare_event c q isw id gap sl s03 : SailM Unit) := by
  unfold ras_spare_event
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ras_record_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_spare_complete (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv Complete (srf_spare r s g t : SailM (BitVec Out.Functions.superlanes)) := by
  unfold srf_spare
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact get_stream_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_s03_complete (ports : Ports) (rw : Bool) (g : Int) (t : Int) (x : Int) : Inv Complete (srf_s03 rw g t x : SailM (BitVec 4)) := by
  unfold srf_s03
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact srf_spare_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ecc_check_value_complete (ports : Ports) (ci : Int) (q : Nat) (kind : Nat) (isw : Bool) (id : Nat) (addr : Nat) (v : vec) (ov : ecc_ov) : Inv Complete (ecc_check_value ci q kind isw id addr v ov : SailM (vec × (BitVec Out.Functions.superlanes))) := by
  unfold ecc_check_value
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ras_ecc_event_complete ports _ _ _ _ _ _ _ _ _
    | exact ras_spare_event_complete ports _ _ _ _ _ _ _
    | exact srf_s03_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_read_chain_ov_complete (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv Complete (srf_read_chain_ov r s g t : SailM (srf_lookup × vec × ecc_ov)) := by
  unfold srf_read_chain_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact get_stream_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_only_downstream_complete (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv Complete (srf_only_downstream r s g t : SailM Bool) := by
  unfold srf_only_downstream
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact get_stream_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_read_ov_complete (ports : Ports) (r : row) (s : Nat) (g : Nat) (t : Int) : Inv Complete (srf_read_ov r s g t : SailM (srf_lookup × vec × ecc_ov)) := by
  unfold srf_read_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact get_stream_complete ports _ _
    | exact srf_only_downstream_complete ports _ _ _ _
    | exact srf_spare_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem read_planes_full_complete (ports : Ports) (s : streamref) (w : Nat) (t : Int) (chain : Bool) (chk : Bool) : Inv Complete (read_planes_full s w t chain chk : SailM (Result ((Vector vec 4) × (Vector ecc_ov 4)) diag)) := by
  unfold read_planes_full
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ecc_check_value_complete ports _ _ _ _ _ _ _ _
    | exact plane_stream_complete ports _ _
    | exact srf_read_chain_ov_complete ports _ _ _ _
    | exact srf_read_ov_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem read_planes_core_complete (ports : Ports) (s : streamref) (w : Nat) (t : Int) (chain : Bool) : Inv Complete (read_planes_core s w t chain : SailM (Result (Vector vec 4) diag)) := by
  unfold read_planes_core
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_full_complete ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem read_planes_complete (ports : Ports) (s : streamref) (w : Nat) (t : Int) : Inv Complete (read_planes s w t : SailM (Result (Vector vec 4) diag)) := by
  unfold read_planes
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_core_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_PCISEND_complete (ports : Ports) (s : streamref) : Inv Complete (execute_C2C_PCISEND s : SailM ExecutionResult) := by
  unfold execute_C2C_PCISEND
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact pci_busy_complete ports _
    | exact read_planes_complete ports _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.host_send _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem c2c_link_complete (ports : Ports) (link : (BitVec 7)) : Inv Complete (c2c_link link : SailM (Option Nat)) := by
  unfold c2c_link
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem c2c_peer_ok_complete (ports : Ports) (chip : (BitVec 9)) (l : Nat) : Inv Complete (c2c_peer_ok chip l : SailM Bool) := by
  unfold c2c_peer_ok
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_peer _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_RECEIVE_complete (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Inv Complete (execute_C2C_RECEIVE chip link s : SailM ExecutionResult) := by
  unfold execute_C2C_RECEIVE
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact c2c_link_complete ports _
    | exact c2c_peer_ok_complete ports _ _
    | exact commit_at_complete ports _
    | exact exec_cycle_complete ports _
    | exact write_planes_complete ports _ _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_recv _ _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_recv_bad _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_recv_poison _ _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_recv_ready _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem c2c_busy_complete (ports : Ports) (l : Nat) (c : Nat) : Inv Complete (c2c_busy l c : SailM Bool) := by
  unfold c2c_busy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem c2c_put_complete (ports : Ports) (l : Nat) (t : c2c_ptype) (v : vec) (poison : Nat) : Inv Complete (c2c_put l t v poison : SailM Unit) := by
  unfold c2c_put
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_send _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_SEND_complete (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Inv Complete (execute_C2C_SEND chip link s : SailM ExecutionResult) := by
  unfold execute_C2C_SEND
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact c2c_busy_complete ports _ _
    | exact c2c_link_complete ports _
    | exact c2c_peer_ok_complete ports _ _
    | exact c2c_put_complete ports _ _ _ _
    | exact exec_cycle_complete ports _
    | exact read_planes_complete ports _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_packet_busy _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_C2C_TRANSMIT_complete (ports : Ports) (chip : (BitVec 9)) (link : (BitVec 7)) : Inv Complete (execute_C2C_TRANSMIT chip link : SailM ExecutionResult) := by
  unfold execute_C2C_TRANSMIT
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact c2c_busy_complete ports _ _
    | exact c2c_link_complete ports _
    | exact c2c_peer_ok_complete ports _ _
    | exact c2c_put_complete ports _ _ _ _
    | exact exec_cycle_complete ports _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_packet_busy _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_CONFIG_complete (ports : Ports) (v : (BitVec 16)) : Inv Complete (execute_CONFIG v : SailM ExecutionResult) := by
  unfold execute_CONFIG
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_arm_complete (ports : Ports) (q : Nat) (st : streamref) (t : Nat) (data : vec) (ov : ecc_ov) : Inv Complete (stream_fetch_arm q st t data ov : SailM Unit) := by
  unfold stream_fetch_arm
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_start_complete (ports : Ports) (st : streamref) : Inv Complete (stream_fetch_start st : SailM ExecutionResult) := by
  unfold stream_fetch_start
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact read_planes_complete ports _ _ _
    | exact stream_fetch_arm_complete ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_IFETCH_complete (ports : Ports) (st : streamref) : Inv Complete (execute_IFETCH st : SailM ExecutionResult) := by
  unfold execute_IFETCH
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact stream_fetch_start_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_IFETCHM_complete (ports : Ports) (app_0 : sram_addr) : Inv Complete (execute_IFETCHM app_0 : SailM ExecutionResult) := by
  unfold execute_IFETCHM
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_ecc _ _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem commit_cycle_complete (ports : Ports) (unitArg : Unit) : Inv Complete (commit_cycle unitArg : SailM Nat) := by
  unfold commit_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem eff_addr_complete (ports : Ports) (app_0 : sram_addr) : Inv Complete (eff_addr app_0 : SailM (Option sram_addr)) := by
  unfold eff_addr
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem eff_stream_complete (ports : Ports) (app_0 : streamref) : Inv Complete (eff_stream app_0 : SailM (Option streamref)) := by
  unfold eff_stream
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem read_cycle_complete (ports : Ports) (unitArg : Unit) : Inv Complete (read_cycle unitArg : SailM Int) := by
  unfold read_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem read_plane_raw_complete (ports : Ports) (s : streamref) (t : Int) : Inv Complete (read_plane_raw s t : SailM (Result (vec × ecc_ov) diag)) := by
  unfold read_plane_raw
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_full_complete ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sram_read_complete (ports : Ports) (s : Nat) (a : sram_addr) : Inv Complete (sram_read s a : SailM (Option vec)) := by
  unfold sram_read
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sram_read_ov_complete (ports : Ports) (s : Nat) (a : sram_addr) : Inv Complete (sram_read_ov s a : SailM ecc_ov) := by
  unfold sram_read_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_ecc _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem write_sram_ecc_complete (ports : Ports) (s : Nat) (a : Nat) (v : ecc_ov) : Inv Complete (write_sram_ecc s a v : SailM Unit) := by
  unfold write_sram_ecc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.raw_write_sram_ecc _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem write_sram_word_complete (ports : Ports) (s : Nat) (a : Nat) (v : vec) : Inv Complete (write_sram_word s a v : SailM Unit) := by
  unfold write_sram_word
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.raw_write_sram_word _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sram_write_ov_complete (ports : Ports) (c : Nat) (s : Nat) (typ_2 : sram_addr) (v : vec) (ov : ecc_ov) : Inv Complete (sram_write_ov c s typ_2 v ov : SailM Bool) := by
  unfold sram_write_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact write_sram_ecc_complete ports _ _ _
    | exact write_sram_word_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem tile_on_complete (ports : Ports) (g : Int) : Inv Complete (tile_on g : SailM Bool) := by
  unfold tile_on
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_DIRECT_complete (ports : Ports) (op : mem_dir) (a0 : sram_addr) (s0 : streamref) (t : (BitVec 1)) : Inv Complete (execute_MEM_DIRECT op a0 s0 t : SailM ExecutionResult) := by
  unfold execute_MEM_DIRECT
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_cycle_complete ports _
    | exact eff_addr_complete ports _
    | exact eff_stream_complete ports _
    | exact exec_cycle_complete ports _
    | exact read_cycle_complete ports _
    | exact read_plane_raw_complete ports _ _
    | exact sram_read_complete ports _ _
    | exact sram_read_ov_complete ports _ _
    | exact sram_write_ov_complete ports _ _ _ _ _
    | exact tile_on_complete ports _
    | exact write_planes_fwd_complete ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ecc_check_sl_complete (ports : Ports) (ci : Int) (q : Nat) (slice_ : Nat) (addr : Nat) (g : Nat) (d : (BitVec 128)) (e : (BitVec 10)) : Inv Complete (ecc_check_sl ci q slice_ addr g d e : SailM ((BitVec 128) × Bool)) := by
  unfold ecc_check_sl
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ras_ecc_event_complete ports _ _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem srf_report_complete (ports : Ports) (r : row) (g : Nat) (c : Nat) (prio : Int) (xs : (BitVec Out.Functions.superlanes)) (sp : (Vector (BitVec Out.Functions.superlanes) 30)) : Inv Complete (srf_report r g c prio xs sp : SailM Bool) := by
  unfold srf_report
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mem_report_complete (ports : Ports) (prio : Int) (bounds : (BitVec Out.Functions.superlanes)) (addrs : (Vector Int Out.Functions.superlanes)) (data : streamref) (map : streamref) (c : Nat) : Inv Complete (mem_report prio bounds addrs data map c : SailM Unit) := by
  unfold mem_report
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact srf_report_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_GATHER_complete (ports : Ports) (d0 : streamref) (m : streamref) (kb : (BitVec 4)) (t : (BitVec 1)) : Inv Complete (execute_MEM_GATHER d0 m kb t : SailM ExecutionResult) := by
  unfold execute_MEM_GATHER
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_cycle_complete ports _
    | exact ecc_check_sl_complete ports _ _ _ _ _ _ _
    | exact eff_addr_complete ports _
    | exact eff_stream_complete ports _
    | exact mem_report_complete ports _ _ _ _ _ _
    | exact read_cycle_complete ports _
    | exact read_planes_complete ports _ _ _
    | exact sram_read_complete ports _ _
    | exact sram_read_ov_complete ports _ _
    | exact tile_on_complete ports _
    | exact write_planes_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_SCATTER_complete (ports : Ports) (s0 : streamref) (m : streamref) (kb : (BitVec 4)) (t : (BitVec 1)) : Inv Complete (execute_MEM_SCATTER s0 m kb t : SailM ExecutionResult) := by
  unfold execute_MEM_SCATTER
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_cycle_complete ports _
    | exact ecc_check_sl_complete ports _ _ _ _ _ _ _
    | exact eff_addr_complete ports _
    | exact eff_stream_complete ports _
    | exact exec_cycle_complete ports _
    | exact mem_report_complete ports _ _ _ _ _ _
    | exact read_cycle_complete ports _
    | exact read_planes_complete ports _ _ _
    | exact sram_read_complete ports _ _
    | exact sram_read_ov_complete ports _ _
    | exact sram_write_ov_complete ports _ _ _ _ _
    | exact tile_on_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MEM_SETSEGMENT_complete (ports : Ports) (n : (BitVec 17)) : Inv Complete (execute_MEM_SETSEGMENT n : SailM ExecutionResult) := by
  unfold execute_MEM_SETSEGMENT
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem macc_busy_in_complete (ports : Ports) (xs : (List abc_entry)) : Inv Complete (macc_busy_in xs : SailM Bool) := by
  induction xs
  case nil =>
    unfold macc_busy_in
    repeat' first
      | exact ih 
      | assumption
      | exact pure_inv _ _
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
  case cons head tail ih =>
    unfold macc_busy_in
    repeat' first
      | exact ih 
      | assumption
      | exact pure_inv _ _
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem macc_busy_complete (ports : Ports) (p : mxm_plane) : Inv Complete (macc_busy p : SailM Bool) := by
  unfold macc_busy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact macc_busy_in_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_stage_acc_complete (ports : Ports) (pl : Nat) (part : mxm_part) (a : Nat) (v : mxm_plane) : Inv Complete (mxm_stage_acc pl part a v : SailM Unit) := by
  unfold mxm_stage_acc
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_stage_complete (ports : Ports) (pl : Nat) (part : mxm_part) (v : mxm_plane) : Inv Complete (mxm_stage pl part v : SailM Unit) := by
  unfold mxm_stage
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact mxm_stage_acc_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem plane_complete (ports : Ports) (unitArg : Unit) : Inv Complete (plane unitArg : SailM Nat) := by
  unfold plane
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_plane_at_complete (ports : Ports) (i : Int) : Inv Complete (mxm_plane_at i : SailM mxm_plane) := by
  unfold mxm_plane_at
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem plane_blocked_complete (ports : Ports) (t : (Option mxm_wtype)) : Inv Complete (plane_blocked t : SailM Bool) := by
  unfold plane_blocked
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact mxm_plane_at_complete ports _
    | exact plane_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ras_mxm_event_complete (ports : Ports) (c : Nat) (q : Int) (unc : Bool) (plane : Nat) (lane : Nat) : Inv Complete (ras_mxm_event c q unc plane lane : SailM Unit) := by
  unfold ras_mxm_event
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact plane_complete ports _
    | exact ras_record_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
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

theorem abcFaultPhase_complete (ports : Ports) (mine : List (Nat × Nat)) (others pend : List mxm_fault) : Inv Complete (abcFaultPhase mine others pend) := by
  unfold abcFaultPhase
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact plane_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro

theorem abcProductPhase_complete (ports : Ports) (t : mxm_wtype) (f : BitVec 1) (p : mxm_plane) (x : planes) (ws xs : Bool) (out : Vector (BitVec 32) Out.Functions.mxm_rows) (wide : Vector wlane Out.Functions.mxm_rows) : Inv Complete (abcProductPhase t f p x ws xs out wide) := by
  unfold abcProductPhase
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact plane_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro

theorem abcPoisonPhase_complete (ports : Ports) (t : mxm_wtype) (p : mxm_plane) (poison : BitVec Out.Functions.superlanes) : Inv Complete (abcPoisonPhase t p poison) := by
  unfold abcPoisonPhase
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact plane_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro

attribute [local irreducible] abcFaultPhase abcProductPhase abcPoisonPhase

theorem abc_exec_complete (ports : Ports) (t : mxm_wtype) (f : (BitVec 1)) (s : streamref) : Inv Complete (abc_exec t f s : SailM ExecutionResult) := by
  rw [actual_abc_factory_source]
  unfold abcExecFactory
  repeat' first
    | exact abcFaultPhase_complete ports _ _ _
    | exact abcProductPhase_complete ports _ _ _ _ _ _ _ _
    | exact abcPoisonPhase_complete ports _ _ _
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | exact exec_cycle_complete ports _
    | exact macc_busy_complete ports _
    | exact mxm_stage_complete ports _ _ _
    | exact plane_complete ports _
    | exact plane_blocked_complete ports _
    | exact ras_mxm_event_complete ports _ _ _ _ _
    | exact read_planes_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_ABC_complete (ports : Ports) (t : mxm_wtype) (s : streamref) : Inv Complete (execute_MXM_ABC t s : SailM ExecutionResult) := by
  unfold execute_MXM_ABC
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact abc_exec_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_ABC8_complete (ports : Ports) (f : (BitVec 1)) (s : streamref) : Inv Complete (execute_MXM_ABC8 f s : SailM ExecutionResult) := by
  unfold execute_MXM_ABC8
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact abc_exec_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_ACC_complete (ports : Ports) (t : mxm_acctype) (m : acc_mode) (d : streamref) (ab : (BitVec 5)) : Inv Complete (execute_MXM_ACC t m d ab : SailM ExecutionResult) := by
  unfold execute_MXM_ACC
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | exact mxm_stage_acc_complete ports _ _ _ _
    | exact plane_complete ports _
    | exact plane_blocked_complete ports _
    | exact write_planes_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_IW_complete (ports : Ports) (t : mxm_wtype) : Inv Complete (execute_MXM_IW t : SailM ExecutionResult) := by
  unfold execute_MXM_IW
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | exact mxm_stage_complete ports _ _ _
    | exact plane_complete ports _
    | exact plane_blocked_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem read_planes16_complete (ports : Ports) (s : streamref) (t : Int) : Inv Complete (read_planes16 s t : SailM (Result (Vector vec 16) diag)) := by
  unfold read_planes16
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ecc_check_value_complete ports _ _ _ _ _ _ _ _
    | exact plane_stream_complete ports _ _
    | exact srf_read_ov_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_IWS_complete (ports : Ports) (t : mxm_wtype) (s : streamref) : Inv Complete (execute_MXM_IWS t s : SailM ExecutionResult) := by
  unfold execute_MXM_IWS
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | exact exec_cycle_complete ports _
    | exact mxm_stage_complete ports _ _ _
    | exact plane_complete ports _
    | exact plane_blocked_complete ports _
    | exact read_planes16_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_MXM_LW_complete (ports : Ports) (t : mxm_wtype) (s : streamref) : Inv Complete (execute_MXM_LW t s : SailM ExecutionResult) := by
  unfold execute_MXM_LW
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | exact exec_cycle_complete ports _
    | exact mxm_stage_complete ports _ _ _
    | exact plane_complete ports _
    | exact plane_blocked_complete ports _
    | exact read_planes16_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SETVL_complete (ports : Ports) (n : (BitVec 6)) : Inv Complete (execute_SETVL n : SailM ExecutionResult) := by
  unfold execute_SETVL
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sx_complete (ports : Ports) (unitArg : Unit) : Inv Complete (sx unitArg : SailM Nat) := by
  unfold sx
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sxm_load_map_complete (ports : Ports) (mi : (BitVec 4)) (lo : vec) (hi : vec) : Inv Complete (sxm_load_map mi lo hi : SailM Unit) := by
  unfold sxm_load_map
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact sx_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sxm_rd_cycle_complete (ports : Ports) (unitArg : Unit) : Inv Complete (sxm_rd_cycle unitArg : SailM Int) := by
  unfold sxm_rd_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_LOADMAP_complete (ports : Ports) (mi : (BitVec 4)) (s : streamref) : Inv Complete (execute_SXM_LOADMAP mi s : SailM ExecutionResult) := by
  unfold execute_SXM_LOADMAP
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_complete ports _ _ _
    | exact sxm_load_map_complete ports _ _ _
    | exact sxm_rd_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_LOADMAPWIDE_complete (ports : Ports) (mi : (BitVec 4)) (s : streamref) (s2 : streamref) : Inv Complete (execute_SXM_LOADMAPWIDE mi s s2 : SailM ExecutionResult) := by
  unfold execute_SXM_LOADMAPWIDE
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_complete ports _ _ _
    | exact sxm_load_map_complete ports _ _ _
    | exact sxm_rd_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sxm_cm_cycle_complete (ports : Ports) (unitArg : Unit) : Inv Complete (sxm_cm_cycle unitArg : SailM Nat) := by
  unfold sxm_cm_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sxm_emit_complete (ports : Ports) (d : streamref) (v : vec) : Inv Complete (sxm_emit d v : SailM ExecutionResult) := by
  unfold sxm_emit
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact sxm_cm_cycle_complete ports _
    | exact write_planes_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_MAP_complete (ports : Ports) (op : sxm_mapop) (mi : (BitVec 4)) (s : streamref) (d : streamref) : Inv Complete (execute_SXM_MAP op mi s d : SailM ExecutionResult) := by
  unfold execute_SXM_MAP
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | exact read_planes_complete ports _ _ _
    | exact sx_complete ports _
    | exact sxm_emit_complete ports _ _
    | exact sxm_rd_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_ROTATE_complete (ports : Ports) (nb : (BitVec 10)) (s : streamref) (d : streamref) : Inv Complete (execute_SXM_ROTATE nb s d : SailM ExecutionResult) := by
  unfold execute_SXM_ROTATE
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_complete ports _ _ _
    | exact sxm_cm_cycle_complete ports _
    | exact sxm_rd_cycle_complete ports _
    | exact write_planes_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_SELECT_complete (ports : Ports) (a : streamref) (b : streamref) (c : streamref) (m : streamref) (d : streamref) : Inv Complete (execute_SXM_SELECT a b c m d : SailM ExecutionResult) := by
  unfold execute_SXM_SELECT
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_complete ports _ _ _
    | exact sxm_emit_complete ports _ _
    | exact sxm_rd_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_SHIFT_complete (ports : Ports) (dir : shift_dir) (n : (BitVec 9)) (s : streamref) (d : streamref) : Inv Complete (execute_SXM_SHIFT dir n s d : SailM ExecutionResult) := by
  unfold execute_SXM_SHIFT
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_complete ports _ _ _
    | exact sxm_emit_complete ports _ _
    | exact sxm_rd_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_SXM_TRANSPOSE_complete (ports : Ports) (s : streamref) (d : streamref) : Inv Complete (execute_SXM_TRANSPOSE s d : SailM ExecutionResult) := by
  unfold execute_SXM_TRANSPOSE
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes16_complete ports _ _
    | exact sxm_cm_cycle_complete ports _
    | exact sxm_rd_cycle_complete ports _
    | exact write_planes_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem sailThrow_complete {α : Type} (ports : Ports) (e : exception) : Inv Complete (sailThrow e : SailM α) := by
  unfold sailThrow PreSail.sailThrow
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem internal_error_complete {α : Type} (ports : Ports) (file : String) (line : Int) (s : String) : Inv Complete (internal_error file line s : SailM α) := by
  unfold internal_error
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact sailThrow_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem fp_transcendental_complete (ports : Ports) (op : Int) (f : fmt) (x : (BitVec 32)) : Inv Complete (fp_transcendental op f x : SailM (BitVec 32)) := by
  unfold fp_transcendental
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact internal_error_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem i8_transcendental_complete (ports : Ports) (op : Int) (x : (BitVec 32)) : Inv Complete (i8_transcendental op x : SailM (BitVec 32)) := by
  unfold i8_transcendental
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact internal_error_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem apply_complete (ports : Ports) (op : vxm_lane_op) (a : (BitVec 32)) (b : (BitVec 32)) (m : (BitVec 32)) : Inv Complete (apply op a b m : SailM (BitVec 32)) := by
  unfold apply
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact fp_transcendental_complete ports _ _ _
    | exact i8_transcendental_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem vxm_commit_cycle_complete (ports : Ports) (unitArg : Unit) : Inv Complete (vxm_commit_cycle unitArg : SailM Nat) := by
  unfold vxm_commit_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact commit_at_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem vxm_read_cycle_complete (ports : Ports) (unitArg : Unit) : Inv Complete (vxm_read_cycle unitArg : SailM Int) := by
  unfold vxm_read_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact exec_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem vxm_read_complete (ports : Ports) (a : vsrc) (w : Nat) : Inv Complete (vxm_read a w : SailM (Result (Vector vec 4) diag)) := by
  unfold vxm_read
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact read_planes_core_complete ports _ _ _ _
    | exact vxm_read_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem vxm_lanes_complete (ports : Ports) (op : vxm_lane_op) (dst : streamref) (a : vsrc) (b : (Option vsrc)) (m : (Option vsrc)) (ws : Nat) (wd : Nat) (g : (BitVec 1)) : Inv Complete (vxm_lanes op dst a b m ws wd g : SailM ExecutionResult) := by
  unfold vxm_lanes
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact apply_complete ports _ _ _ _
    | exact vxm_commit_cycle_complete ports _
    | exact vxm_read_complete ports _ _
    | exact write_planes_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_BIN_complete (ports : Ports) (op : vxm_binop) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (g : (BitVec 1)) : Inv Complete (execute_VXM_BIN op t d a b g : SailM ExecutionResult) := by
  unfold execute_VXM_BIN
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_lanes_complete ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_CMP_complete (ports : Ports) (mode : cmp_mode) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (g : (BitVec 1)) : Inv Complete (execute_VXM_CMP mode t d a b g : SailM ExecutionResult) := by
  unfold execute_VXM_CMP
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_lanes_complete ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_CONV_complete (ports : Ports) (c : vxm_conv) (g__241 : eltype) (d : streamref) (a : vsrc) : Inv Complete (execute_VXM_CONV c g__241 d a : SailM ExecutionResult) := by
  unfold execute_VXM_CONV
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_lanes_complete ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_CVT_complete (ports : Ports) (st : eltype) (dt : eltype) (d : streamref) (a : vsrc) : Inv Complete (execute_VXM_CVT st dt d a : SailM ExecutionResult) := by
  unfold execute_VXM_CVT
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_lanes_complete ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_MASK_complete (ports : Ports) (t : eltype) (d : streamref) (a : vsrc) (m : vsrc) (g : (BitVec 1)) : Inv Complete (execute_VXM_MASK t d a m g : SailM ExecutionResult) := by
  unfold execute_VXM_MASK
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_lanes_complete ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_REDUCE_complete (ports : Ports) (op : reduce_op) (t : eltype) (d : streamref) (a : vsrc) : Inv Complete (execute_VXM_REDUCE op t d a : SailM ExecutionResult) := by
  unfold execute_VXM_REDUCE
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_commit_cycle_complete ports _
    | exact vxm_read_complete ports _ _
    | exact write_planes_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_SELECT_complete (ports : Ports) (t : eltype) (d : streamref) (a : vsrc) (b : vsrc) (m : vsrc) (g : (BitVec 1)) : Inv Complete (execute_VXM_SELECT t d a b m g : SailM ExecutionResult) := by
  unfold execute_VXM_SELECT
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_lanes_complete ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_VXM_UN_complete (ports : Ports) (op : vxm_unop) (t : eltype) (d : streamref) (a : vsrc) (g : (BitVec 1)) : Inv Complete (execute_VXM_UN op t d a g : SailM ExecutionResult) := by
  unfold execute_VXM_UN
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact vxm_lanes_complete ports _ _ _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_complete (ports : Ports) (merge_var : instruction) : Inv Complete (execute merge_var : SailM ExecutionResult) := by
  unfold execute
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact execute_C2C_HOSTNOTIFY_complete ports _
    | exact execute_C2C_PCIRECV_complete ports _
    | exact execute_C2C_PCISEND_complete ports _
    | exact execute_C2C_RECEIVE_complete ports _ _ _
    | exact execute_C2C_SEND_complete ports _ _ _
    | exact execute_C2C_TRANSMIT_complete ports _ _
    | exact execute_CONFIG_complete ports _
    | exact execute_IFETCH_complete ports _
    | exact execute_IFETCHM_complete ports _
    | exact execute_MEM_DIRECT_complete ports _ _ _ _
    | exact execute_MEM_GATHER_complete ports _ _ _ _
    | exact execute_MEM_SCATTER_complete ports _ _ _ _
    | exact execute_MEM_SETSEGMENT_complete ports _
    | exact execute_MXM_ABC_complete ports _ _
    | exact execute_MXM_ABC8_complete ports _ _
    | exact execute_MXM_ACC_complete ports _ _ _ _
    | exact execute_MXM_IW_complete ports _
    | exact execute_MXM_IWS_complete ports _ _
    | exact execute_MXM_LW_complete ports _ _
    | exact execute_SETVL_complete ports _
    | exact execute_SXM_LOADMAP_complete ports _ _
    | exact execute_SXM_LOADMAPWIDE_complete ports _ _ _
    | exact execute_SXM_MAP_complete ports _ _ _ _
    | exact execute_SXM_ROTATE_complete ports _ _ _
    | exact execute_SXM_SELECT_complete ports _ _ _ _ _
    | exact execute_SXM_SHIFT_complete ports _ _ _ _
    | exact execute_SXM_TRANSPOSE_complete ports _ _
    | exact execute_VXM_BIN_complete ports _ _ _ _ _ _
    | exact execute_VXM_CMP_complete ports _ _ _ _ _ _
    | exact execute_VXM_CONV_complete ports _ _ _ _
    | exact execute_VXM_CVT_complete ports _ _ _ _
    | exact execute_VXM_MASK_complete ports _ _ _ _ _
    | exact execute_VXM_REDUCE_complete ports _ _ _ _
    | exact execute_VXM_SELECT_complete ports _ _ _ _ _ _
    | exact execute_VXM_UN_complete ports _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_element_complete (ports : Ports) (e : exec_element) : Inv Complete (execute_element e : SailM Unit) := by
  unfold execute_element
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ecc_begin_complete ports _
    | exact exec_cycle_complete ports _
    | exact execute_complete ports _
    | exact report_complete ports _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem execute_all_complete (ports : Ports) (xs : (List exec_element)) : Inv Complete (execute_all xs : SailM Unit) := by
  induction xs
  case nil =>
    unfold execute_all
    repeat' first
      | exact ih 
      | assumption
      | exact pure_inv _ _
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact execute_element_complete ports _
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
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact execute_element_complete ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem execute_due_complete (ports : Ports) (c : Nat) : Inv Complete (execute_due c : SailM Unit) := by
  unfold execute_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact execute_all_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem iq_fetch_ov_complete (ports : Ports) (q : Nat) (t : Nat) (a : vec) (b : vec) (ova : ecc_ov) (ovb : ecc_ov) : Inv Complete (iq_fetch_ov q t a b ova ovb : SailM Bool) := by
  unfold iq_fetch_ov
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem local_fetch_complete_complete (ports : Ports) (c : Nat) : Inv Complete (local_fetch_complete c : SailM Unit) := by
  unfold local_fetch_complete
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact diag_report_complete ports _ _ _
    | exact iq_fetch_ov_complete ports _ _ _ _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_ecc _ _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem held_complete (ports : Ports) (q : Nat) : Inv Complete (held q : SailM Bool) := by
  unfold held
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_issued_complete (ports : Ports) (q : Nat) (m : Nat) : Inv Complete (scrub_issued q m : SailM Bool) := by
  unfold scrub_issued
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem ras_scrub_event_complete (ports : Ports) (c : Nat) (s : Nat) (unc : Bool) (a : Nat) (sl : Nat) (pos : Nat) : Inv Complete (ras_scrub_event c s unc a sl pos : SailM Unit) := by
  unfold ras_scrub_event
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ras_record_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_slice_complete (ports : Ports) (s : Nat) (m : Nat) : Inv Complete (scrub_slice s m : SailM Unit) := by
  unfold scrub_slice
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact ras_scrub_event_complete ports _ _ _ _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_ecc _ _)
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.read_sram_word _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_take_complete (ports : Ports) (s : Nat) : Inv Complete (scrub_take s : SailM (Option scrub_write)) := by
  unfold scrub_take
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_cycle_complete (ports : Ports) (m : Nat) : Inv Complete (scrub_cycle m : SailM Unit) := by
  unfold scrub_cycle
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact held_complete ports _
    | exact scrub_issued_complete ports _ _
    | exact scrub_slice_complete ports _ _
    | exact scrub_take_complete ports _
    | exact write_sram_ecc_complete ports _ _ _
    | exact write_sram_word_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem scrub_run_complete (ports : Ports) (upto : Int) : Inv Complete (scrub_run upto : SailM Unit) := by
  unfold scrub_run
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact scrub_cycle_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_take_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (stream_fetch_take q c : SailM (Option stream_fetch)) := by
  unfold stream_fetch_take
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_service_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (stream_fetch_service q c : SailM Unit) := by
  unfold stream_fetch_service
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact diag_report_complete ports _ _ _
    | exact ecc_begin_complete ports _
    | exact iq_fetch_ov_complete ports _ _ _ _ _ _
    | exact read_planes_complete ports _ _ _
    | exact stream_fetch_take_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem stream_fetch_complete_complete (ports : Ports) (c : Nat) : Inv Complete (stream_fetch_complete c : SailM Unit) := by
  unfold stream_fetch_complete
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact stream_fetch_service_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_set_complete (ports : Ports) (q : Nat) (p : Nat) (a : agt) : Inv Complete (agt_set q p a : SailM Unit) := by
  unfold agt_set
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_finish_config_complete (ports : Ports) (q : Nat) (p : Nat) : Inv Complete (agt_finish_config q p : SailM Unit) := by
  unfold agt_finish_config
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | exact agt_set_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem deskewrecv_ready_complete (ports : Ports) (c : Nat) (link : (BitVec 7)) : Inv Complete (deskewrecv_ready c link : SailM Bool) := by
  unfold deskewrecv_ready
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact c2c_link_complete ports _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.c2c_recv_ready _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem deskewrecv_diag_complete (ports : Ports) (iu : issuer) (chip : (BitVec 9)) (link : (BitVec 7)) (s : streamref) : Inv Complete (deskewrecv_diag iu chip link s : SailM (Option diag)) := by
  unfold deskewrecv_diag
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact c2c_link_complete ports _
    | exact c2c_peer_ok_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem host_wait_complete (ports : Ports) (q : Nat) (c : Nat) (i : Nat) : Inv Complete (host_wait q c i : SailM Unit) := by
  unfold host_wait
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem occupancy_complete (ports : Ports) (arg0 : instruction) (arg1 : Nat) : Inv Complete (occupancy arg0 arg1 : SailM Nat) := by
  unfold occupancy
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem run_element_complete (ports : Ports) (q : Nat) (i : instruction) (c0 : Nat) (k : Nat) (off : Int) (soff : Int) : Inv Complete (run_element q i c0 k off soff : SailM Unit) := by
  unfold run_element
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact execute_element_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_body_complete (ports : Ports) (q : Nat) (c : Nat) (i : instruction) : Inv Complete (dispatch_body q c i : SailM Unit) := by
  unfold dispatch_body
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact barrier_notify_complete ports _
    | exact barrier_sync_complete ports _ _
    | exact deskewrecv_diag_complete ports _ _ _ _
    | exact host_wait_complete ports _ _ _
    | exact occupancy_complete ports _ _
    | exact report_complete ports _ _ _
    | exact run_element_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_check_complete (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) (strict : Bool) : Inv Complete (hold_check q c i path strict : SailM Bool) := by
  unfold hold_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact hold_view_of_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_check_complete (ports : Ports) (q : Nat) (c : Nat) (i : instruction) (path : (Option Nat)) : Inv Complete (dispatch_check q c i path : SailM (Option diag)) := by
  unfold dispatch_check
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact hold_check_complete ports _ _ _ _ _
    | exact internal_error_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem reserve_local_fetch_complete (ports : Ports) (q : Nat) (i : instruction) (c : Nat) : Inv Complete (reserve_local_fetch q i c : SailM Unit) := by
  unfold reserve_local_fetch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem repeat_issue_complete (ports : Ports) (q : Nat) (i : instruction) (c : Nat) : Inv Complete (repeat_issue q i c : SailM Unit) := by
  unfold repeat_issue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact reserve_local_fetch_complete ports _ _ _
    | exact run_element_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_decoded_complete (ports : Ports) (q : Nat) (c : Nat) (i : instruction) : Inv Complete (dispatch_decoded q c i : SailM Unit) := by
  unfold dispatch_decoded
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact dispatch_body_complete ports _ _ _
    | exact dispatch_check_complete ports _ _ _ _
    | exact repeat_issue_complete ports _ _ _
    | exact report_complete ports _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem bank_report_complete (ports : Ports) (q : Nat) (c : Nat) (a : (instruction × Int × Int)) (b : (instruction × Int × Int)) : Inv Complete (bank_report q c a b : SailM Unit) := by
  unfold bank_report
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact srf_report_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem cfg_path_complete (ports : Ports) (q : Nat) : Inv Complete (cfg_path q : SailM Nat) := by
  unfold cfg_path
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_configure_complete (ports : Ports) (q : Nat) (p : Nat) (f : agt_field) (v : (BitVec 16)) : Inv Complete (agt_configure q p f v : SailM Bool) := by
  unfold agt_configure
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | exact agt_set_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_cfg_complete (ports : Ports) (q : Nat) (c : Nat) (p : Nat) (i : instruction) : Inv Complete (dispatch_cfg q c p i : SailM Unit) := by
  unfold dispatch_cfg
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_configure_complete ports _ _ _ _
    | exact agt_get_complete ports _ _
    | exact agt_set_complete ports _ _ _
    | exact dispatch_check_complete ports _ _ _ _
    | exact internal_error_complete ports _ _ _
    | exact report_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem loop_clash_ref_complete (ports : Ports) (q : Nat) (p : Nat) (s : Int) (ss : Int) : Inv Complete (loop_clash_ref q p s ss : SailM (instruction × Int × Int)) := by
  unfold loop_clash_ref
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem due_ref_complete (ports : Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Inv Complete (due_ref q p st : SailM (Option (instruction × Int × Int))) := by
  unfold due_ref
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact loop_clash_ref_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_due_complete (ports : Ports) (q : Nat) (p : Nat) (c : Nat) (parked : Bool) : Inv Complete (agt_due q p c parked : SailM (Option ((Vector agt_level 4) × Nat × Int × Int))) := by
  unfold agt_due
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem due_refs_complete (ports : Ports) (q : Nat) (c : Nat) (parked : Bool) : Inv Complete (due_refs q c parked : SailM (Vector (Option ((Vector agt_level 4) × Nat × Int × Int)) 2)) := by
  unfold due_refs
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_due_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem hold_path_complete (ports : Ports) (q : Nat) (p : Nat) (c : Nat) : Inv Complete (hold_path q p c : SailM Unit) := by
  unfold hold_path
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem loop_tag_complete (ports : Ports) (q : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) : Inv Complete (loop_tag q p st : SailM (Option ref_tag)) := by
  unfold loop_tag
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem agt_refresh_complete (ports : Ports) (q : Nat) (p : Nat) : Inv Complete (agt_refresh q p : SailM Unit) := by
  unfold agt_refresh
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | exact agt_set_complete ports _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem run_loop_complete (ports : Ports) (q : Nat) (c : Nat) (p : Nat) (st : (Option ((Vector agt_level 4) × Nat × Int × Int))) (blocked : Bool) (partner : (Option (instruction × Int × Int))) : Inv Complete (run_loop q c p st blocked partner : SailM Unit) := by
  unfold run_loop
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_get_complete ports _ _
    | exact agt_refresh_complete ports _ _
    | exact agt_set_complete ports _ _ _
    | exact bank_report_complete ports _ _ _ _
    | exact hold_path_complete ports _ _ _
    | exact loop_clash_ref_complete ports _ _ _ _
    | exact report_complete ports _ _ _
    | exact run_element_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem consumeNewRef_complete (ports : Ports) (q c : Nat) (r : cycle_recs) (loop : Vector (Option ref_tag) 2)
    (lref : Vector (Option clash_ref) 2) (go bank : Vector Bool 4) (k : Int)
    (issued : Nat) (last : Option instruction) : Inv Complete (GrantConsumptionProofs.consumeNewRef q c r loop lref go bank k issued last) := by
  unfold GrantConsumptionProofs.consumeNewRef
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact agt_set_complete ports _ _ _
    | exact bank_report_complete ports _ _ _ _
    | exact cfg_path_complete ports _
    | exact dispatch_cfg_complete ports _ _ _ _
    | exact dispatch_check_complete ports _ _ _ _
    | exact due_ref_complete ports _ _ _
    | exact due_refs_complete ports _ _ _
    | exact hold_path_complete ports _ _ _
    | exact loop_tag_complete ports _ _ _
    | exact report_complete ports _ _ _
    | exact reserve_local_fetch_complete ports _ _ _
    | exact run_element_complete ports _ _ _ _ _ _
    | exact run_loop_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem consumeFetch_complete (ports : Ports) (q c : Nat) (r : cycle_recs) (fm_ok fm_clash : Bool)
    (issued : Nat) (last : Option instruction) : Inv Complete (GrantConsumptionProofs.consumeFetch q c r fm_ok fm_clash issued last) := by
  unfold GrantConsumptionProofs.consumeFetch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact agt_set_complete ports _ _ _
    | exact bank_report_complete ports _ _ _ _
    | exact cfg_path_complete ports _
    | exact dispatch_cfg_complete ports _ _ _ _
    | exact dispatch_check_complete ports _ _ _ _
    | exact due_ref_complete ports _ _ _
    | exact due_refs_complete ports _ _ _
    | exact hold_path_complete ports _ _ _
    | exact loop_tag_complete ports _ _ _
    | exact report_complete ports _ _ _
    | exact reserve_local_fetch_complete ports _ _ _
    | exact run_element_complete ports _ _ _ _ _ _
    | exact run_loop_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro

theorem dispatch_slots_complete (ports : Ports) (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : Inv Complete (dispatch_slots q c r parked : SailM Unit) := by
  rw [GrantConsumptionProofs.actual_dispatch_factory_source]
  unfold GrantConsumptionProofs.dispatchFactory
  repeat' first
    | exact consumeNewRef_complete ports _ _ _ _ _ _ _ _ _ _
    | exact consumeFetch_complete ports _ _ _ _ _ _ _
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_set_complete ports _ _ _
    | exact bank_report_complete ports _ _ _ _
    | exact cfg_path_complete ports _
    | exact dispatch_cfg_complete ports _ _ _ _
    | exact dispatch_check_complete ports _ _ _ _
    | exact due_ref_complete ports _ _ _
    | exact due_refs_complete ports _ _ _
    | exact hold_path_complete ports _ _ _
    | exact loop_tag_complete ports _ _ _
    | exact report_complete ports _ _ _
    | exact reserve_local_fetch_complete ports _ _ _
    | exact run_element_complete ports _ _ _ _ _ _
    | exact run_loop_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem dispatch_complete (ports : Ports) (q : Nat) (c : Nat) (r : cycle_recs) (parked : Bool) : Inv Complete (dispatch q c r parked : SailM Unit) := by
  unfold dispatch
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact dispatch_decoded_complete ports _ _ _
    | exact dispatch_slots_complete ports _ _ _ _
    | exact due_ref_complete ports _ _ _
    | exact due_refs_complete ports _ _ _
    | exact loop_tag_complete ports _ _ _
    | exact report_complete ports _ _ _
    | exact run_loop_complete ports _ _ _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem strict_held_complete (ports : Ports) (q : Nat) (c : Nat) (xs : (List (Result instruction diag))) (held : Bool) : Inv Complete (strict_held q c xs held : SailM Bool) := by
  induction xs generalizing q c held
  case nil =>
    unfold strict_held
    repeat' first
      | exact ih _ _ _
      | assumption
      | exact pure_inv _ _
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact hold_check_complete ports _ _ _ _ _
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
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact hold_check_complete ports _ _ _ _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem parcel_held_complete (ports : Ports) (q : Nat) (c : Nat) (r : cycle_recs) (items : (List (Result instruction diag))) : Inv Complete (parcel_held q c r items : SailM Bool) := by
  unfold parcel_held
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact cfg_path_complete ports _
    | exact held_complete ports _
    | exact hold_check_complete ports _ _ _ _ _
    | exact strict_held_complete ports _ _ _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem tick_queue_complete (ports : Ports) (q : Nat) (c : Nat) : Inv Complete (tick_queue q c : SailM Unit) := by
  unfold tick_queue
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact agt_finish_config_complete ports _ _
    | exact agt_get_complete ports _ _
    | exact deskewrecv_ready_complete ports _ _
    | exact dispatch_complete ports _ _ _ _
    | exact host_parked_complete ports _ _
    | exact iq_pop_complete ports _ _
    | exact iq_text_check_complete ports _ _ _
    | exact occupancy_complete ports _ _
    | exact parcel_earliest_complete ports _ _
    | exact parcel_held_complete ports _ _ _ _
    | exact parcel_scan_complete ports _
    | exact polling_complete ports _
    | exact queue_parked_complete ports _ _
    | exact repeat_issue_complete ports _ _ _
    | exact run_element_complete ports _ _ _ _ _ _
    | exact keeps_pred (fun s : ResourceFrameProofs.St => s.regs) (fun regs => ∀r,∃v,regs.get? r=some v) _ (ports.observe_site _ _ _ _ _ _)
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_complete (ports : Ports) (e : mxm_staged) : Inv Complete (mxm_apply e : SailM Unit) := by
  unfold mxm_apply
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem mxm_apply_ok_complete (ports : Ports) (xs : (List mxm_staged)) (bad : (List Nat)) : Inv Complete (mxm_apply_ok xs bad : SailM Unit) := by
  induction xs generalizing bad
  case nil =>
    unfold mxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact mxm_apply_complete ports _
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
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | exact mxm_apply_complete ports _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem sxm_apply_ok_complete (ports : Ports) (xs : (List sxm_staged)) (bad : (List Nat)) : Inv Complete (sxm_apply_ok xs bad : SailM Unit) := by
  induction xs generalizing bad
  case nil =>
    unfold sxm_apply_ok
    repeat' first
      | exact ih _
      | assumption
      | exact pure_inv _ _
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
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
      | exact throw_complete _
      | exact read_inv _ _
      | exact write_complete _ _
      | exact assert_complete _ _
      | reduce_resource_beta
      | apply forIn_inv
      | apply bind_inv
      | cases_resource_pair
      | split
      | intro
theorem unit_state_flush_complete (ports : Ports) (c : Nat) : Inv Complete (unit_state_flush c : SailM Unit) := by
  unfold unit_state_flush
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact mxm_apply_ok_complete ports _ _
    | exact report_complete ports _ _ _
    | exact sxm_apply_ok_complete ports _ _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
theorem tick_complete (ports : Ports) (c : Nat) : Inv Complete (tick c : SailM Unit) := by
  unfold tick
  repeat' first
    | assumption
    | exact pure_inv _ _
    | exact throw_complete _
    | exact read_inv _ _
    | exact write_complete _ _
    | exact assert_complete _ _
    | exact barrier_retire_complete ports _
    | exact commit_due_complete ports _
    | exact control_complete ports _
    | exact enqueue_complete ports _ _
    | exact execute_due_complete ports _
    | exact local_fetch_complete_complete ports _
    | exact scrub_run_complete ports _
    | exact stream_fetch_complete_complete ports _
    | exact tick_queue_complete ports _ _
    | exact unit_state_flush_complete ports _
    | reduce_resource_beta
    | apply forIn_inv
    | apply bind_inv
    | cases_resource_pair
    | split
    | intro
end ResourceCompleteProofs
