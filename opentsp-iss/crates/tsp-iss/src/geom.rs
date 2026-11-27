//! What the machine takes from its profile (`core/geometry.sail`), and the architectural maxima
//!.
use ots_profile::Profile;

/// Lanes in the widest vector: 32 superlanes of 16 lanes (`core/geometry.sail`).
pub const MAX_LANES: usize = 512;
/// Lanes of the widest profile that has at most 20 superlanes (`ref`, `mini`): a machine of this width keeps its per-cycle
/// records 320 bytes wide (`Machine<NARROW_LANES>`).
pub const NARROW_LANES: usize = 320;
/// SRAM slices (`sram.slices`): 44 per hemisphere.
pub const SLICES: usize = 88;
/// MEM slices per hemisphere at most.
pub const MEM_SLICES_MAX: u64 = 44;
/// Data queues.
pub const DATA_QUEUES: usize = 144;
/// Queue slots: the data queues and the control channel (index 144).
pub const QUEUES: usize = 145;
/// The control channel as a queue id and as an index.
pub const CONTROL_QUEUE: u16 = 0xffff;
pub const CONTROL_INDEX: usize = 144;
/// Chip-to-chip links per chip at most (the link number of Send, Receive and Transmit is 7 bits).
pub const MAX_LINKS: usize = 128;
/// Chips at most (`chip_id` is 9 bits).
pub const MAX_CHIPS: usize = 512;
/// Streams per row.
pub const STREAMS_PER_ROW: usize = 32;

/// The index of a queue id: 0..143, or 144 for the control channel.
pub fn queue_index(q: u16) -> Option<usize> {
    match q {
        0..=143 => Some(q as usize),
        CONTROL_QUEUE => Some(CONTROL_INDEX),
        _ => None,
    }
}

/// The geometry and sizes of one configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geom {
    pub superlanes: u64,
    /// `16 * superlanes`.
    pub vlanes: u64,
    pub words_per_slice: u64,
    pub bank_bit: u64,
    /// `sram.addr_bits`: the width of an SRAM address (13 to 16); a slice has at most `2^addr_bits` words.
    pub addr_bits: u64,
    /// `icu.iq_depth`, in bytes.
    pub iq_depth: u64,
    /// `iq_depth / 4`: the units an IQ holds.
    pub iq_units: usize,
    /// `geometry.mem_slices_per_hemisphere` (44 when the configuration does not say).
    pub mem_slices: u64,
    /// MEM quads per hemisphere: `mem_slices / 4`.
    pub quads: u64,
    pub links: u64,
    pub link_latency: u64,
    pub link_latency_min: u64,
    pub link_latency_max: u64,
    /// `c2c.cycles_per_vector`: a link accepts one vector per that many cycles.
    pub cycles_per_vector: u64,
    /// `c2c.rx_depth`: the vectors a link's receive FIFO holds (the harness's bound).
    pub rx_depth: u64,
    /// `mxm.rows`, when the configuration has the key.
    pub mxm_rows: Option<u64>,
    /// `mxm.planes_per_hemisphere`: MXM planes per hemisphere (1 or 2).
    pub mxm_planes: u64,
    /// `mxm.acc_depth`: accumulators per plane.
    pub mxm_acc_depth: u64,
    /// `mxm.f16_rows`: rows of an f16 matrix.
    pub mxm_f16_rows: u64,
    /// `mxm.fp8_rate`: FP8 ABCs a plane accepts in one cycle (1 or 2).
    pub mxm_fp8_rate: u64,
    /// `icu.fetch_latency`: cycles from an Ifetch's execute cycle to its text entering the IQ.
    pub fetch_latency: u64,
    /// `icu.fetch_staging_depth`: the payloads a queue's fetch staging FIFO holds at most.
    pub fetch_staging_depth: u64,
    /// The host interface (`host.*`, `core/csr.sail`): cycles the PCIe interface needs per vector, the ingress FIFO's depth,
    /// the doorbell count, the chip that has the host interface and whether CSRs of other chips can be reached.
    pub host_cycles_per_vector: u64,
    pub host_ingress_depth: u64,
    pub host_doorbells: u64,
    pub host_remote_csr: bool,
    /// `ras.log_depth`: the entries the error log of the error CSRs holds (`core/csr.sail`).
    pub ras_log_depth: u64,
    /// `ras.diag_depth`: the entries the diagnostic log of the CSR space holds (`core/csr.sail`).
    pub ras_diag_depth: u64,
    /// `scrub.period`: a patrol scrub cycle every this many cycles, 0 = off (`core/csr.sail`).
    pub scrub_period: u64,
    /// `timing.sxm_map_ready`: cycles from a LoadMap's execute cycle to its map being ready for a Permute.
    pub sxm_map_ready: u64,
    /// `timing.mxm_abc.occupancy_f16`: cycles an f16 ABC holds its plane.
    pub mxm_occupancy_f16: u64,
    /// The extension wide_acc is declared (`wide_acc` in `extensions`): ABCs keep their exact products for ACC x128.
    pub ext_wide_acc: bool,
    /// `timing.c2c.rtdeskew_floor`: the shortest hold of a RuntimeDeskew.
    pub rtdeskew_floor: u64,
    /// `icu.fetch_bytes_per_cycle / 4`: units entering an IQ per cycle (`iq_rate`).
    pub iq_rate: usize,
    /// `barrier.release_latency`.
    pub barrier_release_latency: u64,
    /// `c2c.hac_period`.
    pub hac_period: u64,
    /// The `layout` keys (`core/layout.sail`).
    pub layout: Layout,
}

/// Where the units sit on the stream rows (`layout.*`, `core/layout.sail`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub vxm_gap: i64,
    pub mem_up_gap_base: i64,
    pub mem_down_gap_base: i64,
    pub mxm_in_gap: i64,
    pub mxm_acc_gap: i64,
    pub sxm_dst_gap: i64,
    pub sxm_turn_in_gap: i64,
    pub sxm_turn_out_gap: i64,
    pub c2c_tx_gap: i64,
    pub c2c_rx_gap: i64,
}

impl Geom {
    /// The values of a profile. The profile must have passed [`Profile::validate`]: a key that
    /// the schema requires but is missing reads as 0.
    pub fn new(p: &Profile) -> Geom {
        let u = |k: &str| p.uint(k).unwrap_or(0);
        let mem_slices = p.uint("geometry.mem_slices_per_hemisphere").unwrap_or(MEM_SLICES_MAX);
        Geom {
            superlanes: p.superlanes(),
            vlanes: p.vlanes(),
            words_per_slice: p.words_per_slice(),
            bank_bit: p.bank_bit(),
            addr_bits: p.addr_bits(),
            iq_depth: p.iq_depth(),
            iq_units: (p.iq_depth() / 4) as usize,
            mem_slices,
            quads: mem_slices / 4,
            links: u("c2c.links"),
            link_latency: u("c2c.link_latency"),
            link_latency_min: u("c2c.link_latency_min"),
            link_latency_max: u("c2c.link_latency_max"),
            cycles_per_vector: u("c2c.cycles_per_vector"),
            rx_depth: u("c2c.rx_depth"),
            mxm_rows: p.uint("mxm.rows"),
            mxm_planes: u("mxm.planes_per_hemisphere"),
            mxm_acc_depth: u("mxm.acc_depth"),
            mxm_f16_rows: u("mxm.f16_rows"),
            mxm_fp8_rate: u("mxm.fp8_rate"),
            fetch_latency: u("icu.fetch_latency"),
            fetch_staging_depth: u("icu.fetch_staging_depth"),
            host_cycles_per_vector: u("host.cycles_per_vector"),
            host_ingress_depth: u("host.ingress_depth"),
            host_doorbells: u("host.doorbells"),
            host_remote_csr: p.get("host.remote_csr").and_then(|v| v.as_bool()).unwrap_or(false),
            ras_log_depth: u("ras.log_depth"),
            ras_diag_depth: u("ras.diag_depth"),
            scrub_period: u("scrub.period"),
            sxm_map_ready: u("timing.sxm_map_ready"),
            mxm_occupancy_f16: u("timing.mxm_abc.occupancy_f16"),
            ext_wide_acc: p.has_extension("wide_acc"),
            rtdeskew_floor: u("timing.c2c.rtdeskew_floor"),
            iq_rate: (u("icu.fetch_bytes_per_cycle") / 4) as usize,
            barrier_release_latency: u("barrier.release_latency"),
            hac_period: u("c2c.hac_period"),
            layout: Layout {
                vxm_gap: u("layout.vxm_gap") as i64,
                mem_up_gap_base: u("layout.mem_up_gap_base") as i64,
                mem_down_gap_base: u("layout.mem_down_gap_base") as i64,
                mxm_in_gap: u("layout.mxm_in_gap") as i64,
                mxm_acc_gap: u("layout.mxm_acc_gap") as i64,
                sxm_dst_gap: u("layout.sxm_dst_gap") as i64,
                sxm_turn_in_gap: u("layout.sxm_turn_in_gap") as i64,
                sxm_turn_out_gap: u("layout.sxm_turn_out_gap") as i64,
                c2c_tx_gap: u("layout.c2c_tx_gap") as i64,
                c2c_rx_gap: u("layout.c2c_rx_gap") as i64,
            },
        }
    }

    /// Does the MEM slice `slice` (0..87) exist? A slice is absent when its index within its
    /// hemisphere is not below `mem_slices`.
    pub fn slice_present(&self, slice: u64) -> bool {
        slice < SLICES as u64 && slice % MEM_SLICES_MAX < self.mem_slices
    }

    /// The harness holds vectors of at most [`MAX_LANES`] lanes: `Err(message)` otherwise
    /// (`tsp_sim.c` `main`).
    pub fn check_lanes(&self) -> Result<(), String> {
        Self::check_vector_width(self.vlanes)
    }

    pub(crate) fn check_vector_width(lanes: u64) -> Result<(), String> {
        if lanes < 1 || lanes > MAX_LANES as u64 {
            return Err(format!("geometry.superlanes: {lanes} lanes, this harness holds at most {MAX_LANES}"));
        }
        Ok(())
    }

    /// `c2c.link_latency_min <= c2c.link_latency <= c2c.link_latency_max`, else the message.
    pub fn check_link_latency(&self) -> Result<(), String> {
        if self.link_latency_min > self.link_latency || self.link_latency > self.link_latency_max {
            return Err("c2c.link_latency: must be in [c2c.link_latency_min, c2c.link_latency_max]".into());
        }
        Ok(())
    }
}
