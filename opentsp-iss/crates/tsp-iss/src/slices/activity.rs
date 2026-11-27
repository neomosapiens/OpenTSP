//! Activity counters: what each unit class did during a run.
//!
//! Counting is opt-in. A chip counts only while [`Activity::enabled`] is set: `tsp-iss run` sets it for `--stats` and
//! `--energy` (the two outputs that read the counters) and leaves it off otherwise, `tsp-iss bench`, the serving platform and
//! the binding's multi-chip system leave it off, and a library user that builds a [`Machine`](crate::Machine) directly gets the
//! process default ([`set_default_enabled`]), which is off: whoever reads the counters asks for them (`Activity::set_enabled`
//! on the machine, or the default before the machines are built). With counting off each instruction costs one flag test; what
//! the counters mean when on is unchanged.
//!
//! The counters are plain integer additions made where an instruction has finished executing
//! ([`Machine::execute`](crate::Machine::execute) calls the `note_*` method of the unit that ran it), not per element: a
//! MEM reference adds the bytes of its active tiles, a VXM instruction its active lanes times the operations per lane, an
//! ABC the installed rows times the vector length. They never feed back into the model: traces, cycle counts and
//! diagnostics are the same with and without them (`tests/counters.rs`).
//!
//! What counts. An instruction counts when it issues (`Exec::Issued`); one that raises a diagnostic counts nowhere here (the
//! diagnostic is in `diagnostics_by_code`), including the Gather, Scatter and SelectS references whose good tiles or lanes
//! still took effect. Reads of the stream register file are not counted, only its writes (one per vector scheduled for
//! commit, on the row of its stream, including the zero superlanes of a vector cut by SetVL). A replayed schedule executes the
//! same instructions, so the counters are the same with replay on or off.
//!
//! Units. `*_bytes` are bytes of active tiles (`vlanes / superlanes` bytes per tile of an active superlane), `*_vectors` are
//! whole stream vectors, `macs` are multiply-accumulates (a dot-product row over the vector length is `vlanes` of them), and
//! `active_cycles` are the distinct execute cycles in which a unit class executed at least one instruction (the units are
//! otherwise idle; the energy model charges idle cycles separately).
use super::Issue;
use crate::geom::{MAX_LINKS, SLICES};
use crate::Machine;
use ots_isa::{EntryId, Insn, ENTRIES};
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use tsp_numerics::EType;

/// Whether a chip built from now on counts (see the module documentation).
static DEFAULT_ENABLED: AtomicBool = AtomicBool::new(false);

/// Set whether the chips built from now on count their activity (`tsp-iss run` and `bench` call it before they build any).
pub fn set_default_enabled(on: bool) {
    DEFAULT_ENABLED.store(on, Ordering::Relaxed);
}

/// The unit classes the cycle accounting distinguishes, in report order.
pub const CLASSES: [&str; 6] = ["MEM", "VXM", "MXM", "SXM", "C2C", "CU"];
const MEM: usize = 0;
const VXM: usize = 1;
const MXM: usize = 2;
const SXM: usize = 3;
const C2C: usize = 4;
const CU: usize = 5;

/// The element types for the VXM table: the base set in code order (`EType::ALL`), then the extensions (`EType::index`).
const NTYPES: usize = 14;

/// Write gaps the per-gap stream-write table holds per row (a gap is below 88 in every profile).
const SRF_GAPS: usize = 128;

/// The MXM operand types as the counters index them.
pub const MXM_TYPES: [&str; 4] = ["i8", "u8", "f16", "fp8"];

/// What one MEM slice did: SRAM word accesses (the references that issued; IfetchM counts its two words) and the bytes of
/// the active tiles they touched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemSlice {
    pub reads: u64,
    pub writes: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
    /// SRAM word writes as the trace shows them (`T sram`): one per Write, one per good tile of a Scatter.
    pub write_events: u64,
}

/// The matrix unit: instruction counts, rows moved and multiply-accumulates, per operand type where the type matters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MxmActivity {
    /// LW instructions and the weight-buffer rows (stream vectors) they loaded: 16 each.
    pub lw: u64,
    pub lw_rows: u64,
    /// IWS instructions and the weight rows they installed from streams: 16 each.
    pub iws: u64,
    pub iws_rows: u64,
    /// IW instructions and the buffered rows they installed.
    pub iw: u64,
    pub iw_rows: u64,
    /// ABC and ABC8 instructions per operand type ([`MXM_TYPES`]) and the multiply-accumulates they performed: the installed
    /// rows (f16: row pairs) times the vector length (an ABC always runs over all `vlanes` lanes: SetVL does not cut it).
    pub abc: [u64; 4],
    pub macs: [u64; 4],
    /// ABC instructions per plane.
    pub abc_by_plane: [u64; 4],
    /// ACC instructions and the accumulator lanes they produced (`mxm.rows` each).
    /// An ACC x128 (`wide_acc`) still counts one unit per row: priced as a 32-bit lane in `energy.yaml` ( stated limit).
    pub acc: u64,
    pub acc_lanes: u64,
}

/// What the chip is built of, as far as the energy model scales by it: the lanes of a vector, the MEM slices present, the matrix
/// planes and their rows, the chip-to-chip links.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Structure {
    pub vlanes: u64,
    pub mem_slices: u64,
    pub mxm_planes: u64,
    pub mxm_rows: u64,
    pub c2c_links: u64,
}

impl Structure {
    /// The size of each unit class in the units the energy table scales it by: MEM slices times lanes, VXM lanes, matrix
    /// planes times rows times lanes, SXM lanes, C2C links, and 1 for the control unit (one per chip). Indexed as [`CLASSES`].
    pub fn class_sizes(&self) -> [f64; 6] {
        [
            (self.mem_slices * self.vlanes) as f64,
            self.vlanes as f64,
            (self.mxm_planes * self.mxm_rows * self.vlanes) as f64,
            self.vlanes as f64,
            self.c2c_links as f64,
            1.0,
        ]
    }
}

/// One chip's activity counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Activity {
    enabled: bool,
    structure: Structure,
    vector_bytes: u64,
    tile_bytes: u64,
    mxm_rows: u64,
    /// Instructions that issued, per catalogue entry (`ENTRIES` order).
    pub issued: Vec<u64>,
    /// Per MEM slice (global slice 0..87).
    pub mem: Vec<MemSlice>,
    /// Stream-register writes (vectors scheduled for commit), per row: east, west.
    pub srf_writes: [u64; 2],
    /// The same per row and write gap (the position of the writing unit along the row): `row * SRF_GAPS + gap`.
    pub srf_gap: Vec<u64>,
    /// VXM element operations per catalogue entry and element type (`EType` code order).
    pub vxm: Vec<[u64; NTYPES]>,
    pub mxm: MxmActivity,
    /// SXM vectors read and written.
    pub sxm_in_vectors: u64,
    pub sxm_out_vectors: u64,
    /// C2C vectors sent and received and sync packets sent, per link.
    pub c2c_sent: [u64; MAX_LINKS],
    pub c2c_received: [u64; MAX_LINKS],
    pub c2c_sync: [u64; MAX_LINKS],
    /// Host vectors: PciReceive (host to chip), PciSend (chip to host), and HostNotify interrupts.
    pub host_in: u64,
    pub host_out: u64,
    pub host_notify: u64,
    /// Distinct execute cycles per unit class ([`CLASSES`]).
    pub active_cycles: [u64; 6],
    /// Each class's last actual execute cycle; absence is separate from every u64 timestamp.
    last: [Option<u64>; 6],
}

impl Activity {
    /// Zeroed counters for a chip with `vlanes` lanes in `superlanes` superlanes and `mxm_rows` matrix rows, with the other
    /// structure of the `ref` profile (88 MEM slices, 4 matrix planes, 16 C2C links).
    pub fn new(vlanes: u64, superlanes: u64, mxm_rows: u64) -> Activity {
        Activity::with_structure(
            vlanes,
            superlanes,
            Structure { vlanes, mem_slices: SLICES as u64, mxm_planes: 4, mxm_rows, c2c_links: 16 },
        )
    }

    /// Zeroed counters for a chip of the given structure; counting is on or off as [`set_default_enabled`] says.
    pub fn with_structure(vlanes: u64, superlanes: u64, structure: Structure) -> Activity {
        let mxm_rows = structure.mxm_rows;
        Activity {
            enabled: DEFAULT_ENABLED.load(Ordering::Relaxed),
            structure,
            vector_bytes: vlanes,
            tile_bytes: vlanes / superlanes.max(1),
            mxm_rows,
            issued: vec![0; ENTRIES.len()],
            mem: vec![MemSlice::default(); SLICES],
            srf_writes: [0; 2],
            srf_gap: vec![0; 2 * SRF_GAPS],
            vxm: vec![[0; NTYPES]; ENTRIES.len()],
            mxm: MxmActivity::default(),
            sxm_in_vectors: 0,
            sxm_out_vectors: 0,
            c2c_sent: [0; MAX_LINKS],
            c2c_received: [0; MAX_LINKS],
            c2c_sync: [0; MAX_LINKS],
            host_in: 0,
            host_out: 0,
            host_notify: 0,
            active_cycles: [0; 6],
            last: [None; 6],
        }
    }

    /// A run restarts at cycle 0 (`Machine::reset`): the counts stay, the cycle bookkeeping starts over.
    pub fn restart(&mut self) {
        self.last = [None; 6];
    }

    /// Is this chip counting?
    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Count (or stop counting) from now on; the counts so far stay.
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    /// The structure the energy model scales by.
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    /// Bytes of one stream vector.
    pub fn vector_bytes(&self) -> u64 {
        self.vector_bytes
    }

    /// Lanes of the vector length and rows of the matrix unit (for the energy model's peaks).
    pub fn mxm_rows(&self) -> u64 {
        self.mxm_rows
    }

    /// Add `o`'s counts to these (the totals over chips). `active_cycles` add as well: the sum is chip-cycles.
    pub fn add(&mut self, o: &Activity) {
        fn zip(a: &mut [u64], b: &[u64]) {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
        zip(&mut self.issued, &o.issued);
        for (a, b) in self.mem.iter_mut().zip(&o.mem) {
            a.reads += b.reads;
            a.writes += b.writes;
            a.read_bytes += b.read_bytes;
            a.write_bytes += b.write_bytes;
            a.write_events += b.write_events;
        }
        zip(&mut self.srf_writes, &o.srf_writes);
        zip(&mut self.srf_gap, &o.srf_gap);
        for (a, b) in self.vxm.iter_mut().zip(&o.vxm) {
            zip(a, b);
        }
        let (m, n) = (&mut self.mxm, &o.mxm);
        m.lw += n.lw;
        m.lw_rows += n.lw_rows;
        m.iws += n.iws;
        m.iws_rows += n.iws_rows;
        m.iw += n.iw;
        m.iw_rows += n.iw_rows;
        zip(&mut m.abc, &n.abc);
        zip(&mut m.macs, &n.macs);
        zip(&mut m.abc_by_plane, &n.abc_by_plane);
        m.acc += n.acc;
        m.acc_lanes += n.acc_lanes;
        self.sxm_in_vectors += o.sxm_in_vectors;
        self.sxm_out_vectors += o.sxm_out_vectors;
        zip(&mut self.c2c_sent, &o.c2c_sent);
        zip(&mut self.c2c_received, &o.c2c_received);
        zip(&mut self.c2c_sync, &o.c2c_sync);
        self.host_in += o.host_in;
        self.host_out += o.host_out;
        self.host_notify += o.host_notify;
        zip(&mut self.active_cycles, &o.active_cycles);
    }

    /// `n` vectors written to the stream register file on row `west` at write gap `gap`.
    #[inline(always)]
    pub fn srf_write(&mut self, west: bool, gap: u8, n: u64) {
        if self.enabled {
            self.srf_write_counted(west, gap, n);
        }
    }

    #[inline(never)]
    fn srf_write_counted(&mut self, west: bool, gap: u8, n: u64) {
        self.srf_writes[west as usize] += n;
        self.srf_gap[west as usize * SRF_GAPS + (gap as usize).min(SRF_GAPS - 1)] += n;
    }

    /// Count class `class` as active in `cycle` once per distinct cycle (executions arrive in cycle order).
    #[inline]
    fn touch(&mut self, class: usize, cycle: u64) {
        if self.last[class] != Some(cycle) {
            self.last[class] = Some(cycle);
            self.active_cycles[class] += 1;
        }
    }

    /// The MEM counters summed over the slices.
    pub fn mem_total(&self) -> MemSlice {
        let mut t = MemSlice::default();
        for s in &self.mem {
            t.reads += s.reads;
            t.writes += s.writes;
            t.read_bytes += s.read_bytes;
            t.write_bytes += s.write_bytes;
            t.write_events += s.write_events;
        }
        t
    }

    pub fn vxm_ops_total(&self) -> u64 {
        self.vxm.iter().flatten().sum()
    }

    pub fn macs_total(&self) -> u64 {
        self.mxm.macs.iter().sum()
    }

    /// Element operations of VXM entry `e` summed over the types.
    pub fn vxm_ops_of(&self, e: usize) -> u64 {
        self.vxm[e].iter().sum()
    }

    /// Element operations of VXM type code `t` summed over the entries.
    pub fn vxm_ops_of_type(&self, t: usize) -> u64 {
        self.vxm.iter().map(|r| r[t]).sum()
    }

    /// The `activity` object of the `--stats` document.
    pub fn to_json(&self) -> Value {
        let mem = self.mem_total();
        let by_slice: Vec<Value> = self
            .mem
            .iter()
            .enumerate()
            .filter(|(_, s)| **s != MemSlice::default())
            .map(|(i, s)| {
                json!({"slice": i, "reads": s.reads, "writes": s.writes,
                       "read_bytes": s.read_bytes, "write_bytes": s.write_bytes, "write_events": s.write_events})
            })
            .collect();
        let mut by_op = Map::new();
        for (i, e) in ENTRIES.iter().enumerate() {
            let n = self.vxm_ops_of(i);
            if n > 0 {
                by_op.insert(e.name.to_string(), json!(n));
            }
        }
        let mut by_type = Map::new();
        for ty in EType::ALL.iter().chain(EType::EXTENSIONS.iter()) {
            let t = ty.index();
            let n = self.vxm_ops_of_type(t);
            if n > 0 {
                by_type.insert(ty.name().to_string(), json!(n));
            }
        }
        let mut by_class: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
        let mut by_class_type: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        for (i, e) in ENTRIES.iter().enumerate() {
            if e.slice == "VXM" && self.vxm_ops_of(i) > 0 {
                *by_class.entry(vxm_class(e.id)).or_insert(0) += self.vxm_ops_of(i);
                for ty in EType::ALL.iter().chain(EType::EXTENSIONS.iter()) {
                    let t = ty.index();
                    if self.vxm[i][t] > 0 {
                        *by_class_type.entry(format!("{}.{}", vxm_class(e.id), ty.name())).or_insert(0) +=
                            self.vxm[i][t];
                    }
                }
            }
        }
        let mut sxm = Map::new();
        let mut issued = Map::new();
        for (i, e) in ENTRIES.iter().enumerate() {
            if self.issued[i] > 0 {
                issued.insert(e.name.to_string(), json!(self.issued[i]));
                if e.slice == "SXM" {
                    sxm.insert(e.name.to_string(), json!(self.issued[i]));
                }
            }
        }
        let abc: Map<String, Value> = MXM_TYPES
            .iter()
            .enumerate()
            .map(|(k, t)| (t.to_string(), json!({"abc": self.mxm.abc[k], "macs": self.mxm.macs[k]})))
            .collect();
        let links: Vec<Value> = (0..MAX_LINKS)
            .filter(|&l| self.c2c_sent[l] + self.c2c_received[l] + self.c2c_sync[l] > 0)
            .map(|l| {
                json!({"link": l, "sent_vectors": self.c2c_sent[l], "received_vectors": self.c2c_received[l],
                       "sync_packets": self.c2c_sync[l]})
            })
            .collect();
        let by_gap: Vec<Value> = (0..2 * SRF_GAPS)
            .filter(|&k| self.srf_gap[k] > 0)
            .map(|k| json!({"row": if k < SRF_GAPS { "E" } else { "W" }, "gap": k % SRF_GAPS, "writes": self.srf_gap[k]}))
            .collect();
        let (sent, recv): (u64, u64) = (self.c2c_sent.iter().sum(), self.c2c_received.iter().sum());
        let active: Map<String, Value> =
            CLASSES.iter().zip(self.active_cycles).map(|(c, n)| (c.to_string(), json!(n))).collect();
        json!({
            "note": "counts of instructions that issued; see slices/activity.rs for what each counter means",
            "vector_bytes": self.vector_bytes,
            "mem": {
                "reads": mem.reads, "writes": mem.writes,
                "read_bytes": mem.read_bytes, "write_bytes": mem.write_bytes, "write_events": mem.write_events,
                "per_slice": by_slice,
            },
            "srf": {
                "writes_east": self.srf_writes[0], "writes_west": self.srf_writes[1],
                "write_bytes": (self.srf_writes[0] + self.srf_writes[1]) * self.vector_bytes,
                "by_gap": by_gap,
            },
            "vxm": {"element_ops": self.vxm_ops_total(), "by_op": by_op, "by_class": by_class, "by_class_type": by_class_type, "by_type": by_type},
            "mxm": {
                "lw": self.mxm.lw, "lw_rows": self.mxm.lw_rows,
                "iws": self.mxm.iws, "iws_rows": self.mxm.iws_rows,
                "iw": self.mxm.iw, "iw_rows": self.mxm.iw_rows,
                "abc_by_type": abc,
                "abc_by_plane": self.mxm.abc_by_plane,
                "macs": self.macs_total(),
                "acc": self.mxm.acc, "acc_lanes": self.mxm.acc_lanes,
            },
            "sxm": {"ops": sxm, "in_vectors": self.sxm_in_vectors, "out_vectors": self.sxm_out_vectors},
            "c2c": {
                "sent_vectors": sent, "received_vectors": recv, "sync_packets": self.c2c_sync.iter().sum::<u64>(),
                "sent_bytes": sent * self.vector_bytes, "received_bytes": recv * self.vector_bytes,
                "per_link": links,
            },
            "host": {
                "in_vectors": self.host_in, "out_vectors": self.host_out,
                "in_bytes": self.host_in * self.vector_bytes, "out_bytes": self.host_out * self.vector_bytes,
                "notify": self.host_notify,
            },
            "active_cycles": active,
            "issued": issued,
        })
    }
}

fn ety(code: u32) -> EType {
    EType::from_code(code as u8).unwrap_or(EType::I8)
}

/// The element type and the operations per lane of a VXM instruction: an SG4 operand group holds `4 / width` independent
/// operations per lane.
fn vxm_shape(i: &Insn) -> (EType, u64) {
    let grouped = |t: u32, sg4: u32| {
        let t = ety(t);
        let n = if sg4 == 1 {
            match t.width() {
                1 => 4,
                2 => 2,
                _ => 1,
            }
        } else {
            1
        };
        (t, n)
    };
    match *i {
        Insn::Add { r#type, sg4, .. }
        | Insn::Add_sat { r#type, sg4, .. }
        | Insn::Add_mod { r#type, sg4, .. }
        | Insn::Sub { r#type, sg4, .. }
        | Insn::Sub_sat { r#type, sg4, .. }
        | Insn::Sub_mod { r#type, sg4, .. }
        | Insn::Mul { r#type, sg4, .. }
        | Insn::Mul_sat { r#type, sg4, .. }
        | Insn::Mul_mod { r#type, sg4, .. }
        | Insn::Max { r#type, sg4, .. }
        | Insn::Min { r#type, sg4, .. }
        | Insn::And { r#type, sg4, .. }
        | Insn::Or { r#type, sg4, .. }
        | Insn::Xor { r#type, sg4, .. }
        | Insn::Not { r#type, sg4, .. }
        | Insn::Neg { r#type, sg4, .. }
        | Insn::Abs { r#type, sg4, .. }
        | Insn::ReLU { r#type, sg4, .. }
        | Insn::TanH { r#type, sg4, .. }
        | Insn::Exp { r#type, sg4, .. }
        | Insn::RSqrt { r#type, sg4, .. }
        | Insn::Log { r#type, sg4, .. }
        | Insn::Mask { r#type, sg4, .. }
        | Insn::Select { r#type, sg4, .. }
        | Insn::Cmp { r#type, sg4, .. } => grouped(r#type, sg4),
        Insn::Convert { r#type, .. } | Insn::Reduce { r#type, .. } => (ety(r#type), 1),
        _ => {
            // the eight fixed conversions: the type is the source's
            let t = match i.id() {
                EntryId::Conv_I8_F32 | EntryId::Conv_I8_F16 | EntryId::Conv_I8_I32 => EType::I8,
                EntryId::Conv_F32_I8 | EntryId::Conv_F32_F16 => EType::F32,
                EntryId::Conv_F16_I8 | EntryId::Conv_F16_F32 => EType::F16,
                _ => EType::I32,
            };
            (t, 1)
        }
    }
}

/// The energy class of a VXM entry (the keys of `events.vxm_op_pj` in `perf/energy.yaml`).
pub fn vxm_class(id: EntryId) -> &'static str {
    use EntryId as E;
    match id {
        E::Add | E::Add_sat | E::Add_mod | E::Sub | E::Sub_sat | E::Sub_mod | E::Neg | E::Abs => "add_sub",
        E::Mul | E::Mul_sat | E::Mul_mod => "mul",
        E::Max | E::Min | E::ReLU => "minmax",
        E::And | E::Or | E::Xor | E::Not | E::Mask => "logic",
        E::Cmp | E::Select => "compare_select",
        E::Reduce => "reduce",
        E::TanH | E::Exp | E::RSqrt | E::Log => "transcendental",
        _ => "convert",
    }
}

impl<const L: usize> Machine<L> {
    /// Active tiles of a MEM reference of thread `thread` on queue `q` (`core/sram.sail` `tile_active` summed over the tiles).
    fn tiles_on(&self, q: usize, thread: u32) -> u64 {
        let sp = self.geom.superlanes;
        if q > 87 {
            return sp;
        }
        let (vl, split) = (self.masks.queue_vl[q] as u64, self.masks.mem_thread_split[q] as u64);
        let n = if thread == 0 { vl.min(split) } else { vl.saturating_sub(split) };
        n.min(sp)
    }

    /// A MEM-slice instruction that issued.
    #[inline(never)]
    pub(super) fn note_mem(&mut self, i: &Insn, ctx: &Issue) {
        let (reads, writes, tiles) = match *i {
            Insn::Read { thread, .. } | Insn::Gather { thread, .. } => (true, false, self.tiles_on(ctx.queue, thread)),
            Insn::Write { thread, .. } => (false, true, self.tiles_on(ctx.queue, thread)),
            Insn::Scatter { thread, .. } => (true, true, self.tiles_on(ctx.queue, thread)),
            _ => (false, false, 0),
        };
        let a = &mut self.activity;
        a.issued[i.id().index()] += 1;
        if reads || writes {
            let bytes = tiles * a.tile_bytes;
            // a MEM slice's queue is its slice number
            if let Some(s) = a.mem.get_mut(ctx.queue) {
                if reads {
                    s.reads += 1;
                    s.read_bytes += bytes;
                }
                if writes {
                    s.writes += 1;
                    s.write_bytes += bytes;
                    s.write_events += if reads { tiles } else { 1 };
                }
            }
            a.touch(MEM, ctx.exec_cycle());
        }
    }

    /// A control-unit instruction that issued. Only the ones that do work in their own cycle count as activity.
    #[inline(never)]
    pub(super) fn note_cu(&mut self, i: &Insn, ctx: &Issue) {
        let id = i.id();
        let a = &mut self.activity;
        a.issued[id.index()] += 1;
        if matches!(id, EntryId::Ifetch | EntryId::IfetchM | EntryId::Config | EntryId::SetVL) {
            a.touch(CU, ctx.exec_cycle());
        }
        if id == EntryId::IfetchM && ctx.queue <= 87 {
            // IfetchM reads two words of the slice's SRAM
            let vb = a.vector_bytes;
            let s = &mut a.mem[ctx.queue];
            s.reads += 2;
            s.read_bytes += 2 * vb;
            a.touch(MEM, ctx.exec_cycle());
        }
    }

    /// A VXM instruction that issued.
    #[inline(never)]
    pub(super) fn note_vxm(&mut self, i: &Insn, ctx: &Issue) {
        let (t, per_lane) = vxm_shape(i);
        let sp = self.geom.superlanes;
        let active = if ctx.queue <= 143 { (self.masks.queue_vl[ctx.queue] as u64).min(sp) } else { sp };
        let lanes = active * (self.geom.vlanes / sp.max(1));
        let id = i.id().index();
        let a = &mut self.activity;
        a.issued[id] += 1;
        a.vxm[id][t.index()] += lanes * per_lane;
        a.touch(VXM, ctx.exec_cycle());
    }

    /// An MXM instruction that issued. The plane's state is as at the start of the cycle (changes are staged), so its rows
    /// are the ones the instruction used.
    #[inline(never)]
    pub(super) fn note_mxm(&mut self, i: &Insn, ctx: &Issue) {
        let pl = crate::layout::mxm_plane(&self.geom, ctx.queue);
        let vl = self.geom.vlanes;
        let rows = self.mxm.planes[pl].rows as u64;
        let buf = self.mxm.planes[pl].buf_rows as u64;
        let a = &mut self.activity;
        a.issued[i.id().index()] += 1;
        let mrows = a.mxm_rows;
        let m = &mut a.mxm;
        let abc = |m: &mut MxmActivity, k: usize| {
            m.abc[k] += 1;
            m.macs[k] += rows * vl;
            m.abc_by_plane[pl] += 1;
        };
        match *i {
            Insn::LW { .. } => {
                m.lw += 1;
                m.lw_rows += 16;
            }
            Insn::IWS { .. } => {
                m.iws += 1;
                m.iws_rows += 16;
            }
            Insn::IW { .. } => {
                m.iw += 1;
                m.iw_rows += buf;
            }
            Insn::ABC { r#type, .. } => abc(m, mxm_type(r#type)),
            Insn::ABC8 { .. } => abc(m, 3),
            Insn::ACC { .. } => {
                // One counter unit per matrix row; ACC x128 is not scaled (priced as a 32-bit lane; ).
                m.acc += 1;
                m.acc_lanes += mrows;
            }
            _ => {}
        }
        a.touch(MXM, ctx.exec_cycle());
    }

    /// An SXM instruction that issued.
    #[inline(never)]
    pub(super) fn note_sxm(&mut self, i: &Insn, ctx: &Issue) {
        let (vin, vout) = match *i {
            Insn::Shift { .. } | Insn::Permute { .. } | Insn::Distribute { .. } => (1, 1),
            Insn::Rotate { n, .. } => (1, if n == 3 { 9 } else { 16 }),
            Insn::Transpose { .. } => (16, 16),
            Insn::LoadMap { .. } => (1, 0),
            Insn::LoadMapWide { .. } => (2, 0),
            Insn::SelectS { .. } => (4, 1),
            _ => (0, 0),
        };
        let a = &mut self.activity;
        a.issued[i.id().index()] += 1;
        a.sxm_in_vectors += vin;
        a.sxm_out_vectors += vout;
        a.touch(SXM, ctx.exec_cycle());
    }

    /// A C2C-unit instruction that issued.
    #[inline(never)]
    pub(super) fn note_c2c(&mut self, i: &Insn, ctx: &Issue) {
        let a = &mut self.activity;
        a.issued[i.id().index()] += 1;
        match *i {
            Insn::Send { link, .. } => a.c2c_sent[link as usize % MAX_LINKS] += 1,
            Insn::Receive { link, .. } => a.c2c_received[link as usize % MAX_LINKS] += 1,
            Insn::Transmit { link, .. } => a.c2c_sync[link as usize % MAX_LINKS] += 1,
            Insn::PciReceive { .. } => a.host_in += 1,
            Insn::PciSend { .. } => a.host_out += 1,
            Insn::HostNotify { .. } => a.host_notify += 1,
            _ => {}
        }
        // Deskew, RuntimeDeskew, DeskewRecv and HostWait only hold their queue
        if !matches!(i.id(), EntryId::Deskew | EntryId::RuntimeDeskew | EntryId::DeskewRecv | EntryId::HostWait) {
            a.touch(C2C, ctx.exec_cycle());
        }
    }
}

/// The counters' index of an MXM ABC type code (0 i8, 1 f16, 2 u8, 3 fp8, 4 bf16, 5 bfp8 in the encoding; i8, u8, f16, fp8 in the
/// counters).
fn mxm_type(code: u32) -> usize {
    match code {
        0 => 0,
        1 => 2,
        2 => 1,
        4 => 2, // bf16 is counted with f16: the same plane pair and rate
        5 => 2, // bfp8 is counted with f16 too: its storage (a stream pair per row) and its rate are those of f16, not of fp8
        _ => 3,
    }
}
