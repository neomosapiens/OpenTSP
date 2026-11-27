//! Where each unit sits on the stream rows and which unit each queue issues to
//! (`core/layout.sail`, `core/queues.sail`).
//!
//! The functions keep the Sail names and argument order. `layout_matches_encoding_json_stations`
//! checks them against the `stations` of `encoding.json` on every queue and row.
use crate::geom::Geom;

/// `unit_kind` of `core/layout.sail`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnitKind {
    Vxm,
    MemW,
    MemE,
    MxmW,
    MxmE,
    SxmWDst,
    SxmWTsp,
    SxmWShf,
    SxmWPrm,
    SxmWSel,
    SxmEDst,
    SxmETsp,
    SxmEShf,
    SxmEPrm,
    SxmESel,
    C2cWTx,
    C2cWRx,
    C2cETx,
    C2cERx,
}

impl UnitKind {
    /// The unit's name in `encoding.json` (`units`, `stations`).
    pub fn name(self) -> &'static str {
        match self {
            UnitKind::Vxm => "VXM",
            UnitKind::MemW => "MEM_W",
            UnitKind::MemE => "MEM_E",
            UnitKind::MxmW => "MXM_W",
            UnitKind::MxmE => "MXM_E",
            UnitKind::SxmWDst => "SXM_W_DST",
            UnitKind::SxmWTsp => "SXM_W_TSP",
            UnitKind::SxmWShf => "SXM_W_SHF",
            UnitKind::SxmWPrm => "SXM_W_PRM",
            UnitKind::SxmWSel => "SXM_W_SEL",
            UnitKind::SxmEDst => "SXM_E_DST",
            UnitKind::SxmETsp => "SXM_E_TSP",
            UnitKind::SxmEShf => "SXM_E_SHF",
            UnitKind::SxmEPrm => "SXM_E_PRM",
            UnitKind::SxmESel => "SXM_E_SEL",
            UnitKind::C2cWTx => "C2C_W_TX",
            UnitKind::C2cWRx => "C2C_W_RX",
            UnitKind::C2cETx => "C2C_E_TX",
            UnitKind::C2cERx => "C2C_E_RX",
        }
    }
}

/// A stream row (`row` of the model): E flows east, W flows west.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowId {
    E,
    W,
}

/// `issuer` of `core/queues.sail`: a queue's unit and instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Issuer {
    pub unit: UnitKind,
    /// MEM: global slice 0..87; others 0.
    pub slice: u8,
    /// MXM plane 0..3; others 0.
    pub plane: u8,
    /// SXM: 0 west, 1 east.
    pub sxm: u8,
}

const NO_ISSUER: Issuer = Issuer { unit: UnitKind::Vxm, slice: 0, plane: 0, sxm: 0 };

// model: core/queues.sail sxm_unit_at
fn sxm_unit_at(k: usize, east: bool) -> UnitKind {
    use UnitKind::*;
    let pick = |w, e| if east { e } else { w };
    match k {
        0..=3 => pick(SxmWDst, SxmEDst),
        4..=5 => pick(SxmWTsp, SxmETsp),
        6..=7 => pick(SxmWShf, SxmEShf),
        8 => pick(SxmWPrm, SxmEPrm),
        9 => pick(SxmWSel, SxmESel),
        10..=11 => pick(C2cWTx, C2cETx),
        _ => pick(C2cWRx, C2cERx),
    }
}

/// The unit of data queue `q` (0..143).
// model: core/queues.sail issuer_of
pub fn issuer_of(q: usize) -> Issuer {
    assert!(q < 144, "issuer_of({q})");
    match q {
        0..=43 => Issuer { unit: UnitKind::MemW, slice: q as u8, ..NO_ISSUER },
        44..=87 => Issuer { unit: UnitKind::MemE, slice: q as u8, ..NO_ISSUER },
        88..=103 => NO_ISSUER,
        104..=109 => Issuer { unit: UnitKind::MxmW, plane: if q <= 106 { 0 } else { 1 }, ..NO_ISSUER },
        110..=115 => Issuer { unit: UnitKind::MxmE, plane: if q <= 112 { 2 } else { 3 }, ..NO_ISSUER },
        116..=129 => Issuer { unit: sxm_unit_at(q - 116, false), ..NO_ISSUER },
        _ => Issuer { unit: sxm_unit_at(q - 130, true), sxm: 1, ..NO_ISSUER },
    }
}

/// MEM quad of slice `s` (0..87): `(s mod 44) / 4`; quad 0 is nearest the VXM.
// model: core/layout.sail mem_quad
pub fn mem_quad(s: usize) -> i64 {
    ((s % 44) / 4) as i64
}

/// Does MEM slice `s` exist?
// model: core/layout.sail mem_slice_present
pub fn mem_slice_present(g: &Geom, s: usize) -> bool {
    (s % 44) < g.mem_slices as usize
}

/// A data queue exists in every profile; its unit does not when it is the queue of a MEM slice
/// the profile lacks.
// model: core/queues.sail queue_present
pub fn queue_present(g: &Geom, q: usize) -> bool {
    if q <= 87 {
        mem_slice_present(g, q)
    } else if (104..=115).contains(&q) {
        // the index of the queue's plane in its hemisphere is below `mxm.planes_per_hemisphere`
        let k = if q <= 109 { usize::from(q > 106) } else { usize::from(q > 112) };
        (k as u64) < g.mxm_planes
    } else {
        true
    }
}

/// The plane of MXM queue `q`: `Issuer::plane` is the plane of the queue map (0, 1 west; 2, 3 east); the plane of an east
/// queue is `mxm.planes_per_hemisphere` plus its index in the hemisphere.
// model: core/queues.sail issuer_of
pub fn mxm_plane(g: &Geom, q: usize) -> usize {
    let i = issuer_of(q).plane as usize;
    if i >= 2 {
        g.mxm_planes as usize + (i - 2)
    } else {
        i
    }
}

/// Is `q` (a data queue index) a queue of an MXM unit?
// model: core/queues.sail is_mxm_queue
pub fn is_mxm_queue(q: usize) -> bool {
    q <= 143 && matches!(issuer_of(q).unit, UnitKind::MxmW | UnitKind::MxmE)
}

fn gap_or_none(g: i64) -> Option<u8> {
    (0..=43).contains(&g).then_some(g as u8)
}

/// The gap at which unit `u` (instance `inst`: the MEM slice for MEM units, else 0) READS row
/// `r`, if any.
// model: core/layout.sail read_gap
pub fn read_gap(g: &Geom, u: UnitKind, inst: usize, r: RowId) -> Option<u8> {
    use RowId::*;
    use UnitKind::*;
    let l = &g.layout;
    if matches!(u, MemW | MemE) && !mem_slice_present(g, inst) {
        return None;
    }
    let q = mem_quad(inst);
    match (u, r) {
        (Vxm, _) => Some(l.vxm_gap as u8),
        (MemW, E) | (MemE, W) => gap_or_none(l.mem_up_gap_base - q),
        (MemW, W) | (MemE, E) => gap_or_none(l.mem_down_gap_base + q),
        (MxmW, W) | (MxmE, E) => Some(l.mxm_in_gap as u8),
        (SxmWDst, W) | (SxmEDst, E) => Some(l.sxm_dst_gap as u8),
        (SxmWTsp, W) | (SxmWPrm, W) | (SxmWShf, W) | (SxmWSel, W) => Some(l.sxm_turn_in_gap as u8),
        (SxmETsp, E) | (SxmEPrm, E) | (SxmEShf, E) | (SxmESel, E) => Some(l.sxm_turn_in_gap as u8),
        (C2cWTx, W) | (C2cETx, E) => Some(l.c2c_tx_gap as u8),
        // An RX unit reads no operand stream; it fetches its instruction text (Ifetch) where the
        // TX units of its hemisphere read.
        (C2cWRx, W) | (C2cERx, E) => Some(l.c2c_tx_gap as u8),
        _ => None,
    }
}

/// The gap at which unit `u` WRITES row `r`, if any.
// model: core/layout.sail write_gap
pub fn write_gap(g: &Geom, u: UnitKind, inst: usize, r: RowId) -> Option<u8> {
    use RowId::*;
    use UnitKind::*;
    let l = &g.layout;
    match (u, r) {
        (Vxm, _) => Some(l.vxm_gap as u8),
        (MemW, _) | (MemE, _) => read_gap(g, u, inst, r),
        (MxmW, E) | (MxmE, W) => Some(l.mxm_acc_gap as u8),
        (SxmWDst, W) | (SxmEDst, E) => Some(l.sxm_dst_gap as u8),
        (SxmWTsp, E) | (SxmWPrm, E) | (SxmWShf, E) | (SxmWSel, E) => Some(l.sxm_turn_out_gap as u8),
        (SxmETsp, W) | (SxmEPrm, W) | (SxmEShf, W) | (SxmESel, W) => Some(l.sxm_turn_out_gap as u8),
        (C2cWRx, E) | (C2cERx, W) => Some(l.c2c_rx_gap as u8),
        _ => None,
    }
}

/// The row a unit on `east_side` sees data arriving on from the centre, and the outward row.
// model: core/layout.sail inward_row, outward_row
pub fn inward_row(east_side: bool) -> RowId {
    if east_side {
        RowId::W
    } else {
        RowId::E
    }
}

pub fn outward_row(east_side: bool) -> RowId {
    if east_side {
        RowId::E
    } else {
        RowId::W
    }
}
