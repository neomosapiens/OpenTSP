//! Parametrised schedule replay (`opentsp-iss/README.md`).
//! Behind `--features replay`; used by the platform only when its host opts in. Off by default.
//!
//! A successful `arm` starts an exact word-write journal and pins the run's text arena. Complete a low-level run with
//! `Machine::run`, including after bounded `run_until` calls: it checks the whole schedule and text-write history before
//! returning a completed result, then releases its journal. A changed text word or missing history is an optimization
//! error, never an accepted stale result. Such an error leaves the journal available for rollback and a fresh normal run.
//! `Platform` supplies that rollback/fallback and retains the journal through its final gates. A partial or interrupted run
//! is not a completed result; `arm` alone provides neither rollback nor permission to publish partial outputs.
//!
//! The shipped replay (`driver::replay`) serves a run whose program text equals the recorded one byte for byte. A serving
//! decode step is a bucket program whose *sites* (the fields the host patches: embedding words, `wpe`, mask, K and V block
//! addresses and write addresses) take other values at every step, so every step is a different program to that key. A
//! [`ParamSchedule`] is one recorded schedule plus the site table: for another plan (one value per site) it re-resolves the
//! operands of the executed elements, gates the run on the table instead of on a hash of the text, and hands the instantiated
//! schedule to the unchanged compiled replay.
//!
//! How an executed element finds its instruction. The text of a queue is a FIFO of units (initial image, then the fetched
//! blocks, padding dropped); the *n*th dispatch of a queue takes the *n*th complete parcel of that stream (`Iq::parse`: a NOP
//! prefix, instructions of one or two units, a parcel ends at the first instruction whose bundle bit is 0). A static parse of the
//! stream therefore gives the instructions a dispatch at cycle `c0` takes, and every executed element (`ExecElement::c0`: the
//! cycle of the dispatch, also for the references of an AGT loop; a repeated instruction takes the queue's last repeatable one,
//! which is the most recent equal instruction) is matched to the instruction it executes. The match is checked against the
//! recorded instruction: a mapping that is wrong is a [`BuildError`], not a wrong result.
//!
//! What the gate covers (everything else of a run is a function of the program shape and of data the replay executes):
//! * every value is on the table's range and grid and the instruction with the new fields still decodes;
//! * the initial IQ images equal the recorded ones outside the site fields (this replaces the hash of the text);
//! * the bank rule MEM-001 and the loop clash follow the dispatch, which checks banks in places a patched address can reach, and the gate evaluates
//!   the two of them again with the banks of the new plan: (1) the references of one queue issued in one cycle meet in a bank
//!   (`tags_clash`; `Machine::tag_of` at the element's AGT offset); a group with a member that is not a plain reference (a
//!   fetch) keeps the bank of the patched one as it was; (2) an `IfetchM` of address `a` issued in cycle `c`, as an instruction
//!   or as a Repeat iteration, takes the bank of word `a + 1` in cycle `c + 1` (`reserve_local_fetch`) against the Write side of
//!   that cycle (`banks_meet`), so a patched Write issued one cycle after an `IfetchM` of its queue is compared with that bank
//!   even when it is alone in its cycle. A patched reference that is in neither cannot meet another in a bank;
//! * a site is accepted only on the memory address (`addr`) of a Read or a Write, the field the bank rules above judge and the
//!   replay executes again (segment, bounds). A plan that changes any other field (the address of an `IfetchM`, the count of a
//!   `Repeat`, a stream, a thread) is refused ([`Gate::Site`]) and the run takes the normal driver: such a field changes what is
//!   fetched, dispatched or reported, and the recorded schedule does not describe that.
//!
//! The fetched-text hash of the shipped replay is switched off for an instantiated schedule: the text that reaches the queues in a
//! run is the one of the plan, which the host wrote by the same site table.
use super::execute::ExecElement;
use super::replay::{Op, Schedule, TextHash};
use crate::frontend::{decode_text, text_width};
use crate::slices::mem::{banks_meet, tags_clash, BankUse};
use crate::Machine;
use ots_isa::{Code as Diag, Insn, Isa, PADDING};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// One site of the patch table: the field `lo..lo+width` of the instruction of `nunits` units that starts at logical unit
/// `unit` of the queue's whole text (initial image, then the blocks, padding included), and the values it may take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub queue: u16,
    pub unit: usize,
    pub nunits: usize,
    pub lo: u32,
    pub width: u32,
    pub vmin: i64,
    pub vmax: i64,
    pub stride: i64,
    pub span: i64,
}

/// The whole text of a queue as the patch table counts it: initial image, then the blocks (padding included).
#[derive(Clone, Debug)]
pub struct QueueText {
    pub queue: u16,
    pub units: Vec<u32>,
}

/// Where the logical text of a queue lies in the SRAM: `words` consecutive words from (`slice`, `addr`) hold the units
/// `start..` of queue `queue` (the blocks of a paged program: the text package's tensors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub queue: u16,
    pub start: usize,
    pub slice: u8,
    pub addr: u32,
    pub words: u32,
}

/// The table cannot serve the recording: the schedule is not parametrisable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildError(pub String);

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the schedule cannot be parametrised: {}", self.0)
    }
}

impl std::error::Error for BuildError {}

/// Why a plan is not served by the schedule: the run takes the normal driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    /// The values are not a plan of this table (count, range, grid).
    Values(String),
    /// An instruction with the new fields does not decode.
    Decode(String),
    /// A changed site is on a field other than the memory address of a Read or a Write: the schedule is not a function of it.
    Site(String),
    /// A memory reference changed bank: the bank rule would give another result.
    Bank(String),
    /// The initial IQ images differ from the recorded ones outside the sites.
    Images(String),
    /// A word of the text arena in the SRAM is not the one of the plan (the recorded text with the plan's values at the sites).
    Text(String),
    /// An ECC fault is live in the machine (an overlay can change what a fetch or a read finds): the normal driver runs.
    Ecc,
    /// The machine refused the schedule (`Machine::replay_schedule_with`).
    Machine(String),
}

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gate::Values(s) => write!(f, "values: {s}"),
            Gate::Decode(s) => write!(f, "decode: {s}"),
            Gate::Site(s) => write!(f, "site: {s}"),
            Gate::Bank(s) => write!(f, "bank: {s}"),
            Gate::Images(s) => write!(f, "images: {s}"),
            Gate::Text(s) => write!(f, "text: {s}"),
            Gate::Ecc => f.write_str("an ECC fault is live"),
            Gate::Machine(s) => write!(f, "machine: {s}"),
        }
    }
}

impl std::error::Error for Gate {}

/// What [`ParamSchedule::instantiate`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Instantiated {
    /// Sites whose value is not the one of the previous plan.
    pub sites_changed: usize,
    /// Instructions decoded again, and executed elements that took the new instruction.
    pub insns: usize,
    pub elements: usize,
    /// Memory references whose bank was compared.
    pub banks_checked: usize,
}

/// What the build found, for the report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Shape {
    pub ticks: usize,
    pub execs: usize,
    pub sites: usize,
    /// Instructions that hold at least one site, and the ones of them an element executes.
    pub site_insns: usize,
    pub executed_site_insns: usize,
    /// Executed elements of a site instruction (patched at each plan) and of any other instruction (never touched).
    pub patched_elements: usize,
    /// Patched elements that are alone in their (queue, issue cycle): no bank rule can apply to them.
    pub solo_elements: usize,
    pub static_elements: usize,
    /// Executed elements of a queue whose text the caller did not give: their instructions are not matched to the text.
    pub unverified_elements: usize,
    /// Elements whose instruction was found in another parcel than the dispatch at `c0` (repeats).
    pub found_back: usize,
    /// Elements that fit several instructions of their parcel, all equal and without sites.
    pub equal_static: usize,
    /// The instruction kinds that hold sites (catalogue name, instructions).
    pub kinds: Vec<(String, usize)>,
    /// Retained bytes of the instantiated schedule.
    pub schedule_bytes: usize,
}

struct StreamItem {
    start: u32,
    insn: Result<Insn, Diag>,
}

/// The complete parcels of the text of one queue, padding dropped: the items and, per parcel, its first item and item count.
struct Stream {
    items: Vec<StreamItem>,
    parcels: Vec<(u32, u32)>,
}

fn parse(isa: &Isa, units: &[u32]) -> Stream {
    let s: Vec<(u32, u32)> =
        units.iter().enumerate().filter(|(_, u)| **u != PADDING).map(|(i, u)| (*u, i as u32)).collect();
    let (mut items, mut parcels, mut pend, mut pos) = (Vec::new(), Vec::new(), 0usize, 0usize);
    while pos < s.len() {
        let u0 = s[pos].0;
        let w = text_width(u0);
        if w == 2 && pos + 1 >= s.len() {
            break;
        }
        let r = decode_text(isa, u0, if w == 2 { s[pos + 1].0 } else { 0 });
        let bundle = u0 >> 31 == 1;
        let nop = match r {
            Ok(Insn::NOP { n }) if !bundle && pend == 0 => n as u64,
            _ => 0,
        };
        let start = s[pos].1;
        pos += w;
        if nop > 0 {
            continue;
        }
        items.push(StreamItem { start, insn: r });
        pend += 1;
        if !bundle {
            parcels.push(((items.len() - pend) as u32, pend as u32));
            pend = 0;
        }
    }
    Stream { items, parcels }
}

/// The bank an `IfetchM` of address `addr` takes in the cycle after its issue (the second read, word `addr + 1`), as the
/// dispatch reserves it, without the check against the memory segment: a gate may only refuse more than the dispatch.
fn fetch_second_bank(m: &Machine, addr: u32) -> Option<BankUse> {
    let e = u64::from(addr) + 1;
    (e < m.geom.words_per_slice).then(|| m.addr_bank(e as i64))
}

fn field_mask(s: &Site) -> u64 {
    ((1u64 << s.width) - 1) << s.lo
}

/// Set the field of `s` to `v` in the units of its instruction (`units[0]` is the first, the high half of a two-unit one).
fn put(units: &mut [u32; 2], s: &Site, v: i64) {
    let mask = field_mask(s);
    let w = if s.nunits == 2 { ((units[0] as u64) << 32) | units[1] as u64 } else { units[0] as u64 };
    let n = (w & !mask) | ((v as u64) << s.lo);
    if s.nunits == 2 {
        units[0] = (n >> 32) as u32;
        units[1] = n as u32;
    } else {
        units[0] = n as u32;
    }
}

/// An instruction that holds sites.
struct Item {
    queue: u16,
    /// The units of the instruction with the fields of the recorded plan.
    units: [u32; 2],
    nunits: usize,
    sites: Vec<u32>,
    /// The recorded instruction and the one in the elements now.
    base: Insn,
    cur: Insn,
}

/// A schedule recorded for one plan, and what is needed to serve any other plan of the same table.
pub struct ParamSchedule {
    sites: Vec<Site>,
    items: Vec<Item>,
    site_item: Vec<u32>,
    /// CSR: the executed elements of item `i` are `elems[off[i]..off[i + 1]]`; `solo[k]` says whether `elems[k]` is the only
    /// reference of its queue issued in its cycle, `group` has the elements issued with each of the others, `elem_item` the
    /// item of each patched element.
    elems: Vec<u32>,
    off: Vec<u32>,
    solo: Vec<bool>,
    group: HashMap<u32, Vec<u32>>,
    /// For a patched Write issued one cycle after an `IfetchM` of its queue: the bank of the fetch's second read.
    follow: HashMap<u32, BankUse>,
    elem_item: HashMap<u32, u32>,
    /// The values the elements hold now (the plan of the last successful instantiation).
    cur: Vec<i32>,
    sched: Arc<Schedule>,
    /// The recorded initial images, and the bits of each unit that belong to sites.
    base_images: Vec<Vec<u32>>,
    image_mask: Vec<Vec<u32>>,
    /// The whole text of the queues for the plan the elements hold now, and where the SRAM holds it.
    text: HashMap<u16, Vec<u32>>,
    arena: Vec<Run>,
    /// Times the schedule was copied because a machine of an earlier plan still held it (drop that machine, or reset its
    /// `rp`, before the next plan).
    pub copies: usize,
    shape: Shape,
}

impl ParamSchedule {
    /// Parametrise `sched`, recorded by the normal driver on `m` (the machine of the recording, with its images loaded) for the
    /// plan `values` (one per site of `sites`); `text` is the whole text of every queue that holds a site, as the table counts
    /// it, with any plan's values (they are replaced by `values` here).
    pub fn build(
        m: &Machine,
        mut sched: Schedule,
        sites: Vec<Site>,
        text: &[QueueText],
        values: &[i32],
    ) -> Result<ParamSchedule, BuildError> {
        let err = |s: String| BuildError(s);
        if values.len() != sites.len() {
            return Err(err(format!("{} values for {} sites", values.len(), sites.len())));
        }
        if !sched.compiled() {
            return Err(err("the schedule is not compiled (no operations)".into()));
        }
        if sched.ops.iter().any(|o| matches!(o, Op::Diag(_, Diag::Mem001))) {
            return Err(
                err("the recording has a bank clash (MEM-001): equal banks would not give equal checks".into()),
            );
        }
        // the text of the recorded plan
        let mut by_q: HashMap<u16, Vec<u32>> = text.iter().map(|t| (t.queue, t.units.clone())).collect();
        let mut site_units = Vec::with_capacity(sites.len());
        for (i, s) in sites.iter().enumerate() {
            let units = by_q.get_mut(&s.queue).ok_or_else(|| err(format!("no text for queue {}", s.queue)))?;
            if s.unit + s.nunits > units.len() || !(1..=2).contains(&s.nunits) || s.width == 0 || s.lo + s.width > 64 {
                return Err(err(format!("site {i}: outside the text of queue {}", s.queue)));
            }
            let mut u = [units[s.unit], if s.nunits == 2 { units[s.unit + 1] } else { 0 }];
            put(&mut u, s, values[i] as i64);
            units[s.unit] = u[0];
            if s.nunits == 2 {
                units[s.unit + 1] = u[1];
            }
            site_units.push(s.unit);
        }
        // the items that hold sites
        let mut item_at: HashMap<(u16, usize), u32> = HashMap::new();
        let mut items: Vec<Item> = Vec::new();
        let mut site_item = Vec::with_capacity(sites.len());
        for (i, s) in sites.iter().enumerate() {
            let id = *item_at.entry((s.queue, s.unit)).or_insert_with(|| {
                let units = &by_q[&s.queue];
                items.push(Item {
                    queue: s.queue,
                    units: [units[s.unit], if s.nunits == 2 { units[s.unit + 1] } else { 0 }],
                    nunits: s.nunits,
                    sites: Vec::new(),
                    base: Insn::NOP { n: 0 },
                    cur: Insn::NOP { n: 0 },
                });
                (items.len() - 1) as u32
            });
            if items[id as usize].nunits != s.nunits {
                return Err(err(format!(
                    "sites of one instruction disagree on its width (queue {}, unit {})",
                    s.queue, s.unit
                )));
            }
            items[id as usize].sites.push(i as u32);
            site_item.push(id);
        }
        // the streams of the queues that hold sites, and the dispatch cycles of every queue
        let mut streams: HashMap<u16, Stream> = HashMap::new();
        for (q, units) in &by_q {
            streams.insert(*q, parse(&m.isa, units));
        }
        let mut disp: Vec<Vec<u64>> = vec![Vec::new(); 144];
        for i in 0..sched.cycles.len() {
            for op in &sched.ops[sched.ops_off[i] as usize..sched.ops_off[i + 1] as usize] {
                if let Op::Dispatch(q) = op {
                    if (*q as usize) < 144 {
                        disp[*q as usize].push(sched.cycles[i]);
                    }
                }
            }
        }
        // every instruction with sites starts a unit that the parse found as an instruction
        let mut start_item: HashMap<(u16, u32), u32> = HashMap::new(); // (queue, logical start) -> stream item index
        for (q, st) in &streams {
            for (k, it) in st.items.iter().enumerate() {
                start_item.insert((*q, it.start), k as u32);
            }
        }
        for it in items.iter_mut() {
            let first = sites[it.sites[0] as usize].unit as u32;
            let k = *start_item
                .get(&(it.queue, first))
                .ok_or_else(|| err(format!("queue {}: no instruction starts at unit {first}", it.queue)))?;
            let insn = streams[&it.queue].items[k as usize].insn.map_err(|d| {
                err(format!("queue {}: the instruction at unit {first} does not decode ({d:?})", it.queue))
            })?;
            it.base = insn;
            it.cur = insn;
        }
        let mut shape =
            Shape { ticks: sched.cycles.len(), execs: sched.execs.len(), sites: sites.len(), ..Shape::default() };
        // the origin of every executed element
        let mut pairs: Vec<(u32, u32)> = Vec::new(); // (item, element)
        let mut stream_item_to_item: HashMap<(u16, u32), u32> = HashMap::new();
        for (id, it) in items.iter().enumerate() {
            stream_item_to_item.insert((it.queue, sites[it.sites[0] as usize].unit as u32), id as u32);
        }
        for (j, e) in sched.execs.iter().enumerate() {
            let q = e.q as u16;
            let Some(st) = streams.get(&q) else {
                shape.static_elements += 1;
                shape.unverified_elements += 1;
                continue;
            };
            let d = &disp[e.q];
            let n = d.partition_point(|&c| c <= e.c0);
            let want = Ok(e.instr);
            let mut cands: Vec<u32> = Vec::new();
            if n > 0 && d[n - 1] == e.c0 {
                if let Some(&(f, cnt)) = st.parcels.get(n - 1) {
                    cands.extend((f..f + cnt).filter(|k| st.items[*k as usize].insn == want));
                }
            }
            if cands.is_empty() {
                // a repeat: the most recent equal instruction dispatched before
                let mut found = None;
                'back: for p in (0..n.min(st.parcels.len())).rev().take(1 << 16) {
                    let (f, cnt) = st.parcels[p];
                    for k in (f..f + cnt).rev() {
                        if st.items[k as usize].insn == want {
                            found = Some(k);
                            break 'back;
                        }
                    }
                }
                match found {
                    Some(k) => {
                        cands.push(k);
                        shape.found_back += 1;
                    }
                    None => {
                        return Err(err(format!(
                            "element {j} (queue {q}, dispatched at cycle {}, {:?}) is in no parcel of the text",
                            e.c0, e.instr
                        )))
                    }
                }
            }
            let ids: Vec<Option<u32>> =
                cands.iter().map(|k| stream_item_to_item.get(&(q, st.items[*k as usize].start)).copied()).collect();
            if ids.iter().all(Option::is_none) {
                shape.static_elements += 1;
                if cands.len() > 1 {
                    shape.equal_static += 1;
                }
                continue;
            }
            if ids.len() > 1 {
                return Err(err(format!(
                    "element {j} (queue {q}, cycle {}) fits {} instructions of its parcel, some with sites: {:?}",
                    e.c0,
                    ids.len(),
                    e.instr
                )));
            }
            pairs.push((ids[0].expect("some"), j as u32));
            shape.patched_elements += 1;
        }
        pairs.sort_unstable();
        let mut off = vec![0u32; items.len() + 1];
        for (it, _) in &pairs {
            off[*it as usize + 1] += 1;
        }
        for i in 0..items.len() {
            off[i + 1] += off[i];
        }
        let elems: Vec<u32> = pairs.iter().map(|p| p.1).collect();
        // the references issued in one cycle of one queue
        let mut at_cycle: HashMap<(usize, u64), Vec<u32>> = HashMap::new();
        let site_queues: std::collections::HashSet<u16> = sites.iter().map(|s| s.queue).collect();
        for (j, e) in sched.execs.iter().enumerate() {
            if site_queues.contains(&(e.q as u16)) {
                at_cycle.entry((e.q, e.c0 + e.k)).or_default().push(j as u32);
            }
        }
        // the second reads of the `IfetchM`s issued in the cycle before, by queue
        let mut fetch2: HashMap<(usize, u64), BankUse> = HashMap::new();
        for e in &sched.execs {
            if let (Insn::IfetchM { addr }, true) = (e.instr, site_queues.contains(&(e.q as u16))) {
                if let Some(b) = fetch_second_bank(m, addr) {
                    fetch2.insert((e.q, e.c0 + e.k + 1), b);
                }
            }
        }
        let mut solo = Vec::with_capacity(elems.len());
        let mut group: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut follow: HashMap<u32, BankUse> = HashMap::new();
        let mut elem_item: HashMap<u32, u32> = HashMap::new();
        for (it, j) in &pairs {
            let e = &sched.execs[*j as usize];
            let g = &at_cycle[&(e.q, e.c0 + e.k)];
            elem_item.insert(*j, *it);
            let after_fetch = match (e.instr, fetch2.get(&(e.q, e.c0 + e.k))) {
                (Insn::Write { .. }, Some(b)) => {
                    follow.insert(*j, *b);
                    true
                }
                _ => false,
            };
            solo.push(g.len() == 1 && !after_fetch);
            if g.len() > 1 {
                group.insert(*j, g.clone());
            }
        }
        shape.solo_elements = solo.iter().filter(|s| **s).count();
        shape.site_insns = items.len();
        shape.executed_site_insns = (0..items.len()).filter(|i| off[*i + 1] > off[*i]).count();
        let mut kinds: HashMap<String, usize> = HashMap::new();
        for it in &items {
            *kinds.entry(it.base.entry().name.to_string()).or_default() += 1;
        }
        let mut kinds: Vec<(String, usize)> = kinds.into_iter().collect();
        kinds.sort();
        shape.kinds = kinds;
        // the images and the bits of them that are sites
        let base_images: Vec<Vec<u32>> = (0..crate::geom::QUEUES).map(|q| m.images[q].clone()).collect();
        let mut image_mask: Vec<Vec<u32>> = vec![Vec::new(); crate::geom::QUEUES];
        // the images drop padding: the place of a site in its image is its logical unit minus the padding before it
        let mut before: HashMap<u16, Vec<u32>> = HashMap::new();
        for s in &sites {
            let q = s.queue as usize;
            if q >= crate::geom::QUEUES {
                continue;
            }
            let pre = before.entry(s.queue).or_insert_with(|| {
                let mut n = 0u32;
                text.iter()
                    .find(|t| t.queue == s.queue)
                    .map(|t| {
                        t.units
                            .iter()
                            .map(|u| {
                                let r = n;
                                n += u32::from(*u != PADDING);
                                r
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            });
            let at = pre[s.unit] as usize;
            if at + s.nunits <= base_images[q].len() {
                if image_mask[q].is_empty() {
                    image_mask[q] = vec![0; base_images[q].len()];
                }
                let mask = field_mask(s);
                if s.nunits == 2 {
                    image_mask[q][at] |= (mask >> 32) as u32;
                    image_mask[q][at + 1] |= mask as u32;
                } else {
                    image_mask[q][at] |= mask as u32;
                }
            }
        }
        // the gate of the instantiated text is the site table: the hash of every tick is not compared
        sched.text = vec![0; sched.cycles.len()];
        // Parametrised replay has its own exact arena/site gates; do not retain
        // an unused copy of the recorded plan's payloads in its cache budget.
        sched.fetched = Vec::new();
        sched.fetched_off = Vec::new();
        shape.schedule_bytes = sched.bytes();
        let mut ps = ParamSchedule {
            sites,
            items,
            site_item,
            elems,
            off,
            solo,
            group,
            follow,
            elem_item,
            cur: values.to_vec(),
            sched: Arc::new(sched),
            base_images,
            image_mask,
            text: by_q,
            arena: Vec::new(),
            copies: 0,
            shape,
        };
        // the recorded plan must reproduce the recorded instructions: the table, the text and the values agree
        for it in 0..ps.items.len() {
            let u = ps.items[it].units;
            let got = decode_text(&m.isa, u[0], u[1])
                .map_err(|d| err(format!("recorded instruction does not decode: {d:?}")))?;
            if got != ps.items[it].base {
                return Err(err(format!(
                    "queue {}: the text of the recorded plan decodes to another instruction",
                    ps.items[it].queue
                )));
            }
        }
        for it in 0..ps.items.len() {
            for k in ps.off[it]..ps.off[it + 1] {
                let e = ps.sched.execs[ps.elems[k as usize] as usize];
                if e.instr != ps.items[it].base {
                    return Err(err("an element does not hold the instruction its origin has".into()));
                }
            }
        }
        let _ = &mut ps;
        Ok(ps)
    }

    /// The heap bytes this schedule retains: the instantiated [`Schedule`] and every table of its own (the instructions that hold
    /// sites with their element lists, the groups, the origins of the patched elements, the images and the text of the queues).
    /// What the cache budget counts for a parametrised schedule.
    pub fn bytes(&self) -> usize {
        use std::mem::size_of;
        let vec = |len: usize, item: usize| len * item;
        let map = |m: usize, k: usize, v: usize| m * (k + v + 8); // the entries and the table's control bytes, about
        let mut n = size_of::<ParamSchedule>() + self.sched.bytes();
        n += vec(self.sites.capacity(), size_of::<Site>());
        n += vec(self.items.capacity(), size_of::<Item>())
            + self.items.iter().map(|i| i.sites.capacity() * 4).sum::<usize>();
        n += vec(self.site_item.capacity(), 4) + vec(self.elems.capacity(), 4) + vec(self.off.capacity(), 4);
        n += self.solo.capacity();
        n += map(self.group.capacity(), 4, size_of::<Vec<u32>>())
            + self.group.values().map(|v| v.capacity() * 4).sum::<usize>();
        n += map(self.follow.capacity(), 4, size_of::<BankUse>());
        n += map(self.elem_item.capacity(), 4, 4);
        n += vec(self.cur.capacity(), 4);
        n += self
            .base_images
            .iter()
            .chain(&self.image_mask)
            .map(|v| v.capacity() * 4 + size_of::<Vec<u32>>())
            .sum::<usize>();
        n += map(self.text.capacity(), 2, size_of::<Vec<u32>>())
            + self.text.values().map(|v| v.capacity() * 4).sum::<usize>();
        n += vec(self.arena.capacity(), size_of::<Run>());
        n
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// The values the elements hold now.
    pub fn values(&self) -> &[i32] {
        &self.cur
    }

    /// The schedule as instantiated now.
    pub fn schedule(&self) -> &Arc<Schedule> {
        &self.sched
    }

    fn words(&self, it: usize, values: &[i32]) -> [u32; 2] {
        let item = &self.items[it];
        let mut u = item.units;
        for &s in &item.sites {
            put(&mut u, &self.sites[s as usize], values[s as usize] as i64);
        }
        u
    }

    /// Site `i` is the memory address (`addr`) of a Read or a Write: the only kind of field whose new values this gate covers. The
    /// bank rules above judge such an address, and the replay executes the element again with it (segment, bounds). Any other
    /// field (the address of an `IfetchM`, the count of a `Repeat`, a stream or a thread) changes what is fetched, dispatched or
    /// reported, which the recorded schedule does not describe.
    fn is_address_site(&self, i: usize) -> bool {
        let (s, base) = (&self.sites[i], &self.items[self.site_item[i] as usize].base);
        matches!(base, Insn::Read { .. } | Insn::Write { .. })
            && base
                .entry()
                .vars
                .iter()
                .any(|v| v.name == "addr" && v.hi_n == 0 && u32::from(v.lo) == s.lo && u32::from(v.n) == s.width)
    }

    /// Give the executed elements the instructions of the plan `values` (one per site): the plan work of a step. Nothing changes
    /// when the plan is refused. Only the sites whose value changed since the last plan are looked at.
    pub fn instantiate(&mut self, m: &Machine, values: &[i32]) -> Result<Instantiated, Gate> {
        if values.len() != self.sites.len() {
            return Err(Gate::Values(format!("{} values for a table of {} sites", values.len(), self.sites.len())));
        }
        let mut stats = Instantiated::default();
        let mut dirty: Vec<u32> = Vec::new();
        for (i, &value) in values.iter().enumerate() {
            if value == self.cur[i] {
                continue;
            }
            let (s, v) = (&self.sites[i], value as i64);
            if v < s.vmin || v > s.vmax {
                return Err(Gate::Values(format!("site {i}: value {v} outside [{}, {}]", s.vmin, s.vmax)));
            }
            if (v - s.vmin) % s.stride >= s.span {
                return Err(Gate::Values(format!(
                    "site {i}: value {v:#x} is not on the grid of stride {} from {:#x}",
                    s.stride, s.vmin
                )));
            }
            if !self.is_address_site(i) {
                return Err(Gate::Site(format!(
                    "site {i}: the field {:#x}..{:#x} of {:?} (queue {}) is not the memory address of a Read or a Write",
                    s.lo,
                    s.lo + s.width,
                    self.items[self.site_item[i] as usize].base,
                    s.queue
                )));
            }
            stats.sites_changed += 1;
            dirty.push(self.site_item[i]);
        }
        dirty.sort_unstable();
        dirty.dedup();
        let mut new: Vec<(u32, Insn)> = Vec::with_capacity(dirty.len());
        for &it in &dirty {
            let u = self.words(it as usize, values);
            let insn = decode_text(&m.isa, u[0], u[1])
                .map_err(|d| Gate::Decode(format!("queue {}: {d:?}", self.items[it as usize].queue)))?;
            if insn.id() != self.items[it as usize].base.id() {
                return Err(Gate::Decode(format!(
                    "queue {}: the patch changed the instruction",
                    self.items[it as usize].queue
                )));
            }
            new.push((it, insn));
        }
        // the bank rule, for the references that have company in their queue and cycle
        let newmap: HashMap<u32, Insn> = new.iter().copied().collect();
        for &(it, insn) in &new {
            let base = self.items[it as usize].base;
            if !matches!(insn, Insn::Read { .. } | Insn::Write { .. }) {
                continue;
            }
            for k in self.off[it as usize] as usize..self.off[it as usize + 1] as usize {
                if self.solo[k] {
                    continue;
                }
                let j = self.elems[k];
                if let Some(fb) = self.follow.get(&j) {
                    stats.banks_checked += 1;
                    let off = self.sched.execs[j as usize].off;
                    if banks_meet(m.ref_bank(&insn, off), *fb) {
                        return Err(Gate::Bank(format!(
                            "queue {}: {insn:?} issued the cycle after a fetch meets the bank of the fetch's second read",
                            self.items[it as usize].queue
                        )));
                    }
                }
                let Some(members) = self.group.get(&j) else { continue };
                stats.banks_checked += 1;
                let insn_of = |mj: u32| -> Insn {
                    match self.elem_item.get(&mj).and_then(|i| newmap.get(i)) {
                        Some(i) => *i,
                        None => self.sched.execs[mj as usize].instr,
                    }
                };
                let plain = |i: &Insn| {
                    matches!(i, Insn::Read { .. } | Insn::Write { .. } | Insn::Gather { .. } | Insn::Scatter { .. })
                };
                if members.iter().all(|&mj| plain(&insn_of(mj))) {
                    let tags: Vec<_> =
                        members.iter().map(|&mj| m.tag_of(&insn_of(mj), self.sched.execs[mj as usize].off)).collect();
                    for a in 0..tags.len() {
                        for b in a + 1..tags.len() {
                            if tags_clash(tags[a], tags[b]) {
                                return Err(Gate::Bank(format!(
                                    "queue {}: {:?} and {:?} issued in one cycle meet in a bank",
                                    self.items[it as usize].queue,
                                    insn_of(members[a]),
                                    insn_of(members[b])
                                )));
                            }
                        }
                    }
                } else {
                    // a member that is not a plain reference (a fetch): the bank of this one must be the recorded one
                    let off = self.sched.execs[j as usize].off;
                    let (x, y) = (m.ref_bank(&base, off), m.ref_bank(&insn, off));
                    if x != y {
                        return Err(Gate::Bank(format!(
                            "queue {}: {base:?} has {x:?}, {insn:?} has {y:?} at offset {off}",
                            self.items[it as usize].queue
                        )));
                    }
                }
            }
        }
        // the plan is served: write it
        let units: Vec<[u32; 2]> = new.iter().map(|(it, _)| self.words(*it as usize, values)).collect();
        if Arc::get_mut(&mut self.sched).is_none() {
            self.copies += 1; // a machine still holds the schedule: copy it
        }
        let execs: &mut Vec<ExecElement> = &mut Arc::make_mut(&mut self.sched).execs;
        for ((it, insn), u) in new.iter().zip(&units) {
            let item = &mut self.items[*it as usize];
            item.cur = *insn;
            let start = self.sites[item.sites[0] as usize].unit;
            let t = self.text.get_mut(&item.queue).expect("a queue with sites");
            t[start] = u[0];
            if item.nunits == 2 {
                t[start + 1] = u[1];
            }
            for k in self.off[*it as usize]..self.off[*it as usize + 1] {
                execs[self.elems[k as usize] as usize].instr = *insn;
                stats.elements += 1;
            }
        }
        stats.insns = new.len();
        self.cur.copy_from_slice(values);
        Ok(stats)
    }

    /// Tell where the SRAM holds the text that the queues fetch (see [`Run`]); [`ParamSchedule::arm`] then checks every word of it.
    pub fn set_arena(&mut self, runs: Vec<Run>) {
        self.arena = runs;
    }

    /// Every word of the text arena in the SRAM of `m` is the text of the plan the elements hold now: the gate that replaces
    /// the hash of the fetched text. A word that differs (a flipped bit, another program's text, a site that is not the plan's)
    /// is a plan this schedule does not describe.
    pub fn text_gate(&self, m: &Machine) -> Result<(), Gate> {
        let upw = m.geom.vlanes as usize / 4;
        for r in &self.arena {
            if r.slice as usize >= crate::geom::SLICES
                || u64::from(r.addr) + u64::from(r.words) > m.geom.words_per_slice
            {
                return Err(Gate::Text("the text arena is outside SRAM".into()));
            }
            if m.sram.journal_wrote_range(r.slice as usize, u64::from(r.addr), u64::from(r.words)) {
                return Err(Gate::Text("instruction text was written during replay".into()));
            }
            let Some(units) = self.text.get(&r.queue) else { continue };
            for w in 0..r.words as usize {
                let at = r.start + w * upw;
                let Some(want) = units.get(at..at + upw) else {
                    return Err(Gate::Text(format!("queue {}: the arena is beyond the text", r.queue)));
                };
                let got = m.sram.read(r.slice as usize, u64::from(r.addr) + w as u64);
                for (k, u) in want.iter().enumerate() {
                    if got[4 * k..4 * k + 4] != u.to_le_bytes() {
                        return Err(Gate::Text(format!(
                            "queue {}: unit {} (slice {}, word {:#x}, unit {k}) is not the text of the plan",
                            r.queue,
                            at + k,
                            r.slice,
                            u64::from(r.addr) + w as u64
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// The images of `m` equal the recorded ones outside the sites.
    fn images_gate(&self, m: &Machine) -> Result<(), Gate> {
        for q in 0..crate::geom::QUEUES {
            let (b, c) = (&self.base_images[q], &m.images[q]);
            if b.len() != c.len() {
                return Err(Gate::Images(format!("queue {q}: {} units, recorded {}", c.len(), b.len())));
            }
            let mask = &self.image_mask[q];
            for (u, (x, y)) in b.iter().zip(c).enumerate() {
                let keep = if mask.is_empty() { !0 } else { !mask[u] };
                if (x ^ y) & keep != 0 {
                    return Err(Gate::Images(format!("queue {q}: unit {u} differs outside the sites")));
                }
            }
        }
        Ok(())
    }

    /// Start the replay of the instantiated schedule on `m` (a machine that has not ticked, with the plan's images and SRAM
    /// loaded): the images gate, the key of the run, then the compiled replay (`check`: the normal driver compared with the
    /// schedule tick by tick). The fetched-text hash is off: the site table is the gate. A refusal leaves `m` as it was.
    pub fn arm(&mut self, m: &mut Machine, check: bool) -> Result<(), Gate> {
        if m.ecc.live {
            return Err(Gate::Ecc);
        }
        self.images_gate(m)?;
        self.text_gate(m)?;
        let key = m.schedule_key();
        if Arc::get_mut(&mut self.sched).is_none() {
            self.copies += 1;
        }
        Arc::make_mut(&mut self.sched).key = key;
        m.replay_schedule_with(self.sched.clone(), check).map_err(|r| Gate::Machine(r.to_string()))?;
        m.rp.text_watch =
            Some(self.arena.iter().map(|r| (r.slice as usize, u64::from(r.addr), u64::from(r.words))).collect());
        m.sram.journal_begin();
        m.iq.th = TextHash::default();
        Ok(())
    }
}
