//! The host patcher of `opentsp_tools.patch` (`Table`, `apply`), in Rust: the table is the one
//! `bindings.npz` holds (`names`, `queue`, `unit`, `nunits`, `lo`, `width`, `vmin`, `vmax`, `placeholder`, `fixed`,
//! `stride`, `span`) and a value is checked and written exactly as `apply` does. The images are `(queue, units)`.
use std::collections::HashMap;

/// A one-unit instruction keeps its tag in bits 30..28 and its bundle bit in bit 31; a two-unit one its tag in 63..60.
const NARROW_BITS: u32 = 28;
const WIDE_BITS: u32 = 60;

pub type Images = Vec<(u16, Vec<u32>)>;

/// The columns of a table, as `patch.Table` stores them.
#[derive(Default)]
pub struct Columns {
    pub names: Vec<String>,
    pub queue: Vec<i64>,
    pub unit: Vec<i64>,
    pub nunits: Vec<i64>,
    pub lo: Vec<i64>,
    pub width: Vec<i64>,
    pub vmin: Vec<i64>,
    pub vmax: Vec<i64>,
    pub placeholder: Vec<i64>,
    pub fixed: Vec<u64>,
    pub stride: Vec<i64>,
    pub span: Vec<i64>,
    /// The cut of a paged program's text (`image_queues`, `image_units`, `block_units` of `bindings.npz`): present when
    /// the sites' `unit` counts the whole text of the queue, initial image then blocks.
    pub paging: Option<Paging>,
}

#[derive(Clone, Debug, Default)]
pub struct Paging {
    pub image_queues: Vec<i64>,
    pub image_units: Vec<i64>,
    pub block_units: i64,
}

struct Site {
    queue: u16,
    unit: usize,
    nunits: usize,
    lo: u32,
    width: u32,
    vmin: i64,
    vmax: i64,
    fixed: u64,
    stride: i64,
    span: i64,
}

pub struct Table {
    names: Vec<String>,
    index: HashMap<String, usize>,
    sites: Vec<Site>,
}

/// The values of one step: by site name (`apply`'s dict) or one int32 per site in table order.
pub enum Values {
    Named(Vec<(String, i128)>),
    Dense(Vec<i32>),
}

fn uint(what: &str, v: i64, max: i64) -> Result<usize, String> {
    if (0..=max).contains(&v) {
        Ok(v as usize)
    } else {
        Err(format!("patch table: {what} {v} is out of range"))
    }
}

impl Table {
    /// Validates the columns as `Table.validate` does: equal lengths, distinct names, a field inside the instruction
    /// and clear of its fixed bits (opcode, tag), a range and a placeholder that fit the field, a placeholder on the
    /// grid of stride and span, and no two sites on the same bits.
    pub fn new(c: Columns) -> Result<Table, String> {
        let n = c.names.len();
        let lens = [
            c.queue.len(),
            c.unit.len(),
            c.nunits.len(),
            c.lo.len(),
            c.width.len(),
            c.vmin.len(),
            c.vmax.len(),
            c.placeholder.len(),
            c.fixed.len(),
            c.stride.len(),
            c.span.len(),
        ];
        if lens.iter().any(|l| *l != n) {
            return Err("patch table: a column has the wrong length".into());
        }
        let mut index = HashMap::with_capacity(n);
        let mut used: HashMap<(i64, i64), u64> = HashMap::new();
        let mut sites = Vec::with_capacity(n);
        for (i, name) in c.names.iter().enumerate() {
            if index.insert(name.clone(), i).is_some() {
                return Err("patch table: duplicate site names".into());
            }
            let (nunits, lo, width) =
                (uint("nunits", c.nunits[i], 2)?, uint("lo", c.lo[i], 63)?, uint("width", c.width[i], 63)?);
            if !(1..=2).contains(&nunits) || width == 0 || lo + width > 64 {
                return Err(format!("patch table: site {name}: the field does not fit its instruction"));
            }
            let limit = if nunits == 2 { WIDE_BITS } else { NARROW_BITS };
            if (lo + width) as u32 > limit {
                return Err(format!("patch table: site {name}: the field does not fit the {nunits}-unit instruction"));
            }
            let (vmin, vmax) = (c.vmin[i], c.vmax[i]);
            if !(0 <= vmin && vmin <= c.placeholder[i] && c.placeholder[i] <= vmax && (vmax as u64) < 1u64 << width) {
                return Err(format!("patch table: site {name}: the range does not fit the {width}-bit field"));
            }
            if (((1u64 << width) - 1) << lo) & c.fixed[i] != 0 {
                return Err(format!(
                    "patch table: site {name}: bits {lo}..{} overlap the opcode of the instruction",
                    lo + width - 1
                ));
            }
            if c.stride[i] < 1 || c.span[i] < 1 || c.span[i] > c.stride[i] {
                return Err(format!("patch table: site {name}: stride {} and span {}", c.stride[i], c.span[i]));
            }
            if (c.placeholder[i] - vmin) % c.stride[i] >= c.span[i] {
                return Err(format!(
                    "patch table: site {name}: stride {}, span {} and placeholder {} from {vmin}",
                    c.stride[i], c.span[i], c.placeholder[i]
                ));
            }
            let field = ((1u64 << width) - 1) << lo;
            for k in 0..nunits {
                let part = if nunits == 2 { [field >> 32, field & 0xffff_ffff][k] } else { field };
                let slot = used.entry((c.queue[i], c.unit[i].saturating_add(k as i64))).or_insert(0);
                if *slot & part != 0 {
                    return Err(format!("patch table: site {name} overlaps another site"));
                }
                *slot |= part;
            }
            sites.push(Site {
                queue: uint("queue", c.queue[i], 0xffff)? as u16,
                unit: uint("unit", c.unit[i], i64::from(u32::MAX))?,
                nunits,
                lo: lo as u32,
                width: width as u32,
                vmin,
                vmax,
                fixed: c.fixed[i],
                stride: c.stride[i],
                span: c.span[i],
            });
        }
        Ok(Table { names: c.names, index, sites })
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    fn check(&self, i: usize, v: i128) -> Result<(), String> {
        let (s, n) = (&self.sites[i], &self.names[i]);
        if !(i128::from(s.vmin)..=i128::from(s.vmax)).contains(&v) {
            return Err(format!("site {n}: value {v} outside [{}, {}]", s.vmin, s.vmax));
        }
        if (v - i128::from(s.vmin)) % i128::from(s.stride) >= i128::from(s.span) {
            return Err(format!("site {n}: value {v:#x} is not on the grid of stride {} from {:#x}", s.stride, s.vmin));
        }
        Ok(())
    }

    /// The values as (site, value) pairs, every one checked and, as `apply(strict=True)`, every site given a value.
    fn resolve(&self, values: &Values) -> Result<Vec<(usize, i128)>, String> {
        match values {
            Values::Named(v) => {
                let unknown: Vec<&str> =
                    v.iter().map(|(n, _)| n.as_str()).filter(|n| !self.index.contains_key(*n)).take(5).collect();
                if !unknown.is_empty() {
                    return Err(format!("unknown patch sites {unknown:?}"));
                }
                let mut seen = vec![false; self.len()];
                let pairs: Vec<(usize, i128)> = v
                    .iter()
                    .map(|(n, x)| {
                        let i = self.index[n];
                        seen[i] = true;
                        (i, *x)
                    })
                    .collect();
                let absent: Vec<&str> =
                    seen.iter().enumerate().filter(|(_, s)| !**s).map(|(i, _)| self.names[i].as_str()).collect();
                if !absent.is_empty() {
                    let more = if absent.len() > 5 { format!(" and {} more", absent.len() - 5) } else { String::new() };
                    return Err(format!("no value for patch sites {:?}{more}", &absent[..absent.len().min(5)]));
                }
                Ok(pairs)
            }
            Values::Dense(v) => {
                if v.len() != self.len() {
                    return Err(format!("{} patch values for a table of {} sites", v.len(), self.len()));
                }
                Ok(v.iter().enumerate().map(|(i, x)| (i, i128::from(*x))).collect())
            }
        }
    }

    /// The values of a step as one int32 per site in table order, every one checked as `apply` checks it (the values the
    /// parametrised replay of a step takes).
    pub fn dense(&self, values: &Values) -> Result<Vec<i32>, String> {
        let pairs = self.resolve(values)?;
        let mut out = vec![0i32; self.len()];
        for (i, v) in pairs {
            self.check(i, v)?;
            out[i] = i32::try_from(v).map_err(|_| format!("site {}: value {v} does not fit 32 bits", self.names[i]))?;
        }
        Ok(out)
    }

    /// The sites as the platform's parametrised replay counts them (`tsp_iss::driver::replay_param::Site`).
    #[cfg(feature = "replay")]
    pub fn param_sites(&self) -> Vec<tsp_iss::driver::replay_param::Site> {
        self.sites
            .iter()
            .map(|s| tsp_iss::driver::replay_param::Site {
                queue: s.queue,
                unit: s.unit,
                nunits: s.nunits,
                lo: s.lo,
                width: s.width,
                vmin: s.vmin,
                vmax: s.vmax,
                stride: s.stride,
                span: s.span,
            })
            .collect()
    }

    /// The queue, the first unit and the number of units of site `i`.
    pub fn position(&self, i: usize) -> (u16, usize, usize) {
        let s = &self.sites[i];
        (s.queue, s.unit, s.nunits)
    }

    pub fn name(&self, i: usize) -> &str {
        &self.names[i]
    }

    /// `patch.apply` on `images`: every value is checked first, then the fields are set in table order, each on the
    /// word the earlier ones left. A refused value leaves the images as they were; a refused site (opcode or length
    /// changed, a unit outside the initial image) may leave earlier sites written, so the caller applies to a copy.
    pub fn apply(&self, images: &mut Images, values: &Values) -> Result<(), String> {
        self.apply_to(images, values)
    }

    /// `apply` on any holder of the units (the initial images, or the images and text blocks of a paged program).
    pub fn apply_to(&self, store: &mut impl Units, values: &Values) -> Result<(), String> {
        let pairs = self.resolve(values)?;
        for &(i, v) in &pairs {
            self.check(i, v)?;
        }
        for &(i, v) in &pairs {
            let (s, n) = (&self.sites[i], &self.names[i]);
            let word = store.read(i, s.queue, s.unit, s.nunits).map_err(|e| format!("site {n}: {e}"))?;
            let mask = ((1u64 << s.width) - 1) << s.lo;
            let new = (word & !mask) | ((v as u64) << s.lo); // checked above: 0 <= v < 2^width
            let limit = if s.nunits == 2 { WIDE_BITS } else { NARROW_BITS };
            if (new ^ word) & s.fixed != 0 || (new ^ word) >> limit != 0 {
                return Err(format!("site {n}: the patch would change the opcode or the length of the instruction"));
            }
            store.write(i, s.queue, s.unit, s.nunits, new);
        }
        Ok(())
    }
}

/// Where the units of the sites are held: `read` gives the instruction (one unit, or two with the first in bits 63..32)
/// of site `i`, `write` puts it back.
pub trait Units {
    fn read(&self, site: usize, queue: u16, unit: usize, nunits: usize) -> Result<u64, String>;
    fn write(&mut self, site: usize, queue: u16, unit: usize, nunits: usize, word: u64);
}

impl Units for Images {
    fn read(&self, _: usize, queue: u16, unit: usize, nunits: usize) -> Result<u64, String> {
        let units = self
            .iter()
            .find(|(q, _)| *q == queue)
            .map(|(_, u)| u)
            .filter(|u| unit + nunits <= u.len())
            .ok_or_else(|| {
                format!("no unit {unit} of queue {queue} in the initial image (a site in a text block needs the pager)")
            })?;
        Ok(if nunits == 2 {
            (u64::from(units[unit]) << 32) | u64::from(units[unit + 1])
        } else {
            u64::from(units[unit])
        })
    }

    fn write(&mut self, _: usize, queue: u16, unit: usize, nunits: usize, word: u64) {
        let units = &mut self.iter_mut().find(|(q, _)| *q == queue).expect("read found the queue").1;
        if nunits == 2 {
            units[unit] = (word >> 32) as u32;
            units[unit + 1] = word as u32;
        } else {
            units[unit] = word as u32;
        }
    }
}
