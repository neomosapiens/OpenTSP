//! The patcher of a paged program (`opentsp_tools.patch.apply_paged`): the initial images and the text blocks
//! are held in Rust, a step sets the sites in them in place and writes the words that hold sites into the text arena.
//!
//! Every step gives every site a value (the table is strict), and a site sets only its own bits, so the units after a step do
//! not depend on the ones before it: patching the held units in place is the same as patching a fresh copy, and the words
//! that hold no site never change after the bucket's text has been written.
use crate::patch::{Images, Paging, Table, Units};
use ots_format::weights::Tensor;
use ots_format::Extern;
use std::collections::{BTreeMap, HashMap};

/// The text blocks of a paged program as bytes, by (queue, index).
pub type BlockBytes = Vec<((u16, u32), Vec<u8>)>;

/// A block of a paged queue as `program.json` records it: queue, index in the queue, the tensors (name, slice, address,
/// words) it is made of, in order.
pub struct BlockSpec {
    pub queue: u16,
    pub index: u32,
    pub tensors: Vec<(String, u8, u32, u32)>,
}

#[derive(Clone, Copy)]
enum Place {
    Image(usize),
    Block(usize),
}

pub struct Pager {
    images: Images,
    /// The units of each block, as patched so far.
    blocks: Vec<Vec<u32>>,
    keys: Vec<(u16, u32)>,
    /// Per block, the (slice, address) of each of its words.
    words: Vec<Vec<(u8, u32)>>,
    /// Per site, where its units are.
    loc: Vec<[(Place, usize); 2]>,
    /// The words that hold a site: (block, word in the block), in order.
    site_words: Vec<(usize, usize)>,
    units_per_word: usize,
}

impl Pager {
    /// Locates every site of `table` in `images` and the blocks, as `Paged.where` does, and checks that the blocks are what
    /// the table was made for: whole blocks of `paging.block_units`, one tensor of the text package for each piece at the
    /// extern's place, the blocks of a queue complete from index 0.
    pub fn new(
        table: &Table,
        paging: &Paging,
        images: &Images,
        specs: Vec<BlockSpec>,
        text: &BTreeMap<String, Tensor>,
        externs: &[Extern],
        vlanes: usize,
    ) -> Result<Pager, String> {
        let bu = usize::try_from(paging.block_units)
            .ok()
            .filter(|b| *b > 0)
            .ok_or("paged table: block_units is not positive")?;
        if !vlanes.is_multiple_of(4) || !(4 * bu).is_multiple_of(vlanes) {
            return Err(format!("paged table: a block of {bu} units is not a whole number of {vlanes}-byte words"));
        }
        if paging.image_queues.len() != paging.image_units.len() {
            return Err("paged table: image_queues and image_units differ in length".into());
        }
        let mut image_len: HashMap<u16, usize> = HashMap::new();
        for (q, n) in paging.image_queues.iter().zip(&paging.image_units) {
            let (q, n) = (
                u16::try_from(*q).map_err(|_| "paged table: a queue out of range")?,
                usize::try_from(*n).map_err(|_| "paged table: image units out of range")?,
            );
            image_len.insert(q, n);
        }
        let upw = vlanes / 4;
        let by_name: HashMap<&str, &Extern> = externs.iter().map(|x| (x.name.as_str(), x)).collect();
        let mut specs = specs;
        specs.sort_by_key(|b| (b.queue, b.index));
        let (mut blocks, mut keys, mut words) = (Vec::new(), Vec::new(), Vec::new());
        for b in &specs {
            if keys.last() == Some(&(b.queue, b.index)) {
                return Err(format!("block {} of queue {} is given twice", b.index, b.queue));
            }
            let (mut units, mut ws) = (Vec::with_capacity(bu), Vec::new());
            for (name, slice, addr, n) in &b.tensors {
                let t = text.get(name).ok_or_else(|| {
                    format!("block {} of queue {}: tensor {name} is not in the text package", b.index, b.queue)
                })?;
                if t.len() != *n as usize {
                    return Err(format!(
                        "block {} of queue {}: tensor {name} has {} words, the block record {n}",
                        b.index,
                        b.queue,
                        t.len()
                    ));
                }
                match by_name.get(name.as_str()) {
                    Some(x) if x.slice == u64::from(*slice) && x.addr == u64::from(*addr) && x.size == u64::from(*n) => {}
                    Some(x) => {
                        return Err(format!(
                            "block {} of queue {}: tensor {name} is at slice {} address {:#x} in the object, {slice} {addr:#x} in the block record",
                            b.index, b.queue, x.slice, x.addr
                        ))
                    }
                    None => return Err(format!("block {} of queue {}: tensor {name} is not an extern of the program", b.index, b.queue)),
                }
                for (k, w) in t.words().enumerate() {
                    units.extend(w.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))));
                    ws.push((*slice, *addr + k as u32));
                }
            }
            if units.len() != bu {
                return Err(format!("block {} of queue {} has {} units, not {bu}", b.index, b.queue, units.len()));
            }
            keys.push((b.queue, b.index));
            blocks.push(units);
            words.push(ws);
        }
        let mut next: HashMap<u16, u32> = HashMap::new();
        for (q, j) in &keys {
            let want = next.entry(*q).or_insert(0);
            if j != want {
                return Err(format!("queue {q}: block {want} is missing (the next block given is {j})"));
            }
            *want += 1;
            match images.iter().find(|(iq, _)| iq == q) {
                Some((_, u)) if image_len.get(q) == Some(&u.len()) => {}
                _ => {
                    return Err(format!(
                        "queue {q}: its initial image is not the one of {:?} units the sites were located in",
                        image_len.get(q)
                    ))
                }
            }
        }
        let block_at: HashMap<(u16, u32), usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let image_at: HashMap<u16, usize> = images.iter().enumerate().map(|(i, (q, _))| (*q, i)).collect();
        let mut loc = Vec::with_capacity(table.len());
        let mut site_words = Vec::new();
        for i in 0..table.len() {
            let (q, unit, n) = table.position(i);
            let mut at = [(Place::Image(0), 0); 2];
            for (k, slot) in at.iter_mut().enumerate().take(n) {
                let logical = unit + k;
                *slot = match image_len.get(&q) {
                    Some(&il) if logical >= il => {
                        let (j, u) = (((logical - il) / bu) as u32, (logical - il) % bu);
                        let b = block_at
                            .get(&(q, j))
                            .ok_or_else(|| format!("site {}: block {j} of queue {q} does not exist", table.name(i)))?;
                        site_words.push((*b, u / upw));
                        (Place::Block(*b), u)
                    }
                    _ => {
                        let im = image_at
                            .get(&q)
                            .ok_or_else(|| format!("site {}: queue {q} has no initial image", table.name(i)))?;
                        if logical >= images[*im].1.len() {
                            return Err(format!(
                                "site {}: unit {logical} is beyond the initial image of queue {q}",
                                table.name(i)
                            ));
                        }
                        (Place::Image(*im), logical)
                    }
                };
            }
            loc.push(at);
        }
        site_words.sort_unstable();
        site_words.dedup();
        Ok(Pager { images: images.clone(), blocks, keys, words, loc, site_words, units_per_word: upw })
    }

    /// What the platform's parametrised replay needs to know of this program's text: the whole text of every queue (initial image,
    /// then the blocks in order) as patched now, and where the SRAM holds it (runs of consecutive words).
    #[cfg(feature = "replay")]
    pub fn param_text(
        &self,
    ) -> (Vec<tsp_iss::driver::replay_param::QueueText>, Vec<tsp_iss::driver::replay_param::Run>) {
        use tsp_iss::driver::replay_param::{QueueText, Run};
        let mut text: BTreeMap<u16, Vec<u32>> = self.images.iter().map(|(q, u)| (*q, u.clone())).collect();
        let mut runs = Vec::new();
        for (b, (q, _)) in self.keys.iter().enumerate() {
            let units = text.entry(*q).or_default();
            let start = units.len();
            units.extend_from_slice(&self.blocks[b]);
            let words = &self.words[b];
            let mut k = 0;
            while k < words.len() {
                let (slice, addr) = words[k];
                let mut n = 1;
                while k + n < words.len() && words[k + n] == (slice, addr + n as u32) {
                    n += 1;
                }
                runs.push(Run { queue: *q, start: start + k * self.units_per_word, slice, addr, words: n as u32 });
                k += n;
            }
        }
        (text.into_iter().map(|(queue, units)| QueueText { queue, units }).collect(), runs)
    }

    pub fn images(&self) -> &Images {
        &self.images
    }

    /// The text blocks as bytes, by (queue, index).
    pub fn block_bytes(&self) -> BlockBytes {
        self.keys
            .iter()
            .zip(&self.blocks)
            .map(|(k, u)| (*k, u.iter().flat_map(|x| x.to_le_bytes()).collect()))
            .collect()
    }

    /// The (slice, address, bytes) of every word that holds a site, with the blocks as patched.
    pub fn site_words(&self) -> impl Iterator<Item = (u8, u32, Vec<u8>)> + '_ {
        self.site_words.iter().map(move |&(b, w)| {
            let (slice, addr) = self.words[b][w];
            let units = &self.blocks[b][w * self.units_per_word..(w + 1) * self.units_per_word];
            (slice, addr, units.iter().flat_map(|x| x.to_le_bytes()).collect())
        })
    }

    pub fn site_word_count(&self) -> usize {
        self.site_words.len()
    }

    /// Patches the sites in place. The range checks of all values run before the first write (`Table::apply_to`), but a unit that
    /// refuses its site (an opcode or length mismatch) after earlier sites were written leaves the pager PARTLY patched, unlike
    /// the copy `Table::apply` works on for a program that is not paged. That is harmless: every step gives every site a value
    /// (strict) and a site sets only its own bits, so the next step overwrites whatever was left; the step that failed ran
    /// nothing.
    pub fn apply(&mut self, table: &Table, values: &crate::patch::Values) -> Result<(), String> {
        table.apply_to(self, values)
    }

    fn units(&mut self, p: Place) -> &mut Vec<u32> {
        match p {
            Place::Image(i) => &mut self.images[i].1,
            Place::Block(b) => &mut self.blocks[b],
        }
    }
}

impl Units for Pager {
    fn read(&self, site: usize, _: u16, _: usize, nunits: usize) -> Result<u64, String> {
        let at = |k: usize| {
            let (p, u) = self.loc[site][k];
            u64::from(match p {
                Place::Image(i) => self.images[i].1[u],
                Place::Block(b) => self.blocks[b][u],
            })
        };
        Ok(if nunits == 2 { (at(0) << 32) | at(1) } else { at(0) })
    }

    fn write(&mut self, site: usize, _: u16, _: usize, nunits: usize, word: u64) {
        let loc = self.loc[site];
        let parts: [u32; 2] = if nunits == 2 { [(word >> 32) as u32, word as u32] } else { [word as u32, 0] };
        for k in 0..nunits {
            let (p, u) = loc[k];
            self.units(p)[u] = parts[k];
        }
    }
}
