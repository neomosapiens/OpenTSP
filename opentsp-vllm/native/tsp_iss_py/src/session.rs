//! One chip behind the binding, with no Python in it: packages, programs, the restart modes and the run of a step.
//! `lib.rs` is the glue.
use crate::paged::{BlockBytes, BlockSpec, Pager};
use crate::patch::{Columns, Images, Table, Values};
use ots_format::weights::Tensor;
use ots_format::{bind_weights, merge_preloads, object_preloads, read_object, read_weights, Extern};
use ots_profile::Profile;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::time::Instant;
use tsp_iss::platform::{Platform, PlatformError, Record};

/// How the SRAM survives from one run to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `Platform::restart`: the SRAM store stays where it is.
    Warm,
    /// The normative sequence: `reset`, then every SRAM word written back through the aperture.
    Replay,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Warm => "warm",
            Mode::Replay => "replay",
        }
    }
}

pub enum Error {
    /// The chip failed or did not finish: a defect of the program or of the model, never a recoverable condition.
    Chip(String),
    UnsafeCompletion(String),
    /// The run raised diagnostics (`TspChipError` with the first one's cycle, queue and code).
    Diagnostics {
        message: String,
        records: Vec<Diagnostic>,
    },
    /// A bad argument: an unknown program, a malformed argument.
    Value(String),
    /// A patch table or a patch value the table refuses (`patch.PatchError` of the Python patcher).
    Patch(String),
    /// A Rust panic caught at the boundary.
    Panic(String),
}

impl From<PlatformError> for Error {
    fn from(e: PlatformError) -> Error {
        Error::Chip(e.to_string())
    }
}

type Result<T> = std::result::Result<T, Error>;

/// A region of words a step returns: slice, first address, word count.
pub type Region = (u8, u32, u32);

/// A diagnostic as Python sees it: cycle, queue, code.
pub type Diagnostic = (u64, u16, String);

struct Program {
    images: Images,
    table: Option<Table>,
    outputs: Vec<Region>,
    /// A paged program: its images and text blocks, patched in place (`paged.rs`).
    pager: Option<Pager>,
    /// What a paged program's text is written from when another bucket's text has replaced it: the text package and the
    /// externs (the blocks are some of them).
    text: Option<(BTreeMap<String, Tensor>, Vec<Extern>)>,
}

/// An extern of a model package: slice, address, words, tensor, package.
type WrittenKey = (u64, u64, u64, String, usize);

/// Where the words of an extern of a program being loaded come from.
enum Source {
    Text,
    Model(WrittenKey),
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub steps: u64,
    pub cycles_total: u64,
    pub cycles_last: u64,
    pub restarts_warm: u64,
    pub restarts_replay: u64,
    pub replay_words: u64,
    pub patches_applied: u64,
    pub extern_words: u64,
    /// Times a paged bucket's whole text was written (the bucket changed), and the words written for it.
    pub text_switches: u64,
    pub text_words: u64,
    /// Words that hold patch sites, written after the patching of a paged step.
    pub patch_words: u64,
    pub run_seconds: f64,
    pub restart_seconds: f64,
    pub load_seconds: f64,
    pub patch_seconds: f64,
}

pub struct Session {
    profile: Profile,
    platform: Platform,
    geom: tsp_iss::Geom,
    vlanes: usize,
    words_per_slice: u64,
    packages: Vec<BTreeMap<String, Tensor>>,
    programs: HashMap<String, Program>,
    /// Externs of a model package already written: (slice, address, words, tensor, package). Weights are read-only for
    /// the programs, so a second program that declares the same extern does not rewrite it.
    written: HashSet<WrittenKey>,
    mode: Mode,
    /// The paged program whose text is in the arena. None whenever the arena may hold something else: another program's text
    /// was written, a `write_words` met a text region, a text write or a step did not finish.
    resident: Option<String>,
    /// The next step panics after its patching (the test of the boundary, `Chip._panic_in_next_step_for_test`).
    pub panic_in_next_step: bool,
    /// The non-zero SRAM words as they were when `sram_changes` last ran.
    synced: HashMap<(u8, u32), Vec<u8>>,
    pub qualification_flips: Vec<(u8, u32, u8, u8)>,
    pub stats: Stats,
    /// The opt-in parametrised replay of the paged programs' steps (`set_param_replay`); the default is the normal driver.
    param_replay: bool,
}

/// Run `f`, turning a panic into [`Error::Panic`].
pub fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => {
            let m = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
            Err(Error::Panic(m.unwrap_or_else(|| "a panic of the simulator".into())))
        }
    }
}

fn asm<T>(r: std::result::Result<T, ots_format::AsmError>) -> Result<T> {
    r.map_err(|e| Error::Value(e.render("")))
}

fn rec(r: &Record) -> Diagnostic {
    (r.cycle, r.queue, r.code.as_str().to_string())
}

impl Session {
    pub fn new(profile: &str) -> Result<Session> {
        let profile = Profile::load(profile).map_err(|e| Error::Value(e.to_string()))?;
        let platform = Platform::new(&profile, 0)?;
        let geom = tsp_iss::Geom::new(&profile);
        let (vlanes, words_per_slice) = (geom.vlanes as usize, geom.words_per_slice);
        Ok(Session {
            profile,
            platform,
            geom,
            vlanes,
            words_per_slice,
            packages: Vec::new(),
            programs: HashMap::new(),
            written: HashSet::new(),
            mode: Mode::Warm,
            resident: None,
            panic_in_next_step: false,
            synced: HashMap::new(),
            qualification_flips: Vec::new(),
            stats: Stats::default(),
            param_replay: false,
        })
    }

    /// Switch the parametrised replay of paged steps on or off (`Platform::set_param_replay`): a paged program's step is then served
    /// from the schedule recorded for its bucket program shape, with the step's site values; the results, diagnostics and cycle
    /// counts are the normal driver's. A binding built without the `replay` feature refuses to switch it on.
    pub fn set_param_replay(&mut self, on: bool) -> Result<()> {
        #[cfg(feature = "replay")]
        {
            self.param_replay = on;
            self.platform.set_param_replay(on);
            if on {
                let names: Vec<String> = self.programs.keys().cloned().collect();
                for name in names {
                    self.register_shape(&name);
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "replay"))]
        {
            let _ = on;
            Err(Error::Value(
                "this binding was built without the `replay` feature (build it with `just vllm-binding`)".into(),
            ))
        }
    }

    pub fn param_replay(&self) -> bool {
        self.param_replay
    }

    /// Tell the platform the shape of the paged program `name`: its patch table, the text of its queues and where the SRAM holds it.
    #[cfg(feature = "replay")]
    fn register_shape(&mut self, name: &str) {
        let Some(program) = self.programs.get(name) else { return };
        let (Some(table), Some(pager)) = (&program.table, &program.pager) else { return };
        let (text, arena) = pager.param_text();
        self.platform.add_param_shape(name, tsp_iss::platform::ParamSpec { sites: table.param_sites(), text, arena });
    }

    /// What the platform's parametrised replay did (`Platform::replay_status`).
    #[cfg(feature = "replay")]
    pub fn replay_status(&self) -> tsp_iss::platform::ParamStatus {
        self.platform.replay_status()
    }

    pub fn vlanes(&self) -> usize {
        self.vlanes
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    pub fn profile_name(&self) -> &str {
        self.profile.name()
    }

    pub fn sram_digest(&self) -> [u8; 32] {
        self.platform.sram_digest()
    }

    pub fn state_digest(&self) -> [u8; 32] {
        self.platform.state_digest()
    }

    pub fn service_ras(&self) -> &tsp_iss::host::ServiceRas {
        self.platform.service_ras()
    }

    /// The restart of the current mode without a run, then the state digest: a test hook (what a restart leaves of the registers is
    /// not visible in a step, which installs what it uses).
    pub fn restart_probe(&mut self) -> Result<[u8; 32]> {
        self.restart()?;
        Ok(self.platform.state_digest())
    }

    /// The words whose contents differ from those at the previous call (the first call: every non-zero word), as
    /// (slice, address, bytes) in slice then address order; a word that became zero is reported with zero bytes.
    /// Applying the changes of every call in order to a fresh chip's SRAM through the aperture rebuilds this SRAM.
    pub fn sram_changes(&mut self) -> Vec<(u8, u32, Vec<u8>)> {
        let now = self.platform.sram_words();
        let mut changes: Vec<(u8, u32, Vec<u8>)> = Vec::new();
        let mut kept: HashMap<(u8, u32), Vec<u8>> = HashMap::with_capacity(now.len());
        for (slice, addr, bytes) in now {
            if self.synced.get(&(slice, addr)) != Some(&bytes) {
                changes.push((slice, addr, bytes.clone()));
            }
            kept.insert((slice, addr), bytes);
        }
        for ((slice, addr), old) in &self.synced {
            if !kept.contains_key(&(*slice, *addr)) {
                changes.push((*slice, *addr, vec![0; old.len()]));
            }
        }
        changes.sort_by_key(|(s, a, _)| (*s, *a));
        self.synced = kept;
        changes
    }

    /// A weight package kept for the externs of the programs loaded afterwards.
    pub fn load_package(&mut self, data: &[u8]) -> Result<usize> {
        let tensors = asm(read_weights(data, Some(self.profile.name()), None))?;
        let n = tensors.len();
        self.packages.push(tensors);
        Ok(n)
    }

    /// Read a host-owned tensor from the validated model package.
    pub fn weight_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self
            .packages
            .iter()
            .find_map(|pkg| pkg.get(name))
            .ok_or_else(|| Error::Value(format!("unknown weight tensor {name:?}")))?;
        Ok(tensor.words().flatten().copied().collect())
    }

    /// Words `addr..addr+n` of a slice as `n * vlanes` bytes.
    pub fn read_words(&mut self, slice: u8, addr: u32, n: u32) -> Result<Vec<u8>> {
        // the range is checked before the buffer is sized: a huge `n` is an error, not an allocation
        self.check_range("read_words", slice, addr, u64::from(n))?;
        let mut out = Vec::with_capacity(n as usize * self.vlanes);
        for k in 0..n {
            out.extend(self.platform.read_sram_word(slice, addr + k)?);
        }
        Ok(out)
    }

    /// Whether words `addr..addr+n` of `slice` overlap a text tensor of the resident paged program.
    fn meets_resident_text(&self, slice: u8, addr: u32, n: u64) -> bool {
        let Some(name) = self.resident.as_deref() else { return false };
        let Some((text, externs)) = self.programs.get(name).and_then(|p| p.text.as_ref()) else {
            return true; // the program is gone or is out for a step: nothing says the arena is still its text
        };
        let (lo, hi) = (u64::from(addr), u64::from(addr) + n);
        externs
            .iter()
            .any(|x| text.contains_key(&x.name) && x.slice == u64::from(slice) && x.addr < hi && lo < x.addr + x.size)
    }

    /// `n` words from `addr` must lie inside the slice.
    fn check_range(&self, what: &str, slice: u8, addr: u32, n: u64) -> Result<()> {
        let end = u64::from(addr) + n;
        if end > self.words_per_slice {
            return Err(Error::Chip(format!(
                "{what}: slice {slice} words {addr:#x}..{end:#x} go beyond the slice end {:#x}",
                self.words_per_slice
            )));
        }
        Ok(())
    }

    /// Whole words from `addr` on; `data` is a whole number of `vlanes`-byte words. All or nothing: a range that
    /// reaches beyond the slice writes no word.
    pub fn write_words(&mut self, slice: u8, addr: u32, data: &[u8]) -> Result<()> {
        if !data.len().is_multiple_of(self.vlanes) {
            return Err(Error::Value(format!(
                "{} bytes are not a whole number of {}-byte words",
                data.len(),
                self.vlanes
            )));
        }
        let words = (data.len() / self.vlanes) as u64;
        self.check_range("write_words", slice, addr, words)?;
        if self.meets_resident_text(slice, addr, words) {
            self.resident = None; // the arena no longer holds that bucket's text: the next step writes it again
        }
        for (k, w) in data.chunks(self.vlanes).enumerate() {
            self.platform.write_sram_word(slice, addr + k as u32, w)?;
        }
        Ok(())
    }

    /// Register a program under `name`: its object, the text package its externs are bound to (written now), the patch
    /// table of its initial images and the regions a step returns (default: the extern named `logits`). The externs
    /// found in the loaded model packages are written too unless `bind_model` is false or they already are.
    // One argument per piece of a program, as the Python signature has them.
    #[allow(clippy::too_many_arguments)]
    pub fn load_program(
        &mut self,
        name: &str,
        elf: &[u8],
        text_package: Option<&[u8]>,
        table: Option<Columns>,
        outputs: Option<Vec<Region>>,
        bind_model: bool,
        blocks: Option<Vec<BlockSpec>>,
        resident_text: bool,
    ) -> Result<()> {
        let started = Instant::now();
        let (_, imgs, externs) = asm(read_object(elf, Some(self.profile.name()), None))?;
        let mut embedded = asm(object_preloads(elf, self.profile.name(), &externs, None, 1))?;
        if let Some(data) = text_package {
            let external = asm(read_weights(data, Some(self.profile.name()), None))?;
            asm(merge_preloads(&mut embedded, external))?;
        }
        let text = (!embedded.is_empty()).then_some(embedded);
        let mut table = table;
        let paging = table.as_mut().and_then(|c| c.paging.take());
        let table = table.map(Table::new).transpose().map_err(Error::Patch)?;
        let outputs = outputs.unwrap_or_else(|| {
            externs
                .iter()
                .filter(|x| x.name == "logits")
                .map(|x| (x.slice as u8, x.addr as u32, x.size as u32))
                .collect()
        });
        let images: Images = imgs.iter().map(|(q, u)| (q, u.to_vec())).collect();
        let pager =
            match (&table, &paging, blocks) {
                (Some(t), Some(p), Some(b)) => {
                    let text =
                        text.as_ref().ok_or_else(|| Error::Value("a paged program needs its text package".into()))?;
                    Some(Pager::new(t, p, &images, b, text, &externs, self.vlanes).map_err(Error::Patch)?)
                }
                (_, Some(_), None) => {
                    return Err(Error::Value("a paged patch table needs the blocks of the program (`blocks=`)".into()))
                }
                (_, None, Some(_)) => return Err(Error::Value(
                    "blocks were given for a patch table that is not paged (no image_queues, image_units, block_units)"
                        .into(),
                )),
                _ => None,
            };
        // the text of a paged program is written when its first step runs, and again whenever another bucket's text has been
        // written since: the buckets share the text arena
        let defer_text = resident_text || pager.is_some();
        let lazy = text.as_ref().filter(|_| defer_text);
        let now = if lazy.is_some() { None } else { text.as_ref() };
        // all or nothing: every extern is checked (its tensor, its words) before the first word is written, so a refused
        // load leaves the SRAM, the written externs and the loaded programs as they were
        if let Some(t) = lazy {
            for x in externs.iter().filter(|x| t.contains_key(&x.name)) {
                self.check_extern(x, t)?;
            }
        }
        let mut planned = HashSet::new();
        let mut plan = Vec::new();
        for x in &externs {
            if let Some(source) = self.plan_extern(x, now, bind_model, &mut planned)? {
                plan.push((x, source));
            }
        }
        if plan.iter().any(|(_, s)| matches!(s, Source::Text)) {
            self.resident = None; // this program's text is written now, over whatever the arena held
        }
        for (x, source) in plan {
            self.write_extern(x, source, now)?;
        }
        let text = text.filter(|_| defer_text).map(|t| (t, externs));
        self.programs.insert(name.to_string(), Program { images, table, outputs, pager, text });
        if self.resident.as_deref() == Some(name) {
            self.resident = None;
        }
        #[cfg(feature = "replay")]
        if self.param_replay {
            self.register_shape(name);
        }
        self.stats.load_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// The tensor of extern `x` in `tensors` has its size, and its words lie in a present slice.
    fn check_extern(&self, x: &Extern, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
        asm(bind_weights(std::slice::from_ref(x), tensors))?;
        let end = x.addr.checked_add(x.size).filter(|&e| e <= self.words_per_slice);
        if !self.geom.slice_present(x.slice) || end.is_none() {
            return Err(Error::Value(format!(
                "extern {}: slice {} words {:#x}+{:#x} are not inside a slice of the profile {} ({:#x} words)",
                x.name,
                x.slice,
                x.addr,
                x.size,
                self.profile.name(),
                self.words_per_slice
            )));
        }
        Ok(())
    }

    /// Where the words of extern `x` come from, checked: the text package, a model package not yet written (its key is
    /// added to `planned`), or nowhere.
    fn plan_extern(
        &self,
        x: &Extern,
        text: Option<&BTreeMap<String, Tensor>>,
        bind_model: bool,
        planned: &mut HashSet<WrittenKey>,
    ) -> Result<Option<Source>> {
        if let Some(t) = text.filter(|t| t.contains_key(&x.name)) {
            self.check_extern(x, t)?;
            return Ok(Some(Source::Text));
        }
        let Some(i) = bind_model.then(|| self.packages.iter().position(|p| p.contains_key(&x.name))).flatten() else {
            return Ok(None);
        };
        let key = (x.slice, x.addr, x.size, x.name.clone(), i);
        if self.written.contains(&key) || planned.contains(&key) {
            return Ok(None);
        }
        self.check_extern(x, &self.packages[i])?;
        planned.insert(key.clone());
        Ok(Some(Source::Model(key)))
    }

    /// The words of a planned extern into the SRAM.
    fn write_extern(&mut self, x: &Extern, source: Source, text: Option<&BTreeMap<String, Tensor>>) -> Result<()> {
        let one = std::slice::from_ref(x);
        let (words, key) = match source {
            Source::Text => (asm(bind_weights(one, text.expect("a text extern is planned with its package")))?, None),
            Source::Model(key) => (asm(bind_weights(one, &self.packages[key.4]))?, Some(key)),
        };
        for w in &words {
            self.platform.write_sram_word(w.slice as u8, w.addr as u32, w.word)?;
        }
        self.stats.extern_words += words.len() as u64;
        self.written.extend(key);
        Ok(())
    }

    /// The state of a run: the SRAM of the last one, the registers of a fresh chip.
    fn restart(&mut self) -> Result<()> {
        let started = Instant::now();
        match self.mode {
            Mode::Warm => {
                self.platform.restart()?;
                self.stats.restarts_warm += 1;
            }
            Mode::Replay => {
                let words = self.platform.sram_words();
                self.platform.reset()?;
                for (slice, addr, bytes) in &words {
                    self.platform.write_sram_word(*slice, *addr, bytes)?;
                }
                self.stats.restarts_replay += 1;
                self.stats.replay_words += words.len() as u64;
            }
        }
        self.stats.restart_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// The text of the paged program `name` into the arena, from its package.
    fn write_text(&mut self, program: &Program, name: &str) -> Result<()> {
        let Some((text, externs)) = &program.text else { return Ok(()) };
        self.resident = None; // set again only when every word is written
        for x in externs.iter().filter(|x| text.contains_key(&x.name)) {
            let words = asm(bind_weights(std::slice::from_ref(x), text))?;
            for w in &words {
                self.platform.write_sram_word(w.slice as u8, w.addr as u32, w.word)?;
            }
            self.stats.text_words += words.len() as u64;
        }
        self.stats.text_switches += 1;
        self.resident = Some(name.to_string());
        Ok(())
    }

    /// Everything of a step before the restart: the values checked and patched in, the text of a paged bucket in the arena
    /// (all of it when the bucket changed, else the words that hold sites). Returns the images to load, when the program is
    /// not paged (the patched copy; a paged program's are held by its pager).
    fn prepare(&mut self, name: &str, program: &mut Program, values: Option<&Values>) -> Result<Option<Images>> {
        let mut owned = None;
        match (&program.table, values) {
            (Some(t), Some(v)) => {
                match program.pager.as_mut() {
                    Some(p) => p.apply(t, v),
                    None => {
                        let mut images = program.images.clone();
                        t.apply(&mut images, v).map(|_| owned = Some(images))
                    }
                }
                .map_err(Error::Patch)?;
                self.stats.patches_applied += t.len() as u64;
            }
            (Some(t), None) if !t.is_empty() => {
                return Err(Error::Patch(format!(
                    "program {name:?} has {} patch sites and no values were given",
                    t.len()
                )))
            }
            (None, Some(_)) => return Err(Error::Value(format!("program {name:?} has no patch table"))),
            _ => {}
        }
        if program.text.is_some() && self.resident.as_deref() != Some(name) {
            self.write_text(program, name)?;
        }
        if let Some(p) = &program.pager {
            for (slice, addr, bytes) in p.site_words() {
                self.platform.write_sram_word(slice, addr, &bytes)?;
            }
            self.stats.patch_words += p.site_word_count() as u64;
        }
        Ok(owned.or_else(|| program.pager.is_none().then(|| program.images.clone())))
    }

    /// The step's patching and text, without the run: the initial images and the text blocks as they are after it (bytes; the
    /// blocks by (queue, index), none for a program that is not paged). A test hook.
    pub fn prepared(&mut self, name: &str, values: Option<&Values>) -> Result<(Images, BlockBytes)> {
        let mut program =
            self.programs.remove(name).ok_or_else(|| Error::Value(format!("no program {name:?} is loaded")))?;
        let r = catch_unwind(AssertUnwindSafe(|| self.prepare(name, &mut program, values)));
        let r = match r {
            Ok(r) => r,
            Err(p) => {
                self.keep_after_panic(name, program);
                resume_unwind(p)
            }
        };
        let out = r.map(|owned| match &program.pager {
            Some(p) => (p.images().clone(), p.block_bytes()),
            None => (owned.unwrap_or_else(|| program.images.clone()), Vec::new()),
        });
        self.programs.insert(name.to_string(), program);
        out
    }

    /// One run of the program `bucket`: the images with `values` patched in, on a restarted chip, to done within
    /// `max_cycles`. Returns the words of the program's output regions and the cycle counter. Diagnostics, a chip fault
    /// and a chip not done by the limit are errors.
    pub fn step(&mut self, bucket: &str, values: Option<&Values>, max_cycles: u64) -> Result<(Vec<u8>, u64)> {
        let started = Instant::now();
        let mut program =
            self.programs.remove(bucket).ok_or_else(|| Error::Value(format!("no program {bucket:?} is loaded")))?;
        // a panic inside the step must not drop the loaded program: put it back, then let the panic go on to `guard`
        match catch_unwind(AssertUnwindSafe(|| self.step_program(bucket, &mut program, values, max_cycles, started))) {
            Ok(r) => {
                self.programs.insert(bucket.to_string(), program);
                r
            }
            Err(p) => {
                self.keep_after_panic(bucket, program);
                resume_unwind(p)
            }
        }
    }

    /// After a panic in the middle of a step: the program stays loaded, and the arena is not known to hold its text.
    fn keep_after_panic(&mut self, name: &str, program: Program) {
        self.programs.insert(name.to_string(), program);
        self.resident = None;
    }

    fn step_program(
        &mut self,
        bucket: &str,
        program: &mut Program,
        values: Option<&Values>,
        max_cycles: u64,
        started: Instant,
    ) -> Result<(Vec<u8>, u64)> {
        let owned = self.prepare(bucket, program, values)?;
        // the values of a paged step, one per site in table order: what the platform serves a step from a recorded schedule with
        let plan = match (self.param_replay, &program.pager, &program.table, values) {
            (true, Some(_), Some(t), Some(v)) => Some(t.dense(v).map_err(Error::Patch)?),
            _ => None,
        };
        if std::mem::take(&mut self.panic_in_next_step) {
            panic!("deliberate panic of _panic_in_next_step_for_test");
        }
        let outputs = program.outputs.clone();
        let images = match (&owned, &program.pager) {
            (Some(i), _) => i,
            (None, Some(p)) => p.images(),
            (None, None) => &program.images,
        };
        self.stats.patch_seconds += started.elapsed().as_secs_f64();
        self.restart()?;
        for (q, units) in images {
            self.platform.select_iq(*q)?;
            self.platform.write_iq_window(units)?;
        }
        for (slice, addr, lane, bit) in self.qualification_flips.drain(..) {
            self.platform.service_flip_sram(slice, addr, lane, bit).map_err(Error::Value)?;
        }
        match &plan {
            #[cfg(feature = "replay")]
            Some(values) => self.platform.start_plan(bucket, values)?,
            _ => self.platform.start()?,
        }
        let ran = Instant::now();
        let done = self.platform.run_to_done(max_cycles);
        self.stats.run_seconds += ran.elapsed().as_secs_f64();
        if let Err(e) = done {
            self.platform.stop();
            return Err(Error::Chip(format!("program {bucket:?}: {e} (limit {max_cycles} cycles)")));
        }
        let cycles = self.platform.cycle();
        self.stats.steps += 1;
        self.stats.cycles_total += cycles;
        self.stats.cycles_last = cycles;
        self.platform.service_scrub();
        let records: Vec<_> = self.platform.diagnostics().iter().map(rec).collect();
        if let Some((cycle, queue, code)) = records.first() {
            let more = if records.len() > 1 { format!(" (and {} more)", records.len() - 1) } else { String::new() };
            return Err(Error::Diagnostics {
                message: format!("program {bucket:?}: {code} on queue {queue:#x} at cycle {cycle}{more}"),
                records,
            });
        }
        if self.platform.service_ras().fatal_count > 0 {
            return Err(Error::UnsafeCompletion(format!(
                "program {bucket:?}: unsafe result in the RAS completion record"
            )));
        }
        let mut out = Vec::new();
        for (slice, addr, n) in outputs {
            out.extend(self.read_words(slice, addr, n)?);
        }
        Ok((out, cycles))
    }
}
