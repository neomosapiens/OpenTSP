//! `opentsp_vllm._tsp_iss`: the PyO3 binding of the ISS (binding API
//! §5.6). One `Chip` is one `Platform` plus the host's view of what is loaded on it. The run of a step releases the GIL.
//!
//! A diagnostic, a chip fault, a chip not done by the cycle limit and a Rust panic all raise `TspChipError`
//! (`opentsp_vllm.chip.TspChipError` when the package is importable, else `_tsp_iss.TspIssError`) with the attributes
//! `cycle`, `queue`, `code` (the first diagnostic; `None` for the others) and `diagnostics` (all of them).
mod paged;
mod patch;
mod session;
mod system;

use paged::BlockSpec;
use patch::{Columns, Paging, Values};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBool, PyBytes, PyDict, PyInt};
use session::{guard, Diagnostic, Error, Mode, Region, Session};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use system::{Links, Reason, RunOutcome};

pyo3::create_exception!(_tsp_iss, TspIssError, PyRuntimeError, "The chip failed, or a run raised a diagnostic.");
pyo3::create_exception!(
    _tsp_iss,
    PatchError,
    PyValueError,
    "A patch table or a patch value the table refuses: the counterpart of `opentsp_tools.patch.PatchError`, and a `ValueError`."
);

fn ras_snapshot<'py>(py: Python<'py>, ras: &tsp_iss::host::ServiceRas) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("events", ras.events)?;
    out.set_item("causes", ras.causes)?;
    out.set_item("fatal_count", ras.fatal_count)?;
    out.set_item("overflow", ras.overflow)?;
    out.set_item("first", ras.first)?;
    out.set_item("first_fatal", ras.first_fatal)?;
    out.set_item("sram_word", ras.sram_word)?;
    out.set_item("stored_report", ras.stored_report)?;
    Ok(out)
}

fn chip_error_class(py: Python<'_>) -> Bound<'_, PyAny> {
    py.import("opentsp_vllm.chip")
        .and_then(|m| m.getattr("TspChipError"))
        .unwrap_or_else(|_| py.get_type::<TspIssError>().into_any())
}

/// The class to raise: the package's `TspChipError`, so a caller handles one type whichever chip it runs.
fn chip_error(py: Python<'_>, message: String, records: Vec<Diagnostic>) -> PyErr {
    let built = chip_error_class(py).call1((message.as_str(),)).and_then(|e| {
        let (cycle, queue, code) = match records.first() {
            Some((c, q, k)) => (Some(*c), Some(*q), Some(k.clone())),
            None => (None, None, None),
        };
        e.setattr("cycle", cycle)?;
        e.setattr("queue", queue)?;
        e.setattr("code", code)?;
        e.setattr("diagnostics", records)?;
        Ok(PyErr::from_value(e))
    });
    built.unwrap_or_else(|e| e)
}

fn to_pyerr(py: Python<'_>, e: Error) -> PyErr {
    match e {
        Error::UnsafeCompletion(m) => {
            let error = chip_error(py, m, Vec::new());
            if let Err(e) = error.value(py).setattr("ras_completion", true) {
                return e;
            }
            error
        }
        Error::Chip(m) => chip_error(py, m, Vec::new()),
        Error::Diagnostics { message, records } => chip_error(py, message, records),
        Error::Panic(m) => chip_error(py, format!("panic in the simulator: {m}"), Vec::new()),
        Error::Value(m) => PyValueError::new_err(m),
        Error::Patch(m) => PatchError::new_err(m),
    }
}

/// Bytes or the path of a file.
#[derive(FromPyObject)]
enum Source {
    Bytes(PyBackedBytes),
    Path(PathBuf),
}

impl Source {
    fn read(&self) -> Result<Vec<u8>, Error> {
        match self {
            Source::Bytes(b) => Ok(b.to_vec()),
            Source::Path(p) => std::fs::read(p).map_err(|e| Error::Value(format!("{}: {e}", p.display()))),
        }
    }
}

/// An integer patch value: a Python `int` or anything with `__index__` (numpy integers), a bool excepted, as
/// `patch.apply` takes any `numbers.Integral` but not a bool. A value beyond 128 bits saturates: it is out of every range.
fn index_of(name: &str, x: &Bound<'_, PyAny>) -> PyResult<i128> {
    if x.is_instance_of::<PyBool>() {
        return Err(PatchError::new_err(format!("site {name}: value {x} is not an integer")));
    }
    // under the limited API the integer extraction does not call `__index__` itself: do it
    let int =
        if x.is_instance_of::<PyInt>() || !x.hasattr("__index__")? { x.clone() } else { x.call_method0("__index__")? };
    match int.extract::<i128>() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyOverflowError>(x.py()) => Ok(if x.lt(0)? { i128::MIN } else { i128::MAX }),
        Err(_) => Err(PatchError::new_err(format!("site {name}: value {x} is not an integer"))),
    }
}

/// The values of a step: `{site name: int}` or int32 little-endian bytes, one per site in table order. A value that is
/// not an integer (a float, a bool, a string) or a name that is not a string is a `PatchError`, as in `patch.apply`.
fn values_of(v: &Bound<'_, PyAny>) -> PyResult<Values> {
    if let Ok(b) = v.extract::<PyBackedBytes>() {
        if b.len() % 4 != 0 {
            return Err(PatchError::new_err("dense patch values are int32 little-endian: a multiple of 4 bytes"));
        }
        return Ok(Values::Dense(
            b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes"))).collect(),
        ));
    }
    let d = v
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("patch_values: a dict {site name: int}, int32 bytes or None"))?;
    let mut out = Vec::with_capacity(d.len());
    for (k, x) in d.iter() {
        let name: String =
            k.extract().map_err(|_| PatchError::new_err(format!("patch site name {k} is not a string")))?;
        let value = index_of(&name, &x)?;
        out.push((name, value));
    }
    Ok(Values::Named(out))
}

/// `patch.Table`'s columns as a dict of lists (`np.load` of `bindings.npz` gives the arrays; `.tolist()` them).
fn columns(t: &Bound<'_, PyDict>) -> PyResult<Columns> {
    fn col<'py, T: FromPyObjectOwned<'py>>(t: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
        let v = t.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("patch table: no column {key:?}")))?;
        v.extract::<T>().map_err(|e| PyValueError::new_err(format!("patch table: column {key:?}: {}", e.into())))
    }
    Ok(Columns {
        names: col(t, "names")?,
        queue: col(t, "queue")?,
        unit: col(t, "unit")?,
        nunits: col(t, "nunits")?,
        lo: col(t, "lo")?,
        width: col(t, "width")?,
        vmin: col(t, "vmin")?,
        vmax: col(t, "vmax")?,
        placeholder: col(t, "placeholder")?,
        fixed: col(t, "fixed")?,
        stride: col(t, "stride")?,
        span: col(t, "span")?,
        paging: if t.contains("block_units")? {
            Some(Paging {
                image_queues: col(t, "image_queues")?,
                image_units: col(t, "image_units")?,
                block_units: col(t, "block_units")?,
            })
        } else {
            None
        },
    })
}

/// The `replay` entry of `Chip.stats()`: whether the parametrised replay is built in and on, the totals and one row per shape.
fn replay_stats<'py>(py: Python<'py>, session: &Session) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("built", cfg!(feature = "replay"))?;
    d.set_item("enabled", session.param_replay())?;
    #[cfg(feature = "replay")]
    {
        let st = session.replay_status();
        for (k, v) in
            [("served", st.served), ("refused", st.refused), ("fallback", st.fallback), ("normal_runs", st.normal_runs)]
        {
            d.set_item(k, v)?;
        }
        d.set_item("min_gain", st.min_gain)?;
        let shapes = PyDict::new(py);
        for s in &st.shapes {
            let row = PyDict::new(py);
            row.set_item("state", s.state)?;
            row.set_item("why", s.why.as_ref().map(|w| format!("{w:?}")))?;
            for (k, v) in [
                ("served", s.served),
                ("refused", s.refused),
                ("fallback", s.fallback),
                ("normal_runs", s.normal_runs),
                ("retained_bytes", s.retained_bytes),
            ] {
                row.set_item(k, v)?;
            }
            row.set_item("gain", s.gain)?;
            row.set_item("normal_us", s.normal_us)?;
            row.set_item("replay_us", s.replay_us)?;
            shapes.set_item(&s.shape, row)?;
        }
        d.set_item("shapes", shapes)?;
    }
    Ok(d)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHA-256 of a shipped profile's file as lower-case hex: the `profile_sha256` of a program cache entry.
#[pyfunction]
fn profile_sha256(py: Python<'_>, profile: &str) -> PyResult<String> {
    let p = guard(|| ots_profile::Profile::load(profile).map_err(|e| Error::Value(e.to_string())))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(hex(&Sha256::digest(p.bytes())))
}

/// The text blocks `load_program` takes: `(queue, index, [(tensor, slice, address, words)])`.
type BlockArg = Vec<(u16, u32, Vec<(String, u8, u32, u32)>)>;

/// One chip of a profile (`"ref"`, `"mini"`).
#[pyclass(unsendable)]
struct Chip {
    session: Session,
}

#[pymethods]
impl Chip {
    #[new]
    fn new(py: Python<'_>, profile: &str) -> PyResult<Chip> {
        guard(|| Session::new(profile)).map(|session| Chip { session }).map_err(|e| to_pyerr(py, e))
    }

    /// The profile's name.
    #[getter]
    fn profile(&self, py: Python<'_>) -> PyResult<String> {
        guard(|| Ok(self.session.profile_name().to_string())).map_err(|e| to_pyerr(py, e))
    }

    /// The SHA-256 of the profile's file, as hex (`profile_sha256()` of the module for this chip's profile).
    #[getter]
    fn profile_sha256(&self, py: Python<'_>) -> PyResult<String> {
        profile_sha256(py, self.session.profile_name())
    }

    /// Bytes per SRAM word.
    #[getter]
    fn vlanes(&self) -> usize {
        self.session.vlanes()
    }

    /// Read an `.safetensors` (bytes or a path) and keep its tensors for the externs of the programs loaded after it.
    /// Returns the number of tensors.
    fn load_package(&mut self, py: Python<'_>, package: Source) -> PyResult<usize> {
        guard(|| self.session.load_package(&package.read()?)).map_err(|e| to_pyerr(py, e))
    }

    /// Host embedding tables come from the same validated package as the device weights.
    fn weight_tensor<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.session.weight_tensor(name).map_err(|e| to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Register the program `name`: an `.elf` (bytes or a path), the text package its externs are bound to, the patch
    /// table (`bindings.npz`'s columns as lists) and the output regions `[(slice, addr, words)]` (default: the extern
    /// `logits`). Externs bound to a package are written to SRAM now. A paged program (a table with `image_queues`,
    /// `image_units`, `block_units`) also takes its `blocks=[(queue, index, [(tensor, slice, addr, words)])]`
    /// (`program.json`): its text is written to the arena by the first step that runs it, and the words of the blocks that hold
    /// sites by every step. With `resident_text=True`, immutable stage text is restored on program switches.
    /// Such stages can run with no patch table or values.
    // The Python signature is the interface: one argument per piece of a program.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, elf, text_package=None, patch_table=None, outputs=None, bind_model=true, blocks=None, resident_text=false))]
    fn load_program(
        &mut self,
        py: Python<'_>,
        name: &str,
        elf: Source,
        text_package: Option<Source>,
        patch_table: Option<Bound<'_, PyDict>>,
        outputs: Option<Vec<Region>>,
        bind_model: bool,
        blocks: Option<BlockArg>,
        resident_text: bool,
    ) -> PyResult<()> {
        let table = patch_table.as_ref().map(columns).transpose()?;
        let blocks =
            blocks.map(|b| b.into_iter().map(|(queue, index, tensors)| BlockSpec { queue, index, tensors }).collect());
        let r = guard(|| {
            let elf = elf.read()?;
            let text = text_package.as_ref().map(Source::read).transpose()?;
            self.session.load_program(name, &elf, text.as_deref(), table, outputs, bind_model, blocks, resident_text)
        });
        r.map_err(|e| to_pyerr(py, e))
    }

    /// Run the program `bucket` with `patch_values` patched into its images, on a restarted chip, to done within
    /// `max_cycles`: `(output words as bytes, cycles, diagnostics)`. Diagnostics raise `TspChipError`, so the third
    /// element is `()` for a run that returns.
    #[pyo3(signature = (bucket, patch_values=None, max_cycles=1_000_000_000))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        bucket: &str,
        patch_values: Option<Bound<'py, PyAny>>,
        max_cycles: u64,
    ) -> PyResult<(Bound<'py, PyBytes>, u64, Vec<Diagnostic>)> {
        let values = patch_values.as_ref().filter(|v| !v.is_none()).map(|v| values_of(v)).transpose()?;
        let session = &mut self.session;
        let r = py.detach(|| guard(|| session.step(bucket, values.as_ref(), max_cycles)));
        let (words, cycles) = r.map_err(|e| to_pyerr(py, e))?;
        Ok((PyBytes::new(py, &words), cycles, Vec::new()))
    }

    /// Schedule one F sram bit flip after the next stage has loaded its operands.
    fn inject_sram_next_step(&mut self, slice: u8, addr: u32, lane: u8, bit: u8) {
        self.session.qualification_flips.push((slice, addr, lane, bit));
    }

    fn ras_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        ras_snapshot(py, self.session.service_ras())
    }

    /// `"warm"` (`Platform::restart`) or `"replay"` (reset and write every SRAM word back through the aperture);
    /// without an argument it only reports. Returns the mode in force.
    #[pyo3(signature = (mode=None))]
    fn restart_mode(&mut self, mode: Option<&str>) -> PyResult<&'static str> {
        match mode {
            None => {}
            Some("warm") => self.session.set_mode(Mode::Warm),
            Some("replay") => self.session.set_mode(Mode::Replay),
            Some(other) => {
                return Err(PyValueError::new_err(format!("restart mode {other:?}: expected \"warm\" or \"replay\"")))
            }
        }
        Ok(self.session.mode().name())
    }

    /// Opt in to (or out of) the parametrised schedule replay of paged steps (default: off, the normal driver). A step is then served
    /// from the schedule recorded for its bucket program shape, with the step's site values; the logits, the diagnostics and the cycle
    /// counts are the normal driver's, and a run the gate refuses, or a fault during a run, takes the normal driver. Raises
    /// `ValueError` when the binding was built without the `replay` feature. Without an argument it only reports. Returns the setting.
    #[pyo3(signature = (on=None))]
    fn param_replay(&mut self, py: Python<'_>, on: Option<bool>) -> PyResult<bool> {
        if let Some(on) = on {
            guard(|| self.session.set_param_replay(on)).map_err(|e| to_pyerr(py, e))?;
        }
        Ok(self.session.param_replay())
    }

    /// `n` words of `slice` from `addr` as `n * vlanes` bytes (lane 0 first).
    fn read_words<'py>(&mut self, py: Python<'py>, slice: u8, addr: u32, n: u32) -> PyResult<Bound<'py, PyBytes>> {
        let words = guard(|| self.session.read_words(slice, addr, n)).map_err(|e| to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &words))
    }

    /// Write whole words (`len(data)` a multiple of `vlanes`) from `addr` on.
    fn write_words(&mut self, py: Python<'_>, slice: u8, addr: u32, data: PyBackedBytes) -> PyResult<()> {
        guard(|| self.session.write_words(slice, addr, &data)).map_err(|e| to_pyerr(py, e))
    }

    /// The SHA-256 of the SRAM contents (non-zero words only: the same value whichever pages are allocated).
    fn sram_digest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let digest = guard(|| Ok(self.session.sram_digest())).map_err(|e| to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &digest))
    }

    /// The SHA-256 of all chip state except the SRAM (registers, IQs, SRF, MXM, SXM, clock, diagnostics): after a restart it is a
    /// new chip's, and the same on a warm and a replay chip after the same program.
    fn state_digest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let digest = guard(|| Ok(self.session.state_digest())).map_err(|e| to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &digest))
    }

    /// The restart of the current mode without a run, returning the state digest (not part of the interface): what a restart leaves
    /// of the registers cannot be seen in a step, which installs what it uses.
    #[doc(hidden)]
    fn _restart_for_test<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let digest = guard(|| self.session.restart_probe()).map_err(|e| to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &digest))
    }

    /// The SRAM words that differ from those at the previous call (the first call: every non-zero word) as
    /// `[(slice, addr, bytes)]`, a word that became zero with zero bytes. The host's record of a program's effect on the SRAM: writing
    /// the changes of every call in order into a new chip rebuilds this chip's SRAM (`sram_digest` equal).
    fn sram_changes<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<(u8, u32, Bound<'py, PyBytes>)>> {
        let changes = guard(|| Ok(self.session.sram_changes())).map_err(|e| to_pyerr(py, e))?;
        Ok(changes.into_iter().map(|(s, a, b)| (s, a, PyBytes::new(py, &b))).collect())
    }

    /// Counters and wall times since construction.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = guard(|| Ok(self.session.stats.clone())).map_err(|e| to_pyerr(py, e))?;
        let d = PyDict::new(py);
        d.set_item("mode", self.session.mode().name())?;
        for (k, v) in [
            ("steps", s.steps),
            ("cycles_total", s.cycles_total),
            ("cycles_last", s.cycles_last),
            ("restarts_warm", s.restarts_warm),
            ("restarts_replay", s.restarts_replay),
            ("replay_words", s.replay_words),
            ("patches_applied", s.patches_applied),
            ("extern_words", s.extern_words),
            ("text_switches", s.text_switches),
            ("text_words", s.text_words),
            ("patch_words", s.patch_words),
        ] {
            d.set_item(k, v)?;
        }
        for (k, v) in [
            ("run_seconds", s.run_seconds),
            ("restart_seconds", s.restart_seconds),
            ("load_seconds", s.load_seconds),
            ("patch_seconds", s.patch_seconds),
        ] {
            d.set_item(k, v)?;
        }
        d.set_item("replay", replay_stats(py, &self.session)?)?;
        Ok(d)
    }

    /// The patching and the text of a step without its run: `(images, blocks)`, the initial image of each queue and each text block
    /// of a paged program as bytes, after `patch_values` were patched in and the arena written (not part of the interface).
    #[doc(hidden)]
    #[pyo3(signature = (bucket, patch_values=None))]
    fn _prepare_for_test<'py>(
        &mut self,
        py: Python<'py>,
        bucket: &str,
        patch_values: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let values = patch_values.as_ref().filter(|v| !v.is_none()).map(|v| values_of(v)).transpose()?;
        let (images, blocks) = guard(|| self.session.prepared(bucket, values.as_ref())).map_err(|e| to_pyerr(py, e))?;
        let (di, db) = (PyDict::new(py), PyDict::new(py));
        for (q, units) in images {
            let bytes: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
            di.set_item(q, PyBytes::new(py, &bytes))?;
        }
        let blocks: BTreeMap<(u16, u32), Vec<u8>> = blocks.into_iter().collect();
        for (k, b) in blocks {
            db.set_item(k, PyBytes::new(py, &b))?;
        }
        Ok((di, db))
    }

    /// The next `step` panics after its patching and text writes: the test that the program stays loaded (not part of the
    /// interface).
    fn _panic_in_next_step_for_test(&mut self) {
        self.session.panic_in_next_step = true;
    }

    /// Raises the failure a Rust panic becomes: the test of the boundary (not part of the interface).
    #[doc(hidden)]
    fn _panic_for_test(&mut self, py: Python<'_>) -> PyResult<()> {
        guard(|| -> Result<(), Error> { panic!("deliberate panic of _panic_for_test") }).map_err(|e| to_pyerr(py, e))
    }
}

/// The error of a system run that reported diagnostics: `TspChipError` with `cycle`, `queue`, `code` of the first, `chip`
/// (the chip that reported it), `diagnostics` as `(cycle, queue, code)` and `chip_diagnostics` as `(chip, cycle, queue, code)`.
fn system_error(py: Python<'_>, found: Vec<(usize, Diagnostic)>) -> PyErr {
    let (chip, (cycle, queue, code)) = found.first().cloned().expect("a diagnostic was reported");
    let more = if found.len() > 1 { format!(" (and {} more)", found.len() - 1) } else { String::new() };
    let message = format!("chip {chip}: {code} on queue {queue:#x} at cycle {cycle}{more}");
    let records: Vec<Diagnostic> = found.iter().map(|(_, d)| d.clone()).collect();
    let built = chip_error_class(py).call1((message.as_str(),)).and_then(|e| {
        e.setattr("cycle", cycle)?;
        e.setattr("queue", queue)?;
        e.setattr("code", code)?;
        e.setattr("chip", chip)?;
        e.setattr("diagnostics", records)?;
        e.setattr(
            "chip_diagnostics",
            found.iter().map(|(c, (cy, q, k))| (*c, *cy, *q, k.clone())).collect::<Vec<_>>(),
        )?;
        Ok(PyErr::from_value(e))
    });
    built.unwrap_or_else(|e| e)
}

/// The result of a run as a dict: `reason` (`"done"`, `"notify"`, `"diagnostic"`, `"limit"`), `cycle` (where every chip
/// stands), `notifies` `[(chip, cycle, vector)]` and `diagnostics` `[(chip, cycle, queue, code)]`, both since the last run.
fn outcome_dict<'py>(py: Python<'py>, o: RunOutcome) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("reason", o.reason.name())?;
    d.set_item("cycle", o.cycle)?;
    d.set_item("notifies", o.notifies)?;
    d.set_item("diagnostics", o.diagnostics.into_iter().map(|(c, (cy, q, k))| (c, cy, q, k)).collect::<Vec<_>>())?;
    Ok(d)
}

/// N chips of one profile wired as `tsp_sim --chips N` wires them, run in lock step, each with its own host interface
/// (the host's DMA engine and ingress FIFO, CSR space, doorbells, interrupts, and the vectors PciSend gives the host).
/// `link_latency`, `link_latency_seed` and `link_jitter_seed` are the link options of `tsp_sim` (at most one).
/// `window` (default: the smallest link latency, as `tsp_sim`) is a shorter lock-step window, `tsp_sim --window N`: the results do not depend on
/// it, and a run that stops at an interrupt or a limit stands at most `window` cycles after the event, so the host can act close to it.
/// With `trace` the `T srf` and `T sram` events of every chip are kept (`trace(chip)`).
#[pyclass(unsendable, subclass)]
struct System {
    inner: system::System,
}

#[pymethods]
impl System {
    // The Python signature is the interface: one argument per option of the system.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (profile, chips=1, link_latency=None, link_latency_seed=None, link_jitter_seed=None, window=None, trace=false))]
    fn new(
        py: Python<'_>,
        profile: &str,
        chips: usize,
        link_latency: Option<u64>,
        link_latency_seed: Option<u64>,
        link_jitter_seed: Option<u64>,
        window: Option<u64>,
        trace: bool,
    ) -> PyResult<System> {
        let links = match (link_latency, link_latency_seed, link_jitter_seed) {
            (None, None, None) => Links::Nominal,
            (Some(n), None, None) => Links::Fixed(n),
            (None, Some(s), None) => Links::Seed(s),
            (None, None, Some(s)) => Links::Jitter(s),
            _ => {
                return Err(PyValueError::new_err(
                    "link_latency, link_latency_seed and link_jitter_seed exclude one another",
                ))
            }
        };
        guard(|| system::System::new(profile, chips, links, window, trace))
            .map(|inner| System { inner })
            .map_err(|e| to_pyerr(py, e))
    }

    #[getter]
    fn chips(&self) -> usize {
        self.inner.chips()
    }

    #[getter]
    fn profile(&self) -> String {
        self.inner.profile_name().to_string()
    }

    /// Bytes per SRAM word and per vector.
    #[getter]
    fn vlanes(&self) -> usize {
        self.inner.vlanes()
    }

    /// The lock-step window: the smallest connected link latency.
    #[getter]
    fn window(&self) -> u64 {
        self.inner.window()
    }

    /// The cycle every chip stands at (every cycle below it is ticked).
    #[getter]
    fn cycle(&self) -> u128 {
        self.inner.cycle()
    }

    /// The object (an `.elf`, bytes or a path) of chip `chip`, before the first run.
    fn load_object(&mut self, py: Python<'_>, chip: usize, elf: Source) -> PyResult<()> {
        guard(|| self.inner.load_object(chip, &elf.read()?)).map_err(|e| to_pyerr(py, e))
    }

    /// The restart of every chip (`platform.md` section 1.1) with the SRAM kept: cycle 0, no diagnostics, an empty host interface, the objects loaded again. The next
    /// `run` is the run of a new system whose SRAM holds what it held now; the host writes the words that change and the vectors of the next run.
    fn restart(&mut self, py: Python<'_>) -> PyResult<()> {
        guard(|| self.inner.restart()).map_err(|e| to_pyerr(py, e))
    }

    /// The objects of a manifest (an `.otsm`, bytes or a path) on its chips, in chip order; its chip count is the system's.
    fn load_manifest(&mut self, py: Python<'_>, manifest: Source) -> PyResult<()> {
        let r = guard(|| {
            let data = manifest.read()?;
            let (_, objects) = ots_format::manifest::read_manifest(&data, Some(self.inner.profile_name()), None)
                .map_err(|e| Error::Value(e.render("")))?;
            if objects.len() != self.inner.chips() {
                return Err(Error::Value(format!(
                    "the manifest holds {} objects, the system has {} chips",
                    objects.len(),
                    self.inner.chips()
                )));
            }
            for (c, o) in objects.iter().enumerate() {
                self.inner.load_object(c, o)?;
            }
            Ok(())
        });
        r.map_err(|e| to_pyerr(py, e))
    }

    /// `n` words of `slice` from `addr` of chip `chip` as `n * vlanes` bytes.
    fn read_words<'py>(
        &self,
        py: Python<'py>,
        chip: usize,
        slice: u8,
        addr: u32,
        n: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let w = guard(|| self.inner.read_words(chip, slice, addr, n)).map_err(|e| to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &w))
    }

    /// Whole words into the SRAM of chip `chip` from `addr` on.
    fn write_words(&mut self, py: Python<'_>, chip: usize, slice: u8, addr: u32, data: PyBackedBytes) -> PyResult<()> {
        guard(|| self.inner.write_words(chip, slice, addr, &data)).map_err(|e| to_pyerr(py, e))
    }

    /// The host's DMA engine of chip `chip` takes one vector (`vlanes` bytes), requested at `cycle` (default: now). It
    /// enters the ingress FIFO paced by `host.cycles_per_vector`, and waits while the FIFO is full: nothing is dropped
    /// (`dma_events`, `ingress`). Returns the cycle of the request.
    #[pyo3(signature = (chip, vector, cycle=None))]
    fn dma_write(&mut self, py: Python<'_>, chip: usize, vector: PyBackedBytes, cycle: Option<u64>) -> PyResult<u64> {
        guard(|| self.inner.dma_write(chip, &vector, cycle)).map_err(|e| to_pyerr(py, e))
    }

    /// Queue text for one chip at an absolute management-supplied cycle (the harness H inject operation).
    fn inject_text(&mut self, py: Python<'_>, chip: usize, cycle: u64, images: Vec<(u16, Vec<u32>)>) -> PyResult<()> {
        guard(|| self.inner.inject_text(chip, cycle, images)).map_err(|e| to_pyerr(py, e))
    }

    /// A write of CSR `addr` by the host of chip `chip` at `cycle` (default: now). Returns the cycle.
    #[pyo3(signature = (chip, addr, value, cycle=None))]
    fn csr_write(&mut self, py: Python<'_>, chip: usize, addr: u64, value: u64, cycle: Option<u64>) -> PyResult<u64> {
        guard(|| self.inner.csr_write(chip, addr, value, cycle)).map_err(|e| to_pyerr(py, e))
    }

    /// The host rings doorbell `index` of chip `chip` at `cycle` (default: now): `csr_write(chip, 768 + index, 1)`.
    #[pyo3(signature = (chip, index, cycle=None))]
    fn ring(&mut self, py: Python<'_>, chip: usize, index: u64, cycle: Option<u64>) -> PyResult<u64> {
        guard(|| self.inner.csr_write(chip, 768 + index, 1, cycle)).map_err(|e| to_pyerr(py, e))
    }

    /// CSR `addr` of chip `chip` as the system stands.
    fn inject_sram(&mut self, py: Python<'_>, chip: usize, slice: u8, addr: u32, lane: u8, bit: u8) -> PyResult<()> {
        self.inner.inject_sram(chip, slice, addr, lane, bit).map_err(|e| to_pyerr(py, e))
    }

    fn service_scrub(&mut self, py: Python<'_>) -> PyResult<()> {
        guard(|| self.inner.service_scrub()).map_err(|e| to_pyerr(py, e))
    }

    fn ras_snapshot<'py>(&self, py: Python<'py>, chip: usize) -> PyResult<Bound<'py, PyDict>> {
        ras_snapshot(py, self.inner.service_ras(chip).map_err(|e| to_pyerr(py, e))?)
    }

    fn csr_read(&self, py: Python<'_>, chip: usize, addr: u64) -> PyResult<u64> {
        guard(|| self.inner.csr_read(chip, addr)).map_err(|e| to_pyerr(py, e))
    }

    fn _prepare_linked_panel(&mut self) {
        self.inner.prepare_linked_panel();
    }

    fn _run_linked_panel<'py>(&mut self, py: Python<'py>, limit: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        let inner = &mut self.inner;
        let outcome = py.detach(|| guard(|| inner.run_linked_panel(limit))).map_err(|e| to_pyerr(py, e))?;
        outcome.map(|result| outcome_dict(py, result)).transpose()
    }

    /// Run in lock-step windows until every chip is quiescent (`"done"`), a chip delivers an interrupt (`"notify"`, with
    /// `stop_on_notify`), a chip reports a diagnostic (`"diagnostic"`: raised as `TspChipError` unless
    /// `raise_on_diag=False`, which runs on), or the next event lies beyond `max_cycles` (`"limit"`). A run stops at the end
    /// of a window, so that every chip stands at `cycle`; calling `run` again resumes exactly where it stopped.
    #[pyo3(signature = (max_cycles=None, stop_on_notify=true, raise_on_diag=true))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        max_cycles: Option<u64>,
        stop_on_notify: bool,
        raise_on_diag: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inner = &mut self.inner;
        let o = py
            .detach(|| guard(|| inner.run(max_cycles, stop_on_notify, raise_on_diag)))
            .map_err(|e| to_pyerr(py, e))?;
        if raise_on_diag && o.reason == Reason::Diagnostic {
            return Err(system_error(py, o.diagnostics));
        }
        outcome_dict(py, o)
    }

    /// The end-of-run report of every chip (`model_finish`) after a run that ended `done`: a queue left with text that cannot
    /// dispatch raises OTS-E-IQ-004. Raises `TspChipError` for it unless `raise_on_diag=False`; returns `[(chip, cycle, queue, code)]`.
    #[pyo3(signature = (raise_on_diag=true))]
    fn finish(&mut self, py: Python<'_>, raise_on_diag: bool) -> PyResult<Vec<(usize, u64, u16, String)>> {
        let added = guard(|| self.inner.finish()).map_err(|e| to_pyerr(py, e))?;
        if raise_on_diag && !added.is_empty() {
            return Err(system_error(py, added));
        }
        Ok(added.into_iter().map(|(c, (cy, q, k))| (c, cy, q, k)).collect())
    }

    /// The vectors PciSend gave the host of chip `chip`, from the `start`-th on: `[(cycle, bytes)]`.
    #[pyo3(signature = (chip, start=0))]
    fn egress<'py>(&self, py: Python<'py>, chip: usize, start: usize) -> PyResult<Vec<(u64, Bound<'py, PyBytes>)>> {
        let v = guard(|| self.inner.egress(chip, start)).map_err(|e| to_pyerr(py, e))?;
        Ok(v.into_iter().map(|(c, b)| (c, PyBytes::new(py, &b))).collect())
    }

    /// Record successful C2C vector receives until restart.
    fn observe_link_receives(&mut self) {
        self.inner.observe_link_receives();
    }

    /// Consume the captured (cycle, link) pairs for one chip.
    fn drain_link_receives(&mut self, py: Python<'_>, chip: usize) -> PyResult<Vec<(u64, u8)>> {
        guard(|| self.inner.drain_link_receives(chip)).map_err(|e| to_pyerr(py, e))
    }

    /// The queues of chip `chip` parked by a HostWait that no doorbell ring has released: `[(queue, doorbell, cycle)]`.
    fn host_waiting(&self, py: Python<'_>, chip: usize) -> PyResult<Vec<(usize, u8, u64)>> {
        guard(|| self.inner.host_waiting(chip)).map_err(|e| to_pyerr(py, e))
    }

    /// `(vectors in the ingress FIFO, DMA requests not yet in it, whether the head request waits for a free slot)`.
    fn ingress(&self, py: Python<'_>, chip: usize) -> PyResult<(usize, usize, bool)> {
        guard(|| self.inner.ingress(chip)).map_err(|e| to_pyerr(py, e))
    }

    /// Every diagnostic chip `chip` reported so far: `[(cycle, queue, code)]`.
    fn diagnostics(&self, py: Python<'_>, chip: usize) -> PyResult<Vec<Diagnostic>> {
        guard(|| self.inner.diagnostics(chip).map(|d| d.to_vec())).map_err(|e| to_pyerr(py, e))
    }

    /// The `E` line of chip `chip`: `(code, cycle, count)` of its diagnostics (`None`, -1, 0 when there are none).
    fn error_line(&self, py: Python<'_>, chip: usize) -> PyResult<(Option<String>, i64, usize)> {
        let all = guard(|| self.inner.diagnostics(chip).map(|d| d.to_vec())).map_err(|e| to_pyerr(py, e))?;
        Ok(match all.first() {
            Some((cy, _, k)) => (Some(k.clone()), *cy as i64, all.len()),
            None => (None, -1, 0),
        })
    }

    /// The interrupts chip `chip` delivered: `[(cycle, vector)]` (`T irq`).
    fn irqs(&self, py: Python<'_>, chip: usize) -> PyResult<Vec<(u64, u8)>> {
        guard(|| self.inner.irqs(chip).map(|d| d.to_vec())).map_err(|e| to_pyerr(py, e))
    }

    /// What the PCIe interface of chip `chip` did: `[(cycle, "in" | "wait" | "out")]` (`T dma`).
    fn dma_events(&self, py: Python<'_>, chip: usize) -> PyResult<Vec<(u64, &'static str)>> {
        guard(|| self.inner.dma_events(chip).map(|d| d.to_vec())).map_err(|e| to_pyerr(py, e))
    }

    /// The cycles chip `chip` executed: one more than the last cycle it ticked.
    fn cycles(&self, py: Python<'_>, chip: usize) -> PyResult<u128> {
        guard(|| self.inner.cycles(chip)).map_err(|e| to_pyerr(py, e))
    }

    fn active_ticks(&self, py: Python<'_>, chip: usize) -> PyResult<u64> {
        guard(|| self.inner.active_ticks(chip)).map_err(|e| to_pyerr(py, e))
    }

    /// Enable or stop activity counting (initially off). Collected counts survive disabling and restart.
    fn set_activity_counting(&mut self, on: bool) {
        self.inner.set_activity_counting(on);
    }

    /// Cumulative instructions chip `chip` issued while counting was enabled, including before restart.
    fn activity_issued(&self, py: Python<'_>, chip: usize) -> PyResult<u64> {
        guard(|| self.inner.activity_issued(chip)).map_err(|e| to_pyerr(py, e))
    }

    /// With `trace=True`: `(srf, sram)` of chip `chip`, the `T srf` events `(cycle, "E"|"W", stream, gap, bytes)` and the
    /// `T sram` events `(cycle, slice, addr, bytes)`.
    #[allow(clippy::type_complexity)]
    fn trace<'py>(
        &self,
        py: Python<'py>,
        chip: usize,
    ) -> PyResult<(Vec<(u64, char, u8, u8, Bound<'py, PyBytes>)>, Vec<(u64, u8, u32, Bound<'py, PyBytes>)>)> {
        let (srf, sram) =
            guard(|| self.inner.trace(chip).map(|(a, b)| (a.to_vec(), b.to_vec()))).map_err(|e| to_pyerr(py, e))?;
        Ok((
            srf.into_iter().map(|(c, r, s, g, v)| (c, r, s, g, PyBytes::new(py, &v))).collect(),
            sram.into_iter().map(|(c, sl, a, v)| (c, sl, a, PyBytes::new(py, &v))).collect(),
        ))
    }
}

#[pymodule]
fn _tsp_iss(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chip>()?;
    m.add_class::<System>()?;
    m.add_function(wrap_pyfunction!(profile_sha256, m)?)?;
    m.add("SPEC_VERSION", ots_profile::SPEC_VERSION)?;
    m.add("TspIssError", m.py().get_type::<TspIssError>())?;
    m.add("PatchError", m.py().get_type::<PatchError>())?;
    Ok(())
}
