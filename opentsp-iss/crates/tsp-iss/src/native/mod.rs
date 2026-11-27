//! The native subcommands `run`, `vectors` and `bench`.
//!
//! * [`run`]: load a program through `ots-format` (an `.elf` object, a Q-line text file, or an `.otsm`
//!   manifest) with its `.safetensors` weight packages, run it on `--chips K` chips and print what the
//!   compatible mode prints for the harness text of `ots-asm objinput`.
//! * [`vectors`]: run the golden vectors of a conformance file in-process.
//! * [`bench`]: run the benchmark programs and print simulated cycles per second.
//!
//! Errors on stderr start with `tsp-iss: <subcommand>: `; usage and input errors exit 2.
mod bench;
mod obs_out;
mod run;
pub mod threads;
mod vectors;

pub use bench::bench;
pub use run::{load_manifest, run, ChipLoad};
pub use vectors::{canonical_trace, check_config_digest, check_vector, check_vector_lanes, config_digest, vectors};

use crate::geom::Geom;
use crate::harness::{load_config, PROG};
use ots_profile::Profile;
use std::ffi::OsStr;
use std::io::Write;
use std::path::Path;

/// `tsp-iss: <cmd>: <msg>`.
pub(crate) fn fail(stderr: &mut dyn Write, cmd: &str, msg: &str) -> i32 {
    let _ = writeln!(stderr, "{PROG}: {cmd}: {msg}");
    2
}

/// The profile of a shipped name (`--profile`), checked as a configuration file is.
pub(crate) fn load_named(name: &str) -> Result<Profile, String> {
    let p = Profile::load(name).map_err(|e| e.to_string())?;
    p.validate().map_err(|e| format!("config {name}: {e}"))?;
    Ok(p)
}

/// The profile of a configuration file (`--config`), with `tsp_sim`'s message on error.
pub(crate) fn load_file(path: &OsStr) -> Result<Profile, String> {
    let p = Path::new(path);
    load_config(p).map_err(|m| format!("config {}: {m}", p.display()))
}

/// The checks of `tsp_sim` after the configuration is valid: link latency range and `vlanes <= MAX_LANES` (512).
pub(crate) fn check_geom(profile: &Profile, origin: &str) -> Result<Geom, String> {
    let geom = Geom::new(profile);
    geom.check_link_latency().map_err(|m| format!("config {origin}: {m}"))?;
    geom.check_lanes().map_err(|m| format!("config {origin}: {m}"))?;
    Ok(geom)
}
