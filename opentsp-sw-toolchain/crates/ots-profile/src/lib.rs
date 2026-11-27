//! Profiles (configurations) of the opentsp architecture, specification 0.1.
//!
//! A profile is a JSON file (`opentsp-arch-spec/config/<name>.json`) that fixes the geometry,
//! latencies and layout of one machine. This crate
//! - loads a profile by name (the shipped `ref` and `mini` are embedded at build time, so their
//!   digests are those of the shipped files) or from a path;
//! - evaluates the profile variables of `encoding.json` (`vlanes`);
//! - holds the timing table, one row per catalogue entry (`timing`);
//! - validates a configuration against `config/schema.json` and the timing rules of the model
//!   (`validate`; a port of `tsp_config_check.c` and `timing_config_error`).
//!
//! Loading does not validate (the assembler never does; the ISS calls [`Profile::validate`]),
//! and it reads only the keys the toolchain needs, so keys that a later revision of the
//! schema adds are carried and reachable through [`Profile::get`] and [`Profile::uint`].
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fmt, fs, path::Path};

pub mod json;
pub mod schema;
mod timing;

pub use schema::ConfigError;
pub use timing::{eltype_name, timing_class_key, CatalogueEntry, Latency, TimingClass, TimingTable};

include!(concat!(env!("OUT_DIR"), "/catalogue.rs"));

/// The text of `config/schema.json` the validator uses.
pub const SCHEMA_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/schema.json"));

static REF_JSON: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/ref.json"));
static MINI_JSON: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/mini.json"));

/// The specification version this toolchain implements.
pub const SPEC_VERSION: &str = "0.1";

#[derive(Debug)]
pub enum ProfileError {
    /// No profile of this name (names are `[a-z0-9_]+`), or the file does not exist.
    Unknown(String),
    /// The file is not JSON, or not a JSON object.
    Json {
        name: String,
        detail: String,
    },
    /// `spec_version` is not `0.1`.
    SpecVersion {
        name: String,
        found: String,
    },
    /// A key the toolchain needs is missing or is not an unsigned integer.
    Key {
        name: String,
        key: String,
    },
    Io(String),
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::Unknown(n) => write!(f, "unknown profile {n:?}"),
            ProfileError::Json { name, detail } => write!(f, "profile {name:?}: not a JSON object: {detail}"),
            ProfileError::SpecVersion { name, found } => {
                write!(f, "profile {name:?} is spec version {found}, not {SPEC_VERSION}")
            }
            ProfileError::Key { name, key } => {
                write!(f, "profile {name:?}: {key} is missing or not an unsigned integer")
            }
            ProfileError::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ProfileError {}

/// One loaded profile: the bytes of its file, the parsed JSON and the values every user needs.
#[derive(Debug, Clone)]
pub struct Profile {
    name: String,
    /// The path of the file given by `--config` (as written), when the profile was read by [`Profile::load_file`].
    source_path: Option<String>,
    bytes: Vec<u8>,
    validation: std::sync::OnceLock<Result<(), ConfigError>>,
    cfg: Value,
    superlanes: u64,
    vlanes: u64,
    words_per_slice: u64,
    bank_bit: u64,
    addr_bits: u64,
    iq_depth: u64,
    /// The value of every `profile_variables` entry that evaluates, in the order of [`PROFILE_VARIABLES`], computed once at load: a
    /// decode asks for them per operand check.
    variables: Vec<Option<u64>>,
}

/// Insert the keys a profile may leave out (the schema's `default` of a property of a section such as `ras`) into the parsed
/// configuration, as `tsp_config_load` does for the text it gives the model. The bytes of the file, and so the profile digest, stay as
/// written.
fn apply_defaults(cfg: &mut Value) {
    for (section, key, default) in SCHEMA_DEFAULTS {
        let Some(obj) = cfg.get_mut(*section).and_then(Value::as_object_mut) else { continue };
        if !obj.contains_key(*key) {
            if let Ok(d) = serde_json::from_str::<Value>(default) {
                obj.insert((*key).to_string(), d);
            }
        }
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 16
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

impl Profile {
    /// A profile by name: `ref` and `mini` are the embedded copies of the shipped files; any other
    /// name is looked up as `config/<name>.json` in the specification tree (the tree this crate was
    /// built in, or `$OTS_SPEC_DIR/config`).
    pub fn load(name: &str) -> Result<Profile, ProfileError> {
        if !valid_name(name) {
            return Err(ProfileError::Unknown(name.to_string()));
        }
        match name {
            "ref" => Profile::from_bytes(name, REF_JSON),
            "mini" => Profile::from_bytes(name, MINI_JSON),
            _ => {
                let dir = std::env::var("OTS_SPEC_DIR")
                    .unwrap_or_else(|_| format!("{}/../../../opentsp-arch-spec", env!("CARGO_MANIFEST_DIR")));
                // a profile named in the specification tree is still "a profile", as `tsp_as.py --profile` reports it
                let mut p = Profile::load_file(name, format!("{dir}/config/{name}.json"))?;
                p.source_path = None;
                Ok(p)
            }
        }
    }

    /// A profile of the given name read from `path` (`--config`). The name must still be
    /// `[a-z0-9_]{1,16}` and the file must exist (`tsp_as.load_profile`).
    pub fn load_file(name: &str, path: impl AsRef<Path>) -> Result<Profile, ProfileError> {
        if !valid_name(name) || !path.as_ref().exists() {
            return Err(ProfileError::Unknown(name.to_string()));
        }
        let bytes =
            fs::read(path.as_ref()).map_err(|e| ProfileError::Io(format!("{}: {e}", path.as_ref().display())))?;
        let mut p = Profile::from_bytes(name, &bytes)?;
        p.source_path = Some(path.as_ref().display().to_string());
        Ok(p)
    }

    /// A profile from the bytes of its file. The last of two equal keys wins (as `json.load`),
    /// [`Profile::validate`] refuses duplicates.
    pub fn from_bytes(name: &str, bytes: &[u8]) -> Result<Profile, ProfileError> {
        if !valid_name(name) {
            return Err(ProfileError::Unknown(name.to_string()));
        }
        let mut cfg: Value = serde_json::from_slice(bytes)
            .map_err(|e| ProfileError::Json { name: name.to_string(), detail: e.to_string() })?;
        apply_defaults(&mut cfg);
        if !cfg.is_object() {
            return Err(ProfileError::Json { name: name.to_string(), detail: "not an object".into() });
        }
        match cfg.get("spec_version").and_then(Value::as_str) {
            Some(SPEC_VERSION) => {}
            other => {
                let found = other
                    .map(str::to_string)
                    .unwrap_or_else(|| cfg.get("spec_version").map_or("(none)".into(), |v| v.to_string()));
                return Err(ProfileError::SpecVersion { name: name.to_string(), found });
            }
        }
        let uint = |key: &str| {
            key.split('.')
                .try_fold(&cfg, |n, k| n.get(k))
                .and_then(Value::as_u64)
                .ok_or_else(|| ProfileError::Key { name: name.to_string(), key: key.to_string() })
        };
        let superlanes = uint("geometry.superlanes")?;
        let words_per_slice = uint("sram.words_per_slice")?;
        let bank_bit = uint("sram.bank_bit")?;
        let addr_bits = uint("sram.addr_bits")?;
        let iq_depth = uint("icu.iq_depth")?;
        let mut p = Profile {
            name: name.to_string(),
            source_path: None,
            bytes: bytes.to_vec(),
            validation: Default::default(),
            cfg,
            superlanes,
            vlanes: 0,
            words_per_slice,
            bank_bit,
            addr_bits,
            iq_depth,
            variables: Vec::new(),
        };
        p.variables = PROFILE_VARIABLES.iter().map(|(_, expr)| p.evaluate(expr)).collect();
        p.vlanes =
            p.variable("vlanes").ok_or_else(|| ProfileError::Key { name: name.to_string(), key: "vlanes".into() })?;
        Ok(p)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The path of the `--config` file the profile was read from, as given; `None` for a profile read by name.
    pub fn source_path(&self) -> Option<&str> {
        self.source_path.as_deref()
    }

    /// The configuration in force, for messages (`tsp_as.py` `where`): `file <path>` for a `--config` file, else `profile <name>`.
    pub fn describe(&self) -> String {
        match &self.source_path {
            Some(p) => format!("file {p}"),
            None => format!("profile {}", self.name),
        }
    }

    /// The bytes of the file the profile was read from.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The parsed configuration.
    pub fn value(&self) -> &Value {
        &self.cfg
    }

    /// The value at a dotted key (`sram.words_per_slice`).
    pub fn get(&self, key: &str) -> Option<&Value> {
        key.split('.').try_fold(&self.cfg, |n, k| n.get(k))
    }

    /// The names in `extensions`: the extensions the profile declares (none when the key is absent).
    // model: core/types.sail extensions
    pub fn extensions(&self) -> Vec<&str> {
        self.cfg
            .get("extensions")
            .and_then(|v| v.as_array())
            .map_or(Vec::new(), |a| a.iter().filter_map(|x| x.as_str()).collect())
    }

    /// Does the profile declare the extension `name`? // model: core/types.sail has_extension
    pub fn has_extension(&self, name: &str) -> bool {
        self.cfg
            .get("extensions")
            .and_then(|v| v.as_array())
            .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(name)))
    }

    /// The unsigned integer at a dotted key.
    pub fn uint(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    /// The first 12 bytes of SHA-256 of the file (the object's profile digest).
    pub fn digest12(&self) -> [u8; 12] {
        let d: [u8; 32] = Sha256::digest(&self.bytes).into();
        d[..12].try_into().unwrap()
    }

    /// A profile variable of `encoding.json` `profile_variables`: `<int> * <key> * ...`, each
    /// factor an integer, a dotted configuration key, or an integer raised to a key (`2 ^ sram.addr_bits`).
    pub fn variable(&self, name: &str) -> Option<u64> {
        let i = PROFILE_VARIABLES.iter().position(|(n, _)| *n == name)?;
        self.variables[i]
    }

    /// The first `profile_variables` entry that has no value in this profile (a key of its expression is absent): `(name, expression)`.
    /// A validated profile has none; a file read without validation (`--no-validate`) may.
    pub fn missing_variable(&self) -> Option<(&'static str, &'static str)> {
        PROFILE_VARIABLES.iter().zip(&self.variables).find(|(_, v)| v.is_none()).map(|((n, e), _)| (*n, *e))
    }

    /// The value of one `profile_variables` expression (read once per variable at load).
    fn evaluate(&self, expr: &str) -> Option<u64> {
        let mut val: u64 = 1;
        for f in expr.split('*') {
            let f = f.trim();
            let x = match f.parse::<u64>() {
                Ok(x) => x,
                Err(_) => match f.split_once('^') {
                    Some((base, key)) => {
                        base.trim().parse::<u64>().ok()?.checked_pow(u32::try_from(self.uint(key.trim())?).ok()?)?
                    }
                    None => self.uint(f)?,
                },
            };
            val = val.checked_mul(x)?;
        }
        Some(val)
    }

    /// `geometry.superlanes`.
    pub fn superlanes(&self) -> u64 {
        self.superlanes
    }
    /// The lane count (`vlanes` = 16 * `geometry.superlanes`).
    pub fn vlanes(&self) -> u64 {
        self.vlanes
    }
    /// `sram.addr_bits`, the width of an SRAM address (13 to 16).
    pub fn addr_bits(&self) -> u64 {
        self.addr_bits
    }
    /// `sram.words_per_slice`.
    pub fn words_per_slice(&self) -> u64 {
        self.words_per_slice
    }
    /// `sram.bank_bit`.
    pub fn bank_bit(&self) -> u64 {
        self.bank_bit
    }
    /// `icu.iq_depth`, in bytes.
    pub fn iq_depth(&self) -> u64 {
        self.iq_depth
    }

    /// The effective timing table: one row per catalogue entry (`timing.overrides`, else the entry's class).
    pub fn timing_table(&self) -> TimingTable {
        timing::table(self)
    }

    /// The rules the model checks on the effective timing table (`timing_config_error` of
    /// `model/postlude/timing_check.sail`): `None` when valid, else `<key>: <reason>`.
    pub fn timing_config_error(&self) -> Option<String> {
        timing::config_error(self)
    }

    /// Parse a simulator configuration in the same diagnostic order as the Sail harness.
    pub fn from_validated_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        if bytes.contains(&0) {
            return Err(ConfigError { message: "NUL byte in file".into(), schema_error: false });
        }
        schema::check_text(bytes, SCHEMA_JSON)?;
        let p = Self::from_bytes("config", bytes)
            .map_err(|e| ConfigError { message: e.to_string(), schema_error: false })?;
        if let Some(message) = p.timing_config_error() {
            return Err(ConfigError { message, schema_error: false });
        }
        let _ = p.validation.set(Ok(()));
        Ok(p)
    }

    /// Full validation, in the order `tsp_sim` applies it: the schema and its relations, then the
    /// timing rules. The error message names the key (`<key>: <reason>`).
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validation.get_or_init(|| validate_bytes(&self.bytes)).clone()
    }
}

/// Validate the text of a configuration file: not NUL-free, valid JSON, the schema (with
/// duplicate keys refused and every integer a decimal literal), the schema's relations, then
/// `timing_config_error` (which needs a profile that loads).
pub fn validate_bytes(bytes: &[u8]) -> Result<(), ConfigError> {
    Profile::from_validated_bytes(bytes).map(|_| ())
}
