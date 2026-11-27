//! The topology file of `--topology` (`tsp_sim.c`, `topology_load`; `topology.md` section 7): the adjacency of every
//! `(chip, link)`, optionally with a latency per link, a default latency and a jitter seed.
//!
//! ```json
//! {"chips": 12, "latency": 216, "jitter_seed": 5, "cycles_per_vector": 24,
//!  "links": [{"a": [0, 0], "b": [1, 3], "latency": 220}, {"a": [2, 0], "b": [2, 0]}]}
//! ```
//!
//! Each entry joins two link ends (the same end twice loops a link back on itself) and names every end once. An entry's
//! `latency` is the latency of both directions of that link, the top-level `latency` is the default of the others (both
//! within `[c2c.link_latency_min, c2c.link_latency_max]`), and `cycles_per_vector`, when present, must be the profile's
//! `c2c.cycles_per_vector` (the link rate is a profile key, not wiring).
use crate::geom::{Geom, MAX_CHIPS, MAX_LINKS};
use ots_profile::json::{self, Json};

/// A parsed topology file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub chips: usize,
    /// `[chip * MAX_LINKS + link]`: the far end.
    peer: Vec<Option<(u16, u8)>>,
    /// `[chip * MAX_LINKS + link]`: the latency of the directed link, 0 when the file gives none.
    lat: Vec<u64>,
    /// The file's `latency`.
    pub latency_default: Option<u64>,
    /// The file's `jitter_seed`.
    pub jitter_seed: Option<u64>,
}

fn int(j: &Json, what: &str, lo: i64, hi: i64) -> Result<i64, String> {
    let f = match j {
        Json::Num(literal) => literal.parse::<f64>().ok(),
        _ => None,
    };
    // Match the C harness binary64/integer round-trip, including signed underflow.
    // An overflowing JSON number is valid syntax, but is not an integer field.
    let f = match f {
        Some(f) if f.is_finite() && f == (f as i64) as f64 => f,
        _ => return Err(format!("{what} must be an integer")),
    };
    if f < lo as f64 || f > hi as f64 {
        return Err(format!("{what}: {f:.0} is outside {lo}..{hi}"));
    }
    Ok(f as i64)
}

fn end(j: Option<&Json>, what: &str, chips: usize, links: usize) -> Result<(usize, usize), String> {
    let a = match j {
        Some(Json::Arr(a)) if a.len() == 2 => a,
        _ => return Err(format!("{what} must be [chip, link]")),
    };
    let chip = int(&a[0], "chip", 0, chips as i64 - 1)?;
    let link = int(&a[1], "link", 0, links as i64 - 1)?;
    Ok((chip as usize, link as usize))
}

// Keep the lossless parse: duplicate members and numeric literals are semantic inputs. C validates
// top-level duplicates before unknown keys, but link-object unknown keys before duplicates.
// Interpret objects in that same order; nested errors must not hide an earlier outer error.
// The C strict syntax reader rejects NUL strings and values below depth 128.
// The shared lossless reader allows deeper profiles, so retain this topology cap.
fn invalid_value(value: &Json, depth: usize) -> bool {
    if depth > 128 {
        return true;
    }
    match value {
        Json::Str(s) => s.contains('\0'),
        Json::Arr(values) => values.iter().any(|v| invalid_value(v, depth + 1)),
        Json::Obj(members) => members.iter().any(|(key, value)| key.contains('\0') || invalid_value(value, depth + 1)),
        _ => false,
    }
}

fn check_duplicates(members: &[(String, Json)]) -> Result<(), String> {
    let mut seen = Vec::new();
    for (key, _) in members {
        if seen.contains(&key.as_str()) {
            return Err(format!("duplicate key \"{key}\""));
        }
        seen.push(key.as_str());
    }
    Ok(())
}

fn check_known_keys(members: &[(String, Json)], allowed: &[&str], prefix: &str) -> Result<(), String> {
    for (key, _) in members {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{prefix}unknown key \"{key}\""));
        }
    }
    Ok(())
}

impl Topology {
    /// Parse and check the text of a topology file against the profile's geometry. The error is the text after
    /// `topology <path>: `.
    // model: c_emulator/tsp_sim.c topology_load
    pub fn parse(text: &[u8], g: &Geom) -> Result<Topology, String> {
        let root = json::parse(text).map_err(|_| "not a JSON object".to_string())?;
        if invalid_value(&root, 0) {
            return Err("not a JSON object".into());
        }
        let Json::Obj(raw_members) = &root else { return Err("not a JSON object".into()) };
        check_duplicates(raw_members)?;
        check_known_keys(raw_members, &["chips", "links", "latency", "jitter_seed", "cycles_per_vector"], "")?;
        let (lo, hi) = (g.link_latency_min as i64, g.link_latency_max as i64);
        let Some(jc) = root.get("chips") else { return Err("\"chips\" is missing".into()) };
        let chips = int(jc, "chips", 1, MAX_CHIPS as i64)? as usize;
        let Some(Json::Arr(jl)) = root.get("links") else {
            return Err("\"links\" must be an array".into());
        };
        let latency_default = match root.get("latency") {
            Some(j) => Some(int(j, "latency", lo, hi)? as u64),
            None => None,
        };
        let jitter_seed = match root.get("jitter_seed") {
            Some(j) => Some(int(j, "jitter_seed", 0, 2_147_483_647)? as u64),
            None => None,
        };
        if let Some(j) = root.get("cycles_per_vector") {
            let v = int(j, "cycles_per_vector", 1, 1 << 20)? as u64;
            if v != g.cycles_per_vector {
                return Err(format!(
                    "cycles_per_vector {v} is not the profile's c2c.cycles_per_vector {}",
                    g.cycles_per_vector
                ));
            }
        }
        let links = g.links as usize;
        let mut peer: Vec<Option<(u16, u8)>> = vec![None; MAX_CHIPS * MAX_LINKS];
        let mut lat = vec![0u64; MAX_CHIPS * MAX_LINKS];
        for (idx, e) in jl.iter().enumerate() {
            let Json::Obj(raw_members) = e else { return Err(format!("links[{idx}] must be an object")) };
            check_known_keys(raw_members, &["a", "b", "latency"], &format!("links[{idx}]: "))?;
            check_duplicates(raw_members)?;
            let (ac, al) = end(e.get("a"), &format!("links[{idx}].a"), chips, links)?;
            let (bc, bl) = end(e.get("b"), &format!("links[{idx}].b"), chips, links)?;
            let (a, b) = (ac * MAX_LINKS + al, bc * MAX_LINKS + bl);
            if peer[a].is_some() {
                return Err(format!("links[{idx}]: chip {ac} link {al} is joined twice"));
            }
            if peer[b].is_some() {
                return Err(format!("links[{idx}]: chip {bc} link {bl} is joined twice"));
            }
            let l = match e.get("latency") {
                Some(j) => int(j, &format!("links[{idx}].latency"), lo, hi)? as u64,
                None => 0,
            };
            peer[a] = Some((bc as u16, bl as u8));
            peer[b] = Some((ac as u16, al as u8));
            lat[a] = l;
            lat[b] = l;
        }
        Ok(Topology { chips, peer, lat, latency_default, jitter_seed })
    }

    /// The far end of chip `c`'s link `l`.
    pub fn peer(&self, c: usize, l: usize) -> Option<(usize, usize)> {
        self.peer.get(c * MAX_LINKS + l).copied().flatten().map(|(pc, pl)| (pc as usize, pl as usize))
    }

    /// The latency the file gives the directed link, 0 for none.
    pub fn latency(&self, c: usize, l: usize) -> u64 {
        self.lat.get(c * MAX_LINKS + l).copied().unwrap_or(0)
    }
}
