//! The lane count of a machine is a type parameter: a machine of `NARROW_LANES` lanes runs the profiles of at most
//! that width (`ref`, `mini`) with per-cycle records of that width, a machine of `MAX_LANES` lanes runs every
//! profile, and the width changes no result.
mod common;
use common::*;
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::geom::{Geom, MAX_LANES, NARROW_LANES};
use tsp_iss::links::NoLinks;
use tsp_iss::machine::Machine;
use tsp_iss::native::{check_vector, check_vector_lanes};

fn profile(name: &str) -> Profile {
    let loaded = match name.strip_prefix("test/") {
        Some(t) => Profile::load_file(t, spec().join("config/test").join(format!("{t}.json"))),
        None => Profile::load(name),
    };
    loaded.unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn golden(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(spec().join("conformance/model").join(format!("{name}.json"))).unwrap())
        .unwrap()
}

#[test]
fn a_machine_holds_the_profiles_up_to_its_width() {
    let (narrow, wide) = (profile("ref"), profile("test/sl32"));
    assert_eq!((Geom::new(&narrow).vlanes as usize, Geom::new(&wide).vlanes as usize), (NARROW_LANES, MAX_LANES));
    assert!(Machine::<NARROW_LANES>::with_lanes(&narrow, 0, Box::new(NoLinks)).is_ok());
    assert!(Machine::<MAX_LANES>::with_lanes(&narrow, 0, Box::new(NoLinks)).is_ok());
    assert!(Machine::<MAX_LANES>::with_lanes(&wide, 0, Box::new(NoLinks)).is_ok());
    let e = Machine::<NARROW_LANES>::with_lanes(&wide, 0, Box::new(NoLinks)).err().expect("512 lanes do not fit 320");
    assert!(e.contains("512 lanes") && e.contains("at most 320"), "{e}");
    // `Machine::new` is the widest machine
    assert!(Machine::new(&wide, 0, Box::new(NoLinks)).is_ok());
}

#[test]
fn the_width_changes_no_result_on_the_golden_vectors() {
    for name in ["ref", "mini"] {
        let doc = golden(name);
        let p = profile(name);
        let geom = Geom::new(&p);
        let mut checked = 0;
        // every 6th vector: the narrow, the wide and the dispatching check agree with the file
        for v in doc["vectors"].as_array().unwrap().iter().step_by(6) {
            let n = v["name"].as_str().unwrap();
            assert_eq!(check_vector_lanes::<NARROW_LANES>(&p, &geom, v), Ok(()), "{name} {n} at {NARROW_LANES} lanes");
            assert_eq!(check_vector_lanes::<MAX_LANES>(&p, &geom, v), Ok(()), "{name} {n} at {MAX_LANES} lanes");
            assert_eq!(check_vector(&p, &geom, v), Ok(()), "{name} {n}");
            checked += 1;
        }
        assert!(checked >= 50, "{name}: {checked} vectors");
    }
}

#[test]
fn a_vector_check_on_a_machine_that_is_too_narrow_says_so() {
    let doc = golden("ref");
    let (p, wide) = (profile("ref"), profile("test/sl32"));
    let v = &doc["vectors"][0];
    // `ref` runs on both widths; a 32-superlane profile does not run on 320 lanes
    assert_eq!(check_vector_lanes::<MAX_LANES>(&p, &Geom::new(&p), v), Ok(()));
    let e = check_vector_lanes::<NARROW_LANES>(&wide, &Geom::new(&wide), v).unwrap_err();
    assert!(e.starts_with("machine: ") && e.contains("at most 320"), "{e}");
}
