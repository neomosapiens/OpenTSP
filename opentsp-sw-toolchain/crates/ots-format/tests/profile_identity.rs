//! The longest permitted profile name survives all three serialized formats.
use ots_format::{
    manifest::{read_manifest, write_manifest},
    read_object, read_weights,
    text::Images,
    write_object, write_weights,
};
use ots_profile::Profile;

#[test]
fn full_width_profile_name_roundtrips() {
    let config = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/mini.json");
    let name = "abcdefghijklmnop";
    let profile = Profile::load_file(name, config).unwrap();
    let object = write_object(&profile, &Images::new(), &[]);
    let weights = write_weights(&profile, &[]).unwrap();
    let manifest = write_manifest(name, std::slice::from_ref(&object), Some(config)).unwrap();
    assert_eq!(read_object(&object, Some(name), Some(config)).unwrap().0, name);
    assert!(weights.windows(name.len()).any(|window| window == name.as_bytes()));
    assert!(read_weights(&weights, Some(name), Some(config)).unwrap().is_empty());
    assert_eq!(read_manifest(&manifest, Some(name), Some(config)).unwrap().0, name);
}
