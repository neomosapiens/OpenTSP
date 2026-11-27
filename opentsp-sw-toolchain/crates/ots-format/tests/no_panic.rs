//! The readers `read_object`, `read_weights` and `read_manifest` take bytes from files the tool did not write:
//! whatever the bytes, they return an error or a value and never panic. Seeded generators (no external crate):
//! random byte strings, with and without a valid prefix, are always refused; seeded mutations of valid files
//! (bit flips, random bytes, truncation, extension, integer fields set to boundary values) end in `Ok` or `Err`.
use ots_format::{read_manifest, read_object, read_weights, AsmError};
use std::panic::{catch_unwind, AssertUnwindSafe};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../ots-asm/tests/data");

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

type Reader = fn(&[u8]) -> Result<(), AsmError>;

fn object(d: &[u8]) -> Result<(), AsmError> {
    read_object(d, None, None).map(|_| ())
}

fn weights(d: &[u8]) -> Result<(), AsmError> {
    read_weights(d, None, None).map(|_| ())
}

fn manifest(d: &[u8]) -> Result<(), AsmError> {
    read_manifest(d, None, None).map(|_| ())
}

/// (reader, header bytes, the valid files of the data directory).
fn readers() -> Vec<(&'static str, Reader, usize, Vec<Vec<u8>>)> {
    let load = |names: &[&str]| -> Vec<Vec<u8>> {
        names.iter().map(|n| std::fs::read(format!("{DATA}/{n}")).unwrap_or_else(|e| panic!("{n}: {e}"))).collect()
    };
    vec![
        (
            "read_object",
            object,
            64,
            load(&["example.elf", "example_mini.elf", "bind.elf", "bind_mini.elf", "parcel_src.elf"]),
        ),
        ("read_weights", weights, 32, load(&["example.safetensors", "example_mini.safetensors", "rich.safetensors"])),
        ("read_manifest", manifest, 64, load(&["ref1.otsm", "ref2.otsm", "mini3.otsm", "mini16.otsm"])),
    ]
}

/// Run `read` on `data`; a panic is reported with the input.
fn call(name: &str, read: Reader, data: &[u8], seed: u64) -> Result<(), AsmError> {
    match catch_unwind(AssertUnwindSafe(|| read(data))) {
        Ok(r) => r,
        Err(_) => panic!("{name} panicked (seed {seed}) on {} bytes: {}", data.len(), hex(data)),
    }
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn the_valid_files_are_read() {
    for (name, read, _, files) in readers() {
        for (i, f) in files.iter().enumerate() {
            assert!(read(f).is_ok(), "{name}: file {i}");
        }
    }
}

#[test]
fn random_byte_strings_are_refused() {
    for (name, read, header, files) in readers() {
        let magic = files[0][..4].to_vec();
        let mut rng = Rng(0x5eed_0001);
        for i in 0..CASES {
            let len = [0, 1, 3, 4, 5, 31, 32, 63, 64, 65, 200, 400][rng.below(12)];
            let n = rng.below(len + 1);
            let data = match i % 3 {
                0 => rng.bytes(n),
                // the magic, then random bytes
                1 => [magic.clone(), rng.bytes(n)].concat(),
                // the header of a valid file, then random bytes: the checks of the tables and entries
                _ => [files[rng.below(files.len())][..header].to_vec(), rng.bytes(n)].concat(),
            };
            let r = call(name, read, &data, i);
            assert!(r.is_err(), "{name}: accepted {} random bytes (case {i}): {}", data.len(), hex(&data));
        }
    }
}

/// Cases per reader and test: every call loads the profile and hashes its interface files.
const CASES: u64 = 3_000;

const BOUNDARY: [u32; 12] = [0, 1, 2, 7, 8, 63, 64, 65, 255, 0xffff, 0x7fff_ffff, 0xffff_ffff];

fn mutate(rng: &mut Rng, d: &mut Vec<u8>) {
    match rng.below(8) {
        0 if !d.is_empty() => {
            let i = rng.below(d.len());
            d[i] ^= 1 << rng.below(8);
        }
        1 if !d.is_empty() => {
            let i = rng.below(d.len());
            d[i] = rng.next() as u8;
        }
        2 => d.truncate(rng.below(d.len() + 1)),
        3 => {
            let n = rng.below(40);
            d.extend(rng.bytes(n));
        }
        // a 32-bit field: a boundary value, or the length of the file plus or minus a little
        4 | 5 if d.len() >= 4 => {
            let i = rng.below(d.len() - 3);
            let v = match rng.below(3) {
                0 => BOUNDARY[rng.below(BOUNDARY.len())],
                1 => (d.len() as u32).wrapping_add(rng.below(5) as u32).wrapping_sub(2),
                _ => rng.next() as u32,
            };
            d[i..i + 4].copy_from_slice(&v.to_le_bytes());
        }
        // the tail after the header replaced by random bytes (units and tensor data are not checked, so this can be accepted)
        6 if d.len() > 64 => {
            let cut = 64 + rng.below(d.len() - 64);
            d.truncate(cut);
            let n = rng.below(80);
            d.extend(rng.bytes(n));
        }
        // a field of the header, at the offsets the three layouts share (count, revision, reserved)
        _ if d.len() >= 40 => {
            let i = [4, 6, 8, 24, 28, 32, 36][rng.below(7)];
            d[i] = rng.next() as u8;
        }
        _ => {}
    }
}

#[test]
fn mutated_files_are_read_or_refused_without_a_panic() {
    for (name, read, _, files) in readers() {
        let mut rng = Rng(0x5eed_0002);
        let (mut ok, mut err) = (0u32, 0u32);
        for i in 0..CASES {
            let mut d = files[rng.below(files.len())].clone();
            for _ in 0..1 + rng.below(4) {
                mutate(&mut rng, &mut d);
            }
            match call(name, read, &d, i) {
                Ok(()) => ok += 1,
                Err(_) => err += 1,
            }
        }
        assert!(err > ok, "{name}: the mutations hardly reach the checks ({ok} accepted, {err} refused)");
    }
}
