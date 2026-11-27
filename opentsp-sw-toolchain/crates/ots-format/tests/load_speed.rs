//! The two costs of loading a program and its weights that grew with the size of the package, and the
//! rules they must keep: `ExternIndex` answers the extern rules ([`check_extern`]) without scanning every
//! earlier extern, and `digest16` is the first 16 bytes of SHA-256 whatever backend computes it.
//! `check_extern` is the reference: the index is compared with it on sequences that make every rule fire.
use ots_format::{
    check_extern,
    externs::ExternIndex,
    weights::{digest16, digest_backend},
    Extern,
};
use ots_profile::Profile;
use sha2::{Digest, Sha256};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn ext(name: &str, slice: u64, addr: u64, size: u64) -> Extern {
    Extern { name: name.into(), slice, addr, size }
}

/// Feeds `seq` to both the reference (a list of the accepted so far) and the index; every verdict, message
/// included, must agree. Returns how many were accepted.
fn agree(p: &Profile, seq: &[Extern]) -> usize {
    let mut known: Vec<Extern> = Vec::new();
    let mut idx = ExternIndex::new();
    for (i, x) in seq.iter().enumerate() {
        let want = check_extern(x, p, &known);
        let got = idx.check(x, p);
        assert_eq!(got, want, "extern {i}: {x:?}");
        if want.is_ok() {
            idx.push(x);
            known.push(x.clone());
        }
    }
    assert_eq!(idx.len(), known.len());
    known.len()
}

#[test]
fn the_index_gives_the_verdict_and_the_message_of_the_reference_on_random_sequences() {
    let p = Profile::load("ref").unwrap();
    let words = p.words_per_slice();
    let mut accepted = 0;
    let mut rejected = 0;
    for seed in 1..=60u64 {
        let mut r = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut seq = Vec::new();
        for i in 0..80 {
            // few names, few slices and short ranges inside a small window: duplicates and overlaps are common
            let name = match r.below(10) {
                0 => format!("n{}", r.below(8)),
                1 => "bad name".to_string(),
                2 => "a_name_far_too_long".to_string(),
                _ => format!("t{i}"),
            };
            let slice = match r.below(12) {
                0 => 1 << 40,
                1 => 43,
                _ => r.below(3) * 44 + r.below(2),
            };
            let size = match r.below(12) {
                0 => 0,
                1 => words,
                _ => 1 + r.below(24),
            };
            let addr = match r.below(12) {
                0 => words - 1,
                1 => u64::MAX,
                _ => r.below(96),
            };
            seq.push(ext(&name, slice, addr, size));
        }
        let ok = agree(&p, &seq);
        accepted += ok;
        rejected += seq.len() - ok;
    }
    // the sequences exercise both sides
    assert!(accepted > 200 && rejected > 200, "accepted {accepted}, rejected {rejected}");
}

#[test]
fn the_first_overlapped_extern_in_declaration_order_is_the_one_named() {
    let p = Profile::load("ref").unwrap();
    // `c` covers `a` and `b`, which were declared in the other order than their addresses
    let seq = [ext("b", 4, 0x30, 4), ext("a", 4, 0x10, 4), ext("c", 4, 0x00, 0x100)];
    let mut idx = ExternIndex::new();
    idx.push(&seq[0]);
    idx.push(&seq[1]);
    let e = idx.check(&seq[2], &p).unwrap_err();
    assert_eq!(e.msg, "extern c overlaps b in slice 4");
    assert_eq!(agree(&p, &seq), 2);
}

#[test]
fn a_big_package_is_checked_in_linear_time() {
    // 11,509 externs, the size of the P(8,4,128) benchmark: the scan of the reference takes seconds in a
    // debug build, the index a few milliseconds. The bound is loose (it only tells linear from quadratic).
    let p = Profile::load("ref").unwrap();
    let mut idx = ExternIndex::new();
    let started = std::time::Instant::now();
    let mut n = 0;
    'fill: for slice in 0..p.uint("sram.slices").unwrap() {
        if slice % 44 >= p.uint("geometry.mem_slices_per_hemisphere").unwrap_or(44) {
            continue;
        }
        for k in 0..(p.words_per_slice() / 4).min(512) {
            let x = ext(&format!("w{n}"), slice, k * 4, 4);
            idx.check(&x, &p).unwrap();
            idx.push(&x);
            n += 1;
            if n == 11_509 {
                break 'fill;
            }
        }
    }
    assert_eq!(n, 11_509);
    let took = started.elapsed();
    assert!(took.as_millis() < 300, "11,509 externs took {took:?}");
}

#[test]
fn digest16_is_the_head_of_sha256_for_every_length_around_the_block_sizes() {
    let mut r = Rng(7);
    let big: Vec<u8> = (0..200_000).map(|_| r.next() as u8).collect();
    let mut lens: Vec<usize> = (0..=300).collect();
    lens.extend([511, 512, 513, 4095, 4096, 4097, 65_535, 65_536, 65_537, 200_000]);
    for n in lens {
        let want: [u8; 32] = Sha256::digest(&big[..n]).into();
        assert_eq!(digest16(&big[..n]), want[..16], "length {n}");
    }
}

#[test]
fn digest16_matches_the_published_vectors() {
    // FIPS 180-4: the empty message and "abc"
    assert_eq!(digest16(b"")[..], hex("e3b0c44298fc1c149afbf4c8996fb924")[..]);
    assert_eq!(digest16(b"abc")[..], hex("ba7816bf8f01cfea414140de5dae2223")[..]);
    // the 448-bit message of the standard
    assert_eq!(
        digest16(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")[..],
        hex("248d6a61d20638b8e5c026930c3e6039")[..]
    );
}

#[test]
fn the_fast_digest_backend_is_in_use_where_there_is_one() {
    let b = digest_backend();
    if cfg!(target_os = "macos") {
        assert_eq!(b, "commoncrypto");
    } else {
        assert_eq!(b, "sha2");
    }
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

// ---- read_weights on a package big enough to be hashed on several threads: the same verdict, in table order

use ots_format::weights::{read_weights, write_weights, TensorIn};
use ots_isa::Code;

const N: usize = 40;

/// 40 tensors of about 150 KiB each, 6 MiB in all: more than the reader hashes on one thread.
fn big() -> (Vec<u8>, usize) {
    let p = Profile::load("ref").unwrap();
    let vl = p.vlanes() as usize;
    let words = (150 << 10) / vl;
    let mut r = Rng(11);
    let tensors: Vec<TensorIn> = (0..N)
        .map(|i| TensorIn {
            name: format!("t{i}"),
            dtype: None,
            words: (0..words).map(|_| (0..vl).map(|_| r.next() as u8).collect()).collect(),
        })
        .collect();
    (write_weights(&p, &tensors).unwrap(), words * vl)
}

fn metadata_value(bytes: &[u8], name: &str) -> usize {
    let key = format!("\"{name}\":\"");
    bytes.windows(key.len()).position(|window| window == key.as_bytes()).unwrap() + key.len()
}

fn bad_digest(b: &mut [u8], i: usize) {
    let pos = metadata_value(b, &format!("sha256.t{i}"));
    b[pos] = if b[pos] == b'0' { b'1' } else { b'0' };
}

fn bad_dtype(b: &mut [u8], i: usize) {
    let pos = metadata_value(b, &format!("dtype.t{i}"));
    b[pos] = b'x';
}

fn verdict(b: &[u8]) -> Result<usize, (Code, String)> {
    read_weights(b, Some("ref"), None).map(|m| m.len()).map_err(|e| (e.code, e.msg))
}

#[test]
fn a_big_package_reads() {
    let (b, _) = big();
    assert!(b.len() > 4 << 20, "{} bytes", b.len());
    assert_eq!(verdict(&b), Ok(N));
}

#[test]
fn the_first_bad_digest_in_table_order_is_the_one_named() {
    let (mut b, _) = big();
    bad_digest(&mut b, 30);
    bad_digest(&mut b, 7);
    bad_digest(&mut b, 22);
    assert_eq!(verdict(&b), Err((Code::Asm004, "tensor t22: image does not match its digest".into())));
}

#[test]
fn a_digest_before_a_bad_entry_wins_and_a_bad_entry_before_a_digest_wins() {
    // digest of 30 is wrong, the dtype of 35 is: the reader reaches the image of 30 first
    let (mut b, _) = big();
    bad_digest(&mut b, 30);
    bad_dtype(&mut b, 35);
    assert_eq!(verdict(&b), Err((Code::Asm004, "tensor t30: image does not match its digest".into())));
    // the dtype of 3 is bad, the digest of 5 is: the entry is reached first
    let (mut b, _) = big();
    bad_dtype(&mut b, 3);
    bad_digest(&mut b, 5);
    let e = verdict(&b).unwrap_err();
    assert_eq!(e.0, Code::Asm003);
    assert!(e.1.starts_with("tensor t3: invalid element type metadata"), "{}", e.1);
    // a wrong digest is not hidden by a truncated file either: the last image is cut short
    let (mut b, _) = big();
    bad_digest(&mut b, 2);
    b.truncate(b.len() - 10);
    assert_eq!(verdict(&b), Err((Code::Asm004, "tensor t2: image does not match its digest".into())));
}

#[test]
fn a_wrong_digest_is_found_wherever_it_is_in_the_table() {
    for i in [0, 1, N / 2, N - 2, N - 1] {
        let (mut b, _) = big();
        bad_digest(&mut b, i);
        assert_eq!(verdict(&b), Err((Code::Asm004, format!("tensor t{i}: image does not match its digest"))), "{i}");
    }
}
