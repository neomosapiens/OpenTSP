//! The lookups a decode makes per operand check: a profile variable and an extension test. They run for every parsed
//! instruction, so they must read stored values and allocate nothing.
use ots_profile::Profile;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocations<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCS.with(Cell::get);
    let r = f();
    (r, ALLOCS.with(Cell::get) - before)
}

fn declaring(list: &str) -> Profile {
    let base = String::from_utf8(Profile::load("mini").unwrap().bytes().to_vec()).unwrap();
    let old = "\"extensions\": [],";
    assert!(base.contains(old));
    Profile::from_bytes("mini", base.replacen(old, &format!("\"extensions\": {list},"), 1).as_bytes()).unwrap()
}

#[test]
fn an_extension_test_allocates_nothing() {
    let p = declaring("[\"block_fp8\", \"wide_acc\"]");
    let (answers, n) =
        allocations(|| (p.has_extension("wide_acc"), p.has_extension("block_fp8"), p.has_extension("nope")));
    assert_eq!(answers, (true, true, false));
    assert_eq!(n, 0, "has_extension allocated");
    let none = Profile::load("mini").unwrap();
    assert!(!none.has_extension("wide_acc"));
}

#[test]
fn a_profile_variable_is_its_expression_and_allocates_nothing() {
    for name in ["ref", "mini"] {
        let p = Profile::load(name).unwrap();
        let (got, n) = allocations(|| {
            (p.variable("vlanes"), p.variable("sram_words"), p.variable("acc_depth"), p.variable("no_such_variable"))
        });
        assert_eq!(n, 0, "{name}: variable allocated");
        assert_eq!(got.0, Some(16 * p.uint("geometry.superlanes").unwrap()), "{name}");
        assert_eq!(got.1, Some(1u64 << p.uint("sram.addr_bits").unwrap()), "{name}");
        assert_eq!(got.2, p.uint("mxm.acc_depth"), "{name}");
        assert_eq!(got.3, None, "{name}");
        assert_eq!(p.vlanes(), got.0.unwrap(), "{name}");
    }
}
