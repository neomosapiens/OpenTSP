// The CLI uses mimalloc on the hosts qualified in the ISS benchmark report.
#[cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "x86_64")))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    std::process::exit(tsp_iss::harness::run_process());
}
