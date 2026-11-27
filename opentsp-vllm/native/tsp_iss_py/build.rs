// A module is not linked against libpython; macOS needs `-undefined dynamic_lookup` for that
// (with PYO3_BUILD_EXTENSION_MODULE=1, which `just vllm-binding` sets).
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
