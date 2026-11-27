use std::process::Command;

fn main() {
    if let Ok(dir) = Command::new("pkg-config").args(["--variable=libdir", "libzstd"]).output() {
        if dir.status.success() {
            if let Ok(path) = std::str::from_utf8(&dir.stdout) {
                println!("cargo:rustc-link-search=native={}", path.trim());
            }
        }
    }
}
