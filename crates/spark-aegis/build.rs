//! Builds the optional Mojo `libsimd_matcher.so`, but ONLY when the `mojo-bridge` feature is on.
//!
//! A default build does not run `mojo`, does not copy into the source tree and does not export a
//! build-directory path (`SPARK_AEGIS_SO_BUILT`), so the same sources give the same binaries in
//! any directory. The pure Rust engine is the default and the fallback.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=mojo/simd_matcher.mojo");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_MOJO_BRIDGE").is_none() {
        return;
    }
    let out_dir = match std::env::var("OUT_DIR") {
        Ok(d) => d,
        Err(_) => return,
    };
    let dest_path = Path::new(&out_dir).join("libsimd_matcher.so");

    if let Ok(status) = Command::new("mojo")
        .args([
            "build",
            "--emit",
            "shared-lib",
            "mojo/simd_matcher.mojo",
            "-o",
        ])
        .arg(&dest_path)
        .status()
    {
        if status.success() {
            println!(
                "cargo:rustc-env=SPARK_AEGIS_SO_BUILT={}",
                dest_path.display()
            );
            let _ = std::fs::copy(&dest_path, "mojo/libsimd_matcher.so");
        } else {
            println!("cargo:warning=mojo build failed, fallback to pure Rust engine");
        }
    } else {
        println!("cargo:warning=mojo toolchain unavailable, fallback to pure Rust engine");
    }
}
