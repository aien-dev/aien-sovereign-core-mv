//! Builds `libspark_max.so` with Mojo, but ONLY when the `mojo-bridge` feature is on.
//!
//! The Modular MAX bridge is a comparison yardstick (Drake 2026-10-02: NVIDIA and MAX tools
//! are for comparison runs only, never baked into AIEN). A default build therefore does not
//! look for `mojo` on PATH, does not run it, does not copy anything into the source tree and
//! does not export a build-directory path (`SPARK_MAX_SO_BUILT`) into the compiled code, so
//! the same sources give the same binaries on any machine and in any directory.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=mojo/spark_max_bridge.mojo");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_MOJO_BRIDGE").is_none() {
        return;
    }
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("libspark_max.so");

    if let Ok(status) = Command::new("mojo")
        .args([
            "build",
            "--emit",
            "shared-lib",
            "mojo/spark_max_bridge.mojo",
            "-o",
        ])
        .arg(&dest_path)
        .status()
    {
        if status.success() {
            println!("cargo:rustc-env=SPARK_MAX_SO_BUILT={}", dest_path.display());
            let _ = std::fs::copy(&dest_path, "mojo/libspark_max.so");
        } else {
            println!("cargo:warning=mojo build exited with non-zero status");
        }
    } else {
        println!("cargo:warning=mojo binary not found on PATH, skipping libspark_max.so build");
    }
}
