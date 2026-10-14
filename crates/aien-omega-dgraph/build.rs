//! Decides whether omega's one-list decode token (`src/omega_gpu_decode_graph.h`, OWN-30B L2
//! cut 2 on the omega side) is linked.
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script (this crate only declares
//! the extra symbols), so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: native when `nm` finds `omega_gpu_decode_graph_create`
//!     defined in that archive.
//!   * `AIEN_OMEGA_DIR=<checkout>`: native when the checkout has `src/omega_gpu_decode_graph.h`.
//!   * otherwise: stub (every call returns `Unavailable`).
//!
//! `cfg(has_omega_dgraph)` is set only when native. A stub never fails the build: the engine
//! keeps its launch-at-a-time decode, and `is_native()` says which one a binary carries.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_dgraph)");
    for v in ["AIEN_FORCE_CPU_STUB", "AIEN_OMEGA_GPU_LIB", "AIEN_OMEGA_DIR"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_GPU_LIB") {
        println!("cargo:rerun-if-changed={lib}");
        if archive_defines(Path::new(&lib), "omega_gpu_decode_graph_create") {
            println!("cargo:rustc-cfg=has_omega_dgraph");
        } else {
            println!("cargo:warning=aien-omega-dgraph: {lib} does not define the decode graph API; the one-list decode is stubbed");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") {
        let header = Path::new(&dir).join("src/omega_gpu_decode_graph.h");
        println!("cargo:rerun-if-changed={}", header.display());
        if header.is_file() {
            println!("cargo:rustc-cfg=has_omega_dgraph");
        } else {
            println!("cargo:warning=aien-omega-dgraph: {dir} has no src/omega_gpu_decode_graph.h (omega before OWN-30B L2 cut 2); the one-list decode is stubbed");
        }
    }
}

fn archive_defines(lib: &Path, symbol: &str) -> bool {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-dgraph: nm not available, assuming the archive lacks the decode graph API");
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.ends_with(&format!(" T {symbol}")))
}
