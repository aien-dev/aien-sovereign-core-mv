//! Decides whether omega's one-launch decode attention (`src/omega_gpu_decode_attn_api.h`,
//! OWN-30B cut 8 on the omega side) is linked.
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script (this crate only declares
//! the extra symbols), so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: native when `nm` finds `omega_gpu_decode_attn` defined
//!     in that archive.
//!   * `AIEN_OMEGA_DIR=<checkout>`: native when the checkout has `src/omega_gpu_decode_attn_api.h`.
//!   * otherwise: stub (every call returns `Unavailable`).
//!
//! `cfg(has_omega_dattn)` is set only when native. A stub never fails the build: the engine
//! keeps its split attention path (host norm/rope, paged KV), and `is_native()` says which
//! one a binary carries.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_dattn)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_dattn_rows)");
    for v in ["AIEN_FORCE_CPU_STUB", "AIEN_OMEGA_GPU_LIB", "AIEN_OMEGA_DIR"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_GPU_LIB") {
        println!("cargo:rerun-if-changed={lib}");
        if archive_defines(Path::new(&lib), "omega_gpu_decode_attn") {
            println!("cargo:rustc-cfg=has_omega_dattn");
            // W-L4d: the multi-row exact verify attention (omega own30b/spec-exact) is optional on top.
            if archive_defines(Path::new(&lib), "omega_gpu_decode_attn_rows") {
                println!("cargo:rustc-cfg=has_omega_dattn_rows");
            }
        } else {
            println!("cargo:warning=aien-omega-dattn: {lib} does not define the decode attention API; fused attention is stubbed");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") {
        let header = Path::new(&dir).join("src/omega_gpu_decode_attn_api.h");
        println!("cargo:rerun-if-changed={}", header.display());
        if header.is_file() {
            println!("cargo:rustc-cfg=has_omega_dattn");
            if std::fs::read_to_string(&header).map(|h| h.contains("omega_gpu_decode_attn_rows")).unwrap_or(false) {
                println!("cargo:rustc-cfg=has_omega_dattn_rows");
            }
        } else {
            println!("cargo:warning=aien-omega-dattn: {dir} has no src/omega_gpu_decode_attn_api.h (omega before OWN-30B cut 8); fused attention is stubbed");
        }
    }
}

fn archive_defines(lib: &Path, symbol: &str) -> bool {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-dattn: nm not available, assuming the archive lacks the decode attention API");
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.ends_with(&format!(" T {symbol}")))
}
