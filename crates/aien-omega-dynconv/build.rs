//! Decides whether omega's DFlash 2 dynamic conv (`omega_gpu_dynconv2_f32` in
//! `src/omega_gpu_elementwise_api.h`, omega branch cascade/dflash-drafter) is linked.
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script (this crate only declares the
//! extra symbol), so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: native when `nm` finds `omega_gpu_dynconv2_f32` defined.
//!   * `AIEN_OMEGA_DIR=<checkout>`: native when the checkout's elementwise header declares it.
//!   * otherwise: stub (every call returns `Unavailable`; the drafter keeps the host conv).
//!
//! `cfg(has_omega_dynconv)` is set only when native.
use std::path::Path;
use std::process::Command;

const SYMBOL: &str = "omega_gpu_dynconv2_f32";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_dynconv)");
    for v in [
        "AIEN_FORCE_CPU_STUB",
        "AIEN_OMEGA_GPU_LIB",
        "AIEN_OMEGA_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_GPU_LIB") {
        println!("cargo:rerun-if-changed={lib}");
        if archive_defines(Path::new(&lib)) {
            println!("cargo:rustc-cfg=has_omega_dynconv");
        } else {
            println!("cargo:warning=aien-omega-dynconv: {lib} does not define {SYMBOL}; stubbed");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") {
        let header = Path::new(&dir).join("src/omega_gpu_elementwise_api.h");
        println!("cargo:rerun-if-changed={}", header.display());
        let declared = std::fs::read_to_string(&header)
            .map(|s| s.contains(SYMBOL))
            .unwrap_or(false);
        if declared {
            println!("cargo:rustc-cfg=has_omega_dynconv");
        } else {
            println!("cargo:warning=aien-omega-dynconv: {dir} does not declare {SYMBOL}; stubbed");
        }
    }
}

fn archive_defines(lib: &Path) -> bool {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-dynconv: nm not available, assuming {SYMBOL} is absent");
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.ends_with(&format!(" T {SYMBOL}")))
}
