//! Decides whether omega's FP8 GEMV API (`src/omega_gpu_fp8_api.h`, OWN-30B cuts 5-6) is linked.
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script (this crate only declares
//! the extra symbols), so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: native when `nm` finds
//!     `omega_gpu_fp8_gemv_grouped_f32` defined in that archive.
//!   * `AIEN_OMEGA_DIR=<checkout>`: native when the checkout has `src/omega_gpu_fp8_api.h`.
//!
//! `cfg(has_omega_fp8gemm)` (the prefill grouped GEMM) is set when the archive defines
//! `omega_gpu_fp8gemm_f32` or the checkout has `src/omega_gpu_fp8gemm_api.h`.
//! `cfg(has_omega_fp8_grouped_rule)` (L310 one-wave lever, `omega_gpu_fp8_grouped_ctas_auto`) is set when the
//! archive defines that symbol or the checkout header declares it.
//! `cfg(has_omega_fp8_single_rule)` (L310 lever 2, `omega_gpu_fp8_single_rows_rule`) likewise.
//!   * otherwise: stub (every call returns `Unavailable`).
//!
//! `cfg(has_omega_fp8)` is set only when native. A stub never fails the build: the bf16
//! path in the model code stays usable, and `is_native()` says which one a binary carries.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8_dev)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8gemm)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8_grouped_rule)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8_single_rule)");
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
        if archive_defines(Path::new(&lib), "omega_gpu_fp8_gemv_grouped_f32") {
            println!("cargo:rustc-cfg=has_omega_fp8");
            if archive_defines(Path::new(&lib), "omega_gpu_fp8_gemv_grouped_dev")
                && archive_defines(Path::new(&lib), "omega_gpu_fp8_tensor_entry")
            {
                println!("cargo:rustc-cfg=has_omega_fp8_dev");
            }
            if archive_defines(Path::new(&lib), "omega_gpu_fp8gemm_f32") {
                println!("cargo:rustc-cfg=has_omega_fp8gemm");
            }
            if archive_defines(Path::new(&lib), "omega_gpu_fp8_grouped_ctas_auto") {
                println!("cargo:rustc-cfg=has_omega_fp8_grouped_rule");
            }
            if archive_defines(Path::new(&lib), "omega_gpu_fp8_single_rows_rule") {
                println!("cargo:rustc-cfg=has_omega_fp8_single_rule");
            }
        } else {
            println!("cargo:warning=aien-omega-fp8: {lib} does not define the FP8 GEMV API; FP8 calls are stubbed");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") {
        let header = Path::new(&dir).join("src/omega_gpu_fp8_api.h");
        println!("cargo:rerun-if-changed={}", header.display());
        if header.is_file() {
            println!("cargo:rustc-cfg=has_omega_fp8");
            if std::fs::read_to_string(&header).is_ok_and(|t| {
                t.contains("omega_gpu_fp8_gemv_grouped_dev")
                    && t.contains("omega_gpu_fp8_tensor_entry")
            }) {
                println!("cargo:rustc-cfg=has_omega_fp8_dev");
            }
            if Path::new(&dir)
                .join("src/omega_gpu_fp8gemm_api.h")
                .is_file()
            {
                println!("cargo:rustc-cfg=has_omega_fp8gemm");
            }
            if std::fs::read_to_string(&header)
                .is_ok_and(|t| t.contains("omega_gpu_fp8_grouped_ctas_auto"))
            {
                println!("cargo:rustc-cfg=has_omega_fp8_grouped_rule");
            }
            if std::fs::read_to_string(&header)
                .is_ok_and(|t| t.contains("omega_gpu_fp8_single_rows_rule"))
            {
                println!("cargo:rustc-cfg=has_omega_fp8_single_rule");
            }
        } else {
            println!("cargo:warning=aien-omega-fp8: {dir} has no src/omega_gpu_fp8_api.h (omega before OWN-30B cut 5); FP8 calls are stubbed");
        }
    }
}

fn archive_defines(lib: &Path, symbol: &str) -> bool {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-fp8: nm not available, assuming the archive lacks the FP8 GEMV API");
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.ends_with(&format!(" T {symbol}")))
}
