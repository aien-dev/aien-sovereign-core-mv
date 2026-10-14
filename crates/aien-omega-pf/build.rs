//! Decides which parts of omega's prefill API are linked (pattern of `aien-omega-fp8/build.rs`).
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script; this crate only declares
//! the extra symbols, so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: `nm` decides per symbol.
//!   * `AIEN_OMEGA_DIR=<checkout>`: the headers decide (`src/omega_gpu_prefill.h`,
//!     `src/omega_gpu_prefill_attn_api.h`).
//!
//! `cfg(has_omega_pf)`: `omega_gpu_pf_qgemm` defined (quantiser, QMMA grouped GEMM, pf buffers).
//! `cfg(has_omega_pattn)`: `omega_gpu_prefill_attn` defined (host-pointer prefill attention).
//! `cfg(has_omega_pattn_dev)`: `omega_gpu_prefill_attn_dev` defined (device-pointer entry).
//! Otherwise every call returns `Unavailable`; a stub never fails the build.
use std::path::Path;
use std::process::Command;

fn main() {
    for c in ["has_omega_pf", "has_omega_pattn", "has_omega_pattn_dev"] {
        println!("cargo:rustc-check-cfg=cfg({c})");
    }
    println!("cargo:rustc-check-cfg=cfg(has_omega_pfhost)");
    for v in ["AIEN_FORCE_CPU_STUB", "AIEN_OMEGA_GPU_LIB", "AIEN_OMEGA_DIR"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_GPU_LIB") {
        println!("cargo:rerun-if-changed={lib}");
        let syms = defined_symbols(Path::new(&lib));
        let has = |s: &str| syms.iter().any(|d| d == s);
        if has("omega_gpu_pf_qgemm") && has("omega_gpu_pf_quant") && has("omega_gpu_pf_alloc_dev") {
            println!("cargo:rustc-cfg=has_omega_pf");
        } else {
            println!("cargo:warning=aien-omega-pf: {lib} does not define omega_gpu_pf_qgemm; prefill GEMM calls are stubbed");
        }
        if has("omega_gpu_prefill_attn") {
            println!("cargo:rustc-cfg=has_omega_pattn");
            if has("omega_gpu_prefill_attn_dev") {
                println!("cargo:rustc-cfg=has_omega_pattn_dev");
            }
        } else {
            println!("cargo:warning=aien-omega-pf: {lib} does not define omega_gpu_prefill_attn; prefill attention is stubbed");
        }
        // W-L1G-D: router logits + MoE combine (omega_gpu_pfhost.h)
        if has("omega_gpu_pf_qgemm") && has("omega_gpu_pf_router_logits") && has("omega_gpu_pf_moe_combine") && has("omega_gpu_pf_router_pack") && has("omega_gpu_pf_router_rows") {
            println!("cargo:rustc-cfg=has_omega_pfhost");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") {
        let pf = Path::new(&dir).join("src/omega_gpu_prefill.h");
        let pa = Path::new(&dir).join("src/omega_gpu_prefill_attn_api.h");
        println!("cargo:rerun-if-changed={}", pf.display());
        println!("cargo:rerun-if-changed={}", pa.display());
        // W-L1G-D: router logits + MoE combine (omega_gpu_pfhost.h)
        let ph = Path::new(&dir).join("src/omega_gpu_pfhost.h");
        println!("cargo:rerun-if-changed={}", ph.display());
        if std::fs::read_to_string(&ph).is_ok_and(|t| t.contains("omega_gpu_pf_router_logits(") && t.contains("omega_gpu_pf_moe_combine(")) {
            println!("cargo:rustc-cfg=has_omega_pfhost");
        }
        if std::fs::read_to_string(&pf).is_ok_and(|t| t.contains("omega_gpu_pf_qgemm(") && t.contains("omega_gpu_pf_quant(")) {
            println!("cargo:rustc-cfg=has_omega_pf");
        }
        if let Ok(t) = std::fs::read_to_string(&pa) {
            if t.contains("omega_gpu_prefill_attn(") {
                println!("cargo:rustc-cfg=has_omega_pattn");
                if t.contains("omega_gpu_prefill_attn_dev(") {
                    println!("cargo:rustc-cfg=has_omega_pattn_dev");
                }
            }
        }
    }
}

fn defined_symbols(lib: &Path) -> Vec<String> {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-pf: nm not available, assuming the archive lacks the prefill API");
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_once(" T ").map(|(_, s)| s.to_string()))
        .collect()
}
