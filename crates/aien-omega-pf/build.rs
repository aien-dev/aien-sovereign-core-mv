//! Decides which parts of omega's prefill API are linked.
//!
//! Canonical capabilities are inspected from the admitted canonical archive exported by
//! `aien-omega-gpu` via Cargo DEP metadata.
//!
//! `cfg(has_omega_pf)`: `omega_gpu_pf_qgemm`, `omega_gpu_pf_quant`, `omega_gpu_pf_alloc_dev` defined (quantiser, QMMA grouped GEMM, pf buffers).
//! `cfg(has_omega_pattn)`: `omega_gpu_prefill_attn` defined (host-pointer prefill attention).
//! `cfg(has_omega_pattn_dev)`: `omega_gpu_prefill_attn_dev` defined (device-pointer entry).
//! `cfg(has_omega_pfhost)`: prefill router logits, MoE combine, router pack and rows.
//! Otherwise every call returns `Unavailable`; a stub never fails the build.
use aien_abi_conformance::CanonicalFeatures;

fn main() {
    for c in ["has_omega_pf", "has_omega_pattn", "has_omega_pattn_dev", "has_omega_pfhost"] {
        println!("cargo:rustc-check-cfg=cfg({c})");
    }
    let Some(cf) = CanonicalFeatures::from_dep("aien-omega-pf") else {
        return;
    };
    let has_pf = cf.has_symbol("omega_gpu_pf_qgemm")
        && cf.has_symbol("omega_gpu_pf_quant")
        && cf.has_symbol("omega_gpu_pf_alloc_dev");
    if has_pf {
        println!("cargo:rustc-cfg=has_omega_pf");
    } else {
        println!(
            "cargo:warning=aien-omega-pf: {} does not define omega_gpu_pf_qgemm; prefill GEMM calls are stubbed",
            cf.archive().display()
        );
    }
    if cf.has_symbol("omega_gpu_prefill_attn") {
        println!("cargo:rustc-cfg=has_omega_pattn");
        if cf.has_symbol("omega_gpu_prefill_attn_dev") {
            println!("cargo:rustc-cfg=has_omega_pattn_dev");
        }
    } else {
        println!(
            "cargo:warning=aien-omega-pf: {} does not define omega_gpu_prefill_attn; prefill attention is stubbed",
            cf.archive().display()
        );
    }
    // W-L1G-D: router logits + MoE combine (omega_gpu_pfhost.h)
    if has_pf
        && cf.has_symbol("omega_gpu_pf_router_logits")
        && cf.has_symbol("omega_gpu_pf_moe_combine")
        && cf.has_symbol("omega_gpu_pf_router_pack")
        && cf.has_symbol("omega_gpu_pf_router_rows")
    {
        println!("cargo:rustc-cfg=has_omega_pfhost");
    }
}
