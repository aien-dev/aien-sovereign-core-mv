//! Decides whether omega's MoE window API (`src/omega_gpu_moe_api.h`, OWN-30B cut 2) is linked.
//!
//! Canonical capabilities are inspected from the admitted canonical archive exported by
//! `aien-omega-gpu` via Cargo DEP metadata.
//!
//! `cfg(has_omega_moe)` is set only when native. A stub never fails the build: the whole-tensor
//! fallback in the model code stays usable, and `is_native()` says which one a binary carries.
use aien_abi_conformance::CanonicalFeatures;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_moe)");
    let Some(cf) = CanonicalFeatures::from_dep("aien-omega-moe") else {
        return;
    };
    if cf.has_symbol("omega_gpu_matmul_resident_windows_f32") {
        println!("cargo:rustc-cfg=has_omega_moe");
    } else {
        println!(
            "cargo:warning=aien-omega-moe: {} does not define the MoE window API; window calls are stubbed",
            cf.archive().display()
        );
    }
}
