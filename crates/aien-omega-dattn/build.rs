//! Decides whether omega's one-launch decode attention (`src/omega_gpu_decode_attn_api.h`,
//! OWN-30B cut 8 on the omega side) is linked.
//!
//! Canonical capabilities are inspected from the admitted canonical archive exported by
//! `aien-omega-gpu` via Cargo DEP metadata.
//!
//! `cfg(has_omega_dattn)` is set only when native. A stub never fails the build: the engine
//! keeps its split attention path (host norm/rope, paged KV), and `is_native()` says which
//! one a binary carries.
use aien_abi_conformance::CanonicalFeatures;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_dattn)");
    let Some(cf) = CanonicalFeatures::from_dep("aien-omega-dattn") else {
        return;
    };
    if cf.has_symbol("omega_gpu_decode_attn") {
        println!("cargo:rustc-cfg=has_omega_dattn");
    } else {
        println!(
            "cargo:warning=aien-omega-dattn: {} does not define the decode attention API; fused attention is stubbed",
            cf.archive().display()
        );
    }
}
