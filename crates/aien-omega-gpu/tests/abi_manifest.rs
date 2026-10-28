//! The Rust mirror of the canonical CUDApp GPU ABI must equal abi/omega_gpu.manifest,
//! field by field (offset, size, kind, order), constant by constant and signature by
//! signature. Runs in every build, the stub included.

#[test]
fn rust_ffi_matches_the_manifest() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("abi/omega_gpu.manifest");
    aien_abi_conformance::assert_manifest(&aien_omega_gpu::ffi::rust_abi(), &path);
}

/// Both actual link inputs, with the canonical ABI and companion admission kept separate.
/// The companion has no Rust-facing GPU ABI; it must not fabricate `abi=pass`.
#[test]
fn native_identity_matches_the_link() {
    let id = aien_omega_gpu::native_identity();
    let companion = aien_omega_gpu::native_companion_identity();
    println!("AIEN_NATIVE_IDENTITY {}", id.to_json());
    println!("AIEN_NATIVE_IDENTITY {}", companion.to_json());
    println!("AIEN_NATIVE_PAIR_QUALIFIES {}", aien_omega_gpu::native_pair_qualifies());
    assert_eq!(id.component, "libcudapp.a");
    assert_eq!(companion.component, "libomega_gpu.a");
    assert_eq!(companion.abi, "none");
    if cfg!(has_omega_gpu) {
        assert_ne!(id.link, "stub", "{id:?}");
        assert!(id.digest != "none", "{id:?}");
        assert_eq!(id.source_sha, aien_omega_gpu::PINNED_CUDAPP_SHA);
        assert_eq!(companion.source_sha, aien_omega_gpu::PINNED_OMEGA_SHA);
        assert_eq!(companion.link, "pinned-source");
        assert_ne!(companion.digest, "none");
    } else {
        assert_eq!(id.link, "stub", "{id:?}");
        assert!(!id.qualifies());
        assert_eq!(companion.link, "stub");
        assert!(!aien_omega_gpu::native_pair_qualifies());
    }
    if std::env::var("AIEN_ABI_QUALIFY").as_deref() == Ok("1") {
        assert!(
            aien_omega_gpu::native_pair_qualifies(),
            "canonical GPU/companion pair does not qualify: {id:?}, {companion:?}"
        );
    }
}
