//! Shared build-time helpers for optional omega crates to discover admitted canonical capabilities.
//!
//! All optional crates derive their native symbols and configuration flags from the admitted
//! canonical archive exported by `aien-omega-gpu` via Cargo DEP metadata.
//! No obsolete raw headers or prebuilt library paths (`AIEN_OMEGA_GPU_LIB`) are used.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::build;
use crate::identity;
use crate::probe;

/// Admitted canonical archive metadata and defined global symbols.
#[derive(Debug, Clone)]
pub struct CanonicalFeatures {
    archive: PathBuf,
    sha256: String,
    symbols: BTreeSet<String>,
}

impl CanonicalFeatures {
    /// Inspect the admitted canonical archive exported by `aien-omega-gpu` via Cargo DEP metadata.
    ///
    /// Fails closed (panics) on malformed metadata, hash mismatch, or inspection failure.
    /// Returns `None` only when `aien-omega-gpu` built as a stub and native is not required.
    pub fn from_dep(krate: &str) -> Option<Self> {
        println!("cargo:rerun-if-env-changed=AIEN_FORCE_CPU_STUB");
        println!("cargo:rerun-if-env-changed=AIEN_REQUIRE_NATIVE_LIBS");
        println!("cargo:rerun-if-env-changed=AIEN_OMEGA_GPU_LIB");
        println!("cargo:rerun-if-env-changed=DEP_AIEN_OMEGA_GPU_CANONICAL_ARCHIVE");
        println!("cargo:rerun-if-env-changed=DEP_AIEN_OMEGA_GPU_CANONICAL_SHA256");

        assert!(
            std::env::var_os("AIEN_OMEGA_GPU_LIB").is_none(),
            "{krate}: AIEN_OMEGA_GPU_LIB is obsolete; canonical capability decisions derive only from admitted dependency metadata"
        );

        if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
            build::refuse_stub_if_required(krate);
            return None;
        }

        let archive_str = match std::env::var("DEP_AIEN_OMEGA_GPU_CANONICAL_ARCHIVE") {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                if build::require_native() {
                    panic!(
                        "{krate}: AIEN_REQUIRE_NATIVE_LIBS=1 refuses the stub build (canonical archive metadata not provided by aien-omega-gpu)"
                    );
                }
                return None;
            }
        };

        let sha256_expected = match std::env::var("DEP_AIEN_OMEGA_GPU_CANONICAL_SHA256") {
            Ok(s) if identity::is_sha256(&s) => s,
            _ => {
                panic!(
                    "{krate}: malformed or missing DEP_AIEN_OMEGA_GPU_CANONICAL_SHA256 metadata for canonical archive {archive_str}"
                );
            }
        };

        let archive = PathBuf::from(archive_str);
        if !archive.is_file() {
            panic!(
                "{krate}: canonical archive metadata points to non-existent file: {}",
                archive.display()
            );
        }

        println!("cargo:rerun-if-changed={}", archive.display());

        let actual_hash = identity::sha256_file(&archive).unwrap_or_else(|e| {
            panic!(
                "{krate}: cannot hash canonical archive {}: {e}",
                archive.display()
            );
        });

        if actual_hash != sha256_expected {
            panic!(
                "{krate}: canonical archive digest mismatch for {}: metadata expected {sha256_expected}, got {actual_hash}",
                archive.display()
            );
        }

        let sym_vec = probe::defined_symbols(&archive).unwrap_or_else(|e| {
            panic!(
                "{krate}: cannot inspect defined symbols in canonical archive {}: {e}",
                archive.display()
            );
        });

        let symbols = sym_vec.into_iter().collect();

        Some(Self {
            archive,
            sha256: actual_hash,
            symbols,
        })
    }

    pub fn archive(&self) -> &Path {
        &self.archive
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn has_symbol(&self, name: &str) -> bool {
        self.symbols.contains(name)
    }

    pub fn has_all_symbols(&self, names: &[&str]) -> bool {
        names.iter().all(|name| self.has_symbol(name))
    }
}
