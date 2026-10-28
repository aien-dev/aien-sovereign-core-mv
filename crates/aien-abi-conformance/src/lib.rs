//! ABI conformance for the omega C boundaries that sovereign-core mirrors by hand
//! (`aien-omega-compose` over `rxc_host_abi.h`, `aien-omega-gpu` over the `omega_gpu_*` headers).
//!
//! One checked-in **manifest** per boundary is the contract. It is checked from both sides:
//!
//! * **Rust side** (every `cargo test`, stub builds included): the Rust declarations are written
//!   through [`abi_struct!`], [`abi_externs!`], [`abi_fnptr!`] and [`abi_consts!`], which record
//!   every field (name, offset, size, kind, count, in declaration order), every extern signature,
//!   every callback type and every mirrored constant. [`rust_manifest`] renders them in manifest
//!   form and the crate's test compares that text with the checked-in file.
//! * **C side** (`build.rs`, whenever the real library is built from an omega checkout):
//!   [`check_c`] parses the pinned headers (field names and order, prototypes, callback typedefs),
//!   compiles and runs a probe that prints `sizeof`, `_Alignof`, `offsetof`, field widths and
//!   integer kinds (C11 `_Generic`) and the constant values, and lists the archive's defined
//!   symbols with `nm`. Any difference fails the build before anything links.
//!
//! So a field reorder that keeps the total size, a width or signedness change, a changed
//! parameter, a changed constant or a missing symbol cannot reach a qualified build.
//! [`identity`] adds the native archive digest checks (`native-artifacts.lock`).
//!
//! No dependencies; the C side needs `cc` and `nm` (both already required by the native build).

pub mod build;
pub mod features;
pub mod header;
pub mod identity;
pub mod manifest;
mod native;
pub mod probe;
mod rust;
pub mod source;

pub use manifest::{canon_sig, canon_type, Manifest};
pub use features::CanonicalFeatures;
pub use native::NativeIdentity;
pub use probe::{check_c, defined_symbols, CCheck};
pub use rust::{assert_manifest, rust_manifest, AbiStruct, CKind, Field, FnSig, RustAbi};
