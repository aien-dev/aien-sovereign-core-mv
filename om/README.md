# om: Sovereign Core in seeded Omega

First-party Omega (`.om`) replacements for parts of Sovereign Core, built by the seeded Omega compiler
(`seed/out/omc` from omega-mv, pinned by sha256 in each test). Rust stays the production path until a
replacement passes parity and real-execution tests against it ([#748](https://github.com/aien-dev/aien-sovereign-core-mv/issues/748),
aien-architecture-mv#300).

| Path | What it is | Status |
|---|---|---|
| `effects/world.om` | The restart world check of the effect ledger: reads the target file, hashes it, decides DONE / NOT_DONE / UNRESOLVED like `effects::state_from`, and writes the reconcile record text and its sha256 like `effects::reconcile_text`. | Byte-identical to the Rust on 24 fixtures (`effects/run.sh`). Not wired into the daemon. |
| `tur-ledger/tur-ledger.om` | Aienet-0 test-credit ledger CLI: v2 genesis pins issuer, independent verifier and account/node policy. Canonical mutations require issuer signatures; settlement also requires verifier authorization over signed-job, result and attestation digests and the reserved quote. Single-writer hash chain, fsync, recovery, integer conservation and exactly-once settle remain. See `tur-ledger/README.md`. | Host admission/durability tests and guard mutants in `tur-ledger/check.sh`; no wallet, Core RPC or public economic qualification. Historical unsigned v1 receipts remain historical. |
| `vendor/sha256.om` | Incremental SHA-256 in Omega, vendored from benchmarks-mv. | Exercised through `world.om` and `tur-ledger.om`. |
| `vendor/ec.om` | Native Omega ECDSA P-256/P-384 verification, byte-identical to omega-mv `6999fa9e6dae4847c155ef4eacaece74f491d3ca` `om/tls/ec.om`. | P-256/SHA-256 strict DER verification used by `tur-ledger.om`; OpenSSL is a test-only signing oracle, not a runtime dependency. |

Run: `OMEGA_OMC=<path to seed omc> bash om/effects/run.sh` (needs the offline cargo cache for the Rust
reference, `crates/aien-runtime/examples/om_world_reference.rs`).

Not covered yet: the confined check (`confined_world_state`, which needs a descriptor-relative open
with resolve flags in Omega), the journal write, ledger rebuild, and the steps before the effect
(objective, inference, typed tool, approval). The list with owners is in #748.
