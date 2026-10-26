# om: Sovereign Core in seeded Omega

First-party Omega (`.om`) replacements for parts of Sovereign Core, built by the seeded Omega compiler
(`seed/out/omc` from omega-mv, pinned by sha256 in each test). Rust stays the production path until a
replacement passes parity and real-execution tests against it ([#748](https://github.com/aien-dev/aien-sovereign-core-mv/issues/748),
aien-architecture-mv#300).

| Path | What it is | Status |
|---|---|---|
| `effects/world.om` | The restart world check of the effect ledger: reads the target file, hashes it, decides DONE / NOT_DONE / UNRESOLVED like `effects::state_from`, and writes the reconcile record text and its sha256 like `effects::reconcile_text`. | Byte-identical to the Rust on 24 fixtures (`effects/run.sh`). Not wired into the daemon. |
| `vendor/sha256.om` | Incremental SHA-256 in Omega, vendored from benchmarks-mv. | Exercised through `world.om`. |
| `dsv4/fp4dq.om` | DeepSeek-V4-Flash routed-expert decoder: packed 4-bit e2m1 codes (low nibble first) times the E8M0 scale of each 32 weights, emitted as exact float32 bit patterns with integers only. | Host only. Bit-identical to DeepSeek's own decoder (`inference/convert.py` FP4_TABLE x torch E8M0) on all 16 x 256 code/scale pairs and on six real layer-5 expert matrices; 7 refusals, 3 mutants caught (`OMEGA_OMC=<seed omc> [DSV4_SHARD7=<shard>] bash om/dsv4/run.sh`). Not wired into the engine. |

Run: `OMEGA_OMC=<path to seed omc> bash om/effects/run.sh` (needs the offline cargo cache for the Rust
reference, `crates/aien-runtime/examples/om_world_reference.rs`).

Not covered yet: the confined check (`confined_world_state`, which needs a descriptor-relative open
with resolve flags in Omega), the journal write, ledger rebuild, and the steps before the effect
(objective, inference, typed tool, approval). The list with owners is in #748.
