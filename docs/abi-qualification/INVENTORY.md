# Omega and AIEN ABI qualification: inventory (2026-10-11)

Scope: mission sections 4 to 10 (`~/handoffs/2026-10-11-abi-qualification-mission.md`).
Sources read at: omega `6c6180cf` (omega.lock; ABI headers identical to omega main `82f31173`),
physics `6d7cf0d4`, aienos `b84c0a67` (omega aienos.lock) and aienos main `3b2004b6`,
sovereign-core main `bd438bcc`. Tags: **implemented** (true before this work), **fixed here**
(defect or gap closed by the abi-qual PRs), **partial**, **missing**, **unnecessary**.
`sc` = this repository, `om` = omega, `ao` = aienos.

## Boundaries

| Boundary | Producer | Consumer | Linked into production |
|---|---|---|---|
| rxc_host v2 (composition facade) | om `src/runtime/rxc_host_abi.h`, `.c`; archive `librx_compose.a` (`mk/rx_compose_lib.mk`) | sc `crates/aien-omega-compose` | yes (aien-runtime, aien-cli) |
| omega GPU engine | om `src/omega_gpu_{matmul_api,session,serving,elementwise_api,attention_api}.h`; `libomega_gpu.a` | sc `crates/aien-omega-gpu`, used by `aien-inference-abi` | yes on the GB10 build |
| Aienux capability authority (C, 64-bit generation) | ao `native/capability/aienos_capability.h` | om `src/runtime/aienos_cap.h` (hand copy), merged into `librx_compose.a` | yes, through librx_compose.a |
| Aienux capability (Rust, 32-bit generation) | ao `crates/aienos-capability-ffi` | nothing in sc, om or ao links it (grep, 2026-10-11) | no |

## Section 4: ABI compatibility (rxc_host and GPU)

| Item | Before | Now | Evidence |
|---|---|---|---|
| 4.1 size and alignment | partial: rxc_host checks version + 4 sizes at open (om `rxc_host_abi.c:458-465`, sc `aien-omega-compose/src/lib.rs:238`); GPU none | fixed here | manifests `crates/*/abi/*.manifest` (`struct Name size align`), C probe `sizeof`/`_Alignof` in build.rs |
| 4.2 offsets and order | missing (size only; an equal-size reorder passed) | fixed here | manifest `field` lines in order; header parse checks names and order; test `c_side.rs` equal-size reorder |
| 4.3 widths and signedness | missing | fixed here | C11 `_Generic` kind per field and per constant; tests width + signedness |
| 4.4 enum representation | unnecessary as a type: rxc_host enums are anonymous constant lists and every mirrored function takes and returns plain integers; the one typed enum (`OmegaGpuEwOp`, om `omega_gpu_elementwise_api.h:57-88`) appears only in functions Rust does not mirror | values checked as constants | `const` lines (RXC_HOST_E_*, OMEGA_GPU_RC_*) |
| 4.5 bool representation | unnecessary: no `bool` field or parameter crosses; flags are u32 | guarded | kind check refuses bool vs u8 (test `bool_to_u8`) |
| 4.6 function pointers, calling convention | partial: typed `extern "C"` only | fixed here | `fnptr` + `fn` signature lines from header prototypes; tests param + callback change |
| 4.7 pointer validity, ownership | implemented: safe wrappers own buffers, check lengths before FFI | kept; GPU `pool` param drift fixed (`*const u8` vs `const void *`) | sc `aien-omega-gpu/src/ffi.rs`, found by the C check |
| 4.8 buffer lengths, overflow | partial: GPU length checks multiplied `usize` unchecked (wraps in release) | fixed here | `prod()` sc `aien-omega-gpu/src/lib.rs:121`; test `overflowing_shapes_are_refused_before_ffi` (red on main: overflow panic at lib.rs:745) |
| 4.9 callback lifetime, concurrency, cancellation | implemented: boxes live until `rxc_host_close` returns, close stops workers (sc lib.rs:480-485, om rxc_host_abi.c:448-452); slot table under `g_mu` (om :290) | tested here | `tests/callbacks.rs` 12 open/close cycles (> 4 slots), 3 concurrent homes |
| 4.10 panic and error propagation | implemented: `catch_unwind` in both trampolines (sc lib.rs:218, 232); release `panic = "abort"` (sc Cargo.toml:64) so no unwind reaches C in either profile | strengthened: null `result`/`ctx` refused | `tests/callbacks.rs` panicking Skill and verifier |
| runtime negotiation | rxc_host version + sizes before any other call | kept, unchanged | no new producer descriptor needed: the build-time C check covers layout fully |

## Section 5: handwritten drift

Implemented here: one checked-in manifest per boundary, checked from both sides
(`crates/aien-abi-conformance`). Rust side: the `abi_struct!`/`abi_externs!`/`abi_fnptr!`/
`abi_consts!` declarations render a manifest compared with the file in every `cargo test`
(stub CI included). C side, in build.rs before linking: header parse, compiled layout probe,
`nm` symbol list. No binding generator, no new external dependency (needs `cc` and `nm`,
already required). Results on the Spark: compose 4 structs, 13 functions, 2 callback types,
95 facts; GPU 6 structs, 31 functions, 108 facts; all match. Twelve synthetic drift tests
(`crates/aien-abi-conformance/tests/c_side.rs`) cover reorder, width, signedness, bool,
parameter, callback, version constant, removed field, missing symbol, truncated archive,
missing struct.

## Section 6: Aienux capability

Ruling: two separate implementations that exported the same 17 `aienos_cap_*` symbols with
incompatible `AienosCapRef` layouts (C 16 bytes, 64-bit generation; Rust 8 bytes, 32-bit).
Not linked together today, but interchangeable by link order. aienos `abi-qual/cap-width`
renames the Rust exports to `aienos_cap32_*`, adds `aienos_cap32_abi_layout`, pins the C
layout with `_Static_assert`, adds a symbol-isolation and negative-link script, and fixes a
Rust restart defect (generation reuse after restart, already fixed in C by b1fa747).
Generations were not widened. omega `abi-qual/aienos-cap-conformance` checks
`src/runtime/aienos_cap.h` against the pinned aienos header (25 layout facts, 27
declarations, 8 refused mutants). Stale ref, recycled slot, exhaustion, unauthorized use:
C side implemented before (`ao native/capability/capability_test.c`); Rust side fixed there.

## Section 7: native artifact identity

| Item | Before | Now |
|---|---|---|
| omega.lock checkout check | implemented (compose build.rs, GPU build.rs) | kept |
| prebuilt `*_LIB` | accepted with no digest | fixed: `*_LIB_SHA256` required and matched, every manifest function must be defined; identity `prebuilt`, never qualifies |
| stub under `AIEN_REQUIRE_NATIVE_LIBS=1` | `AIEN_FORCE_CPU_STUB=1` silently won | fixed: refused |
| expected vs actual digest | missing | fixed: `native-artifacts.lock` (archive, omega sha, target, cc version, sha256); mismatch fails the build, unlisted combination warns and does not qualify |
| provenance in the binary | missing | fixed: `native_identity()` in both crates (link kind, ABI result, digest state, archive sha256, omega, aienos.lock, physics.lock, target, cc) |
| sovereign-core commit, flags, tools | missing | `scripts/abi-qualify.sh` evidence.json |

## Section 8: safe cross-language execution

Implemented before and confirmed: no unwind across C (catch_unwind + release abort);
callback contexts outlive close; `Compose` is `Send` not `Sync` (sc lib.rs:196);
`ResidentTensor` is `Send` not `Sync` and held in a `Mutex` (sc aien-omega-gpu lib.rs:248-261,
aien-inference-abi resident.rs:36); omega serializes GPU calls with the session lock
(om `omega_gpu_session.c:28,67`). Fixed here: null result pointer in the Skill trampoline,
usize overflow in length checks. Noted, not changed: `check_pool` checks only
`pool.len() >= pool_bytes` and trusts omega for stride bounds (sc lib.rs:939);
`rxc_host_set_wait_ms` returns `RX_ERR_ARG`, not `RXC_HOST_E_ARG` (om rxc_host_abi.c:338-339;
both are -1 (om `rx_world.h:117`, `rxc_host_abi.h` RXC_HOST_E_ARG), so callers see the right code).

## Sections 9 and 10: tests, CI, entry point

Covered by tests: compatible pinned build, version mismatch, equal-size reorder, signature
change, wrong prebuilt hash, wrong omega commit, missing symbol, truncated archive, callback
failure and panic, invalid lengths and overflow, repeated init/destroy, concurrent
callbacks, explicit stub, required-native with no library (`scripts/test-native-identity.sh`,
16 cases, in the required CI job), CPU vs GPU path identity (`native_identity()`), GPU
parity (existing chip suites, run by `scripts/abi-qualify.sh --chip`).
Entry point: `scripts/abi-qualify.sh [--chip]`, exit 0 PASS, 1 FAIL, 2 INCOMPLETE.
Registered in the local verification gates (ADR 0033): `tests/gates/sc-native-identity.gate`
(step `native-identity` in `tests/gates/run.sh`), and `native-artifacts.lock` is an input of
SC-TEST and SC-LINKED-COMPOSE. Partial: the GB10 part of `abi-qualify.sh --chip` stays an
operator run with its own evidence.json (no GB10 gate class exists in the manifests yet).
Not wired: the GPU C-side check on hosted CI (the GPU archive builds only on the Spark).
