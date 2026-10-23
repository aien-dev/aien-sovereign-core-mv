# Core #679 readiness and retirement ledger

Inspected 2026-10-23. None of #710..#713 is complete. No serving artifact or
speed/quality result is claimed by this change.

## Reconciled source versus effective consumer

| Input | Inspected identity | Disposition |
|---|---|---|
| Core GitHub main | 754c07b118f41ad899607b60329bd851b1ccb13e | No qwen38 path in this tree |
| Core main omega.lock | 4799f9eba563e9c69579b47c477e7d9c2b0f15cb | Different from experimental backend |
| Omega GitHub main | a8ffd2b912c84197d08f963b05f9c7d8b30a3f8a | Not automatically the linked backend |
| CUDApp GitHub main | 232332422bfdc31c8918085b50f13c1e18ab702e | Qualification #33 open; local primary checkout differs |
| core feat/qwen38-27b | 3581a9882fa48dca761c0b4024c983c43377351e | Existing proposed implementation |
| core cascade/mtp-chain | 230574e0f731ca441f5d64b03a7946710d788f08 | Experimental Rust MTP; HTTP ordinary decode |
| core cascade/w4-decode | 1c36c3d3b9e9a13ecefca7ff73cf4bdb2877e59a | Existing quantization study; not final native W4 |
| core cascade/tree-verify-wire | 3b2dcd9168d73925d8dc0836adaf9a9e4dca022c | Recorded first hardware attempt failed |
| all four branch omega.lock files | 7c2ae49e22b5d4d771e6ae8757a03975a73c5abd | Declared pin only |
| historical MTP actual linked archive | Omega 37034f03, SHA256 6e90520913032ee8653ebbe325c485d329daa0a749840303a44b68b97ca17bff | Differs from declared lock; must bind actual binary |
| historical MTP executable | SHA256 9aece7168e8ce5baa42fd0dce97f19858afcb2899dfd2e7f06d54fb475d051f8 | Contaminated receipt excluded |

Branch identities above come from inspected remote-tracking refs. Revalidate
against the server before integration; current main identities were checked
against GitHub. Open Omega #547..#553/#559 and #614..#618 retain their ownership.
No PR has been treated as merged qualification. Core #642 and #676 and CUDApp
#33 remain the serving/layout/qualification boundaries.

Read-only process inspection found the actual port-18091 service running
Qwen3.6-35B from `models/qwen3.6-35b-a3b-fp8-95a723d0`, max context 49152,
token cap 16384, thinking off, batch slots 8. Its actual executable is
`.local/state/aien-q36/bin-19173f9e/aien-serve-http`, SHA256
`19173f9ec3b609fe79af644f84741d8808b92ea8eb3ecec4bc945af6f22b3aeb`.
It is not the target 27B baseline. It was neither restarted nor measured.
The colocated build receipt names Core `ship/motor-l387` f0230a59 (code
59b05a71) and Omega `ship/motor-l387` 6de999da. The colocated archive was
rehashed as `110295ef23dc7561eea371636347c005835f37b82c22f92401aa8a887a3ca990`.
This establishes the observed artifact plus recorded build provenance; no
independent rebuild or process-environment verification was performed. These
35B identities are not substituted for missing Qwen3.8 candidate provenance.

## Acceptance mapping and blockers

| Issue | Delivered here | Remaining acceptance |
|---|---|---|
| #710 | Native per-phase median/p95 and useful-output accounting, contamination/order refusals, synthetic controls, predeclared thresholds, source/pin reconciliation, restored and rehashed 81-file checkpoint | Freeze exact token arrays and held-out corpora/reference outputs; bind full native serving binary; calibrate hardware envelope; all real context/profile paired trials; transient-state memory and confidence intervals |
| #711 | No W4 acceleration claim; old FP8-backed simulation identified | Native packed layout/loader, GEMV and small-row GEMM, all packed/layout numerical controls, quality study, physical traffic/allocation and full-model W4/W4+MTP measurements |
| #712 | Receipt accounting includes draft/verify/commit and pending consumed prefix | Qualified MTP head, real endpoint integration, bounded adaptation/fallback, sampling distribution and all rejection/cancel/state tests, four-arm measurements |
| #713 | Arithmetic boundary refuses mismatched consumed count or failed state assertion | Actual KV/GDN/conv/drafter commitment, restore-versus-prefill, branch/dependency controls, transient bytes and commit costs, measured tree disposition |

L408 owns compiler/FR4 floats, FR15 clocks, FR16 sockets and the seed/IR path.
L605 owns the active native device continuation. Its open/ioctl work is not yet
an acknowledged model allocation/upload/dispatch/wait/readback interface.
L419 is the existing runtime/GPU migration slot but currently unassigned;
no active implementer is established by that ledger entry. L739's existing native serving
integration and L720 numerical primitives must be reused where applicable;
their host primitives do not establish a Qwen3.8 forward pass. No dirty owner
tree was edited. The native device/serving boundary needs immutable qualified
interfaces before full-model integration and replacement deletion.

## Replacement audit (all pending)

| Active implementation to retire | Existing successor boundary | Delete only after |
|---|---|---|
| `crates/aien-inference-abi/src/qwen38_omega/w4.rs` and W4 study examples | #711 and shared layouts #676 | Native packing/numerics and actual consumer/quality gates |
| `qwen38_omega/spec.rs`, speculative examples, experimental tree glue | #712/#713 | Native verification/sampling/all-state integration gates |
| `qwen38_omega/session.rs`, `prefill.rs`, qwen38 engine/model loader | #642/#713 | Complete native session/prefill/continuation equivalence |
| `crates/aien-serve-http/src/qwen38_engine.rs` and its Rust consumer/build references | #642 and native serving owners | Actual `.om` endpoint, protocol, cancellation and hardware tests |
| `crates/aien-omega-q38` and corresponding GPU ABI/build glue | L419/L605, CUDApp #5/#33 | Native model-to-device execution closure |
| Scoped Omega/CUDApp C kernels, codegen/runtime and their `.osc` sources/tests/build entries | Existing Omega #599/#638 migration owners | Native seeded implementation, independent controls and final linked consumer |
| First-party qualification scripts/tests in these paths | #710 and existing native test/host facility owners | Native executable test/build orchestration plus full acceptance replay |

This table identifies retirement groups, not a completed file-by-file deletion
audit. Final delivery must enumerate every removed file/build reference and
the replacement gate. No implementation was deleted prematurely. Historical
source and immutable evidence stay in Git history. No retained compatibility
implementation or wrapper is accepted as the final state.
