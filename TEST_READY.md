# TEST_READY: Sovereign Distillation Workshop Test Suite

**Record, not current release status (2026-10-16).** This report describes a run on 2026-09-20 and is kept as a record. It is not part of the release path. Its Tier 4 "model seat on port 18006" roundtrip used the Modular MAX serving endpoint of that time; MAX was removed from the project on 2026-10-15 (#529), and the inference path now being qualified runs on AIEN's own Omega kernels (see `README.md`). The tier test files now live in `crates/spark-adapters/tests/`, not in a top-level `tests/` folder.

## 1. Status: Ready for Milestone Verification

The end-to-end test harness for the Sovereign Distillation Workshop is complete, verified, and operational on NVIDIA DGX Spark (`spark`).

The suite provides 140 deterministic test cases across 4 tiers covering all acceptance criteria in R1 through R6 of ORIGINAL_REQUEST.md.

## 2. Test Suite Composition

| Tier | Category | Test Count | Description |
|------|----------|------------|-------------|
| Tier 1 | Feature Coverage | 65 | 5 tests per feature across 13 core features (vault resolution, prompt sanitizer, cortex auth, resident bounds, crawler flags, cli workbench, key verification, axum routes, cortex storage, dataset emission, verification gates, test harness, git parity) |
| Tier 2 | Boundary and Corner Cases | 65 | Negative, boundary, and stress tests (empty inputs, missing secrets, invalid CLI flags, malformed JSON, delimiter parsing, sycophancy detection, unslop penalties, zero/negative DPO deltas, duplicate rejection) |
| Tier 3 | Cross-Feature Combinations | 5 | Multi-subsystem interactions: Sanitizer + Router + local seat dispatch, Verifier + Consensus + Cortex gate, Crawler + Curriculum + SFT/DPO dataset emission, Vault token + Cortex HTTP dispatch, Verifier unslop + rustc + SFT emission |
| Tier 4 | Real-World Scenarios | 5 | Live execution on DGX Spark: spark-distill verify-keys catalog audit, live model seat inference roundtrip on port 18006, systems curriculum batch generation, live Cortex daemon handshake on port 18080, end-to-end SFT/DPO filesystem dataset emission |
| **Total** | **All Tiers** | **140** | **Complete opaque-box distillation workshop verification** |

## 3. Baseline Execution Results on NVIDIA DGX Spark

Baseline executed against live workspace state on 2026-09-20:
- Total tests executed: 140
- Passed: 140
- Failed: 0
- Ignored: 0
- Execution duration: 132 seconds
- Results artifact: `tests_results.json`

### Tier Execution Summary

| Tier | Target Name | Passed | Failed | Status |
|------|-------------|--------|--------|--------|
| Tier 1 | `tests/tier1_features.rs` | 65 | 0 | PASSED |
| Tier 2 | `tests/tier2_boundary.rs` | 65 | 0 | PASSED |
| Tier 3 | `tests/tier3_cross_feature.rs` | 5 | 0 | PASSED |
| Tier 4 | `tests/tier4_real_world.rs` | 5 | 0 | PASSED |
| **Total** | **All 4 Targets** | **140** | **0** | **100% PASS** |

## 4. Feature Checklist Table

| # | Feature | Requirement | Tier 1 Tests | Tier 2 Tests | Status |
|---|---------|-------------|--------------|--------------|--------|
| F1 | In-Memory Vault Secret Resolution | ORIGINAL_REQUEST §R1 | T1_F01 to T1_F05 (5) | T2_F01 to T2_F05 (5) | VERIFIED |
| F2 | Outbound Prompt Sanitizer Hardening | ORIGINAL_REQUEST §R2 | T1_F06 to T1_F10 (5) | T2_F06 to T2_F10 (5) | VERIFIED |
| F3 | Cortex Token Dynamic Auth | ORIGINAL_REQUEST §R1 | T1_F11 to T1_F15 (5) | T2_F11 to T2_F15 (5) | VERIFIED |
| F4 | Resident Generation Bounds | ORIGINAL_REQUEST §R1 | T1_F16 to T1_F20 (5) | T2_F16 to T2_F20 (5) | VERIFIED |
| F5 | Crawler Flag Collision Fix | ORIGINAL_REQUEST §R4 | T1_F21 to T1_F25 (5) | T2_F21 to T2_F25 (5) | VERIFIED |
| F6 | Interactive CLI Workbench | ORIGINAL_REQUEST §R4 | T1_F26 to T1_F30 (5) | T2_F26 to T2_F30 (5) | VERIFIED |
| F7 | Key Verification Command | ORIGINAL_REQUEST §R4 | T1_F31 to T1_F35 (5) | T2_F31 to T2_F35 (5) | VERIFIED |
| F8 | Axum Web Workshop Service | ORIGINAL_REQUEST §R4 | T1_F36 to T1_F40 (5) | T2_F36 to T2_F40 (5) | VERIFIED |
| F9 | Closed-Loop Cortex Storage | ORIGINAL_REQUEST §R5 | T1_F41 to T1_F45 (5) | T2_F41 to T2_F45 (5) | VERIFIED |
| F10 | SFT and DPO Dataset Emission | ORIGINAL_REQUEST §R5 | T1_F46 to T1_F50 (5) | T2_F46 to T2_F50 (5) | VERIFIED |
| F11 | Four-Tier Verification Gates | ORIGINAL_REQUEST §R3 | T1_F51 to T1_F55 (5) | T2_F51 to T2_F55 (5) | VERIFIED |
| F12 | E2E Opaque-Box Test Suite | ORIGINAL_REQUEST §Acceptance | T1_F56 to T1_F60 (5) | T2_F56 to T2_F60 (5) | VERIFIED |
| F13 | Remote Git Parity & PR Lifecycle | ORIGINAL_REQUEST §R6 | T1_F61 to T1_F65 (5) | T2_F61 to T2_F65 (5) | VERIFIED |

## 5. Test Execution Instructions

Execute tests on the NVIDIA DGX Spark from the repository root:

```bash
# Execute entire 140-test suite via runner script
./scripts/e2e_distill_test.sh --all

# Execute by tier
./scripts/e2e_distill_test.sh --tier 1
./scripts/e2e_distill_test.sh --tier 2
./scripts/e2e_distill_test.sh --tier 3
./scripts/e2e_distill_test.sh --tier 4

# Execute by feature pattern
./scripts/e2e_distill_test.sh --feature sanitizer
./scripts/e2e_distill_test.sh --feature vault
./scripts/e2e_distill_test.sh --feature crawler

# Output structured JSON results
./scripts/e2e_distill_test.sh --all --json

# Direct cargo test invocations
cargo test -p spark-adapters --test tier1_features
cargo test -p spark-adapters --test tier2_boundary
cargo test -p spark-adapters --test tier3_cross_feature
cargo test -p spark-adapters --test tier4_real_world
```

## 6. Artifact Index

- Tier 1 Feature Test Suite: `crates/spark-adapters/tests/tier1_features.rs`
- Tier 2 Boundary Test Suite: `crates/spark-adapters/tests/tier2_boundary.rs`
- Tier 3 Cross-Feature Suite: `crates/spark-adapters/tests/tier3_cross_feature.rs`
- Tier 4 Real-World Suite: `crates/spark-adapters/tests/tier4_real_world.rs`
- Unified Shell Test Runner: `./scripts/e2e_distill_test.sh`
- Machine-Readable Results: `tests_results.json`
- Readiness Report: `TEST_READY.md`
