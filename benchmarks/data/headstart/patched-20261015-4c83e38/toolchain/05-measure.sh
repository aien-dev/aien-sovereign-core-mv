#!/usr/bin/env bash
# Stage 5: the M5 A/B measurement through the shared M0 runner (extended).
# Same patched toolchain for both arms; only CARGO_UNSTABLE_HEADSTART differs.
set -euo pipefail
E=~/workspace/headstart-exp
H=$E/headstart
WT=~/aien-sovereign-core/.claude/worktrees/agent-adc52ad674d7b621a
export HS_COMMIT=4c83e389e8fa8790af6f3199958e6570f976db2d
export HS_WORK=$E/aien-work
export HS_DATE=20261015
export HS_RUN_ID=headstart-m5-20261015-4c83e38
export HS_OUT=${HS_OUT:-$WT/benchmarks/data/headstart/patched-20261015-4c83e38}
# Per-sample hold of the shared quiet flag (orchestrator, 2026-10-15): the
# peer tokenizer pauses while someone holds it; held one sample at a time.
export HS_HOLD=${HS_HOLD-1}
export HS_HOLD_OWNER=hs-m5
export HS_HOLD_MINUTES=5
export HS_HOLD_REASON="headstart M5 timing sample"
export HS_CARGO=$H/cargo/target/release/cargo
export HS_RUSTC=$H/rustc/build/host/stage1/bin/rustc
export HS_ARM_ENV_OFF="CARGO_UNSTABLE_HEADSTART=false"
export HS_ARM_ENV_ON="CARGO_UNSTABLE_HEADSTART=true"
# Host baseline load is about 2 to 3 from non-compiler services (ola_runner,
# aien-serve-http), so the M0 value 2 would cap nearly every wait.
export HS_LOAD_MAX=${HS_LOAD_MAX:-3}
export HS_AB_JOBS=${HS_AB_JOBS-"4 8 16"}
export HS_AB_JOBS_SAMPLES=3
export HS_VERIFY_TEST_PKGS="spark-harness aien-proof"
export HS_VERIFY_ENV_ON="RUSTFLAGS=-Zearly-metadata-verify"
export HS_BODY_FILE=crates/cortex-rs/src/lib.rs
export HS_BODY_ANCHOR='    pub fn in_memory() -> Result<Self, String> {'
export HS_IFACE_FILE=crates/cortex-rs/src/lib.rs
export HS_EDIT_REASON="same edit targets as the M0 stable baseline (baseline-20261015-4c83e38 manifest.incremental.reason), so the incremental series are comparable"
exec bash "$WT/scripts/bench/headstart-aien.sh" "$@"
