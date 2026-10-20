#!/usr/bin/env bash
# Builds the ALLEN fixture subject for the campaign driver (run-campaign-allen.sh). Host-built, test-only encoder
# (crates/aien-allen/tests/e2e_demo_subject.rs), the same one scripts/allen_e2e_demo.sh S1 and the WHOLE-SYSTEM-E2E
# harness use. Real subjects come only from Aienux cs_provision. Shell only.
#   OQ3_SC_DIR=<sovereign-core checkout the binaries were built from> [OQ3_ALLEN_TARGET=<cargo target dir>] \
#     build-allen-subject.sh OUT LINEAGE_HEX NAME      prints AGENT=<64 hex> and ROOT=<64 hex>
set -u
OUT=${1:?OUT} LINEAGE=${2:?LINEAGE_HEX} NAME=${3:?NAME}
SC=${OQ3_SC_DIR:?OQ3_SC_DIR}
TGT=${OQ3_ALLEN_TARGET:-$(dirname "$OUT")/allen-target}
cd "$SC" || exit 2
env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR CARGO_TARGET_DIR="$TGT" \
  DEMO_SUBJECT_OUT="$OUT" DEMO_LINEAGE_HEX="$LINEAGE" DEMO_SUBJECT_NAME="$NAME" \
  cargo test --release -p aien-allen --test e2e_demo_subject -- --ignored --nocapture 2>&1
