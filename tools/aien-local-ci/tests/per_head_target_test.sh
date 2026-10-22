#!/usr/bin/env bash
# tests/per_head_target_test.sh: why every head gets its own CARGO_TARGET_DIR (sc #605 flakes, #452 signature).
#
# Cargo calls a workspace crate fresh when its outputs are newer than its sources. It compares file times and a
# path RELATIVE to the workspace, never the absolute checkout path. Two heads that share one target dir therefore
# share binaries: when head A builds after head B was checked out, head B's gate links and runs head A's build,
# with head A's CARGO_MANIFEST_DIR baked in. Seen on the Spark as SC-VERIFY-CLOSURE "…/wt/aien-sovereign-core/
# 2692c1e8…/tests/fixtures/c-cycle: No such file" while judging 9b8e09eb, and as PREFILL-E2E-2 running a mutant.
#
# Leg 1 (shared dir, the old layout): must show head B running head A's binary. If it does not, this cargo no
#        longer has the hazard and the test says so (exit 1) instead of passing silently.
# Leg 2 (one dir per head, the new layout): head B must run its own build.
# Needs only cargo; no sandbox, no network.
set -uo pipefail
R=$(mktemp -d); trap 'rm -rf "$R"' EXIT
pass=0; fail=0
ok()  { pass=$((pass+1)); printf 'PASS %s\n' "$1"; }
bad() { fail=$((fail+1)); printf 'FAIL %s: %s\n' "$1" "$2"; }
mk() {  # mk <dir>: one probe crate whose test prints and checks the checkout path baked in at build time
  mkdir -p "$1/src"
  printf '[package]\nname = "probe"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n' > "$1/Cargo.toml"
  cat > "$1/src/lib.rs" <<'EOF'
pub fn dir() -> &'static str { env!("CARGO_MANIFEST_DIR") }
#[test]
fn where_am_i() {
    println!("BUILT_IN={}", dir());
    assert!(std::path::Path::new(dir()).exists(), "baked path gone: {}", dir());
}
EOF
}
leg() {  # leg <name> <target-of-A> <target-of-B>: B checked out first, A built after it, A's checkout removed
  local n=$1 ta=$2 tb=$3
  mk "$R/$n/headB"; sleep 1.1; mk "$R/$n/headA"
  (cd "$R/$n/headA" && CARGO_TARGET_DIR=$ta cargo test -q --offline -- --nocapture >/dev/null 2>&1) || { bad "$n" "head A build failed"; return; }
  rm -rf "$R/$n/headA"
  (cd "$R/$n/headB" && CARGO_TARGET_DIR=$tb cargo test -q --offline -- --nocapture 2>&1) | grep -o 'BUILT_IN=[^ ]*' | head -1
}
got=$(leg shared "$R/shared/t" "$R/shared/t")
[[ $got == "BUILT_IN=$R/shared/headA" ]] && ok "shared target dir: head B ran head A's binary (the hazard is real)" \
  || bad "shared hazard not reproduced" "$got"
got=$(leg perhead "$R/perhead/t-A" "$R/perhead/t-B")
[[ $got == "BUILT_IN=$R/perhead/headB" ]] && ok "one target dir per head: head B ran its own build" \
  || bad "per-head" "$got"
printf '%d passed, %d failed\n' "$pass" "$fail"
(( fail == 0 ))
