#!/usr/bin/env bash
# scripts/contract-omega.sh: this repository's consumer check for omega (contracts/omega.contract).
# Run by the aien-sovereign-core repo agent (tools/aien-board) when an omega post on the change board
# touches a watched path. It proves the rxc_host C ABI contract still holds against the posted omega
# commit, without changing omega, omega.lock or anything in this checkout:
#   1. build omega's librx_compose.a at OMEGA_SHA (crates/aien-omega-compose/build.rs, the documented
#      AIEN_OMEGA_COMPOSE_SHA override), physics at that omega's physics.lock, aienos from aienos.lock;
#   2. aien-abi-conformance checks the C header, a compiled layout probe and the archive symbols against
#      crates/aien-omega-compose/abi/rxc_host_abi.manifest (build.rs; a mismatch fails the build);
#   3. run the compose, callbacks and abi_manifest tests against the REAL library (stub refused);
#   4. write OUT_DIR/watch.learned: every omega file the library was compiled from (the objects' -MMD
#      dependency files) plus the files that decide what is built. contracts/omega.watch is that list.
# Usage: scripts/contract-omega.sh OMEGA_DIR OMEGA_SHA OUT_DIR   (AIEN_BOARD_CLONES: physics, aienos clones)
# Exit: 0 PASS, 1 FAIL, 3 setup missing.
set -euo pipefail
OMW=$(cd "${1:?OMEGA_DIR}" && pwd); SHA="${2:?OMEGA_SHA}"; OUT=$(mkdir -p "${3:?OUT_DIR}" && cd "$3" && pwd)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONES="${AIEN_BOARD_CLONES:?AIEN_BOARD_CLONES names the dir holding physics and aienos clones}"
die() { echo "contract-omega: $*" >&2; exit "${2:-1}"; }
[ "$(git -C "$OMW" rev-parse HEAD)" = "$SHA" ] || die "omega checkout is not at $SHA" 3
for r in physics aienos; do [ -e "$CLONES/$r/.git" ] || die "missing clone $CLONES/$r" 3; done
phys_sha=$(tr -d '[:space:]' <"$OMW/physics.lock")
PHW="$OUT/physics"
if [ "$(git -C "$PHW" rev-parse HEAD 2>/dev/null)" != "$phys_sha" ]; then
  rm -rf "$PHW"
  git clone --quiet --shared --no-checkout "$CLONES/physics" "$PHW"
  git -C "$PHW" checkout --quiet --detach "$phys_sha" || die "physics.lock $phys_sha not in $CLONES/physics" 3
fi
export AIEN_OMEGA_COMPOSE_DIR="$OMW" AIEN_OMEGA_COMPOSE_SHA="$SHA" AIEN_PHYSICS_DIR="$PHW"
export AIEN_AIENOS_LOCK_REPO="$CLONES/aienos" AIEN_REQUIRE_NATIVE_LIBS=1
unset AIEN_OMEGA_DIR AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_FORCE_CPU_STUB AIEN_DEV_FALLBACK
cd "$ROOT"
LOG="$OUT/cargo-test.log"
set +e
cargo test -p aien-omega-compose --test compose --test callbacks --test abi_manifest \
  -- --include-ignored --test-threads=1 >"$LOG" 2>&1
rc=$?
set -e
tail -n 40 "$LOG"
# The interface this consumer really reads, learned from the build (pass or fail, if the build got far).
TGT="${CARGO_TARGET_DIR:-$ROOT/target}"
d=$(ls -td "$TGT"/debug/build/aien-omega-compose-*/out/omega-compose-build/rxc_lib 2>/dev/null | head -1 || true)
if [ -n "$d" ]; then
  {
    printf '%s\n' Makefile mk/ physics.lock aienos.lock
    find "$d" -name '*.d' -exec cat {} + | tr ' \\' '\n\n' | sed -n 's/^[[:space:]]*//; /^[^:]*$/p' \
      | grep -v '^$' | grep -v '^/' | sed 's|^\./||' | grep -v '\.o$'
  } | sort -u >"$OUT/watch.learned"
  echo "contract-omega: learned $(wc -l <"$OUT/watch.learned") watched omega paths"
fi
[ "$rc" -eq 0 ] || die "compose contract tests failed against omega $SHA (exit $rc)"
if grep -q "building the stub" "$LOG"; then die "a stub was built: the omega library is not linked"; fi
n=$(grep -E '^test result: ok\. [0-9]+ passed' "$LOG" | awk '{s += $4} END {print s + 0}')
[ "$n" -gt 0 ] || die "no contract test ran"
echo "contract-omega: PASS ($n tests against omega $SHA)"
