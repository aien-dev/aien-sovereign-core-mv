#!/usr/bin/env bash
# Stage 3: digests, smoke test, upstream correctness checks. Output to logs/.
set -uo pipefail
E=~/workspace/headstart-exp
H=$E/headstart
L=$E/logs
R=$H/rustc/build/host/stage1/bin/rustc
C=$H/cargo/target/release/cargo
{
  echo "## digests"
  sha256sum "$R" "$H"/rustc/build/host/stage1/lib/librustc_driver-*.so "$C"
  echo "## versions"
  "$R" -vV; "$C" -V
  echo "## upstream pin"
  (cd "$H" && git rev-parse HEAD && git ls-tree HEAD rustc cargo)
  (cd "$H/rustc" && echo "rustc submodule $(git rev-parse HEAD)")
  (cd "$H/cargo" && echo "cargo submodule $(git rev-parse HEAD)")
  sha256sum "$H"/patches/rustc/*.patch "$H"/patches/cargo/*.patch "$H/config/bootstrap.toml"
} > "$L/03-digests.txt" 2>&1

run() { # run <name> <cmd...>
  local name=$1; shift
  local t0=$(date +%s)
  ( "$@" ) > "$L/03-$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc wall_s=$(( $(date +%s) - t0 ))" | tee -a "$L/03-results.txt"
}
: > "$L/03-results.txt"
smoke() {
  local t; t=$(mktemp -d)
  cp -r "$H/tests/smoke/." "$t/"
  cd "$t"
  for m in false true; do
    rm -rf target
    echo "### headstart=$m"
    /usr/bin/time -f 'wall=%e maxrss_kb=%M' env RUSTC="$R" CARGO_UNSTABLE_HEADSTART=$m "$C" check --color never 2>&1
    echo "exit=$?"
  done
  rm -rf "$t"
}
cd "$H"
run smoke smoke
run check-swap env RUSTC="$R" scripts/check-swap.sh
run check-errors scripts/check-errors.sh
run check-incremental-check scripts/check-incremental.sh check
run check-incremental-build scripts/check-incremental.sh build
echo SANITY_DONE
