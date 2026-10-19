#!/bin/bash -l
# Workaround for install.sh exit 101: the server_build() steps of install.sh, by hand, same commands.
set -uo pipefail
. "$HOME/.cargo/env"
cd ~/aien-sovereign-core-mv
SOURCE_ROOT=$PWD; BIN_DIR=$HOME/.local/bin; pins=$SOURCE_ROOT/target/pins
{
echo "loadavg_start $(cat /proc/loadavg)"; echo "memavail_kB_start $(awk '/MemAvailable/{print $2}' /proc/meminfo)"
echo "quiet_wait_start $(date +%s)"
echo QUIET_LOCK_NOT_HELD_writer_starved; bash -c '
  echo "quiet_acquired $(date +%s) loadavg $(cat /proc/loadavg)"
  t0=$(date +%s)
  nice -n 19 ionice -c3 bash scripts/pins.sh --gpu target/pins || { echo PINS_FAIL; exit 2; }
  . target/pins/env.sh
  echo "AIEN_REQUIRE_NATIVE_LIBS=${AIEN_REQUIRE_NATIVE_LIBS:-} AIEN_OMEGA_DIR=${AIEN_OMEGA_DIR:-}"
  t1=$(date +%s)
  nice -n 19 ionice -c3 cargo build --locked --release -p aien-serve-http || { echo BUILD_FAIL; exit 3; }
  t2=$(date +%s)
  echo "pins_s $((t1-t0)) build_s $((t2-t1)) total_s $((t2-t0))"
  echo "loadavg_end $(cat /proc/loadavg)"
' 9<&-
rc=$?; exec 9<&-; echo "rc $rc quiet_released $(date +%s)"
out=target/release/aien-serve-http
info=$($out --version 2>&1); echo "$info"
if grep -q "^omega gpu engine: native" <<<"$info" && grep -q "^qwen38 engine: native" <<<"$info"; then
  mkdir -p $BIN_DIR; install -m755 $out $BIN_DIR/aien-serve-http; install -m755 tools/serve-27b.sh $BIN_DIR/aien-serve-27b; echo INSTALLED
else echo STUB_OR_MISSING; fi
echo "memavail_kB_end $(awk '/MemAvailable/{print $2}' /proc/meminfo)"
} > ~/server-build.out 2>&1
echo done > ~/server-build.done
