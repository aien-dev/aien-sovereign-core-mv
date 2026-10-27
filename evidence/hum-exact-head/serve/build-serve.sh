#!/bin/bash
# build-serve.sh: aien-serve-http from the L314 serve worktree (motor/serve + exact head) against the frozen exact-head omega lib.
L=$HOME/workspace/targets/hum-exact-serve/omega/libomega_gpu.a
cd $HOME/workspace/sc-hum-exact-serve
AIEN_OMEGA_GPU_LIB=$L AIEN_OMEGA_GPU_LIB_SHA256=$(sha256sum $L | cut -d' ' -f1) CARGO_TARGET_DIR=$HOME/workspace/targets/hum-exact-head/sc \
  quiet-build nice -n 10 cargo build --release -p aien-serve-http > $HOME/handoffs/hum-exact-head/build-serve.log 2>&1
echo "exit $?" >> $HOME/handoffs/hum-exact-head/build-serve.log
