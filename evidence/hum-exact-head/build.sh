#!/bin/bash
# build.sh [cargo args...]: build in the L314 sc worktree against L306's omega lib (read-only use).
L=$HOME/workspace/targets/q36/omega-new/libomega_gpu.a
cd $HOME/workspace/sc-hum-exact-head
AIEN_OMEGA_GPU_LIB=$L AIEN_OMEGA_GPU_LIB_SHA256=$(sha256sum $L | cut -d' ' -f1) CARGO_TARGET_DIR=$HOME/workspace/targets/hum-exact-head/sc quiet-build nice -n 10 cargo "$@"
