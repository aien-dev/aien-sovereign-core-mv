#!/bin/bash
# Build qwen38_lab in the skipped-ideas worktree with its own target dir.
cd ~/workspace/sc-skipped || exit 1
export CARGO_TARGET_DIR=$HOME/workspace/targets/skipped
export AIEN_OMEGA_GPU_LIB=$HOME/workspace/targets/q38-f7-omega/libomega_gpu.a
export AIEN_OMEGA_GPU_LIB_SHA256=6e90520913032ee8653ebbe325c485d329daa0a749840303a44b68b97ca17bff
~/.local/bin/quiet-build nice -n 10 cargo build --release -j 6 -p aien-inference-abi --bin qwen38_lab 2>&1 | grep -E "^(error|warning: unused|warning)|-->|^\s+\||Finished" | head -80
