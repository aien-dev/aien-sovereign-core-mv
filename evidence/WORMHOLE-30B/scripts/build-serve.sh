#!/bin/sh
# L190: build aien-serve-http (with --engine wormhole) natively against the omega lib 18091 runs (qwen38/decode-speed 0010f7d3, sha bd6802a5).
cd /home/drakestapleton/workspace/sc-wormhole-30b || exit 1
export CARGO_TARGET_DIR=/home/drakestapleton/workspace/targets/sc-wormhole-30b-native
export AIEN_OMEGA_GPU_LIB=/home/drakestapleton/workspace/targets/sc-wormhole-30b/omega-lib/libomega_gpu.a
export AIEN_OMEGA_GPU_LIB_SHA256=bd6802a5d493ed6b4589dabb5787db70a37ba10c75d8406f3cce27c4dfa9f9bd
quiet-build cargo build --release -p aien-serve-http --bin aien-serve-http 2>&1 | grep -E "^(warning|error)|-->|Finished" | grep -v "omega-compose"
