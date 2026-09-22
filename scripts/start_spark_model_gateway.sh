#!/usr/bin/env bash
set -euo pipefail

# MAX stays private on 127.0.0.1:18006. The gateway is the client-facing
# compatibility and admission layer on 127.0.0.1:18007.
export AIEN_GATEWAY_UPSTREAM="${AIEN_GATEWAY_UPSTREAM:-http://127.0.0.1:18006}"
export AIEN_GATEWAY_TOKENIZER_PATH="${AIEN_GATEWAY_TOKENIZER_PATH:-/home/drakestapleton/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/tokenizer.json}"

exec /home/drakestapleton/aien-sovereign-core/target/release/spark-model-gateway
