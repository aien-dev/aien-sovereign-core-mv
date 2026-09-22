#!/usr/bin/env bash
set -euo pipefail
if [[ -f /home/drakestapleton/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat /home/drakestapleton/.cache/huggingface/token)"
fi
sync; sudo -n sysctl -w vm.drop_caches=3 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0
export MAX_SERVE_HOST=127.0.0.1
export MAX_SERVE_METRICS_ENDPOINT_PORT=8011

SNAPSHOT_DIR="/home/drakestapleton/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
WEIGHT_ARGS=()
for shard in $(ls -1 "$SNAPSHOT_DIR"/model-*.safetensors | sort); do
  WEIGHT_ARGS+=(--weight-path "$shard")
done

exec /home/drakestapleton/max-env/bin/max serve \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  "${WEIGHT_ARGS[@]}" \
  --served-model-name atlas-lightning-omni \
  --custom-architectures /home/drakestapleton/nemotron_h_kvexp \
  --port 18006 \
  --max-batch-size 8 \
  --max-length 49152 \
  --device-memory-utilization 0.85 \
  --quantization-encoding float4_e2m1fnx2 \
  --enable-structured-output \
  --enable-prefix-caching \
  --kv-cache-format bfloat16 \
  --fold-sampler-into-graph \
  --no-device-graph-capture
