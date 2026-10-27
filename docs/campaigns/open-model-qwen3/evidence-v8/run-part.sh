#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# OPEN-MODEL-QWEN3 v8 (SPEC-v8.md; real parts only once frozen on sc main). One part per quietlock hold. Never killed, never timed out.
#   run-part.sh real <1|2|3|4>      the qualification part
#   run-part.sh dry 1               gate G6: part 1 in dry mode (same tasks file, scratch run base, never scored)
# As evidence-v7/run-part.sh with: the v8 wrapper, the Qwen3-8B model folder, the v8 build and run bases.
set -u
mode=$1 part=$2; V=/home/drakestapleton/workspace/oq3-v8-real; R=$V/records
B=${OQ3_BUILD:?OQ3_BUILD (the oq3-v8-build-<commit> folder)}; M=/home/drakestapleton/models/qwen3-8b
W=$V/wrap/docs/campaigns/open-model-qwen3/run-qwen3-v8.sh
tag=$mode-p$part
{ date -u +%FT%TZ; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } > $R/mem-$tag.txt
journalctl -k --no-pager 2>/dev/null | grep -c NVRM > $R/nvrm-before-$tag.txt
unset AIEN_FORCE_CPU_STUB AIEN_DEV_FALLBACK AIEN_BACKEND AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK AIEN_COMPOSE_MAX_TOKENS AIEN_COMPOSE_DOC_MAX_TOKENS \
      AIEN_COMPOSE_EDIT_BUDGET_MS AIEN_COMPOSE_DOC_BUDGET_MS AIEN_COMPOSE_BUDGET_MS AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET OQ3_DRY_TASKS OQ3_FROZEN
if [ "$mode" = dry ]; then
  base=$V/dry-runs out=$R/dry-out
  export OQ3_DRY_TASKS=$V/wrap/docs/campaigns/open-model-qwen3/tasks-oq3-v5.json
else
  base=/home/drakestapleton/workspace/oq3-v8-runs out=$R/out
fi
AIEN_BIN=$B/target/release/aien-cli AIEN_MODEL_PATH=$M/model.safetensors.index.json AIEN_TOKENIZER_PATH=$M/tokenizer.json \
AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_REQUIRE_CHECKPOINT=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 \
OQ3_PART=$part bash "$W" "$base" "$out" $B/sc $B/pins/omega $B/target/release/examples/np1_reference $B/target/release/examples/np1_edit_merge
rc=$?
journalctl -k --no-pager 2>/dev/null | grep -c NVRM > $R/nvrm-after-$tag.txt
exit $rc
