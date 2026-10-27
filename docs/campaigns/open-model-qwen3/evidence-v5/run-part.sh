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

# OPEN-MODEL-QWEN3 v5 REAL RUN (ACCEPTANCE-v5, FROZEN on sc main). One part per quietlock hold. Never killed, never timed out.
# Binaries: build ~/workspace/oq3-v5-build-30258fc (frozen-v5.json pins). Wrapper: clean clone of sc main after the G8 merge (./wrap).
set -u
part=$1; R=/home/drakestapleton/workspace/oq3-v5-real/records
{ date -u +%FT%TZ; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } > $R/mem-p$part.txt
journalctl -k --no-pager 2>/dev/null | grep -c NVRM > $R/nvrm-before-p$part.txt
unset AIEN_FORCE_CPU_STUB AIEN_DEV_FALLBACK AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK AIEN_COMPOSE_MAX_TOKENS AIEN_COMPOSE_DOC_MAX_TOKENS \
      AIEN_COMPOSE_EDIT_BUDGET_MS AIEN_COMPOSE_DOC_BUDGET_MS AIEN_COMPOSE_BUDGET_MS AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET OQ3_DRY_TASKS
B=/home/drakestapleton/workspace/oq3-v5-build-30258fc; M=/home/drakestapleton/models/qwen3-4b-instruct-2507-cdbee75
AIEN_BIN=$B/target/release/aien-cli AIEN_MODEL_PATH=$M/model.safetensors.index.json AIEN_TOKENIZER_PATH=$M/tokenizer.json \
AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_REQUIRE_CHECKPOINT=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 \
OQ3_PART=$part \
/home/drakestapleton/workspace/oq3-v5-real/wrap/docs/campaigns/open-model-qwen3/run-qwen3-v5.sh \
  /home/drakestapleton/workspace/oq3-v5-runs $R/out $B/sc $B/omega $B/target/release/examples/np1_reference $B/target/release/examples/np1_edit_merge
rc=$?
journalctl -k --no-pager 2>/dev/null | grep -c NVRM > $R/nvrm-after-p$part.txt
exit $rc
