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

# OPEN-MODEL-QWEN3 v3 RUN (ACCEPTANCE-v3 frozen on sovereign-core main cad36d9). One part per quietlock hold.
# Binaries: frozen build from sovereign-core 8f3e8c8 + omega 01f6a74 (sha256 in ACCEPTANCE-v3 Section 7).
# Wrapper, tasks and declaration: clean checkout of main cad36d9 at /home/drakestapleton/workspace/oq3-v3-wrap.
set -u
part=$1
{ date -u +%FT%TZ; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } > /home/drakestapleton/workspace/oq3-v3-records/mem-p$part.txt
unset AIEN_FORCE_CPU_STUB AIEN_COMPOSE_MAX_TOKENS AIEN_COMPOSE_DOC_MAX_TOKENS AIEN_COMPOSE_EDIT_BUDGET_MS AIEN_COMPOSE_DOC_BUDGET_MS AIEN_COMPOSE_BUDGET_MS AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET OQ3_DRY_TASKS
B=/home/drakestapleton/workspace/oq3-v3-build/target/release
AIEN_BIN=$B/aien-cli AIEN_MODEL_PATH=/home/drakestapleton/models/qwen3-4b-instruct-2507-cdbee75/model.safetensors.index.json AIEN_TOKENIZER_PATH=/home/drakestapleton/models/qwen3-4b-instruct-2507-cdbee75/tokenizer.json \
AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 \
OQ3_PART=$part \
/home/drakestapleton/workspace/oq3-v3-wrap/docs/campaigns/open-model-qwen3/run-qwen3-v3.sh /home/drakestapleton/workspace/oq3-v3-runs /home/drakestapleton/workspace/oq3-v3-records/out 8f3e8c8b879e10dd83883cee150f16508284d643 01f6a74636b8383b010cdb95597839582c415c27 01f6a74636b8383b010cdb95597839582c415c27 $B/examples/np1_reference $B/examples/np1_edit_merge
