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

# OPEN-MODEL-QWEN3 v2 RUN (ACCEPTANCE-v2 frozen on sovereign-core main 3e8c998). One part per quietlock hold.
# Binaries: frozen build from sovereign-core 905fdfc + omega 459b461 (sha256 in ACCEPTANCE-v2 Section 2).
# Wrapper, tasks and declaration: clean checkout of main 3e8c998 at /home/drakestapleton/workspace/oq3-v2-build/sc.
set -u
part=$1
{ date -u +%FT%TZ; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } > /home/drakestapleton/workspace/oq3-v2-records/mem-p$part.txt
unset AIEN_FORCE_CPU_STUB AIEN_COMPOSE_MAX_TOKENS AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET OQ3_DRY_TASKS
AIEN_BIN=/home/drakestapleton/workspace/oq3-v2-build/target/release/aien-cli AIEN_MODEL_PATH=/home/drakestapleton/models/qwen3-4b-instruct-2507-cdbee75/model.safetensors.index.json AIEN_TOKENIZER_PATH=/home/drakestapleton/models/qwen3-4b-instruct-2507-cdbee75/tokenizer.json \
AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 \
OQ3_PART=$part \
/home/drakestapleton/workspace/oq3-v2-build/sc/docs/campaigns/open-model-qwen3/run-qwen3-v2.sh /home/drakestapleton/workspace/oq3-v2-runs /home/drakestapleton/workspace/oq3-v2-records/out 905fdfc18933ed47ae1cd075163be1eb8a421ca1 459b46133550a39ed391ea11eeb5aa5968052611 459b46133550a39ed391ea11eeb5aa5968052611 /home/drakestapleton/workspace/oq3-v2-build/target/release/examples/np1_reference /home/drakestapleton/workspace/oq3-v2-build/target/release/examples/np1_edit_merge
