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

# NEXT-PHASE-1 v6: evidence of the Cortex record mark (ACCEPTANCE-v6 Section 6.1),
# one JSON object on stdout. Shell + coreutils + jq only; it judges nothing.
# Usage: mark-evidence.sh MARK_FILE [MACHINE_ID_FILE]
# Mark layout (crates/aien-runtime/src/cortex_mark.rs:10-12, 128 bytes):
# "AIENCXM1" | u32 version | u32 0 | machine id (bytes 16..48) | u64 seq |
# u64 records | digest (32) | sha256 of the first 96 bytes (bytes 96..128).
set -u
M=${1:?MARK_FILE} MID=${2:-}
hex() { od -An -tx1 -v | tr -d ' \n'; }
midhex=null
if [ -n "$MID" ] && [ -f "$MID" ]; then midhex=$(hex <"$MID" | jq -R .); fi
if [ ! -f "$M" ]; then
  jq -nc --argjson mid "$midhex" '{present:false, compose_machine_id_file_hex:$mid}'
  exit 0
fi
jq -nc --argjson size "$(stat -c %s "$M")" --arg magic "$(head -c 8 "$M" | hex)" \
  --arg head96 "$(head -c 96 "$M" | sha256sum | cut -d' ' -f1)" --arg tail "$(tail -c +97 "$M" | hex)" \
  --arg mark_mid "$(tail -c +17 "$M" | head -c 32 | hex)" --arg sha "$(sha256sum "$M" | cut -d' ' -f1)" \
  --argjson mid "$midhex" \
  '{present:true, size:$size, magic_hex:$magic, head96_sha256:$head96, tail_hex:$tail, sha256:$sha,
    machine_id_hex:$mark_mid, compose_machine_id_file_hex:$mid}'
