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

# CAND-4 qualification: shared input handling, sourced by run-cand4.sh and hygiene.sh.
# Reads the ">>> CAND-4 frozen inputs >>>" block of ACCEPTANCE-CAND4.md (or of a TRIAL
# inputs file with the same block) and checks every pinned digest. Shell only.

# load_inputs FILE: keep FILE; refuse a QUALIFICATION block with an UNFROZEN value,
# and a block whose kind is neither QUALIFICATION nor TRIAL.
load_inputs() {
  INPUTS_FILE=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
  INPUTS_BLOCK=$(sed -n '/^>>> CAND-4 frozen inputs >>>$/,/^<<< end of CAND-4 frozen inputs <<<$/p' "$INPUTS_FILE")
  [ -n "$INPUTS_BLOCK" ] || { echo "inputs: no CAND-4 frozen inputs block in $1" >&2; return 1; }
  case $(inp kind) in
    QUALIFICATION)
      if grep -q '= *UNFROZEN *$' <<<"$INPUTS_BLOCK"; then
        echo "inputs: REFUSED, UNFROZEN values in a QUALIFICATION block: $(grep '= *UNFROZEN' <<<"$INPUTS_BLOCK" | cut -d' ' -f1 | paste -sd' ')" >&2
        return 1
      fi;;
    TRIAL) ;;
    *) echo "inputs: kind must be QUALIFICATION or TRIAL" >&2; return 1;;
  esac
}

# inp KEY: the value of KEY (exact key; "a" does not match "a_b").
inp() { sed -n "s/^$1 *= *//p" <<<"$INPUTS_BLOCK" | head -1 | sed 's/ *$//'; }

# out_ok DIR: TRIAL output must say TRIAL and stay out of the evidence dir and the repo.
out_ok() {
  [ "$(inp kind)" = TRIAL ] || return 0
  case $1 in
    *TRIAL*) ;;
    *) echo "inputs: TRIAL output dir must contain TRIAL in its path: $1" >&2; return 1;;
  esac
  case $1 in
    "$HOME"/workspace/evidence-out*|*/docs/campaigns/*) echo "inputs: TRIAL output not allowed in $1" >&2; return 1;;
  esac
}

DIGESTS='[]'
# dg NAME PATH WANT: one pinned file; records the result; returns 1 on a mismatch.
dg() {
  local got=absent
  [ -f "$2" ] && got=$(sha256sum "$2" | cut -d' ' -f1)
  DIGESTS=$(jq -c --arg n "$1" --arg p "$2" --arg w "$3" --arg g "$got" '. + [{name:$n, path:$p, want:$w, got:$g, ok:($w == $g)}]' <<<"$DIGESTS")
  [ "$got" = "$3" ] || { echo "digest MISMATCH $1: $2 is $got, frozen $3" >&2; return 1; }
}

# verify_digests: every pinned file and the source constants at sc_commit. Returns 1 on
# any difference (the caller refuses to run).
verify_digests() {
  local bad=0 md ob src
  DIGESTS='[]'
  dg aien_cli "$(inp aien_cli_path)" "$(inp aien_cli_sha256)" || bad=1
  dg cpu_fault "$(inp cpu_fault_path)" "$(inp cpu_fault_sha256)" || bad=1
  ob=$(inp omega_build_dir)
  dg librx_compose "$ob/librx_compose.a" "$(inp librx_compose_sha256)" || bad=1
  dg libomega_gpu "$ob/libomega_gpu.a" "$(inp libomega_gpu_sha256)" || bad=1
  dg rx_r13_living_host "$ob/rx_r13_living_host" "$(inp rx_r13_host_sha256)" || bad=1
  dg rx_r13_living_silicon "$ob/rx_r13_living_silicon" "$(inp rx_r13_silicon_sha256)" || bad=1
  dg rx_r13_living_testbuild_silicon "$ob/rx_r13_living_testbuild_silicon" "$(inp rx_r13_testbuild_sha256)" || bad=1
  md=$(inp model_dir)
  dg model.safetensors "$(readlink -f "$md/model.safetensors")" "$(inp model_safetensors)" || bad=1
  dg config.json "$(readlink -f "$md/config.json")" "$(inp model_config)" || bad=1
  dg generation_config.json "$(readlink -f "$md/generation_config.json")" "$(inp model_generation_config)" || bad=1
  dg tokenizer.json "$(readlink -f "$md/tokenizer.json")" "$(inp model_tokenizer)" || bad=1
  dg tokenizer_config.json "$(readlink -f "$md/tokenizer_config.json")" "$(inp model_tokenizer_config)" || bad=1
  dg special_tokens_map.json "$(readlink -f "$md/special_tokens_map.json")" "$(inp model_special_tokens)" || bad=1
  dg chat_template.jinja "$(readlink -f "$md/chat_template.jinja")" "$(inp model_chat_template)" || bad=1
  # the snapshot holds exactly these seven files (nothing else the loader could read)
  [ "$(ls -A "$md" | LC_ALL=C sort | paste -sd,)" = "chat_template.jinja,config.json,generation_config.json,model.safetensors,special_tokens_map.json,tokenizer.json,tokenizer_config.json" ] \
    || { echo "model_dir holds other files: $(ls -A "$md" | paste -sd' ')" >&2; bad=1; }
  # source constants at sc_commit (ACCEPTANCE-CAND4 section 2): the frozen values are read from code
  src=$(git -C "$(dirname "${BASH_SOURCE[0]}")" show "$(inp sc_commit):crates/aien-runtime/src/spine.rs" 2>/dev/null)
  grep -q "COMPOSE_ATTEMPT_BUDGET: std::time::Duration = std::time::Duration::from_millis($(inp attempt_budget_ms | sed 's/\(...\)$/_\1/'));" <<<"$src" \
    || { echo "source: COMPOSE_ATTEMPT_BUDGET at sc_commit is not $(inp attempt_budget_ms) ms" >&2; bad=1; }
  grep -q "COMPOSE_SKILL_BUDGET: std::time::Duration = std::time::Duration::from_secs($(( $(inp skill_budget_ms) / 1000 )));" <<<"$src" \
    || { echo "source: COMPOSE_SKILL_BUDGET at sc_commit is not $(inp skill_budget_ms) ms" >&2; bad=1; }
  grep -q "COMPOSE_MAX_ATTEMPTS: u32 = $(inp max_attempts);" <<<"$src" \
    || { echo "source: COMPOSE_MAX_ATTEMPTS at sc_commit is not $(inp max_attempts)" >&2; bad=1; }
  [ "$(git -C "$(dirname "${BASH_SOURCE[0]}")" show "$(inp sc_commit):omega.lock" 2>/dev/null | tr -d '[:space:]')" = "$(inp omega_commit)" ] \
    || { echo "source: omega.lock at sc_commit is not omega_commit" >&2; bad=1; }
  return $bad
}

# wait_quiet WHAT: wait while another lane holds the Spark quiet flag (never interrupt it).
wait_quiet() {
  while [ -e "$HOME/workspace/.spark-quiet" ]; do
    echo "waiting for the Spark quiet flag to clear before $1 ($(date -u +%T))" >&2; sleep 30
  done
}
