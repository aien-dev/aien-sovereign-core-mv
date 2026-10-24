# qwen38_receipt.sh: helpers shared by tools/qwen38_serve_gate.sh and tools/qwen38_serve_speed.sh.
# Sourced, not run. Collects what a run needs to be checked later: which server binary answered
# (sha256 and its --version build line), which model folder, the machine state, and one JSON
# receipt per run. Needs curl, jq, sha256sum and awk.
#
# Which server: the process listening on the URL's port (found with ss, works for your own
# processes), or set SERVER_BIN and MODEL_DIR explicitly. Anything that cannot be found is
# written as "unknown", never guessed.
#
# AIEN_CHECK_MODEL=1 also runs sha256sum -c of the shipped model checksum list in the model
# folder (reads about 31 GB, several minutes).

Q38_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
Q38_MODEL_SUMS="$Q38_REPO/docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256"
# Weight bytes read per generated token and the 273 GB/s nominal memory rate: docs/qwen38/CEILING.md.
# shellcheck disable=SC2034 # read by qwen38_serve_speed.sh
Q38_CEILING_TOK_S=10.15

# q38_server_facts <base-url>: sets Q38_PID, Q38_BIN, Q38_BIN_SHA256, Q38_BUILD, Q38_MODEL_DIR.
q38_server_facts() {
  local url=$1 port pid="" cmd
  port=$(sed -E 's#^[a-z]+://[^:/]+:?([0-9]*).*#\1#' <<<"$url"); port=${port:-80}
  if command -v ss >/dev/null; then
    pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
  fi
  Q38_PID=${pid:-unknown}
  Q38_BIN=${SERVER_BIN:-}
  if [ -z "$Q38_BIN" ] && [ -n "$pid" ] && [ -r "/proc/$pid/exe" ]; then Q38_BIN=/proc/$pid/exe; fi
  if [ -n "$Q38_BIN" ] && [ -r "$Q38_BIN" ]; then
    Q38_BIN_SHA256=$(sha256sum "$Q38_BIN" | cut -d' ' -f1)
    # --version loads nothing (aien-serve-http builds from alpha/serve-path on). Older builds
    # refuse the flag and exit, so their build line is unknown.
    Q38_BUILD=$(timeout 20 "$Q38_BIN" --version 2>/dev/null | grep '^sovereign-core commit:' | sed 's/^sovereign-core commit: //' || true)
    Q38_BUILD=${Q38_BUILD:-unknown (binary has no --version)}
  else
    Q38_BIN_SHA256=unknown; Q38_BUILD=unknown
  fi
  Q38_MODEL_DIR=${MODEL_DIR:-}
  if [ -z "$Q38_MODEL_DIR" ] && [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ]; then
    cmd=$(tr '\0' '\n' < "/proc/$pid/cmdline")
    Q38_MODEL_DIR=$(awk 'p {print; exit} $0 == "--model-dir" {p = 1}' <<<"$cmd")
  fi
  Q38_MODEL_DIR=${Q38_MODEL_DIR:-unknown}
}

# q38_machine_json: driver, kernel, load average and free memory now, as one JSON object.
q38_machine_json() {
  local drv="" load mem
  [ -r /proc/driver/nvidia/version ] && drv=$(grep -o '[0-9]\{3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' /proc/driver/nvidia/version | head -1 || true)
  load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)
  mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
  jq -n --arg drv "${drv:-unknown}" --arg kernel "$(uname -r)" --arg arch "$(uname -m)" \
    --arg load "$load" --argjson mem "${mem:-0}" --arg at "$(date -u +%FT%TZ)" \
    '{at:$at, nvidia_driver:$drv, kernel:$kernel, arch:$arch, loadavg_1_5_15:$load, mem_available_kib:$mem}'
}

# q38_model_check_json: the shipped checksum list, its sha256, and (AIEN_CHECK_MODEL=1) the result.
q38_model_check_json() {
  local result="not run (set AIEN_CHECK_MODEL=1; reads about 31 GB)" list_sha=missing
  [ -f "$Q38_MODEL_SUMS" ] && list_sha=$(sha256sum "$Q38_MODEL_SUMS" | cut -d' ' -f1)
  if [ "${AIEN_CHECK_MODEL:-0}" = 1 ]; then
    if [ -d "$Q38_MODEL_DIR" ] && [ -f "$Q38_MODEL_SUMS" ]; then
      if (cd "$Q38_MODEL_DIR" && sha256sum --quiet -c "$Q38_MODEL_SUMS" >/dev/null 2>&1); then result=MATCH; else result=MISMATCH; fi
    else
      result="not run (model folder or list missing)"
    fi
  fi
  jq -n --arg list "docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256" --arg sha "$list_sha" --arg r "$result" \
    '{sha256_list:$list, sha256_list_sha256:$sha, check:$r}'
}

# q38_log_line <server-log> <reply-id>: the server's own qwen38 log line for one reply ("" if absent).
q38_log_line() {
  [ -f "$1" ] || return 0
  grep -B1 "$2:" "$1" 2>/dev/null | grep 'qwen38:' | head -1 | sed 's/aien-serve-http: qwen38: //' || true
}

# q38_run_json <name> <reply.json> <client-wall-s> <server-log>: one request's figures as JSON.
q38_run_json() {
  local name=$1 f=$2 wall=$3 log=$4 id line ids chip swall lock
  id=$(jq -r '.id // ""' "$f" 2>/dev/null)
  line=$(q38_log_line "$log" "$id")
  ids=$(jq -c '.usage.aien_token_ids // null' "$f" 2>/dev/null || echo null)
  chip=$(grep -o 'graph chip [0-9.]* ms/token' <<<"$line" | grep -o '[0-9.]*' | head -1 || true)
  swall=$(grep -o 'wall [0-9.]* ms/token' <<<"$line" | grep -o '[0-9.]*' | head -1 || true)
  lock=$(grep -o 'lock wait [0-9]* ms' <<<"$line" | grep -o '[0-9]*' | head -1 || true)
  jq -n --arg name "$name" --arg id "$id" --arg wall "${wall:-}" --arg line "$line" \
    --arg chip "${chip:-}" --arg swall "${swall:-}" --arg lock "${lock:-}" \
    --arg ids_sha "$(printf '%s\n' "$ids" | sha256sum | cut -c1-12)" \
    --argjson pt "$(jq '.usage.prompt_tokens // null' "$f" 2>/dev/null || echo null)" \
    --argjson ct "$(jq '.usage.completion_tokens // null' "$f" 2>/dev/null || echo null)" \
    '{name:$name, reply_id:$id, prompt_tokens:$pt, completion_tokens:$ct,
      client_wall_s:($wall|tonumber? // null),
      client_ms_per_token:(if ($ct // 0) > 0 and ($wall|tonumber? // null) != null then (($wall|tonumber) * 1000 / $ct) else null end),
      chip_ms_per_token:($chip|tonumber? // null), server_wall_ms_per_token:($swall|tonumber? // null),
      lock_wait_ms:($lock|tonumber? // null), ids_sha256_12:$ids_sha, server_log_line:$line}'
}

# q38_write_receipt <out-file> <kind> <url> <runs.jsonl> <extra-json>: the run's receipt.
q38_write_receipt() {
  local out=$1 kind=$2 url=$3 runs=$4 extra=$5
  jq -n --arg schema "aien-qwen38-serve-receipt/1" --arg kind "$kind" --arg url "$url" \
    --arg tools_commit "$(git -C "$Q38_REPO" rev-parse HEAD 2>/dev/null || echo unknown)" \
    --arg tools_dirty "$([ -n "$(git -C "$Q38_REPO" status --porcelain --untracked-files=no 2>/dev/null)" ] && echo yes || echo no)" \
    --arg pid "$Q38_PID" --arg bin "${Q38_BIN:-unknown}" --arg bin_sha "$Q38_BIN_SHA256" --arg build "$Q38_BUILD" \
    --arg model "$Q38_MODEL_DIR" \
    --argjson machine_start "$Q38_MACHINE_START" --argjson machine_end "$(q38_machine_json)" \
    --argjson model_check "$(q38_model_check_json)" \
    --slurpfile runs "$runs" --argjson extra "$extra" \
    '{schema:$schema, kind:$kind, server_url:$url,
      tools:{repo_commit:$tools_commit, uncommitted_changes:$tools_dirty},
      server:{pid:$pid, binary:$bin, binary_sha256:$bin_sha, build:$build},
      model:({dir:$model} + $model_check),
      machine:{start:$machine_start, end:$machine_end},
      runs:$runs} + $extra' > "$out"
}
