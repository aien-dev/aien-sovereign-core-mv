#!/bin/bash
# Document memory chip run for the Qwen3.8-27B door (cascade/document-memory).
#
# Starts our own aien-serve-http on a test port with the conversation store on, reads a document
# in the background (POST /v1/documents), then asks questions on it after another conversation
# replaced the live state: from RAM, then after spilling the saved conversation to NVMe. Every
# answer is compared token for token with the same prompt recomputed from scratch
# (aien_reuse=cold). Everything goes into the evidence folder: commands, raw responses, the
# server log, timings with load average and MemAvailable.
#
# usage: tools/qwen38_docmem_run.sh EVIDENCE_DIR BIN SIZES...   (sizes in document bytes)
# env:   PORT (18291), MODEL_DIR, MAX_CONTEXT (40960), RAM (4G), STORE_DIR, MUTANT (state|kv),
#        QUESTIONS_RAM (2), QUESTIONS_NVME (2), MAX_TOKENS (32)
set -u
EV=$1; BIN=$2; shift 2
PORT=${PORT:-18291}
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-fp8-017b9c7a}
MAX_CONTEXT=${MAX_CONTEXT:-40960}
RAM=${RAM:-4G}
STORE_DIR=${STORE_DIR:-$HOME/workspace/targets/docmem/convstore}
QR=${QUESTIONS_RAM:-2}; QN=${QUESTIONS_NVME:-2}
MAXT=${MAX_TOKENS:-32}
URL=http://127.0.0.1:$PORT
mkdir -p "$EV"
LOG=$EV/timings.tsv
[ -f "$LOG" ] || printf 'utc\tstep\tsize_bytes\tprompt_tokens\tcurl_total_s\tloadavg_start\tloadavg_end\tmemavail_kb_start\tmemavail_kb_end\tfile\n' > "$LOG"

memavail() { awk '/MemAvailable/ {print $2}' /proc/meminfo; }
la() { cut -d' ' -f1-3 /proc/loadavg | tr ' ' ','; }
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$EV/run.log"; }

# The document: every license text under /usr/share/common-licenses in a fixed order, cut to N bytes.
make_doc() {
  local n=$1 f=$EV/doc-$1.txt
  if [ ! -f "$f" ]; then
    for x in GPL-3 Apache-2.0 GPL-2 LGPL-2.1 MPL-2.0 GFDL-1.3 MPL-1.1 LGPL-2 GPL-1 CC0-1.0 Artistic LGPL-3 BSD GFDL-1.2; do
      printf '\n===== %s =====\n' "$x"; cat /usr/share/common-licenses/$x
    done | head -c "$n" > "$f"
  fi
  echo "$f"
}

QUESTIONS=(
  "In one sentence: what does the document say about warranty?"
  "List three license names that appear in the document."
  "What does the document say a licensee must do when distributing copies?"
  "Which section talks about patents, and what does it say? Answer briefly."
  "Summarise the very last paragraph of the document in one sentence."
  "Does the document mention the Free Software Foundation? Quote one line."
)

# body DOCFILE QUESTION|"" COLD(0/1)
body() {
  local doc=$1 q=$2 cold=$3
  jq -n --rawfile d "$doc" --arg q "$q" --argjson cold "$cold" --argjson mt "$MAXT" '
    {model:"qwen3.8-27b-fp8", temperature:0, max_tokens:$mt, aien_return_token_ids:true,
     chat_template_kwargs:{enable_thinking:false},
     messages:( [ {role:"user", content:("Here is a document.\n\n" + $d)} ]
       + (if $q == "" then [] else [ {role:"assistant", content:"I have read the document."}, {role:"user", content:$q} ] end) ) }
    + (if $cold == 1 then {aien_reuse:"cold"} else {} end)'
}

# post NAME SIZE ENDPOINT BODYFILE -> response file
post() {
  local name=$1 size=$2 ep=$3 bf=$4 out=$EV/$1.json
  local l0 m0 t
  l0=$(la); m0=$(memavail)
  echo "quiet-measure curl -s -X POST $URL$ep -H 'content-type: application/json' --data-binary @$(basename "$bf")" > "$EV/$name.cmd"
  t=$($HOME/.local/bin/quiet-measure curl -s -o "$out" -w '%{time_total}' -X POST "$URL$ep" -H 'content-type: application/json' --data-binary @"$bf")
  local pt; pt=$(jq -r '.usage.prompt_tokens // .tokens // ""' "$out" 2>/dev/null)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "$size" "$pt" "$t" "$l0" "$(la)" "$m0" "$(memavail)" "$name.json" >> "$LOG"
  echo "$out"
}

ids() { jq -c '.usage.aien_token_ids' "$1"; }

start_server() {
  note "starting server port $PORT (MUTANT=${MUTANT:-none}) memavail $(memavail) kB load $(la)"
  echo "AIEN_CONV_STORE_DIR=$STORE_DIR AIEN_CONV_RAM_BYTES=$RAM AIEN_CONV_TEST_MUTANT=${MUTANT:-} $BIN --engine omega --model-dir $MODEL_DIR --port $PORT --max-context $MAX_CONTEXT --max-tokens-cap 512 --thinking off" >> "$EV/server.cmd"
  AIEN_CONV_STORE_DIR=$STORE_DIR AIEN_CONV_RAM_BYTES=$RAM AIEN_CONV_TEST_MUTANT=${MUTANT:-} \
    setsid nohup "$BIN" --engine omega --model-dir "$MODEL_DIR" --port "$PORT" --max-context "$MAX_CONTEXT" \
    --max-tokens-cap 512 --thinking off >> "$EV/server.log" 2>&1 < /dev/null &
  echo $! > "$EV/server.pid"
  for _ in $(seq 1 900); do
    curl -s "$URL/v1/models" 2>/dev/null | grep -q qwen && { note "server up pid $(cat "$EV/server.pid")"; return 0; }
    kill -0 "$(cat "$EV/server.pid")" 2>/dev/null || { note "server died"; return 1; }
    sleep 2
  done
  note "server did not come up"; return 1
}

stop_server() {
  local p; p=$(cat "$EV/server.pid" 2>/dev/null) || return 0
  kill -TERM "$p" 2>/dev/null
  for _ in $(seq 1 300); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  note "server $p stopped"
}

wait_ready() {
  local id=$1 t0=$SECONDS st
  while true; do
    st=$(curl -s "$URL/v1/documents" | jq -c --arg id "$id" '.entries[] | select(.id == $id) | .status')
    case "$st" in
      '"Ready"') note "ingest $id ready after $((SECONDS - t0)) s"; return 0 ;;
      *Failed*) note "ingest $id failed: $st"; return 1 ;;
    esac
    sleep 2
  done
}

displace() {
  local bf=$EV/displace.body
  jq -n '{model:"qwen3.8-27b-fp8", temperature:0, max_tokens:4, messages:[{role:"user", content:"Say hello."}]}' > "$bf"
  post "displace-$1" 0 /v1/chat/completions "$bf" > /dev/null
}

gate() { # NAME REF GOT
  if [ "$(ids "$2")" = "$(ids "$3")" ] && [ "$(ids "$2")" != "null" ]; then r=MATCH; else r=MISMATCH; fi
  printf '%s\t%s\t%s\t%s\n' "$1" "$r" "$(ids "$2")" "$(ids "$3")" >> "$EV/gate.tsv"
  note "gate $1: $r"
}

run_size() {
  local size=$1 doc id i q n=0
  doc=$(make_doc "$size")
  body "$doc" "" 0 > "$EV/ingest-$size.body"
  local r; r=$(post "ingest-$size" "$size" /v1/documents "$EV/ingest-$size.body")
  id=$(jq -r .id "$r")
  note "ingest $size bytes: id $id, $(jq -r .tokens "$r") tokens"
  wait_ready "$id" || return 1
  curl -s "$URL/v1/documents" > "$EV/documents-$size-after-ingest.json"
  for ((i = 0; i < QR + QN; i++)); do
    q=${QUESTIONS[$i]}
    local where=ram
    if [ "$i" -ge "$QR" ]; then
      where=nvme
      curl -s -X POST "$URL/v1/documents/$id/spill" > "$EV/spill-$size-q$i.json"
    fi
    displace "$size-q$i"
    body "$doc" "$q" 0 > "$EV/q-$size-$i.body"
    local got; got=$(post "q-$size-$i-$where" "$size" /v1/chat/completions "$EV/q-$size-$i.body")
    body "$doc" "$q" 1 > "$EV/ref-$size-$i.body"
    local ref; ref=$(post "ref-$size-$i-cold" "$size" /v1/chat/completions "$EV/ref-$size-$i.body")
    gate "size$size-q$i-$where${MUTANT:+-mutant-$MUTANT}" "$ref" "$got"
  done
  curl -s "$URL/v1/documents" > "$EV/documents-$size-end.json"
}

# Approximate join of two separately read documents (A: the first N bytes of the licence texts,
# B: the N bytes after them). Every answer is compared with the cold recompute of the joined
# prompt: token agreement (length of the common prefix, and position by position).
agree() { # REF GOT -> "prefix positional total"
  jq -n --argjson a "$(ids "$1")" --argjson b "$(ids "$2")" '
    ($a | length) as $n
    | ([range(0; [$n, ($b | length)] | min)] | map(select($a[.] == $b[.])) | length) as $pos
    | (first(range(0; [$n, ($b | length)] | min) | select($a[.] != $b[.])) // ([$n, ($b | length)] | min)) as $pre
    | "\($pre) \($pos) \($n)"' -r
}
run_join() {
  local n=$1 all=$EV/doc-all.txt a=$EV/join-a-$1.txt b=$EV/join-b-$1.txt
  make_doc 400000 > /dev/null; cp "$EV/doc-400000.txt" "$all"
  head -c "$n" "$all" > "$a"; tail -c +"$((n + 1))" "$all" | head -c "$n" > "$b"
  local ia ib
  body "$a" "" 0 > "$EV/join-ingest-a-$n.body"; body "$b" "" 0 > "$EV/join-ingest-b-$n.body"
  ia=$(jq -r .id "$(post "join-ingest-a-$n" "$n" /v1/documents "$EV/join-ingest-a-$n.body")"); wait_ready "$ia" || return 1
  ib=$(jq -r .id "$(post "join-ingest-b-$n" "$n" /v1/documents "$EV/join-ingest-b-$n.body")"); wait_ready "$ib" || return 1
  local i q
  for ((i = 0; i < ${JOIN_QUESTIONS:-2}; i++)); do
    q=${QUESTIONS[$i]}
    jq -n --rawfile da "$a" --rawfile db "$b" --arg q "$q" --argjson mt "$MAXT" '
      {model:"qwen3.8-27b-fp8", temperature:0, max_tokens:$mt, aien_return_token_ids:true,
       chat_template_kwargs:{enable_thinking:false},
       messages:[ {role:"user", content:("Here is a document.\n\n" + $da)},
                  {role:"assistant", content:"I have read the document."},
                  {role:"user", content:("Here is a document.\n\n" + $db)},
                  {role:"assistant", content:"I have read the document."},
                  {role:"user", content:$q} ]}' > "$EV/join-$n-q$i.body"
    jq '. + {aien_reuse:"cold"}' "$EV/join-$n-q$i.body" > "$EV/join-$n-q$i-cold.body"
    local ref; ref=$(post "join-$n-q$i-cold" "$n" /v1/chat/completions "$EV/join-$n-q$i-cold.body")
    # Exact baseline from the store: A restored, B re-read.
    displace "join-$n-q$i"
    local ex; ex=$(post "join-$n-q$i-store-exact" "$n" /v1/chat/completions "$EV/join-$n-q$i.body")
    gate "join$n-q$i-store-exact" "$ref" "$ex"
    printf '%s\t%s\t%s\n' "join$n-q$i" "store-exact" "$(agree "$ref" "$ex")" >> "$EV/join.tsv"
    local mode
    for mode in last first sum; do
      jq --arg r "join:$ia,$ib:$mode" '. + {aien_reuse:$r}' "$EV/join-$n-q$i.body" > "$EV/join-$n-q$i-$mode.body"
      local got; got=$(post "join-$n-q$i-$mode" "$n" /v1/chat/completions "$EV/join-$n-q$i-$mode.body")
      printf '%s\t%s\t%s\n' "join$n-q$i" "$mode" "$(agree "$ref" "$got")" >> "$EV/join.tsv"
      note "join $n q$i $mode: agreement (prefix positional total) $(agree "$ref" "$got")"
    done
  done
}

note "run start: sizes $* bin $BIN sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
start_server || exit 1
for s in "$@"; do
  case "$s" in
    join:*) run_join "${s#join:}" ;;
    *) run_size "$s" ;;
  esac
done
stop_server
note "run end"
