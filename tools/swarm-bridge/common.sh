# Shared helpers for demo.sh and bench.sh (sourced, not run). Bash + jq only.
# shellcheck shell=bash

SB_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SB_REPO=$(cd "$SB_TOOLS/../.." && pwd)
SB_HOME=${SB_HOME:-$HOME/workspace/board83}
SB_DEMO=${SB_DEMO:-$SB_HOME/demo}
SB_BIN=${SB_BIN:-$SB_HOME/target-c/debug/swarm-bridge}
SB_MODEL=${SB_MODEL:-claude-haiku-5-5}
SB_BUDGET=${SB_BUDGET:-1.00}
SB_TARGET=${SB_TARGET:-$SB_DEMO/target-crumbs}
SB_SYNAPSE_URL=${SB_SYNAPSE_URL:-https://github.com/aien-spark/synapse.git}
SB_SC_URL=${SB_SC_URL:-https://github.com/aien-spark/aien-sovereign-core.git}
SB_AGENT_MAIL=${SB_AGENT_MAIL:-$HOME/.local/bin/agent-mail}
SB_CRUMB=${SB_CRUMB:-$HOME/.local/bin/crumb}

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# One shallow seed clone per repository; the crumbs test deps are prebuilt
# into SB_TARGET so neither arm pays the dependency build.
sb_seed() {
  mkdir -p "$SB_DEMO/seed"
  [ -d "$SB_DEMO/seed/synapse/.git" ] || git clone -q --depth 1 "$SB_SYNAPSE_URL" "$SB_DEMO/seed/synapse"
  [ -d "$SB_DEMO/seed/sc/.git" ] || git clone -q --depth 1 "$SB_SC_URL" "$SB_DEMO/seed/sc"
  if [ ! -f "$SB_TARGET/.prebuilt" ]; then
    log "prebuilding crumbs tests into $SB_TARGET (not measured)"
    (cd "$SB_DEMO/seed/sc" && CARGO_TARGET_DIR="$SB_TARGET" cargo test -q -p crumbs --no-run) >&2
    touch "$SB_TARGET/.prebuilt"
  fi
}

# Fresh shallow clone of a seed on a scratch branch with pushing disabled.
sb_clone() { # seed dest branch
  git clone -q --depth 1 "file://$SB_DEMO/seed/$1" "$2"
  git -C "$2" checkout -q -b "$3"
  git -C "$2" remote set-url --push origin "PUSH-DISABLED-scratch-clone"
  git -C "$2" config user.email aien@aienos.com
  git -C "$2" config user.name "swarm-bridge demo agent"
}

# Brief for one agent: task text plus the arm's messaging block.
sb_brief() { # agent arm prefix crumbdir
  local blk
  blk=$(cat "$SB_TOOLS/briefs/$1.$2")
  { cat "$SB_TOOLS/briefs/$1.task"; echo; echo "$blk"; } |
    sed -e "s|@P@|$3|g" -e "s|@CRUMBDIR@|$4|g" -e "s|@TARGET@|$SB_TARGET|g"
}

# Start one headless agent in the background. Extra env as NAME=VALUE args.
# Writes <out>/<name>.{sid,stream.jsonl,err,time,pid}.
sb_start() { # name cwd out settings(or -) prompt [env...]
  local name=$1 cwd=$2 out=$3 settings=$4 prompt=$5
  shift 5
  local sid
  sid=$(uuidgen)
  echo "$sid" >"$out/$name.sid"
  local sargs=()
  [ "$settings" = "-" ] || sargs=(--settings "$settings")
  (
    cd "$cwd" || exit 1
    exec env "$@" /usr/bin/time -f '%e %U %S' -o "$out/$name.time" \
      claude -p --model "$SB_MODEL" --session-id "$sid" --permission-mode bypassPermissions \
      --max-budget-usd "$SB_BUDGET" --output-format stream-json --verbose "${sargs[@]}" \
      "$prompt" >"$out/$name.stream.jsonl" 2>"$out/$name.err"
  ) &
  echo $! >"$out/$name.pid"
}

sb_transcript_of() { # sid
  find "$HOME/.claude/projects" -maxdepth 2 -name "$1.jsonl" 2>/dev/null | head -1
}

# Wait until an agent's transcript exists (the session started).
sb_wait_started() { # out name
  local sid i
  sid=$(cat "$1/$2.sid")
  for i in $(seq 1 240); do
    [ -n "$(sb_transcript_of "$sid")" ] && return 0
    sleep 0.25
  done
  log "agent $2 did not start"
  return 1
}

sb_collect() { # out name
  local sid t
  sid=$(cat "$1/$2.sid")
  t=$(sb_transcript_of "$sid")
  [ -n "$t" ] && cp "$t" "$1/$2.transcript.jsonl"
  [ -d "${t%.jsonl}/subagents" ] && cp -r "${t%.jsonl}/subagents" "$1/$2.subagents"
  return 0
}

# Per-transcript counters: model invocations, tool calls, waiting tool calls,
# token usage (deduplicated by API message id), first/last timestamps.
sb_counts() { # transcript
  jq -s '
    def waiting: test("(^|[;&| ])(sleep|agent-mail read|crumb sniff|swarm-bridge (watch|read))");
    ([.[] | select(.type=="assistant" and .message.id != null)] | group_by(.message.id) | map(.[0])) as $calls
    | ([.[] | select(.type=="assistant") | .message.content[]? | select(.type=="tool_use")] | unique_by(.id)) as $tools
    | {
        model_invocations: ($calls | length),
        tool_calls: ($tools | length),
        waiting_tool_calls: ([$tools[] | select((.input.command // "") | waiting)] | length),
        input_tokens: ([$calls[].message.usage.input_tokens // 0] | add // 0),
        cache_creation_tokens: ([$calls[].message.usage.cache_creation_input_tokens // 0] | add // 0),
        repeated_context_tokens: ([$calls[].message.usage.cache_read_input_tokens // 0] | add // 0),
        output_tokens: ([$calls[].message.usage.output_tokens // 0] | add // 0),
        first_ts: ([.[] | .timestamp // empty] | min),
        last_ts: ([.[] | .timestamp // empty] | max)
      }' "$1"
}

# Every delivery of a marker to a recipient, through a coordination channel:
# hook context attachments, or the results of swarm-bridge / agent-mail /
# crumb sniff commands. Prints one JSON object per delivery: {ts, via}.
sb_deliveries() { # transcript marker
  jq -c -s --arg m "$2" '
    ([.[] | select(.type=="assistant") | .message.content[]? | select(.type=="tool_use")
      | {key: .id, value: (.input.command // "")}] | from_entries) as $cmd
    | .[]
    | if .type=="attachment" and .attachment.type=="hook_additional_context"
           and ((.attachment.content | tostring) | contains($m))
      then {ts: .timestamp, via: "hook", text: (.attachment.content | tostring)}
      elif .type=="user" and (.message.content | type)=="array"
      then .timestamp as $ts | .message.content[]
        | select(.type=="tool_result")
        | ($cmd[.tool_use_id] // "") as $c
        | select($c | test("swarm-bridge|agent-mail|crumb sniff"))
        | select((.content | tostring) | contains($m))
        | {ts: $ts, via: (if ($c|test("swarm-bridge")) then "bridge-cli" else "mail-or-sniff" end), text: (.content | tostring)}
      else empty end' "$1"
}

# When the sender's send command for a marker returned.
sb_sent_at() { # transcript marker
  jq -r -s --arg m "$2" '
    ([.[] | select(.type=="assistant") | .message.content[]? | select(.type=="tool_use")
      | select((.input.command // "") | (contains($m) and test("swarm-bridge (question|answer|post)|agent-mail send")))
      | .id]) as $ids
    | [.[] | select(.type=="user" and (.message.content|type)=="array")
        | .timestamp as $ts | .message.content[] | select(.type=="tool_result")
        | select(.tool_use_id as $i | $ids | index($i)) | $ts] | min // empty' "$1"
}

sb_epoch() { date -u -d "$1" +%s.%N; }

# Keep only what evidence needs from a transcript: the conversation and the
# swarm-board hook context. Other hooks' context, request blobs and harness
# bookkeeping (which can carry operator-private settings) are dropped.
sb_scrub() { # transcript
  jq -c 'select(
      (.type=="assistant") or
      (.type=="user") or
      (.type=="attachment" and .attachment.type=="hook_additional_context"
        and ((.attachment.content|tostring) | test("UNTRUSTED COORDINATION DATA from the swarm board")))
    ) | del(.message.content[]? | select(.type=="thinking") | .signature)' "$1"
}

# Assemble result.json for a run. Pairs: "marker:sender:recipient".
sb_metrics() { # out arm run
  local out=$1 arm=$2 run=$3 a f agents='{}' relays='[]' pairs='[]' cpu=0
  for a in alpha beta gamma; do
    agents=$(jq -c --arg a "$a" --argjson c "$(sb_counts "$out/$a.transcript.jsonl")" \
      --arg t "$(cat "$out/$a.time" 2>/dev/null | tail -1)" \
      '. + {($a): ($c + {time: $t})}' <<<"$agents")
  done
  for f in "$out"/relay*.transcript.jsonl; do
    [ -f "$f" ] || continue
    relays=$(jq -c --argjson c "$(sb_counts "$f")" '. + [$c]' <<<"$relays")
  done
  for f in "$out"/*.time; do
    [ -f "$f" ] || continue
    cpu=$(tail -1 "$f" | awk -v c="$cpu" '{printf "%.2f", c + $2 + $3}')
  done
  local p m s r sent dl
  for p in "[Q1]:alpha:beta" "[Q1]:alpha:gamma" "[A1]:beta:alpha" "[A1]:beta:gamma"; do
    IFS=: read -r m s r <<<"$p"
    sent=$(sb_sent_at "$out/$s.transcript.jsonl" "$m")
    dl=$(sb_deliveries "$out/$r.transcript.jsonl" "$m" | jq -s -c 'map({ts, via})')
    pairs=$(jq -c --arg m "$m" --arg s "$s" --arg r "$r" --arg sent "$sent" --argjson d "$dl" '
      def ep: (.[0:19]+"Z" | fromdateiso8601) + (.[19:-1] | if .=="" then 0 else ("0"+. | tonumber) end);
      . + [{msg: $m, from: $s, to: $r, sent: $sent, deliveries: ($d|length),
            first_via: ($d[0].via // null),
            first_seen: ($d | map(.ts) | min),
            latency_s: (if $sent != "" and ($d|length) > 0 then (($d|map(.ts)|min|ep) - ($sent|ep)) else null end)}]' <<<"$pairs")
  done
  jq -n --arg arm "$arm" --arg run "$run" --argjson agents "$agents" --argjson relays "$relays" \
    --argjson pairs "$pairs" --arg cpu "$cpu" '
    def ep: (.[0:19] + "Z" | fromdateiso8601) + (.[19:-1] | if . == "" then 0 else ("0" + . | tonumber) end);
    def wall($who):
      ([$pairs[] | select(.msg == "[A1]" and .to == $who) | .first_seen][0]) as $x
      | ([$pairs[] | select(.msg == "[Q1]" and .to == "beta") | .sent][0]) as $q
      | if ($x // "") != "" and ($q // "") != "" then (($x | ep) - ($q | ep)) else null end;
    {arm: $arm, run: $run, agents: $agents, relays: $relays,
     ferrying_invocations: ([$relays[].model_invocations] | add // 0),
     totals: {
       agent_model_invocations: ([$agents[].model_invocations] | add),
       model_invocations_all: (([$agents[].model_invocations] | add) + ([$relays[].model_invocations] | add // 0)),
       tool_calls: (([$agents[].tool_calls] | add) + ([$relays[].tool_calls] | add // 0)),
       waiting_tool_calls: ([$agents[].waiting_tool_calls] | add),
       repeated_context_tokens: (([$agents[].repeated_context_tokens] | add) + ([$relays[].repeated_context_tokens] | add // 0)),
       input_side_tokens: ([($agents[], $relays[]) | .input_tokens + .cache_creation_tokens + .repeated_context_tokens] | add),
       cpu_s: ($cpu | tonumber)},
     pairs: $pairs,
     missed: ([$pairs[] | select(.deliveries == 0)] | length),
     duplicates: ([$pairs[] | select(.deliveries > 1) | .deliveries - 1] | add // 0),
     coordination_wall_s: {alpha_has_answer: wall("alpha"), gamma_has_answer: wall("gamma")}
    }'
}

# Task result: alpha's note committed and quoting the answer; gamma's test
# committed, quoting the answer, and passing when rerun here.
sb_task_check() { # run_dir
  local R=$1 note test_ok=false note_ok=false
  note="$R/synapse-alpha/docs/notes/ledger-verify-contract.md"
  if [ -f "$note" ] && git -C "$R/synapse-alpha" log -1 --format=%s | grep -q 'ledger verify contract' &&
    grep -qi 'genesis' "$note"; then note_ok=true; fi
  local t="$R/sc-gamma/crates/crumbs/tests/ledger_verify_empty.rs"
  if [ -f "$t" ] && grep -q 'A1' "$t" && git -C "$R/sc-gamma" log -1 --format=%s | grep -q 'ledger verify empty' &&
    (cd "$R/sc-gamma" && CARGO_TARGET_DIR="$SB_TARGET" cargo test -q -p crumbs --test ledger_verify_empty >"$R/out/gamma-test-rerun.txt" 2>&1); then
    test_ok=true
  fi
  jq -n --argjson n "$note_ok" --argjson t "$test_ok" '{alpha_note_ok: $n, gamma_test_ok: $t}'
}

# Per-run tool shims on PATH. Control arm: agent-mail and crumb run with HOME
# pointed at the run's own home, so inboxes and whispers land in the run and
# never in the real ~/shared/agents. Board arm: swarm-bridge.
sb_shims() { # run_dir
  local R=$1
  mkdir -p "$R/bin" "$R/home/shared/agents"
  cat >"$R/bin/agent-mail" <<SH
#!/usr/bin/env bash
exec env HOME="$R/home" CRUMB_COORD_ROOT="$R/home/shared/agents" PATH="$R/bin:\$PATH" "$SB_AGENT_MAIL" "\$@"
SH
  cat >"$R/bin/crumb" <<SH
#!/usr/bin/env bash
exec env HOME="$R/home" CRUMB_COORD_ROOT="$R/home/shared/agents" "$SB_CRUMB" "\$@"
SH
  ln -sf "$SB_BIN" "$R/bin/swarm-bridge"
  chmod +x "$R/bin/agent-mail" "$R/bin/crumb"
}

# Orchestrator relay step of the control arm: for each new mail in the
# orchestrator inbox, one headless model call reads it and reposts it.
# Runs until the stop file exists. Each call is relay<N>.
sb_relay_loop() { # run_dir prefix
  local R=$1 P=$2 n=0 f inbox mail prompt
  inbox="$R/home/shared/agents/inbox/${P}orch"
  mkdir -p "$inbox/done"
  while [ ! -f "$R/out/relay.stop" ]; do
    for f in "$inbox"/*.md; do
      [ -f "$f" ] || continue
      n=$((n + 1))
      mail=$(cat "$f")
      mv "$f" "$inbox/done/"
      prompt=$(sed -e "s|@P@|$P|g" "$SB_TOOLS/briefs/relay.prompt")
      prompt=${prompt//@MAIL@/$mail}
      log "relay $n: $(basename "$f")"
      sb_start "relay$n" "$R/home" "$R/out" - "$prompt" PATH="$R/bin:$PATH"
      wait "$(cat "$R/out/relay$n.pid")" || true
    done
    sleep 1
  done
}

# One full run of one arm. Writes $SB_DEMO/<run>/out/{result.json,...}.
sb_run_arm() { # arm(board|mail) run
  local arm=$1 run=$2 R P a
  R="$SB_DEMO/$run"
  [ -e "$R" ] && { log "run dir $R exists; refusing to reuse"; return 1; }
  mkdir -p "$R/out" "$R/state"
  P="${run}-"
  sb_shims "$R"
  sb_clone synapse "$R/synapse-alpha" "scratch/$run-alpha"
  sb_clone sc "$R/sc-beta" "scratch/$run-beta"
  sb_clone sc "$R/sc-gamma" "scratch/$run-gamma"
  local cwd_alpha="$R/synapse-alpha" cwd_beta="$R/sc-beta" cwd_gamma="$R/sc-gamma"
  local settings=- relay_pid=
  if [ "$arm" = board ]; then
    "$SB_BIN" init-board "$R/board" >"$R/out/board-init.txt"
    settings="$SB_TOOLS/demo-settings.json"
  else
    sb_relay_loop "$R" "$P" &
    relay_pid=$!
  fi
  date -u +%Y-%m-%dT%H:%M:%S.%3NZ >"$R/out/started"
  for a in beta gamma alpha; do
    local cwd_var="cwd_$a" brief envs
    brief=$(sb_brief "$a" "$arm" "$P" "$R/home/shared/agents")
    envs=(PATH="$R/bin:$PATH")
    if [ "$arm" = board ]; then
      envs+=(SWARM_BRIDGE_BIN="$SB_BIN" SWARM_BRIDGE_BOARD_ROOT="$R/board"
        SWARM_BRIDGE_STATE_DIR="$R/state" SWARM_TAG="$a" SWARM_BRIDGE_SESSION="$run-$a")
      [ "$a" = gamma ] && envs+=(SWARM_BRIDGE_SUBSCRIBE="thread:contract/ledger-verify")
    fi
    sb_start "$a" "${!cwd_var}" "$R/out" "$settings" "$brief" "${envs[@]}"
    sb_wait_started "$R/out" "$a"
  done
  for a in beta gamma alpha; do wait "$(cat "$R/out/$a.pid")" || log "$a exited non-zero"; done
  date -u +%Y-%m-%dT%H:%M:%S.%3NZ >"$R/out/finished"
  if [ -n "$relay_pid" ]; then
    touch "$R/out/relay.stop"
    wait "$relay_pid" || true
  fi
  for f in "$R"/out/*.sid; do sb_collect "$R/out" "$(basename "$f" .sid)"; done
  sb_finish "$R" "$arm" "$run"
}

# Score a finished run (also reruns on an existing run dir; agents are not
# started again).
sb_finish() { # run_dir arm run
  local R=$1 arm=$2 run=$3
  sb_metrics "$R/out" "$arm" "$run" >"$R/out/metrics.json"
  sb_task_check "$R" >"$R/out/task.json"
  if [ "$arm" = board ]; then
    "$SB_BIN" --board-root "$R/board" board-events >"$R/out/board-events.jsonl"
    sb_board_assert "$R" >"$R/out/assert.json"
  else
    echo '{}' >"$R/out/assert.json"
  fi
  jq -s '.[0] + {task: .[1], board: .[2]}' "$R/out/metrics.json" "$R/out/task.json" \
    "$R/out/assert.json" >"$R/out/result.json"
  log "run $run ($arm) done: $(jq -c '{missed, duplicates, ferrying_invocations, task, pass: .board.pass}' "$R/out/result.json")"
}

# Event ids an agent's transcript shows: posted ids from its own commands,
# and seen ids per section (addressed/subscribed) and channel (hook/cli).
sb_seen_ids() { # transcript
  jq -s -c '
    def ids($kind): [scan("#([0-9]+) " + $kind + " from=") | .[0] | tonumber];
    def sections: (split("From your subscriptions:") | {addr: .[0], sub: (.[1] // "")});
    [ .[]
      | if .type=="attachment" and .attachment.type=="hook_additional_context"
             and ((.attachment.content|tostring) | contains("UNTRUSTED COORDINATION DATA from the swarm board"))
        then (.attachment.content|tostring|sections) as $s
          | {via: "hook", addressed_q: ($s.addr|ids("question")), addressed_a: ($s.addr|ids("answer")),
             subscribed_q: ($s.sub|ids("question")), subscribed_a: ($s.sub|ids("answer"))}
        elif .type=="user" and (.message.content|type)=="array"
        then .message.content[] | select(.type=="tool_result") | (.content|tostring) as $t
          | {via: "cli", posted: [$t | scan("posted #([0-9]+) on board") | .[0] | tonumber],
             addressed_q: ($t|sections.addr|ids("question")), addressed_a: ($t|sections.addr|ids("answer")),
             subscribed_q: ($t|sections.sub|ids("question")), subscribed_a: ($t|sections.sub|ids("answer"))}
        else empty end ]
    | {posted: ([.[].posted[]?] | unique),
       hook_sub_q: ([.[] | select(.via=="hook") | .subscribed_q[]] | unique),
       hook_sub_a: ([.[] | select(.via=="hook") | .subscribed_a[]] | unique),
       hook_addr_q: ([.[] | select(.via=="hook") | .addressed_q[]] | unique),
       hook_addr_a: ([.[] | select(.via=="hook") | .addressed_a[]] | unique),
       any_q: ([.[] | (.addressed_q + .subscribed_q)[]] | unique),
       any_a: ([.[] | (.addressed_a + .subscribed_a)[]] | unique)}' "$1"
}

# The board arm success rule (RULE.md items 1-4).
sb_board_assert() { # run_dir
  local R=$1
  jq -n --argjson al "$(sb_seen_ids "$R/out/alpha.transcript.jsonl")" \
    --argjson be "$(sb_seen_ids "$R/out/beta.transcript.jsonl")" \
    --argjson ga "$(sb_seen_ids "$R/out/gamma.transcript.jsonl")" \
    --argjson ev "$(jq -s -c . "$R/out/board-events.jsonl")" \
    --argjson m "$(cat "$R/out/metrics.json")" --argjson t "$(cat "$R/out/task.json")" '
    ($al.posted[0]) as $q | ($be.posted[0]) as $a
    | {question_id: $q, answer_id: $a, alpha: $al, beta: $be, gamma: $ga,
       board_events: ($ev | length),
       r1_same_question_id: ($q != null and ($be.any_q | index($q)) != null and ($ga.any_q | index($q)) != null),
       r1_same_answer_id: ($a != null and ($al.any_a | index($a)) != null and ($ga.any_a | index($a)) != null),
       r1_gamma_subscribed_via_hook: ($q != null and $a != null
          and ($ga.hook_sub_q | index($q)) != null and ($ga.hook_sub_a | index($a)) != null
          and ($ga.hook_addr_q | index($q)) == null and ($ga.hook_addr_a | index($a)) == null),
       r2_no_relay_two_events: ($m.ferrying_invocations == 0 and ($ev | length) == 2
          and ($al.posted | length) == 1 and ($be.posted | length) == 1 and ($ga.posted | length) == 0),
       r3_no_missed_no_dup: ($m.missed == 0 and $m.duplicates == 0),
       r4_task: ($t.alpha_note_ok and $t.gamma_test_ok)}
    | . + {pass: (.r1_same_question_id and .r1_same_answer_id and .r1_gamma_subscribed_via_hook
                  and .r2_no_relay_two_events and .r3_no_missed_no_dup and .r4_task)}'
}

# Copy one run's evidence into the repo: metrics, rule check, board events,
# scrubbed transcripts (conversation + swarm-board hook context only), and
# what each agent committed.
sb_evidence() { # run_dir dest
  local R=$1 D=$2 f n
  mkdir -p "$D/transcripts" "$D/commits"
  for f in result.json metrics.json task.json assert.json board-events.jsonl gamma-test-rerun.txt \
    started finished board-init.txt; do
    [ -f "$R/out/$f" ] && cp "$R/out/$f" "$D/"
  done
  for f in "$R"/out/*.time; do [ -f "$f" ] && cp "$f" "$D/transcripts/"; done
  for f in "$R"/out/*.transcript.jsonl; do
    [ -f "$f" ] || continue
    n=$(basename "$f")
    sb_scrub "$f" >"$D/transcripts/$n"
  done
  for n in synapse-alpha sc-beta sc-gamma; do
    [ -d "$R/$n/.git" ] || continue
    git -C "$R/$n" log --format='%H %s' "$(git -C "$R/$n" rev-list --max-parents=0 HEAD | tail -1)..HEAD" \
      >"$D/commits/$n.log" 2>/dev/null || true
    git -C "$R/$n" diff "$(git -C "$R/$n" rev-list --max-parents=0 HEAD | tail -1)" HEAD \
      >"$D/commits/$n.diff" 2>/dev/null || true
  done
}

# Side-by-side summary of all runs under a results dir (RULE.md comparison).
sb_summary() { # results_dir
  jq -s '
    def pct($p): sort | if length == 0 then null else .[((length - 1) * $p | floor)] end;
    def arm($a): [.[] | select(.arm == $a)];
    (arm("mail")) as $m | (arm("board")) as $b
    | def col($runs; f): [$runs[] | f];
      def side(f): {mail: col($m; f), board: col($b; f),
                    board_lower_every_pair: ([range(0; ([($m|length), ($b|length)] | min))
                       | (col($b; f)[.] < col($m; f)[.])] | (length > 0 and all))};
    {runs: {mail: ($m | length), board: ($b | length)},
     board_rule_pass: [$b[] | {run, pass: .board.pass}],
     ferrying_invocations: side(.ferrying_invocations),
     agent_model_invocations: side(.totals.agent_model_invocations),
     model_invocations_all: side(.totals.model_invocations_all),
     tool_calls: side(.totals.tool_calls),
     waiting_tool_calls: side(.totals.waiting_tool_calls),
     repeated_context_tokens: side(.totals.repeated_context_tokens),
     input_side_tokens: side(.totals.input_side_tokens),
     cpu_s: side(.totals.cpu_s),
     coordination_wall_s_alpha: side(.coordination_wall_s.alpha_has_answer),
     coordination_wall_s_gamma: side(.coordination_wall_s.gamma_has_answer),
     missed: side(.missed), duplicates: side(.duplicates),
     task_ok: {mail: col($m; (.task.alpha_note_ok and .task.gamma_test_ok)),
               board: col($b; (.task.alpha_note_ok and .task.gamma_test_ok))},
     delivery_latency_s: {
       mail: ([$m[].pairs[].latency_s | select(. != null)] | {n: length, p50: pct(0.5), p95: pct(0.95)}),
       board: ([$b[].pairs[].latency_s | select(. != null)] | {n: length, p50: pct(0.5), p95: pct(0.95)})},
     joules: "NOT_MEASURED"}' "$1"/*/result.json
}
