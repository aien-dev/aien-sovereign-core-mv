#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Offline launcher tests. Git, aien-test, process discovery, detached starts
# and kill are stubs. No worker, remote command or device access is performed.
# Pass another launcher path to check the same controls against older source.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLI=${1:-$HERE/../aien-local-ci}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pass=0 fail=0
check() {
  local name=$1; shift
  if "$@"; then pass=$((pass + 1)); printf 'PASS %s\n' "$name"
  else fail=$((fail + 1)); printf 'FAIL %s\n' "$name"; fi
}
expect_argv() {
  local file=$1; shift
  printf '%s\0' "$@" > "$T/expected"
  cmp -s "$T/expected" "$file"
}
export AIEN_STATE="$T/state space" AIEN_CLONES="$T/clones space"
export AIEN_TEST_BIN="$T/bin/aien-test" AIEN_REPOS=alpha,beta AIEN_POLL=17
export AIEN_CLI_TEST_ROOT=$T
export PATH="$T/bin:$PATH"
mkdir -p "$T/bin" "$AIEN_CLONES/omega"
cat > "$T/bin/aien-test" <<'EOF'
#!/usr/bin/env bash
printf '%s\0' "$@" > "$AIEN_CLI_TEST_ROOT/bin.argv"
pwd > "$AIEN_CLI_TEST_ROOT/bin.cwd"
EOF
cat > "$T/bin/git" <<'EOF'
#!/usr/bin/env bash
printf '%s\0' "$@" >> "$AIEN_CLI_TEST_ROOT/git.argv"
[[ $# == 5 && $1 == -C && $2 == "$AIEN_CLONES/omega" ]] || exit 94
if [[ $3 == fetch && $4 == -q && $5 == --all ]]; then exit 0; fi
[[ $3 == rev-parse && $4 == --verify ]] || exit 94
[[ $5 == '01234567^{commit}' ]] || exit 2
printf '%s\n' 0123456789abcdef0123456789abcdef01234567
EOF
cat > "$T/bin/pgrep" <<'EOF'
#!/usr/bin/env bash
# No real process is adopted during a test.
exit 1
EOF
cat > "$T/bin/setsid" <<'EOF'
#!/usr/bin/env bash
printf '%s\0' "$@" > "$AIEN_CLI_TEST_ROOT/start.argv"
# Record the request only. Do not execute nohup or start a worker.
: > "$AIEN_CLI_TEST_ROOT/start.done"
EOF
chmod +x "$T/bin/"*
# Bash resolves this exported function before its kill builtin. The only
# accepted PID is synthetic; neither kill -0 nor kill reaches the host.
kill() {
  printf '%s\0' "$@" >> "$AIEN_CLI_TEST_ROOT/kill.argv"
  [[ $# == 2 && $1 == -0 && $2 == 424242 ]] || [[ $# == 1 && $1 == 424242 ]]
}
export -f kill
fresh() {
  rm -rf "$AIEN_STATE"
  mkdir -p "$AIEN_STATE"
  rm -f "$T/start.done"
  : > "$T/bin.argv"; : > "$T/git.argv"; : > "$T/start.argv"; : > "$T/kill.argv"
}
run() {
  if bash "$CLI" "$@" > "$T/stdout" 2> "$T/stderr"; then rc=0; else rc=$?; fi
}
started() {
  local i
  for ((i=0; i<100; i++)); do
    [[ -f $T/start.done ]] && return 0
    sleep 0.01
  done
  return 1
}
expect_trigger() {
  local kind parent head stamp extra
  local file=$AIEN_STATE/trigger/omega.0123456789abcdef0123456789abcdef01234567
  [[ -f $file ]] || return 1
  read -r kind parent head stamp extra < "$file"
  [[ $kind == manual && $parent == 0000000000000000000000000000000000000000 &&
     $head == 0123456789abcdef0123456789abcdef01234567 && -z $extra &&
     $stamp =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]
}

fresh
run plan --operator --base origin/main
check 'plan exits successfully' test "$rc" -eq 0
check 'plan preserves operator and base flags' expect_argv "$T/bin.argv" plan --operator --base origin/main

fresh
run trigger omega 01234567
check 'trigger exits successfully' test "$rc" -eq 0
check 'trigger resolves the requested repo and commit' expect_argv "$T/git.argv" -C "$AIEN_CLONES/omega" rev-parse --verify '01234567^{commit}'
check 'trigger records the resolved full SHA' expect_trigger
check 'trigger does not execute a worker' test ! -s "$T/bin.argv"

fresh
run trigger omega unknown
check 'unknown trigger commit is refused' test "$rc" -eq 2
check 'unknown trigger commit writes no trigger' test -z "$(find "$AIEN_STATE/trigger" -type f -print -quit)"

fresh
run verify omega 01234567 --operator --base origin/main
check 'verify exits successfully' test "$rc" -eq 0
check 'verify preserves repo, resolved SHA and flags' expect_argv "$T/bin.argv" verify 0123456789abcdef0123456789abcdef01234567 --repo github.com/aien-dev/omega --state-dir "$AIEN_STATE" --operator --base origin/main
check 'verify uses the requested clone' test "$(cat "$T/bin.cwd")" = "$AIEN_CLONES/omega"

fresh
run once omega 01234567 --operator --base origin/main
check 'once exits successfully' test "$rc" -eq 0
check 'once preserves repo, resolved SHA and flags' expect_argv "$T/bin.argv" worker --once omega 0123456789abcdef0123456789abcdef01234567 --state-dir "$AIEN_STATE" --clones "$AIEN_CLONES" --sandbox user --operator --base origin/main
check 'once uses only the expected stub git calls' expect_argv "$T/git.argv" -C "$AIEN_CLONES/omega" fetch -q --all -C "$AIEN_CLONES/omega" rev-parse --verify '01234567^{commit}'

fresh
printf 'one\ntwo\nthree\nfour\n' > "$AIEN_STATE/journal.jsonl"
run log 2
check 'log preserves its line count' test "$(cat "$T/stdout")" = $'three\nfour'

fresh
run start --repos omega --lane big --github
check 'named start exits successfully' test "$rc" -eq 0
check 'named start reaches the detached-start stub' started
check 'named start preserves lane repo arguments' expect_argv "$T/start.argv" nohup "$AIEN_TEST_BIN" worker --poll 17 --state-dir "$AIEN_STATE" --clones "$AIEN_CLONES" --sandbox user --github --repos omega
check 'named start uses its lane pidfile' test -s "$AIEN_STATE/worker-big.pid"
check 'named start leaves the main lane alone' test ! -e "$AIEN_STATE/worker.pid"

fresh
run start
check 'default start exits successfully' test "$rc" -eq 0
check 'default start reaches the detached-start stub' started
check 'default start keeps trigger-only arguments' expect_argv "$T/start.argv" nohup "$AIEN_TEST_BIN" worker --poll 17 --state-dir "$AIEN_STATE" --clones "$AIEN_CLONES" --sandbox user
check 'default start uses the main pidfile' test -s "$AIEN_STATE/worker.pid"

fresh
run start --lane big --repos omega
check 'named start still requires github polling' test "$rc" -eq 2
check 'refused start does not reach the detached-start stub' test ! -e "$T/start.done"
run start --github --lane big
check 'named start still requires a repo list' test "$rc" -eq 2
run start --github --lane Bad --repos omega
check 'invalid lane names are still refused' test "$rc" -eq 2

fresh
printf '424242\n' > "$AIEN_STATE/worker-big.pid"
printf 'main-sentinel\n' > "$AIEN_STATE/worker.pid"
run stop --lane big
check 'named stop exits successfully' test "$rc" -eq 0
check 'named stop checks and signals only its synthetic pid' expect_argv "$T/kill.argv" -0 424242 424242
check 'named stop removes its pidfile' test ! -e "$AIEN_STATE/worker-big.pid"
check 'named stop preserves the main pidfile' test "$(cat "$AIEN_STATE/worker.pid")" = main-sentinel

fresh
printf '424242\n' > "$AIEN_STATE/worker.pid"
run stop
check 'default stop exits successfully' test "$rc" -eq 0
check 'default stop checks and signals only its synthetic pid' expect_argv "$T/kill.argv" -0 424242 424242
check 'default stop removes the main pidfile' test ! -e "$AIEN_STATE/worker.pid"

printf '%d passed, %d failed\n' "$pass" "$fail"
(( fail == 0 ))
