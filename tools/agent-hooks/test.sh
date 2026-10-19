#!/usr/bin/env bash
# test.sh: red and green cases for every hook in tools/agent-hooks/.
# Run from anywhere: bash tools/agent-hooks/test.sh. Exits non-zero on any failure.
#
# Canonical copy: aien-architecture tools/agent-hooks/ (ADR 0038). Every repo in
# the Aienix tree carries an identical copy; edit the canonical copy and recopy.
# The duplication ends when the aienix tree import happens.
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export TMPDIR="$work/tmp"
mkdir -p "$TMPDIR"
pass=0 fail=0 slow=0

json_cmd() { python3 -c 'import json,sys; print(json.dumps({"session_id":"t","tool_name":"Bash","tool_input":{"command":sys.argv[1]}}))' "$1"; }
json_edit() { python3 -c 'import json,sys; print(json.dumps({"session_id":"t","tool_name":"Edit","tool_input":{"file_path":sys.argv[1]}}))' "$1"; }

# expect <exit code> <hook> <json> <label>
expect() {
  local want=$1 hook=$2 in=$3 label=$4 got t0 t1 ms
  t0=$(date +%s%N)
  printf '%s' "$in" | "$here/$hook" >"$work/out" 2>"$work/err"
  got=$?
  t1=$(date +%s%N); ms=$(( (t1 - t0) / 1000000 ))
  if [ "$got" = "$want" ]; then
    pass=$((pass + 1)); printf 'ok   %-18s exit %s %4sms  %s\n' "$hook" "$got" "$ms" "$label"
  else
    fail=$((fail + 1)); printf 'FAIL %-18s exit %s (want %s)  %s  stderr: %s\n' "$hook" "$got" "$want" "$label" "$(head -c 200 "$work/err")"
  fi
  # The 200 ms budget is a target, not a pass/fail line: a loaded machine slows
  # python3 start-up. Slow allow cases are counted and printed.
  if [ "$want" = 0 ] && [ "$ms" -ge 200 ]; then
    slow=$((slow + 1)); printf 'SLOW %-18s took %sms (target under 200ms)  %s\n' "$hook" "$ms" "$label"
  fi
}
# expect_out <hook> <json> <substring or empty for no output> <label>
expect_out() {
  local hook=$1 in=$2 want=$3 label=$4 out
  out=$(printf '%s' "$in" | "$here/$hook" 2>/dev/null)
  if { [ -z "$want" ] && [ -z "$out" ]; } || { [ -n "$want" ] && [[ "$out" == *"$want"* ]]; }; then
    pass=$((pass + 1)); printf 'ok   %-18s output  %s\n' "$hook" "$label"
  else
    fail=$((fail + 1)); printf 'FAIL %-18s output [%s] (want [%s])  %s\n' "$hook" "$out" "$want" "$label"
  fi
}

h=no-admin-merge.sh
expect 2 $h "$(json_cmd 'gh pr merge 5 --admin --squash')" 'red: plain --admin'
expect 2 $h "$(json_cmd 'cd x && gh pr merge 5 --squash --admin')" 'red: after &&'
expect 2 $h "$(json_cmd 'bash -c "gh pr merge 1 --admin"')" 'red: inside bash -c'
expect 0 $h "$(json_cmd 'gh pr merge 5 --squash')" 'green: normal merge'
expect 0 $h "$(json_cmd 'git commit -m "never gh pr merge --admin"')" 'green: quoted text only'
expect 0 $h "$(json_cmd 'ls -la')" 'green: unrelated'
expect 0 $h 'not json' 'green: bad input is ignored'

h=no-force-push.sh
expect 2 $h "$(json_cmd 'git push -f')" 'red: -f'
expect 2 $h "$(json_cmd 'git push --force origin x')" 'red: --force'
expect 2 $h "$(json_cmd 'git push -uf origin b')" 'red: -uf cluster'
expect 2 $h "$(json_cmd 'git -C /x push --force')" 'red: git -C dir push'
expect 2 $h "$(json_cmd 'git push origin +main')" 'red: +refspec'
expect 2 $h "$(json_cmd 'make && git push -f origin b')" 'red: after &&'
expect 0 $h "$(json_cmd 'git push --force-with-lease origin b')" 'green: --force-with-lease'
expect 0 $h "$(json_cmd 'git push origin a:refs/heads/a')" 'green: normal push'
expect 0 $h "$(json_cmd 'git commit -m "do not git push -f"')" 'green: quoted text only'
expect 0 $h "$(json_cmd 'git fetch -f origin')" 'green: fetch -f'
expect 0 $h "$(json_cmd 'ls')" 'green: unrelated'

h=no-kill-jobs.sh
expect 2 $h "$(json_cmd 'pkill -f aien-test')" 'red: pkill aien-test'
expect 2 $h "$(json_cmd 'killall aien-serve-http')" 'red: killall aien-serve-http'
expect 2 $h "$(json_cmd 'pkill -f "claude -p"')" 'red: pkill claude -p'
expect 2 $h "$(json_cmd 'pkill claude')" 'red: pkill claude'
expect 2 $h "$(json_cmd 'kill $(pgrep aien-local-ci)')" 'red: kill $(pgrep ...)'
expect 2 $h "$(json_cmd 'sudo kill -9 `pgrep -f aien-test`')" 'red: backticks'
# A short-lived stand-in whose command line names aien-test; it exits by itself.
bash -c 'exec -a aien-test-standin sleep 3' &
standin=$!
sleep 0.2
expect 2 $h "$(json_cmd "kill $standin")" 'red: kill PID of an aien-test process'
expect 2 $h "$(json_cmd "kill -TERM $standin")" 'red: kill -TERM PID'
expect 0 $h "$(json_cmd 'kill 4194999')" 'green: kill PID that does not exist'
expect 0 $h "$(json_cmd 'pgrep -fa aien-test')" 'green: looking is fine'
expect 0 $h "$(json_cmd 'pkill -f my-own-scratch-loop')" 'green: unrelated target'
expect 0 $h "$(json_cmd 'echo skill')" 'green: unrelated'

h=protected-paths.sh
repo="$work/repo"; mkdir -p "$repo/.git"
for p in .crumb crumb.lock .aien/verify.pub tests/gates/g.toml .github/workflows/ci.yml .forgejo/workflows/x.yml tools/aien-test/src/main.rs sub/.crumb; do
  expect 2 $h "$(json_edit "$repo/$p")" "red: $p"
done
for p in .crumb.local docs/x.md tests/unit/x.rs README.md .github/CODEOWNERS tools/agent-hooks/x.sh; do
  expect 0 $h "$(json_edit "$repo/$p")" "green: $p"
done
expect 0 $h '{"tool_input":{"command":"ls"}}' 'green: no file_path'
AIEN_PIPELINE_EDIT=1 expect 0 $h "$(json_edit "$repo/.crumb")" 'green: .crumb with AIEN_PIPELINE_EDIT=1'

h=skill-context.sh
mkdir -p "$repo/skills/demo-basics" "$repo/skills/hooks" "$repo/src" "$repo/tools/agent-hooks" "$repo/lib/skills/lib-only" "$repo/lib/a"
printf -- '---\nname: demo-basics\ndescription: d\n---\nbody\n' >"$repo/skills/demo-basics/SKILL.md"
printf -- '---\nname: hooks\ndescription: h\npaths:\n  - "tools/agent-hooks/**"\n---\nbody\n' >"$repo/skills/hooks/SKILL.md"
printf -- '---\nname: lib-only\ndescription: l\npaths: ["a/**"]\n---\nbody\n' >"$repo/lib/skills/lib-only/SKILL.md"
printf 'schema = 1\n[component]\nname = "demo"\ntree = "core"\n[[skills]]\npath = "skills/hooks"\n[[skills]]\npath = "skills/demo-basics"\n' >"$repo/agents.toml"
expect 0 $h "$(json_edit "$repo/src/main.rs")" 'green: never blocks'
rm -f "$TMPDIR"/aien-skill-context-*
expect_out $h "$(json_edit "$repo/src/main.rs")" 'Read skills/demo-basics/SKILL.md' 'component skill for src/'
expect_out $h "$(json_edit "$repo/src/other.rs")" '' 'same directory again: silent'
expect_out $h "$(json_edit "$repo/tools/agent-hooks/x.sh")" 'Read skills/hooks/SKILL.md' 'paths: glob match'
expect_out $h "$(json_edit "$repo/lib/a/x.rs")" 'Read lib/skills/lib-only/SKILL.md' 'nearest skills/ dir wins'
expect_out $h "$(json_edit "$repo/lib/b.rs")" 'Read skills/demo-basics/SKILL.md' 'nested non-match falls back to component'
mkdir -p "$repo/docs"
expect_out $h "$(json_edit "$repo/docs/x.md")" '{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "Read skills/demo-basics/SKILL.md before editing here"}}' 'exact PreToolUse hook object' 
bare="$work/bare"; mkdir -p "$bare/.git" "$bare/x"
expect_out $h "$(json_edit "$bare/x/y.rs")" '' 'repo without skills: silent'
expect 0 $h "$(json_edit "$bare/x/y.rs")" 'green: repo without skills'

printf '\n%s passed, %s failed, %s slow\n' "$pass" "$fail" "$slow"
[ "$fail" = 0 ]
