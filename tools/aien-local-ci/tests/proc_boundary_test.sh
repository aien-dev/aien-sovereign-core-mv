#!/usr/bin/env bash
# SC-SEC-2: local/offline same-UID proc escape and profile lock regression.
# Run with AIEN_EXEC=... as an operator with sudo -n for the candidate helper.
# No chip device, GPU lock, or gb10 helper is opened.
set -euo pipefail
X=${AIEN_EXEC:-/usr/local/sbin/aien-run-exec}
lock=/run/aien-local-ci/execution.lock
fixture=$(mktemp -d)
tag=sec2-victim-$$
baseline_pid= host_pid= baseline_job= host_job= exclusive_job= waiting_job=
cleanup() {
  [[ -z $baseline_pid ]] || sudo -n kill -TERM "$baseline_pid" 2>/dev/null || true
  [[ -z $host_pid ]] || sudo -n kill -TERM "$host_pid" 2>/dev/null || true
  [[ -z $waiting_job ]] || kill "$waiting_job" 2>/dev/null || true
  [[ -z $exclusive_job ]] || kill "$exclusive_job" 2>/dev/null || true
  [[ -z $baseline_job ]] || wait "$baseline_job" 2>/dev/null || true
  [[ -z $host_job ]] || wait "$host_job" 2>/dev/null || true
  [[ -z $exclusive_job ]] || wait "$exclusive_job" 2>/dev/null || true
  [[ -z $waiting_job ]] || wait "$waiting_job" 2>/dev/null || true
  rm -rf -- "$fixture"
}
trap cleanup EXIT

# Wait for the real process and its ready output; no hard-coded host PID assumption.
await_victim() {
  local label=$1 logfile=$2 pid= i
  for ((i=0; i<100; i++)); do
    pid=$(pgrep -u aien-run -f "^${label} 45$" || true)
    if [[ -n $pid ]] && grep -q '^READY$' "$logfile"; then printf '%s\n' "$pid"; return 0; fi
    sleep .1
  done
  printf 'victim never became ready (%s): ' "$label" >&2
  cat "$logfile" >&2
  return 1
}

# Old host-profile namespace arrangement: private mounts but the host PID table and proc.
# The victim's /var/tmp is a private mount. A peer cannot use its own /var/tmp to
# reach it, but /proc/<same-UID-pid>/root crosses into the victim's mount namespace.
sudo -n unshare -m -n -i --propagation private -- bash -c '
  mount -t tmpfs -o mode=1777,nosuid,nodev sec2-old-victim /var/tmp
  exec setpriv --reuid aien-run --regid aien-run --init-groups --reset-env -- \
    bash -c '\''touch /var/tmp/marker; echo READY; exec -a "$1" sleep 45'\'' _ "$1"
' old-victim "$tag-old" >"$fixture/old.log" 2>&1 &
baseline_job=$!
baseline_pid=$(await_victim "$tag-old" "$fixture/old.log")
sudo -n unshare -m -n -i --propagation private -- \
  setpriv --reuid aien-run --regid aien-run --init-groups --reset-env -- \
  bash -c 'touch "/proc/$1/root/var/tmp/marker" && kill -CONT "$1"' _ "$baseline_pid"
printf 'PASS old same-UID host proc crosses victim mount and sees victim pid\n'
sudo -n kill -TERM "$baseline_pid"
wait "$baseline_job" 2>/dev/null || true
baseline_pid= baseline_job=

# Both victim and attacker now use the production helper and its actual host profile.
sudo -n "$X" --cwd /tmp --timeout 60 --profile host -- bash -c \
  'touch /var/tmp/marker; echo READY; exec -a "$1" sleep 45' _ "$tag-new" \
  >"$fixture/host.log" 2>&1 &
host_job=$!
host_pid=$(await_victim "$tag-new" "$fixture/host.log")
# This same root-held lock must be shared by a second host or qemu job, but deny
# an exclusive contender (the gb10 reservation mode) until the scope is cleaned.
grep -Fq 'host|qemu) lock_mode=-s ;; gb10) lock_mode=-x' "$X"
if sudo -n flock -n -x -- "$lock" true; then
  echo 'FAIL exclusive contender entered during host scope' >&2; exit 1
fi
sudo -n flock -n -s -- "$lock" true
second=$(sudo -n "$X" --cwd /tmp --timeout 10 --profile qemu -- printf 'SECOND\n')
[[ $second == SECOND ]]
printf 'PASS production host and qemu scopes share lock, exclusive contender blocked\n'

# Actual helper attacker sees only its own PID table and proc mount. Its own
# child remains visible and reachable, including through /proc/<child>/root.
sudo -n "$X" --cwd /tmp --timeout 10 --profile host -- bash -c '
  victim=$1
  if [[ -e /proc/$victim/root ]] || kill -CONT "$victim" 2>/dev/null || \
     touch "/proc/$victim/root/var/tmp/marker" 2>/dev/null; then exit 1; fi
  for fd in /proc/self/fd/*; do
    [[ $(readlink "$fd") != /run/aien-local-ci/execution.lock ]] || exit 1
  done
  sleep 5 & child=$!
  kill -0 "$child" && [[ -d /proc/$child/root ]] && touch "/proc/$child/root/var/tmp/own-child"
  kill "$child"; wait "$child" 2>/dev/null || true
' _ "$host_pid"
printf 'PASS private proc rejects peer root and PID, own child remains reachable\n'
sudo -n kill -TERM "$host_pid"
wait "$host_job" 2>/dev/null || true
host_pid= host_job=
sudo -n flock -n -x -- "$lock" true
printf 'PASS exclusive lock obtainable after production scope cleanup\n'

# Simulate the exclusive reservation without running the gb10 helper or touching
# devices. A host invocation must wait until that reservation is released.
sudo -n flock -x --close -- "$lock" bash -c 'echo HELD; sleep 20' >"$fixture/exclusive.log" 2>&1 &
exclusive_job=$!
for ((i=0; i<100; i++)); do
  grep -q '^HELD$' "$fixture/exclusive.log" && break
  sleep .1
done
grep -q '^HELD$' "$fixture/exclusive.log"
sudo -n "$X" --cwd /tmp --timeout 10 --profile host -- printf 'ADMITTED\n' \
  >"$fixture/waiting.log" 2>&1 &
waiting_job=$!
sleep 1
if grep -q '^ADMITTED$' "$fixture/waiting.log"; then
  echo 'FAIL host entered during exclusive reservation' >&2; exit 1
fi
kill "$exclusive_job"
wait "$exclusive_job" 2>/dev/null || true
exclusive_job=
wait "$waiting_job"
waiting_job=
grep -q '^ADMITTED$' "$fixture/waiting.log"
printf 'PASS exclusive reservation blocks host until release\n'
