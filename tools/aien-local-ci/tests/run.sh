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

# tests/run.sh: proves the C8 exec contract on this host. Run as the operator (uses sudo -n). Exit 1 on any FAIL.
set -uo pipefail
X=${AIEN_EXEC:-/usr/local/sbin/aien-run-exec}   # AIEN_EXEC: test a candidate helper before install.sh installs it
RUN_HOME=/var/lib/aien-run
OP_HOME=${AIEN_OPERATOR_HOME:-$HOME}
pass=0; fail=0
ok()  { pass=$((pass+1)); printf 'PASS %s\n' "$1"; }
bad() { fail=$((fail+1)); printf 'FAIL %s: %s\n' "$1" "$2"; }
run() { sudo -n "$X" "$@"; }   # run [exec args]... -- cmd
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

uid=$(id -u aien-run)
[[ -n $uid ]] || { echo "no aien-run user"; exit 1; }

# 1 runs as uid aien-run
out=$(run --cwd /tmp --timeout 10 --profile host -- id -u 2>&1)
[[ $out == "$uid" ]] && ok "runs as uid aien-run ($uid)" || bad "uid" "$out"

# 2 env cleared: planted secret absent, no SUDO_* leaks, only the allowed names
out=$(SECRET_PLANT=hunter2 run --cwd /tmp --timeout 10 --profile host -- sh -c 'echo "${SECRET_PLANT:-unset}"; env | cut -d= -f1 | grep -v -x -E "PATH|HOME|CARGO_HOME|RUSTUP_HOME|PWD|SHLVL|_|OLDPWD" | tr "\n" ","' 2>&1)
[[ $out == "unset" ]] && ok "env cleared (planted secret absent, no extra names)" || bad "env" "$out"

# 3 --env pass-through and refused names
out=$(run --cwd /tmp --timeout 10 --profile host --env FOO=bar -- sh -c 'echo $FOO' 2>&1)
[[ $out == bar ]] && ok "--env FOO=bar passes" || bad "--env" "$out"
run --cwd /tmp --timeout 10 --profile host --env LD_PRELOAD=x -- true >/dev/null 2>&1 && bad "--env LD_PRELOAD" "accepted" || ok "--env LD_PRELOAD refused"

# 4 no network for host profile
out=$(run --cwd /tmp --timeout 20 --profile host -- sh -c 'curl -s -m 3 https://github.com >/dev/null 2>&1; echo curl=$?; getent hosts github.com >/dev/null 2>&1; echo getent=$?; ip -o link | cut -d: -f2 | tr -d " " | tr "\n" ","' 2>&1)
out=$(tr "\n" " " <<<"$out"); [[ $out == *"curl="[1-9]* && $out == *"getent="[1-9]* && $out == *"lo,"* && $out != *"enP"* && $out != *"tailscale"* ]] && ok "host profile has no network ($out)" || bad "network" "$out"

# 4b loopback is up inside the namespace (a local server works), the host's own 127.0.0.1 services stay unreachable
out=$(run --cwd /tmp --timeout 20 --profile host -- bash -c 'ip -o link show lo | grep -q "state UNKNOWN\|,UP"; echo lo_up=$?; (exec 3<>/dev/tcp/127.0.0.1/9) 2>&1 | grep -o "refused\|unreachable" | head -1; (exec 3<>/dev/tcp/127.0.0.1/3443) 2>&1 | grep -o "refused\|unreachable" | head -1' 2>&1 | tr "\n" " ")
[[ $out == "lo_up=0 refused refused " ]] && ok "host profile loopback up, host 127.0.0.1:3443 not reachable ($out)" || bad "loopback" "$out"

# 5 timeout -> 124
t0=$(date +%s); run --cwd /tmp --timeout 1 --profile host -- sleep 30 >/dev/null 2>&1; rc=$?; t1=$(date +%s)
[[ $rc -eq 124 && $((t1-t0)) -lt 15 ]] && ok "timeout returns 124 in $((t1-t0))s" || bad "timeout" "rc=$rc after $((t1-t0))s"

# 6 exit code passthrough
run --cwd /tmp --timeout 10 --profile host -- sh -c 'exit 7' >/dev/null 2>&1; rc=$?
[[ $rc -eq 7 ]] && ok "exit code passthrough (7)" || bad "exit passthrough" "rc=$rc"

# 7 cwd honored
out=$(run --cwd "$RUN_HOME" --timeout 10 --profile host -- pwd 2>&1)
[[ $out == "$RUN_HOME" ]] && ok "cwd honored" || bad "cwd" "$out"

# 8 memory cap enforced (64M cap, allocate 300M in a shell variable)
out=$(run --cwd /tmp --timeout 60 --profile host --memory-max 64M -- sh -c 'x=$(head -c 300000000 /dev/zero | tr "\0" a); echo survived' 2>&1); rc=$?
[[ $rc -ne 0 && $out != *survived* ]] && ok "memory cap enforced (rc=$rc)" || bad "memory cap" "rc=$rc out=$out"

# 9 devices: host cannot open the GPU, gb10 can
out=$(run --cwd /tmp --timeout 10 --profile host -- sh -c 'exec 3<>/dev/nvidiactl' 2>&1); rc=$?
[[ $rc -ne 0 ]] && ok "host profile cannot open /dev/nvidiactl" || bad "host device policy" "opened nvidiactl"
out=$(run --cwd /tmp --timeout 10 --profile gb10 -- sh -c 'exec 3<>/dev/nvidiactl && exec 4<>/dev/nvidia0 && exec 5<>/dev/nvidia-uvm && exec 6<>/dev/dri/renderD128 && echo opened' 2>&1)
[[ $out == opened ]] && ok "gb10 profile opens /dev/nvidiactl, nvidia0, nvidia-uvm, dri/renderD128" || bad "gb10 devices" "$out"
out=$(run --cwd /tmp --timeout 10 --profile qemu -- sh -c 'exec 3<>/dev/kvm && echo kvm' 2>&1)
run --cwd /tmp --timeout 10 --profile host -- sh -c 'exec 3<>/dev/kvm' >/dev/null 2>&1; rc=$?
[[ $out == kvm && $rc -ne 0 ]] && ok "qemu profile opens /dev/kvm, host profile cannot" || bad "kvm" "qemu=$out host_rc=$rc"

# 10 secrets of the trusted side are unreadable (direct sudo -u, as the inventory demands)
for f in "$OP_HOME/.aien/verify.key" "$OP_HOME/.config/gh/hosts.yml" "$OP_HOME/.ssh/id_ed25519"; do
  sudo -n -u aien-run cat "$f" >/dev/null 2>&1 && bad "secret $f" "READABLE" || ok "denied: $f"
done
sudo -n -u aien-run ls "$OP_HOME/workspace" >/dev/null 2>&1 && bad "workspace" "listable" || ok "denied: $OP_HOME/workspace (0700)"

# 11 toolchain as aien-run
out=$(run --cwd /tmp --timeout 60 --profile host -- cargo --version 2>&1)
[[ $out == cargo\ 1.98.1* ]] && ok "cargo as aien-run: $out" || bad "cargo" "$out"

# 12 crash recovery: kill -9 the running child; exit non-zero, no strays in the scope. Only this test's scope
#    counts (its unit is named after the helper pid): a judge running at the same time holds scopes of its own.
sudo -n "$X" --cwd /tmp --timeout 120 --profile host -- sh -c 'sleep 300 & exec sleep 200' >/dev/null 2>&1 &
bg=$!
sleep 2
victim=$(pgrep -u aien-run -f '^sleep 200$' | head -1)
helper=$(pgrep -P "$bg" | head -1)
[[ -n $victim ]] && sudo -n kill -9 "$victim"
wait $bg; rc=$?
sleep 1
strays=$(pgrep -u aien-run -f '^sleep 300$' | wc -l)
scopes=$(systemctl list-units --no-legend --plain "aien-run-exec-${helper:-none}-*.scope" 2>/dev/null | wc -l)
[[ -n $helper && $rc -ne 0 && $strays -eq 0 && $scopes -eq 0 ]] && ok "crash recovery: rc=$rc, strays=0, scopes=0" || bad "crash recovery" "helper=${helper:-not found} rc=$rc strays=$strays scopes=$scopes"

# 13 worktree protection (if a worktree exists; also used by the tests below)
# Only worktrees made by wt.sh (root-owned .git pointing into the mirror). Other lanes may be creating and removing
# their own worktrees under wt/ while this runs; those are skipped.
wt_sh_worktrees() {
  local d
  # run directory layout wt/<repo>/<sha>/<repo>/ (siblings beside it), or the older flat wt/<repo>/<sha>/
  for d in "$RUN_HOME"/wt/*/*/ "$RUN_HOME"/wt/*/*/*/; do
    d=${d%/}
    [[ -f $d/.git && $(stat -c %u "$d/.git" 2>/dev/null) == 0 ]] || continue
    grep -q "^gitdir: $RUN_HOME/mirror/" "$d/.git" 2>/dev/null && printf '%s\n' "$d"
  done
}
wt=$(wt_sh_worktrees | head -1)
if [[ -n $wt ]]; then
  out=$(run --cwd "$wt" --timeout 30 --profile host -- sh -c 'git rev-parse --short HEAD >/dev/null && echo git-ok; rm -f .git 2>/dev/null && echo REMOVED; m=$(sed "s|gitdir: ||;s|/worktrees/.*||" .git); touch "$m/refs/x" 2>/dev/null && echo MIRROR-WRITABLE; echo t > .probe && rm .probe && echo tree-writable' 2>&1 | tr '\n' ' ')
  [[ $out == *git-ok* && $out != *REMOVED* && $out != *MIRROR-WRITABLE* && $out == *tree-writable* ]] && ok "worktree: git reads, .git and mirror protected, tree writable" || bad "worktree" "$out"
fi

# sibling checkouts (SANDBOX lane): wt.sh sibling puts a read-only checkout at ../<dir> of the checkout under test;
# aien-run reads it and its git history, cannot write it or the run directory; wt.sh remove takes both away.
if [[ -d $RUN_HOME/mirror/aien-protocols.git && -d $RUN_HOME/mirror/aien-architecture.git ]]; then
  sa=$(git -C "$RUN_HOME/mirror/aien-architecture.git" rev-parse --verify -q 'refs/remotes/origin/main^{commit}')
  sp=$(git -C "$RUN_HOME/mirror/aien-protocols.git" rev-parse --verify -q 'refs/remotes/origin/main^{commit}')
  rd="$RUN_HOME/wt/aien-architecture/$sa"
  if [[ -e $rd ]]; then
    printf 'SKIP sibling test: %s is in use\n' "$rd"
  else
    co=$(bash "$here/../wt.sh" add aien-architecture "$sa") && bash "$here/../wt.sh" sibling aien-architecture "$sa" aien-protocols aien-protocols "$sp" >/dev/null
    out=$(run --cwd "$co" --timeout 30 --profile host -- sh -c 'git -C ../aien-protocols rev-parse HEAD; test -f ../aien-protocols/Cargo.toml && echo sib-read; touch ../aien-protocols/x 2>/dev/null && echo SIB-WRITABLE; touch ../x 2>/dev/null && echo RUNDIR-WRITABLE; echo t > .p && rm .p && echo tree-writable' 2>&1 | tr '\n' ' ')
    bash "$here/../wt.sh" remove aien-architecture "$sa" >/dev/null
    left=$(git -C "$RUN_HOME/mirror/aien-protocols.git" worktree list | grep -c "$rd" || true)
    [[ $out == "$sp sib-read tree-writable " && ! -e $rd && $left -eq 0 ]] \
      && ok "sibling checkout: readable with history, read-only, run dir closed, removed with the run" || bad "sibling" "$out left=$left"
  fi
fi

# 14c mirror origin URL (PIPELINE-FINISH): `wt.sh sync` copies the trusted clone's origin URL into the mirror, so a
# gate that reads `git remote get-url origin` in a sandbox checkout sees the GitHub URL, as on a GitHub runner.
cl="$HOME/workspace/pipeline/clones/aien-architecture"
if [[ -d $cl/.git ]]; then
  bash "$here/../wt.sh" sync aien-architecture "$cl" >/dev/null
  want=$(git -C "$cl" remote get-url origin)
  got=$(git -C "$RUN_HOME/mirror/aien-architecture.git" remote get-url origin 2>&1)
  [[ -n $want && $got == "$want" ]] && ok "mirror carries the trusted clone's origin URL ($got)" || bad "mirror origin" "want=$want got=$got"
fi

# ---- review fixes (INFRA-2): nothing a check writes can reach a later check ----
refused() { [[ $1 -eq 2 ]]; }   # exit 2 = refused by aien-run-exec before anything ran
T_SHA=0000000000000000000000000000000000000a15; T_SHA2=0000000000000000000000000000000000000b15
T_PR="$RUN_HOME/target/aien-sovereign-core/pr/$T_SHA"; T_MAIN="$RUN_HOME/target/aien-sovereign-core/main/$T_SHA"
T_PR2="$RUN_HOME/target/aien-sovereign-core/pr/$T_SHA2"

# 14b the operator's ~/models is visible read-only at /var/lib/aien-run/models (model gates), never writable
if [[ -d $HOME/models && -d /var/lib/aien-run/models ]]; then
  probe=$(cd "$HOME/models" && ls | head -1)
  out=$(run --cwd /tmp --timeout 20 --profile host -- sh -c "M=/var/lib/aien-run/models
    [ -e \"\$M/$probe\" ] && echo SEEN
    touch \$M/aien-run-probe 2>/dev/null && echo MODELS-WRITABLE
    echo done" 2>&1 | tr '\n' ' ')
  [[ $out == "SEEN done " ]] && ok "host: ~/models mounted read-only at /var/lib/aien-run/models" || bad "models read-only" "$out"
else
  printf 'SKIP 14b: no ~/models or no /var/lib/aien-run/models mount point\n'
fi
# 14 the sandbox home is read-only inside every profile: no ~/.cargo/config.toml, no CARGO_HOME config,
#    no swapping the registry or the cargo proxies, no new dotfiles
for p in host gb10; do
  out=$(run --cwd /tmp --timeout 20 --profile $p -- sh -c 'H=/var/lib/aien-run
    mkdir -p $H/.cargo 2>/dev/null && echo HOME-WRITABLE
    touch $H/cargo/config.toml 2>/dev/null && echo CARGO_HOME-WRITABLE
    mv $H/cargo/bin $H/cargo/bin.x 2>/dev/null && echo BIN-RENAMED
    mv $H/cargo/registry $H/cargo/r.x 2>/dev/null && echo REGISTRY-RENAMED
    touch $H/.gitconfig 2>/dev/null && echo GITCONFIG-WRITABLE
    echo done' 2>&1 | tr '\n' ' ')
  [[ $out == "done " ]] && ok "$p: sandbox home, CARGO_HOME, registry, proxies, .gitconfig all read-only" || bad "$p home read-only" "$out"
done

# 15 trust classes and heads: a PR run can write its own target/<repo>/pr/<sha> only, never main and never another
#    head's dir; a main run never pr; a run whose cwd is not in a worktree can write no worktree
sudo -n install -d -o aien-run -g aien-run -m 0755 "$T_PR2"
out=$(run --cwd /tmp --timeout 20 --profile host --env CARGO_TARGET_DIR="$T_PR" -- sh -c "touch $T_PR/.p && rm $T_PR/.p && echo pr-rw; touch $T_MAIN/.p 2>/dev/null && echo MAIN-WRITABLE; touch $T_PR2/.p 2>/dev/null && echo OTHER-HEAD-WRITABLE; echo end" 2>&1 | tr '\n' ' ')
[[ $out == "pr-rw end " ]] && ok "class pr: own head dir writable, main and another head's dir read-only" || bad "class pr" "$out"
out=$(run --cwd /tmp --timeout 20 --profile gb10 --env CARGO_TARGET_DIR="$T_MAIN" -- sh -c "touch $T_MAIN/.p && rm $T_MAIN/.p && echo main-rw; touch $T_PR/.p 2>/dev/null && echo PR-WRITABLE; echo end" 2>&1 | tr '\n' ' ')
[[ $out == "main-rw end " ]] && ok "class main (gb10): own head dir writable, target/pr read-only" || bad "class main" "$out"
sudo -n rm -rf "$T_PR" "$T_MAIN" "$T_PR2"
if [[ -n $wt ]]; then
  out=$(run --cwd /tmp --timeout 20 --profile host -- sh -c "touch $wt/.p 2>/dev/null && echo WT-WRITABLE; echo end" 2>&1 | tr '\n' ' ')
  [[ $out == "end " ]] && ok "a run outside a worktree cannot write any worktree" || bad "foreign worktree" "$out"
fi

# 16 argument refusals (exit 2, nothing runs)
run --cwd /tmp --timeout 5 --profile host --env CARGO_TARGET_DIR=/tmp/t -- true >/dev/null 2>&1; r1=$?
run --cwd /tmp --timeout 5 --profile host --env CARGO_TARGET_DIR="$RUN_HOME/target/../target/x/main" -- true >/dev/null 2>&1; r2=$?
run --cwd /tmp --timeout 5 --profile host --env CARGO_TARGET_DIR="$RUN_HOME/target/aien-sovereign-core/other" -- true >/dev/null 2>&1; r3=$?
run --cwd /tmp --timeout 5 --profile host --env FOO=1 --env FOO=2 -- true >/dev/null 2>&1; r4=$?
# the old shared class dir, a head dir with a bad sha, and a dir one level too deep
run --cwd /tmp --timeout 5 --profile host --env CARGO_TARGET_DIR="$RUN_HOME/target/aien-sovereign-core/pr" -- true >/dev/null 2>&1; r5=$?
run --cwd /tmp --timeout 5 --profile host --env CARGO_TARGET_DIR="$RUN_HOME/target/aien-sovereign-core/pr/abc" -- true >/dev/null 2>&1; r6=$?
run --cwd /tmp --timeout 5 --profile host --env CARGO_TARGET_DIR="$T_PR/x" -- true >/dev/null 2>&1; r7=$?
refused $r1 && refused $r2 && refused $r3 && refused $r4 && refused $r5 && refused $r6 && refused $r7 \
  && ok "CARGO_TARGET_DIR other than target/<repo>/pr|main/<sha40> (incl. the old shared dir) and duplicate --env refused" \
  || bad "env refusals" "$r1 $r2 $r3 $r4 $r5 $r6 $r7"
if [[ -n $wt ]]; then
  # inside a worktree the target dir must be that head's own: another head's dir is refused
  wrel=${wt#"$RUN_HOME"/wt/}; wrepo=${wrel%%/*}; wsha=${wrel#*/}; wsha=${wsha%%/*}
  had_class=0; [[ -d $RUN_HOME/target/$wrepo/pr ]] && had_class=1
  run --cwd "$wt" --timeout 5 --profile host --env CARGO_TARGET_DIR="$RUN_HOME/target/$wrepo/pr/$T_SHA2" -- true >/dev/null 2>&1; r1=$?
  run --cwd "$wt" --timeout 5 --profile host --env CARGO_TARGET_DIR="$RUN_HOME/target/$wrepo/pr/$wsha" -- true >/dev/null 2>&1; r2=$?
  sudo -n rm -rf "$RUN_HOME/target/$wrepo/pr/$wsha"
  (( had_class )) || sudo -n rmdir "$RUN_HOME/target/$wrepo/pr" 2>/dev/null || true
  refused $r1 && [[ $r2 -eq 0 ]] && ok "in a worktree only that head's own target dir is accepted" || bad "head-bound target dir" "$r1 $r2"
fi
run --cwd "$RUN_HOME/target" --timeout 5 --profile host -- true >/dev/null 2>&1; r1=$?
run --cwd "$RUN_HOME/cargo" --timeout 5 --profile host -- true >/dev/null 2>&1; r2=$?
run --cwd "$RUN_HOME/wt/../wt" --timeout 5 --profile host -- true >/dev/null 2>&1; r3=$?
r4=2
if [[ -n $wt ]]; then   # a symlink planted by a check inside its worktree, pointing at another directory
  run --cwd "$wt" --timeout 10 --profile host -- ln -sfn /tmp cwdlink >/dev/null 2>&1
  run --cwd "$wt/cwdlink" --timeout 5 --profile host -- true >/dev/null 2>&1; r4=$?
  run --cwd "$wt" --timeout 10 --profile host -- rm -f cwdlink >/dev/null 2>&1
fi
refused $r1 && refused $r2 && refused $r3 && refused $r4 && ok "--cwd outside wt/<repo>/<sha> inside the home, with .., or through a symlink refused" || bad "cwd refusals" "$r1 $r2 $r3 $r4"
run --cwd /tmp --timeout 5 --profile host -- LD_PRELOAD=/nonexistent.so true >/dev/null 2>&1; r1=$?
run --cwd /tmp --timeout 5 --profile host -- -i true >/dev/null 2>&1; r2=$?
refused $r1 && refused $r2 && ok "PROGRAM shaped like an env assignment or option refused" || bad "program refusals" "$r1 $r2"

# 17 private scratch dirs: host cannot see the operator's /tmp; every profile gets a private /var/tmp
probe=/tmp/aien-infra-probe.$$; : > "$probe"; : > /var/tmp/aien-infra-probe.$$
out=$(run --cwd /tmp --timeout 10 --profile host -- sh -c "test -e $probe && echo SEES-HOST-TMP; test -e /var/tmp/aien-infra-probe.$$ && echo SEES-VARTMP; ls -A /tmp | wc -l" 2>&1 | tr '\n' ' ')
[[ $out == "0 " ]] && ok "host: private empty /tmp and /var/tmp" || bad "host private tmp" "$out"
out=$(run --cwd /tmp --timeout 10 --profile gb10 -- sh -c "test -e $probe && echo sees-host-tmp; test -e /var/tmp/aien-infra-probe.$$ && echo SEES-VARTMP; echo end" 2>&1 | tr '\n' ' ')
[[ $out == "sees-host-tmp end " ]] && ok "gb10: shares host /tmp (documented), private /var/tmp" || bad "gb10 tmp" "$out"
rm -f "$probe" /var/tmp/aien-infra-probe.$$
out=$(run --cwd /tmp --timeout 10 --profile host -- sh -c 'echo $(ls -A /run/lock /run/screen /dev/mqueue /dev/shm 2>/dev/null | grep -vc -e "^$" -e ":$") $(ipcs -m | grep -c "^0x") $(ipcs -s | grep -c "^0x")' 2>&1)
host_ipc="$(ipcs -m | grep -c "^0x") $(ipcs -s | grep -c "^0x")"
[[ $out == "0 0 0" ]] && ok "host: private /run/lock, /run/screen, /dev/shm, /dev/mqueue and IPC namespace (host has shm+sem: $host_ipc)" || bad "host ipc" "$out"

# 18 trusted crumb on the sandbox PATH; `crumb verify .` runs as aien-run in a sandboxed worktree; a `crumb`
#    planted inside the worktree is never picked up by a bare name
out=$(run --cwd /tmp --timeout 10 --profile host -- sh -c 'command -v crumb; stat -c "%U %a" "$(command -v crumb)"' 2>&1 | tr '\n' ' ')
[[ $out == "/usr/local/bin/crumb root 755 " ]] && ok "crumb on sandbox PATH: $out" || bad "crumb PATH" "$out"
if [[ -n $wt ]]; then
  # every worktree: the sandboxed verdict equals the trusted side's verdict on a clean trusted checkout of the same
  # commit (operator git refuses to run inside a sandbox worktree: dubious ownership, by design); at least one
  # worktree (a crumb-current commit) must verify OK, so the PASS path is exercised, not only the STALE path
  agree=0; total=0; okrun=0
  for w in $(wt_sh_worktrees); do
    total=$((total+1))
    s_out=$(run --cwd "$w" --timeout 60 --profile host -- crumb verify . 2>&1); s_rc=$?
    tc=$(mktemp -d); rel=${w#"$RUN_HOME"/wt/}; repo=${rel%%/*}; rest=${rel#*/}; sha=${rest%%/*}
    git clone --quiet --shared --no-checkout "$RUN_HOME/mirror/$repo.git" "$tc/t" && git -C "$tc/t" checkout --quiet "$sha"
    t_out=$(cd "$tc/t" && /usr/local/bin/crumb verify . 2>&1); t_rc=$?
    rm -rf "$tc"
    [[ $s_rc -eq $t_rc && $s_out == "$t_out" ]] && agree=$((agree+1))
    [[ $s_rc -eq 0 ]] && okrun=$((okrun+1))
    printf '     %s: sandbox rc=%s, trusted rc=%s, %s\n' "$(basename "$w" | cut -c1-7)" "$s_rc" "$t_rc" "$(tail -1 <<<"$s_out")"
  done
  [[ $agree -eq $total && $okrun -ge 1 ]] && ok "crumb verify . as aien-run in $total sandboxed worktrees (agrees with trusted side; $okrun OK)" \
    || bad "crumb verify" "agree=$agree/$total ok=$okrun"
  run --cwd "$wt" --timeout 10 --profile host -- sh -c 'printf "#!/bin/sh\necho PLANTED\n" > crumb && chmod +x crumb' >/dev/null 2>&1
  o1=$(run --cwd "$wt" --timeout 10 --profile host -- crumb status . 2>&1 | head -3)
  o2=$(run --cwd "$wt" --timeout 10 --profile host -- sh -c 'exec crumb status .' 2>&1 | head -3)
  o3=$(run --cwd "$wt" --timeout 10 --profile host -- sh -c 'test -x ./crumb && echo planted-present' 2>&1)
  run --cwd "$wt" --timeout 10 --profile host -- rm -f crumb >/dev/null 2>&1
  [[ $o3 == planted-present && $o1 != *PLANTED* && $o2 != *PLANTED* && -n $o1 && -n $o2 ]] \
    && ok "planted ./crumb in the worktree ignored (PROGRAM and sh exec both use /usr/local/bin/crumb)" || bad "planted crumb" "o1=$o1 o2=$o2 o3=$o3"
fi
printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
