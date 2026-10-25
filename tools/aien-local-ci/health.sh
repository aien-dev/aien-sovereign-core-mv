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

# health.sh: one-screen health of the sovereign local pipeline host pieces (lane INFRA). Read-only.
set -uo pipefail
RUN_HOME=/var/lib/aien-run
FORGEJO_DATA=/home/drakestapleton/atlas-forgejo/data
FORGEJO_PORT=$(grep -E '^HTTP_PORT' "$FORGEJO_DATA/custom/conf/app.ini" 2>/dev/null | awk -F= '{gsub(/ /,"",$2); print $2}')
FORGEJO_PORT=${FORGEJO_PORT:-3443}
STATE=/home/drakestapleton/.local/state/aien-local-ci
GPU_LOCK=/tmp/aien-gb10.lock
SLOTS=/home/drakestapleton/.local/state/aien-proof/slots

row() { printf '%-22s %s\n' "$1" "$2"; }

if mountpoint -q "$FORGEJO_DATA"; then row "forgejo storage" "mounted ($FORGEJO_DATA)"; else row "forgejo storage" "NOT MOUNTED"; fi
code=$(curl -sk -m 3 -o /dev/null -w '%{http_code}' "https://127.0.0.1:$FORGEJO_PORT/" 2>/dev/null)
if [[ $code == 200 ]]; then row "forgejo web" "up (https://127.0.0.1:$FORGEJO_PORT, $(systemctl is-active atlas-forgejo.service 2>/dev/null))"; else row "forgejo web" "DOWN (http $code, unit $(systemctl is-active atlas-forgejo.service 2>/dev/null))"; fi
if (exec 3<>/dev/tcp/127.0.0.1/2222) 2>/dev/null; then row "forgejo ssh" "listening 127.0.0.1:2222"; else row "forgejo ssh" "NOT listening on 2222"; fi
row "forgejo-runner" "$(systemctl --user is-active forgejo-runner.service 2>/dev/null) (enabled: $(systemctl --user is-enabled forgejo-runner.service 2>/dev/null)); must stay off"

if u=$(getent passwd aien-run); then
  row "aien-run user" "ok uid=$(cut -d: -f3 <<<"$u") groups=$(id -nG aien-run | tr ' ' ',') shell=$(cut -d: -f7 <<<"$u")"
else
  row "aien-run user" "MISSING"
fi
[[ -x /usr/local/sbin/aien-run-exec ]] && row "aien-run-exec" "$(stat -c '%U %a' /usr/local/sbin/aien-run-exec) $(sudo -n -l /usr/local/sbin/aien-run-exec >/dev/null 2>&1 && echo 'sudo ok' || echo 'SUDO RULE MISSING')" || row "aien-run-exec" "MISSING"
if [[ -x /usr/local/bin/crumb ]]; then
  want=$(sed -n 's/^CRUMB_COMMIT=//p' "$(dirname "$0")/install.sh" 2>/dev/null | cut -c1-7)
  row "crumb (sandbox)" "/usr/local/bin/crumb $(stat -c '%U %a' /usr/local/bin/crumb) sha256 $(sha256sum < /usr/local/bin/crumb | cut -c1-12) (install.sh pins aien-architecture ${want:-?})"
else
  row "crumb (sandbox)" "MISSING (run install.sh)"
fi
hl=$(stat -c '%U:%G %a' "$RUN_HOME" "$RUN_HOME/cargo" "$RUN_HOME/target" 2>/dev/null | tr '\n' ' ')
[[ $hl == "root:aien-run 750 root:root 755 root:root 755 " ]] && row "sandbox home layout" "root-owned (home, cargo, target)" || row "sandbox home layout" "UNSAFE: $hl(re-run install.sh)"

for c in cargo/registry cargo/git rustup/toolchains; do
  if [[ -d $RUN_HOME/$c ]]; then row "cache $c" "$(sudo -n du -sh "$RUN_HOME/$c" 2>/dev/null | cut -f1) $(stat -c '%U %a' "$RUN_HOME/$c")"; else row "cache $c" "MISSING"; fi
done
for t in "$RUN_HOME"/target/*/*/; do [[ -d $t ]] && row "target ${t#"$RUN_HOME"/target/}" "$(sudo -n du -sh "$t" 2>/dev/null | cut -f1)"; done
n=0; for w in "$RUN_HOME"/wt/*/*/; do [[ -d $w ]] && n=$((n+1)); done
row "worktrees" "$n under $RUN_HOME/wt"
row "mirrors" "$(ls "$RUN_HOME/mirror" 2>/dev/null | tr '\n' ' ')"
row "scopes running" "$(systemctl list-units --no-legend --plain 'aien-run-exec-*.scope' 2>/dev/null | wc -l)"
row "disk free /" "$(df -h / | awk 'NR==2{print $4" of "$2" ("$5" used)"}')"
row "triggers pending" "$(ls "$STATE/trigger" 2>/dev/null | wc -l)"

if [[ -e $GPU_LOCK ]]; then
  holder=$(cat "$GPU_LOCK" 2>/dev/null | head -c 120)
  if command -v fuser >/dev/null && fuser "$GPU_LOCK" >/dev/null 2>&1; then
    row "gpu lock" "HELD by pid(s) $(fuser "$GPU_LOCK" 2>/dev/null | tr -s ' ') ${holder:+[$holder]}"
  else
    row "gpu lock" "file present, no live holder (flock free) ${holder:+[$holder]}"
  fi
else
  row "gpu lock" "absent"
fi
for s in "$SLOTS"/*.lock; do
  [[ -e $s ]] || continue
  if fuser "$s" >/dev/null 2>&1; then row "slot $(basename "$s")" "HELD by pid $(fuser "$s" 2>/dev/null | tr -s ' ')"; else row "slot $(basename "$s")" "free"; fi
done
if command -v quietlock >/dev/null; then
  q=$(quietlock check 2>&1 | head -1)
  if quietlock check >/dev/null 2>&1; then row "quiet flag" "clear${q:+ ($q)}"; else row "quiet flag" "RAISED${q:+ ($q)}"; fi
else
  row "quiet flag" "quietlock not installed"
fi
