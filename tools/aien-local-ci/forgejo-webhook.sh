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

# forgejo-webhook.sh: offline push trigger for the trusted worker (A3).
#
# Trigger file contract (consumed by `aien-test worker`, lane ENGINE):
#   directory  $OP_HOME/.local/state/aien-local-ci/trigger/   (OP_HOME = $AIEN_OPERATOR_HOME, default $HOME)
#   file name  <repo>.<sha>            (repo = Forgejo repository name, sha = full 40-hex new commit)
#   content    one line: "<ref> <old-sha> <new-sha> <unix-time>"  (ref = refs/heads/<branch>)
#   semantics  the worker treats the file's presence as "please verify <repo> at <sha>"; it deletes the file when
#              the run is recorded. A deleted branch (new-sha all zeros) writes no file. Re-pushing the same sha
#              rewrites the same file (idempotent). No network, no HTTP receiver: Forgejo's post-receive hook
#              runs as the operator on this host and writes the file directly.
#
#   forgejo-webhook.sh install <repo>   install the post-receive hook into the Forgejo bare repo
#                                       $OP_HOME/atlas-forgejo/data/git/repositories/drake/<repo>.git
#                                       (hooks/post-receive runs every hooks/post-receive.d/*, Forgejo's own layout,
#                                       so a later "resynchronize hooks" from the admin page keeps our script)
#   forgejo-webhook.sh hook <repo>      the hook body (reads "old new ref" lines on stdin)
#   forgejo-webhook.sh list             show pending trigger files
set -euo pipefail
OP_HOME=${AIEN_OPERATOR_HOME:-$HOME}
TRIGGER_DIR=$OP_HOME/.local/state/aien-local-ci/trigger
REPOS=$OP_HOME/atlas-forgejo/data/git/repositories/drake

case "${1:-}" in
  hook)
    repo=${2:?repo}
    mkdir -p "$TRIGGER_DIR"
    now=$(date +%s)
    while read -r old new ref; do
      [[ $new =~ ^[0-9a-f]{40}$ ]] || continue
      [[ $new != 0000000000000000000000000000000000000000 ]] || continue
      [[ $ref == refs/heads/* ]] || continue
      tmp="$TRIGGER_DIR/.$repo.$new.$$"
      printf '%s %s %s %s\n' "$ref" "$old" "$new" "$now" > "$tmp"
      mv -f "$tmp" "$TRIGGER_DIR/$repo.$new"
    done
    ;;
  install)
    repo=${2:?repo}
    [[ $repo =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad repo" >&2; exit 2; }
    bare="$REPOS/$repo.git"
    [[ -d $bare && -f $bare/HEAD ]] || { echo "no bare repo at $bare" >&2; exit 2; }
    mkdir -p "$bare/hooks/post-receive.d"
    if [[ ! -f $bare/hooks/post-receive ]]; then
      cat > "$bare/hooks/post-receive" <<'EOF'
#!/usr/bin/env bash
data=$(cat)
exitcodes=""
hookname=$(basename "$0")
GIT_DIR=${GIT_DIR:-$(dirname "$0")/..}
for hook in "$GIT_DIR"/hooks/"$hookname".d/*; do
  test -x "$hook" && test -f "$hook" || continue
  echo "$data" | "$hook"
  exitcodes="$exitcodes $?"
done
for i in $exitcodes; do
  [ "$i" -eq 0 ] || exit "$i"
done
EOF
      chmod 0755 "$bare/hooks/post-receive"
    fi
    {
      printf '#!/usr/bin/env bash\n# installed by tools/aien-local-ci/forgejo-webhook.sh (sovereign-core); writes the worker trigger file\n'
      printf 'TRIGGER_DIR=%q\nrepo=%q\n' "$TRIGGER_DIR" "$repo"
      cat <<'EOF'
mkdir -p "$TRIGGER_DIR"
now=$(date +%s)
while read -r old new ref; do
  [[ $new =~ ^[0-9a-f]{40}$ ]] || continue
  [[ $new != 0000000000000000000000000000000000000000 ]] || continue
  [[ $ref == refs/heads/* ]] || continue
  tmp="$TRIGGER_DIR/.$repo.$new.$$"
  printf '%s %s %s %s\n' "$ref" "$old" "$new" "$now" > "$tmp"
  mv -f "$tmp" "$TRIGGER_DIR/$repo.$new"
done
exit 0
EOF
    } > "$bare/hooks/post-receive.d/aien-local-ci"
    chmod 0755 "$bare/hooks/post-receive.d/aien-local-ci"
    echo "installed $bare/hooks/post-receive.d/aien-local-ci -> $TRIGGER_DIR/$repo.<sha>"
    ;;
  list)
    ls -l "$TRIGGER_DIR" 2>/dev/null || echo "no trigger dir"
    ;;
  *) sed -n '2,18p' "$0" >&2; exit 2 ;;
esac
