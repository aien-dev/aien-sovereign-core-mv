#!/usr/bin/env bash
# Vendored copy of synapse-coord (the synapse#83 board) for swarm-bridge.
#
# The Spark judge builds offline with a read-only cargo home, so a git
# dependency can never resolve there. The crate is copied into
# third_party/synapse-coord at one pinned upstream rev, recorded in
# third_party/synapse-coord.pin.
#
#   vendor-synapse-coord.sh sync REV   fetch REV, regenerate the copy, rewrite the pin
#   vendor-synapse-coord.sh check      offline: the copy matches the pin's file digest
#   vendor-synapse-coord.sh upstream   fetch the pinned rev, check its tree id, regenerate
#                                      into a temp dir and diff against the copy
#
# The copy is byte-identical to upstream crates/synapse-coord except
# Cargo.toml: upstream's manifest is kept as Cargo.toml.orig and Cargo.toml
# has the `x.workspace = true` lines resolved from upstream's root
# [workspace.package] and [workspace.dependencies] (what `cargo package`
# does). SYNAPSE_GIT may name a local clone to fetch from instead of GitHub.
set -euo pipefail

REPO_URL=https://github.com/aien-dev/synapse
CRATE=crates/synapse-coord
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEST="$ROOT/third_party/synapse-coord"
PIN="$ROOT/third_party/synapse-coord.pin"

die() { echo "vendor-synapse-coord: $*" >&2; exit 1; }

pin_get() { # key
  local v
  v=$(grep -E "^$1=" "$PIN" | head -1 | cut -d= -f2-) || true
  [ -n "$v" ] || die "PinIncomplete: no $1= in $PIN"
  printf %s "$v"
}

# sha256 over "sha256  relative/path" lines of every file, sorted by path.
digest() { # dir
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) | sha256sum | cut -c1-64
}

# Value line for KEY from section SECTION of a TOML file (simple `key = value` lines only).
toml_line() { # file section key
  awk -v sec="[$2]" -v key="$3" '
    /^\[/ { insec = ($0 == sec); next }
    insec && $0 ~ "^" key "[ \t]*=" { print; found = 1; exit }
    END { if (!found) exit 1 }' "$1"
}

# Resolve upstream crate manifest against upstream root manifest.
resolve_manifest() { # crate_toml root_toml rev
  local crate=$1 root=$2 rev=$3 section= line key
  printf '# Vendored from %s %s at %s by\n' "$REPO_URL" "$CRATE" "$rev"
  printf '# tools/swarm-bridge/vendor-synapse-coord.sh. Do not edit; upstream manifest: Cargo.toml.orig\n'
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      \[*\]) section=${line#[}; section=${section%]} ;;
    esac
    if [[ "$line" =~ ^([A-Za-z0-9_-]+)\.workspace[[:space:]]*=[[:space:]]*true[[:space:]]*$ ]]; then
      key=${BASH_REMATCH[1]}
      case "$section" in
        package) toml_line "$root" workspace.package "$key" || die "Unresolved: package.$key" ;;
        dependencies | dev-dependencies | build-dependencies)
          toml_line "$root" workspace.dependencies "$key" || die "Unresolved: $section.$key" ;;
        *) die "Unresolved: $key.workspace in [$section]" ;;
      esac
    elif [[ "$line" == *workspace* ]]; then
      die "UnsupportedManifest: can not resolve line: $line"
    else
      printf '%s\n' "$line"
    fi
  done <"$crate"
}

# Fetch REV into a temp bare repo; print its dir.
fetch() { # rev
  local rev=$1 tmp src=${SYNAPSE_GIT:-$REPO_URL}
  tmp=$(mktemp -d)
  git init -q --bare "$tmp/git"
  git -C "$tmp/git" fetch -q --depth 1 "$src" "$rev" || die "FetchFailed: $rev from $src"
  [ "$(git -C "$tmp/git" rev-parse FETCH_HEAD)" = "$rev" ] || die "RevMismatch: fetched $(git -C "$tmp/git" rev-parse FETCH_HEAD), want $rev"
  printf %s "$tmp"
}

# Build the vendored tree for REV from fetched repo TMP into OUT.
generate() { # tmp rev out
  local tmp=$1 rev=$2 out=$3
  mkdir -p "$tmp/x" "$out"
  git -C "$tmp/git" archive "$rev" "$CRATE" Cargo.toml | tar -x -C "$tmp/x"
  cp -R "$tmp/x/$CRATE/." "$out/"
  mv "$out/Cargo.toml" "$out/Cargo.toml.orig"
  resolve_manifest "$out/Cargo.toml.orig" "$tmp/x/Cargo.toml" "$rev" >"$out/Cargo.toml"
}

cmd=${1:-}
case "$cmd" in
  sync)
    rev=${2:-}
    [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || die "BadRev: sync needs a full 40-hex commit id"
    tmp=$(fetch "$rev")
    trap 'rm -rf "$tmp"' EXIT
    tree=$(git -C "$tmp/git" rev-parse "$rev:$CRATE")
    rm -rf "$DEST"
    generate "$tmp" "$rev" "$DEST"
    {
      echo "# Pin for third_party/synapse-coord (tools/swarm-bridge/vendor-synapse-coord.sh)."
      echo "repo=$REPO_URL"
      echo "rev=$rev"
      echo "path=$CRATE"
      echo "tree=$tree"
      echo "digest=$(digest "$DEST")"
    } >"$PIN"
    echo "vendored $CRATE at $rev (tree $tree)"
    ;;
  check)
    [ -f "$PIN" ] || die "NoPin: $PIN"
    [ -d "$DEST" ] || die "NoCopy: $DEST"
    want=$(pin_get digest)
    got=$(digest "$DEST")
    [ "$want" = "$got" ] || die "DigestMismatch: third_party/synapse-coord was edited (pin $want, tree $got)"
    head -2 "$DEST/Cargo.toml" | grep -q "at $(pin_get rev) by" || die "ManifestRevMismatch: Cargo.toml header does not name the pinned rev"
    echo "ok: third_party/synapse-coord matches pin rev $(pin_get rev)"
    ;;
  upstream)
    rev=$(pin_get rev)
    tmp=$(fetch "$rev")
    trap 'rm -rf "$tmp"' EXIT
    tree=$(git -C "$tmp/git" rev-parse "$rev:$CRATE")
    [ "$tree" = "$(pin_get tree)" ] || die "TreeMismatch: upstream $CRATE at $rev is $tree, pin says $(pin_get tree)"
    generate "$tmp" "$rev" "$tmp/out"
    diff -r "$tmp/out" "$DEST" >&2 || die "CopyDiffers: third_party/synapse-coord differs from upstream $rev"
    [ "$(digest "$tmp/out")" = "$(pin_get digest)" ] || die "DigestMismatch: regenerated copy does not match the pin digest"
    echo "ok: third_party/synapse-coord is upstream $REPO_URL $CRATE at $rev (tree $tree)"
    ;;
  *)
    die "usage: vendor-synapse-coord.sh sync REV | check | upstream"
    ;;
esac
