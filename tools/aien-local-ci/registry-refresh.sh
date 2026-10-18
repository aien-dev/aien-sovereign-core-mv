#!/usr/bin/env bash
# registry-refresh.sh: add to the judge's read-only cargo registry the crates a trusted Cargo.lock names
# that it does not have yet (issue #622). The judge builds with `--locked --offline` against
# /var/lib/aien-run/cargo/registry, which install.sh seeded once from the operator's ~/.cargo; nothing
# refreshed it, so every PR that adds or bumps a crate failed on the judge. This is the operator-run refresh
# step. The trust model is unchanged: the registry stays root-owned and read-only to aien-run, nothing here
# touches the network, and the source of every byte is the operator's own cache.
#
#   bash tools/aien-local-ci/registry-refresh.sh <trusted Cargo.lock> [--dry-run]
#
# For each crates.io package in the lockfile that is missing from the judge registry: copy the .crate from
# the operator's ~/.cargo cache, refuse unless its sha256 equals the lockfile checksum, unpack that verified
# .crate into registry/src the way cargo does (with its .cargo-ok marker), then copy the matching index-cache
# entry. The unpacked source is required: the judge's CARGO_HOME is read-only, so its cargo cannot unpack a
# .crate itself and fails with "failed to unpack package ... Permission denied". A crate whose .crate is
# already in the judge cache but whose source was never unpacked (the state an older refresh left) gets its
# source unpacked from the judge's own .crate after the same checksum check. Prints what it did and appends
# one journal line so the next bundle can be traced to the registry state it ran with. Never deletes or
# replaces an existing crate file or source directory.
#
# Test overrides (tests/registry_refresh_test.sh uses them on throwaway directories, no sudo):
#   AIEN_JUDGE_REGISTRY, AIEN_OPERATOR_REGISTRY, AIEN_JOURNAL, AIEN_INSTALL (the privileged copy command),
#   AIEN_AS_ROOT (the privileged prefix for the source tree copy; empty in tests).
set -euo pipefail

lock=${1:-}
dry=0
for a in "${@:2}"; do case $a in --dry-run) dry=1 ;; *) echo "unknown argument: $a" >&2; exit 2 ;; esac; done
[[ -n $lock && -f $lock ]] || { echo "usage: registry-refresh.sh <trusted Cargo.lock> [--dry-run]" >&2; exit 2; }

OP_HOME=${AIEN_OP_HOME:-$HOME}
JUDGE=${AIEN_JUDGE_REGISTRY:-/var/lib/aien-run/cargo/registry}
OPER=${AIEN_OPERATOR_REGISTRY:-$OP_HOME/.cargo/registry}
JOURNAL=${AIEN_JOURNAL:-$OP_HOME/.local/state/aien-local-ci/journal.jsonl}
INSTALL=${AIEN_INSTALL:-sudo -n install -o root -g root}
AS_ROOT=${AIEN_AS_ROOT-sudo -n}
CRATES_IO='registry+https://github.com/rust-lang/crates.io-index'

say() { printf '[registry-refresh] %s\n' "$*"; }
die() { echo "registry-refresh: $*" >&2; exit 1; }

[[ -d $JUDGE/cache && -d $OPER/cache ]] || die "registry layout missing (judge $JUDGE, operator $OPER)"
# One index directory on each side (index.crates.io-<hash>); the names must agree or the judge's cargo
# would never look where we copy.
idx_judge=$(ls "$JUDGE/cache"); idx_oper=$(ls "$OPER/cache")
[[ $idx_judge == "$idx_oper" && $(wc -w <<<"$idx_judge") -eq 1 ]] || die "index directory mismatch: judge '$idx_judge' operator '$idx_oper'"
idx=$idx_judge
SRC_DIR=$JUDGE/src/$idx

# cargo's sparse-index cache path for a crate name: 1/<n>, 2/<n>, 3/<c>/<n>, <ab>/<cd>/<n> (lower case).
index_rel() {
  local n; n=$(printf '%s' "$1" | tr 'A-Z' 'a-z')
  case ${#n} in
    1) printf '1/%s' "$n" ;;
    2) printf '2/%s' "$n" ;;
    3) printf '3/%s/%s' "${n:0:1}" "$n" ;;
    *) printf '%s/%s/%s' "${n:0:2}" "${n:2:2}" "$n" ;;
  esac
}

# Lockfile packages from crates.io: "name version checksum" per line.
packages() {
  awk -v src="$CRATES_IO" '
    function emit() { if (name != "" && source == src) print name, version, checksum }
    /^\[\[package\]\]/ { emit(); name=""; version=""; source=""; checksum="" }
    /^name = /     { gsub(/"/, "", $3); name=$3 }
    /^version = /  { gsub(/"/, "", $3); version=$3 }
    /^source = /   { gsub(/"/, "", $3); source=$3 }
    /^checksum = / { gsub(/"/, "", $3); checksum=$3 }
    END { emit() }' "$lock"
}

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/registry-refresh.XXXXXX")
trap 'chmod -R u+w "$STAGE" 2>/dev/null; rm -rf "$STAGE"' EXIT

# Unpack a verified .crate into $SRC_DIR/<name>-<version> the way cargo would, so the judge's read-only cargo
# finds it already unpacked. Staged as the operator, copied as root under a dot name, made read-only, then
# renamed into place in one step so a judge run never sees a half-written tree. Returns 1 (nothing written)
# unless the archive holds exactly one top-level directory named <name>-<version>.
unpack() {
  local crate=$1 top=$2 s
  s=$(mktemp -d "$STAGE/u.XXXXXX")
  tar -xzf "$crate" -C "$s" --no-same-owner 2>/dev/null || return 1
  [[ $(ls -A "$s") == "$top" && -d $s/$top && ! -L $s/$top ]] || return 1
  # cargo writes this marker after a complete unpack and skips the unpack when it reads {"v":1}.
  printf '{"v":1}' > "$s/$top/.cargo-ok"
  [[ -e $SRC_DIR/$top || -e $SRC_DIR/.$top.tmp ]] && return 1
  $AS_ROOT cp -R "$s/$top" "$SRC_DIR/.$top.tmp"
  $AS_ROOT chmod -R a-w,a+rX "$SRC_DIR/.$top.tmp"
  $AS_ROOT mv "$SRC_DIR/.$top.tmp" "$SRC_DIR/$top"
}

lock_digest=$(sha256sum "$lock" | cut -d' ' -f1)
added=() unpacked=() missing_src=() refused=()
total=0
while read -r name version checksum; do
  [[ -n $name ]] || continue
  total=$((total + 1))
  top="$name-$version"; file="$top.crate"
  have_crate=0 have_src=0
  [[ -e $JUDGE/cache/$idx/$file ]] && have_crate=1
  [[ -e $SRC_DIR/$top/.cargo-ok ]] && have_src=1
  (( have_crate && have_src )) && continue
  if [[ -e $SRC_DIR/$top ]]; then refused+=("$file: $SRC_DIR/$top exists without .cargo-ok (half unpacked; not replaced)"); continue; fi
  [[ -n $checksum ]] || { refused+=("$file: no checksum in lockfile"); continue; }
  # The bytes to unpack: the judge's own .crate if it has one, otherwise the operator's.
  if (( have_crate )); then src="$JUDGE/cache/$idx/$file"; else src="$OPER/cache/$idx/$file"; fi
  if [[ ! -f $src ]]; then missing_src+=("$file"); continue; fi
  have=$(sha256sum "$src" | cut -d' ' -f1)
  if [[ $have != "$checksum" ]]; then refused+=("$file: sha256 $have, lockfile says $checksum"); continue; fi
  if (( dry )); then
    if (( have_crate )); then say "would unpack $top (judge has the .crate, no source)"; unpacked+=("$file")
    else say "would add $file"; added+=("$file"); fi
    continue
  fi
  unpack "$src" "$top" || { refused+=("$file: does not unpack to a single $top/ directory (nothing written)"); continue; }
  if (( have_crate )); then say "unpacked $top from the judge's .crate (sha256 verified)"; unpacked+=("$file"); continue; fi
  # Add only: check again right before the copy, then never overwrite.
  [[ -e $JUDGE/cache/$idx/$file ]] || $INSTALL -m 0444 "$src" "$JUDGE/cache/$idx/$file"
  rel=$(index_rel "$name")
  if [[ -f $OPER/index/$idx/.cache/$rel ]]; then
    $INSTALL -d -m 0755 "$JUDGE/index/$idx/.cache/$(dirname "$rel")"
    # The index-cache entry lists every version cargo has seen; the operator's copy is a superset.
    $INSTALL -m 0444 "$OPER/index/$idx/.cache/$rel" "$JUDGE/index/$idx/.cache/$rel"
  else
    say "note: no operator index-cache entry for $name (cargo will still find the .crate by its lockfile checksum)"
  fi
  say "added $file and its unpacked source (sha256 verified)"
  added+=("$file")
done < <(packages)

(( total > 0 )) || die "no crates.io packages found in $lock"

for m in "${missing_src[@]+"${missing_src[@]}"}"; do say "cannot add $m: not in the operator cache (fetch it as the operator first)"; done
for r in "${refused[@]+"${refused[@]}"}"; do say "REFUSED $r"; done

jlist() { printf '%s\n' "$@" | jq -R . | jq -sc 'map(select(. != ""))'; }
if (( ! dry )); then
  mkdir -p "$(dirname "$JOURNAL")"
  jq -nc --arg utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg action registry-refresh --arg lockfile "$lock" \
    --arg lock_sha256 "$lock_digest" --arg packages "$total" \
    --argjson added "$(jlist "${added[@]+"${added[@]}"}")" \
    --argjson unpacked "$(jlist "${unpacked[@]+"${unpacked[@]}"}")" \
    --argjson missing "$(jlist "${missing_src[@]+"${missing_src[@]}"}")" \
    --argjson refused "$(jlist "${refused[@]+"${refused[@]}"}")" \
    '$ARGS.named' >> "$JOURNAL"
fi
say "lockfile $lock_digest: $total crates.io packages, ${#added[@]} added, ${#unpacked[@]} unpacked, ${#missing_src[@]} not in operator cache, ${#refused[@]} refused"
(( ${#refused[@]} == 0 )) || exit 1
