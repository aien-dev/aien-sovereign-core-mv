#!/bin/sh
# core#688 re-verification after review. Run from the repository checkout,
# inside a memory-capped scope. OLD_BIN (optional) is an aien-browser-mirror
# built at 4965d44c, replayed first as the before arm.
here=$(cd "$(dirname "$0")" && pwd)
cd "$(git -C "$here" rev-parse --show-toplevel)" || exit 9
export AIEN_CHROME=${AIEN_CHROME:-/usr/bin/google-chrome}
target=${CARGO_TARGET_DIR:-target}
src=crates/aien-cli/src/browser_mirror.rs
tagged() { grep -lsa 'AIEN_BROWSER_MIRROR_TAG=' /proc/[0-9]*/environ | grep -vc "^/proc/$$/"; }
echo "== start $(date -u +%FT%TZ) HEAD $(git rev-parse HEAD)"
echo "== $src working blob $(git hash-object $src), at 9f745160 $(git rev-parse 9f745160:$src)"
echo "== code changes vs HEAD: $(git diff --name-only HEAD -- crates skills | wc -l)"
echo "== processes with a mirror tag before: $(tagged)"
cargo build --locked -j1 -p aien-cli --bin aien-browser-mirror 2>&1 | grep -E '^(warning|error)|Finished'
echo "== browser_mirror tests (real Chrome, --test-threads=1)"
cargo test --locked -j1 -p aien-cli --lib browser_mirror -- --test-threads=1 2>&1 | grep -E '^test |test result|^(warning|error)'
echo "== isolated inherited SIGCHLD=SIG_IGN case"
perl -e '$SIG{CHLD}="IGNORE"; exec @ARGV' cargo test --locked -j1 -p aien-cli --lib browser_mirror::tests::cleanup_with_inherited_ignored_sigchld -- --ignored --test-threads=1 2>&1 | grep -E '^test |test result'
sleep 2
echo "== processes with a mirror tag after tests: $(tagged)"
if [ -n "${OLD_BIN:-}" ]; then
  echo "== SIGINT replay, before: $OLD_BIN sha256 $(sha256sum "$OLD_BIN" | cut -c1-16)"
  sh "$here/sigint-replay.sh" "$OLD_BIN"
fi
echo "== SIGINT replay, this tree: $target/debug/aien-browser-mirror"
sh "$here/sigint-replay.sh" "$target/debug/aien-browser-mirror"
echo "== processes with a mirror tag at end: $(tagged)"
echo "== end $(date -u +%FT%TZ)"
