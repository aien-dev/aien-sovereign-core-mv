#!/bin/sh
# core#688 re-verification after the Rev688 fixes. Run inside the 3G scope.
cd /tmp/prfix/t688 || exit 9
export CARGO_TARGET_DIR=/tmp/prfix/t688-target AIEN_CHROME=/usr/bin/google-chrome
PAT='aien-browser-mirror-[0-9]|profile-fixture|owned-sigchld-fixture'
echo "== start $(date -u +%FT%TZ) head-tree $(git rev-parse HEAD) + working changes"
echo "== tagged before: $(pgrep -fc "$PAT")"
cargo build --locked -j1 -p aien-cli --bin aien-browser-mirror 2>&1 | grep -E '^(warning|error)|Finished'
echo "== browser_mirror tests (real Chrome, --test-threads=1)"
cargo test --locked -j1 -p aien-cli --lib browser_mirror -- --test-threads=1 2>&1 | grep -E '^test |test result|^(warning|error)'
echo "== isolated inherited SIGCHLD=SIG_IGN case"
perl -e '$SIG{CHLD}="IGNORE"; exec @ARGV' cargo test --locked -j1 -p aien-cli --lib browser_mirror::tests::cleanup_with_inherited_ignored_sigchld -- --ignored --test-threads=1 2>&1 | grep -E '^test |test result'
sleep 2
echo "== tagged after tests (pattern '$PAT'): $(pgrep -fc "$PAT")"
pgrep -fa "$PAT" | cut -c1-160
echo "== SIGINT replay"
/tmp/prfix/t688-sigint.sh "$CARGO_TARGET_DIR/debug/aien-browser-mirror"
echo "== end $(date -u +%FT%TZ)"
