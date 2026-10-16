#!/usr/bin/env bash
# Proves the ON arm activates headstart: JSON message reasons and the rustc
# flags cargo passes, on the upstream smoke workspace, OFF vs ON.
set -uo pipefail
E=~/workspace/headstart-exp
H=$E/headstart
R=$H/rustc/build/host/stage1/bin/rustc
C=$H/cargo/target/release/cargo
t=$(mktemp -d)
cp -r "$H/tests/smoke/." "$t/"
cd "$t"
for m in false true; do
  rm -rf target
  echo "### CARGO_UNSTABLE_HEADSTART=$m"
  env RUSTC="$R" CARGO_UNSTABLE_HEADSTART=$m "$C" check -v --color never --message-format=json > out.json 2> err.txt
  echo "exit=$?"
  echo "json reasons:"; jq -r .reason out.json | sort | uniq -c
  echo "rustc invocations with -Zearly-metadata: $(grep -c -- '-Zearly-metadata' err.txt)"
  grep -o -- '-Zearly-metadata[^ ]*' err.txt | sort | uniq -c
done
rm -rf "$t"
