#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ARTIFACT="${1:?provide output archive path}"
PACKAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$PACKAGE_DIR"' EXIT
mkdir -p "$PACKAGE_DIR/bin"
for binary in aien-cli spark-cockpit-rs spark-inquisitor cortex-encoder-rs cortex-rs spark-supervisor spark-debugger; do
    [[ -x "target/release/$binary" ]] || { echo "Missing release binary: $binary" >&2; exit 1; }
    name="$binary"
    [[ "$binary" != aien-cli ]] || name=aien
    cp "target/release/$binary" "$PACKAGE_DIR/bin/$name"
done
cp CONSTITUTION.md README.md install.sh "$PACKAGE_DIR/"
if [[ -d imprints/en2-trinity ]]; then
    mkdir -p "$PACKAGE_DIR/imprints"
    cp -R imprints/en2-trinity "$PACKAGE_DIR/imprints/"
fi
tar -czf "$ARTIFACT" -C "$PACKAGE_DIR" .
