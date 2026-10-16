#!/usr/bin/env bash
# Stage 1: fetch pinned submodules and apply headstart patches (mirrors upstream scripts/setup.sh).
set -euo pipefail
H=~/workspace/headstart-exp/headstart
cd "$H"
git submodule update --init --depth 1 rustc cargo
echo "rustc HEAD $(cd rustc && git rev-parse HEAD)"
echo "cargo HEAD $(cd cargo && git rev-parse HEAD)"
for p in patches/rustc/*.patch; do echo "applying $p"; (cd rustc && git apply "$H/$p"); done
for p in patches/cargo/*.patch; do echo "applying $p"; (cd cargo && git apply "$H/$p"); done
cp config/bootstrap.toml rustc/bootstrap.toml
(cd rustc && git diff --stat | tail -1)
(cd cargo && git diff --stat | tail -1)
echo FETCH_OK
