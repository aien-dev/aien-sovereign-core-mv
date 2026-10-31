#!/bin/sh
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

# Run one formal oracle job on the proof board and bind the result to its evidence.
#
#   formal-job.sh formal/jobs/<name>.job
#
# Needs: AIEN_ARCH_DIR (aien-architecture checkout holding the oracle and manifests),
#        aien-proof on PATH (or AIEN_PROOF_BIN), gh authenticated (to fetch the covered file once).
# Optional: AIEN_FORMAL_CACHE (default ~/.cache/aien-formal), AIEN_PROOF_DIR (board dir).
# Prints the receipt ID. Exit code follows the receipt result (0 PASS).
#
# The covered implementation file is fetched at the pinned COVERED_COMMIT into the cache and
# hashed like any input, so the pin is part of the job fingerprint. Changing the job file, this
# script, the invariant manifest, any Lean source, or the covered file changes the job key.
set -eu
job_file=${1:?usage: formal-job.sh JOBFILE}
here=$(cd "$(dirname "$0")" && pwd)
job_file=$(cd "$(dirname "$job_file")" && pwd)/$(basename "$job_file")
arch=${AIEN_ARCH_DIR:?set AIEN_ARCH_DIR to the aien-architecture checkout}
proof=${AIEN_PROOF_BIN:-aien-proof}
cache=${AIEN_FORMAL_CACHE:-$HOME/.cache/aien-formal}

# shellcheck disable=SC1090
. "$job_file"

mode=${2:-start}

covered_abs="$cache/${COVERED_REPO}/$COVERED_COMMIT/$COVERED_PATH"
out="$cache/out/$JOB.out"

inputs_flags() {
  printf '%s\n' --base "$arch" --invariant "$INVARIANT"
  for m in $MODEL; do printf '%s\n' --model "$m"; done
  printf '%s\n' --proof-source "$ORACLE_DIR" --covered "$covered_abs"
}
# word-safe enough: no path here contains spaces
iflags=$(inputs_flags | tr '\n' ' ')

if [ "$mode" = "exec" ]; then
  # Inside the board job: write the digest line, then the oracle output, keep both.
  mkdir -p "$(dirname "$out")"
  # shellcheck disable=SC2086
  d=$("$proof" formal digest $iflags)
  { echo "inputs_digest: $d"; sh "$arch/$ORACLE_DIR/$RUNNER"; } >"$out" 2>&1 && rc=0 || rc=$?
  cat "$out"
  exit "$rc"
fi

# Fetch the covered file at the pinned commit (once).
if [ ! -s "$covered_abs" ]; then
  mkdir -p "$(dirname "$covered_abs")"
  gh api -H "Accept: application/vnd.github.raw" \
    "repos/$COVERED_REPO/contents/$COVERED_PATH?ref=$COVERED_COMMIT" >"$covered_abs.tmp"
  mv "$covered_abs.tmp" "$covered_abs"
fi

export PATH="$HOME/.elan/bin:$PATH"
toolchain="$(tr -d '\n' <"$arch/$ORACLE_DIR/lean-toolchain") / $(lean --version | head -1) / $(lake --version | head -1)"
theorems=$(sed -n 's/^theorem \([A-Za-z_0-9.]*\).*/\1/p' "$arch/$THEOREM_FILE" | sed 's/^/--theorem /' | tr '\n' ' ')
arch_commit=$(git -C "$arch" rev-parse HEAD)
dirty=""
[ -n "$(git -C "$arch" status --porcelain -- "$INVARIANT" "$ORACLE_DIR")" ] && dirty="--dirty"

# The board job. Inputs are the invariant manifest, the Lean sources, this script, the job
# file, and the covered file. Run from the architecture checkout so inputs hash by relative path.
cd "$arch"
rc=0
"$proof" run --job "$JOB" \
  --input "$INVARIANT" --input "$ORACLE_DIR" \
  --input "$here/formal-job.sh" --input "$job_file" --input "$covered_abs" \
  -- sh "$here/formal-job.sh" "$job_file" exec || rc=$?
[ -s "$out" ] || { echo "no output recorded for $JOB (rc=$rc)"; exit 1; }

# shellcheck disable=SC2086
exec "$proof" formal bind $iflags \
  --oracle-toolchain "$toolchain" $theorems \
  --result-prefix "$RESULT_PREFIX" --output-file "$out" \
  --repo "https://github.com/$COVERED_REPO" --commit "$COVERED_COMMIT" \
  --machine "$(hostname)" --procedure "$JOB" \
  --external-ref "aien-architecture@$arch_commit" $dirty
