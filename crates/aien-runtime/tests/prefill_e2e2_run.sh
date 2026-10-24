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

# Driver for the PREFILL-E2E-2 gates (ADR 0028, run by `aien-test run`).
#
# The forge lines for PREFILL-E2E-2 are shell chains with environment
# variables. A gate manifest has no shell and no environment (ADR 0028 P6), so
# every gate runs this one script from the repository root with a fixed
# argument list. The commands below are the forge lines, unchanged, minus the
# HEAD pin (aien-test pins and records the commit itself).
#
#   prefill_e2e2_run.sh main
#       cargo fmt --all -- --check, then cargo clippy -p aien-runtime
#       --all-targets -- -D warnings, then the ignored real-checkpoint test in
#       release mode. Stops at the first failing stage with that stage's exit
#       status (the forge uses &&).
#   prefill_e2e2_run.sh mutant ID MARKER
#       Copy the tree to a private temp directory, apply the patch
#       prefill_e2e2_mutants/mutant_ID.patch there and run the same test
#       there (the checkout is never patched). Succeed only when the test
#       failed and its output holds MARKER (the forge KILLED rule). A mutant
#       that survives, or fails with another message, exits 1. The copy is
#       built in its own target dir (see run_mutant), never the shared one.
#   prefill_e2e2_run.sh patches
#       Check that every prefill_e2e2_mutants/mutant_*.patch still applies to
#       this tree (git apply --check). No model, no build; run it after changing
#       a file a mutant patch touches (sc #452).
#
# Observations ("AIEN-OBS NAME JSON" lines, see tools/aien-test) tell the
# runner what happened; the verdict is derived from the manifest, not here.
#
# Environment (defaults are the paths the forge lines use):
#   AIEN_E2E_CHECKPOINT  TinyLlama-1.1B-Chat-v1.0 directory
#   AIEN_E2E2_OUT_DIR    receipts and mutant logs; must be outside the repository
set -u

here=$(cd "$(dirname "$0")" && pwd) || exit 3
cd "$here/../../.." || exit 3

ckpt=${AIEN_E2E_CHECKPOINT:-$HOME/models/TinyLlama-1.1B-Chat-v1.0}
out=${AIEN_E2E2_OUT_DIR:-$HOME/workspace/test-queue-logs}
sha8=$(git rev-parse HEAD | cut -c1-8)
# The sandbox HOME is read-only: receipts then go to a fresh directory in the sandbox's private /tmp.
if [ -z "${AIEN_E2E2_OUT_DIR:-}" ] && [ ! -w "$HOME" ]; then
	out=$(mktemp -d) || exit 3
fi
# In the user sandbox, aien-run-exec mounts the operator's ~/models read-only at
# /var/lib/aien-run/models, which is $HOME/models there, so the default above finds the checkpoint.
# No checkpoint: NOT_RUN with reason model_missing (exit 77 + not_run_reason, see aien-test
# verdict.rs SELF_NOT_RUN_EXIT), never PASS. The weights' sha256 is recorded either way it runs.
model_check() {
	if [ ! -f "$ckpt/model.safetensors" ] || [ ! -f "$ckpt/tokenizer.json" ]; then
		echo "checkpoint not found: $ckpt (model.safetensors, tokenizer.json)" >&2
		obs not_run_reason '"model_missing"'
		exit 77
	fi
	obs model_sha256 "\"$(sha256sum "$ckpt/model.safetensors" | cut -c1-64)\""
}
mkdir -p "$out" && out=$(cd "$out" && pwd) || exit 3

# The leading newline keeps the line whole when a test left a partial line.
obs() {
	printf '\nAIEN-OBS %s %s\n' "$1" "$2"
}

run_test() {
	AIEN_E2E_CHECKPOINT="$ckpt" AIEN_E2E2_RECEIPT="$1" \
		cargo test -p aien-runtime --release --test prefill_e2e2 -- --ignored --nocapture
}

run_main() {
	model_check
	cargo fmt --all -- --check
	rc=$?
	obs fmt_rc "$rc"
	[ "$rc" -eq 0 ] || exit "$rc"
	cargo clippy -p aien-runtime --all-targets -- -D warnings
	rc=$?
	obs clippy_rc "$rc"
	[ "$rc" -eq 0 ] || exit "$rc"
	run_test "$out/AIENTEST-DOGFOOD-receipt-$sha8.json"
	rc=$?
	obs test_rc "$rc"
	exit "$rc"
}

run_mutant() {
	id=${1:-}
	marker=${2:-}
	case "$id" in
	m[1-8]) ;;
	*)
		echo "bad mutant id: $id" >&2
		exit 2
		;;
	esac
	case "$marker" in
	'' | *[!A-Z_]*)
		echo "bad marker: $marker" >&2
		exit 2
		;;
	esac
	model_check
	patch="crates/aien-runtime/tests/prefill_e2e2_mutants/mutant_$id.patch"
	lg="$out/AIENTEST-DOGFOOD-$sha8-$id.log"
	# The patch goes into a private copy of the tree, never into the checkout: aien-test runs gates in
	# parallel in one checkout, and a patched file there fails their clean-tree checks (and the main leg's
	# fmt check).
	#
	# The copy never builds in the shared target dir, and its files carry the time of the copy, not
	# the checkout's (tar -m). Cargo calls a workspace crate fresh when its library is newer than its
	# sources; it compares file times, not contents, and the library has the same name whichever
	# tree built it. So a library built from the patched copy was later linked as fresh by the main
	# leg on the same target dir, which then ran the mutant (sc #452: PREFILL-E2E-2 failing with
	# M8's RECLAIM_VIOLATION, allocated blocks 4, free 18), and a mutant whose patched file was older
	# than a library built after it linked that one instead and survived. Fresh file times make every
	# workspace crate of the copy rebuild; third-party crates build once in the mutant dir and stay.
	work=$(mktemp -d) || exit 3
	trap 'rm -rf "$work"' EXIT
	trap 'rm -rf "$work"; exit 143' HUP INT TERM
	git ls-files -z --cached --others --exclude-standard --deduplicate |
		tar --null --ignore-failed-read -T - -cf - | tar -xmf - -C "$work" || exit 3
	CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target}/prefill-e2e2-mutants
	export CARGO_TARGET_DIR
	cd "$work" || exit 3
	git apply "$patch" || exit 3
	run_test "$out/AIENTEST-DOGFOOD-receipt-$sha8-$id.json" >"$lg" 2>&1
	rc=$?
	echo "LEG $id rc=$rc"
	grep -n "$marker" "$lg" | head -3
	obs mutant_test_rc "$rc"
	if [ "$rc" -ne 0 ] && grep -q "$marker" "$lg"; then
		obs mutant_killed true
		echo "MUTANT $id KILLED"
		exit 0
	fi
	obs mutant_killed false
	echo "MUTANT $id SURVIVED or wrong failure"
	tail -5 "$lg"
	exit 1
}

check_patches() {
	rc=0
	for p in crates/aien-runtime/tests/prefill_e2e2_mutants/mutant_m*.patch; do
		if git apply --check "$p"; then
			echo "applies: $p"
		else
			echo "STALE: $p no longer applies; regenerate it so it still plants its bug" >&2
			rc=1
		fi
	done
	exit "$rc"
}

case "${1:-}" in
main) run_main ;;
patches) check_patches ;;
mutant)
	shift
	run_mutant "$@"
	;;
*)
	echo "usage: $0 main | mutant ID MARKER | patches" >&2
	exit 2
	;;
esac
