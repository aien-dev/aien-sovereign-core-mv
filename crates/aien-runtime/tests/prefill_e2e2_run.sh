#!/bin/sh
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
#       that survives, or fails with another message, exits 1.
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
	# fmt check). Each mutant builds in its own target dir under CARGO_TARGET_DIR: cargo names artifacts
	# by package, not by source path, so a mutant's patched rlibs and the clean tree's rlibs would overwrite
	# each other in one shared dir and a clean run that starts while a mutant copy still exists reuses the
	# mutant's artifacts as fresh (PREFILL-E2E-2 failed 18 of 26 clean runs with the m7 or m3 shape when
	# the mutant gates ran alongside, 2026-10-16 to 10-18; mutants survived the same way). The per-mutant
	# dir persists across heads, so after the first build only the patched crates recompile.
	work=$(mktemp -d) || exit 3
	trap 'rm -rf "$work"' EXIT
	trap 'rm -rf "$work"; exit 143' HUP INT TERM
	git ls-files -z --cached --others --exclude-standard --deduplicate |
		tar --null --ignore-failed-read -T - -cf - | tar -xf - -C "$work" || exit 3
	CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target}/mutants/$id
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

case "${1:-}" in
main) run_main ;;
mutant)
	shift
	run_mutant "$@"
	;;
*)
	echo "usage: $0 main | mutant ID MARKER" >&2
	exit 2
	;;
esac
