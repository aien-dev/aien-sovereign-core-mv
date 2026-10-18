#!/usr/bin/env bash
# test-pkv-lib.sh: host-only self-test of scripts/own30b-pkv-lib.sh (sc#523 items 7 and 8) with
# stub curl, ss and sleep on PATH: no server, no chip, no network, no real port. Proves:
# wait_parks settles only on a number (a null or empty parked_chunks never counts as settled),
# and a port still held at exit makes the EXIT trap fail the run without skipping its cleanup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$ROOT/scripts/own30b-pkv-lib.sh"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0
chk() { if eval "$1"; then echo "  ok   $2"; else echo "  FAIL $2"; fail=1; fi; }

mkdir "$T/bin"
# curl: the live port never answers; any other URL answers with $T/health when it exists.
cat >"$T/bin/curl" <<EOF
#!/bin/sh
case "\$*" in *:18091*) exit 7 ;; esac
[ -f "$T/health" ] || exit 7
cat "$T/health"
EOF
# ss: a listener on the port while $T/listening exists.
cat >"$T/bin/ss" <<EOF
#!/bin/sh
[ -f "$T/listening" ] && echo "LISTEN 0 128 127.0.0.1:18098 0.0.0.0:*"
exit 0
EOF
printf '#!/bin/sh\nexit 0\n' >"$T/bin/sleep"
chmod +x "$T/bin/curl" "$T/bin/ss" "$T/bin/sleep"

# wait_parks against a /health body; prints its exit status.
parks() {
	printf '%s\n' "$1" >"$T/health"
	(PATH="$T/bin:$PATH"; set -euo pipefail; . "$LIB"; wait_parks) >/dev/null 2>&1 && echo 0 || echo $?
}
echo "test-pkv-lib: wait_parks (item 7)"
chk '[ "$(parks "{\"persistent_kv\":{\"parked_chunks\":3}}")" = 0 ]' "a steady number settles"
chk '[ "$(parks "{\"persistent_kv\":{\"parked_chunks\":0}}")" != 0 ]' "zero never settles"
chk '[ "$(parks "{\"persistent_kv\":{\"parked_chunks\":null}}")" != 0 ]' "null never settles"
chk '[ "$(parks "{\"persistent_kv\":{}}")" != 0 ]' "a missing counter never settles"
chk '[ "$(parks "{\"status\":\"ok\"}")" != 0 ]' "persistence absent never settles"
chk '[ "$(parks "")" != 0 ]' "an empty reply never settles"
rm -f "$T/health"

echo "test-pkv-lib: the EXIT trap with the port still held (item 8)"
# pkv_begin with a free port, then something keeps listening after our server is gone.
rc=0
(
	PATH="$T/bin:$PATH"
	set -euo pipefail
	. "$LIB"
	dir="$T/ev" BIN=/bin/true
	pkv_begin
	echo "$work" >"$T/work"
	echo 999999 >"$PIDF" # not an aien-serve-http: stop_server sends no signal
	touch "$T/listening"
) >"$T/trap.out" 2>&1 || rc=$?
chk '[ "$rc" != 0 ]' "the run fails (exit $rc)"
chk 'grep -q "still listens" "$T/trap.out"' "the leftover listener is reported"
chk '[ -s "$T/work" ] && [ ! -e "$(cat "$T/work")" ]' "the trap still removed the private work dir"

rm -f "$T/listening"
rc=0
(
	PATH="$T/bin:$PATH"
	set -euo pipefail
	. "$LIB"
	dir="$T/ev2" BIN=/bin/true
	pkv_begin
	echo "$work" >"$T/work2"
) >/dev/null 2>&1 || rc=$?
chk '[ "$rc" = 0 ]' "a clean stop exits 0 (control)"
chk '[ ! -e "$(cat "$T/work2")" ]' "and removes the work dir"

[ "$fail" = 0 ] && echo "test-pkv-lib: PASS" || { echo "test-pkv-lib: FAIL"; exit 1; }
