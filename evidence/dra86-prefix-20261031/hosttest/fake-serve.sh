#!/usr/bin/env bash
# dra86 mock door for the host check of window-dra86.sh run (never the real server): takes the server's flags,
# keeps --port, and serves mock86.py with this process's environment (the step's server env).
port=; while [ $# -gt 0 ]; do [ "$1" = --port ] && port=$2; shift; done
echo "fake-serve pid $$ sid $(ps -o sid= -p $$ | tr -d " ") MOCK_M=${MOCK_M:-exact} key_len=${#AIEN_KV_STORE_KEY} lock_fds=$(ls -l /proc/$$/fd | grep -c "$(basename "$DRA86_TEST_LOCK")")" >&2
# The store key the door received goes to a test-only file outside the receipts (runtests.sh greps the receipts for it).
[ -n "${AIEN_KV_STORE_KEY:-}" ] && [ -n "${DRA86_KEYSEEN:-}" ] && echo "$AIEN_KV_STORE_KEY" >> "$DRA86_KEYSEEN"
exec python3 "$(dirname "$0")/mock86.py" "$port"
