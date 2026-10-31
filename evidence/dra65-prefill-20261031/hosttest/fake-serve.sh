#!/usr/bin/env bash
# dra65 mock door for the host check of window-dra65.sh run (never the real server)
port=; while [ $# -gt 0 ]; do [ "$1" = --port ] && port=$2; shift; done
M=exact
if [ "${AIEN_Q36_TEST_MUTANT:-}" = 4 ] || { [ "${AIEN_OMEGA_Q36_PF_MUTANT:-}" = 2 ] && [ "${AIEN_Q36_PREFILL_CHUNK:-128}" != 0 ]; }; then M=mut; fi
[ -n "${MOCK_M:-}" ] && M=$MOCK_M
echo "fake-serve pid $$ sid $(ps -o sid= -p $$ | tr -d " ") M=$M lock_fds=$(ls -l /proc/$$/fd | grep -c "$(basename "$DRA65_TEST_LOCK")") $(grep Cpus_allowed_list /proc/self/status)" >&2
M=$M exec python3 "$(dirname "$0")/mock2.py" "$REPO" "$port"
