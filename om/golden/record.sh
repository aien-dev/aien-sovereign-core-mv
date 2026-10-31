#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# om/golden/record.sh: record greedy golden token ids from an aien-serve-http endpoint.
# Sequencing only: golden (om/golden/golden.om) validates prompts, writes every request body, judges
# every reply and writes the record; curl only moves bytes.
# Usage: OMC=... bash om/golden/record.sh --url http://HOST:PORT --prompts P.json --model-repo ORG/NAME
#   --model-rev SHA40 --max-tokens N --engine-sha256 SHA64 --host-id ID --label REAL|FAKE --out FILE
#   [--header-file F] [--timeout SECONDS]
# GOLDEN=path skips the build and uses that golden binary.
# Exit: 0 written, 2 usage or existing --out, 3 compiler not pinned, 4/5 refused by golden, 6 transport.
set -u
here=$(cd "$(dirname "$0")" && pwd)
url= prompts= repo= rev= maxt= eng= host= label= out= hdr= tmo=600
while [ $# -gt 1 ]; do
  case $1 in
    --url) url=$2 ;; --prompts) prompts=$2 ;; --model-repo) repo=$2 ;; --model-rev) rev=$2 ;;
    --max-tokens) maxt=$2 ;; --engine-sha256) eng=$2 ;; --host-id) host=$2 ;; --label) label=$2 ;;
    --out) out=$2 ;; --header-file) hdr=$2 ;; --timeout) tmo=$2 ;;
    *) echo "record: refused 2 unknown option $1" >&2; exit 2 ;;
  esac
  shift 2
done
[ $# -eq 0 ] || { echo "record: refused 2 option without value" >&2; exit 2; }
for v in url prompts repo rev maxt eng host label out; do
  [ -n "${!v}" ] || { echo "record: refused 2 missing $v" >&2; exit 2; }
done
if [ -e "$out" ] || [ -L "$out" ]; then echo "record: refused 2 $out exists" >&2; exit 2; fi
work=$(mktemp -d "$(dirname "$out")/.golden-record.XXXXXX") || exit 2
trap 'rm -rf "$work"' EXIT
if [ -z "${GOLDEN:-}" ]; then
  bash "$here/build.sh" "$work/golden" || exit $?
  GOLDEN=$work/golden
fi
mkdir "$work/req" "$work/resp"
n=$("$GOLDEN" req "$prompts" "$repo" "$maxt" "$work/req") || exit $?
hargs=()
[ -z "$hdr" ] || hargs=(-H "@$hdr")
for ((i = 0; i < n; i++)); do
  curl --silent --show-error --fail --max-time "$tmo" --noproxy '*' --proto '=http,https' \
    -H 'Content-Type: application/json' -H 'Expect:' "${hargs[@]}" \
    --data-binary "@$work/req/$i.json" -o "$work/resp/$i.json" "$url/v1/chat/completions" \
    || { echo "record: refused 6 transport prompt $i" >&2; exit 6; }
done
"$GOLDEN" rec "$prompts" "$repo" "$rev" "$maxt" "$eng" "$host" "$label" "$work/resp" > "$work/record.json" || exit $?
ln "$work/record.json" "$out" || { echo "record: refused 2 $out exists" >&2; exit 2; }
