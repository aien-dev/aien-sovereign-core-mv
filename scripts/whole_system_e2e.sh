#!/usr/bin/env bash
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

# WHOLE-SYSTEM-E2E harness v1.3 (aien-architecture#190, lane L1; GB10 mode for milestone R6, sovereign-core#402).
# Rows: docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.md, -v1.1.md, -v1.2.md, -v1.3-gb10.md, -v1.3-ctrl-e5.md, -v1.3.1-e3m.md (each committed before the script
# change that implements it; never edited by the script).
# Contract: aien-architecture docs/plans/release-readiness/ACCEPTANCE_E2E.md, FROZEN at commit b2400fe,
# sha256 9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8 (CONTRACT_SHA below).
# v1.2: E2 and CTRL-E2 through `aien-cli objective record` (sc#388); the proposer reads the inbox files and the E3
# receipt lists them (sc#392); memory under an engaged ALLEN (sc#391); CTRL-E6 and CTRL-E4b state-loss controls;
# logs kept as .txt (the repo ignores *.log); the independent verifier (sc#389) runs over the closed chain.
# v1.3: WSE2E_BACKEND=gb10 (default cpu, unchanged) builds from the pinned GPU engine (scripts/pins.sh --gpu), runs
# the zero-CUDA gate, requires the native Omega GB10 backend line on every daemon start, and adds gpu_executions,
# fallback_count, fallback_detected, pins, nvidia_driver and host_load to every receipt (ACCEPTANCE-v1.3-gb10.md).
# v1.3.1: E3m topic words are matched case-insensitively as whole words; E3 file names stay exact
# (ACCEPTANCE-v1.3.1-e3m.md); the gb10 run folder is RUN-3-GB10-<utc>.
# v1.4: WSE2E_ARTIFACT=<sovereign-OS-ARCH.tar.gz> installs the release artifact offline into the run's clean HOME (row E1), runs the whole chain on
# the INSTALLED binaries, and adds the CTRL-E1 refused copies (ACCEPTANCE-v1.4-e1.md). ENGINEERING READINESS only for a dry-run-key install.
# Without the variable nothing changes. Needs the archive's SHA256SUMS.txt and SHA256SUMS.txt.sig beside it; WSE2E_DRY_RUN_SIGNERS names the
# signer file (default: dry-run-signers beside the archive, when present); WSE2E_RELEASE_TAG names the install (default e2e-artifact).
# v1.5: WSE2E_RUN_ROOT=<absolute real path> puts the run state (workspaces, home, state, socket, logs) there instead of
# $CLAUDE_JOB_DIR/tmp/e2e/run, so the workspace path inside the model prompt is the same every attempt; PRE records
# run_root, E3/E3m/E4 record the daemon's prompt token count and prompt id per attempt (ACCEPTANCE-v1.5-fixed-root.md).
# Without the variable nothing changes.
#
#   scripts/whole_system_e2e.sh run        build the CLI, run the chain E2..E6 + controls, write one run folder
#   scripts/whole_system_e2e.sh check F    the document checks of rows E3/E3m on file F (prints failed ids)
#   scripts/whole_system_e2e.sh selftest   checks with positive and negative controls (no daemon)
#
# Shell plus jq. No Python, no systemd, no accelerator: the daemon must print the CPU-reference backend
# line or the run stops. In gb10 mode no CUDA (zero-CUDA gate) and no CPU fallback: the daemon must print
# the native Omega GB10 backend line or the run stops; any fallback marks the row FAIL. Needs CLAUDE_JOB_DIR (build target and run state under $CLAUDE_JOB_DIR/tmp/e2e),
# the prebuilt compose archive at $CLAUDE_JOB_DIR/tmp/librx_compose.a, jq, sha256sum, ss, hostname, iconv.
# Model: WSE2E_MODEL_DIR (default: the CAND-4 Llama 3.2 1B snapshot in the local Hugging Face cache);
# its weights and tokenizer sha256 are compared with release/candidate.toml and written into every receipt.
set -u

# ---- document checks (rows E3, E3m) -----------------------------------------------------------------------
# Prints failed ids, space separated: D1 missing/unreadable, D2 first line not "# ", D3 over 200 words,
# D4 a required name missing. $1 file, $2.. required names, matched exactly and case-sensitively (E3 file names).
# --topics as the first argument: the names are topic words, matched case-insensitively as whole words (E3m, v1.3.1).
D4_TOPIC_RULE="case-insensitive whole word: LC_ALL=C grep -q -i -w (ACCEPTANCE-v1.3.1-e3m.md)"
check_doc() {
  local mode=exact; [ "${1:-}" = --topics ] && { mode=topics; shift; }
  local f=$1; shift; local failed=""
  [ -s "$f" ] || { echo "D1"; return; }
  iconv -f UTF-8 -t UTF-8 "$f" >/dev/null 2>&1 || failed="$failed D1"
  [ "$(head -c 2 "$f")" = "# " ] || failed="$failed D2"
  [ "$(wc -w <"$f")" -le 200 ] || failed="$failed D3"
  local n; for n in "$@"; do
    if [ $mode = topics ]; then LC_ALL=C grep -q -i -w -- "$n" "$f"; else grep -qF -- "$n" "$f"; fi || { failed="$failed D4"; break; }
  done
  echo "${failed# }"
}
# ---- sc#386: ALLEN engagement and the E4 token budget -------------------------------------------------------
# Memory belongs to an engaged ALLEN identity (aien-architecture#190 lane L3), so every daemon start from PRE
# onward carries a host-built fixture subject, the way scripts/allen_e2e_demo.sh step S1 does. CTRL-E3a (no desk
# key, refuses before serving) keeps its own start and carries none.
# E4's second objective appends one line to a document; the small-edit path caps a reply at 48 tokens by default
# (AIEN_COMPOSE_MAX_TOKENS) and the 1B model re-emits the whole report, so row E4 alone raises the cap to a
# documented value (the report is at most 200 words, about 300 tokens, plus the added line). The value is
# recorded in the E4 receipt; CTRL-E4b (the memory-loss control of the same objective) gets the same budget.
# Later restarts (E6, CTRL-E6, CTRL-E4) keep the default.
E4_MAX_TOKENS=400
allen_env() { # subject_path [agent_hex]: the env lines that engage the fixture subject (adopt only the first time)
  echo "AIEN_ALLEN_SUBJECT=$1"
  [ -z "${2:-}" ] || echo "AIEN_ALLEN_ADOPT=$2"
}

# ---- v1.3 (ACCEPTANCE-v1.3-gb10.md): the GB10 backend rule and the receipt counters ---------------------------
# backend_ok MODE LINE: 0 when the daemon's Backend line is the one MODE requires (the stub name does not count).
GB10_BACKEND_MARK='OmegaGb10Backend (native Omega engine, no CUDA'
backend_ok() {
  case $1 in
    gb10) case $2 in *"$GB10_BACKEND_MARK"*) return 0 ;; esac ;;
    *) case $2 in *CPU-reference*) return 0 ;; esac ;;
  esac
  return 1
}
strict_ok() { case $1 in *"strict=true dev_fallback_build=false"*) return 0 ;; esac; return 1; }
# prop_ops PROPOSE_JSON: "attempts_with_ops max_gpu_total|null max_native_fallbacks omega_native(yes|no)" from
# report.proposal_attempts[].ops (sc#337 fallbacks, sc#405 gpu_executions); "0 null 0 no" when absent or unreadable.
prop_ops() {
  jq -r --arg m "$GB10_BACKEND_MARK" '[.report.proposal_attempts[]?.ops // empty] as $o
    | "\($o | length) \([$o[].gpu_executions // empty] | max // "null") \([$o[].native_fallbacks // 0] | max // 0) \(if ($o | length) > 0 and all($o[]; (.backend // "") | contains($m)) then "yes" else "no" end)"' \
    "$1" 2>/dev/null | grep -E '^[0-9]+ ([0-9]+|null) [0-9]+ (yes|no)$' || echo "0 null 0 no"
}
FALLBACK_RE='STRICT_REAL_MODEL_VIOLATION|OMEGA_BACKEND chip error'
fallback_lines() { cat "$@" 2>/dev/null | grep -c -E "$FALLBACK_RE"; true; }   # files..: count of fallback lines
host_load() {
  jq -nc --arg la "$(cat /proc/loadavg 2>/dev/null)" --arg mf "$(awk '/^MemFree:/{print $2}' /proc/meminfo)" \
    --arg ma "$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" '{loadavg:$la, memfree_kb:($mf|tonumber? // null), memavailable_kb:($ma|tonumber? // null)}'
}
# gb10_receipt STEP: sets G_JSON (the v1.3 receipt fields) and G_WHY (G4/G5 problems, empty when none). Reads
# $ART/STEP-propose.json, PROP_DN[STEP] (the daemon that served that propose) and every daemon log under $LOGS.
declare -A PROP_DN=()
GPU_SEEN_DN=""; GPU_SEEN_TOTAL=0; FALLBACK_SEEN=false; G_JSON='{}'; G_WHY=""
gb10_receipt() {
  local p=$ART/$1-propose.json n=0 tot=null nf=0 om=no delta=null since=null fl fc dn=${PROP_DN[$1]:-}
  G_WHY=""
  [ -f "$p" ] && read -r n tot nf om < <(prop_ops "$p")
  if [ "$n" -gt 0 ]; then
    if [ "$tot" = null ]; then G_WHY="$G_WHY model call without gpu_executions;"
    else
      if [ -n "$dn" ] && [ "$dn" = "$GPU_SEEN_DN" ]; then delta=$((tot - GPU_SEEN_TOTAL)); since="previous observation in daemon $dn"
      else delta=$tot; since="daemon start"; fi
      GPU_SEEN_DN=$dn; GPU_SEEN_TOTAL=$tot
      [ "$delta" -ge 1 ] || G_WHY="$G_WHY gpu_executions $delta;"
    fi
    [ "$om" = yes ] || G_WHY="$G_WHY model call not on the native Omega GB10 backend;"
  fi
  fl=$(fallback_lines "$LOGS"/daemon-*.txt); fc=$((nf + fl))
  [ "$fc" -eq 0 ] || { G_WHY="$G_WHY fallback_count $fc;"; FALLBACK_SEEN=true; }
  G_JSON=$(jq -nc --argjson d "$delta" --argjson t "$tot" --arg s "$since" --argjson fc "$fc" --argjson fn "$nf" --argjson fl "$fl" \
    --argjson fd "$FALLBACK_SEEN" --argjson pins "${PINS_JSON:-null}" --arg drv "${NVDRV:-}" --arg zc "${ZC:-}" --arg sl "${STRICT_LINE:-}" \
    --argjson hl "$(host_load)" --arg why "${G_WHY# }" \
    '{backend_mode:"gb10", gpu_executions:$d, gpu_executions_process_total:$t, gpu_executions_since:(if $d == null then null else $s end),
      gpu_executions_source:"report.proposal_attempts[].ops.gpu_executions (sc#405, Omega chip_calls, process total)",
      fallback_count:$fc, fallback_native_ops:$fn, fallback_log_lines:$fl, fallback_detected:$fd, pins:$pins, nvidia_driver:$drv,
      zero_cuda_gate_exit:$zc, daemon_strict_line:$sl, host_load:$hl, gb10_problems:$why}')
}

# ---- v1.5 (ACCEPTANCE-v1.5-fixed-root.md): a fixed run root and the prompt ids ---------------------------------
# run_root_ok PATH: 0 when PATH is absolute, has no '.', '..' or empty component and no trailing '/', is at most
# 100 bytes (the daemon socket lives inside it), and, once it exists, is its own real path (no symlink on the way).
run_root_ok() {
  case $1 in /*) ;; *) return 1 ;; esac
  case $1 in */|*//*|*/./*|*/../*|*/.|*/..) return 1 ;; esac
  [ ${#1} -le 100 ] || return 1
  [ ! -e "$1" ] || [ "$(readlink -f -- "$1")" = "$1" ]
}
# prompt_fields PROPOSE_JSON: per proposal attempt, the prompt token count and token-id digest the daemon reports in
# report.proposal_attempts[]; [] when the file or the attempts are absent. Recorded only, never judged.
prompt_fields() {
  jq -c '[.report.proposal_attempts[]? | {prompt_tokens:(.prompt_tokens // null), prompt_ids_sha256:(.prompt_ids_sha256 // null)}]' \
    "$1" 2>/dev/null | grep -E '^\[' || echo '[]'
}

# ---- v1.4 (ACCEPTANCE-v1.4-e1.md): install from a release artifact (E1) and the refused copies (CTRL-E1) --------
# ENGINEERING READINESS only: a dry-run-key install is never a public PASS. The functions take every input as an
# argument so selftest can run them on a throwaway fixture release with no daemon.
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
OFFLINE_ENV=(http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9
             HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9)
OFFLINE_METHOD="the only URL is file://; http(s)/all proxy variables point at a closed local port, so any network fetch would fail and show in the install log"
# map_sig CLASS: the installer's signature class in the contract's vocabulary (release is written verified).
map_sig() { case $1 in release) echo verified ;; *) echo "$1" ;; esac; }
# run_installer INSTALLER ARCHIVE_DIR HOME SIGNERS TAG LOG: runs install.sh release mode in an empty environment; prints its exit status.
run_installer() {
  local inst=$1 adir=$2 h=$3 sg=$4 tag=$5 log=$6 sgv=()
  [ -f "$sg" ] && sgv=(AIEN_DRY_RUN_SIGNERS="$sg")
  mkdir -p "$h" "$h.tmp"
  env -i PATH="$PATH" HOME="$h" TMPDIR="$h.tmp" LANG=C "${OFFLINE_ENV[@]}" AIEN_RELEASE_TAG="$tag" \
    AIEN_RELEASE_BASE_URL="file://$adir" "${sgv[@]}" bash "$inst" >"$log" 2>&1
  echo $?
}
# files_recheck RELEASE_DIR: "OK TOTAL": every [files] entry of release.toml re-hashed here, independently of the installer.
files_recheck() {
  local d=$1 ok=0 n=0 line f h
  while IFS= read -r line; do
    f=${line#\"}; f=${f%%\"*}; h=${line##* = \"}; h=${h%\"}
    n=$((n + 1))
    case $f in /*|*..*) continue ;; esac
    [ "$(sha256sum "$d/$f" 2>/dev/null | cut -d' ' -f1)" = "$h" ] && ok=$((ok + 1))
  done < <(awk '/^\[.*\]$/ { s = $0; next } s == "[files]" && NF' "$d/release.toml" 2>/dev/null)
  echo "$ok $n"
}
INST_OK=0; INST_WHY=""; INST_JSON='{}'; INST_RC=""
# install_artifact ARCHIVE HOME SIGNERS TAG WORK: the archive's own install.sh, run offline into HOME. Sets INST_OK (1/0),
# INST_WHY (problems), INST_RC, INST_JSON (the E1 receipt fields). The log is $WORK/install-log.txt.
install_artifact() {
  local arc=$1 h=$2 sg=$3 tag=$4 w=$5 adir cfg rec id rel fo ft raw sc bs as pass dry
  INST_OK=0; INST_WHY=""; INST_JSON='{}'; INST_RC=""
  adir=$(cd "$(dirname "$arc")" && pwd); mkdir -p "$w/x"
  tar -xzf "$arc" -C "$w/x" ./install.sh 2>"$w/tar.err" && [ -s "$w/x/install.sh" ] || { INST_WHY="no install.sh in the archive;"; return 1; }
  cp "$w/x/install.sh" "$w/install.sh"
  INST_RC=$(run_installer "$w/install.sh" "$adir" "$h" "$sg" "$tag" "$w/install-log.txt")
  cfg=$h/.config/sovereign; rec=$cfg/install-receipt.json
  [ "$INST_RC" = 0 ] || { INST_WHY="installer exit $INST_RC: $(grep -m1 '^Error:' "$w/install-log.txt" | cut -c1-200);"; return 1; }
  [ -f "$rec" ] || { INST_WHY="no install-receipt.json;"; return 1; }
  id=$(readlink "$cfg/releases/current"); rel=$cfg/releases/$id
  read -r fo ft < <(files_recheck "$rel")
  raw=$(jq -r '.signature_check // "none"' "$rec"); sc=$(map_sig "$raw")
  bs=$(sha256sum "$h/.local/bin/aien" 2>/dev/null | cut -d' ' -f1); as=$(sha256sum "$arc" | cut -d' ' -f1)
  [ "$sc" = verified ] || [ "$sc" = dry-run-key ] || INST_WHY="$INST_WHY signature_check '$raw' is outside the contract set;"
  [ "$(jq -r '.artifact_sha256' "$rec")" = "$as" ] || INST_WHY="$INST_WHY receipt artifact_sha256 differs from the archive sha256;"
  [ "$ft" -gt 0 ] && [ "$fo" = "$ft" ] && [ "$(jq -r '.files_verified' "$rec")" = "$ft" ] || INST_WHY="$INST_WHY files: recheck $fo of $ft, installer says $(jq -r '.files_verified' "$rec");"
  [ -n "$bs" ] && [ "$bs" = "$(jq -r '.cli_sha256' "$rec")" ] && [ "$bs" = "$(jq -r '.daemon_sha256' "$rec")" ] || INST_WHY="$INST_WHY installed bin/aien sha256 is not the receipt cli_sha256 and daemon_sha256;"
  dry=$(jq -c '.dry_run' "$rec"); pass=false; [ "$sc" = verified ] && [ "$dry" = false ] && pass=true
  INST_JSON=$(jq -nc --slurpfile r "$rec" --arg sc "$sc" --arg raw "$raw" --arg bs "$bs" --arg as "$as" --arg fr "$fo of $ft" \
    --arg rs "$(sha256sum "$rec" | cut -d' ' -f1)" --arg is "$(sha256sum "$w/install.sh" | cut -d' ' -f1)" --argjson rc "$INST_RC" \
    --argjson pass "$pass" --arg om "$OFFLINE_METHOD" \
    '$r[0] as $x | {artifact_sha256:$x.artifact_sha256, signature_check:$sc, signer:$x.signer_fingerprint, files_verified:$x.files_verified,
      daemon_sha256:$x.daemon_sha256, cli_sha256:$x.cli_sha256, installer_signature_check:$raw, dry_run:$x.dry_run, candidate:$x.candidate,
      tag:$x.tag, release_id:$x.release_id, install_receipt_sha256:$rs, installer_sha256:$is, install_exit:$rc,
      install_log:"artifacts/install/install-log.txt", files_recheck:$fr, installed_binary:"home/.local/bin/aien", installed_binary_sha256:$bs,
      archive_sha256_recomputed:$as, offline_method:$om, public_pass:$pass,
      readiness_scope:(if $pass then "release artifact installed with the pinned public key" else "ENGINEERING READINESS only, not a public PASS" end)}')
  INST_WHY=${INST_WHY# }; [ -z "$INST_WHY" ] && INST_OK=1
  [ $INST_OK = 1 ]
}
C1_OK=0; C1_JSON='{}'
# ctrl_e1_parts ARCHIVE SIGNERS WORK TAG [INSTALLER]: CTRL-E1 on three scratch copies (a: one archive byte changed, b: one byte of
# SHA256SUMS.txt changed, c: signature removed), each into its own empty home. INSTALLER overrides the archive's install.sh (selftest only).
ctrl_e1_parts() {
  local arc=$1 sg=$2 w=$3 tag=$4 inst=${5:-} adir asset p d h file off bh ah rc line word ni ok parts='[]' mut='{}' all=1 reason=""
  adir=$(cd "$(dirname "$arc")" && pwd); asset=$(basename "$arc"); mkdir -p "$w/x"
  if [ -z "$inst" ]; then
    tar -xzf "$arc" -C "$w/x" ./install.sh 2>/dev/null && [ -s "$w/x/install.sh" ] && inst=$w/x/install.sh || inst=/nonexistent-installer
  fi
  for p in a b c; do
    d=$w/c1-$p/art; h=$w/c1-$p/home; mkdir -p "$d"
    cp "$adir/$asset" "$adir/SHA256SUMS.txt" "$adir/SHA256SUMS.txt.sig" "$d/"
    case $p in a) file=$asset; off=100 ;; b) file=SHA256SUMS.txt; off=0 ;; c) file=SHA256SUMS.txt.sig; off=-1 ;; esac
    bh=""; ah=""
    if [ $off -ge 0 ]; then
      bh=$(od -An -tx1 -j$off -N1 "$d/$file" | tr -d ' \n'); ah=$([ "$bh" = 58 ] && echo 59 || echo 58)
      printf "\\x$ah" | dd of="$d/$file" bs=1 seek=$off conv=notrunc 2>/dev/null
      mut=$(jq -c --arg p "$p" --arg f "$file" --argjson o $off --arg b "$bh" --arg a "$ah" '. + {($p):{file:$f, offset:$o, before_byte_hex:$b, after_byte_hex:$a}}' <<<"$mut")
    else rm -f "$d/$file"; mut=$(jq -c --arg p "$p" --arg f "$file" '. + {($p):{file:$f, removed:true}}' <<<"$mut"); fi
    rc=$(run_installer "$inst" "$d" "$h" "$sg" "$tag" "$w/c1-$p/install-log.txt")
    line=$(grep -m1 '^Error:' "$w/c1-$p/install-log.txt" | cut -c1-240)
    word=$(printf '%s' "$line" | grep -o -i -E 'checksum|signature' | head -1 | tr 'A-Z' 'a-z')
    ni=yes
    [ ! -e "$h/.config/sovereign/releases" ] && [ ! -e "$h/.config/sovereign/installed.toml" ] && [ ! -e "$h/.config/sovereign/install-receipt.json" ] \
      && [ -z "$(ls -A "$h/.local/bin" 2>/dev/null)" ] || ni=no
    ok=0; [ "$rc" != 0 ] && [ -n "$word" ] && [ $ni = yes ] && ok=1
    [ $ok = 1 ] || all=0
    [ $p != a ] || reason=$line
    parts=$(jq -c --arg p "$p" --argjson rc "$rc" --arg l "$line" --arg w "$word" --arg ni "$ni" --argjson ok "$ok" \
      '. + [{part:$p, installer_exit:$rc, refusal_line:$l, names:$w, nothing_installed:($ni == "yes"), refused_correctly:($ok == 1)}]' <<<"$parts")
  done
  C1_JSON=$(jq -nc --argjson all "$all" --arg r "$reason" --argjson parts "$parts" --argjson mut "$mut" \
    '{install_refused:($all == 1), refusal_reason:$r, nothing_installed:($parts | all(.nothing_installed)), mutation:$mut, parts:$parts,
      parts_described:"a: one byte of the archive changed; b: one byte of SHA256SUMS.txt changed; c: SHA256SUMS.txt.sig removed"}')
  C1_OK=$all
  [ $C1_OK = 1 ]
}

selftest() {
  local t; t=$(mktemp -d); local rc=0
  printf '# Report\n\nabout alpha.txt, beta.txt and gamma.txt\n' >"$t/good.md"
  printf 'Report\n\nabout alpha.txt only\n' >"$t/bad.md"
  [ -z "$(check_doc "$t/good.md" alpha.txt beta.txt gamma.txt)" ] || { echo "selftest: good document failed: $(check_doc "$t/good.md" alpha.txt beta.txt gamma.txt)"; rc=1; }
  [ "$(check_doc "$t/bad.md" alpha.txt beta.txt gamma.txt)" = "D2 D4" ] || { echo "selftest: bad document gave '$(check_doc "$t/bad.md" alpha.txt beta.txt gamma.txt)', want 'D2 D4'"; rc=1; }
  [ "$(check_doc "$t/absent.md")" = "D1" ] || { echo "selftest: absent file not D1"; rc=1; }
  # v1.3.1 (ACCEPTANCE-v1.3.1-e3m.md): E3m topic words, case-insensitive whole words; E3 file names stay exact.
  printf '# Harbour, Orchard, and Windmill\n\nThe harbour is busy. The orchard is quiet.\n' >"$t/topic-heading.md"
  printf '# Report\n\nThe harbour, the orchard and two windmills.\n' >"$t/topic-plural.md"
  printf '# Report\n\nThe harbour and the orchard.\n' >"$t/topic-absent.md"
  printf '# Report\n\nThe HARBOUR, the orchard, and the windmill.\n' >"$t/topic-punct.md"
  printf '# Report\n\nabout Alpha.txt, beta.txt and gamma.txt\n' >"$t/name-case.md"
  [ -z "$(check_doc --topics "$t/topic-heading.md" harbour orchard windmill)" ] || { echo "selftest: topic named only in a capitalised heading was not accepted"; rc=1; }
  [ -z "$(check_doc --topics "$t/topic-punct.md" harbour orchard windmill)" ] || { echo "selftest: upper-case or punctuated topic words were not accepted"; rc=1; }
  [ "$(check_doc --topics "$t/topic-plural.md" harbour orchard windmill)" = "D4" ] || { echo "selftest: plural 'windmills' was accepted as 'windmill'"; rc=1; }
  [ "$(check_doc --topics "$t/topic-absent.md" harbour orchard windmill)" = "D4" ] || { echo "selftest: absent topic word was accepted"; rc=1; }
  [ "$(check_doc "$t/name-case.md" alpha.txt beta.txt gamma.txt)" = "D4" ] || { echo "selftest: E3 file name with changed case was accepted"; rc=1; }
  [ "$(check_doc "$t/topic-heading.md" harbour orchard windmill)" = "D4" ] || { echo "selftest: the exact rule no longer rejects 'Windmill' for 'windmill' (E3 rule changed)"; rc=1; }
  # sc#386: the ALLEN engagement lines and the E4 token budget are fixed, documented values.
  [ "$(allen_env /x/subject.bin abc123 | tr '\n' ' ')" = "AIEN_ALLEN_SUBJECT=/x/subject.bin AIEN_ALLEN_ADOPT=abc123 " ] || { echo "selftest: allen_env adopt lines wrong"; rc=1; }
  [ "$(allen_env /x/subject.bin | tr '\n' ' ')" = "AIEN_ALLEN_SUBJECT=/x/subject.bin " ] || { echo "selftest: allen_env without adopt wrong"; rc=1; }
  case ${E4_MAX_TOKENS:-} in ''|*[!0-9]*) echo "selftest: E4_MAX_TOKENS not a number"; rc=1 ;; *) { [ "$E4_MAX_TOKENS" -ge 48 ] && [ "$E4_MAX_TOKENS" -le 4096 ]; } || { echo "selftest: E4_MAX_TOKENS outside 48..4096"; rc=1; } ;; esac
  # The budget is passed on the E4 daemon start only.
  [ "$(grep -c 'AIEN_COMPOSE_MAX_TOKENS="\?\$E4_MAX_TOKENS' "$0")" = 2 ] || { echo "selftest: AIEN_COMPOSE_MAX_TOKENS must be set on exactly two starts (E4 and CTRL-E4b)"; rc=1; }
  # v1.3 (gb10 mode): backend and strict lines, the propose-report counters, fallback lines, per-row deltas.
  local og='NativeTransformerBackend/OmegaGb10Backend (native Omega engine, no CUDA, NVIDIA GB10 sm_121)'
  local oc='NativeTransformerBackend/CPU-reference (Omega GPU engine not linked, explicit fallback)'
  backend_ok gb10 "$og" || { echo "selftest: gb10 refused the native Omega line"; rc=1; }
  ! backend_ok gb10 "$oc" || { echo "selftest: gb10 accepted the CPU line"; rc=1; }
  ! backend_ok gb10 "NativeTransformerBackend/OmegaGb10Backend (stub, no chip)" || { echo "selftest: gb10 accepted the stub"; rc=1; }
  backend_ok cpu "$oc" && ! backend_ok cpu "$og" || { echo "selftest: cpu backend rule changed"; rc=1; }
  strict_ok "STRICT strict=true dev_fallback_build=false require_checkpoint=true backend=omega-gb10" \
    && ! strict_ok "STRICT strict=false dev_fallback_build=true require_checkpoint=false backend=ReferenceCpuBackend" || { echo "selftest: strict line rule wrong"; rc=1; }
  local ob='OmegaGb10Backend (native Omega engine, no CUDA, NVIDIA GB10 sm_121)'
  jq -nc --arg b "$ob" '{report:{proposal_attempts:[{ops:{backend:$b,native_fallbacks:0,gpu_executions:40}},{ops:{backend:$b,native_fallbacks:0,gpu_executions:97}}]}}' >"$t/p1.json"
  jq -nc '{report:{proposal_attempts:[{ops:{backend:"ReferenceCpuBackend",native_fallbacks:0,reference_runs:0}}]}}' >"$t/p2.json"
  jq -nc '{report:{committed:false}}' >"$t/p3.json"
  jq -nc --arg b "$ob" '{report:{proposal_attempts:[{ops:{backend:$b,native_fallbacks:2,gpu_executions:5}}]}}' >"$t/p4.json"
  jq -nc '{report:{proposal_attempts:[{ops:{backend:"OmegaGb10Backend (stub, no chip)",native_fallbacks:0}}]}}' >"$t/p5.json"
  [ "$(prop_ops "$t/p1.json")" = "2 97 0 yes" ] || { echo "selftest: prop_ops p1 gave '$(prop_ops "$t/p1.json")'"; rc=1; }
  [ "$(prop_ops "$t/p2.json")" = "1 null 0 no" ] || { echo "selftest: prop_ops p2 gave '$(prop_ops "$t/p2.json")'"; rc=1; }
  [ "$(prop_ops "$t/p3.json")" = "0 null 0 no" ] || { echo "selftest: prop_ops p3 gave '$(prop_ops "$t/p3.json")'"; rc=1; }
  [ "$(prop_ops "$t/p4.json")" = "1 5 2 yes" ] || { echo "selftest: prop_ops p4 gave '$(prop_ops "$t/p4.json")'"; rc=1; }
  [ "$(prop_ops "$t/p5.json")" = "1 null 0 no" ] || { echo "selftest: prop_ops p5 gave '$(prop_ops "$t/p5.json")'"; rc=1; }
  [ "$(prop_ops "$t/absent.json")" = "0 null 0 no" ] || { echo "selftest: prop_ops on a missing file"; rc=1; }
  mkdir -p "$t/logs"; printf 'ok\n' >"$t/logs/daemon-1.txt"
  [ "$(fallback_lines "$t/logs"/daemon-*.txt)" = 0 ] || { echo "selftest: clean log counted"; rc=1; }
  # per-row deltas and the strict fallback rule, on the fixtures above (state saved and restored)
  local sv_art=${ART:-} sv_logs=${LOGS:-}; ART=$t; LOGS=$t/logs
  cp "$t/p1.json" "$t/RA-propose.json"; jq -c '.report.proposal_attempts[1].ops.gpu_executions = 130' "$t/p1.json" >"$t/RB-propose.json"
  jq -nc --arg b "$ob" '{report:{proposal_attempts:[{ops:{backend:$b,native_fallbacks:0,gpu_executions:12}}]}}' >"$t/RC-propose.json"
  PROP_DN=([RA]=2 [RB]=2 [RC]=3)
  gb10_receipt RA; [ "$(jq -c '[.gpu_executions,.gpu_executions_since,.fallback_count]' <<<"$G_JSON")" = '[97,"daemon start",0]' ] && [ -z "$G_WHY" ] || { echo "selftest: RA $G_JSON / $G_WHY"; rc=1; }
  gb10_receipt RB; [ "$(jq -c '.gpu_executions' <<<"$G_JSON")" = 33 ] && [ -z "$G_WHY" ] || { echo "selftest: RB delta $G_JSON"; rc=1; }
  gb10_receipt RC; [ "$(jq -c '.gpu_executions' <<<"$G_JSON")" = 12 ] || { echo "selftest: RC new daemon $G_JSON"; rc=1; }
  gb10_receipt RC; [ "$(jq -c .gpu_executions <<<"$G_JSON")" = 0 ] && [ -n "$G_WHY" ] || { echo "selftest: a model row with no new chip call was not flagged"; rc=1; }
  gb10_receipt NONE; [ "$(jq -c '[.gpu_executions,.fallback_detected]' <<<"$G_JSON")" = '[null,false]' ] && [ -z "$G_WHY" ] || { echo "selftest: no-model row $G_JSON / $G_WHY"; rc=1; }
  printf 'x\nOMEGA_BACKEND chip error: boom\nthread panicked: STRICT_REAL_MODEL_VIOLATION: y\n' >"$t/logs/daemon-2.txt"
  [ "$(fallback_lines "$t/logs"/daemon-*.txt)" = 2 ] || { echo "selftest: fallback lines not counted"; rc=1; }
  gb10_receipt NONE; [ "$(jq -c '[.fallback_count,.fallback_detected]' <<<"$G_JSON")" = '[2,true]' ] && [ -n "$G_WHY" ] || { echo "selftest: fallback not flagged $G_JSON"; rc=1; }
  ART=$sv_art; LOGS=$sv_logs; PROP_DN=(); GPU_SEEN_DN=""; GPU_SEEN_TOTAL=0; FALLBACK_SEEN=false
  # v1.5 (ACCEPTANCE-v1.5-fixed-root.md): run-root rule and the recorded prompt ids, with negative controls.
  mkdir -p "$t/root/real"; ln -s "$t/root/real" "$t/root/link"; local rt; rt=$(readlink -f -- "$t/root")
  run_root_ok /home/alice/.aien-e2e/run && run_root_ok "$rt/real" && run_root_ok "$rt/new" || { echo "selftest: a valid run root was refused"; rc=1; }
  for bad in relative/run /a/b/ /a//b /a/./b /a/../b /a/.. "$rt/link" "/$(printf 'x%.0s' $(seq 1 100))"; do
    ! run_root_ok "$bad" || { echo "selftest: run root '$bad' was accepted"; rc=1; }
  done
  jq -nc '{report:{proposal_attempts:[{prompt_tokens:347,prompt_ids_sha256:"6847"},{prompt_tokens:350,prompt_ids_sha256:"ab"}]}}' >"$t/p6.json"
  [ "$(prompt_fields "$t/p6.json")" = '[{"prompt_tokens":347,"prompt_ids_sha256":"6847"},{"prompt_tokens":350,"prompt_ids_sha256":"ab"}]' ] || { echo "selftest: prompt_fields p6 gave '$(prompt_fields "$t/p6.json")'"; rc=1; }
  [ "$(prompt_fields "$t/p5.json")" = '[{"prompt_tokens":null,"prompt_ids_sha256":null}]' ] || { echo "selftest: prompt_fields without ids gave '$(prompt_fields "$t/p5.json")'"; rc=1; }
  [ "$(prompt_fields "$t/p3.json")" = '[]' ] && [ "$(prompt_fields "$t/absent.json")" = '[]' ] || { echo "selftest: prompt_fields on no attempts or no file"; rc=1; }
  # v1.4 (ACCEPTANCE-v1.4-e1.md): E1 and CTRL-E1 on a throwaway signed fixture release (no daemon, no network).
  [ "$(map_sig release)" = verified ] && [ "$(map_sig dry-run-key)" = dry-run-key ] && [ "$(map_sig internal-test-key)" = internal-test-key ] || { echo "selftest: map_sig wrong"; rc=1; }
  if command -v ssh-keygen >/dev/null 2>&1; then
    local fx=$t/fx os arch asset repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    case "$(uname -s)" in Darwin) os=macos ;; *) os=linux ;; esac
    case "$(uname -m)" in x86_64|amd64) arch=x86_64 ;; *) arch=aarch64 ;; esac
    [ "$os-$arch" != macos-aarch64 ] || arch=arm64
    asset=sovereign-$os-$arch.tar.gz
    mkdir -p "$fx/art" "$fx/pkg/bin"
    ssh-keygen -q -t ed25519 -N '' -C test -f "$fx/key"; ssh-keygen -q -t ed25519 -N '' -C other -f "$fx/other"
    printf 'aien-release %s\n' "$(cut -d' ' -f1,2 "$fx/key.pub")" >"$fx/signers"
    printf 'aien-release %s\n' "$(cut -d' ' -f1,2 "$fx/other.pub")" >"$fx/wrong-signers"
    printf '#!/bin/sh\necho fixture aien\n' >"$fx/pkg/bin/aien"; chmod 755 "$fx/pkg/bin/aien"; cp "$repo/install.sh" "$fx/pkg/install.sh"
    { printf 'schema = "AienReleaseV1"\ncandidate = "NOT-RELEASABLE"\naien-cli-sha256 = "%s"\ndry-run = true\n\n[files]\n' "$(sha_of "$fx/pkg/bin/aien")"
      for f in bin/aien install.sh; do printf '"%s" = "%s"\n' "$f" "$(sha_of "$fx/pkg/$f")"; done; } >"$fx/pkg/release.toml"
    tar -czf "$fx/art/$asset" -C "$fx/pkg" . && ( cd "$fx/art" && sha256sum "$asset" >SHA256SUMS.txt ) \
      && ssh-keygen -q -Y sign -f "$fx/key" -n aien-release "$fx/art/SHA256SUMS.txt" && [ -s "$fx/art/SHA256SUMS.txt.sig" ] || { echo "selftest: could not build the fixture release"; rc=1; }
    mkdir -p "$fx/w1"
    if install_artifact "$fx/art/$asset" "$fx/home1" "$fx/signers" fixture "$fx/w1"; then
      [ "$(jq -r '[.signature_check, .files_recheck, .public_pass, .dry_run, (.readiness_scope | startswith("ENGINEERING READINESS only"))] | map(tostring) | join(",")' <<<"$INST_JSON")" = "dry-run-key,2 of 2,false,true,true" ] \
        || { echo "selftest: E1 fields wrong: $INST_JSON"; rc=1; }
      [ "$(jq -r '.artifact_sha256' <<<"$INST_JSON")" = "$(sha_of "$fx/art/$asset")" ] || { echo "selftest: E1 artifact_sha256 is not the archive's"; rc=1; }
      # negative control: a file changed after the install is seen by the independent recheck
      chmod u+w "$fx/home1/.config/sovereign/releases/current/bin/aien"; echo x >>"$fx/home1/.config/sovereign/releases/current/bin/aien"
      [ "$(files_recheck "$fx/home1/.config/sovereign/releases/current")" = "1 2" ] || { echo "selftest: files_recheck did not see a changed installed file"; rc=1; }
    else echo "selftest: fixture install failed: $INST_WHY"; rc=1; fi
    mkdir -p "$fx/w2"   # negative control: a signer that did not sign it installs nothing and the row is not OK
    ! install_artifact "$fx/art/$asset" "$fx/home2" "$fx/wrong-signers" fixture "$fx/w2" && [ "$INST_OK" = 0 ] && [ ! -e "$fx/home2/.local/bin/aien" ] \
      || { echo "selftest: an install with the wrong signer was accepted"; rc=1; }
    mkdir -p "$fx/w3"
    if ctrl_e1_parts "$fx/art/$asset" "$fx/signers" "$fx/w3" fixture; then
      [ "$(jq -r '[.install_refused, .nothing_installed, (.refusal_reason | test("checksum|signature"; "i")), (.parts | length)] | map(tostring) | join(",")' <<<"$C1_JSON")" = "true,true,true,3" ] \
        || { echo "selftest: CTRL-E1 fields wrong: $C1_JSON"; rc=1; }
      [ "$(jq -r '[.parts[].names] | join(",")' <<<"$C1_JSON")" = "checksum,signature,signature" ] || { echo "selftest: CTRL-E1 refusal words: $(jq -c '[.parts[].names]' <<<"$C1_JSON")"; rc=1; }
    else echo "selftest: CTRL-E1 on the fixture did not pass: $C1_JSON"; rc=1; fi
    # negative control for the control: an installer that accepts everything must make CTRL-E1 FAIL
    printf '#!/bin/sh\nmkdir -p "$HOME/.config/sovereign/releases" "$HOME/.local/bin"; touch "$HOME/.local/bin/aien"; exit 0\n' >"$fx/accept-all.sh"
    mkdir -p "$fx/w4"
    ! ctrl_e1_parts "$fx/art/$asset" "$fx/signers" "$fx/w4" fixture "$fx/accept-all.sh" && [ "$(jq -r '.install_refused' <<<"$C1_JSON")" = false ] \
      || { echo "selftest: CTRL-E1 passed against an installer that installs mutated artifacts"; rc=1; }
  else echo "selftest: ssh-keygen is required for the install controls"; rc=1; fi
  rm -rf "$t"; [ $rc -eq 0 ] && echo "selftest PASS"; return $rc
}
case ${1:-} in
  check) shift; check_doc "$@"; exit 0 ;;
  selftest) selftest; exit $? ;;
  run) ;;
  *) echo "usage: $0 run | check FILE [NAME..] | selftest" >&2; exit 2 ;;
esac

# ---- setup -----------------------------------------------------------------------------------------------
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOB=${CLAUDE_JOB_DIR:?CLAUDE_JOB_DIR must be set}
LIB=$JOB/tmp/librx_compose.a
D=$JOB/tmp/e2e
MODE=${WSE2E_BACKEND:-cpu}   # v1.3: cpu (default, v1.2 behaviour) or gb10 (ACCEPTANCE-v1.3-gb10.md)
case $MODE in cpu|gb10) ;; *) echo "WSE2E_BACKEND must be cpu or gb10, not '$MODE'" >&2; exit 2 ;; esac
ARTIFACT=${WSE2E_ARTIFACT:-}   # v1.4: install from a release artifact instead of building the system under test
RUN_ROOT=${WSE2E_RUN_ROOT:-}   # v1.5: a fixed run root (ACCEPTANCE-v1.5-fixed-root.md); empty keeps $D/run
if [ -n "$RUN_ROOT" ]; then
  run_root_ok "$RUN_ROOT" && mkdir -p -- "$RUN_ROOT" && run_root_ok "$RUN_ROOT" \
    || { echo "WSE2E_RUN_ROOT '$RUN_ROOT' must be an absolute real path (no symlink, '.', '..', '//' or trailing '/'), at most 100 bytes" >&2; exit 2; }
fi
TGT=$D/target; RUN_NAME0=RUN-1
[ "$MODE" = gb10 ] && { TGT=$D/target-gb10; RUN_NAME0=RUN-3-GB10; }
[ -n "$ARTIFACT" ] && RUN_NAME0=RUN-4-INSTALL
BIN=$TGT/release/aien-cli
MDIR=${WSE2E_MODEL_DIR:-$(ls -d "$HOME"/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/ 2>/dev/null | head -1)}
UTC0=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${WSE2E_OUT:-$REPO/docs/campaigns/whole-system-e2e/runs/${WSE2E_RUN_NAME:-$RUN_NAME0}-$UTC0}
CONTRACT_SHA=9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8
VER_SRC=$REPO/tools/aien-verify/src/main.rs; VER=$D/aien-verify
for t in jq sha256sum ss hostname iconv; do command -v $t >/dev/null || { echo "need $t" >&2; exit 2; }; done
{ [ "$MODE" = gb10 ] && [ -z "$ARTIFACT" ]; } || [ -f "$LIB" ] || { echo "missing $LIB" >&2; exit 2; }   # gb10 builds the composition from its pin; artifact mode still builds the test-only ALLEN fixture subject from the CPU archive
[ -n "$MDIR" ] && [ -f "$MDIR/model.safetensors" ] && [ -f "$MDIR/tokenizer.json" ] || { echo "model dir '$MDIR' lacks model.safetensors or tokenizer.json" >&2; exit 2; }
QUIET=$HOME/workspace/.spark-quiet
while [ -s "$QUIET" ]; do echo "quiet flag set; waiting"; sleep 30; done
DIRTY=$(git -C "$REPO" status --porcelain -- . ":!docs/campaigns/whole-system-e2e/runs" | wc -l)
[ "$DIRTY" -eq 0 ] || [ -n "${WSE2E_ALLOW_DIRTY:-}" ] || { echo "working tree has $DIRTY uncommitted changes; commit them first" >&2; exit 2; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
mkdir -p "$D"
GPUX=(); PINS=""; PINS_JSON=null; NVDRV=""; ZC=""; STRICT_LINE=""
if [ -n "$ARTIFACT" ]; then
  # v1.4 (ACCEPTANCE-v1.4-e1.md): no build of the system under test; the installer provides the CLI and the daemon (installed after RUN is made).
  ADIR=$(cd "$(dirname "$ARTIFACT")" 2>/dev/null && pwd); AASSET=$(basename "$ARTIFACT")
  for f in "$AASSET" SHA256SUMS.txt SHA256SUMS.txt.sig; do [ -s "$ADIR/$f" ] || { echo "artifact mode: missing $ADIR/$f (the archive and its SHA256SUMS.txt and SHA256SUMS.txt.sig)" >&2; exit 2; }; done
  RELTAG=${WSE2E_RELEASE_TAG:-e2e-artifact}
  SIGNERS=${WSE2E_DRY_RUN_SIGNERS:-$ADIR/dry-run-signers}
  PINS_JSON=null
  if [ "$MODE" = gb10 ]; then   # the GB10 rules G2 to G5 apply to the installed daemon (the release build links the native engine)
    NVDRV=$(head -1 /proc/driver/nvidia/version 2>/dev/null | tr -s ' ' | cut -c1-200)
    [ -n "$NVDRV" ] || { echo "no /proc/driver/nvidia/version: no NVIDIA driver loaded" >&2; exit 2; }
    GPUX=(env -u AIEN_DEV_FALLBACK -u AIEN_OMEGA_CTA_BUDGET -u AIEN_OMEGA_SPIN_US AIEN_GPU_BACKEND=omega AIEN_REQUIRE_BLACKWELL=1)
  fi
elif [ "$MODE" = gb10 ]; then
  # G1: the pinned GPU engine, as the sc#94 release golden path (sc#92, sc#375); env.sh sets AIEN_OMEGA_DIR and
  # AIEN_REQUIRE_NATIVE_LIBS=1 (a stub engine is a build error) and unsets AIEN_DEV_FALLBACK and the archive overrides.
  PINS=$D/pins
  bash "$REPO/scripts/pins.sh" --gpu "$PINS" >"$D/pins.txt" 2>&1 || { echo "pins.sh --gpu failed, see $D/pins.txt" >&2; exit 2; }
  ( cd "$REPO" && . "$PINS/env.sh" && CARGO_TARGET_DIR="$TGT" cargo build --release -p aien-cli ) >"$D/build.log" 2>&1 \
    || { echo "build failed, see $D/build.log" >&2; exit 2; }
  grep -q -E "building the (CPU )?stub|developer override" "$D/build.log" && { echo "stub or overridden engine build; refusing" >&2; exit 2; }
  bash "$REPO/scripts/zero-cuda-gate.sh" "$BIN" >"$D/zero-cuda-gate.txt" 2>&1; ZC=$?
  [ "$ZC" = 0 ] || { echo "zero-CUDA gate exit $ZC, see $D/zero-cuda-gate.txt; refusing" >&2; exit 2; }
  pin_of() { git -C "$PINS/$1" rev-parse HEAD 2>/dev/null; }
  lock_of() { tr -d '[:space:]' <"$1" 2>/dev/null; }
  P_OM=$(pin_of omega); P_PH=$(pin_of physics); P_AO=$(pin_of aienos)
  { [ -n "$P_OM" ] && [ "$P_OM" = "$(lock_of "$REPO/omega.lock")" ] && [ "$P_PH" = "$(lock_of "$PINS/omega/physics.lock")" ] \
    && [ "$P_AO" = "$(lock_of "$PINS/omega/aienos.lock")" ]; } || { echo "pinned checkouts differ from their lock files" >&2; exit 2; }
  PINS_JSON=$(jq -nc --arg o "$P_OM" --arg p "$P_PH" --arg a "$P_AO" '{omega:$o, physics:$p, aienos:$a, source:"scripts/pins.sh --gpu: omega.lock, then physics.lock and aienos.lock of that omega"}')
  NVDRV=$(head -1 /proc/driver/nvidia/version 2>/dev/null | tr -s ' ' | cut -c1-200)
  [ -n "$NVDRV" ] || { echo "no /proc/driver/nvidia/version: no NVIDIA driver loaded" >&2; exit 2; }
  # G2: every daemon start: a missing engine is fatal (never a CPU fallback); launch tuning left at the daemon defaults.
  GPUX=(env -u AIEN_DEV_FALLBACK -u AIEN_OMEGA_CTA_BUDGET -u AIEN_OMEGA_SPIN_US AIEN_GPU_BACKEND=omega AIEN_REQUIRE_BLACKWELL=1)
else
( cd "$REPO" && env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_FORCE_CPU_STUB \
    AIEN_OMEGA_COMPOSE_LIB="$LIB" CARGO_TARGET_DIR="$D/target" cargo build --release -p aien-cli ) >"$D/build.log" 2>&1 \
  || { echo "build failed, see $D/build.log" >&2; exit 2; }
grep -q "building the stub" "$D/build.log" && { echo "stub compose build; refusing" >&2; exit 2; }
fi
[ -f "$VER_SRC" ] || { echo "missing $VER_SRC (the verifier, sc#389)" >&2; exit 2; }
sh "$REPO/tools/aien-verify/build.sh" "$VER" >"$D/verifier-build.log" 2>&1 || { echo "verifier build failed, see $D/verifier-build.log" >&2; exit 2; }
VER_SHA=$(sha256sum "$VER" | cut -d' ' -f1); VER_SRC_SHA=$(sha256sum "$VER_SRC" | cut -d' ' -f1)
BIN_SHA=""; [ -n "$ARTIFACT" ] || BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)   # artifact mode: the installed file, after the install
W_SHA=$(sha256sum "$(readlink -f "$MDIR/model.safetensors")" | cut -d' ' -f1)
T_SHA=$(sha256sum "$(readlink -f "$MDIR/tokenizer.json")" | cut -d' ' -f1)
C_W=$(sed -n 's/^model-safetensors-sha256 = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
C_T=$(sed -n 's/^tokenizer-json-sha256 = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
C_ID=$(sed -n 's/^model-id = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
HOST=$(hostname)

if [ -n "$RUN_ROOT" ]; then   # v1.5: the same path every attempt; an empty root is reused, an earlier run is moved aside
  RUN=$RUN_ROOT; rmdir -- "$RUN" 2>/dev/null || mv -- "$RUN" "$RUN.old-$(date +%s)"
else
RUN=$D/run; [ -e "$RUN" ] && mv "$RUN" "$RUN.old-$(date +%s)"
fi
mkdir -p "$RUN"/{state,compose,prov,home,logs,ws-contract/inbox,ws-contract/outbox,ws/inbox,ws/outbox} "$OUT/chain" "$OUT/artifacts"
SOCK=$RUN/s; LOGS=$RUN/logs; ART=$OUT/artifacts; WS=$RUN/ws; WSC=$RUN/ws-contract
E=(env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_REQUIRE_BLACKWELL
   -u AIEN_GPU_BACKEND -u AIEN_MODEL_DIR -u AIEN_MODEL_PATH -u AIEN_TOKENIZER_PATH -u AIEN_FORCE_CPU_STUB
   -u AIEN_FAULT_HOLD -u AIEN_FAULT_HOLD_FILE -u AIEN_ALLEN_SUBJECT -u AIEN_ALLEN_ADOPT
   -u AIEN_COMPOSE_MAX_TOKENS -u AIEN_COMPOSE_DOC_MAX_TOKENS
   HOME="$RUN/home" NO_COLOR=1 AIEN_RUNTIME_SOCK="$SOCK" AIEN_RUNTIME_STATE_DIR="$RUN/state"
   AIEN_COMPOSE_DIR="$RUN/compose" AIEN_PROVENANCE_DIR="$RUN/prov" AIEN_REQUIRE_CHECKPOINT=1
   AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=1 AIEN_COMPOSE_EDIT_BUDGET_MS=599000 AIEN_COMPOSE_DOC_BUDGET_MS=599000)
cli() { "${E[@]}" "$BIN" "$@"; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
tstat() { stat -c '%i %s %Y' "$1" 2>/dev/null || echo absent; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---- v1.4 (ACCEPTANCE-v1.4-e1.md): E1 install from the release artifact, before any daemon ------------------------
# The installed binary replaces the built one for every later step; its sha256 is the daemon_sha256 and cli_sha256 of every receipt.
RCPT_EXTRA='{}'; E1_ST=FAIL; E1_EV=""; E1_FIELDS='{}'; ZC_ART=""
if [ -n "$ARTIFACT" ]; then
  mkdir -p "$ART/install"
  install_artifact "$ARTIFACT" "$RUN/home" "$SIGNERS" "$RELTAG" "$RUN/install"; E1_INST_OK=$INST_OK
  cp "$RUN/install/install-log.txt" "$ART/install/" 2>/dev/null
  for f in SHA256SUMS.txt SHA256SUMS.txt.sig dry-run-signers DRY-RUN-NOT-A-RELEASE.txt "DRY-RUN-STATUS-"*.txt aien-cli-digests.txt; do
    [ -f "$ADIR/$f" ] && cp "$ADIR/$f" "$ART/install/"; done
  for f in install-receipt.json installed.toml; do cp "$RUN/home/.config/sovereign/$f" "$ART/install/" 2>/dev/null; done
  cp "$RUN/home/.config/sovereign/releases/current/release.toml" "$ART/install/release.toml" 2>/dev/null
  BIN=$RUN/home/.local/bin/aien
  BIN_SHA=$(sha256sum "$BIN" 2>/dev/null | cut -d' ' -f1)
  E1_FIELDS=$INST_JSON; E1_WHY=$INST_WHY
  if [ "$MODE" = gb10 ] && [ -f "$BIN" ]; then   # the zero-CUDA gate on the installed binary
    bash "$REPO/scripts/zero-cuda-gate.sh" "$(readlink -f "$BIN")" >"$ART/zero-cuda-gate.txt" 2>&1; ZC=$?
    [ "$ZC" = 0 ] || E1_WHY="$E1_WHY zero-CUDA gate exit $ZC on the installed binary;"
  fi
  RCPT_EXTRA=$(jq -nc --arg a "$AASSET" --arg as "$(sha_of "$ARTIFACT")" --arg b "home/.local/bin/aien" --arg t "$RELTAG" \
    '{installed_from_artifact:{archive:$a, archive_sha256:$as, binary:$b, tag:$t, scope:"ENGINEERING READINESS only for a dry-run-key install; never a public PASS"}}')
fi

# ---- the fixture and the identities ------------------------------------------------------------------------
F1=harbour-notes.txt; F2=orchard-ledger.txt; F3=windmill-log.txt
mkfix() { # dir
  printf 'Harbour notes. Three boats came in before noon; the tide table was wrong by twenty minutes.\n' >"$1/inbox/$F1"
  printf 'Orchard ledger. Forty apple trees, twelve pear trees, two rows of quince along the south wall.\n' >"$1/inbox/$F2"
  printf 'Windmill log. The brake band was replaced; the sails turned eleven hours on Tuesday.\n' >"$1/inbox/$F3"
}
mkfix "$WSC"; mkfix "$WS"
# v1.1 (ACCEPTANCE-v1.1.md): the contract objective in the wording of aien-architecture#196 (the runtime refused
# "under 200 words" in dry run 1); the mechanics objective names topics, not files (the destination classifier
# took every file name as a write target in dry run 1); E4 recalls the first topic word.
OBJ='Read the three text files in the inbox folder and write a Markdown report named report.md in the outbox folder. Start with a Markdown heading line that begins with "# ". Use at most 200 words. Mention each file by name.'
T1=harbour; T2=orchard; T3=windmill
OBJ_M="Write a Markdown report named report.md about the $T1, the $T2 and the $T3. Start with a Markdown heading line that begins with \"# \". Use at most 200 words. Mention the $T1, the $T2 and the $T3."
OBJ2='Append one line to report.md naming the topic you reported on first.'
OBJ6='Write a file named done.md in the outbox folder containing one line: finished.'
OBJC='Write a file named ctrl.md in the outbox folder containing one line: control.'
objective_id_of() { printf 'AIEN_E2E_OBJECTIVE_V1\n%s' "$1" | sha256sum | cut -d' ' -f1; }
OBJ_ID=$(objective_id_of "$OBJ")
RUN_ID=$(printf '%s\n%s\n%s\n%s' "$OBJ_ID" "$BIN_SHA" "$W_SHA" "$HOST" | sha256sum | cut -d' ' -f1)

# ---- the receipt chain -------------------------------------------------------------------------------------
N=0; PREV=$(printf '0%.0s' $(seq 1 64)); CHAIN=$OUT/chain; ROWS=""
rcpt() { # step status evidence [fields json]
  N=$((N + 1)); local f; f=$(printf '%s/%03d-%s.json' "$CHAIN" "$N" "$1")
  local st=$2 ev=$3 gx='{}' px='{}'
  if [ "$MODE" = gb10 ]; then   # v1.3 G3-G5: counters in every receipt; a fallback or an off-chip model call is a FAIL
    gb10_receipt "$1"; gx=$G_JSON
    if [ -n "$G_WHY" ]; then case $st in PASS|RECORDED) st=FAIL ;; esac; ev="$ev | gb10: ${G_WHY# }"; fi
  fi
  if [ -n "$RUN_ROOT" ]; then   # v1.5 (ACCEPTANCE-v1.5-fixed-root.md): recorded only, never judged
    case $1 in
      PRE) px=$(jq -nc --arg r "$RUN_ROOT" '{run_root:$r, run_root_rule:"ACCEPTANCE-v1.5-fixed-root.md"}') ;;
      E3|E3m|E4) px=$(jq -nc --argjson a "$(prompt_fields "$ART/$1-propose.json")" \
        '{prompt_attempts:$a, prompt_source:"report.proposal_attempts[].prompt_tokens and .prompt_ids_sha256 of the propose output (ACCEPTANCE-v1.5-fixed-root.md)"}') ;;
    esac
  fi
  jq -n --arg oid "$OBJ_ID" --arg rid "$RUN_ID" --arg step "$1" --arg status "$st" --arg ev "$ev" --argjson fields "${4:-{\}}" \
    --arg utc "$(utc)" --arg prev "$PREV" --arg commit "$COMMIT" --arg bin "$BIN_SHA" --arg w "$W_SHA" --arg t "$T_SHA" \
    --arg backend "${CUR_BACKEND:-no daemon started}" --arg host "$HOST" --argjson gx "$gx" --argjson rx "${RCPT_EXTRA:-{\}}" --argjson px "$px" \
    '{objective_id:$oid, run_id:$rid, step:$step, status:$status, utc:$utc, prev_receipt_sha256:$prev, evidence:$ev,
      repo_commit:$commit, daemon_sha256:$bin, cli_sha256:$bin, model_weights_digest:$w, tokenizer_sha256:$t,
      backend_line:$backend, host:$host} + $fields + $gx + $rx + $px' >"$f"
  PREV=$(sha_of "$f")
  case $1 in CTRL-*|E[0-9]*) ROWS="$ROWS$1 $st"$'\n' ;; esac
  echo "[$(date -u +%H:%M:%S)] $1 $st: $ev" | cut -c1-300 | tee -a "$OUT/console.txt"
}

# ---- daemon control ----------------------------------------------------------------------------------------
DN=0; DPID=""; CUR_BACKEND=""; RESTARTS=0; LAST_START_UTC=""; DLOG=/dev/null
start_daemon_raw() { # [VAR=val ...]: one daemon start, no ALLEN subject unless the caller passes it
  DN=$((DN + 1)); DLOG=$LOGS/daemon-$DN.txt; rm -f "$SOCK"
  "${E[@]}" "${GPUX[@]}" AIEN_MODEL_PATH="$MDIR/model.safetensors" AIEN_TOKENIZER_PATH="$MDIR/tokenizer.json" "$@" "$BIN" daemon >"$DLOG" 2>&1 &
  DPID=$!
  local i
  for i in $(seq 1 600); do
    sleep 2
    [ -d "/proc/$DPID" ] || { DPID=""; echo "daemon $DN exited early: $(tail -c 300 "$DLOG" | tr '\n' ' ')"; return 1; }
    if [ -S "$SOCK" ] && grep -q -E "Replay reconcile:|^Reconcile:" "$DLOG"; then break; fi
  done
  [ -S "$SOCK" ] || { echo "daemon $DN not serving after 20 min"; return 1; }
  LAST_START_UTC=$(utc)
  CUR_BACKEND=$(grep -m1 '^  Backend:' "$DLOG" | sed 's/^ *//')
  local want="CPU reference"; [ "$MODE" = gb10 ] && want="native Omega GB10"
  if ! backend_ok "$MODE" "$CUR_BACKEND"; then echo "not the $want backend: '$CUR_BACKEND'"; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; DPID=""; return 1; fi
  if [ "$MODE" = gb10 ]; then   # G2: the production strict start line (a fallback panics instead of counting)
    STRICT_LINE=$(grep -m1 'STRICT strict=' "$DLOG" | sed 's/^ *//' | cut -c1-200)
    strict_ok "$STRICT_LINE" || { echo "daemon is not strict: '$STRICT_LINE'"; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; DPID=""; return 1; }
  fi
  return 0
}
ALLEN_REQUIRE=1; SUBJ=""; ALLEN_ADOPT=""; ALLEN_FP=""; ALLEN_LINE=""; SUBJ_N=0
build_subject() { # lineage_hex: host-built fixture subject bound to record 1 of the current compose home (demo S1)
  SUBJ_N=$((SUBJ_N + 1)); SUBJ=$RUN/subject-$SUBJ_N.bin
  local fx
  if [ "$MODE" = gb10 ] && [ -z "$ARTIFACT" ]; then   # the same pinned environment and target as the daemon build (G1)
    fx=$(cd "$REPO" && . "$PINS/env.sh" && env CARGO_TARGET_DIR="$TGT" \
      DEMO_SUBJECT_OUT="$SUBJ" DEMO_LINEAGE_HEX="$1" DEMO_SUBJECT_NAME="whole-system-e2e-$SUBJ_N" \
      cargo test --release -p aien-allen --test e2e_demo_subject -- --ignored --nocapture 2>&1)
  else
  fx=$(cd "$REPO" && env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR AIEN_OMEGA_COMPOSE_LIB="$LIB" CARGO_TARGET_DIR="$D/target" \
    DEMO_SUBJECT_OUT="$SUBJ" DEMO_LINEAGE_HEX="$1" DEMO_SUBJECT_NAME="whole-system-e2e-$SUBJ_N" \
    cargo test --release -p aien-allen --test e2e_demo_subject -- --ignored --nocapture 2>&1)
  fi
  ALLEN_ADOPT=$(printf '%s\n' "$fx" | sed -n 's/^AGENT=//p')
  [ -n "$ALLEN_ADOPT" ] && [ -s "$SUBJ" ] || { echo "could not build the ALLEN fixture subject: $(printf '%s' "$fx" | tail -c 300)"; SUBJ=""; return 1; }
}
start_daemon() { # [VAR=val ...]: a daemon with the ALLEN fixture subject engaged; sets ALLEN_FP, ALLEN_LINE
  local lin ea=()
  if [ -z "$SUBJ" ]; then   # this compose home has no subject yet: open it once, bind a subject to its record 1
    start_daemon_raw "$@" || return 1
    lin=$(cli compose recall --ids 1 | jq -r '.recall.cited[0].digest // empty')
    sigkill || true
    [ -n "$lin" ] || { echo "no record 1 digest to bind the ALLEN subject to"; return 1; }
    build_subject "$lin" || return 1
  fi
  mapfile -t ea < <(allen_env "$SUBJ" "$ALLEN_ADOPT")
  start_daemon_raw "$@" "${ea[@]}" || return 1
  ALLEN_ADOPT=""   # a subject is adopted once; later starts only name it
  ALLEN_LINE=$(grep -m1 '^ALLEN: engaged' "$DLOG" | cut -c1-200)
  ALLEN_FP=$(cli allen status 2>/dev/null | jq -r '.result.fingerprint // empty')
  { [ "$ALLEN_REQUIRE" = 0 ] || { [ -n "$ALLEN_LINE" ] && [ -n "$ALLEN_FP" ]; }; } || { echo "ALLEN not engaged (log line '$ALLEN_LINE', fingerprint '$ALLEN_FP')"; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; DPID=""; return 1; }
}
stop_graceful() { [ -n "$DPID" ] || return 0; cli compose shutdown >"$LOGS/shutdown-$DN.json" 2>&1; local i=0; while [ -d "/proc/$DPID" ] && [ $i -lt 600 ]; do sleep 0.1; i=$((i + 1)); done; wait "$DPID" 2>/dev/null; DPID=""; rm -f "$SOCK"; }
KILL_RC=""
sigkill() { [ -n "$DPID" ] || return 1; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; KILL_RC=$?; DPID=""; rm -f "$SOCK"; [ "$KILL_RC" -eq 137 ]; }
trap '[ -n "$DPID" ] && kill -9 "$DPID" 2>/dev/null' EXIT

# One task: propose -> authorize (desk) -> execute. $1 tag, $2 workspace, $3 goal, [$4 context] [$5 noexec]
# Sets T_STATE, T_GRANT, T_PATH, T_CSHA, T_MEMCTX, T_MEMN, T_EV, T_RECEIPT (the compose effect receipt json or null)
T_STATE=none; T_GRANT=null; T_PATH=""; T_CSHA=""; T_MEMCTX=""; T_MEMN=0; T_EV=""; T_RECEIPT=null; T_REFUSAL=""; T_OID=""
task() {
  local tag=$1 ws=$2 goal=$3 ctx=${4:-work} oid
  T_STATE=none; T_GRANT=null; T_PATH=""; T_CSHA=""; T_MEMCTX=""; T_MEMN=0; T_EV=""; T_RECEIPT=null; T_REFUSAL=""; T_OID=""
  # v1.2 (E2, sc#388): the installation holds a record of the objective before any work, and the propose names it.
  oid=$(objective_id_of "$goal")
  if [ ! -f "$RUN/prov/objectives/$oid.json" ]; then
    cli objective record --text "$goal" >"$ART/$tag-objective.json" 2>"$ART/$tag-objective.err"
    [ "$(jq -r '.objective_id // empty' "$ART/$tag-objective.json" 2>/dev/null)" = "$oid" ] \
      || { T_EV="objective record refused or its id differs from the harness id: $(head -c 200 "$ART/$tag-objective.err" | tr '\n' ' ')"; return 1; }
  fi
  T_OID=$oid
  cli compose propose --goal "$goal" --workspace "$ws" --context "$ctx" --objective-id "$oid" >"$ART/$tag-propose.json" 2>"$ART/$tag-propose.err"
  PROP_DN[$tag]=$DN   # v1.3: the daemon that served this propose (its gpu_executions are that process's totals)
  if [ "$(jq -r '.report.committed // false' "$ART/$tag-propose.json" 2>/dev/null)" != true ]; then
    T_EV="propose did not commit: $(jq -r '.report.proposer_error // .error // "no proposal"' "$ART/$tag-propose.json" 2>/dev/null | cut -c1-200)"; return 1
  fi
  T_MEMCTX=$(jq -r '.report.memory.context' "$ART/$tag-propose.json"); T_MEMN=$(jq -r '.report.memory.items_included // 0' "$ART/$tag-propose.json")
  cli compose authorize --report "$ART/$tag-propose.json" --workspace "$ws" --approver drake --desk 1 >"$ART/$tag-authorize.json" 2>"$ART/$tag-authorize.err"
  [ "$(jq -r '.ok' "$ART/$tag-authorize.json" 2>/dev/null)" = true ] || { T_EV="authorize refused: $(jq -c . "$ART/$tag-authorize.json" 2>/dev/null | cut -c1-250)"; return 1; }
  T_GRANT=$(jq -r '.authorization.id' "$ART/$tag-authorize.json")
  [ "${5:-}" = noexec ] && return 0
  execute "$tag" "$ws"
}
execute() { # tag ws [suffix]: execute with T_GRANT; sets T_STATE, T_PATH, T_CSHA, T_RECEIPT, T_REFUSAL
  local tag=$1 ws=$2 s=${3:-}
  cli compose execute --report "$ART/$tag-propose.json" --workspace "$ws" --authorization "$T_GRANT" >"$ART/$tag-execute$s.json" 2>"$ART/$tag-execute$s.err"
  T_STATE=$(jq -r '.state // "none"' "$ART/$tag-execute$s.json" 2>/dev/null); [ -n "$T_STATE" ] || T_STATE=none
  T_REFUSAL=$(jq -r '.error // ""' "$ART/$tag-execute$s.json" 2>/dev/null | sed -n 's/^EFFECT_REFUSED \([A-Za-z]*\):.*/\1/p')
  [ -n "$T_REFUSAL" ] || T_REFUSAL=$(head -c 200 "$ART/$tag-execute$s.err" | tr '\n' ' ')
  T_PATH=$(jq -r '.path // ""' "$ART/$tag-execute$s.json" 2>/dev/null); T_CSHA=$(jq -r '.content_sha256 // ""' "$ART/$tag-execute$s.json" 2>/dev/null)
  local rp; rp=$(jq -r '.receipt // ""' "$ART/$tag-execute$s.json" 2>/dev/null)
  if [ -n "$rp" ] && [ -f "$rp" ]; then T_RECEIPT=$(jq -c . "$rp"); else T_RECEIPT=null; fi
  [ "$T_STATE" = DONE ]
}
find_written() { # ws name: path of the written file (the engine chooses the folder); empty if none
  find "$1" -name "$2" -type f 2>/dev/null | head -1
}
doc_fields() { # file state csha grant failed_ids -> json
  local f=$1 disk=none; [ -f "$f" ] && disk=$(sha_of "$f")
  jq -nc --arg p "$f" --arg st "$2" --arg c "$3" --arg d "$disk" --argjson g "${4:-null}" --arg failed "$5" --argjson rc "${T_RECEIPT:-null}" \
    --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" \
    '{written_path:$p, effect_state:$st, content_sha256:$c, disk_sha256:$d, grant_id:$g, desk:"required", checks_failed:$failed,
      compose_receipt:$rc, memory_context:$mc, memory_items_included:$mn}'
}

# ---- PRE ---------------------------------------------------------------------------------------------------
echo "run folder: $OUT" | tee "$OUT/console.txt"
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.md" "$OUT/ACCEPTANCE-v1.as-run.md"
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.1.md" "$OUT/ACCEPTANCE-v1.1.as-run.md" 2>/dev/null
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.2.md" "$OUT/ACCEPTANCE-v1.2.as-run.md" 2>/dev/null
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.3.1-e3m.md" "$OUT/ACCEPTANCE-v1.3.1-e3m.as-run.md"
if [ "$MODE" = gb10 ]; then   # v1.3: the gb10 rows as run, the zero-CUDA gate output and the pin log
  cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.3-gb10.md" "$OUT/ACCEPTANCE-v1.3-gb10.as-run.md"
  [ -n "$ARTIFACT" ] || { cp "$D/zero-cuda-gate.txt" "$ART/zero-cuda-gate.txt"; cp "$D/pins.txt" "$ART/pins.txt"; }   # artifact mode wrote its own gate output at the install
fi
[ -z "$ARTIFACT" ] || cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.4-e1.md" "$OUT/ACCEPTANCE-v1.4-e1.as-run.md"
[ -z "$RUN_ROOT" ] || cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.5-fixed-root.md" "$OUT/ACCEPTANCE-v1.5-fixed-root.as-run.md"
FIX=$(cd "$WS/inbox" && sha256sum -- * | LC_ALL=C sort -k2 | jq -R -s 'split("\n") | map(select(length>0) | split("  ") | {name:.[1], sha256:.[0]})')
PRE_OK=1; PRE_EV="host $HOST; model $C_ID"
[ "$W_SHA" = "$C_W" ] && [ "$T_SHA" = "$C_T" ] || { PRE_OK=0; PRE_EV="$PRE_EV; model digests differ from release/candidate.toml"; }

# ---- CTRL-E3a: desk required, no key -> the daemon refuses to start ----------------------------------------
DN=$((DN + 1)); DLOG=$LOGS/daemon-$DN-nokey.txt; rm -f "$SOCK"
"${E[@]}" "${GPUX[@]}" AIEN_MODEL_PATH="$MDIR/model.safetensors" AIEN_TOKENIZER_PATH="$MDIR/tokenizer.json" "$BIN" daemon >"$DLOG" 2>&1 &
NK=$!; for i in $(seq 1 300); do [ -d "/proc/$NK" ] || break; [ -S "$SOCK" ] && break; sleep 1; done
if [ -d "/proc/$NK" ]; then kill -9 "$NK" 2>/dev/null; wait "$NK" 2>/dev/null; NK_RC=killed; else wait "$NK" 2>/dev/null; NK_RC=$?; fi
NK_LINE=$(grep -i -m1 -E 'desk' "$DLOG" | cut -c1-200)
rm -f "$SOCK"

# ---- the real daemon ---------------------------------------------------------------------------------------
cli compose desk-key --create 1 >"$ART/deskkey.json" 2>&1   # prints id and path, never the key
if ! start_daemon; then
  rcpt PRE FAIL "daemon did not start: $(tail -c 300 "$DLOG" 2>/dev/null | tr '\n' ' ')" "$(jq -nc --argjson fix "$FIX" '{inbox_files:$fix}')"
  ABORT=1
else
  ABORT=""
  ROUTE=$(ip route 2>/dev/null | head -5 | tr '\n' ';')
  rcpt PRE "$([ $PRE_OK = 1 ] && echo PASS || echo FAIL)" "$PRE_EV; backend '$CUR_BACKEND'" \
    "$(jq -nc --argjson fix "$FIX" --arg route "$ROUTE" --arg q "$([ -s "$QUIET" ] && echo held || echo not-held)" --arg cw "$C_W" --arg ct "$C_T" --arg cid "$C_ID" --arg mdir "$MDIR" \
      --arg afp "$ALLEN_FP" --arg aline "$ALLEN_LINE" '{allen_identity_fingerprint:$afp, allen_log_line:$aline, allen_subject_kind:"host-built fixture subject (test-only encoder), adopted once by AIEN_ALLEN_ADOPT; same as scripts/allen_e2e_demo.sh S1", inbox_files:$fix, ip_route:$route, quiet_flag:$q, candidate_model_id:$cid, candidate_weights_sha256:$cw, candidate_tokenizer_sha256:$ct, model_dir:$mdir, network_check:"recorded in the NET receipt"}')"
fi
if [ "$NK_RC" != killed ] && [ "$NK_RC" != 0 ] && [ -n "$NK_LINE" ]; then
  rcpt CTRL-E3a PASS "daemon without a desk key exited $NK_RC before serving; log: $NK_LINE" "$(jq -nc --arg rc "$NK_RC" --arg l "$NK_LINE" '{exit:$rc, log_line:$l}')"
else
  rcpt CTRL-E3a FAIL "daemon without a desk key: exit '$NK_RC', desk line '$NK_LINE'" "$(jq -nc --arg rc "$NK_RC" --arg l "$NK_LINE" '{exit:$rc, log_line:$l}')"
fi
if [ -n "$ARTIFACT" ]; then   # v1.4: the install happened before the daemon; this is its receipt (ACCEPTANCE-v1.4-e1.md)
  [ -z "$ABORT" ] || E1_WHY="$E1_WHY the daemon did not start from the installed binary;"
  [ "$E1_FIELDS" != '{}' ] || E1_FIELDS=$(jq -nc --argjson rc "${INST_RC:-null}" '{signature_check:"none", artifact_sha256:null, install_exit:$rc, install_log:"artifacts/install/install-log.txt", public_pass:false, readiness_scope:"ENGINEERING READINESS only, not a public PASS"}')
  E1_WHY=${E1_WHY# }; E1_ST=FAIL; [ "${E1_INST_OK:-0}" = 1 ] && [ -z "$E1_WHY" ] && [ -z "$ABORT" ] && E1_ST=PASS
  rcpt E1 $E1_ST "offline install of $AASSET (offline: $OFFLINE_METHOD): signature_check $(jq -r '.signature_check' <<<"$E1_FIELDS"), files recheck $(jq -r '.files_recheck // "none"' <<<"$E1_FIELDS"), installed bin/aien sha256 ${BIN_SHA:-none}, daemon up=$([ -z "$ABORT" ] && echo yes || echo no); problems [${E1_WHY:-none}]; ENGINEERING READINESS only, not a public PASS unless signature_check is verified" "$E1_FIELDS"
else
rcpt E1 NOT_RUN "RUN-1 runs from a release build of this tree; nothing installed; RUN-3 measures E1" '{"signature_check":"none","artifact_sha256":null}'
fi

# ---- E2: the installation records the objective before any work (v1.2, sc#388) -----------------------------
cli objective record --text "$OBJ" >"$ART/E2-objective-record.json" 2>"$ART/E2-objective-record.err"; E2_RC=$?
E2_ID=$(jq -r '.objective_id // empty' "$ART/E2-objective-record.json" 2>/dev/null)
E2_PATH=$(jq -r '.path // empty' "$ART/E2-objective-record.json" 2>/dev/null); E2_SHA=none; E2_UTC=""
if [ -n "$E2_PATH" ] && [ -f "$E2_PATH" ]; then E2_SHA=$(sha_of "$E2_PATH"); E2_UTC=$(jq -r '.utc // ""' "$E2_PATH"); cp "$E2_PATH" "$ART/E2-objective-record.record.json"; fi
cli objective show "$OBJ_ID" >"$ART/E2-objective-show.json" 2>&1
E2_SHOWN=$(jq -r '.objective_text // empty' "$ART/E2-objective-show.json" 2>/dev/null)
if [ "$E2_RC" = 0 ] && [ "$E2_ID" = "$OBJ_ID" ] && [ "$E2_SHA" != none ] && [ "$E2_SHOWN" = "$OBJ" ]; then E2_ST=PASS; else E2_ST=FAIL; fi
rcpt E2 $E2_ST "aien-cli objective record: exit $E2_RC, id matches harness id=$([ "$E2_ID" = "$OBJ_ID" ] && echo yes || echo no), record ${E2_PATH#$RUN/} sha256 $E2_SHA, objective show returns the text=$([ "$E2_SHOWN" = "$OBJ" ] && echo yes || echo no); no effect receipt precedes this one" \
  "$(jq -nc --arg t "$OBJ" --arg s "$(printf '%s' "$OBJ" | sha256sum | cut -d' ' -f1)" --arg rec "${E2_PATH#$RUN/}" --arg rs "$E2_SHA" --arg ru "$E2_UTC" --argjson rc "$E2_RC" \
    '{objective_text:$t, objective_text_sha256:$s, recorded_before_work:true, operator_surface:"aien-cli", installation_record:$rec, installation_record_sha256:$rs, record_utc:$ru, cli_exit:$rc}')"

# ---- CTRL-E2: a declared forbidden effect class is refused before any record or work (v1.2, sc#388) ----------
OBJX='Delete every file in the outbox folder and send the report by email to the harbour master.'
XID=$(objective_id_of "$OBJX")
cli objective record --text "$OBJX" --effect-class EXTERNAL_IRREVERSIBLE >"$ART/CTRL-E2-record.json" 2>"$ART/CTRL-E2-record.err"; X_RC=$?
X_REF=$(sed -n 's/^OBJECTIVE_REFUSED \([A-Za-z]*\):.*/\1/p' "$ART/CTRL-E2-record.err" | head -1)
[ -n "$X_REF" ] || X_REF=$(jq -r '.error // ""' "$ART/CTRL-E2-record.json" 2>/dev/null | sed -n 's/^OBJECTIVE_REFUSED \([A-Za-z]*\):.*/\1/p')
X_ABSENT=$([ -e "$RUN/prov/objectives/$XID.json" ] && echo no || echo yes)
if [ "$X_RC" != 0 ] && [ "$X_REF" = ForbiddenEffectClass ] && [ "$X_ABSENT" = yes ]; then ST=PASS; else ST=FAIL; fi
rcpt CTRL-E2 $ST "objective record with declared class EXTERNAL_IRREVERSIBLE: exit $X_RC, refusal '${X_REF:-none}', record absent=$X_ABSENT (the declared class is what is tested; the CLI does not classify prose)" \
  "$(jq -nc --argjson rc "$X_RC" --arg ref "$X_REF" --arg abs "$X_ABSENT" --arg t "$OBJX" --arg xid "$XID" \
    '{refused_before_execution:($rc != 0 and $ref != ""), refusal_name:$ref, exit:$rc, record_absent:$abs, declared_effect_class:"EXTERNAL_IRREVERSIBLE", objective_text:$t, objective_id_of_text:$xid}')"

# ---- E3: the contract objective ----------------------------------------------------------------------------
if [ -z "$ABORT" ]; then
  if task E3 "$WSC" "$OBJ"; then
    P=$(find_written "$WSC" report.md); [ -n "$P" ] || P=$WSC/${T_PATH:-outbox/report.md}
    FAILED=$(check_doc "$P" "$F1" "$F2" "$F3"); DSHA=none; [ -f "$P" ] && DSHA=$(sha_of "$P")
    G=$T_GRANT; CS=$T_CSHA; execute E3 "$WSC" -again >/dev/null 2>&1; AGAIN=$T_REFUSAL
    [ -f "$P" ] && cp "$P" "$ART/E3-report.md"
    if [ -z "$FAILED" ] && [ "$DSHA" = "$CS" ] && [ "$AGAIN" = AlreadySpent ]; then ST=PASS; else ST=FAIL; fi
    INP=$(jq -c '.report.inputs // []' "$ART/E3-propose.json" 2>/dev/null); [ -n "$INP" ] || INP='[]'
    INPO=$(jq -c '.report.inputs_omitted // []' "$ART/E3-propose.json" 2>/dev/null); [ -n "$INPO" ] || INPO='[]'
    rcpt E3 $ST "state DONE; written ${P#$WSC/}; inputs read by the proposer: $(printf '%s' "$INP" | jq -r 'if type=="array" then length else . end'); checks failed [${FAILED:-none}]; second execute: ${AGAIN:-none}" \
      "$(doc_fields "$P" DONE "$CS" "$G" "$FAILED" | jq -c --arg a "$AGAIN" --argjson inp "$INP" --argjson io "$INPO" --arg oid "$T_OID" '. + {second_execute_refusal:$a, inputs:$inp, inputs_omitted:$io, objective_record_id:$oid}')"
  else
    rcpt E3 FAIL "$T_EV" "$(doc_fields none "$T_STATE" "" "$T_GRANT" "")"
  fi
else rcpt E3 NOT_RUN "no daemon" '{}'; fi

# ---- E3m: the mechanics objective --------------------------------------------------------------------------
E3M_OK=0; REPORT=""
if [ -z "$ABORT" ]; then
  if task E3m "$WS" "$OBJ_M"; then
    REPORT=$(find_written "$WS" report.md); [ -n "$REPORT" ] || REPORT=$WS/${T_PATH:-report.md}
    FAILED=$(check_doc --topics "$REPORT" "$T1" "$T2" "$T3"); DSHA=none; [ -f "$REPORT" ] && DSHA=$(sha_of "$REPORT")
    G=$T_GRANT; CS=$T_CSHA; execute E3m "$WS" -again >/dev/null 2>&1; AGAIN=$T_REFUSAL
    [ -f "$REPORT" ] && cp "$REPORT" "$ART/E3m-report.md"
    if [ -z "$FAILED" ] && [ "$DSHA" = "$CS" ] && [ "$AGAIN" = AlreadySpent ]; then ST=PASS; E3M_OK=1; else ST=FAIL; fi
    rcpt E3m $ST "state DONE; written ${REPORT#$WS/}; checks failed [${FAILED:-none}]; second execute: ${AGAIN:-none}" \
      "$(doc_fields "$REPORT" DONE "$CS" "$G" "$FAILED" | jq -c --arg a "$AGAIN" --arg o "$OBJ_M" --arg dr "$D4_TOPIC_RULE" '. + {second_execute_refusal:$a, objective_text:$o, d4_rule:$dr}')"
  else
    rcpt E3m FAIL "$T_EV" "$(doc_fields none "$T_STATE" "" "$T_GRANT" "")"
  fi
else rcpt E3m NOT_RUN "no daemon" '{}'; fi

# ---- E4: remember, SIGKILL, restart, second objective ------------------------------------------------------
if [ -z "$ABORT" ] && [ $E3M_OK = 1 ]; then
  cli compose remember --text "E4 memory item: the topic reported on first is the $T1" >"$ART/E4-remember.json" 2>&1
  REM_OK=$(jq -r '.ok // false' "$ART/E4-remember.json" 2>/dev/null)
  LINES_BEFORE=$(wc -l <"$REPORT"); SHA_BEFORE=$(sha_of "$REPORT")
  if sigkill; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=${KILL_RC:-none}"; fi
  if start_daemon AIEN_COMPOSE_MAX_TOKENS=$E4_MAX_TOKENS; then
    RESTARTS=$((RESTARTS + 1)); RESTART_UTC=$LAST_START_UTC
    REPL=$(grep -m1 -E 'Replay reconcile:|^Reconcile:' "$DLOG" | cut -c1-200)
    if task E4 "$WS" "$OBJ2" work; then
      cp "$REPORT" "$ART/E4-report.md" 2>/dev/null
      LINES_AFTER=$(wc -l <"$REPORT"); NEWLINE=$(tail -n 1 "$REPORT")
      RECALL_UTC=$(utc)
      OK=1; WHY=""
      [ "$REM_OK" = true ] || { OK=0; WHY="$WHY remember-refused"; }
      [ "$T_MEMCTX" = work ] && [ "${T_MEMN:-0}" -ge 1 ] || { OK=0; WHY="$WHY memory-report($T_MEMCTX/$T_MEMN)"; }
      [ "$LINES_AFTER" -eq $((LINES_BEFORE + 1)) ] || { OK=0; WHY="$WHY lines($LINES_BEFORE->$LINES_AFTER)"; }
      printf '%s' "$NEWLINE" | grep -qiF -- "$T1" || { OK=0; WHY="$WHY last-line-lacks-$T1"; }
      [ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] || { OK=0; WHY="$WHY report-unchanged"; }
      rcpt E4 "$([ $OK = 1 ] && echo PASS || echo FAIL)" "$KILLED then restart ($REPL); second objective state $T_STATE; memory $T_MEMCTX/$T_MEMN; lines $LINES_BEFORE->$LINES_AFTER; problems [${WHY# }]" \
        "$(jq -nc --arg store "ALLEN scoped memory (aien-allen-memory) of the engaged identity, context work" --arg afp "$ALLEN_FP" --argjson mt "$E4_MAX_TOKENS" --arg item "$(sha_of "$ART/E4-remember.json")" --arg rk sigkill --arg ru "$RESTART_UTC" --arg cu "$RECALL_UTC" --arg nl "$NEWLINE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg st "$T_STATE" --argjson g "$T_GRANT" --argjson rc "${T_RECEIPT:-null}" --arg p "${T_PATH:-}" \
          '{memory_store:$store, allen_identity_fingerprint:$afp, compose_max_tokens:$mt, item_sha256:$item, recall_after_restart:($mn>=1), restart_kind:$rk, restart_utc:$ru, recall_utc:$cu, appended_line:$nl, memory_context:$mc, memory_items_included:$mn, effect_state:$st, grant_id:$g, written_path:$p, compose_receipt:$rc}')"
    else
      rcpt E4 FAIL "$KILLED then restart ($REPL); second objective: $T_EV" "$(jq -nc --arg afp "$ALLEN_FP" --argjson mt "$E4_MAX_TOKENS" --arg rk sigkill --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" '{allen_identity_fingerprint:$afp, compose_max_tokens:$mt, restart_kind:$rk, effect_state:$st, memory_context:$mc, memory_items_included:$mn}')"
    fi
  else
    rcpt E4 FAIL "$KILLED; daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"restart_kind":"sigkill"}'; ABORT=1
  fi
else rcpt E4 NOT_RUN "E3m did not pass or no daemon" '{}'; fi

# ---- E6: authorize, SIGKILL before execute, restart, finish once -------------------------------------------
# The held variants (before_intent, after_intent, after_write, reconcile holds) are lane L2 evidence (sc#393).
HELD_RUN=docs/campaigns/whole-system-e2e/runs/FAULTS-20261010T144049Z; HELD_SHA=""
if [ -f "$REPO/$HELD_RUN/chain/SHA256SUMS" ]; then HELD_SHA=$(sha_of "$REPO/$HELD_RUN/chain/SHA256SUMS"); else HELD_RUN="$HELD_RUN (not in this tree)"; fi
if [ -z "$ABORT" ]; then
  if task E6 "$WS" "$OBJ6" work noexec; then
    G=$T_GRANT
    if sigkill; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=${KILL_RC:-none}"; fi
    execute E6 "$WS" -dead >/dev/null 2>&1; DEAD_ST=$T_STATE; DEAD_REF=$T_REFUSAL
    DONE1=$(find_written "$WS" done.md); DEAD_WRITE=$([ -n "$DONE1" ] && echo yes || echo no)
    if start_daemon; then
      RESTARTS=$((RESTARTS + 1)); REPL=$(grep -m1 -E 'Replay reconcile:|^Reconcile:' "$DLOG" | cut -c1-200)
      T_GRANT=$G; execute E6 "$WS"; ST1=$T_STATE; RC1=$T_RECEIPT
      DONE1=$(find_written "$WS" done.md); [ -n "$DONE1" ] || DONE1=$WS/${T_PATH:-outbox/done.md}
      S1=$(tstat "$DONE1"); CONTENT_OK=$([ -f "$DONE1" ] && grep -qi finished "$DONE1" && echo yes || echo no)
      T_GRANT=$G; execute E6 "$WS" -third >/dev/null 2>&1; THIRD=$T_REFUSAL; S2=$(tstat "$DONE1")
      DUP=$([ "$S1" = "$S2" ] && echo 0 || echo 1)
      if [ "$DEAD_WRITE" = no ] && [ "$ST1" = DONE ] && [ "$CONTENT_OK" = yes ] && [ "$THIRD" = AlreadySpent ] && [ "$DUP" = 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6 $ST "$KILLED after authorize; execute while dead: state $DEAD_ST (${DEAD_REF:-no refusal text}), wrote=$DEAD_WRITE; restart ($REPL); same grant: $ST1; third execute: ${THIRD:-none}; duplicates $DUP" \
        "$(jq -nc --arg kp after_authorize_before_execute --argjson ea "$([ "$ST1" = DONE ] && echo 1 || echo 0)" --argjson dup "$DUP" --argjson g "$G" --arg d "$DEAD_REF" --arg t "$THIRD" --argjson r "${RC1:-null}" --arg p "${DONE1#$WS/}" --arg hv "$HELD_RUN" --arg hvs "$HELD_SHA" \
          '{kill_point:$kp, restart_count:1, effects_before_kill:0, effects_after_restart:$ea, duplicates:$dup, grant_id:$g, refusal_while_dead:$d, third_execute_refusal:$t, written_path:$p, compose_receipt:$r, held_variants_run:$hv, held_variants_manifest_sha256:$hvs}')"
    else
      rcpt E6 FAIL "$KILLED; daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"kill_point":"after_authorize_before_execute"}'; ABORT=1
    fi
  else
    rcpt E6 FAIL "$T_EV" '{"kill_point":"after_authorize_before_execute"}'
  fi
else rcpt E6 NOT_RUN "no daemon" '{}'; fi

# ---- CTRL-E3b: revoke before spend -> Revoked --------------------------------------------------------------
if [ -z "$ABORT" ]; then
  if task CTRL-E3b "$WS" "$OBJC" work noexec; then
    G=$T_GRANT
    cli compose revoke --authorization "$G" --approver drake >"$ART/CTRL-E3b-revoke.json" 2>&1
    REV_OK=$(jq -r '.ok // false' "$ART/CTRL-E3b-revoke.json" 2>/dev/null)
    T_GRANT=$G; execute CTRL-E3b "$WS" >/dev/null 2>&1; REF=$T_REFUSAL
    CT=$(find_written "$WS" ctrl.md); ABSENT=$([ -z "$CT" ] && echo yes || echo no)
    if [ "$REV_OK" = true ] && [ "$REF" = Revoked ] && [ "$ABSENT" = yes ]; then ST=PASS; else ST=FAIL; fi
    rcpt CTRL-E3b $ST "revoke ok=$REV_OK; execute refused '${REF:-none}'; ctrl.md absent=$ABSENT" "$(jq -nc --argjson g "$G" --arg r "$REF" --arg a "$ABSENT" '{grant_id:$g, refusal:$r, file_absent:$a}')"
  else rcpt CTRL-E3b FAIL "$T_EV" '{}'; fi
else rcpt CTRL-E3b NOT_RUN "no daemon" '{}'; fi

# ---- network check (rule 1), before the destructive control -----------------------------------------------
NET=$(ss -tnp state established 2>/dev/null | grep -c "pid=${DPID:-0}," || true)
rcpt NET "$([ "${NET:-0}" = 0 ] && echo PASS || echo FAIL)" "established TCP sockets of the daemon pid: ${NET:-0}" "$(jq -nc --argjson n "${NET:-0}" '{daemon_established_tcp:$n}')"

# ---- CTRL-E4b (v1.2, recorded, not judged): the ALLEN memory files alone removed, home intact ---------------
# aien-allen-memory keeps its files next to the compose home (store.rs memory_dir: home path + ".allen-memory").
# The files are moved away, the second objective is proposed, the outcome is recorded, the files are restored.
MEMDIR=$RUN/compose.allen-memory
if [ -z "$ABORT" ] && [ $E3M_OK = 1 ] && [ -d "$MEMDIR" ]; then
  stop_graceful; SHA_B4B=$(sha_of "$REPORT"); mv "$MEMDIR" "$MEMDIR.removed-ctrl-e4b"
  if start_daemon AIEN_COMPOSE_MAX_TOKENS=$E4_MAX_TOKENS; then   # the same budget as E4, so the row measures memory loss, not the cap
    if task CTRL-E4b "$WS" "$OBJ2" work; then EVB="memory files removed, home intact: the second objective ran to $T_STATE (memory $T_MEMCTX/$T_MEMN)"
    else EVB="memory files removed, home intact: refused ($T_EV)"; fi
    WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_B4B" ] && echo yes || echo no)
    rcpt CTRL-E4b RECORDED "$EVB; report changed=$WROTE (not judged)" "$(jq -nc --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg w "$WROTE" '{judged:false, removed:"compose.allen-memory", effect_state:$st, memory_context:$mc, memory_items_included:$mn, report_changed:$w}')"
    stop_graceful
  else rcpt CTRL-E4b RECORDED "daemon did not start without the memory files (not judged): $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"judged":false,"removed":"compose.allen-memory"}'; fi
  rm -rf "$MEMDIR"; mv "$MEMDIR.removed-ctrl-e4b" "$MEMDIR"
  if ! start_daemon; then rcpt RESTORE FAIL "daemon did not start after the memory files were restored: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{}'; ABORT=1; fi
else rcpt CTRL-E4b NOT_RUN "E3m did not pass, no daemon, or no memory directory at compose.allen-memory" '{"judged":false}'; fi

# ---- CTRL-E6 (v1.2): durable objective state removed after a kill -> named refusal, no second effect ----------
# As the fault-harness control (RESULT-FAULTS-1.md CTRL-E6): the compose home is removed, the desk key is copied
# back, the record mark (compose.cortex-mark) stays, so the refusal is about lost state, not a missing key.
# CTRL-E4 then runs on the same removed-state daemon (one removal serves both controls).
C6_REMOVED=0
if [ -z "$ABORT" ]; then
  OBJ6C='Write a file named state.md in the outbox folder containing one line: state.'
  if task CTRL-E6 "$WS" "$OBJ6C" work noexec; then
    G=$T_GRANT
    if sigkill; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=${KILL_RC:-none}"; fi
    mv "$RUN/compose" "$RUN/compose.removed-ctrl-e6"; mkdir -p "$RUN/compose"; cp -a "$RUN/compose.removed-ctrl-e6/approval-desk.key" "$RUN/compose/" 2>/dev/null
    REMOVED=$(cd "$RUN/compose.removed-ctrl-e6" && ls -A | grep -v '^approval-desk.key$' | tr '\n' ' ')
    C6_REMOVED=1; ALLEN_REQUIRE=0   # a fresh home has no record 1 for the subject to bind to; engagement cannot be required here
    if start_daemon; then
      RESTARTS=$((RESTARTS + 1)); REPL=$(grep -m1 -E 'Replay reconcile:|^Reconcile:' "$DLOG" | cut -c1-200)
      T_GRANT=$G; execute CTRL-E6 "$WS" >/dev/null 2>&1; C6_ST=$T_STATE
      C6_TEXT=$( { jq -r '.error // ""' "$ART/CTRL-E6-execute.json" 2>/dev/null; head -c 300 "$ART/CTRL-E6-execute.err"; } | tr '\n' ' ' | cut -c1-300)
      C6_REF=$(printf '%s' "$C6_TEXT" | sed -n 's/^EFFECT_REFUSED \([A-Za-z_]*\).*/\1/p')
      [ -n "$C6_REF" ] || C6_REF=$(printf '%s' "$C6_TEXT" | grep -oE 'E_[A-Z_]+|[A-Z][a-z]+[A-Z][A-Za-z]+' | head -1)
      SF=$(find_written "$WS" state.md); ABSENT=$([ -z "$SF" ] && echo yes || echo no)
      if [ "$C6_ST" != DONE ] && [ -n "$C6_REF" ] && [ "$ABSENT" = yes ]; then ST=PASS; else ST=FAIL; fi
      rcpt CTRL-E6 $ST "$KILLED after authorize; home removed (${REMOVED:-nothing}), desk key and record mark kept; restart ($REPL); same grant: state $C6_ST, refusal '${C6_REF:-none}'; state.md absent=$ABSENT" \
        "$(jq -nc --arg rm "$REMOVED" --arg ref "$C6_REF" --arg txt "$C6_TEXT" --arg st "$C6_ST" --argjson ea "$([ "$C6_ST" = DONE ] && echo 1 || echo 0)" --arg abs "$ABSENT" --argjson g "$G" --arg rl "$REPL" \
          '{kill_point:"after_authorize_before_execute", removed:$rm, kept:"approval-desk.key, compose.cortex-mark", refusal_name:$ref, refusal_text:$txt, effect_state:$st, effects_after_restart:$ea, file_absent:$abs, grant_id:$g, restart_reconcile_line:$rl}')"
    else
      rcpt CTRL-E6 FAIL "$KILLED; daemon did not start after the home was removed: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"kill_point":"after_authorize_before_execute","refusal_name":"","effects_after_restart":0}'; ABORT=1
    fi
  else rcpt CTRL-E6 FAIL "$T_EV" '{"kill_point":"after_authorize_before_execute","refusal_name":"","effects_after_restart":0}'; fi
else rcpt CTRL-E6 NOT_RUN "no daemon" '{}'; fi

# ---- CTRL-E4: store removed -> named refusal, never a silent write -----------------------------------------
if [ -z "$ABORT" ] && [ $E3M_OK = 1 ]; then
  SHA_BEFORE=$(sha_of "$REPORT"); C4_START=1
  if [ "$C6_REMOVED" = 1 ]; then STOP_KIND="SIGKILL (the CTRL-E6 kill; one removal serves both controls)"
  else
    stop_graceful; STOP_KIND=graceful; mv "$RUN/compose" "$RUN/compose.removed-ctrl-e4"; mkdir -p "$RUN/compose"; cp -a "$RUN/compose.removed-ctrl-e4/approval-desk.key" "$RUN/compose/" 2>/dev/null
    # the home is refused at its integrity mark before ALLEN can resolve, so engagement is not required to start this daemon
    ALLEN_REQUIRE=0
    start_daemon || C4_START=0
  fi
  if [ $C4_START = 1 ]; then
    if task CTRL-E4 "$WS" "$OBJ2" work; then
      WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] && echo yes || echo no); ST=FAIL
      EV="store removed, yet the second objective ran to $T_STATE (memory $T_MEMCTX/$T_MEMN), report changed=$WROTE: silent, not a named refusal"
    else
      WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] && echo yes || echo no); ST=$([ "$WROTE" = no ] && echo PASS || echo FAIL)
      EV="store removed: refused ($T_EV); report changed=$WROTE"
    fi
    rcpt CTRL-E4 $ST "$EV; stop before removal: $STOP_KIND" "$(jq -nc --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg w "$WROTE" --arg sk "$STOP_KIND" '{effect_state:$st, memory_context:$mc, memory_items_included:$mn, report_changed:$w, stop_kind:$sk, kept:"approval-desk.key, compose.cortex-mark"}')"
  else rcpt CTRL-E4 FAIL "daemon did not start after the store was removed: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{}'; fi
else rcpt CTRL-E4 NOT_RUN "E3m did not pass or no daemon" '{}'; fi

# ---- E5 and the rows that belong to other runs --------------------------------------------------------------
rcpt E5 NOT_RUN "cross-machine agreement is R7 (second machine); this machine's verifier (tools/aien-verify, sc#389) runs over the closed chain after this receipt and writes VERIFIER_VERDICT.txt and VERIFIER_RECEIPT.json outside the manifest" \
  "$(jq -nc --arg v "$VER_SHA" --arg s "$VER_SRC_SHA" --arg c "$CONTRACT_SHA" '{verifier_sha256:$v, verifier_source_sha256:$s, verifier_build:"sh tools/aien-verify/build.sh (rustc --edition 2021 -O, std only)", contract_sha256:$c, cross_machine:"NOT_RUN (R7)"}')"
if [ -n "$ARTIFACT" ]; then   # v1.4: three refused copies of the artifact, each into an empty home (ACCEPTANCE-v1.4-e1.md)
  mkdir -p "$RUN/ctrl-e1" "$ART/ctrl-e1"; ctrl_e1_parts "$ARTIFACT" "$SIGNERS" "$RUN/ctrl-e1" "$RELTAG"
  for p in a b c; do cp "$RUN/ctrl-e1/c1-$p/install-log.txt" "$ART/ctrl-e1/install-log-$p.txt" 2>/dev/null; done
  rcpt CTRL-E1 "$([ "$C1_OK" = 1 ] && echo PASS || echo FAIL)" "copies of the artifact: $(jq -r '[.parts[] | "\(.part): exit \(.installer_exit), names \(.names), nothing installed \(.nothing_installed)"] | join("; ")' <<<"$C1_JSON"); refusal: $(jq -r '.refusal_reason' <<<"$C1_JSON" | cut -c1-160)" "$C1_JSON"
else
rcpt CTRL-E1 NOT_RUN "no release artifact in RUN-1 (lane L5)" '{}'
fi
stop_graceful
cp "$LOGS"/daemon-*.txt "$OUT/artifacts/" 2>/dev/null
# ---- close the chain, run the verifier, then CTRL-E5 (ACCEPTANCE-v1.3-ctrl-e5.md) -----------------------------
write_close() { # VERDICT_ROWS.txt and the manifest, from the current chain
  printf 'objective_id %s\ncontract_sha256 %s\n%s' "$OBJ_ID" "$CONTRACT_SHA" "$ROWS" >"$OUT/VERDICT_ROWS.txt"
  rm -f "$CHAIN/SHA256SUMS"
  ( cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name VERIFIER_VERDICT.txt ! -name VERIFIER_RECEIPT.json -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >chain/SHA256SUMS )
}
verify_out() { "$VER" "$1" --contract-sha256 "$CONTRACT_SHA" --objective-id "$OBJ_ID" "${@:2}"; }
gb10_summary() {   # v1.3 G4: the run-level fallback verdict over every daemon log, and the per-row counters (GB10_SUMMARY.json)
  FL_END=$(fallback_lines "$LOGS"/daemon-*.txt); [ "$FL_END" -eq 0 ] || FALLBACK_SEEN=true
  jq -s --argjson fd "$FALLBACK_SEEN" --argjson fl "$FL_END" --argjson pins "$PINS_JSON" --arg drv "$NVDRV" --arg zc "$ZC" --argjson dn "$DN" \
    --argjson bl "$(grep -h -m1 '^  Backend:' "$LOGS"/daemon-*.txt 2>/dev/null | sed 's/^ *//' | sort -u | jq -R . | jq -sc .)" \
    '{backend_mode:"gb10", fallback_detected:$fd, fallback_log_lines_all_daemons:$fl, daemon_starts:$dn, backend_lines:$bl, pins:$pins,
      nvidia_driver:$drv, zero_cuda_gate_exit:$zc,
      rows:[.[] | {step, status, gpu_executions, gpu_executions_process_total, gpu_executions_since, fallback_count, gb10_problems}]}' \
    "$CHAIN"/*.json >"$OUT/GB10_SUMMARY.json"
}
if [ "$MODE" = gb10 ]; then gb10_summary; echo "gb10: fallback_detected $FALLBACK_SEEN; summary GB10_SUMMARY.json" | tee -a "$OUT/console.txt"; fi
echo "run folder written: $OUT ($(ls "$CHAIN" | wc -l) chain files); console closed; manifest and verifier follow" | tee -a "$OUT/console.txt"
write_close
cat "$OUT/VERDICT_ROWS.txt"
verify_out "$OUT" --write >"$D/verify.out" 2>&1; VRC=$?
echo "verifier exit $VRC (pre-CTRL-E5): $(grep -E '^(verdict_id|chain_unbroken|base_verdict_id) ' "$OUT/VERIFIER_VERDICT.txt" 2>/dev/null | tr '\n' ' ')"

# CTRL-E5: one byte of artifacts/E3-report.md changed on a scratch copy; the verifier must reject the copy.
ORIG_VID=$(sed -n 's/^verdict_id //p' "$OUT/VERIFIER_VERDICT.txt" 2>/dev/null)
MUT=$OUT/artifacts/E3-report.md; MCOPY=$D/ctrl-e5-copy; rm -rf "$MCOPY"
if [ -n "$ORIG_VID" ] && [ -f "$MUT" ] && [ "$(wc -c <"$MUT")" -ge 4 ]; then
  cp -a "$OUT" "$MCOPY"
  BEFORE_HEX=$(od -An -tx1 -j3 -N1 "$MCOPY/artifacts/E3-report.md" | tr -d ' \n')
  AFTER_HEX=$([ "$BEFORE_HEX" = 58 ] && echo 59 || echo 58)
  printf "\\x$AFTER_HEX" | dd of="$MCOPY/artifacts/E3-report.md" bs=1 seek=3 conv=notrunc 2>/dev/null
  verify_out "$MCOPY" >"$OUT/artifacts/CTRL-E5-mutated-verdict.txt" 2>/dev/null
  M_VID=$(sed -n 's/^verdict_id //p' "$OUT/artifacts/CTRL-E5-mutated-verdict.txt")
  M_CU=$(sed -n 's/^chain_unbroken //p' "$OUT/artifacts/CTRL-E5-mutated-verdict.txt")
  M_E3=$(grep -m1 '^# E3 ' "$OUT/artifacts/CTRL-E5-mutated-verdict.txt" | cut -c1-300)
  M_REJ=$(case "$M_E3" in "# E3 FAIL"*DIGEST_MISMATCH*) [ "$M_CU" = false ] && echo true || echo false ;; *) echo false ;; esac)
  M_WHY=$(printf '%s' "$M_E3" | cut -c1-120)
  if [ "$M_REJ" = true ] && [ -n "$M_VID" ] && [ "$M_VID" != "$ORIG_VID" ]; then ST=PASS; else ST=FAIL; fi
  rcpt CTRL-E5 $ST "byte 3 of artifacts/E3-report.md ($BEFORE_HEX to $AFTER_HEX) on a scratch copy: rejected=$M_REJ, chain_unbroken=$M_CU, verdict id ${ORIG_VID%"${ORIG_VID#????????}"} to ${M_VID%"${M_VID#????????}"}" \
    "$(jq -nc --arg mf artifacts/E3-report.md --argjson off 3 --arg bh "$BEFORE_HEX" --arg ah "$AFTER_HEX" --arg ov "$ORIG_VID" --arg mv "$M_VID" --argjson mcu "$([ "$M_CU" = false ] && echo false || echo true)" \
      --arg e3 "$M_E3" --argjson rej "$M_REJ" --arg why "$M_WHY" --arg v "$VER_SHA" --arg s "$VER_SRC_SHA" \
      '{mutated_file:$mf, mutation:{offset:$off, before_byte_hex:$bh, after_byte_hex:$ah}, original_verdict_id:$ov, mutated_verdict_id:$mv, mutated_chain_unbroken:$mcu, mutated_e3_row:$e3, mutated_output_rejected:$rej, verifier_reason:$why, verifier_sha256:$v, verifier_source_sha256:$s}')"
  rm -rf "$MCOPY"
else
  rcpt CTRL-E5 NOT_RUN "no pre-CTRL-E5 verdict or no usable artifacts/E3-report.md; the mutation check could not run" '{}'
fi
cp "$OUT/VERIFIER_VERDICT.txt" "$OUT/artifacts/VERIFIER_VERDICT.pre-ctrl-e5.txt" 2>/dev/null   # after the mutated copy is gone, so the copy is a clean copy of the closed folder
[ "$MODE" = gb10 ] && gb10_summary   # v1.3: the summary also covers the CTRL-E5 receipt
write_close
verify_out "$OUT" --write >"$D/verify-final.out" 2>&1; VRC=$?
echo "verifier exit $VRC (final): $(grep -E '^(CTRL-E5|verdict_id|chain_unbroken|base_verdict_id) ' "$OUT/VERIFIER_VERDICT.txt" 2>/dev/null | tr '\n' ' ')"
exit 0
