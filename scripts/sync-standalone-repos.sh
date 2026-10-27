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

#
# sync-standalone-repos.sh: Monorepo Crate Synchronization Pipeline
#
# Synchronizes crates from aien-sovereign-core to standalone publish repositories on DGX Spark.
# Note (2026-10-10, #93): every standalone repository on GitHub is archived read-only
# with a pointer back to its crate here. --check is the drift check against local
# publish checkouts; sync mode only matters if they are ever unarchived.
# Adheres strictly to Sovereign Voice: zero em dashes, zero en dashes.
#
# Preserved standalone-specific files:
# action.yml, README.md, LICENSE, Cargo.lock, AGENTS.md, AGENT_CODE_OF_CONDUCT.md,
# CONSTITUTION.md, CONTRIBUTING.md, SECURITY.md, PHILOSOPHY.md, agent.json, llms.txt,
# evals.json, workflows.json, schemas/, kb/, failures/, assets/, scripts/, install.sh,
# Dockerfile, .github/, .git/, .gitignore, .crumb, .crumb.local.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOVEREIGN_CORE="$(cd "$SCRIPT_DIR/.." && pwd)"
# Parent directory holding the <publish_dirname> checkouts. Override for a CI
# runner or a scratch directory; defaults to the monorepo's parent.
WORKSPACE="${AIEN_MIRROR_WORKSPACE:-$(cd "$SOVEREIGN_CORE/.." && pwd)}"
# Git source used for monorepo path dependencies in mirrored manifests.
MONOREPO_GIT_URL="https://github.com/aien-spark/aien-sovereign-core"

# Source [dependencies] that sync_manifest may APPEND to a mirror manifest that
# lacks them, per repo (space-separated dep names). Repos not listed get no
# appended deps. Keep this list explicit: appending a dep changes what the
# mirror builds, so each entry must be a reviewed decision.
#   spark-harness: aien-proof and blake3 back the verified completion gate (#533).
#   cortex-rs, spark-supervisor, spark-dream, spark-debugger: aien-localhttp
#   replaces reqwest for their local http calls (#553).
manifest_append_deps() {
  case "$1" in
    spark-harness) echo "aien-proof blake3" ;;
    cortex-rs | spark-supervisor | spark-dream | spark-debugger) echo "aien-localhttp" ;;
    *) echo "" ;;
  esac
}

# Source [dev-dependencies] that sync_manifest may rewrite or APPEND in a
# mirror manifest, per repo. Same rule as manifest_append_deps: explicit,
# reviewed entries only.
#   cortex-rs, spark-dream, spark-debugger: their #[cfg(test)] code uses
#   aien_localhttp::test_server, which needs the `test-server` feature (#553).
manifest_append_dev_deps() {
  case "$1" in
    cortex-rs | spark-dream | spark-debugger) echo "aien-localhttp" ;;
    *) echo "" ;;
  esac
}

# Repository mapping: <cli_name>:<crate_subpath>:<publish_dirname>
REPOS=(
  "spark-hive:crates/spark-hive:spark-hive-publish"
  "spark-supervisor:crates/spark-supervisor:spark-supervisor-publish"
  "spark-dream:crates/spark-dream:spark-dream-publish"
  "cortex-rs:crates/cortex-rs:cortex-rs-publish"
  "spark-adapters:crates/spark-adapters:spark-adapters-publish"
  "spark-crumbs:crates/spark-crumbs:spark-crumbs-publish"
  "spark-debugger:crates/spark-debugger:spark-debugger-publish"
  "spark-harness:crates/spark-harness:aien-harness-publish"
  "spark-inquisitor:crates/spark-inquisitor:spark-inquisitor"
  "rad-id-sync:crates/rad-id-sync:rad-id-sync-publish"
)

CHECK_MODE=false
ALL_REPOS=false
TARGET_REPO=""
VERBOSE=false
RUN_TESTS=false

usage() {
  cat << 'HELP_EOF'
Usage: sync-standalone-repos.sh [OPTIONS]

Synchronizes member crates from aien-sovereign-core to standalone publish repositories.

Options:
  --check              Dry run and parity check mode (does not modify files on disk)
  --all                Synchronize or check all standalone repositories
  --repo <name>        Synchronize or check a specific repository by name
  --verbose, -v        Enable verbose logging and diff inspection
  --test               Run cargo test on synced repositories in addition to cargo check
  --help, -h           Show this help message and exit

Preserved standalone files:
  action.yml, README.md, LICENSE, Cargo.lock, AGENTS.md, AGENT_CODE_OF_CONDUCT.md,
  CONSTITUTION.md, CONTRIBUTING.md, SECURITY.md, PHILOSOPHY.md, agent.json, llms.txt,
  evals.json, workflows.json, schemas/, kb/, failures/, assets/, scripts/, install.sh,
  Dockerfile, .github/, .git/, .gitignore, .crumb, .crumb.local.
HELP_EOF
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_MODE=true
      shift
      ;;
    --all)
      ALL_REPOS=true
      shift
      ;;
    --repo)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --repo requires a repository name." >&2
        exit 1
      fi
      TARGET_REPO="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --test)
      RUN_TESTS=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$ALL_REPOS" = false && -z "$TARGET_REPO" ]]; then
  echo "ERROR: Specify either --all or --repo <name>." >&2
  echo "Run with --help for detailed usage." >&2
  exit 1
fi

check_dirty_worktree() {
  local target_dir="$1"
  local repo_name="$2"
  if [[ -d "$target_dir/.git" ]]; then
    local git_status
    git_status=$(git -C "$target_dir" status --porcelain 2>/dev/null || true)
    if [[ -n "$git_status" ]]; then
      echo "[WARN] Target worktree for $repo_name is dirty (uncommitted modifications detected):"
      echo "$git_status"
      return 1
    fi
  fi
  return 0
}

# Value of `key = "..."` inside the [package] table of a manifest (first match).
pkg_field() {
  awk -v key="$2" '
    /^[[:space:]]*\[/ { in_pkg = ($0 ~ /^[[:space:]]*\[package\][[:space:]]*$/); next }
    in_pkg && $0 ~ "^[[:space:]]*" key "[[:space:]]*=[[:space:]]*\"" {
      s = $0; sub(/^[^"]*"/, "", s); sub(/".*$/, "", s); print s; exit
    }
  ' "$1"
}

# Bring the destination [package] table in line with the source: license from the
# workspace root (single source of truth), version and edition from the source crate.
# Also: in-repo path deps become pinned git deps, and deps named by
# manifest_append_deps for this repo are added when missing (see below).
# Pure awk, no Python. Exit 0 parity or synced, 2 disparity in check mode, 1 error.
sync_manifest() {
  local src_manifest="$1"
  local dest_manifest="$2"
  local is_check="$3"
  local repo_name="$4"
  local ws_manifest="$SOVEREIGN_CORE/Cargo.toml"

  local license_id
  license_id=$(awk '
    /^[[:space:]]*\[/ { in_ws = ($0 ~ /^[[:space:]]*\[workspace\.package\][[:space:]]*$/); next }
    in_ws && /^[[:space:]]*license[[:space:]]*=[[:space:]]*"/ {
      s = $0; sub(/^[^"]*"/, "", s); sub(/".*$/, "", s); print s; exit
    }
  ' "$ws_manifest")
  if [[ -z "$license_id" ]]; then
    echo "[ERROR] No [workspace.package] license in $ws_manifest"
    return 1
  fi

  local src_version src_edition
  src_version=$(pkg_field "$src_manifest" version)
  src_edition=$(pkg_field "$src_manifest" edition)

  local tmp
  tmp=$(mktemp)
  awk -v lic="$license_id" -v ver="$src_version" -v ed="$src_edition" '
    function flush_license() {
      if (in_pkg && !license_set) { printf "license = \"%s\"\n", lic; license_set = 1 }
    }
    /^[[:space:]]*\[package\][[:space:]]*$/ { in_pkg = 1; print; next }
    /^[[:space:]]*\[/ { flush_license(); in_pkg = 0 }
    in_pkg && /^[[:space:]]*license(\.workspace)?[[:space:]]*=/ { printf "license = \"%s\"\n", lic; license_set = 1; next }
    in_pkg && ver != "" && /^[[:space:]]*version[[:space:]]*=/ { printf "version = \"%s\"\n", ver; next }
    in_pkg && ed != "" && /^[[:space:]]*edition[[:space:]]*=/ { printf "edition = \"%s\"\n", ed; next }
    { print }
    END { flush_license() }
  ' "$dest_manifest" > "$tmp"

  # [dependencies]: a source dep of the form `name = { path = "../<crate>" ... }`
  # cannot build outside the monorepo, and Cargo refuses `path` and `git` on one
  # dependency ("Only one of `git` or `path` is allowed", cargo 1.98.1). So it is
  # rewritten to a git dep on this monorepo pinned to `rev = <monorepo HEAD at
  # sync time>` and replaces the mirror's entry of the same name. The pin lives
  # in the manifest itself: the aien-harness mirror ships no Cargo.lock, so a
  # floating branch would build whatever main is at build time. Consequences:
  # sync from a commit that is pushed to the monorepo (cargo fetches it), and
  # `--check` reports disparity for a mirror with path deps after every
  # monorepo commit until it is re-synced. Only deps named by
  # manifest_append_deps are appended when missing; every other mirror line,
  # and every mirror with no path deps and no list entry, is left byte-identical.
  local mono_rev append_deps
  mono_rev=$(git -C "$SOVEREIGN_CORE" rev-parse HEAD) || {
    echo "[ERROR] Cannot resolve monorepo HEAD in $SOVEREIGN_CORE"
    return 1
  }
  append_deps=$(manifest_append_deps "$repo_name")
  local tmp_deps
  tmp_deps=$(mktemp)
  awk -v url="$MONOREPO_GIT_URL" -v rev="$mono_rev" -v allow="$append_deps" '
    function key_of(line,   k) { k = line; sub(/[[:space:]]*=.*/, "", k); gsub(/^[[:space:]]+/, "", k); return k }
    function emit_missing(   i) { for (i = 1; i <= n; i++) if (!(keys[i] in seen)) print lines[i]; emitted = 1 }
    function missing_count(   i, c) { c = 0; for (i = 1; i <= n; i++) if (!(keys[i] in seen)) c++; return c }
    BEGIN { na = split(allow, al, " "); for (i = 1; i <= na; i++) allowed[al[i]] = 1 }
    FNR == NR {
      if ($0 ~ /^[[:space:]]*\[/) { in_src = ($0 ~ /^[[:space:]]*\[dependencies\][[:space:]]*$/); next }
      if (in_src && $0 ~ /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/) {
        line = $0
        if (line ~ /path[[:space:]]*=[[:space:]]*"\.\.\//) {
          gsub(/path[[:space:]]*=[[:space:]]*"\.\.\/[^"]*"/, "git = \"" url "\", rev = \"" rev "\"", line)
          rewritten[key_of(line)] = line
        }
        if (key_of(line) in allowed) { n++; keys[n] = key_of(line); lines[n] = line }
      }
      next
    }
    /^[[:space:]]*\[/ {
      if (in_dst && !emitted) emit_missing()
      in_dst = ($0 ~ /^[[:space:]]*\[dependencies\][[:space:]]*$/)
      if (in_dst) has_dst = 1
      print; next
    }
    in_dst && /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/ {
      k = key_of($0); seen[k] = 1
      if (k in rewritten) { print rewritten[k]; next }
    }
    { print }
    END {
      if (in_dst && !emitted) emit_missing()
      else if (!has_dst && missing_count() > 0) { print ""; print "[dependencies]"; emit_missing() }
    }
  ' "$src_manifest" "$tmp" > "$tmp_deps"
  mv "$tmp_deps" "$tmp"

  # [dev-dependencies]: only keys named by manifest_append_dev_deps are touched.
  # Each is rewritten from its source line (path to the pinned git dep) and
  # replaces or is appended to the mirror's entry; mirrors with no list entry
  # stay byte-identical.
  local append_dev_deps
  append_dev_deps=$(manifest_append_dev_deps "$repo_name")
  if [[ -n "$append_dev_deps" ]]; then
    tmp_deps=$(mktemp)
    awk -v url="$MONOREPO_GIT_URL" -v rev="$mono_rev" -v allow="$append_dev_deps" '
      function key_of(line,   k) { k = line; sub(/[[:space:]]*=.*/, "", k); gsub(/^[[:space:]]+/, "", k); return k }
      function is_dev(line) { return line ~ /^[[:space:]]*\[dev-dependencies\][[:space:]]*$/ }
      function emit_missing(   i) { for (i = 1; i <= n; i++) if (!(keys[i] in seen)) print lines[i]; emitted = 1 }
      BEGIN { na = split(allow, al, " "); for (i = 1; i <= na; i++) allowed[al[i]] = 1 }
      FNR == NR {
        if ($0 ~ /^[[:space:]]*\[/) { in_src = is_dev($0); next }
        if (in_src && $0 ~ /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/ && key_of($0) in allowed) {
          line = $0
          gsub(/path[[:space:]]*=[[:space:]]*"\.\.\/[^"]*"/, "git = \"" url "\", rev = \"" rev "\"", line)
          n++; keys[n] = key_of(line); lines[n] = line; src_line[keys[n]] = line
        }
        next
      }
      /^[[:space:]]*\[/ {
        if (in_dst && !emitted) emit_missing()
        in_dst = is_dev($0)
        if (in_dst) has_dst = 1
        print; next
      }
      in_dst && /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/ {
        k = key_of($0)
        if (k in src_line) { seen[k] = 1; print src_line[k]; next }
      }
      { print }
      END {
        if (in_dst && !emitted) emit_missing()
        else if (!has_dst && n > 0) { print ""; print "[dev-dependencies]"; emit_missing() }
      }
    ' "$src_manifest" "$tmp" > "$tmp_deps"
    mv "$tmp_deps" "$tmp"
  fi

  if cmp -s "$tmp" "$dest_manifest"; then
    rm -f "$tmp"
    echo "[PARITY OK] Manifest $dest_manifest in parity (license = $license_id)"
    return 0
  fi
  if [[ "$is_check" == "true" ]]; then
    rm -f "$tmp"
    echo "[DISPARITY] $dest_manifest requires manifest update (license $license_id, version/edition or dependencies)"
    return 2
  fi
  mv "$tmp" "$dest_manifest"
  echo "[SYNCED] Manifest updated: $dest_manifest (license = $license_id)"
  return 0
}

check_unslop_invariants() {
  local target_dir="$1"
  local violations=0
  while IFS= read -r -d '' file; do
    if grep -q -P '[\x{2014}\x{2013}]' "$file" 2>/dev/null; then
      echo "[UNSLOP VIOLATION] Prohibited dashes detected in: $file"
      violations=$((violations + 1))
    fi
  done < <(find "$target_dir" -type f \( -name "*.rs" -o -name "*.md" -o -name "*.toml" \) -not -path "*/.git/*" -not -path "*/target/*" -print0)
  return $violations
}

process_repo() {
  local entry="$1"
  IFS=':' read -r cli_name crate_subpath publish_dirname <<< "$entry"

  local src_dir="$SOVEREIGN_CORE/$crate_subpath"
  if [[ ! -d "$src_dir" && "$crate_subpath" =~ ^basecamp/ ]]; then
    src_dir="$WORKSPACE/$crate_subpath"
  fi

  local dest_dir="$WORKSPACE/$publish_dirname"

  echo "============================================================"
  echo "Target: $publish_dirname (Source: $crate_subpath)"
  echo "============================================================"

  if [[ ! -d "$dest_dir" ]]; then
    echo "ERROR: Target publish directory does not exist: $dest_dir" >&2
    return 1
  fi

  if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: Source directory does not exist: $src_dir" >&2
    return 1
  fi

  local repo_has_disparity=0

  # Step 1: Check dirty target worktree
  check_dirty_worktree "$dest_dir" "$publish_dirname" || true

  # Step 2: Clean stray metadata files from source before sync
  if [[ "$CHECK_MODE" = false ]]; then
    find "$src_dir/src" -name "*.metadata.json" -delete 2>/dev/null || true
  fi

  # Step 3: Source directory synchronization and parity check
  if [[ "$CHECK_MODE" = true ]]; then
    local src_diff
    src_diff=$(diff -ru --exclude='*.metadata.json' --exclude='.git*' "$src_dir/src" "$dest_dir/src" 2>&1 || true)
    if [[ -n "$src_diff" ]]; then
      echo "[DISPARITY] src/ tree difference detected for $publish_dirname"
      if [[ "$VERBOSE" = true ]]; then
        echo "$src_diff"
      fi
      repo_has_disparity=1
    else
      echo "[PARITY OK] src/ tree in parity for $publish_dirname"
    fi

    # Check tests directory if present in source
    if [[ -d "$src_dir/tests" && -d "$dest_dir/tests" ]]; then
      local tests_diff
      tests_diff=$(diff -ru --exclude='.git*' "$src_dir/tests" "$dest_dir/tests" 2>&1 || true)
      if [[ -n "$tests_diff" ]]; then
        echo "[DISPARITY] tests/ tree difference detected for $publish_dirname"
        if [[ "$VERBOSE" = true ]]; then
          echo "$tests_diff"
        fi
        repo_has_disparity=1
      else
        echo "[PARITY OK] tests/ tree in parity for $publish_dirname"
      fi
    fi

    # Check schemas directory for spark-harness
    if [[ -d "$src_dir/schemas" && -d "$dest_dir/schemas" ]]; then
      local schemas_diff
      schemas_diff=$(diff -ru --exclude='.git*' "$src_dir/schemas" "$dest_dir/schemas" 2>&1 || true)
      if [[ -n "$schemas_diff" ]]; then
        echo "[DISPARITY] schemas/ tree difference detected for $publish_dirname"
        if [[ "$VERBOSE" = true ]]; then
          echo "$schemas_diff"
        fi
        repo_has_disparity=1
      else
        echo "[PARITY OK] schemas/ tree in parity for $publish_dirname"
      fi
    fi
  else
    echo "[SYNC] Synchronizing src/ from $src_dir/src to $dest_dir/src..."
    rsync -a --delete --exclude='*.metadata.json' --exclude='.git*' "$src_dir/src/" "$dest_dir/src/"

    if [[ -d "$src_dir/tests" && -d "$dest_dir/tests" ]]; then
      echo "[SYNC] Synchronizing tests/ from $src_dir/tests to $dest_dir/tests..."
      rsync -a --delete --exclude='.git*' "$src_dir/tests/" "$dest_dir/tests/"
    fi

    if [[ -d "$src_dir/schemas" && -d "$dest_dir/schemas" ]]; then
      echo "[SYNC] Synchronizing schemas/ from $src_dir/schemas to $dest_dir/schemas..."
      rsync -a --delete --exclude='.git*' "$src_dir/schemas/" "$dest_dir/schemas/"
    fi

    if [[ -f "$src_dir/evals.json" && -f "$dest_dir/evals.json" ]]; then
      cp "$src_dir/evals.json" "$dest_dir/evals.json"
    fi

    if [[ -f "$src_dir/workflows.json" && -f "$dest_dir/workflows.json" ]]; then
      cp "$src_dir/workflows.json" "$dest_dir/workflows.json"
    fi

    diff -ru --exclude='*.metadata.json' --exclude='.git*' "$src_dir/src" "$dest_dir/src"
    echo "[PARITY VERIFIED] src/ tree parity verified with diff -ru"
  fi

  # Step 4: Selective manifest synchronization
  if [[ -f "$src_dir/Cargo.toml" && -f "$dest_dir/Cargo.toml" ]]; then
    local manifest_code=0
    if sync_manifest "$src_dir/Cargo.toml" "$dest_dir/Cargo.toml" "$CHECK_MODE" "$cli_name"; then
      manifest_code=0
    else
      manifest_code=$?
    fi
    if [[ $manifest_code -eq 2 ]]; then
      repo_has_disparity=1
    elif [[ $manifest_code -ne 0 ]]; then
      echo "ERROR: Failed to process manifest for $publish_dirname" >&2
      return 1
    fi
  fi

  # Step 5: Unslop Invariants Scan
  if ! check_unslop_invariants "$dest_dir"; then
    echo "[WARN] Unslop violations found in $publish_dirname"
    if [[ "$CHECK_MODE" = true ]]; then
      repo_has_disparity=1
    fi
  fi

  # Step 6: Code Formatting
  if [[ "$CHECK_MODE" = false ]]; then
    echo "[CARGO FMT] Formatting code in $dest_dir..."
    cargo fmt --manifest-path "$dest_dir/Cargo.toml" --quiet 2>/dev/null || cargo fmt --manifest-path "$dest_dir/Cargo.toml"
  fi

  # Step 7: Compilation and Test Validation (in sync mode)
  if [[ "$CHECK_MODE" = false ]]; then
    echo "[CARGO CHECK] Verifying compilation in $dest_dir..."
    cargo check --manifest-path "$dest_dir/Cargo.toml" --quiet
    echo "[CARGO CHECK PASSED] $publish_dirname compiles cleanly."

    if [[ "$RUN_TESTS" = true ]]; then
      echo "[CARGO TEST] Running test suite in $dest_dir..."
      cargo test --manifest-path "$dest_dir/Cargo.toml" --quiet
      echo "[CARGO TEST PASSED] $publish_dirname passed all tests."
    fi
  fi

  return $repo_has_disparity
}

# Main execution loop
TOTAL_FAILED=0
PROCESSED_COUNT=0

for entry in "${REPOS[@]}"; do
  IFS=':' read -r cli_name crate_subpath publish_dirname <<< "$entry"

  if [[ "$ALL_REPOS" = false ]]; then
    clean_target="${TARGET_REPO%-publish}"
    clean_cli="${cli_name%-publish}"
    clean_pub="${publish_dirname%-publish}"
    if [[ "$TARGET_REPO" != "$cli_name" && "$TARGET_REPO" != "$publish_dirname" && "$clean_target" != "$clean_cli" && "$clean_target" != "$clean_pub" ]]; then
      continue
    fi
  fi

  PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
  if process_repo "$entry"; then
    repo_rc=0
  else
    repo_rc=$?
  fi
  if [[ $repo_rc -ne 0 ]]; then
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
  fi
done

if [[ $PROCESSED_COUNT -eq 0 ]]; then
  echo "ERROR: Repository '$TARGET_REPO' not recognized." >&2
  echo "Available repositories:" >&2
  for entry in "${REPOS[@]}"; do
    IFS=':' read -r cli_name _ publish_dirname <<< "$entry"
    echo "  - $cli_name (target: $publish_dirname)" >&2
  done
  exit 1
fi

echo "============================================================"
if [[ "$CHECK_MODE" = true ]]; then
  if [[ $TOTAL_FAILED -eq 0 ]]; then
    echo "[STATUS: ALL IN PARITY] All $PROCESSED_COUNT repositories match source state and workspace license."
    exit 0
  else
    echo "[STATUS: DISPARITY DETECTED] $TOTAL_FAILED of $PROCESSED_COUNT repositories exhibit drift or disparities."
    exit 1
  fi
else
  if [[ $TOTAL_FAILED -eq 0 ]]; then
    echo "[STATUS: SYNC SUCCESS] Successfully synchronized $PROCESSED_COUNT repositories."
    exit 0
  else
    echo "[STATUS: SYNC FAILED] $TOTAL_FAILED of $PROCESSED_COUNT repositories failed synchronization."
    exit 1
  fi
fi
