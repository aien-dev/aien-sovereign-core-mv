#!/usr/bin/env bash
# Exercise piped installation (install.sh fed to bash on stdin) from an
# arbitrary working directory without network access or real builds.
# git, cargo and uname are replaced by small stand-in scripts on PATH.
# Also checks the opt-in model server path (--with-server): it refuses the wrong
# machine and a release install before building anything, refuses a CPU stub
# build, and installs aien-serve-http plus aien-serve-27b from a native build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# aien-protocols is a git dependency pinned by revision in Cargo.lock, not a sibling checkout.
PROTOCOLS_REV="$(sed -n 's/^source = "git+https:\/\/github.com\/aien-dev\/aien-protocols?rev=\([0-9a-f]\{40\}\)#.*"$/\1/p' "$ROOT/Cargo.lock" | sort -u)"
[[ $(printf '%s\n' "$PROTOCOLS_REV" | grep -c .) -eq 1 ]] || { echo "FAIL: Cargo.lock must pin aien-protocols to exactly one revision" >&2; exit 1; }
SOURCE_REV="0123456789abcdef0123456789abcdef01234567"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/aien-test-install.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/mocks"

fail() {
    echo "FAIL: $*" >&2
    [[ -f "$WORK/install.log" ]] && sed 's/^/    install: /' "$WORK/install.log" >&2
    exit 1
}

# Stand-in git: clone creates a directory holding a Cargo.toml (plus, for the server
# cases, stand-in scripts/pins.sh and the real tools/serve-27b.sh), rev-parse reports
# the pinned protocols revision, config answers operator identity.
cat > "$WORK/mocks/git" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "\$MOCK_STATE/git.calls"
case "\$1" in
    clone)
        dest="\${!#}"
        mkdir -p "\$dest/scripts" "\$dest/tools"
        printf '[workspace]\n' > "\$dest/Cargo.toml"
        cp "$WORK/mocks/pins.sh" "\$dest/scripts/pins.sh"
        cp "$ROOT/tools/serve-27b.sh" "\$dest/tools/serve-27b.sh"
        ;;
    -C)
        case "\$3" in
            rev-parse) echo "$SOURCE_REV" ;;
        esac
        ;;
    rev-parse) echo "$SOURCE_REV" ;;
    config)
        if [[ "\${!#}" == user.name ]]; then echo "Fixture Operator"; else echo "fixture@example.test"; fi
        ;;
esac
exit 0
EOF

# Stand-in pins.sh: writes the env.sh the real one writes for --gpu, no network.
cat > "$WORK/mocks/pins.sh" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == --gpu ]] || { echo "pins.sh called without --gpu" >&2; exit 2; }
mkdir -p "$2/omega"
printf "export AIEN_OMEGA_DIR='%s/omega'\nexport AIEN_REQUIRE_NATIVE_LIBS=1\n" "$2" > "$2/env.sh"
echo "pins.sh --gpu $2" >> "$MOCK_STATE/pins.calls"
EOF

# Stand-in cargo: checks it runs inside a checkout with a locked build. The CLI build
# must include every installed package; the server build must see the native-only
# environment from env.sh. MOCK_SERVER_KIND=native|stub picks what --version reports.
cat > "$WORK/mocks/cargo" <<'EOF'
#!/usr/bin/env bash
[[ -f Cargo.toml ]] || { echo "installer did not locate checkout" >&2; exit 1; }
args=" $* "
[[ "$args" == *" --locked "* ]] || { echo "cargo build is not --locked" >&2; exit 1; }
mkdir -p target/release
if [[ "$args" == *" -p aien-serve-http "* ]]; then
    [[ "${AIEN_REQUIRE_NATIVE_LIBS:-}" == 1 && -n "${AIEN_OMEGA_DIR:-}" ]] \
        || { echo "server build without the pins env (AIEN_REQUIRE_NATIVE_LIBS, AIEN_OMEGA_DIR)" >&2; exit 1; }
    echo "server build" >> "$MOCK_STATE/cargo.calls"
    if [[ "${MOCK_SERVER_KIND:-native}" == native ]]; then e="native"; else e="STUB (not usable with --engine omega)"; fi
    printf '#!/bin/sh\nprintf "aien-serve-http 0.1.0\\nomega gpu engine: %s\\nqwen38 engine: %s\\n"\n' "$e" "$e" > target/release/aien-serve-http
    chmod 755 target/release/aien-serve-http
    exit 0
fi
for pkg in aien-cli spark-cockpit-rs cortex-rs spark-supervisor spark-debugger spark-harness spark-crumbs spark-aegis; do
    [[ "$args" == *" -p $pkg "* ]] || { echo "cargo build is missing package $pkg" >&2; exit 1; }
    printf '#!/bin/sh\nexit 0\n' > "target/release/$pkg"
    chmod 755 "target/release/$pkg"
done
echo "cli build" >> "$MOCK_STATE/cargo.calls"
EOF

# Stand-in uname for the server platform check (MOCK_UNAME_S, MOCK_UNAME_M).
cat > "$WORK/mocks/uname" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    -s) echo "${MOCK_UNAME_S:-Linux}" ;;
    -m) echo "${MOCK_UNAME_M:-aarch64}" ;;
    *) echo "${MOCK_UNAME_S:-Linux}" ;;
esac
EOF
chmod 755 "$WORK/mocks/git" "$WORK/mocks/cargo" "$WORK/mocks/pins.sh" "$WORK/mocks/uname"

# run_case <name> [install.sh args...]: fresh state dirs, piped install from an empty cwd.
# Extra environment comes from the caller (VAR=value run_case ...). Sets $status.
run_case() {
    local name="$1"; shift
    C="$WORK/case-$name"
    mkdir -p "$C/cwd" "$C/tmp" "$C/state"
    set +e
    (
        cd "$C/cwd"
        unset AIEN_SOURCE_DIR AIEN_REV AIEN_INSTALL_SERVER CARGO_TARGET_DIR AIEN_PINS_DIR
        # CASE_INSTALL_SERVER=1 tests the environment form of --with-server.
        [[ -z "${CASE_INSTALL_SERVER:-}" ]] || export AIEN_INSTALL_SERVER=1
        PATH="$WORK/mocks:$PATH" \
        MOCK_STATE="$C/state" \
        TMPDIR="$C/tmp" \
        AIEN_BIN_DIR="$C/bin" \
        AIEN_CONFIG_DIR="$C/config" \
        AIEN_INSTALL_NO_PROFILE=1 \
            bash -s -- "$@" < "$ROOT/install.sh"
    ) > "$WORK/install.log" 2>&1
    status=$?
    set -e
}

# 1. Default install: unchanged behaviour, no server.
run_case default
[[ $status -eq 0 ]] || fail "piped install.sh exited with status $status"
for bin in aien spark-cockpit cortex spark-supervisor spark-debugger spark-harness spark-crumbs spark-aegis; do
    [[ -x "$C/bin/$bin" ]] || fail "binary $bin was not installed"
done
[[ ! -e "$C/bin/aien-serve-http" && ! -e "$C/bin/aien-serve-27b" ]] || fail "default install must not install the model server"
[[ ! -e "$C/state/pins.calls" ]] || fail "default install must not fetch the omega pins"
[[ -f "$C/config/operator.toml" ]] || fail "operator.toml was not generated"
grep -q '^name = "Fixture Operator"$' "$C/config/operator.toml" || fail "operator.toml lacks git identity"
grep -q 'clone --quiet https://github.com/aien-dev/aien-sovereign-core-mv.git' "$C/state/git.calls" || fail "installer did not clone the core repository (aien-dev/aien-sovereign-core-mv)"
! grep -q "aien-protocols" "$C/state/git.calls" || fail "installer must not clone aien-protocols (it is a pinned git dependency in Cargo.lock)"
[[ -z "$(ls -A "$C/tmp")" ]] || fail "installer left its temporary checkout behind"
[[ -z "$(ls -A "$C/cwd")" ]] || fail "installer wrote into the caller's working directory"
grep -q "aien compose desk-key --create 1" "$WORK/install.log" || fail "quickstart must list the desk-key step before aien start (sc#414)"
! grep -q "aien harness" "$WORK/install.log" || fail "quickstart must not offer aien harness, which is not shipped (sc#414)"
echo "PASS: piped installer bootstraps a checkout and installs the CLI from an arbitrary cwd"

# 2. --with-server on a machine that is not aarch64 Linux: refused before anything is built.
MOCK_UNAME_S=Linux MOCK_UNAME_M=x86_64 run_case wrong-arch --with-server
[[ $status -ne 0 ]] || fail "--with-server on x86_64 must fail"
grep -q "only runs on an NVIDIA DGX Spark" "$WORK/install.log" || fail "--with-server on x86_64 must say why"
[[ ! -e "$C/state/cargo.calls" && ! -e "$C/state/git.calls" ]] || fail "--with-server on x86_64 must stop before cloning or building"
echo "PASS: --with-server refuses a machine that is not aarch64 Linux, before any build"

# 3. --with-server with a release tag: refused (no release carries the server).
AIEN_RELEASE_TAG=v0.0.0 run_case release-tag --with-server
[[ $status -ne 0 ]] || fail "--with-server with AIEN_RELEASE_TAG must fail"
grep -q "cannot be combined with AIEN_RELEASE_TAG" "$WORK/install.log" || fail "--with-server with a release tag must say why"
echo "PASS: --with-server refuses a release install"

# 4. A build that linked only the CPU stub: refused loudly, nothing installed for the server.
AIEN_TEST_SERVER_SKIP_HW_CHECKS=1 MOCK_SERVER_KIND=stub run_case stub --with-server
[[ $status -ne 0 ]] || fail "a CPU stub server build must fail the install"
grep -q "only the CPU stub" "$WORK/install.log" || fail "a CPU stub server build must say so"
[[ ! -e "$C/bin/aien-serve-http" && ! -e "$C/bin/aien-serve-27b" ]] || fail "a CPU stub server must not be installed"
echo "PASS: --with-server refuses to install a CPU stub build"

# 5. Native build (AIEN_INSTALL_SERVER=1 form): pins fetched with --gpu, server and start script installed.
AIEN_TEST_SERVER_SKIP_HW_CHECKS=1 MOCK_SERVER_KIND=native CASE_INSTALL_SERVER=1 run_case native
[[ $status -eq 0 ]] || fail "native server install exited with status $status"
grep -q -- "pins.sh --gpu" "$C/state/pins.calls" || fail "server install must run scripts/pins.sh --gpu"
grep -q "server build" "$C/state/cargo.calls" || fail "server install must build aien-serve-http"
[[ -x "$C/bin/aien-serve-http" ]] || fail "aien-serve-http was not installed"
[[ -x "$C/bin/aien-serve-27b" ]] && cmp -s "$C/bin/aien-serve-27b" "$ROOT/tools/serve-27b.sh" || fail "aien-serve-27b is not tools/serve-27b.sh"
[[ -x "$C/bin/aien" ]] || fail "server install must still install the CLI"
[[ -z "$(ls -A "$C/tmp")" ]] || fail "server install left its temporary checkout behind"
echo "PASS: --with-server (AIEN_INSTALL_SERVER=1) builds with the pinned GPU env and installs the server and its start script"
