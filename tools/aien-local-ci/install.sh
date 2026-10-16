#!/usr/bin/env bash
# install.sh: idempotent host setup for the sovereign local pipeline sandbox (lane INFRA, contracts C3/C8/A2).
# Run as drakestapleton; uses sudo -n inside. Safe to re-run: every step checks before it acts.
#
# What it sets up on the host:
#   * system user aien-run (home /var/lib/aien-run root:aien-run 0750 + ACL u:drakestapleton:r-x, nologin,
#     groups video+render). The home and everything a later run could read config from is root-owned, so a
#     check cannot plant ~/.cargo/config.toml, swap the registry or replace the cargo proxies (review finding 1).
#   * /var/lib/aien-run/{cargo,target,wt,mirror,rustup} with the ownership described in README.md
#   * cargo registry + git cache seeded from ~/.cargo (root-owned, read-only), toolchain copied from ~/.rustup
#     (root-owned, read-only), rustup proxies in /var/lib/aien-run/cargo/bin (no second rustup install)
#   * /usr/local/bin/crumb (root 0755), built from aien-architecture at CRUMB_COMMIT below (the trusted copy
#     on the sandbox PATH; never the operator's ~/.local/bin copy, never a copy from a tree under test)
#   * /usr/local/sbin/aien-run-exec (root 0755) and /etc/sudoers.d/aien-local-ci (validated with visudo -c)
#   * ~/.local/state/aien-local-ci/{trigger,evidence,bundles} for the trusted side
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_USER=aien-run
RUN_HOME=/var/lib/aien-run
OPERATOR=${AIEN_OPERATOR:-drakestapleton}
OP_HOME=$(getent passwd "$OPERATOR" | cut -d: -f6)
TOOLCHAIN=${AIEN_TOOLCHAIN:-1.98.1-aarch64-unknown-linux-gnu}
STATE="$OP_HOME/.local/state/aien-local-ci"
# crumb source: aien-architecture main at this commit (2026-10-11, on GitHub main), file tools/crumb/crumb.c
# (git blob 841f5317480b6643086061585c4309097975d8f9). Built with gcc 13.3.0 the binary's sha256 is
# 22015e6b156e038ecea26ca8cf5c03f944d0c526282c4314fc5891751ffa67b0 (same as the operator's copy that day).
CRUMB_COMMIT=94fffbcf88492c5a1c54211b5126c8a6c370a98d
CRUMB_BLOB=841f5317480b6643086061585c4309097975d8f9
ARCH_CLONE=${AIEN_ARCH_CLONE:-$OP_HOME/workspace/aien-architecture}

say() { printf '[install] %s\n' "$*"; }
S() { sudo -n "$@"; }

[[ $(id -un) == "$OPERATOR" ]] || { echo "run as $OPERATOR" >&2; exit 1; }
sudo -n true 2>/dev/null || { echo "passwordless sudo required" >&2; exit 1; }
# private scratch dir (never a predictable /tmp name: the sudoers line is staged here before install)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# 1. user and groups (device groups: /dev/dri/card0 is root:video, /dev/dri/renderD128 is root:render,
#    /dev/kvm is root:kvm 0660 for the qemu profile, /dev/nvidia* are 0666 root:root so they need no group).
#    The cgroup device policy still keeps /dev/kvm closed for the host profile.
if ! getent passwd "$RUN_USER" >/dev/null; then
  S useradd --system --home-dir "$RUN_HOME" --create-home --shell /usr/sbin/nologin --user-group \
    --groups video,render "$RUN_USER"
  say "created user $RUN_USER"
else
  say "user $RUN_USER exists"
fi
for g in video render kvm; do
  getent group "$g" >/dev/null || continue
  id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx "$g" || S usermod -aG "$g" "$RUN_USER"
done
# home: root-owned, group aien-run (traverse), operator via ACL; skeleton dotfiles removed (nothing may be
# sourced or read from an aien-run-writable file in the home).
S chown root:"$RUN_USER" "$RUN_HOME"
S chmod 0750 "$RUN_HOME"
S setfacl -m "u:$OPERATOR:r-x" "$RUN_HOME"
for f in .bashrc .profile .bash_logout .config Desktop; do
  if S test -e "$RUN_HOME/$f"; then S rm -rf "${RUN_HOME:?}/$f"; say "removed skeleton $RUN_HOME/$f"; fi
done

# 2. directory layout (cargo/ and target/ root-owned; only target/<repo>/{pr,main} belong to aien-run)
S install -d -o root -g root -m 0755 "$RUN_HOME/cargo" "$RUN_HOME/target" "$RUN_HOME/models"
S chown root:root "$RUN_HOME/cargo" "$RUN_HOME/target"; S chmod 0755 "$RUN_HOME/cargo" "$RUN_HOME/target"
for f in "$RUN_HOME"/cargo/.package-cache "$RUN_HOME"/cargo/.package-cache-mutate "$RUN_HOME"/cargo/.global-cache; do
  if S test -e "$f"; then S chown root:root "$f"; S chmod 0644 "$f"; fi
done
for r in "$RUN_HOME"/target/*/; do
  [[ -d $r ]] || continue
  S chown root:root "$r"; S chmod 0755 "$r"
  for c in pr main; do S install -d -o "$RUN_USER" -g "$RUN_USER" -m 0755 "$r$c"; done
done
S install -d -o "$OPERATOR" -g "$OPERATOR" -m 0755 "$RUN_HOME/wt" "$RUN_HOME/mirror"
S install -d -o root -g root -m 0755 "$RUN_HOME/rustup" "$RUN_HOME/rustup/toolchains" "$RUN_HOME/rustup/update-hashes" "$RUN_HOME/rustup/downloads" "$RUN_HOME/rustup/tmp"
install -d -m 0755 "$STATE/trigger" "$STATE/evidence" "$STATE/bundles"

# 3. caches: registry + git cache copied once from the operator's ~/.cargo, then root-owned read-only.
seed_ro() { # seed_ro SRC DST
  local src=$1 dst=$2
  if [[ ! -e $dst ]]; then
    [[ -d $src ]] || { say "skip $dst (no $src)"; return; }
    S rm -rf "$dst.tmp"
    S cp -a "$src" "$dst.tmp"
    S mv "$dst.tmp" "$dst"
    say "seeded $dst from $src"
  fi
  S chown -R root:root "$dst"
  S chmod -R a+rX,u-w,go-w "$dst"
}
seed_ro "$OP_HOME/.cargo/registry" "$RUN_HOME/cargo/registry"
seed_ro "$OP_HOME/.cargo/git" "$RUN_HOME/cargo/git"
seed_ro "$OP_HOME/.rustup/toolchains/$TOOLCHAIN" "$RUN_HOME/rustup/toolchains/$TOOLCHAIN"

# 4. rustup: the operator's rustup binary acts as the proxy (cargo/rustc/... are symlinks to it), with
#    RUSTUP_HOME=/var/lib/aien-run/rustup holding only settings.toml + the copied toolchain. No rustup-init,
#    no downloads: rust-toolchain.toml pins resolve against the copied toolchain only.
S install -d -o root -g root -m 0755 "$RUN_HOME/cargo/bin"
S install -o root -g root -m 0755 "$OP_HOME/.cargo/bin/rustup" "$RUN_HOME/cargo/bin/rustup"
for p in cargo rustc rustdoc rustfmt cargo-fmt cargo-clippy clippy-driver rust-gdb rust-lldb; do
  S ln -sfn rustup "$RUN_HOME/cargo/bin/$p"
done
printf 'version = "12"\ndefault_toolchain = "%s"\nprofile = "minimal"\n\n[overrides]\n' "$TOOLCHAIN" > "$tmp/settings.toml"
S install -o root -g root -m 0644 "$tmp/settings.toml" "$RUN_HOME/rustup/settings.toml"
S chown root:root "$RUN_HOME/rustup"; S chmod 0755 "$RUN_HOME/rustup"

# 4b. git safe.directory for aien-run: worktree roots are operator-owned (sticky), so git must be told they are safe.
#     git 2.43 has no path globs for safe.directory, so it is "*"; aien-run can only reach /var/lib/aien-run anyway.
printf "[safe]\n\tdirectory = *\n" > "$tmp/gitconfig"
S install -o root -g root -m 0644 "$tmp/gitconfig" "$RUN_HOME/.gitconfig"

# 4c. trusted crumb on the sandbox PATH, built from the pinned aien-architecture commit (source read straight
#     from the git object store, so the clone's checked-out branch does not matter).
git -C "$ARCH_CLONE" cat-file -e "$CRUMB_COMMIT^{commit}" 2>/dev/null \
  || git -C "$ARCH_CLONE" fetch --quiet origin main \
  || { echo "cannot find $CRUMB_COMMIT in $ARCH_CLONE (set AIEN_ARCH_CLONE)" >&2; exit 1; }
blob=$(git -C "$ARCH_CLONE" rev-parse "$CRUMB_COMMIT:tools/crumb/crumb.c")
[[ $blob == "$CRUMB_BLOB" ]] || { echo "crumb.c blob $blob != expected $CRUMB_BLOB" >&2; exit 1; }
git -C "$ARCH_CLONE" cat-file blob "$blob" > "$tmp/crumb.c"
cc -O2 -Wall -std=c11 -o "$tmp/crumb" "$tmp/crumb.c"
if ! cmp -s "$tmp/crumb" /usr/local/bin/crumb 2>/dev/null; then
  S install -o root -g root -m 0755 "$tmp/crumb" /usr/local/bin/crumb
  say "installed /usr/local/bin/crumb from aien-architecture $CRUMB_COMMIT (sha256 $(sha256sum < "$tmp/crumb" | cut -c1-16)...)"
else
  say "/usr/local/bin/crumb current (aien-architecture $CRUMB_COMMIT)"
fi

# 5. exec contract + sudoers
S install -o root -g root -m 0755 "$here/aien-run-exec" /usr/local/sbin/aien-run-exec
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/aien-run-exec\n' "$OPERATOR" > "$tmp/sudoers"
S visudo -cf "$tmp/sudoers" >/dev/null
S install -o root -g root -m 0440 "$tmp/sudoers" /etc/sudoers.d/aien-local-ci
S visudo -c >/dev/null && say "sudoers OK"

# 6. cron tick (operator-owned; the crontab line itself is the operator's: */2 * * * * $HOME/.local/lib/aien-local-ci/cron-tick.sh)
LIB=${AIEN_CI_LIB:-$OP_HOME/.local/lib/aien-local-ci}
install -d -m 0755 "$LIB"
if ! cmp -s "$here/cron-tick.sh" "$LIB/cron-tick.sh" 2>/dev/null; then
  install -m 0755 "$here/cron-tick.sh" "$LIB/cron-tick.sh"
  say "installed $LIB/cron-tick.sh"
else
  say "$LIB/cron-tick.sh current"
fi

say "done. verify with: $here/tests/run.sh"
