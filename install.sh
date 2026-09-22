#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# AIEN Sovereign Stack Universal Multi-OS Developer Installer
# Single-command install for the entire sovereign monorepo ecosystem.
# Free and open-source for humanity. Zero telemetry. Zero subscriptions.
# ==============================================================================

echo "=================================================================="
echo "      ⚡ AIEN Sovereign Stack: Universal Developer Installer       "
echo "=================================================================="

OS="$(uname -s | tr "[:upper:]" "[:lower:]")"
ARCH="$(uname -m)"

case "$ARCH" in
    x86_64|amd64) TARGET_ARCH="x86_64" ;;
    aarch64|arm64) TARGET_ARCH="aarch64" ;;
    *) echo "Error: Unsupported CPU architecture: $ARCH"; exit 1 ;;
esac

case "$OS" in
    linux) TARGET_OS="linux" ;;
    darwin) TARGET_OS="macos" ;;
    *) echo "Error: Unsupported operating system: $OS"; exit 1 ;;
esac

echo "[*] Detected Platform: $TARGET_OS ($TARGET_ARCH)"

# Verify Rust Toolchain
if ! command -v cargo >/dev/null 2>&1; then
    echo "[!] Cargo not found. Installing Rust toolchain via rustup..."
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/sovereign"
IMPRINT_DIR="$CONFIG_DIR/imprints/en2"

mkdir -p "$BIN_DIR" "$CONFIG_DIR" "$IMPRINT_DIR"

# Ensure ~/.local/bin is in PATH for current and future shells
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    export PATH="$BIN_DIR:$PATH"
fi

for RC in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile"; do
    if [ -f "$RC" ]; then
        if ! grep -q "\$HOME/.local/bin" "\$RC"; then
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$RC"
        fi
    fi
done

echo "[*] Building Core Monorepo Binaries (High-Performance Release Mode)..."
echo "    - aien-cli (Universal Terminal CLI & Orchestrator)"
echo "    - spark-cockpit-rs (3D Sovereign Glass Cockpit Server)"
echo "    - cortex-rs (Canonical Memory & Vector Store)"
echo "    - spark-supervisor (Resilient Process Supervisor)"
echo "    - spark-debugger (Zero-Leak Health & Telemetry Auditor)"
echo "    - spark-harness (5-Layer Verification Engine)"
echo "    - spark-crumbs (Blake3 Scent & Action Vector Tracker)"
echo "    - spark-aegis (Defensive Boundary & Containment Engine)"

cargo build --release -p aien-cli -p spark-cockpit-rs -p cortex-rs -p spark-supervisor -p spark-debugger -p spark-harness -p spark-crumbs -p spark-aegis

echo "[*] Installing Binaries to $BIN_DIR..."
cp target/release/aien-cli "$BIN_DIR/aien"
cp target/release/spark-cockpit-rs "$BIN_DIR/spark-cockpit"
cp target/release/cortex-rs "$BIN_DIR/cortex"
cp target/release/spark-supervisor "$BIN_DIR/spark-supervisor"
cp target/release/spark-debugger "$BIN_DIR/spark-debugger"
cp target/release/spark-harness "$BIN_DIR/spark-harness"
cp target/release/spark-crumbs "$BIN_DIR/spark-crumbs"
cp target/release/spark-aegis "$BIN_DIR/spark-aegis"
chmod +x "$BIN_DIR/aien" "$BIN_DIR/spark-cockpit" "$BIN_DIR/cortex" "$BIN_DIR/spark-supervisor" "$BIN_DIR/spark-debugger" "$BIN_DIR/spark-harness" "$BIN_DIR/spark-crumbs" "$BIN_DIR/spark-aegis"

echo "[+] Binaries successfully linked:"
echo "    - aien -> $BIN_DIR/aien"
echo "    - spark-cockpit -> $BIN_DIR/spark-cockpit"
echo "    - cortex -> $BIN_DIR/cortex"
echo "    - spark-supervisor -> $BIN_DIR/spark-supervisor"
echo "    - spark-debugger -> $BIN_DIR/spark-debugger"
echo "    - spark-harness -> $BIN_DIR/spark-harness"
echo "    - spark-crumbs -> $BIN_DIR/spark-crumbs"
echo "    - spark-aegis -> $BIN_DIR/spark-aegis"

echo "[*] Initializing Sovereign Operator Profile..."
if [ ! -f "$CONFIG_DIR/operator.toml" ]; then
    GIT_NAME="$(git config user.name 2>/dev/null || echo "Sovereign Operator")"
    GIT_EMAIL="$(git config user.email 2>/dev/null || echo "operator@local")"
    
    cat << CFG > "$CONFIG_DIR/operator.toml"
[operator]
name = "$GIT_NAME"
email = "$GIT_EMAIL"
handle = "operator"
sign_commits = false

[engine]
mode = "max"
    api_base_url = "http://127.0.0.1:18007/v1"
api_key = ""
model_id = "atlas-lightning-omni"
    max_port = 18006
    gateway_port = 18007
    context_window = 49152
temperature = 0.7
CFG
    echo "[+] Generated default configuration at $CONFIG_DIR/operator.toml"
else
    echo "[+] Existing configuration found at $CONFIG_DIR/operator.toml"
fi

echo "[*] Installing Free EN2 Trinity Imprint..."
if [ -d "imprints/en2-trinity" ]; then
    cp -r imprints/en2-trinity/* "$IMPRINT_DIR/"
    echo "[+] EN2 Trinity Imprint installed locally to $IMPRINT_DIR"
fi

echo "=================================================================="
echo "  ⚡ Installation Complete. The Entire AIEN Stack is Ready."
echo "=================================================================="
echo "  Primary Command: aien"
echo ""
echo "  Quickstart:"
echo "    aien start      # Starts Cortex, Cockpit, and background daemons"
echo "    aien status     # Real-time health, ports, and memory status"
echo "    aien cockpit    # Launches Sovereign Glass Cockpit in browser"
echo "    aien chat       # Interactive sovereign terminal pairing"
echo "    aien harness    # Run 5-layer verification checks"
echo "    aien aegis      # Run defensive boundary security audit"
echo "    aien doctor     # System diagnostic and TPM key vault audit"
echo "=================================================================="
