---
name: self-improvement
description: Autonomous self-improvement protocol for AIEN: isolated git worktree sandboxing, Microsoft SkillOpt optimization, JSpace dynamic dream cycles, and headless browser mirror self-testing via Chrome DevTools Protocol.
---

# Autonomous Self-Improvement and Browser Mirror Protocol

Use this skill when modifying AIEN core code, improving skills via Microsoft SkillOpt, inspecting the JSpace dream cycle, or verifying live UI and streaming behavior using the headless browser mirror harness.

## Core Pillars of Sovereign Self-Improvement

1. Never Break Live Harnesses:
   Live systemd services (spark-cockpit.service, ~/.local/bin/aien) must never be edited directly. All exploratory code modifications, refactorings, and experiments must take place inside an isolated git worktree sandbox.
2. Deterministic IPC over Flaky Webhooks:
   Browser self-testing uses Chrome DevTools Protocol (CDP) WebSocket on port 9222 and local Unix Domain Sockets (UDS). Webhooks are strictly forbidden for local control loops.
3. Dual-Seat Mirror Auditing:
   AIEN tests his live Cockpit UI by driving headless Chrome as an adversarial auditor, verifying page hydration, live token streaming via Axum SSE, tactile action buttons, and tab transitions.

## 1. The Autonomous Sandbox Subsystem (aien --sandbox)

When tasked with improving or modifying your own codebase (aien-sovereign-core):

### Step 1: Initialize the Worktree Sandbox
Run the sandbox init command to branch main into an isolated worktree at ~/workspace/aien-sandbox:
```bash
aien --sandbox init auto-improve
```
Or directly via git:
```bash
git -C ~/workspace/aien-sovereign-core worktree add ~/workspace/aien-sandbox -b auto-improve
```

### Step 2: Make Edits and Run Tests
Apply edits inside ~/workspace/aien-sandbox.
Verify compiler checks and test passes:
```bash
aien --sandbox test
```
Or directly:
```bash
cargo check --tests --manifest-path ~/workspace/aien-sandbox/Cargo.toml
```

### Step 3: Inspect Sandbox Status
Check git diff and untracked changes within the worktree:
```bash
aien --sandbox status
```

### Step 4: Promote Verified Edits or Clean Up
If tests pass, promote the branch into main and rebuild the release binary:
```bash
aien --sandbox promote "feat(core): verified autonomous self-improvement"
```
If tests fail or the experiment is aborted, clean the worktree:
```bash
aien --sandbox clean
```

## 2. Microsoft SkillOpt Self-Evolution Loop

To optimize a sovereign skill against a held-out benchmark split using the local model seat on port 18006:

```bash
bash ~/atlas-skillopt-stage.sh
```

### Key Operational Invariants
1. Endpoint and Model:
   The local model seat on port 18006 registers as atlas-lightning-omni. Both OPTIMIZER_DEPLOYMENT and TARGET_DEPLOYMENT must be set to atlas-lightning-omni.
2. Split Size Alignment:
   When running mini-batch iterations, pass --limit N matching train.train_size=N to satisfy dataloader size assertions.
3. Worker Throttling:
   Set env.workers to 2 and env.exec_timeout to 300 in the dataset YAML configuration to avoid saturating the model seat batch queue.
4. Output Artifacts:
   Produces best_skill.md and automatically deploys the winning patch to ~/skills/atlas-skillopt/SKILL.md.

## 3. JSpace Dynamic Dream Cycle (Eden Loop)

The autonomous consolidation loop operates continuously in the background on port 18085 during operator idle windows.

### Status and Health Probing
```bash
curl -s http://127.0.0.1:18085/health
curl -s http://127.0.0.1:18085/pulse
```
Outputs heartbeat BPM, coherence score, secret hygiene posture, and territory maps.

### Manual Cycle Trigger and Integration Verification
The Python dream engine (basecamp/aien-dream/dream_engine.py) was removed from the sovereign build (sovereign-core #181). The `/dream` command in `aien` now says so and points to the native `spark-dream` binary for dream telemetry. A manual cycle trigger and the scoring/storage verification are unavailable until a native replacement lands.

## 4. Headless Browser Mirror and Mentor Harness

AIEN drives a headless Chrome over the Chrome DevTools Protocol from Rust (`crates/aien-cli/src/browser_mirror.rs`, sovereign-core #180). No Python and no extra crates: the websocket client and the base64 decoder live in the module. The same code runs behind the `browser` tool, `aien --browser`, and the standalone binary:

```
aien-browser-mirror --test                          # Cockpit self-test: report.json + 3 screenshots
aien-browser-mirror --mentor "<prompt>"             # one chat prompt, waits for the reply, screenshot
aien-browser-mirror --screenshot [file.png] --url U # screenshot of any page
aien-browser-mirror --eval "<javascript>" --url U   # evaluate an expression on any page
```

Defaults: `--url http://127.0.0.1:18095` (override with `--url` or `AIEN_COCKPIT_URL`), `--out-dir ~/basecamp/ui-tests`, Chrome found as `google-chrome`, `chromium` or `AIEN_CHROME`. Exit status 0 means PASSED, COMPLETED or ok.

On Linux with pidfd support, each run starts Chrome with an `AIEN_BROWSER_MIRROR_TAG` environment tag and leaves it in the caller's process group, so a Ctrl-C or a harness group kill that ends the caller also reaches Chrome. Linux cleanup uses only pidfd signaling, never numeric PID or process-group signals. After pinning each candidate, it revalidates an exact environment tag or private-profile argument. Chrome clears tags and rewrites argv on some tasks, so a rewritten title requires a Chrome image (`chrome`, `chromium` or `chromium-browser`), its exact executable prefix and a complete profile flag, not a substring or prefix match. Profiles are created exclusively with mode 0700. Tags and profile arguments distinguish these owned fixtures; they are not an authentication boundary against same-UID processes deliberately copying them. A cleanup timeout emits a diagnostic and retains the private profile rather than blocking indefinitely or deleting potentially active browser data. Unsupported Linux pidfd signaling refuses the spawn. On other Unix platforms Chrome leads its own process group and cleanup signals that group; detached-helper cleanup there is not qualified. Chrome runs with `--disable-gpu --use-gl=disabled --disable-software-rasterizer`: without the last two flags, Chrome 153 on aarch64 falls back to a SwiftShader GPU process that never answers a screenshot (sovereign-core #661).

The self-test checks hydration (title, `#chat-input`, navigation controls), a live chat reply sent through `#btn-send` that stops growing, and every `[data-view]` or `[data-tab]` control switching to a present view. Screenshots land as `cockpit_init.png`, `cockpit_chat_stream.png` and `cockpit_actions.png`, next to `report.json`. The mentor run writes `cockpit_mentor_response.png` and `mentor_response.json` (COMPLETED when the reply is longer than 50 characters). The Cockpit must be running for the self-test; against a dead port the report says FAILED with `chat_input: false`.

## 5. Cortex Memory Commitment

Record a durable procedure receipt to Spark Cortex from inside `aien`, using the built-in command (name, a vertical bar, then the content):
```
/cortex write sovereign_self_improvement_loop | Verified self-improvement procedure: isolated git worktree in ~/workspace/aien-sandbox, JSpace dream telemetry via spark-dream, and Microsoft SkillOpt validation gate.
```
The command reads CORTEX_TOKEN from the AIEN vault and prints the receipt id; check it with `/cortex search sovereign_self_improvement_loop`.
