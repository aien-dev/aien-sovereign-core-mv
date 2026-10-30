# aien-sovereign-core

The Rust workspace for the AIEN host runtime on Linux (DGX Spark target): the `aien` command line, a single-process runtime daemon, the inference ABI and scheduler, and the supporting services around them. This file lists what exists and who owns what. It makes no claim that any part is complete or released; the README states the release state, and `docs/audits/` in aien-architecture holds the integration audit.

Ownership: this repository has no `CODEOWNERS` file, so per-component owners are not recorded here (UNVERIFIED who owns each crate). Cross-repo plans and decisions are owned by aien-architecture (`PLAN_AUTHORITY.md`, `CURRENT_EXECUTION_PLAN.md`, ADRs). Decisions made in this repo live in `docs/adr/`.

## Language

**Runtime daemon**:
The long-running process started by `aien start` (it runs as `aien --daemon`). It owns the model, the KV memory and the socket the command line talks to. Code: `crates/aien-runtime`, started from `crates/aien-cli/src/commands.rs`.
_Avoid_: server, service

**Approval desk**:
The operator's approval step for effects. It is required by default; a desk key created by `aien compose desk-key --create 1` must exist before the daemon starts. Code: `aien-runtime/src/compose/bridge.rs`, `approved_auth.rs`.
_Avoid_: login, auth

**Objective record**:
The operator's written objective, stored as `objectives/<id>.json` under the provenance folder, re-read by the daemon when a grant is bound to it. Code: `aien-runtime/src/objective.rs`, `aien-cli/src/objective.rs`.
_Avoid_: task, ticket

**Compose home**:
The state folder holding the desk key and composition state (`AIEN_COMPOSE_DIR`, else under `AIEN_RUNTIME_STATE_DIR`, else XDG state, else `~/.local/state/aien-runtime/compose`).

**Candidate**:
The release candidate this tree may be released as, named in `release/candidate.toml` with its model digests and source pins. `omega.lock` pins the omega commit whose composition library is linked; `cudapp.lock` pins the CUDApp commit the GPU engine is built from. Changed only with aien-architecture's candidate manifest.

**Generation controller**:
The one module that runs a generation for any model: it keeps the sequence ledger (history and position), reuses a prefix, rewinds, exports and restores the KV cache, picks each token, applies constraints, stops and streams. Written once; drives a forward pass through the `ForwardPass` trait (ADR 0004). Planned code: `aien-inference-abi`, type `GenerationController<M>`.
_Avoid_: engine, sequence driver, inference loop, runtime (that word means the daemon)

**Forward pass**:
One model's implementation behind the seam: its weights, layer maths, kernels, attention path and cache, prefill strategy, decode graph, output head, traces and mutants. The 30B MoE, the 8B dense and the host fake are each one forward pass. It never loops over tokens itself.
_Avoid_: engine, model (ambiguous with the weights file), backend

**KV geometry**:
The numbers that fix the KV cache's layout and payload bytes (layers, query heads, KV heads, head width, rope theta, block size), produced from a model's config and read by every cache constructor and by persistence. Type `KvGeometry` (ADR 0004). Model-independent: no cache code reads `Qwen3MoeConfig` directly.
_Avoid_: config, shape, dims

**Head**:
The pending output a forward call leaves behind for one position, from which the next token is chosen. Read once, as either the chip's own argmax or the full logits. One after prefill, one per decoded token.
_Avoid_: pending token, logits buffer, output (overloaded)

**Step rule**:
What a constraint allows at one decode step: anything, an allowed set of token ids, one forced token, or finish. Applied by the generation controller between the head and the sampler. Type `StepRule` (ADR 0004).
_Avoid_: mask (that is one way to apply a rule, not the rule), grammar

**Host fake**:
The test-only forward pass that runs with no chip and no weights, on a real host KV cache, with tokens that depend on every stored KV row. Used to test the generation controller end to end, including constraints and persistence. Type `EchoChainModel` (ADR 0004).
_Avoid_: mock, stub (the serve-http stub engine is a different thing)

## Components (from crate descriptions)

Runtime and inference
- `aien-runtime`: single-process runtime spine (daemon, compose bridge, objective and desk logic).
- `aien-cli`: the `aien` command line and terminal agent harness.
- `aien-inference-abi`, `aien-abi-core`, `aien-inference-runtime`, `aien-inference-service`: inference ABI, backend trait, weights loader, service protocols.
- `aien-kv-cache`, `aien-scheduler`: paged KV cache and continuous-batching scheduler.
- `aien-omega-gpu` (and the other `aien-omega-*` GPU crates): bindings to the Omega GPU engine built from aien-dev/CUDApp (pinned by `cudapp.lock`). `aien-omega-compose`: binding to omega composition (pinned by `omega.lock`).
- `aien-platform`, `aien-platform-linux`: platform ABI and the Linux / GB10 driver cell.

Identity, memory, evidence
- `aien-allen`, `aien-allen-memory`, `aien-allen-profile`: ALLEN identity gate, scoped memory, persona profile.
- `cortex-rs`, `cortex-encoder-rs`, `cortex-path-m1`: Cortex memory engine, embeddings, a read-only retrieval experiment.
- `aien-capability`, `aien-mcp`, `aien-security`: tool and effect descriptors, MCP sessions, security abstractions.
- `aien-proof`, `aien-closure`, `aien-trace`, `aien-dual-observation`, `aien-replay`, `aien-campaign-verifier`, `crumbs`: test board, closure check, correlation trace, observation snapshot, transcript reader, campaign verifier, curriculum substrate.

Services and tools
- `spark-*` crates (cockpit, supervisor, debugger, harness, aegis, hive, dream, mail, discord hub, harvester, inquisitor, adapters, crumbs) and `rad-id-sync`, `aien-switchboard`, `aien-drive`: supporting services and tools; see each crate's Cargo description.
- `tools/aien-verify`: independent verifier of the whole-system end-to-end run folders; standard library only. Tests: `tools/aien-verify/tests/run.sh` (run in CI).

## Where things are

- Release machinery: `install.sh`, `scripts/check-release-candidate.sh`, `scripts/package-release.sh`, `docs/release/`.
- First-run steps for an operator: `docs/FIRST_RUN.md`.
- End-to-end and campaign material: `scripts/whole_system_e2e.sh`, `docs/campaigns/`.
