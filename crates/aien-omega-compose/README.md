# aien-omega-compose

Rust binding to omega COMPOSITION-2 through the rxc_host C ABI
(`src/runtime/rxc_host_abi.h`, archive `build/librx_compose.a`, make target
in omega `mk/rx_compose_lib.mk`). NEXT-PHASE-1 cut 1b.

## Building against omega

The library is built from an omega checkout whose HEAD must equal `omega.lock`
(the same pin aien-omega-gpu uses; on 2026-10-19 `omega.lock` names omega
4799f9eb). Omega has shipped `librx_compose.a` since NEXT-PHASE-1 (omega 62b6a28). Default build, no override:

```text
AIEN_OMEGA_DIR=<omega checkout at omega.lock>
AIEN_PHYSICS_DIR=<physics checkout at omega physics.lock>
AIEN_AIENOS_LOCK_REPO=<aienos clone>
cargo test -p aien-omega-compose
```

| Variable | Meaning |
|----------|---------|
| `AIEN_OMEGA_DIR` | omega checkout at `omega.lock`; shared with aien-omega-gpu |
| `AIEN_OMEGA_COMPOSE_DIR` | a different omega checkout for this crate only (takes precedence over `AIEN_OMEGA_DIR`) |
| `AIEN_OMEGA_COMPOSE_SHA` | **developer override**: full 40-hex sha the checkout must be at instead of `omega.lock`, for building against an unmerged omega branch. Prints a build warning; never used for a release build or a campaign receipt. |
| `AIEN_PHYSICS_DIR` | physics checkout at omega's `physics.lock` (default `<omega>/../physics`) |
| `AIEN_AIENOS_LOCK_REPO` | aienos clone holding omega's `aienos.lock` commit (passed as `AIENOS_LOCK_REPO`) |
| `AIEN_OMEGA_COMPOSE_LIB` | link a prebuilt `librx_compose.a` instead (not sha-checked) |
| `AIEN_FORCE_CPU_STUB=1` | stub: every call returns `ComposeError::Unavailable` (build time only; for a CPU-only run use `AIEN_BACKEND=cpu`, which makes the daemon refuse a GPU build) |

With no checkout and no library set the crate builds the stub and prints a warning.
Use `AIEN_OMEGA_COMPOSE_DIR` together with `AIEN_OMEGA_COMPOSE_SHA`, so an override
never changes the checkout aien-omega-gpu checks against `omega.lock`.

The integration test (`tests/compose.rs`) is ignored in a stub build.

## Semantics (from the C header)

- `open` binds the directory to one AienMachineId (`machine.id`); another
  machine is refused. The Cortex journal is `<dir>/cortex.cx`.
- `register_skill` before the first `run`/`recall`/`info`. At most 2 Skills.
  Callbacks run on omega World worker threads.
- `run(task, now_us)`: the Skills compete on staged J-Space branches, AEGIS
  checks each result with the `set_verify` callback, the winner commits.
  The result carries the Cortex record ids, branch count and verdict.
- A reopen appends records; pre-restart records stay an unchanged prefix.
- Torn journal tail: `refuse_torn = true` refuses without touching the file;
  otherwise it is repaired when it lies past the last commit, and refused
  explicitly (stably) when it lies inside the committed record.
