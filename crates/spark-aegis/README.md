# spark-aegis

A command-line safety tool for the DGX Spark workstation. It scans files and diffs for secrets, banned writing patterns and dangerous commands, audits the live machine (listening sockets, workspace secrets, Cortex reachability), and runs four containment actions. Its motto, printed by `spark-aegis doctrine`: "The network is the house. Cut the session. Isolate the host. Close the door. Keep the evidence."

## What it checks

The invariant names below are the ones `spark-aegis doctrine` prints.

- **SECURE_TPM_ONLY**: no plaintext secrets on disk. `spark-aegis audit` reports this as true only when the workspace has no `.env` files and no exposed keys (`src/audit.rs`). It does not itself talk to a TPM.
- **UNSLOP_COMPLIANCE**: no em dashes, en dashes or banned filler words in scanned text.
- **PURE_NATIVE_SYSTEMS**: the tool is plain compiled Rust.
- **BLAKE3_CORTEX_TRAIL**: containment actions write an incident record with a BLAKE3 digest to Cortex memory (`src/containment.rs`, token from `CORTEX_TOKEN` or `~/.config/cortex/token`).

## Parts

- `doctrine`: the four defensive actions and the invariants.
- `scanner`: finds secrets, banned writing patterns and dangerous commands in files, directories and diffs.
- `byte_scan`: the byte and pattern search the scanner uses.
- `containment`: the four actions (`cut_session`, `isolate_host`, `close_door`, `keep_evidence`).
- `pr_triage`: reviews a diff and prints a verdict that carries a BLAKE3 digest of the report (a digest, not a signature).
- `audit`: the live machine audit.

## Commands

From `spark-aegis --help` (built from this crate on 2026-10-19):

```bash
spark-aegis scan /path/to/target
spark-aegis review-pr diff.patch
git diff | spark-aegis review-pr -
spark-aegis audit
spark-aegis contain 12345
spark-aegis contain --session pts/2 pts/2
spark-aegis contain --close /path/to/file /path/to/file
spark-aegis doctrine
```

`contain` always needs the target as its last argument. On its own, a number is taken as a process ID, an existing path as a path, and anything else as a session. `--pid`, `--session` or `--close` (each with the target as its value) says the kind explicitly and takes precedence. The earlier form `spark-aegis contain --session pts/2`, with no target argument, is refused by the argument parser. Containment acts on the real machine (it ends sessions, isolates processes and quarantines paths), so run it only on purpose.

Build it from the root of the aien-sovereign-core checkout with `cargo build -p spark-aegis`.

## License

AGPL-3.0-or-later, like the rest of the workspace (`license.workspace = true` in `Cargo.toml`). See [LICENSE](../../LICENSE). An earlier copy of this page said Apache-2.0 WITH LLVM-exception; that did not match `Cargo.toml`.
