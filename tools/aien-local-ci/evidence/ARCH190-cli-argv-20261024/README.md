# Launcher arguments: 35 controls pass after the repair

The same offline controls report **23 passed, 12 failed** on Core main
`2f28ee89a26be1d326e29a85982bbc1c5714e437` and **35 passed, 0 failed** with this
repair. `before.log` and `after.log` retain the complete results.

Tracking: [architecture #190](https://github.com/aien-dev/aien-architecture-mv/issues/190).
The merge desk owns this bounded dispatch repair. The recorded changed-file
lists and titles for all 52 open Core PRs were checked; none already changes
`tools/aien-local-ci/aien-local-ci`.

## Defect and change

Only `start` and `stop` fill the launcher's lane-parser argument array. The
unconditional `set --` then replaced every command's arguments with that array.
As a result, `trigger omega <sha>` failed before resolving its repo, while
`plan --operator --base origin/main` silently invoked `plan` without its flags.

The fix places that existing reset inside a `start`/`stop` condition. Other
commands retain their original arguments. Lane parsing, worker construction,
verification rules and required gate selection are unchanged.

## Offline controls

Run:

```sh
bash tools/aien-local-ci/tests/cli_args_test.sh
```

The controls cover exact repo and SHA resolution, unknown-commit refusal,
operator and base flags, `verify` and `once` argument forwarding, `log` line
counts, named and default worker lanes, missing lane requirements, invalid
lane names and lane-specific stop behavior. NUL-delimited argument comparison
checks argument boundaries. State and clone paths contain spaces.

The fixture substitutes Git, the test binary, process discovery, detached
starts and Bash's signal builtin. Detached-start requests are recorded without
executing a worker. Both signal checks and signal delivery use a synthetic PID
inside the stub. No real worker, remote command, device access or activation
occurs. The fixture can take an alternate launcher path to test original source:

```sh
baseline=$(mktemp)
git show 2f28ee89a26be1d326e29a85982bbc1c5714e437:tools/aien-local-ci/aien-local-ci > "$baseline"
bash tools/aien-local-ci/tests/cli_args_test.sh "$baseline"
rm -f "$baseline"
```

The original source is expected to return 1 with 12 failed controls. Both old
and new source pass all existing lane parsing controls in this fixture.

## Qualification limits

Shell syntax and whitespace checks pass. The repository-required
`bash scripts/agent-preflight.sh` was attempted and **failed, exit 1** because
Cargo is absent. Branch, secrets and text checks passed; RustSec audit was
skipped because cargo-audit is absent. `preflight.log.gz` retains the exact output, including its original whitespace.

The pinned Crumb source is `94fffbcf88492c5a1c54211b5126c8a6c370a98d`; its
locally built binary has SHA-256
`d010857bfbd176f8fc1701fdafe9a6bd3d5bf512ad09f208876b0fcc8fa09c57`.
Regenerate and verify metadata with that tool after committing source.

This candidate is suitable for draft publication. Full preflight and the
existing operator-qualified pipeline gates remain required before merge or
installation. There is no Spark runtime qualification or deployment claim.
