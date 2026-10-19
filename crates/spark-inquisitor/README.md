# spark-inquisitor

A command-line reviewer for pull requests and issues. It reads a diff or an issue body from a file and writes review text and suggested labels. It checks for telemetry or tracking code, em dashes and en dashes, and banned filler words, and it runs the contributor interview.

## Commands

From `spark-inquisitor --help` (built from this crate on 2026-10-19):

| Command | What it does |
|---|---|
| `review --author <user> --pr <num> --diff <file> [--title <title>] [--output <file>] [--output-labels <file>]` | review a PR diff and write the review comment as markdown |
| `triage-issue --author <user> --issue <num> --body <file> [--title <title>]` | classify an issue (bug report, complaint, feature request, inquiry) and suggest labels |
| `interview --author <user> --pr <num> [--title <title>]` | write the contributor interview comment for a pull request |
| `audit --diff <file> [--advisory-style]` | audit a diff for telemetry, surveillance and writing-style violations; `--advisory-style` turns style findings into warnings |
| `fix --file <file>` | replace em dashes and en dashes in the file |
| `evaluate --testimony <file> --author <user>` | judge a contributor's interview answers against the project standards |
| `doctor` | run the built-in self-checks |

Run `spark-inquisitor <command> --help` for the full options of each command.

## Build

From the root of the aien-sovereign-core checkout:

```bash
cargo build -p spark-inquisitor
./target/debug/spark-inquisitor --help
```

`install-inquisitor.sh` at the repository root is its installer (`--install-dir` sets where it goes). In the judge, the SC-INQUISITOR gate (`tests/gates/sc-inquisitor.gate`, step `inquisitor` in `tests/gates/run.sh`) builds it and reviews the pull request diff; that step needs the diff and a GitHub token.

## License

AGPL-3.0-or-later, like the rest of the workspace. See [LICENSE](../../LICENSE).
