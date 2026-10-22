# spark-inquisitor

Automated review gate for the AIENOS project: PR code reviewer, contributor interview and issue triage.

## Architecture

`spark-inquisitor` is a pure native compiled Rust binary that enforces the project standards (the Stapleton Doctrine; values in `COVENANT.md`) across GitHub repositories:

1. **Zero Telemetry**: Flags tracking scripts, analytics SDKs, and data collection calls.
2. **Sovereign Voice and Anti-Slop**: Flags em dashes, en dashes, and banned AI buzzwords.
3. **Contributor Alignment**: Runs the contributor interview and checks the certification reply (zero telemetry, AGPL-3.0-or-later license and attribution preserved downstream).
4. **Autonomous Issue Triage**: Classifies bug reports, complaints, proposals, and community inquiries.

## Commands

- `spark-inquisitor review --author <user> --pr <num> --title <title> --diff <file>`: Runs the doctrine diff audit and generates the GitHub PR review comment.
- `spark-inquisitor triage-issue --author <user> --issue <num> --title <title> --body <file>`: Analyzes issue description and outputs triage guidance with labels.
- `spark-inquisitor evaluate --testimony <file> --author <user>`: Evaluates contributor interview replies against the project standards.
- `spark-inquisitor doctor`: Runs local self-diagnostics.
