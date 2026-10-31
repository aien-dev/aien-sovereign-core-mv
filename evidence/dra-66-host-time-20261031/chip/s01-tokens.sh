#!/usr/bin/env bash
# DRA-66 (sovereign #857, PR #863) chip slice, no args: golden identity through the door (mutant 4 red
# first), the per-token receipt at 1 and 8 conversations, seed control, the token checker. Body in
# dra66-lib.sh; see PLAN.md. Run only by chip/run-queue.sh; `DRY_RUN=1 bash s01-tokens.sh` = host dry path.
# pdra4-gb10: none
SLICE=s01-tokens
GATES=(mutant4-red golden-identity receipt-1conv receipt-8conv seed-t07 check-tokens)
source "$(dirname "${BASH_SOURCE[0]}")/dra66-lib.sh"
