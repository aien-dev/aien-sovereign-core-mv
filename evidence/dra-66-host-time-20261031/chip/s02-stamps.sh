#!/usr/bin/env bash
# DRA-66 (sovereign #857, PR #863) chip slice, no args: the stamped launch timeline at one conversation
# (submit + poll per token), its checker, and Main's Wolfram inputs gate-red.wl then gate.wl (with
# s01-tokens' summaries). Body in dra66-lib.sh; see PLAN.md. Queue it after s01-tokens.
# pdra4-gb10: none
SLICE=s02-stamps
GATES=(stamps-1conv check-stamps)
NEEDS=s01-tokens
source "$(dirname "${BASH_SOURCE[0]}")/dra66-lib.sh"
