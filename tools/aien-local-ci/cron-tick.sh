#!/bin/bash
# AIEN Spark pipeline tick (cron, every 5 min; no systemd by rule). Installed copy: ~/.local/lib/aien-local-ci/.
#   */5 * * * * $HOME/.local/lib/aien-local-ci/cron-tick.sh >> $HOME/.local/state/aien-local-ci/cron-tick.log 2>&1
# 1) keeps both worker lanes alive (--github): main (all enrolled repos) and small (the small repos, so a long
#    aienos/sc/omega job cannot starve them). `aien-local-ci start` is a no-op while a lane runs.
# 2) rebase train: moves ready-to-merge PRs that conflict only in .crumb files (or are behind) onto main.
# 3) runs one pipeline round: publish signed bundles, rerun stale checks, merge green PRs labelled ready-to-merge.
# Every 5 min, not 2: the 2-min round plus the worker's polling used up the 5000/h GitHub API limit (2026-10-18).
# Stop everything: touch ~/.local/state/aien-local-ci/PAUSE ; aien-local-ci stop ; aien-local-ci stop --lane small
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
S=$HOME/.local/state/aien-local-ci
L=$HOME/.local/lib/aien-local-ci
[ -e "$S/PAUSE" ] && exit 0
nice -n 10 aien-local-ci start --github >/dev/null
nice -n 10 aien-local-ci start --github --lane small >/dev/null
nice -n 10 /bin/bash "$L/rebase-train.sh" || echo "cron-tick: rebase-train.sh exit $?"
nice -n 10 /bin/bash "$L/pipeline.sh" round
