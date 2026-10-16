#!/usr/bin/env bash
# Start the held cold A/B rerun detached from the agent's shell, so a tool
# time limit cannot stop it mid-sample. Writes its exit code to a marker file.
E=~/workspace/headstart-exp
rm -f "$E/logs/12-ab-held.rc"
setsid nohup bash -c "HS_AB_JOBS='' '$E/bin/05-measure.sh' ab > '$E/logs/05-measure-ab-held.log' 2>&1; echo \$? > '$E/logs/12-ab-held.rc'" > /dev/null 2>&1 < /dev/null &
echo "started pid $!"
