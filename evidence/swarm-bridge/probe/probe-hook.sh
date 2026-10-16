#!/bin/sh
inp=$(cat)
printf '%s\n' "$inp" >> /home/drakestapleton/workspace/board83/probe/hook-stdin.jsonl
ev=$(printf '%s' "$inp" | jq -r .hook_event_name)
jq -cn --arg e "$ev" '{hookSpecificOutput:{hookEventName:$e,additionalContext:("PROBE-TOKEN-7731 from " + $e)}}'
