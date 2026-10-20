#!/bin/bash
# L190 endpoint receipt: the same prompts through aien-serve-http --engine wormhole on 18190, first the 27B
# alone (--no-draft), then the pair (--draft-dir 30B); outputs must be byte-identical. Leaves the pair serving.
set -u
cd /home/drakestapleton/workspace/sc-wormhole-30b || exit 1
BIN=/home/drakestapleton/workspace/targets/sc-wormhole-30b-native/release/aien-serve-http
M=/home/drakestapleton/models
E=evidence/WORMHOLE-30B/05-endpoint
I=evidence/WORMHOLE-ACCEPT-69825c92/inputs
mkdir -p $E /tmp/l190
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
start() { # $1 label, rest: extra flags
  local label=$1; shift
  until [ "$(avail)" -ge 56 ]; do sleep 30; done
  systemctl --user stop l190-wormhole-18190 2>/dev/null; systemctl --user reset-failed l190-wormhole-18190 2>/dev/null
  systemd-run --user --unit=l190-wormhole-18190 --property=StandardOutput=file:/tmp/l190/serve-$label.log \
    --property=StandardError=file:/tmp/l190/serve-$label.log $BIN --engine wormhole --model-dir $M/qwen3.8-27b-fp8-017b9c7a --port 18190 --bind 127.0.0.1 \
    --max-context 8192 --max-tokens-cap 4096 --thinking off "$@"
  sleep 2; systemctl --user show -p MainPID --value l190-wormhole-18190 > /tmp/l190/serve.pid
  until curl -sf http://127.0.0.1:18190/v1/models > /dev/null; do
    systemctl --user is-active -q l190-wormhole-18190 || { echo "server $label died"; tail -5 /tmp/l190/serve-$label.log; return 1; }
    sleep 5
  done
}
ask() { # $1 label
  for f in q38-code-golden q38-prose q38-codelong; do
    user=$(/home/drakestapleton/workspace/targets/sc-wormhole-30b/release/wormhole30 user $M/qwen3.8-27b-fp8-017b9c7a/tokenizer.json $I/$f.json)
    jq -n --arg u "$user" '{model:"aien-wormhole-27b-30b", messages:[{role:"user",content:$u}], max_tokens:256, temperature:0}' \
      | curl -s http://127.0.0.1:18190/v1/chat/completions -H 'Content-Type: application/json' -d @- > $E/$1-$f.json
    jq -r '.choices[0].message.content' $E/$1-$f.json > $E/$1-$f.txt
  done
}
stop() { systemctl --user stop l190-wormhole-18190; }
start alone --no-draft || exit 2
ask alone; stop
start pair --draft-dir $M/qwen3-30b-a3b-instruct-2507-fp8 --draft-k 4 --ring-slots 1024 || exit 3
ask pair
same=0; for f in q38-code-golden q38-prose q38-codelong; do cmp -s $E/alone-$f.txt $E/pair-$f.txt && same=$((same+1)); done
grep "wormhole:" /tmp/l190/serve-pair.log > $E/pair-acceptance.log
grep "wormhole" /tmp/l190/serve-alone.log | head -3 > $E/alone.log
echo "identical $same/3 (pair vs 27B alone, same binary); pair left serving on 127.0.0.1:18190 pid $(cat /tmp/l190/serve.pid)"
