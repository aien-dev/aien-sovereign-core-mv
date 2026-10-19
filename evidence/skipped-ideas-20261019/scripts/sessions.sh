#!/bin/bash
# Ask the live 27B server (port 18091) for two ~2,000-token greedy sessions. Retries on 503.
W=$HOME/workspace/skipped-work/sessions; mkdir -p $W
cat > $W/code-prompt.txt <<'EOF'
Write a complete, well-commented C program of about 300 lines that implements a small in-memory key-value store: a hash table with separate chaining that grows when it gets full, a command loop that reads GET, SET, DEL and LIST commands from standard input, and a set of self-tests that run when the program is started with --test. Briefly explain the design first, then give the full program.
EOF
cat > $W/story-prompt.txt <<'EOF'
Write a long short story, about 1500 words, about a lighthouse keeper who finds a message in a bottle. Use several scenes and some dialogue.
EOF
for name in code story; do
  [ -s $W/$name-reply.json ] && grep -q '"content"' $W/$name-reply.json && continue
  body=$(jq -n --rawfile p $W/$name-prompt.txt '{model:"qwen3.8-27b-fp8", messages:[{role:"user", content:$p}], max_tokens:2100, temperature:0}')
  for try in $(seq 1 120); do
    code=$(curl -s -o $W/$name-reply.json -w '%{http_code}' -m 1800 -H 'Content-Type: application/json' -d "$body" 127.0.0.1:18091/v1/chat/completions)
    echo "$(date -u +%FT%TZ) $name try $try http $code" >> $W/requests.log
    [ "$code" = 200 ] && break
    sleep 60
  done
  jq -r '.choices[0].message.content' $W/$name-reply.json > $W/$name-reply.txt
  jq -c '.usage' $W/$name-reply.json >> $W/requests.log
done
sha256sum $W/* > $W/../sessions.SHA256SUMS
