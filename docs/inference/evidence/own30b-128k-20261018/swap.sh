#!/bin/bash
# swap.sh DIR EXTRA_ARGS... : stop the running server, start DIR/aien-serve-http with graph on, wait for the port
L=$HOME/.local/state/aien-own30b; D=$1; shift
P=$(cat $L/serve.pid); kill -TERM $P; while kill -0 $P 2>/dev/null; do sleep 1; done
cd $D && OWN30B_DECODE_GRAPH=1 setsid nohup bash -c "echo \$\$ > $L/serve.pid; exec ./aien-serve-http --engine omega --model-dir $HOME/models/qwen3-30b-a3b-instruct-2507-fp8 --port 18091 $*" > $L/serve-swap.log 2>&1 < /dev/null &
sleep 2; P=$(cat $L/serve.pid); until ss -ltn | grep -q ':18091 ' || ! kill -0 $P 2>/dev/null; do sleep 2; done; grep -E 'ready in' $L/serve-swap.log | cut -c1-120
