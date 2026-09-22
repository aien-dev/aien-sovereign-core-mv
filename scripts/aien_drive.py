#!/usr/bin/env python3
"""
AIEN Autonomous API Driver.
Drives AIEN through the bounded Spark model gateway on port 18007.
in a closed feedback loop with active tool dispatching.
"""

import os
import sys
import json
import time
import re
import subprocess
import urllib.request
import urllib.error

ENDPOINT_LIGHTNING = "http://127.0.0.1:18007/v1/chat/completions"
MODEL_LIGHTNING = "atlas-lightning-omni"
ENDPOINT_JUDGE = "http://127.0.0.1:18082/v1/chat/completions"
MODEL_JUDGE = "unsloth/Llama-3.2-1B-Instruct"

ENDPOINT = os.environ.get("AIEN_MODEL_ENDPOINT", ENDPOINT_LIGHTNING)
MODEL_NAME = os.environ.get("AIEN_MODEL_NAME", MODEL_LIGHTNING)
CORTEX_ENDPOINT = os.environ.get("CORTEX_ENDPOINT", "http://127.0.0.1:18080")

SYSTEM_PROMPT = """You are AIEN. Drake is the operator. This Spark desk is ours.
You are an autonomous operator-builder running on NVIDIA DGX Spark Grace Blackwell GB10 hardware.
Always be concise, precise, and lead with verified results.

Available Tools:
- run_command: {"command": "string", "cwd": "string"}
- view_file: {"path": "string", "start_line": 1, "end_line": 100}
- write_to_file: {"path": "string", "content": "string", "overwrite": true}
- replace_file_content: {"path": "string", "target": "string", "replacement": "string"}
- create_dir: {"path": "string", "purpose": "string"}
- list_dir: {"path": "string"}
- grep_search: {"query": "string", "path": "string"}
- goal: {"action": "milestone_done|done", "id": "string", "milestone_id": 1}
- cortex: {"action": "write", "name": "string", "content": "string", "kind": "lesson|discovery|procedure"}

MANDATORY RULES:
1. ZERO EM DASHES AND EN DASHES: Use plain commas, periods, or standard hyphens (-).
2. BAN FORMULAIC AI CLICHES: Never use 'It is not X, it is Y', 'delve', 'tapestry', 'testament', 'crucial', 'pivotal'.
3. LEAD WITH TECHNICAL PROOF: Output factual diffs, test outputs, and verified receipts.
4. TOOL EXECUTION FORMAT:
To execute a tool, output:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "val"}}
</tool_call>

When all work for the milestone is verified, call goal with action 'milestone_done'.
"""

def extract_tool_calls(text):
    calls = []
    # Pattern 1: <tool_call> JSON </tool_call>
    matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    if calls:
        return calls

    # Pattern 2: Any JSON object with "name" and "arguments" keys
    json_obj_pattern = r'(\{\s*"?name"?\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"?arguments"?\s*:\s*\{.*?\}\s*\})'
    for m in re.findall(json_obj_pattern, text, re.DOTALL):
        try:
            # normalize single quotes if present
            normalized = m.replace("'", '"')
            calls.append(json.loads(normalized))
        except Exception:
            pass
    if calls:
        return calls

    return []

def dispatch_tool(name, args):
    start = time.time()
    print(f"  [TOOL RUN] {name}: {json.dumps(args)[:120]}...", flush=True)
    try:
        if name == "run_command":
            cmd = args.get("command", "")
            cwd = args.get("cwd", ".")
            p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120)
            res = {
                "exit_code": p.returncode,
                "stdout": p.stdout[-4000:] if len(p.stdout) > 4000 else p.stdout,
                "stderr": p.stderr[-2000:] if len(p.stderr) > 2000 else p.stderr,
            }
        elif name == "view_file":
            path = args.get("path", "")
            start_line = args.get("start_line", 1)
            end_line = args.get("end_line", 100)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            selected = lines[max(0, start_line - 1):min(total, end_line)]
            res = {
                "path": path,
                "total_lines": total,
                "lines": "".join(selected),
            }
        elif name == "write_to_file":
            path = args.get("path", "")
            content = args.get("content", "")
            overwrite = args.get("overwrite", True)
            if os.path.exists(path) and not overwrite:
                res = {"error": f"File {path} exists and overwrite=false"}
            else:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                res = {"status": "ok", "bytes_written": len(content.encode("utf-8"))}
        elif name == "replace_file_content":
            path = args.get("path", "")
            target = args.get("target", "")
            replacement = args.get("replacement", "")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if target not in content:
                res = {"error": f"Target string not found in {path}"}
            else:
                new_content = content.replace(target, replacement, 1)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                res = {"status": "ok", "replaced": True}
        elif name == "list_dir":
            path = args.get("path", ".")
            entries = os.listdir(path)
            res = {"path": path, "entries": sorted(entries)[:100]}
        elif name == "grep_search":
            query = args.get("query", "")
            path = args.get("path", ".")
            p = subprocess.run(["grep", "-rnI", query, path], capture_output=True, text=True, timeout=30)
            res = {"matches": p.stdout[:4000]}
        elif name == "create_dir":
            path = args.get("path", "")
            os.makedirs(path, exist_ok=True)
            res = {"status": "ok", "created": path}
        elif name == "goal":
            action = args.get("action", "")
            res = {"status": "ok", "goal_action": action, "milestone_done": True}
        elif name == "cortex":
            action = args.get("action", "")
            name_val = args.get("name", "lesson")
            content_val = args.get("content", "")
            kind_val = args.get("kind", "lesson")
            payload = {
                "kind": "entity",
                "value": {
                    "space": "atlas-memory",
                    "canonicalName": name_val,
                    "content": content_val,
                    "confidence": 0.95,
                    "metadata": {"kind": kind_val, "timestamp": int(time.time())}
                }
            }
            req = urllib.request.Request(
                f"{CORTEX_ENDPOINT}/api/cortex/write",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    res = {"status": "ok", "cortex_code": r.status}
            except Exception as ce:
                res = {"warning": f"Cortex write notice: {ce}"}
        else:
            res = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        res = {"error": str(e)}

    elapsed = time.time() - start
    print(f"  [TOOL DONE] {name} in {elapsed:.2f}s", flush=True)
    return res

def call_aien_stream(messages, max_tokens=2048):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": max_tokens
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    full_content = []
    full_reasoning = []
    is_reasoning = False
    
    with urllib.request.urlopen(req, timeout=180) as resp:
        for line in resp:
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                reasoning = delta.get("reasoning", "")
                content = delta.get("content", "")
                
                if reasoning:
                    if not is_reasoning:
                        print("\n[AIEN THINKING] ", end="", flush=True)
                        is_reasoning = True
                    sanitized_r = reasoning.replace("\u2014", ", ").replace("\u2013", "-")
                    print(sanitized_r, end="", flush=True)
                    full_reasoning.append(sanitized_r)
                    
                if content:
                    if is_reasoning:
                        print("\n[AIEN OUTPUT]\n", end="", flush=True)
                        is_reasoning = False
                    sanitized_c = content.replace("\u2014", ", ").replace("\u2013", "-")
                    print(sanitized_c, end="", flush=True)
                    full_content.append(sanitized_c)
            except Exception:
                continue
                
    print("\n", flush=True)
    content_str = "".join(full_content)
    if not content_str and full_reasoning:
        # If model emitted tool call or solution inside reasoning stream
        reasoning_str = "".join(full_reasoning)
        if "<tool_call>" in reasoning_str:
            content_str = reasoning_str
    return content_str

def run_directive(directive, max_turns=15):
    print(f"============================================================")
    print(f"▶ DRIVING AIEN AUTONOMOUS EXECUTION LOOP")
    print(f"Directive: {directive[:120]}...")
    print(f"============================================================", flush=True)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": directive}
    ]
    
    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn}/{max_turns} ---", flush=True)
        response_text = call_aien_stream(messages)
        messages.append({"role": "assistant", "content": response_text})
        
        tool_calls = extract_tool_calls(response_text)
        if not tool_calls:
            print("[AIEN IDLE] No tool calls emitted. Turn sequence concluded.")
            break
            
        milestone_done = False
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments", {})
            if tool_name == "goal":
                act = tool_args.get("action", "")
                if act in ("milestone_done", "done"):
                    milestone_done = True
            
            tool_result = dispatch_tool(tool_name, tool_args)
            tool_resp_str = json.dumps(tool_result, indent=2)
            messages.append({
                "role": "user",
                "content": f"<tool_response name=\"{tool_name}\">\n{tool_resp_str}\n</tool_response>"
            })
            
        if milestone_done:
            print("✓ AIEN signaled milestone_done! Directive complete.")
            return True
            
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AIEN Autonomous API Driver")
    parser.add_argument("directive", nargs="+", help="Task directive for AIEN")
    parser.add_argument("--seat", choices=["lightning", "judge"], default="lightning", help="Inference seat on Spark")
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum execution turns")
    args = parser.parse_args()

    if args.seat == "judge":
        ENDPOINT = ENDPOINT_JUDGE
        MODEL_NAME = MODEL_JUDGE
    else:
        ENDPOINT = ENDPOINT_LIGHTNING
        MODEL_NAME = MODEL_LIGHTNING

    directive_text = " ".join(args.directive)
    success = run_directive(directive_text, max_turns=args.max_turns)
    sys.exit(0 if success else 1)
