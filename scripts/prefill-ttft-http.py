#!/usr/bin/env python3
"""HTTP TTFT / ITL client for the door (P-DRA-4 Lane P, #877). Test equipment, not product code.

Every number this prints is labelled HTTP: wall clock at the client, measured from just before the request
is sent. That clock includes the door, the tokenizer, queueing and the network. Never put it next to the
bench's CHIP_ONLY or TTFT_INPROC_HOST clocks.

  prefill-ttft-http.py ttft  URL MODEL PROMPT_FILE [--runs N] [--max-tokens K] [--out J]
  prefill-ttft-http.py probe URL MODEL LONG_FILE SHORT_FILE [--max-tokens K] [--out J]

ttft streams /v1/chat/completions (stream, temperature 0, enable_thinking false) N times. Per run it
reports the time to the first content delta, the time to the first byte, the gaps between later content
deltas (ITL), and the server's usage when it is sent.
probe first starts a short decode stream (SHORT_FILE). Once that stream's first content arrives it sends
the long prompt (LONG_FILE) on a second connection. It then reports the short stream's ITL before the
long request was sent, during the long request's prefill (send to first content), and after that. This
measures how much a long read stalls a concurrent decode.
Exit 0 ok, 2 usage/error.
"""
import argparse, json, statistics, sys, threading, time, urllib.request

LABEL = "HTTP"


def stream(url, model, prompt, max_tokens, t0=None, on_first=None):
    """Return dict with t_send, first_byte_s, content times (s since t0), text, usage; on_first() fires at the first content."""
    body = json.dumps({
        "model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic() if t0 is None else t0
    out = {"t_send": time.monotonic() - t0, "first_byte_s": None, "content_t": [], "text": "", "usage": None}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            now = time.monotonic() - t0
            if out["first_byte_s"] is None:
                out["first_byte_s"] = now
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                out["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if not out["content_t"] and on_first:
                        on_first()
                    out["content_t"].append(now)
                    out["text"] += piece
    return out


def gaps(ts):
    return [b - a for a, b in zip(ts, ts[1:])]


def summ(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median_ms": statistics.median(xs) * 1e3, "max_ms": max(xs) * 1e3,
            "mean_ms": statistics.fmean(xs) * 1e3}


def cmd_ttft(a):
    prompt = open(a.prompt, encoding="utf-8").read()
    runs = []
    for i in range(a.runs):
        s = stream(a.url, a.model, prompt, a.max_tokens)
        if not s["content_t"]:
            raise SystemExit(f"run {i}: no content delta")
        ttft = s["content_t"][0] - s["t_send"]
        runs.append({"ttft_ms": ttft * 1e3, "first_byte_ms": (s["first_byte_s"] - s["t_send"]) * 1e3,
                     "itl": summ(gaps(s["content_t"])), "usage": s["usage"], "text_head": s["text"][:80]})
        print(f"{LABEL} run {i}: ttft_ms {ttft * 1e3:.1f} itl_median_ms {runs[-1]['itl'].get('median_ms', 0):.2f}"
              f" prompt_tokens {(s['usage'] or {}).get('prompt_tokens', '-')}")
    t = [r["ttft_ms"] for r in runs]
    res = {"label": LABEL, "clock": "client wall, send to first content delta", "url": a.url,
           "prompt_file": a.prompt, "runs": runs, "ttft_median_ms": statistics.median(t)}
    print(f"{LABEL} ttft_median_ms {res['ttft_median_ms']:.1f} over {len(t)} runs")
    return res


def cmd_probe(a):
    long_p = open(a.long, encoding="utf-8").read()
    short_p = open(a.short, encoding="utf-8").read()
    t0 = time.monotonic()
    box, err = {}, []
    started = threading.Event()

    def short_run():
        try:
            box["short"] = stream(a.url, a.model, short_p, a.max_tokens, t0, started.set)
        except Exception as e:  # reported below
            err.append(repr(e))
        finally:
            started.set()

    th = threading.Thread(target=short_run)
    th.start()
    started.wait()
    time.sleep(a.lead_s)  # a few decode gaps before the long request, for the "before" phase
    lng = stream(a.url, a.model, long_p, 1, t0)
    th.join()
    if err or "short" not in box:
        raise SystemExit(f"short stream failed: {err}")
    s = box["short"]
    send, first = lng["t_send"], (lng["content_t"] or [lng["first_byte_s"]])[0]
    ct = s["content_t"]
    phase = {"before": [], "during": [], "after": []}
    for a_t, b_t in zip(ct, ct[1:]):
        key = "before" if b_t <= send else ("during" if a_t < first else "after")
        phase[key].append(b_t - a_t)
    res = {"label": LABEL, "clock": "client wall", "long_send_s": send, "long_first_content_s": first,
           "long_ttft_ms": (first - send) * 1e3, "short_itl": {k: summ(v) for k, v in phase.items()},
           "short_tokens": len(ct)}
    print(f"{LABEL} probe long_ttft_ms {res['long_ttft_ms']:.1f} short_itl_median_ms before "
          f"{res['short_itl']['before'].get('median_ms', float('nan')):.2f} during "
          f"{res['short_itl']['during'].get('median_ms', float('nan')):.2f} max_during "
          f"{res['short_itl']['during'].get('max_ms', float('nan')):.2f}")
    return res


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("ttft")
    t.add_argument("url"); t.add_argument("model"); t.add_argument("prompt")
    t.add_argument("--runs", type=int, default=3); t.add_argument("--max-tokens", type=int, default=16)
    t.add_argument("--out")
    q = sp.add_parser("probe")
    q.add_argument("url"); q.add_argument("model"); q.add_argument("long"); q.add_argument("short")
    q.add_argument("--max-tokens", type=int, default=256); q.add_argument("--lead-s", type=float, default=1.0)
    q.add_argument("--out")
    a = p.parse_args()
    try:
        res = cmd_ttft(a) if a.cmd == "ttft" else cmd_probe(a)
    except (OSError, ValueError, SystemExit) as e:
        print(f"prefill-ttft-http: error {e}", file=sys.stderr)
        return 2
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
