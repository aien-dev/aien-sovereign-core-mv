# dra86 mock batch door (host check of window-dra86.sh only, never the chip): greedy ids are a hash of the
# messages; the prompt cache is modelled from the door's env and log wording (qwen36_batch.rs, q36_prefix.rs).
# MOCK_M: exact | nomut (the position mutant changes nothing) | norestore (turn 2 never restored) |
#         ignore_ns (the store loads nodes of another namespace)
import hashlib, json, os, sys, time, http.server
E = os.environ; M = E.get('MOCK_M', 'exact')
ON = E.get('AIEN_Q36_PREFIX') == '1'; MUT = E.get('AIEN_Q36_PREFIX_MUTANT') == '1'
GB = float(E.get('AIEN_Q36_PREFIX_POOL_GB', '16')); BUDGET = GB * 1024
NS = hashlib.sha256(json.dumps(sorted((k, v) for k, v in E.items() if (k.startswith('AIEN_Q36_') or k.startswith('AIEN_OMEGA_'))
     and (not k.startswith('AIEN_Q36_PREFIX_') or k == 'AIEN_Q36_PREFIX_MUTANT'))).encode()).hexdigest()[:16]
DIR = E.get('AIEN_Q36_PREFIX_DIR'); KEY = E.pop('AIEN_KV_STORE_KEY', None)
def say(s): sys.stderr.write('aien-serve-http: qwen36-batch: ' + s + '\n'); sys.stderr.flush()
pool = []  # [conversation key, MiB]
def kv_mib(tok): return tok * 20480 / 2**20
def total(): return sum(m for _, m in pool)
if not ON or GB <= 0:
    say('prompt cache off (needs AIEN_Q36_PREFIX=1 and AIEN_Q36_PREFIX_POOL_GB > 0)')
else:
    say('prompt cache on: budget %.1f GiB, checkpoint every 2048 tok, K/V 20480 B per token, state 65.6 MiB per checkpoint%s'
        % (GB, ', POSITION MUTANT' if MUT else ''))
    if DIR:
        if MUT: say('prompt cache store %s off: a mutant or test switch is set (its states are never stored); the pool stays in memory' % DIR)
        elif KEY is None or len(KEY) != 64: say('prompt cache store %s off: key missing; the pool stays in memory' % DIR)
        else:
            ld = dc = 0
            for f in sorted(os.listdir(DIR)):
                if not f.endswith('.q36p'): continue
                ns, key, m = open(os.path.join(DIR, f)).read().split('\t')
                if ns == NS or M == 'ignore_ns': pool.append([key, float(m)]); ld += 1
                else:
                    sys.stderr.write('aien-serve-http: q36 prefix: discarding %s: another namespace\n' % f); os.remove(os.path.join(DIR, f)); dc += 1
            sys.stderr.write('aien-serve-http: q36 prefix: store %s open: %d nodes loaded (%.1f MiB), %d discarded, in 0.0 s\n' % (DIR, ld, total(), dc))
            sys.stderr.flush()
evicted = 0
def ntok(msgs): return sum(max(1, len(m['content']) // 4) for m in msgs) + 4 * len(msgs) + 3
def save(key, tok):
    global evicted
    nodes = max(1, tok // 2048) + 1; per = 65.6 + kv_mib(tok) / nodes
    for i in range(nodes):
        pool.append([key, per])
        while total() > BUDGET and len(pool) > 1: pool.pop(0); evicted += 1
        res = sum(m for k, m in pool if k == key); rn = sum(1 for k, _ in pool if k == key)
        say('prompt cache save: %d tok (K/V from %d) from row 0, %.1f MiB in 2 ms; conversation resident %d nodes %.1f MiB; pool %d nodes %.1f MiB of %.1f GiB, %d evicted, 0 refused'
            % ((i + 1) * tok // nodes, i * tok // nodes, per, rn, res, len(pool), total(), GB, evicted))
        if DIR and KEY and not MUT:
            open(os.path.join(DIR, '%s-%d.q36p' % (hashlib.sha256(key.encode()).hexdigest()[:16], i)), 'w').write('%s\t%s\t%f' % (NS, key, per))
class H(http.server.BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def out(s, code, b, ct='application/json'):
        s.send_response(code); s.send_header('Content-Type', ct); s.send_header('Content-Length', str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s): s.out(200, json.dumps({'data': [{'id': 'mock86'}]}).encode())
    def do_POST(s):
        q = json.loads(s.rfile.read(int(s.headers['Content-Length']))); msgs = q['messages']; n = q['max_tokens']
        h = hashlib.sha256(json.dumps(msgs).encode()).digest(); ids = [h[i % 32] + 256 * i for i in range(n)]
        key = msgs[0]['content']; tok = ntok(msgs); reuse = 0; t0 = time.time()
        if len(msgs) == 3 and ON and GB > 0 and M != 'norestore' and any(k == key for k, _ in pool):
            reuse = ntok(msgs[:2]); res = sum(m for k, m in pool if k == key)
            say('prompt cache restore: %d tok into row 0, %.1f MiB in 3 ms, conversation resident %.1f MiB%s' % (reuse, res, res, ' (POSITION MUTANT)' if MUT else ''))
            if MUT and M != 'nomut': ids[min(2, n - 1)] += 1
        say('join: prompt %d tok, reuse %d tok (%s, row 0), 0 cache checkpoints, queued 0 ms, now 1 rows' % (tok, reuse, 'prompt cache' if reuse else 'cold'))
        if ON and GB > 0 and len(msgs) == 1: save(key, tok)
        say('leave: row 0, prompt %d tok (reused %d), prefill %d ms, decode %d tok in 1 ms over %d beats, mean rows 1.00, wall 1.0 ms/token, Stop'
            % (tok, reuse, int((time.time() - t0) * 1000) + 1, n, n))
        text = 'answer %s' % h.hex()[:12]
        if q.get('stream'):
            ev = [{'choices': [{'delta': {'content': text}}]}, {'choices': [], 'usage': {'prompt_tokens': tok, 'completion_tokens': n}}]
            return s.out(200, (''.join('data: ' + json.dumps(e) + '\n\n' for e in ev) + 'data: [DONE]\n\n').encode(), 'text/event-stream')
        s.out(200, json.dumps({'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                                'usage': {'prompt_tokens': tok, 'completion_tokens': n, 'aien_token_ids': ids}}).encode())
http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
