# dra65 mock door (host check of window-dra65.sh only): serves set32 goldens; M=exact|mut|nosnap|slow|slow1
import json, os, sys, time, http.server
R=sys.argv[1]; D=R+'/evidence/dra65-prefill-20261031'; M=os.environ.get('M','exact')
g={p['user']:p['golden_ids'] for p in json.load(open(D+'/set32.json'))['prompts']}
for p in json.load(open(D+'/long-goldens.json'))['prompts']:
    if 'file' in p: g[open(R+'/'+p['file']).read()]=p['golden_ids']
    else: g['TURN2']=p['golden_ids']
class H(http.server.BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def out(s,code,b,ct='application/json'):
        s.send_response(code); s.send_header('Content-Type',ct); s.send_header('Content-Length',str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s): s.out(200,json.dumps({'data':[{'id':'mock'}]}).encode())
    def do_POST(s):
        q=json.loads(s.rfile.read(int(s.headers['Content-Length'])))
        if M=='500': return s.out(500,b'{"error":"x"}')
        msgs=q['messages']; key='TURN2' if len(msgs)==3 else msgs[0]['content']; ids=list(g.get(key,[1,2,3]))
        if M=='slow': time.sleep(30)
        if M=='slow1': time.sleep(0.25)
        if M=='mut' and key=='TURN2': ids[2]+=1
        sys.stderr.write('aien-serve-http: qwen38: prompt 9 tok, reuse %s\n' % ('Snapshot(2479) (2479 tok)' if key=='TURN2' and M!='nosnap' else 'Cold (0 tok)')); sys.stderr.flush()
        if M=='empty': ids=[]
        if M=='short': ids=ids[:-1]
        if q.get('stream'):
            ev=[{'choices':[{'delta':{'content':'x'}}]}]
            if M!='nousage': ev.append({'choices':[],'usage':{'prompt_tokens':9,'completion_tokens':len(ids)}})
            b=''.join('data: '+json.dumps(e)+'\n\n' for e in ev)+'data: [DONE]\n\n'
            return s.out(200,b.encode(),'text/event-stream')
        s.out(200,json.dumps({'choices':[{'message':{'content':'x'},'finish_reason':'stop'}],'usage':{'aien_token_ids':ids}}).encode())
http.server.ThreadingHTTPServer(('127.0.0.1',int(sys.argv[2])),H).serve_forever()
