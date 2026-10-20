import json,struct,glob,os,collections,sys
def headers(d):
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with open(f,"rb") as fh:
            n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n))
        for k,v in h.items():
            if k=="__metadata__": continue
            yield os.path.basename(f),k,v["dtype"],v["shape"],v["data_offsets"][1]-v["data_offsets"][0]
def cls(k):
    if ".mlp.experts." in k and not k.startswith("mtp"): return "routed_experts"
    if "ple" in k and "ngram_embedding" in k: return "ngram_table"
    if k.startswith("mtp"): return "mtp"
    if "hyper_connection" in k: return "hyper_conn"
    if "embed_tokens" in k: return "embed"
    if k.startswith("lm_head"): return "lm_head"
    if "visual" in k or "vision" in k: return "vision"
    if "shared_expert" in k: return "shared_expert"
    if ".mlp.gate." in k: return "router"
    if "linear_attn" in k: return "deltanet"
    if "self_attn" in k: return "attention"
    if ".ple." in k: return "ple_other"
    return "other"
for d in sys.argv[1:]:
    c=collections.Counter(); tot=0; ex=collections.Counter()
    for f,k,dt,sh,nb in headers(d):
        c[cls(k)]+=nb; tot+=nb
        if cls(k)=="routed_experts":
            parts=k.split(".mlp.experts.")[1].split(".",1); ex[(k.split(".mlp.experts.")[0],parts[0])]+=nb
    print(d, "total", tot)
    for k,v in c.most_common(): print("  %-15s %15d"%(k,v))
    if ex:
        s=set(ex.values()); print("  experts", len(ex), "bytes/expert", s)
