//! Full-model external quality oracle. Never link or deploy as the serving path.
#[cfg(not(feature="rotation-qualification"))]
fn main(){eprintln!("rotation-qualification feature required");std::process::exit(2);}
#[cfg(feature="rotation-qualification")]
mod qualified {
use aien_inference_abi::qwen38::{Qwen38Reference,Logits,argmax,negative_log_likelihood};
use aien_inference_abi::rotation_reference;
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::path::Path;
use std::time::Instant;
fn ids(v:&Value,key:&str)->Result<Vec<u32>,String>{v[key].as_array().ok_or(format!("missing {key}"))?.iter().map(|x|x.as_u64().and_then(|x|u32::try_from(x).ok()).ok_or(format!("invalid {key}"))).collect()}
fn margin(v:&[f32])->f64 {let first=argmax(v);let second=v.iter().enumerate().filter(|(i,_)|*i!=first).map(|(_,x)|*x).fold(f32::NEG_INFINITY,f32::max);v[first] as f64-second as f64}
fn write(path:&Path,value:&Value)->Result<(),String>{
    let bytes=serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?;
    let mut options=std::fs::OpenOptions::new();options.write(true).create_new(true);
    use std::io::Write;
    options.open(path).map_err(|e|format!("{}: {e}",path.display()))?.write_all(&bytes).map_err(|e|e.to_string())
}
fn digest_f32(values:&[f32])->String {let mut h=Sha256::new();for chunk in values.chunks(8192){let bytes:Vec<u8>=chunk.iter().flat_map(|v|v.to_le_bytes()).collect();h.update(bytes);}format!("{:x}",h.finalize())}
fn compare(a:&[f32],b:&[f32])->(f64,f64) {
    let (mut ab,mut aa,mut bb)=(0f64,0f64,0f64);
    let ma=a.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
    let mb=b.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
    let za=a.iter().map(|&x|(x as f64-ma).exp()).sum::<f64>().ln()+ma;
    let zb=b.iter().map(|&x|(x as f64-mb).exp()).sum::<f64>().ln()+mb;
    let mut kl=0f64;
    for (&x,&y) in a.iter().zip(b){let (x,y)=(x as f64,y as f64);ab+=x*y;aa+=x*x;bb+=y*y;kl+=(x-za).exp()*((x-za)-(y-zb));}
    (ab/(aa*bb).sqrt(),kl)
}
fn read_span<R:std::io::Read+std::io::Seek>(file:&mut R,base:u64,start:u64,end:u64,size:u64)->Result<Vec<u8>,String>{
    if start>=end || base.checked_add(end).is_none_or(|x|x>size) {return Err("invalid tensor extent".into());}
    let count=usize::try_from(end-start).map_err(|_|"tensor too large")?;
    let mut bytes=vec![0u8;count];file.seek(std::io::SeekFrom::Start(base+start)).map_err(|e|e.to_string())?;
    file.read_exact(&mut bytes).map_err(|e|e.to_string())?;Ok(bytes)
}
fn precompute(args:&[String])->Result<(),String>{
    use std::io::Read;
    let model=Path::new(&args[2]);let first:usize=args[3].parse().map_err(|_|"invalid first layer")?;let end:usize=args[4].parse().map_err(|_|"invalid end layer")?;
    if first>=end || end>40 || std::env::var_os("AIEN_ROTQUANT_CACHE").is_none() || std::env::var_os("AIEN_ROTQUANT_MODE").is_none(){return Err("precompute requires cache, mode and layers within0..40".into());}
    let out=Path::new(&args[5]);std::fs::create_dir(out).map_err(|e|e.to_string())?;
    let mut total=0usize;
    for layer in first..end {
        let path=model.join(format!("layers-{layer}.safetensors"));let mut file=std::fs::File::open(&path).map_err(|e|e.to_string())?;
        let before=file.metadata().map_err(|e|e.to_string())?;let mut len=[0u8;8];file.read_exact(&mut len).map_err(|e|e.to_string())?;let size=u64::from_le_bytes(len);
        if size>64*1024*1024 || size.checked_add(8).is_none_or(|x|x>before.len()){return Err("invalid safetensors header".into());}
        let mut bytes=vec![0u8;size as usize];file.read_exact(&mut bytes).map_err(|e|e.to_string())?;
        let header:Value=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
        let object=header.as_object().ok_or("header must be an object")?;
        let extent=|v:&Value|->Result<(u64,u64),String>{Ok((v["data_offsets"][0].as_u64().ok_or("missing extent")?,v["data_offsets"][1].as_u64().ok_or("missing extent")?))};
        let mut count=0usize;
        for (name,item) in object {
            if item["dtype"]!="F8_E4M3" || !(name.contains(".mlp.experts.") || name.contains(".mlp.shared_expert.")) || ![".gate_proj.weight",".up_proj.weight",".down_proj.weight"].iter().any(|x|name.ends_with(x)){continue;}
            let n=item["shape"][0].as_u64().ok_or("missing rows")? as usize;let k=item["shape"][1].as_u64().ok_or("missing columns")? as usize;
            let (lo,hi)=extent(item)?;let raw=read_span(&mut file,8+size,lo,hi,before.len())?;
            if n.checked_mul(k)!=Some(raw.len()) || n%128!=0 || k%128!=0{return Err("invalid matrix dimensions".into());}
            let scale=&header[format!("{name}_scale_inv")];
            if scale["dtype"]!="BF16" || scale["shape"]!=json!([n/128,k/128]){return Err("invalid scale grid".into());}
            let (lo,hi)=extent(scale)?;let bytes=read_span(&mut file,8+size,lo,hi,before.len())?;
            if bytes.len()!=2*(n/128)*(k/128){return Err("invalid scale byte count".into());}
            let scales:Vec<u16>=bytes.chunks_exact(2).map(|v|u16::from_le_bytes(v.try_into().unwrap())).collect();
            let values=rotation_reference::convert(name,n,k,&raw,&scales)?.ok_or("expert conversion missing")?;
            if values.len()!=raw.len(){return Err("incomplete reconstructed matrix".into());}count+=1;
        }
        let after=file.metadata().map_err(|e|e.to_string())?;
        if before.len()!=after.len() || before.modified().ok()!=after.modified().ok(){return Err("source shard changed".into());}
        let tensors=rotation_reference::finish_report()?;
        if count==0 || tensors["unique_expert_tensors"]!=json!(count){return Err("precompute tensor-count disagreement".into());}
        write(&out.join(format!("layer-{layer}.json")),&json!({"layer":layer,"tensor_count":count,"header_sha256":format!("{:x}",Sha256::digest(&bytes)),"native":tensors}))?;
        total+=count;eprintln!("precompute layer{layer}: {count} native expert matrices");
    }
    write(&out.join("complete.json"),&json!({"complete":true,"first_layer":first,"end_layer":end,"tensor_count":total,"reference_expanded_f32_cache_not_packed_storage":true}))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_tensor_span_and_bounds() {
        let mut bytes=std::io::Cursor::new([1u8,2,3,4]);
        assert_eq!(read_span(&mut bytes,0,1,3,4).unwrap(),[2,3]);
        assert!(read_span(&mut bytes,0,0,5,4).is_err());
        assert!(read_span(&mut bytes,u64::MAX,1,2,4).is_err());
        assert!(read_span(&mut bytes,0,3,2,4).is_err());
    }
}
pub fn run()->Result<(),String>{
    let args:Vec<String>=std::env::args().collect();
    if args.len()!=6{return Err("usage: q36_rotation_quality MODEL GOLDEN_JSON_OR_PASSAGE COUNT OUT_NEW_DIR identity,hadamard,zero-aware".into());}
    if args[1]=="--precompute" {return precompute(&args);}
    let model_dir=Path::new(&args[1]);let input=Path::new(&args[2]);let count:usize=args[3].parse().map_err(|_|"invalid count")?;
    if !(2..=512).contains(&count){return Err("count must be2..512".into());}
    let out=Path::new(&args[4]);std::fs::create_dir(out).map_err(|e|format!("fresh output required: {e}"))?;
    let input_bytes=std::fs::read(input).map_err(|e|e.to_string())?;
    let (tokens,targets,start,golden)=if input.extension().is_some_and(|e|e=="json") {
        let g:Value=serde_json::from_slice(&input_bytes).map_err(|e|e.to_string())?;
        let p=ids(&g,"prompt_ids")?;let r=ids(&g,"ref_ids")?;
        if p.is_empty() || r.len()<count || g["ref_margin"].as_array().map_or(0,Vec::len)<count {return Err("incomplete golden".into());}
        let mut t=p.clone();t.extend_from_slice(&r[..count-1]);(t,r[..count].to_vec(),p.len()-1,Some(g))
    } else {
        let mut text=String::from_utf8(input_bytes.clone()).map_err(|e|e.to_string())?;
        if let Some((cut,_))=text.char_indices().nth(count*8){text.truncate(cut);}
        let tokenizer=ChatTokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e|e.to_string())?;
        let mut t=tokenizer.encode(&text).map_err(|e|e.to_string())?;
        if t.len()<count {return Err("passage shorter than requested count".into());}
        t.truncate(count);let target=t[1..].to_vec();(t,target,0,None)
    };
    std::env::remove_var("AIEN_ROTQUANT_MODE");std::env::set_var("AIEN_ROTQUANT_AUDIT","1");
    let t0=Instant::now();eprintln!("FP8 baseline: {} input tokens, {} scored positions",tokens.len(),targets.len());
    let mut model=Qwen38Reference::open(model_dir)?;model.moe_capture=true;
    let vocab=model.config.vocab;let mut state=model.new_state();
    let baseline=model.forward(&mut state,&tokens,Logits::All)?;
    if baseline.len()!=tokens.len()*vocab || !baseline.iter().all(|v|v.is_finite()){return Err("invalid baseline logits".into());}
    let routes=model.routes.clone();
    let activation_hashes:Vec<String>=model.moe_x.iter().map(|v|digest_f32(v)).collect();
    let source=rotation_reference::finish_report()?;drop(model);drop(state);
    let mut rows=Vec::new();let mut exact=true;
    for (j,&target) in targets.iter().enumerate(){
        let row=&baseline[(start+j)*vocab..(start+j+1)*vocab];
        let id=argmax(row) as u32;let m=margin(row);
        let control=if let Some(g)=&golden {id==target && (m-g["ref_margin"][j].as_f64().ok_or("invalid golden margin")?).abs()<=1e-3}else{true};
        exact &=control;
        rows.push(json!({"position":j,"target":target,"argmax":id,"margin":m,"nll":negative_log_likelihood(row,target as usize),"exact_control":control}));
    }
    write(&out.join("fp8-tensors.json"),&source)?;
    write(&out.join("fp8.json"),&json!({"arm":"fp8","positions":rows,"input_sha256":format!("{:x}",Sha256::digest(&input_bytes)),"logits_sha256":digest_f32(&baseline),"moe_activation_sha256":activation_hashes,"exact_control":exact,"reference_wall_seconds_not_a_runtime_benchmark":t0.elapsed().as_secs_f64(),"tokens":tokens,"scored_positions":targets.len()}))?;
    if !exact{return Err("FP8 exact-control failure; no approximate arm ran".into());}
    eprintln!("FP8 control passed in {:.1}s; {} expert tensors",t0.elapsed().as_secs_f64(),source["unique_expert_tensors"]);
    for arm in args[5].split(',').filter(|s|!s.is_empty()) {
        let mode=match arm {"identity"=>0,"hadamard"=>1,"zero-aware"=>2,_=>return Err(format!("unknown arm {arm}"))};
        std::env::set_var("AIEN_ROTQUANT_MODE",mode.to_string());let clock=Instant::now();eprintln!("{arm}: full model start");
        let mut candidate=Qwen38Reference::open(model_dir)?;candidate.moe_capture=true;
        let mut state=candidate.new_state();let logits=candidate.forward(&mut state,&tokens,Logits::All)?;
        if logits.len()!=baseline.len() || !logits.iter().all(|v|v.is_finite()){return Err("invalid candidate logits".into());}
        let mut result=Vec::new();
        for (j,&target) in targets.iter().enumerate(){
            let a=&baseline[(start+j)*vocab..(start+j+1)*vocab];let b=&logits[(start+j)*vocab..(start+j+1)*vocab];let (cosine,kl)=compare(a,b);
            let route_changes=routes.iter().zip(&candidate.routes).filter(|(a,b)|a[(start+j)*8..(start+j+1)*8]!=b[(start+j)*8..(start+j+1)*8]).count();
            result.push(json!({"position":j,"target":target,"argmax":argmax(b),"reference_argmax":argmax(a),"reference_margin":margin(a),"reference_nll":negative_log_likelihood(a,target as usize),"nll":negative_log_likelihood(b,target as usize),"logit_cosine":cosine,"kl_reference_candidate":kl,"layers_with_changed_ordered_routes":route_changes}));
        }
        let tensors=rotation_reference::finish_report()?;
        write(&out.join(format!("{arm}-tensors.json")),&tensors)?;
        write(&out.join(format!("{arm}.json")),&json!({"arm":arm,"mode":mode,"positions":result,"scored_positions":targets.len(),"full_model_layers":candidate.config.layers,"logits_sha256":digest_f32(&logits),"moe_activation_sha256":candidate.moe_x.iter().map(|x|digest_f32(x)).collect::<Vec<_>>(),"reference_wall_seconds_not_a_runtime_benchmark":clock.elapsed().as_secs_f64(),"native_binary_sha256":std::env::var("AIEN_ROTQUANT_SHA256").map_err(|e|e.to_string())?,"direct_packed_gpu_execution":false}))?;
        eprintln!("{arm}: finished {:.1}s; {} expert tensors",clock.elapsed().as_secs_f64(),tensors["unique_expert_tensors"]);
    }
    std::env::remove_var("AIEN_ROTQUANT_MODE");
    write(&out.join("complete.json"),&json!({"completed":true,"arms":args[5],"scored_positions":targets.len(),"exact_control":true,"production_changed":false,"gpu_tested":false}))?;
    Ok(())
}
}
#[cfg(feature="rotation-qualification")]
fn main(){if let Err(e)=qualified::run(){eprintln!("QUALIFICATION FAILED: {e}");std::process::exit(1);}}
