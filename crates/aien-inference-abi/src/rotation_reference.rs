//! Test-only transport to the seeded Omega codec, never a serving fallback.
use std::path::Path;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use sha2::{Digest, Sha256};

pub struct Bridge {
    child: Child,
    input: Option<ChildStdin>,
    output: Option<ChildStdout>,
    pub frames: usize,
}
fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap())
}
fn ready(fd: i32, event: i16, until: std::time::Instant) -> Result<(),String> {
    let ms=until.saturating_duration_since(std::time::Instant::now()).as_millis();
    if ms==0 { return Err("native frame timeout".into()); }
    let mut p=libc::pollfd {fd,events:event,revents:0};
    let rc=unsafe {libc::poll(&mut p,1,ms.min(30000) as i32)};
    if rc<=0 {return Err(format!("native frame poll failed: {rc}"));}
    Ok(())
}
impl Bridge {
    pub fn new(path: &Path, sha: &str) -> Result<Self,String> {
        let bytes=std::fs::read(path).map_err(|e|e.to_string())?;
        if sha.len()!=64 || format!("{:x}",Sha256::digest(&bytes))!=sha {
            return Err("native binary fingerprint mismatch".into());
        }
        if !bytes.starts_with(b"\x7fELF\x02\x01") || bytes.get(16..20)!=Some(&[2,0,183,0]) {
            return Err("native executable is not AArch64 EXEC".into());
        }
        let mut child=Command::new(path).stdin(Stdio::piped()).stdout(Stdio::piped())
            .stderr(Stdio::inherit()).spawn().map_err(|e|e.to_string())?;
        let input=child.stdin.take(); let output=child.stdout.take();
        for fd in [input.as_ref().unwrap().as_raw_fd(),output.as_ref().unwrap().as_raw_fd()] {
            let flags=unsafe {libc::fcntl(fd,libc::F_GETFL)};
            if flags<0 || unsafe {libc::fcntl(fd,libc::F_SETFL,flags|libc::O_NONBLOCK)}<0 {
                return Err("cannot bound native pipe IO".into());
            }
        }
        Ok(Self{child,input,output,frames:0})
    }
    pub fn exchange(&mut self, frame: &[u8]) -> Result<Vec<u8>,String> {
        if frame.len()<12 || &frame[..2]!=b"RQ" || ![b'1',b'2'].contains(&frame[3]) {
            return Err("invalid native input header".into());
        }
        let count=word(frame,4) as usize; let mode=word(frame,8);
        let (unit,kind,outunit,header)=match frame[2] {b'U'=>(128,b'P',68,12),b'P'=>(68,b'R',1024,16),_=>return Err("invalid native input kind".into())};
        if !(1..=256).contains(&count) || mode>2 || (frame[3]==b'1' && mode>1) || frame.len()!=12+count*unit || self.frames>=999999 {
            return Err("invalid native input bounds".into());
        }
        let until=std::time::Instant::now()+std::time::Duration::from_secs(30);
        let input=self.input.as_mut().ok_or("closed native input")?;
        let mut sent=0;
        while sent<frame.len() {
            ready(input.as_raw_fd(),libc::POLLOUT,until)?;
            match input.write(&frame[sent..]) {
                Ok(0)=>return Err("native input made no progress".into()),Ok(n)=>sent+=n,
                Err(e) if e.kind()==std::io::ErrorKind::WouldBlock=>(),Err(e)=>return Err(e.to_string()),
            }
        }
        let mut out=vec![0u8;header+count*outunit]; let mut got=0;
        let output=self.output.as_mut().ok_or("closed native output")?;
        while got<out.len() {
            ready(output.as_raw_fd(),libc::POLLIN,until)?;
            match output.read(&mut out[got..]) {
                Ok(0)=>return Err("native output truncated".into()),Ok(n)=>got+=n,
                Err(e) if e.kind()==std::io::ErrorKind::WouldBlock=>(),Err(e)=>return Err(e.to_string()),
            }
        }
        if &out[..2]!=b"RQ" || out[2]!=kind || out[3]!=frame[3] || out[4..12]!=frame[4..12] {
            return Err("native output identity mismatch".into());
        }
        if kind==b'R' && word(&out,12)!=(if mode==0 {3584} else {458752}) {
            return Err("native reconstruction denominator mismatch".into());
        }
        self.frames+=1;
        Ok(out)
    }
    pub fn finish(&mut self)->Result<(),String> {
        self.input.take(); self.output.take();
        let status=self.child.wait().map_err(|e|e.to_string())?;
        if !status.success() {return Err(format!("native child failed: {status}"));}
        Ok(())
    }
}
impl Drop for Bridge {
    fn drop(&mut self) { self.input.take();self.output.take();let _=self.child.wait(); }
}
// Everything below is external-oracle byte transport/measurement, not a product quantizer.
use std::collections::BTreeMap;
use std::sync::{Mutex,OnceLock};
use serde_json::{Value,json};
static BRIDGE: OnceLock<Mutex<Option<Bridge>>> = OnceLock::new();
static RECORDS: OnceLock<Mutex<BTreeMap<String,Value>>> = OnceLock::new();
static AUDITS: OnceLock<Mutex<Vec<Value>>> = OnceLock::new();
fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
pub fn audit_tensor(name:&str)->bool {
    ["layers.0.mlp.experts.0.gate_proj.weight","layers.0.mlp.experts.0.up_proj.weight",
     "layers.0.mlp.experts.0.down_proj.weight","layers.20.mlp.experts.128.up_proj.weight",
     "layers.39.mlp.experts.255.down_proj.weight"].iter().any(|suffix|name.ends_with(suffix))
}
pub fn audit_output(name:&str,x:&[f32],t:usize,y:&[f32],reference:&[f32]) {
    let (mut ss,mut err,mut max)=(0f64,0f64,0f64);
    for (&a,&b) in y.iter().zip(reference) {let d=a as f64-b as f64;ss+=(b as f64).powi(2);err+=d*d;max=max.max(d.abs());}
    let xb:Vec<u8>=x.iter().flat_map(|v|v.to_le_bytes()).collect();
    AUDITS.get_or_init(Default::default).lock().unwrap().push(json!({"tensor":name,"input_vectors":t,
        "activation_origin":"current candidate full-model forward, actual FFN matmul input",
        "activation_sha256":digest(&xb),"reference_squared_norm":ss,"error_squared_norm":err,
        "relative_output_error":if ss>0.0 {Some((err/ss).sqrt())}else{None},"max_absolute_error":max}));
}
pub fn convert(name:&str,n:usize,k:usize,raw:&[u8],scales:&[u16])->Result<Option<Vec<f32>>,String> {
    if !(name.contains(".mlp.experts.") || name.contains(".mlp.shared_expert.")) {return Ok(None);}
    if ![".gate_proj.weight",".up_proj.weight",".down_proj.weight"].iter().any(|s|name.ends_with(s)) {return Ok(None);}
    let mode=match std::env::var("AIEN_ROTQUANT_MODE") {Ok(s)=>Some(s.parse::<u32>().map_err(|_|"invalid native quantizer mode")?),Err(std::env::VarError::NotPresent)=>None,Err(e)=>return Err(e.to_string())};
    if mode.is_some_and(|m|m>2) {return Err("invalid native quantizer mode".into());}
    if mode.is_none() && std::env::var("AIEN_ROTQUANT_AUDIT").as_deref()!=Ok("1") {return Ok(None);}
    let scale_bytes:Vec<u8>=scales.iter().flat_map(|v|v.to_le_bytes()).collect();
    let mut receipt=json!({"tensor":name,"shape":[n,k],"mode":mode,"source_sha256":digest(raw),"scale_sha256":digest(&scale_bytes),"raw_bytes":raw.len(),"scale_bytes":scale_bytes.len()});
    let mut result=None;
    if let Some(mode)=mode {
        if n%128!=0 || k%128!=0 || raw.len()!=n*k || scales.len()!=n/128*(k/128) {return Err("unsupported source-scale layout".into());}
        let path=std::env::var_os("AIEN_ROTQUANT_BIN").ok_or("native codec path required")?;
        let sha=std::env::var("AIEN_ROTQUANT_SHA256").map_err(|_|"native codec fingerprint required")?;
        receipt["native_binary_sha256"]=json!(sha);
        let identity=receipt.clone();
        let cache=std::env::var_os("AIEN_ROTQUANT_CACHE").map(std::path::PathBuf::from);
        if let Some(root)=&cache {
            if digest(&std::fs::read(&path).map_err(|e|e.to_string())?)!=sha{return Err("cached native binary fingerprint mismatch".into());}
            if let Some((values,stored))=cache_read(root,&identity,raw.len())? {
                let mut records=RECORDS.get_or_init(Default::default).lock().map_err(|_|"records poisoned")?;
                if records.get(name).is_some_and(|old|old!=&stored){return Err("cached source identity changed".into());}
                records.insert(name.into(),stored);return Ok(Some(values));
            }
        }
        let mut guard=BRIDGE.get_or_init(Default::default).lock().map_err(|_|"native bridge poisoned")?;
        let needed=raw.len().div_ceil(32768)*2;
        if guard.as_ref().is_some_and(|b|b.frames+needed>=999999) {
            if let Some(mut b)=guard.take(){b.finish()?;}
        }
        if guard.is_none(){*guard=Some(Bridge::new(Path::new(&path),&sha)?);}
        let bridge=guard.as_mut().unwrap(); let mut values=Vec::with_capacity(raw.len());
        let mut packed_hash=Sha256::new();let mut packed_bytes=0usize;
        let (mut zeros,mut lost)=(0usize,0usize); let mut rotated=0usize;
        for (chunk_index,chunk) in raw.chunks(32768).enumerate() {
            let blocks=chunk.len()/128;
            let mut input=b"RQU2".to_vec();input.extend_from_slice(&(blocks as u32).to_le_bytes());input.extend_from_slice(&mode.to_le_bytes());input.extend_from_slice(chunk);
            let packed=bridge.exchange(&input)?;packed_bytes+=packed.len();packed_hash.update(&packed);
            if mode==2 {for b in 0..blocks {rotated+=usize::from(word(&packed,12+b*68)>>31!=0);}}
            else if mode==1 {rotated+=blocks;}
            let reconstructed=bridge.exchange(&packed)?;let denominator=word(&reconstructed,12) as f64;
            for (j,bytes) in reconstructed[16..].chunks_exact(8).enumerate() {
                let numerator=i64::from_le_bytes(bytes.try_into().unwrap());
                let idx=chunk_index*32768+j;
                let scale=f32::from_bits((scales[(idx/k/128)*(k/128)+(idx%k/128)] as u32)<<16);
                if !scale.is_finite() || scale<=0.0 {return Err("invalid original scale".into());}
                let value=(numerator as f64/denominator) as f32*scale;
                if !value.is_finite(){return Err("nonfinite native reconstruction".into());}
                if raw[idx]&127==0 {zeros+=1;lost+=usize::from(numerator!=0);}
                values.push(value);
            }
        }
        if values.len()!=raw.len() || (mode==2 && lost!=0) {return Err("native zero-support/length violation".into());}
        receipt["packed_bytes_with_scales"]=json!(packed_bytes+scale_bytes.len());
        receipt["packed_sha256"]=json!(format!("{:x}",packed_hash.finalize()));
        receipt["original_zero_weights"]=json!(zeros);receipt["lost_zero_weights"]=json!(lost);
        receipt["rotated_blocks"]=json!(rotated);receipt["blocks"]=json!(raw.len()/128);
        if let Some(root)=&cache {cache_write(root,&identity,&receipt,&values)?;}
        result=Some(values);
    }
    let mut records=RECORDS.get_or_init(Default::default).lock().map_err(|_|"record mutex poisoned")?;
    if let Some(previous)=records.get(name) {if previous!=&receipt{return Err(format!("source/packed identity changed for {name}"));}}
    records.insert(name.into(),receipt);
    Ok(result)
}
pub fn finish_report()->Result<Value,String> {
    let final_child_frames=if let Some(mut b)=BRIDGE.get_or_init(Default::default).lock().map_err(|_|"bridge poisoned")?.take(){let frames=b.frames;b.finish()?;frames}else{0};
    let records=std::mem::take(&mut *RECORDS.get_or_init(Default::default).lock().map_err(|_|"records poisoned")?);
    let audits=std::mem::take(&mut *AUDITS.get_or_init(Default::default).lock().map_err(|_|"audits poisoned")?);
    Ok(json!({"unique_expert_tensors":records.len(),"tensors":records,"real_activation_probes":audits,"final_child_frames":final_child_frames}))
}

fn cache_path(root:&Path,identity:&Value)->Result<std::path::PathBuf,String> {
    if !root.is_absolute(){return Err("cache root must be absolute".into());}
    for part in root.ancestors(){let m=std::fs::symlink_metadata(part).map_err(|e|e.to_string())?;if m.file_type().is_symlink() || !m.is_dir(){return Err("unsafe cache directory".into());}}
    Ok(root.join(format!("{}.orc",digest(&serde_json::to_vec(identity).map_err(|e|e.to_string())?))))
}
fn cache_write(root:&Path,identity:&Value,receipt:&Value,values:&[f32])->Result<(),String>{
    use std::os::unix::fs::OpenOptionsExt;
    let path=cache_path(root,identity)?;
    match std::fs::symlink_metadata(&path){Ok(_)=>return Err("cache entry already exists".into()),Err(e) if e.kind()==std::io::ErrorKind::NotFound=>(),Err(e)=>return Err(e.to_string())}
    if !values.iter().all(|v|v.is_finite()){return Err("nonfinite cache payload".into());}
    let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
    let metadata=serde_json::to_vec(&json!({"input":identity,"receipt":receipt,"f32_sha256":digest(&bytes)})).map_err(|e|e.to_string())?;
    if metadata.len()>65536{return Err("cache metadata too large".into());}
    let temporary=path.with_extension(format!("partial-{}",std::process::id()));
    let mut f=std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(&temporary).map_err(|e|e.to_string())?;
    f.write_all(b"ORC1").and_then(|_|f.write_all(&(metadata.len() as u32).to_le_bytes())).and_then(|_|f.write_all(&Sha256::digest(&metadata))).and_then(|_|f.write_all(&metadata)).and_then(|_|f.write_all(&bytes)).and_then(|_|f.sync_all()).map_err(|e|e.to_string())?;
    // Atomic no-clobber publication; only the producer owns this key. A crash leaving
    // two links is refused rather than mistaken for a complete cache transaction.
    std::fs::hard_link(&temporary,&path).map_err(|e|e.to_string())?;
    std::fs::remove_file(temporary).map_err(|e|e.to_string())?;
    Ok(())
}
fn cache_read(root:&Path,identity:&Value,length:usize)->Result<Option<(Vec<f32>,Value)>,String>{
    use std::os::unix::fs::{OpenOptionsExt,MetadataExt};
    let path=cache_path(root,identity)?;
    match std::fs::symlink_metadata(&path){Ok(m)=>{if !m.is_file() || m.file_type().is_symlink() || m.nlink()!=1{return Err("unsafe cache file".into());}},Err(e) if e.kind()==std::io::ErrorKind::NotFound=>return Ok(None),Err(e)=>return Err(e.to_string())}
    let mut f=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path).map_err(|e|e.to_string())?;
    let m=f.metadata().map_err(|e|e.to_string())?;
    if !m.is_file() || m.nlink()!=1{return Err("unsafe cache file".into());}
    let mut header=[0u8;8];f.read_exact(&mut header).map_err(|e|e.to_string())?;
    let size=word(&header,4) as usize;
    if &header[..4]!=b"ORC1" || size>65536 || m.len()!=40+size as u64+(length as u64).checked_mul(4).ok_or("cache size overflow")? {return Err("cache size/magic mismatch".into());}
    let mut meta_sha=[0u8;32];f.read_exact(&mut meta_sha).map_err(|e|e.to_string())?;
    let mut meta=vec![0u8;size];f.read_exact(&mut meta).map_err(|e|e.to_string())?;
    if Sha256::digest(&meta)[..]!=meta_sha{return Err("cache receipt checksum mismatch".into());}
    let meta:Value=serde_json::from_slice(&meta).map_err(|e|e.to_string())?;
    let mut bytes=vec![0u8;length.checked_mul(4).ok_or("cache size overflow")?];f.read_exact(&mut bytes).map_err(|e|e.to_string())?;
    if &meta["input"]!=identity || meta["f32_sha256"]!=digest(&bytes) {return Err("cache identity/content mismatch".into());}
    let values:Vec<f32>=bytes.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect();
    if !values.iter().all(|v|v.is_finite()){return Err("nonfinite cache payload".into());}
    Ok(Some((values,meta["receipt"].clone())))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn binary() -> std::path::PathBuf { std::env::var_os("AIEN_ROTQUANT_BIN").expect("qualified native binary required").into() }
    fn hash() -> String { std::env::var("AIEN_ROTQUANT_SHA256").expect("qualified binary hash required") }
    #[test]
    fn native_frames_match_independent_fraction_oracle() {
        let mut bridge=Bridge::new(&binary(), &hash()).unwrap();
        let mut data=&include_bytes!("rotation-fixtures.bin")[..];
        let mut count=0;
        while !data.is_empty() {
            let n=u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
            let m=u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
            data=&data[8..];
            assert_eq!(bridge.exchange(&data[..n]).unwrap(), data[n..n+m]);
            data=&data[n+m..]; count+=1;
        }
        assert_eq!(count,6);
    }
    #[test]
    fn native_reference_cache_is_exact_and_fails_closed() {
        let root=std::path::PathBuf::from(std::env::var_os("AIEN_ROTQUANT_CACHE_TESTROOT").expect("owned cache test root required")).join(format!("cache-test-{}",std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let identity=json!({"fixture":"aaaa"});let receipt=json!({"native":"pinned"});let values=[0.0f32,-0.0,4.0/7.0];
        cache_write(&root,&identity,&receipt,&values).unwrap();
        let (loaded,meta)=cache_read(&root,&identity,3).unwrap().expect("native reconstruction cache must be reused");
        assert_eq!(loaded.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),values.iter().map(|v|v.to_bits()).collect::<Vec<_>>());assert_eq!(meta,receipt);
        assert!(cache_write(&root,&identity,&receipt,&values).is_err());
        assert!(cache_read(&root,&identity,4).is_err());
        let p=root.join(format!("{}.orc",digest(&serde_json::to_vec(&identity).unwrap())));
        let original=std::fs::read(&p).unwrap();
        let mut changed_meta=original.clone();let at=changed_meta.windows(6).position(|w|w==b"pinned").unwrap();changed_meta[at]=b'b';std::fs::write(&p,&changed_meta).unwrap();assert!(cache_read(&root,&identity,3).is_err(),"receipt metadata corruption must be refused");
        let mut corrupt=original.clone();*corrupt.last_mut().unwrap()^=1;std::fs::write(&p,&corrupt).unwrap();assert!(cache_read(&root,&identity,3).is_err());
        std::fs::write(&p,&original).unwrap();let victim=root.join("retained");std::fs::rename(&p,&victim).unwrap();std::fs::hard_link(&victim,&p).unwrap();assert!(cache_read(&root,&identity,3).unwrap_err().contains("unsafe"));
        std::fs::remove_file(&p).unwrap();std::os::unix::fs::symlink(&victim,&p).unwrap();assert!(cache_read(&root,&identity,3).unwrap_err().contains("unsafe"));assert_eq!(std::fs::read(&victim).unwrap(),original);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn wrong_native_fingerprint_is_refused() {
        assert!(Bridge::new(&binary(), &"0".repeat(64)).is_err());
    }
    #[test]
    fn successful_native_noop_is_not_a_result() {
        use sha2::{Digest,Sha256};
        let p=std::path::PathBuf::from(std::env::var_os("AIEN_ROTQUANT_NOOP").expect("native no-op required"));
        let sha=format!("{:x}",Sha256::digest(std::fs::read(&p).unwrap()));
        let mut bridge=Bridge::new(&p,&sha).unwrap();
        assert!(bridge.exchange(&include_bytes!("rotation-fixtures.bin")[8..276]).is_err());
    }
}
