//! Loading the student checkpoint and writing the merged (private) model directory.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen3_moe::{bf16_to_f32, f32_to_bf16};
use aien_inference_abi::qwen3_moe_omega::{ShardSource, TensorSource};

use crate::gemm::Packed;
use crate::model::{Attn, Gdn, Layer, Lin, Mixer, Mlp, Model};

fn read_f32(src: &mut dyn TensorSource, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let n: usize = shape.iter().product();
    match src.dtype_of(name)?.as_deref() {
        Some("BF16") => {
            let mut w = vec![0u16; n];
            // SAFETY: u8 has no alignment need and the slice covers exactly the words' bytes;
            // the checkpoint is little-endian like this host.
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u8>(), n * 2) };
            src.read_into(name, "BF16", shape, bytes)?;
            Ok(w.into_iter().map(bf16_to_f32).collect())
        }
        Some("F32") => {
            let mut w = vec![0f32; n];
            // SAFETY: as above; every bit pattern is a valid f32.
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u8>(), n * 4) };
            src.read_into(name, "F32", shape, bytes)?;
            Ok(w)
        }
        Some(d) => Err(format!("{name}: dtype {d}, expected BF16 or F32")),
        None => Err(format!("{name}: not in the checkpoint")),
    }
}

/// Where the text model lives: `model.language_model.` (multimodal) or `model.`.
pub fn text_prefix(src: &mut dyn TensorSource) -> Result<String, String> {
    for p in ["model.language_model.", "model."] {
        if matches!(
            src.dtype_of(&format!("{p}embed_tokens.weight")),
            Ok(Some(_))
        ) {
            return Ok(p.to_string());
        }
    }
    Err("checkpoint: no embed_tokens.weight".into())
}

/// The dense text model from a checkpoint directory (tied embeddings only: the 0.8B student).
pub fn load(dir: &Path) -> Result<Model, String> {
    let c = Qwen38Config::from_checkpoint(dir)?;
    if c.moe.is_some() || !c.tied_embeddings {
        return Err("load: the trainer takes the dense, tied-embedding student only".into());
    }
    let mut src = ShardSource::open(dir)?;
    let src: &mut dyn TensorSource = &mut src;
    let p = text_prefix(src)?;
    let h = c.hidden;
    let lin = |src: &mut dyn TensorSource, name: &str, n: usize, k: usize| {
        read_f32(src, name, &[n, k]).map(|w| Lin::new(w, n, k))
    };
    let mut layers = Vec::with_capacity(c.layers);
    for l in 0..c.layers {
        let lp = format!("{p}layers.{l}");
        let mixer = if c.full_attention[l] {
            let a = format!("{lp}.self_attn");
            Mixer::Attn(Attn {
                q: lin(
                    src,
                    &format!("{a}.q_proj.weight"),
                    2 * c.heads * c.head_dim,
                    h,
                )?,
                k: lin(src, &format!("{a}.k_proj.weight"), c.kv_dim(), h)?,
                v: lin(src, &format!("{a}.v_proj.weight"), c.kv_dim(), h)?,
                o: lin(src, &format!("{a}.o_proj.weight"), h, c.heads * c.head_dim)?,
                q_norm: read_f32(src, &format!("{a}.q_norm.weight"), &[c.head_dim])?,
                k_norm: read_f32(src, &format!("{a}.k_norm.weight"), &[c.head_dim])?,
            })
        } else {
            let a = format!("{lp}.linear_attn");
            Mixer::Gdn(Gdn {
                qkv: lin(src, &format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h)?,
                z: lin(src, &format!("{a}.in_proj_z.weight"), c.lin_value_dim(), h)?,
                a: lin(src, &format!("{a}.in_proj_a.weight"), c.lin_v_heads, h)?,
                b: lin(src, &format!("{a}.in_proj_b.weight"), c.lin_v_heads, h)?,
                conv: read_f32(
                    src,
                    &format!("{a}.conv1d.weight"),
                    &[c.conv_dim(), 1, c.conv_kernel],
                )?,
                a_log: read_f32(src, &format!("{a}.A_log"), &[c.lin_v_heads])?,
                dt_bias: read_f32(src, &format!("{a}.dt_bias"), &[c.lin_v_heads])?,
                norm: read_f32(src, &format!("{a}.norm.weight"), &[c.lin_v_dim])?,
                out: lin(src, &format!("{a}.out_proj.weight"), h, c.lin_value_dim())?,
            })
        };
        let m = format!("{lp}.mlp");
        layers.push(Layer {
            input_norm: read_f32(src, &format!("{lp}.input_layernorm.weight"), &[h])?,
            post_norm: read_f32(src, &format!("{lp}.post_attention_layernorm.weight"), &[h])?,
            mixer,
            mlp: Mlp {
                gate: lin(src, &format!("{m}.gate_proj.weight"), c.intermediate, h)?,
                up: lin(src, &format!("{m}.up_proj.weight"), c.intermediate, h)?,
                down: lin(src, &format!("{m}.down_proj.weight"), h, c.intermediate)?,
            },
        });
    }
    let embed = read_f32(src, &format!("{p}embed_tokens.weight"), &[c.vocab, h])?;
    let final_norm = read_f32(src, &format!("{p}norm.weight"), &[h])?;
    let head_p = Packed::new(&crate::gemm::transpose(&embed, c.vocab, h), h, c.vocab);
    let head_bwd_p = Packed::new(&embed, c.vocab, h);
    Ok(Model {
        c,
        embed,
        head_p,
        head_bwd_p,
        final_norm,
        layers,
    })
}

/// One tensor of a safetensors file: dtype, shape, absolute byte range.
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

fn read_header(path: &Path) -> Result<BTreeMap<String, Entry>, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|e| e.to_string())?;
    let hlen = u64::from_le_bytes(len);
    if hlen > 100 << 20 {
        return Err(format!("{}: header of {hlen} bytes", path.display()));
    }
    let mut hdr = vec![0u8; hlen as usize];
    f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_slice(&hdr).map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    for (name, e) in v.as_object().ok_or("header is not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let off = e["data_offsets"].as_array().ok_or("no data_offsets")?;
        let (a, b) = (
            off[0].as_u64().ok_or("offset")?,
            off[1].as_u64().ok_or("offset")?,
        );
        out.insert(
            name.clone(),
            Entry {
                dtype: e["dtype"].as_str().ok_or("no dtype")?.to_string(),
                shape: e["shape"]
                    .as_array()
                    .ok_or("no shape")?
                    .iter()
                    .map(|d| d.as_u64().unwrap_or(0) as usize)
                    .collect(),
                start: 8 + hlen + a,
                end: 8 + hlen + b,
            },
        );
    }
    Ok(out)
}

/// Writes `out_dir`: the base checkpoint's text tensors with every adapted map replaced by
/// `W + s B A` in bf16 (round to nearest even), one shard, its index, and the base's config,
/// tokenizer and template files. Returns the (name, sha-free) list of replaced tensors.
pub fn save_merged(
    model: &Model,
    base: &Path,
    out_dir: &Path,
    readme: &str,
) -> Result<usize, String> {
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(base.join("model.safetensors.index.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let shards: std::collections::BTreeSet<String> = index["weight_map"]
        .as_object()
        .ok_or("index: no weight_map")?
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    let mut src = ShardSource::open(base)?;
    let prefix = text_prefix(&mut src)?;
    let replaced: BTreeMap<String, &Lin> = model
        .lins()
        .into_iter()
        .filter(|(_, l)| l.lora.is_some())
        .map(|(n, l)| (format!("{prefix}{n}"), l))
        .collect();
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let mut entries: Vec<(String, std::path::PathBuf, Entry)> = Vec::new();
    for s in &shards {
        let path = base.join(s);
        for (name, e) in read_header(&path)? {
            if name.starts_with(&prefix) {
                entries.push((name, path.clone(), e));
            }
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        serde_json::json!({"format": "pt", "aien": "small Allen (L389), private: never publish"}),
    );
    let mut off = 0u64;
    for (name, _, e) in &entries {
        let len = e.end - e.start;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": e.dtype, "shape": e.shape, "data_offsets": [off, off + len]}),
        );
        off += len;
    }
    let mut hbytes =
        serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|e| e.to_string())?;
    while hbytes.len() % 8 != 0 {
        hbytes.push(b' ');
    }
    let shard_name = "model.safetensors-00001-of-00001.safetensors";
    let mut out =
        std::io::BufWriter::new(File::create(out_dir.join(shard_name)).map_err(|e| e.to_string())?);
    out.write_all(&(hbytes.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    out.write_all(&hbytes).map_err(|e| e.to_string())?;
    let mut n_replaced = 0;
    for (name, path, e) in &entries {
        if let Some(lin) = replaced.get(name) {
            if e.dtype != "BF16" || e.shape != [lin.n, lin.k] {
                return Err(format!(
                    "{name}: base is {} {:?}, merge has [{}, {}]",
                    e.dtype, e.shape, lin.n, lin.k
                ));
            }
            let w = lin.merged();
            let mut bytes = Vec::with_capacity(w.len() * 2);
            for v in w {
                bytes.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
            }
            out.write_all(&bytes).map_err(|e| e.to_string())?;
            n_replaced += 1;
        } else {
            let mut f = File::open(path).map_err(|e| e.to_string())?;
            f.seek(SeekFrom::Start(e.start))
                .map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; (e.end - e.start) as usize];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            out.write_all(&buf).map_err(|e| e.to_string())?;
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    if n_replaced != replaced.len() {
        return Err(format!(
            "merge: {} adapted maps, {n_replaced} written",
            replaced.len()
        ));
    }
    let map: serde_json::Map<String, serde_json::Value> = entries
        .iter()
        .map(|(n, _, _)| (n.clone(), serde_json::Value::String(shard_name.into())))
        .collect();
    std::fs::write(
        out_dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"metadata": {"total_size": off}, "weight_map": map}),
        )
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for f in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "LICENSE",
        "generation_config.json",
    ] {
        if base.join(f).exists() {
            std::fs::copy(base.join(f), out_dir.join(f)).map_err(|e| format!("{f}: {e}"))?;
        }
    }
    std::fs::write(out_dir.join("README.md"), readme).map_err(|e| e.to_string())?;
    Ok(n_replaced)
}

/// Writes the model as one flat file for the iMac GPU kernels (`aienos native/kernel/gpu/kernels/allen`).
///
/// Little-endian throughout. A header of 16 `u32`: magic `AGP1`, version 1, hidden, layers, vocab,
/// intermediate, heads, kv_heads, head_dim, rotary_dim, lin_k_heads, lin_v_heads, lin_k_dim,
/// lin_v_dim, conv_kernel, the bits of eps. Then one `u32` per layer (0 DeltaNet, 1 attention).
/// Then the tensors, each a `u32` dtype (0 bf16, 1 f32), a `u32` element count and the data:
/// embed `[vocab][hidden]` bf16, final_norm f32; per layer input_norm, post_norm (f32); DeltaNet
/// qkv, z, a, b (bf16 `[n][k]`), conv `[conv_dim][kernel]`, a_log, dt_bias, norm (f32), out
/// (bf16); attention q, k, v, o (bf16), q_norm, k_norm (f32); then mlp gate, up, down (bf16).
/// Adapters are folded in (rounded to bf16 as [`save_merged`] does).
pub fn export_gpu(model: &Model, out: &Path) -> Result<(), String> {
    let c = &model.c;
    let mut f = std::io::BufWriter::new(File::create(out).map_err(|e| e.to_string())?);
    let put = |v: u32, f: &mut std::io::BufWriter<File>| {
        f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    };
    for v in [
        u32::from_le_bytes(*b"AGP1"),
        1,
        c.hidden as u32,
        c.layers as u32,
        c.vocab as u32,
        c.intermediate as u32,
        c.heads as u32,
        c.kv_heads as u32,
        c.head_dim as u32,
        c.rotary_dim as u32,
        c.lin_k_heads as u32,
        c.lin_v_heads as u32,
        c.lin_k_dim as u32,
        c.lin_v_dim as u32,
        c.conv_kernel as u32,
        c.eps.to_bits(),
    ] {
        put(v, &mut f)?;
    }
    for l in &model.layers {
        put(matches!(l.mixer, Mixer::Attn(_)) as u32, &mut f)?;
    }
    fn tensor(f: &mut std::io::BufWriter<File>, v: &[f32], bf16: bool) -> Result<(), String> {
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        e(f.write_all(&(!bf16 as u32).to_le_bytes()))?;
        e(f.write_all(&(v.len() as u32).to_le_bytes()))?;
        for &x in v {
            if bf16 {
                e(f.write_all(&f32_to_bf16(x).to_le_bytes()))?;
            } else {
                e(f.write_all(&x.to_le_bytes()))?;
            }
        }
        Ok(())
    }
    let lin = |f: &mut std::io::BufWriter<File>, l: &Lin| tensor(f, &l.merged(), true);
    tensor(&mut f, &model.embed, true)?;
    tensor(&mut f, &model.final_norm, false)?;
    for l in &model.layers {
        tensor(&mut f, &l.input_norm, false)?;
        tensor(&mut f, &l.post_norm, false)?;
        match &l.mixer {
            Mixer::Gdn(d) => {
                for m in [&d.qkv, &d.z, &d.a, &d.b] {
                    lin(&mut f, m)?;
                }
                for v in [&d.conv, &d.a_log, &d.dt_bias, &d.norm] {
                    tensor(&mut f, v, false)?;
                }
                lin(&mut f, &d.out)?;
            }
            Mixer::Attn(a) => {
                for m in [&a.q, &a.k, &a.v, &a.o] {
                    lin(&mut f, m)?;
                }
                tensor(&mut f, &a.q_norm, false)?;
                tensor(&mut f, &a.k_norm, false)?;
            }
        }
        for m in [&l.mlp.gate, &l.mlp.up, &l.mlp.down] {
            lin(&mut f, m)?;
        }
    }
    f.flush().map_err(|e| e.to_string())
}
