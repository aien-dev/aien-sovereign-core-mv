//! Document memory pieces measured without the model (cascade/document-memory):
//!
//! - `kv ROWS`: the device KV cache copies the store uses: rows `0 .. ROWS` of 16 attention
//!   layers x 4 KV heads read out one row per call (`omega_gpu_decode_attn256_cache_read`) and
//!   written back one call per (layer, head) (`..._cache_write`), checked bit for bit. Needs the
//!   chip (a 16-layer cache of ROWS positions, no weights): run it under the GPU lock.
//! - `disk ROWS CKPTS DIR`: a conversation of ROWS KV rows and CKPTS DeltaNet checkpoints (the
//!   27B's shapes, synthetic bytes) written to DIR with `O_DIRECT` and loaded back, checked bit
//!   for bit (page cache bypassed both ways, so the load reads the NVMe).
//!
//! Prints one JSON line per measurement.
use std::path::PathBuf;
use std::time::Instant;

use aien_inference_abi::qwen38_omega::Qwen38Snapshot;
use aien_serve_http::conv_store::{
    CkptKind, ConvStore, EntryStatus, Shape, StoreConfig, DEFAULT_IO_THREADS, KV_HEADS,
};

const N_ATTN: usize = 16;
const N_GDN: usize = 48;

fn shape(max_pos: usize) -> Shape {
    Shape {
        n_attn: N_ATTN,
        n_gdn: N_GDN,
        tail_floats: aien_omega_q38::GDN_TAIL_FLOATS,
        state_floats: aien_omega_q38::GDN_STATE_FLOATS,
        max_pos,
    }
}

fn word(layer: usize, head: usize, plane: usize, i: usize) -> u16 {
    ((layer * 7919 + head * 104_729 + plane * 31 + i * 2_654_435_761) % 65_521) as u16
}

fn kv(rows: usize) -> Result<(), String> {
    let max_pos = rows.next_multiple_of(256);
    let mut c = aien_omega_q38::Attn256Cache::alloc(N_ATTN, max_pos, 1e7, 64)
        .map_err(|e| format!("alloc: {e}"))?;
    let w = aien_omega_q38::HEAD_DIM;
    let t = Instant::now();
    for layer in 0..N_ATTN {
        for head in 0..KV_HEADS {
            let k: Vec<u16> = (0..rows * w).map(|i| word(layer, head, 0, i)).collect();
            let v: Vec<u16> = (0..rows * w).map(|i| word(layer, head, 1, i)).collect();
            c.write_rows(layer, head, 0, &k, &v)
                .map_err(|e| format!("write: {e}"))?;
        }
    }
    let fill_s = t.elapsed().as_secs_f64();
    // Read back as the store's export does (one row per call), then compare.
    let t = Instant::now();
    let mut back = Vec::with_capacity(N_ATTN * KV_HEADS);
    for layer in 0..N_ATTN {
        for head in 0..KV_HEADS {
            let (mut k, mut v) = (vec![0u16; rows * w], vec![0u16; rows * w]);
            for p in 0..rows {
                c.read_row(layer, head, p, &mut k[p * w..(p + 1) * w], &mut v[p * w..(p + 1) * w])
                    .map_err(|e| format!("read: {e}"))?;
            }
            back.push((k, v));
        }
    }
    let read_s = t.elapsed().as_secs_f64();
    let mut bad = 0usize;
    for (j, (k, v)) in back.iter().enumerate() {
        let (layer, head) = (j / KV_HEADS, j % KV_HEADS);
        bad += k.iter().enumerate().filter(|&(i, &x)| x != word(layer, head, 0, i)).count();
        bad += v.iter().enumerate().filter(|&(i, &x)| x != word(layer, head, 1, i)).count();
    }
    // Write the read bytes back (the restore path) and read a sample again.
    let t = Instant::now();
    for (j, (k, v)) in back.iter().enumerate() {
        c.write_rows(j / KV_HEADS, j % KV_HEADS, 0, k, v)
            .map_err(|e| format!("write: {e}"))?;
    }
    let write_s = t.elapsed().as_secs_f64();
    let bytes = (N_ATTN * KV_HEADS * 2 * rows * 512) as f64;
    println!(
        "{}",
        serde_json::json!({
            "what": "kv device copies", "rows": rows, "bytes": bytes as u64,
            "fill_s": fill_s, "export_read_s": read_s, "export_gb_s": bytes / 1e9 / read_s,
            "restore_write_s": write_s, "restore_gb_s": bytes / 1e9 / write_s,
            "mismatched_words": bad,
        })
    );
    if bad != 0 {
        return Err(format!("{bad} words differ after the round trip"));
    }
    Ok(())
}

fn disk(rows: usize, ckpts: usize, dir: PathBuf) -> Result<(), String> {
    let sh = shape(rows.next_multiple_of(256));
    let st = ConvStore::open(StoreConfig {
        dir: Some(dir.clone()),
        ram_budget: u64::MAX,
        disk_budget: u64::MAX,
        checkpoint_every: 4096,
        namespace: "docmem-bench".into(),
        shape: sh.clone(),
        io_threads: DEFAULT_IO_THREADS,
        direct_io: true,
    })?;
    let tokens: Vec<u32> = (0..rows as u32).collect();
    let key = st.create(&tokens, Some("bench".into()), EntryStatus::Ready);
    let tf = N_GDN * sh.tail_floats;
    let sf = N_GDN * sh.state_floats;
    st.with_entry(&key, |e| -> Result<(), String> {
        let kv = e.kv.as_mut().ok_or("not resident")?;
        kv.append_rows(rows, |l, h, p, w| {
            for (i, x) in w.iter_mut().enumerate() {
                *x = word(l, h, p, i);
            }
            Ok(())
        })?;
        e.kv_rows = rows;
        for c in 0..ckpts {
            let pos = rows * (c + 1) / ckpts;
            let tails = (0..tf).map(|i| (i + c) as f32 * 0.5).collect();
            let states = (0..sf).map(|i| (i ^ c) as f32 * 0.25).collect();
            e.put_checkpoint(CkptKind::Chunk, Qwen38Snapshot::from_parts(pos, tails, states));
        }
        Ok(())
    })?
    .ok_or("entry vanished")??;
    st.unpin();
    let t = Instant::now();
    st.spill(&key)?;
    let spill_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let check = st
        .with_entry(&key, |e| -> (usize, usize) {
            let kv = e.kv.as_ref().expect("resident");
            let mut bad = 0;
            for l in 0..N_ATTN {
                for h in 0..KV_HEADS {
                    for p in 0..2 {
                        bad += kv
                            .plane_words(l, h, p, 0, rows)
                            .iter()
                            .enumerate()
                            .filter(|&(i, &x)| x != word(l, h, p, i))
                            .count();
                    }
                }
            }
            (bad, e.checkpoints.len())
        })?
        .ok_or("entry vanished")?;
    let load_s_with_check = t.elapsed().as_secs_f64();
    let status = st.status();
    let c = &status["counters"];
    let load_s = c["load_ns"].as_u64().unwrap_or(0) as f64 / 1e9;
    let file_bytes = status["entries"][0]["file_bytes"].as_u64().unwrap_or(0);
    println!(
        "{}",
        serde_json::json!({
            "what": "conversation spill and load, O_DIRECT", "rows": rows, "checkpoints": ckpts,
            "file_bytes": file_bytes, "spill_s": spill_s, "spill_gb_s": file_bytes as f64 / 1e9 / spill_s,
            "load_s": load_s, "load_gb_s": file_bytes as f64 / 1e9 / load_s,
            "load_plus_check_s": load_s_with_check,
            "mismatched_words": check.0, "checkpoints_back": check.1,
            "dir": dir.display().to_string(),
        })
    );
    st.remove(&key);
    if check.0 != 0 || check.1 != ckpts {
        return Err("round trip differs".into());
    }
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("kv") => a
            .get(2)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| "usage: docmem_bench kv ROWS".to_string())
            .and_then(kv),
        Some("disk") => match (a.get(2).and_then(|s| s.parse().ok()), a.get(3).and_then(|s| s.parse().ok()), a.get(4)) {
            (Some(r), Some(c), Some(d)) => disk(r, c, PathBuf::from(d)),
            _ => Err("usage: docmem_bench disk ROWS CKPTS DIR".into()),
        },
        _ => Err("usage: docmem_bench kv ROWS | disk ROWS CKPTS DIR".into()),
    };
    if let Err(e) = r {
        eprintln!("docmem_bench: {e}");
        std::process::exit(1);
    }
}
