//! L308 typewriter experts, the smallest full loop on the chip (H3, the heartbeat, one swap).
//!
//!   typewriter_chip <checkpoint> <layers> <tokens> <swap-every> <work-dir> <out.json>
//!
//! Three tiers, one beat (one decode token):
//! - math cores: the engine's kernels, built at load; nothing is compiled or allocated per beat.
//! - fast memory: every expert of the first `layers` layers is planted a second time, in a
//!   different order, in a separate GPU region (a "type head"); the device router's directory is
//!   pointed at it. Routing is the model's own, so every logit must stay bit-identical.
//! - big base: the next head lives on NVMe in a file packed in beat order (layer by layer, in the
//!   exact upload layout). The ring thread streams it into host memory while the beats run; at a
//!   swap the streamed bytes land in a new GPU region and the directory is repointed.
//!
//! Runs, same prompt, greedy: A today's path (experts where the loader put them); B head pool,
//! no ring; C ring streaming, no swap; D ring streaming + a head swap every `swap-every` beats.
//! Gate: tokens and logit bits of B, C, D equal A. Report: every beat's time per run, the ring's
//! bytes and rate, each swap's landing and repoint time. The truncated model's tokens are not the
//! 30B's tokens; only their equality across placements is the gate.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use aien_inference_abi::qwen3_moe_omega::{
    AttnPath, DevicePaths, ExpertWeights, Fp8Gemv, ModelUploader, OmegaModelUploader,
    Qwen3MoeConfig, Qwen3MoeOmegaEngine, ShardSource, TensorSource, WeightFormat,
};
use serde_json::json;

const BLOCK: usize = 128;

/// One planted head for one layer: `order[slot]` is the expert held in that slot.
struct Pool {
    gu: Arc<dyn Fp8Gemv>,
    dn: Arc<dyn Fp8Gemv>,
    order: Vec<usize>,
}

/// One layer's experts in upload layout, in `order`: gate_up fp8 `[n][2i][h]`, its scales
/// `[n][2i/128][h/128]`, down fp8 `[n][h][i]`, its scales `[n][h/128][i/128]`.
struct LayerBytes {
    gu: Vec<u8>,
    gu_s: Vec<u16>,
    dn: Vec<u8>,
    dn_s: Vec<u16>,
}

fn as_bytes(v: &mut [u16]) -> &mut [u8] {
    // SAFETY: a u16 slice viewed as twice as many bytes; little-endian host, little-endian file.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 2) }
}

fn read_layer(src: &mut ShardSource, c: &Qwen3MoeConfig, l: usize, order: &[usize]) -> Result<LayerBytes, String> {
    let (h, i, n) = (c.hidden, c.moe_intermediate, order.len());
    let (ib, kb) = (i / BLOCK, h / BLOCK);
    let mut b = LayerBytes {
        gu: vec![0; n * 2 * i * h],
        gu_s: vec![0; n * 2 * ib * kb],
        dn: vec![0; n * h * i],
        dn_s: vec![0; n * kb * ib],
    };
    let p = format!("model.layers.{l}.mlp.experts");
    for (slot, &ex) in order.iter().enumerate() {
        for (j, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
            let (w0, s0) = ((slot * 2 + j) * i * h, (slot * 2 + j) * ib * kb);
            src.read_into(&format!("{p}.{ex}.{proj}.weight"), "F8_E4M3", &[i, h], &mut b.gu[w0..w0 + i * h])?;
            src.read_into(&format!("{p}.{ex}.{proj}.weight_scale_inv"), "BF16", &[ib, kb], as_bytes(&mut b.gu_s[s0..s0 + ib * kb]))?;
        }
        let (w0, s0) = (slot * h * i, slot * kb * ib);
        src.read_into(&format!("{p}.{ex}.down_proj.weight"), "F8_E4M3", &[h, i], &mut b.dn[w0..w0 + h * i])?;
        src.read_into(&format!("{p}.{ex}.down_proj.weight_scale_inv"), "BF16", &[kb, ib], as_bytes(&mut b.dn_s[s0..s0 + kb * ib]))?;
    }
    Ok(b)
}

fn plant(c: &Qwen3MoeConfig, b: &LayerBytes, order: &[usize]) -> Result<Pool, String> {
    let (h, i, n) = (c.hidden, c.moe_intermediate, order.len());
    let up = OmegaModelUploader;
    Ok(Pool {
        gu: up.upload_fp8(n, 2 * i, h, &b.gu, &b.gu_s)?,
        dn: up.upload_fp8(n, h, i, &b.dn, &b.dn_s)?,
        order: order.to_vec(),
    })
}

/// Points layer `l`'s router directory at `pool` (experts it holds) or the loader's tensors.
fn point(engine: &Qwen3MoeOmegaEngine, l: usize, pool: Option<&Pool>) -> Result<(), String> {
    let w = engine.weights();
    let router = w.device_router.as_ref().ok_or("no device router: the directory seam is missing")?;
    let ExpertWeights::Fp8 { gate_up, down } = &w.layers[l].experts else {
        return Err(format!("layer {l}: experts are not FP8"));
    };
    let e = w.config.experts;
    let mut slot_of = vec![usize::MAX; e];
    if let Some(p) = pool {
        p.order.iter().enumerate().for_each(|(s, &ex)| slot_of[ex] = s);
    }
    let (mut gu, mut dn) = (Vec::with_capacity(2 * e), Vec::with_capacity(2 * e));
    for ex in 0..e {
        let ((gw, gs), (dw, ds)) = match (pool, slot_of[ex]) {
            (Some(p), s) if s != usize::MAX => (p.gu.tensor_entry(s)?, p.dn.tensor_entry(s)?),
            _ => (gate_up.tensor_entry(ex)?, down.tensor_entry(ex)?),
        };
        gu.extend([gw, gs]);
        dn.extend([dw, ds]);
    }
    router.set_directory(l, &gu, &dn)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3))
}

#[cfg(target_arch = "aarch64")]
const O_DIRECT: i32 = 0o200000;
#[cfg(not(target_arch = "aarch64"))]
const O_DIRECT: i32 = 0o40000;
const CHUNK: usize = 64 << 20;

/// The ring: streams `path` with O_DIRECT into `dst` (host memory) over and over until `stop`;
/// counts bytes. One pass = the next head in beat order.
fn ring(path: PathBuf, len: usize, stop: Arc<AtomicBool>, bytes: Arc<AtomicU64>, passes: Arc<AtomicU64>) -> Result<Vec<u8>, String> {
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    let f = std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(&path).map_err(|e| e.to_string())?;
    let padded = len.div_ceil(4096) * 4096;
    let layout = std::alloc::Layout::from_size_align(padded, 4096).map_err(|e| e.to_string())?;
    // SAFETY: non-zero size, 4096 alignment; zeroed so pages are faulted before timing.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err("ring buffer: out of memory".into());
    }
    // SAFETY: ptr owns `padded` bytes until dealloc below.
    let buf = unsafe { std::slice::from_raw_parts_mut(ptr, padded) };
    let mut out = Ok(());
    'outer: while !stop.load(Ordering::Relaxed) {
        let mut off = 0usize;
        while off < padded {
            if stop.load(Ordering::Relaxed) {
                break 'outer;
            }
            let n = (padded - off).min(CHUNK);
            match f.read_at(&mut buf[off..off + n], off as u64) {
                Ok(0) => break,
                Ok(got) => {
                    off += got;
                    bytes.fetch_add(got as u64, Ordering::Relaxed);
                }
                Err(e) => {
                    out = Err(e.to_string());
                    break 'outer;
                }
            }
        }
        passes.fetch_add(1, Ordering::Relaxed);
    }
    let v = buf[..len].to_vec();
    // SAFETY: allocated above with this layout.
    unsafe { std::alloc::dealloc(ptr, layout) };
    out.map(|_| v)
}

struct Run {
    tokens: Vec<u32>,
    hashes: Vec<u64>,
    beats_ns: Vec<u64>,
    swaps: Vec<serde_json::Value>,
}

fn decode(
    engine: &mut Qwen3MoeOmegaEngine,
    prompt: &[u32],
    n: usize,
    mut at_beat: impl FnMut(usize, &mut Qwen3MoeOmegaEngine) -> Result<Option<serde_json::Value>, String>,
) -> Result<Run, String> {
    engine.reset();
    let mut hidden = Vec::new();
    for &t in prompt {
        hidden = engine.step(t)?;
    }
    let mut logits = engine.logits(&hidden)?;
    let mut r = Run { tokens: Vec::new(), hashes: Vec::new(), beats_ns: Vec::new(), swaps: Vec::new() };
    for s in 0..n {
        let next = argmax(&logits);
        r.tokens.push(next);
        r.hashes.push(fnv(&logits));
        if let Some(sw) = at_beat(s, engine)? {
            r.swaps.push(sw);
        }
        let t = Instant::now();
        hidden = engine.step(next)?;
        logits = engine.logits(&hidden)?;
        r.beats_ns.push(t.elapsed().as_nanos() as u64);
    }
    Ok(r)
}

fn summary(b: &[u64]) -> serde_json::Value {
    let mut s: Vec<u64> = b.to_vec();
    s.sort_unstable();
    let n = s.len().max(1);
    let mean = s.iter().sum::<u64>() as f64 / n as f64;
    let var = s.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let q = |p: f64| s[((n - 1) as f64 * p).round() as usize] as f64 / 1e3;
    json!({"beats": s.len(), "mean_us": mean / 1e3, "sd_us": var.sqrt() / 1e3, "min_us": q(0.0),
        "p50_us": q(0.5), "p90_us": q(0.9), "p99_us": q(0.99), "max_us": q(1.0)})
}

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1e6)
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: typewriter_chip <checkpoint> <layers> <tokens> <swap-every> <work-dir> <out.json>".into());
    }
    let dir = Path::new(&a[1]);
    let layers: usize = a[2].parse().map_err(|_| "layers")?;
    let n: usize = a[3].parse().map_err(|_| "tokens")?;
    let every: usize = a[4].parse().map_err(|_| "swap-every")?;
    let work = PathBuf::from(&a[5]);
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let mut c = Qwen3MoeConfig::from_checkpoint(dir)?;
    c.layers = c.layers.min(layers);
    let e = c.experts;
    let layer_bytes = e * 3 * c.moe_intermediate * c.hidden;
    let need_gb = (c.vocab * c.hidden * 4 + layer_bytes * c.layers * 5) as f64 / 1e9 + 4.0;
    let avail = mem_available_gb();
    eprintln!("memory: need about {need_gb:.1} GB, available {avail:.1} GB");
    if avail < need_gb + 8.0 {
        return Err(format!("REFUSED: MemAvailable {avail:.1} GB < need {need_gb:.1} + 8 GB margin (the live server shares this memory)"));
    }
    // Two orders: pool A reversed, pool B rotated by half; both hold every expert of the layer.
    let order_a: Vec<usize> = (0..e).rev().collect();
    let order_b: Vec<usize> = (0..e).map(|x| (x + e / 2) % e).collect();
    // The next head on NVMe, packed in beat order in the exact upload layout (written once).
    let mut src = ShardSource::open(dir)?;
    let packed = work.join(format!("head-b-{layers}l.bin"));
    let mut sizes = Vec::new();
    {
        let mut all = Vec::new();
        for l in 0..c.layers {
            let b = read_layer(&mut src, &c, l, &order_b)?;
            sizes.push((b.gu.len(), b.gu_s.len(), b.dn.len(), b.dn_s.len()));
            all.extend_from_slice(&b.gu);
            all.extend(b.gu_s.iter().flat_map(|x| x.to_le_bytes()));
            all.extend_from_slice(&b.dn);
            all.extend(b.dn_s.iter().flat_map(|x| x.to_le_bytes()));
        }
        let padded = all.len().div_ceil(4096) * 4096;
        if std::fs::metadata(&packed).map(|m| m.len() as usize).ok() != Some(padded) {
            all.resize(padded, 0);
            std::fs::write(&packed, &all).map_err(|e| e.to_string())?;
        }
    }
    let head_bytes: usize = sizes.iter().map(|s| s.0 + 2 * s.1 + s.2 + 2 * s.3).sum();

    let t0 = Instant::now();
    let (mut engine, _report) = Qwen3MoeOmegaEngine::load_omega_layers(
        dir, Some(layers), 4096, WeightFormat::Fp8All, AttnPath::Auto, DevicePaths::default(), &mut |_| {},
    )?;
    let load_s = t0.elapsed().as_secs_f64();
    if engine.weights().device_router.is_none() {
        return Err("the device router is off: no directory to repoint".into());
    }
    let prompt: Vec<u32> = vec![785, 943, 315, 279, 4161, 374, 11, 323, 1817, 3409, 3171, 1550, 13, 576];
    // Pool A planted ahead of time (from the checkpoint, not timed as a beat).
    let t = Instant::now();
    let pool_a: Vec<Pool> = (0..c.layers)
        .map(|l| read_layer(&mut src, &c, l, &order_a).and_then(|b| plant(&c, &b, &order_a)))
        .collect::<Result<_, _>>()?;
    let plant_a_s = t.elapsed().as_secs_f64();

    // A: today's path.
    let ra = decode(&mut engine, &prompt, n, |_, _| Ok(None))?;
    // B: head pool A, no ring.
    for (l, p) in pool_a.iter().enumerate() {
        point(&engine, l, Some(p))?;
    }
    let rb = decode(&mut engine, &prompt, n, |_, _| Ok(None))?;
    // C: head pool A + ring streaming the next head from NVMe the whole time.
    let stop = Arc::new(AtomicBool::new(false));
    let (bytes, passes) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let start_ring = |stop: &Arc<AtomicBool>, bytes: &Arc<AtomicU64>, passes: &Arc<AtomicU64>| {
        let (p, s, b, q) = (packed.clone(), stop.clone(), bytes.clone(), passes.clone());
        std::thread::spawn(move || ring(p, head_bytes, s, b, q))
    };
    let h = start_ring(&stop, &bytes, &passes);
    let tc = Instant::now();
    let rc = decode(&mut engine, &prompt, n, |_, _| Ok(None))?;
    let c_s = tc.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let streamed = h.join().map_err(|_| "ring panicked")??;
    let c_bytes = bytes.swap(0, Ordering::Relaxed);
    let c_passes = passes.swap(0, Ordering::Relaxed);
    // D: ring streaming + a swap every `every` beats: land the streamed head (B order) in new GPU
    // regions, repoint, and on the next swap go back to pool A.
    stop.store(false, Ordering::Relaxed);
    let h = start_ring(&stop, &bytes, &passes);
    let mut landed: Option<Vec<Pool>> = None;
    let mut on_b = false;
    let streamed_ref = &streamed;
    let td = Instant::now();
    let rd = decode(&mut engine, &prompt, n, |s, eng| {
        if s == 0 || s % every != 0 {
            return Ok(None);
        }
        let t = Instant::now();
        if landed.is_none() {
            let mut off = 0usize;
            let mut pools = Vec::with_capacity(sizes.len());
            for &(g, gs, d, ds) in &sizes {
                let take = |off: &mut usize, len: usize| {
                    let s = &streamed_ref[*off..*off + len];
                    *off += len;
                    s
                };
                let gu = take(&mut off, g).to_vec();
                let gu_s: Vec<u16> = take(&mut off, 2 * gs).chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
                let dn = take(&mut off, d).to_vec();
                let dn_s: Vec<u16> = take(&mut off, 2 * ds).chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
                pools.push(plant(&c, &LayerBytes { gu, gu_s, dn, dn_s }, &order_b)?);
            }
            landed = Some(pools);
        }
        let land_ns = t.elapsed().as_nanos() as u64;
        let t = Instant::now();
        on_b = !on_b;
        for l in 0..c.layers {
            let p = if on_b { &landed.as_ref().expect("landed")[l] } else { &pool_a[l] };
            point(eng, l, Some(p))?;
        }
        let repoint_ns = t.elapsed().as_nanos() as u64;
        Ok(Some(json!({"beat": s, "to": if on_b { "B (landed from the ring)" } else { "A" }, "land_us": land_ns as f64 / 1e3, "repoint_us": repoint_ns as f64 / 1e3})))
    })?;
    let d_s = td.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    h.join().map_err(|_| "ring panicked")??;
    let d_bytes = bytes.load(Ordering::Relaxed);
    // Back to today's path for whoever uses the engine next.
    for l in 0..c.layers {
        point(&engine, l, None)?;
    }
    let same = |r: &Run| r.tokens == ra.tokens && r.hashes == ra.hashes;
    let gate = same(&rb) && same(&rc) && same(&rd);
    // Negative control: one wrong directory entry (layer 0 expert 0 -> expert 1's bytes) must move the logits.
    let ctrl = {
        let w = engine.weights();
        let router = w.device_router.as_ref().expect("router");
        let ExpertWeights::Fp8 { gate_up, down } = &w.layers[0].experts else { unreachable!() };
        let (mut gu, mut dn) = (Vec::new(), Vec::new());
        for ex in 0..e {
            let src_ex = if ex == 0 { 1 } else { ex };
            let (a1, a2) = gate_up.tensor_entry(src_ex)?;
            let (b1, b2) = down.tensor_entry(src_ex)?;
            gu.extend([a1, a2]);
            dn.extend([b1, b2]);
        }
        router.set_directory(0, &gu, &dn)?;
        let rx = decode(&mut engine, &prompt, n, |_, _| Ok(None))?;
        point(&engine, 0, None)?;
        rx.hashes != ra.hashes
    };
    let gbs = |b: u64, s: f64| b as f64 / s / 1e9;
    let out = json!({
        "mode": "typewriter_chip", "checkpoint": dir, "layers": c.layers, "experts": e, "tokens": n,
        "swap_every": every, "load_seconds": load_s, "plant_a_seconds": plant_a_s,
        "head_bytes_per_swap": head_bytes, "packed_file": packed,
        "gate_identical_tokens_and_logit_bits": gate,
        "identical": {"B": same(&rb), "C": same(&rc), "D": same(&rd)},
        "negative_control_wrong_directory_moves_logits": ctrl,
        "A_today": summary(&ra.beats_ns), "B_head_pool": summary(&rb.beats_ns),
        "C_head_pool_ring": summary(&rc.beats_ns), "D_ring_and_swaps": summary(&rd.beats_ns),
        "ring": {"C_bytes": c_bytes, "C_passes": c_passes, "C_gb_s": gbs(c_bytes, c_s),
                  "D_bytes": d_bytes, "D_gb_s": gbs(d_bytes, d_s)},
        "swaps": rd.swaps,
        "beats_ns": {"A": ra.beats_ns, "B": rb.beats_ns, "C": rc.beats_ns, "D": rd.beats_ns},
        "tokens": ra.tokens,
    });
    std::fs::write(&a[6], serde_json::to_vec_pretty(&out).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    eprintln!("gate {} control {} | A {} | B {} | C {} | D {}", gate, ctrl, out["A_today"], out["B_head_pool"], out["C_head_pool_ring"], out["D_ring_and_swaps"]);
    if !gate || !ctrl {
        return Err("GATE FAILED (see out.json)".into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("typewriter_chip: {e}");
        std::process::exit(1);
    }
}
