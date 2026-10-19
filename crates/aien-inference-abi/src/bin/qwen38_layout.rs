//! The pinned resident layout of Qwen3.8-27B (sc #676, FORTRAN idea 4): build, check and load it.
//!
//!   qwen38_layout build <checkpoint-dir> <out-dir> [<layers>]
//!       Stages every image from the checkpoint (no chip) and writes `<out-dir>/<id>.q38layout`
//!       (0444) plus `<id>.manifest.json`. `<layers>` builds a prefix only (for tests).
//!       `AIEN_Q38_LAYOUT_MUTANT_NOFOLD=<entry>` skips one norm's `1 + w` fold: the
//!       wrong-transform negative control (its id must differ from the pinned one).
//!   qwen38_layout id <pack>
//!       Prints the pack's id (read, not trusted).
//!   qwen38_layout verify <pack> [<id>]
//!       Opens against `<id>` (default: the pinned id) and hashes every image against the manifest.
//!   qwen38_layout covers <pack> <manifest.json>...
//!       Every entry of each manifest (a window a checkpoint load recorded) is in the pack with the
//!       same fingerprint.
//!   qwen38_layout load <checkpoint-dir> <first> <count> [--layout <pack>] [<out.json>]
//!       Uploads the window to the chip from the checkpoint or the pack, prints where the seconds
//!       and the resident bytes went and the window's layout id, runs one decode step.
//!
//!   qwen38_layout cached <file>... / evict <file>...
//!       Page-cache residency of files (mincore); evict drops their clean pages (posix_fadvise
//!       DONTNEED, nothing else touched) so a load is measured from the disk.
//!
//! Exit 0 pass, 1 refused or mismatched, 2 usage or I/O.

use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::layout::{self, Manifest, Pack, PackWriter, ShapeFacts};
use aien_inference_abi::qwen38_omega::Qwen38Omega;
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use serde_json::json;

const KEEP_FREE: u64 = 12 << 30;

fn mem_available() -> Result<u64, String> {
    let s = std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    s.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}

fn build(dir: &Path, out: &Path, layers: Option<usize>) -> Result<bool, String> {
    let full = Qwen38Config::from_checkpoint(dir)?;
    let c = match layers {
        None => full,
        Some(n) => full
            .prefix(n)
            .ok_or_else(|| format!("layers {n} outside the model"))?,
    };
    let mutant = layout::StageMutant {
        no_fold: std::env::var("AIEN_Q38_LAYOUT_MUTANT_NOFOLD")
            .ok()
            .filter(|s| !s.is_empty()),
    };
    if let Some(n) = &mutant.no_fold {
        eprintln!("build: MUTANT: {n} is NOT folded to 1 + w (negative control pack)");
    }
    let t = Instant::now();
    let mut w = PackWriter::create(out, ShapeFacts::of(&c, 0))?;
    let (mut images, mut bytes) = (0usize, 0u64);
    let mut src = ShardSource::open(dir)?;
    let read_s = layout::stage(&c, 0, &mut src, &mutant, &mut |img| {
        images += 1;
        bytes += img.bytes().len() as u64;
        w.add(&img)
    })?;
    let (path, id) = w.finish()?;
    let s = t.elapsed().as_secs_f64();
    println!(
        "built {} ({images} images, {bytes} bytes) in {s:.2} s (checkpoint read {read_s:.2} s)",
        path.display()
    );
    println!("layout id {id}");
    if id == layout::QWEN38_27B_FP8_017B9C7A {
        println!("matches the pinned id");
    } else {
        println!(
            "differs from the pinned id {}",
            layout::QWEN38_27B_FP8_017B9C7A
        );
    }
    Ok(true)
}

fn verify(pack: &Path, id: &str) -> Result<bool, String> {
    let t = Instant::now();
    let p = Pack::open(pack, id)?;
    let mut bytes = 0u64;
    for e in &p.manifest.entries {
        let spec = layout::EntrySpec {
            name: e.name.clone(),
            dtype: e.dtype,
            dims: e.dims.clone(),
        };
        let img = p.take(&spec, false)?;
        bytes += img.bytes().len() as u64;
        img.release();
    }
    println!(
        "verified {} images, {bytes} bytes of {} against id {id} in {:.2} s",
        p.manifest.entries.len(),
        pack.display(),
        t.elapsed().as_secs_f64()
    );
    Ok(true)
}

fn covers(pack: &Path, manifests: &[String]) -> Result<bool, String> {
    let id = layout::peek_id(pack)?;
    let p = Pack::open(pack, &id)?;
    let mut ok = true;
    for m in manifests {
        let text = std::fs::read(m).map_err(|e| format!("{m}: {e}"))?;
        let got: Manifest = serde_json::from_slice(&text).map_err(|e| format!("{m}: {e}"))?;
        match p.manifest.covers(&got) {
            Ok(n) => println!("{m}: all {n} entries match pack {id}"),
            Err(e) => {
                println!("{m}: MISMATCH {e}");
                ok = false;
            }
        }
    }
    Ok(ok)
}

fn load(args: &mut Vec<String>) -> Result<bool, String> {
    let pack = layout::take_layout_arg(args)?
        .map(|p| layout::open_for_tool(&p))
        .transpose()?;
    let (dir, first, count) = (
        Path::new(&args[2]),
        args[3].parse::<usize>().map_err(|e| e.to_string())?,
        args[4].parse::<usize>().map_err(|e| e.to_string())?,
    );
    let full = Qwen38Config::from_checkpoint(dir)?;
    let c = full
        .window(first, count)
        .ok_or_else(|| format!("window {first}+{count} outside the model"))?;
    let need: u64 = layout::plan(&c, first)
        .iter()
        .map(|s| s.bytes() as u64)
        .sum::<u64>()
        + (64 << 20);
    let avail = mem_available()?;
    if avail < need + KEEP_FREE {
        return Err(format!(
            "window {first}+{count}: needs {} MiB with {} MiB kept free, MemAvailable {} MiB; refusing",
            need >> 20,
            KEEP_FREE >> 20,
            avail >> 20
        ));
    }
    let before = layout::resident_split()?;
    let t = Instant::now();
    let mut engine = match &pack {
        Some(p) => Qwen38Omega::open_pack(
            dir,
            p,
            Some((first, count)),
            256,
            layout::flip_from_env().as_deref(),
        )?,
        None => Qwen38Omega::open_window(dir, first, count, 256)?,
    };
    let wall = t.elapsed().as_secs_f64();
    let after = layout::resident_split()?;
    let avail_after = mem_available()?;
    let ld = engine.load_stats().clone();
    let id = engine.layout().id();
    let (tok, st) = engine.step(0)?;
    println!(
        "{}: window {first}+{count} loaded in {wall:.3} s (read {:.3}, hash {:.3} (overlaps upload on the pack door), upload {:.3}); {} images {} bytes; window layout {id}",
        ld.source,
        ld.read_s,
        ld.hash_s,
        ld.upload_s,
        ld.images,
        ld.image_bytes
    );
    println!("resident before {before:?}");
    println!(
        "resident after  {after:?}; MemAvailable {} -> {} MiB",
        avail >> 20,
        avail_after >> 20
    );
    println!("one step: token {tok}, chip {} us", st.chip_ns / 1000);
    if let Some(p) = &pack {
        p.manifest.covers(engine.layout())?;
        println!("every uploaded image matches pack {}", p.id);
    }
    if let Some(out) = args.get(5) {
        let v = json!({
            "source": ld.source, "first": first, "count": count, "wall_s": wall,
            "total_s": ld.total_s, "read_s": ld.read_s, "hash_s": ld.hash_s, "upload_s": ld.upload_s,
            "images": ld.images, "image_bytes": ld.image_bytes, "window_layout_id": id,
            "resident_before": before, "resident_after": after,
            "mem_available_before": avail, "mem_available_after": avail_after,
            "step_token": tok, "step_chip_ns": st.chip_ns,
        });
        std::fs::write(out, serde_json::to_string_pretty(&v).unwrap())
            .map_err(|e| format!("{out}: {e}"))?;
        let mp = format!("{out}.manifest.json");
        std::fs::write(&mp, engine.layout().encode()).map_err(|e| format!("{mp}: {e}"))?;
    }
    Ok(true)
}

/// Bytes of `path` in the page cache (mincore over a read-only mapping) and its length.
fn cached_bytes(path: &str) -> Result<(u64, u64), String> {
    use std::os::fd::AsRawFd;
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let len = f.metadata().map_err(|e| format!("{path}: {e}"))?.len() as usize;
    if len == 0 {
        return Ok((0, 0));
    }
    // SAFETY: a fresh read-only mapping, only passed to mincore and unmapped below.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            f.as_raw_fd(),
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(format!("{path}: mmap failed"));
    }
    let page = 4096usize;
    let mut v = vec![0u8; len.div_ceil(page)];
    // SAFETY: v holds one byte per page of the mapping.
    let rc = unsafe { libc::mincore(p, len, v.as_mut_ptr()) };
    // SAFETY: p/len from the mmap above.
    unsafe { libc::munmap(p, len) };
    if rc != 0 {
        return Err(format!("{path}: mincore failed"));
    }
    let resident = v.iter().filter(|&&b| b & 1 == 1).count() as u64 * page as u64;
    Ok((resident.min(len as u64), len as u64))
}

/// `cached <file>...`: page-cache residency per file and in total.
fn cached(files: &[String]) -> Result<bool, String> {
    let (mut r, mut t) = (0u64, 0u64);
    for f in files {
        let (a, b) = cached_bytes(f)?;
        r += a;
        t += b;
    }
    println!(
        "page cache: {r} of {t} bytes resident over {} files",
        files.len()
    );
    Ok(true)
}

/// `evict <file>...`: asks the kernel to drop these files' clean pages from the page cache
/// (posix_fadvise DONTNEED; other files and other processes' memory are untouched), so a load
/// can be measured from the disk.
fn evict(files: &[String]) -> Result<bool, String> {
    use std::os::fd::AsRawFd;
    for f in files {
        let file = std::fs::File::open(f).map_err(|e| format!("{f}: {e}"))?;
        // SAFETY: plain advice call on an open descriptor.
        let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if rc != 0 {
            return Err(format!("{f}: posix_fadvise {rc}"));
        }
    }
    cached(files)
}

fn run() -> Result<bool, String> {
    let mut a: Vec<String> = std::env::args().collect();
    let usage = "usage: qwen38_layout build <checkpoint-dir> <out-dir> [<layers>] | id <pack> | verify <pack> [<id>] | covers <pack> <manifest.json>... | load <checkpoint-dir> <first> <count> [--layout <pack>] [<out.json>] | cached <file>... | evict <file>...";
    match a.get(1).map(String::as_str) {
        Some("build") if a.len() == 4 || a.len() == 5 => {
            let layers = a
                .get(4)
                .map(|s| s.parse::<usize>().map_err(|_| usage.to_string()))
                .transpose()?;
            build(Path::new(&a[2]), Path::new(&a[3]), layers)
        }
        Some("id") if a.len() == 3 => {
            println!("{}", layout::peek_id(Path::new(&a[2]))?);
            Ok(true)
        }
        Some("verify") if a.len() == 3 || a.len() == 4 => {
            let id = a
                .get(3)
                .cloned()
                .unwrap_or_else(|| layout::QWEN38_27B_FP8_017B9C7A.to_string());
            verify(Path::new(&a[2]), &id)
        }
        Some("covers") if a.len() >= 4 => covers(Path::new(&a[2]), &a[3..]),
        Some("load") if a.len() >= 5 => load(&mut a),
        Some("cached") if a.len() >= 3 => cached(&a[2..]),
        Some("evict") if a.len() >= 3 => evict(&a[2..]),
        _ => Err(usage.into()),
    }
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen38_layout: {e}");
            std::process::exit(if e.starts_with(layout::LAYOUT_REFUSAL_PREFIX) {
                1
            } else {
                2
            });
        }
    }
}
