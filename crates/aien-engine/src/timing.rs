//! The timing diagram. Per beat four stamps in nanoseconds from the run's t0 (CLOCK_MONOTONIC through `Instant`):
//! fill issued and fill done (movement engine, for the buffer this beat fires on), fire start and fire done (power
//! engine). Ignition gap = fire_start - fill_done: positive means the fuel sat waiting (early), negative means the
//! piston waited for fuel (late, a misfire). Text table + PNG written by this module alone.
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BeatTrace {
    pub b: u32,
    pub token: u32,
    pub layer: u16,
    pub buffer: u32,
    pub bytes: u64,
    pub fill_issue_ns: i64,
    pub fill_done_ns: i64,
    pub fire_start_ns: i64,
    pub fire_done_ns: i64,
    pub misfire: bool,
}

impl BeatTrace {
    pub fn gap_ns(&self) -> i64 {
        self.fire_start_ns - self.fill_done_ns
    }
    pub fn fill_ns(&self) -> i64 {
        self.fill_done_ns - self.fill_issue_ns
    }
    pub fn fire_ns(&self) -> i64 {
        self.fire_done_ns - self.fire_start_ns
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Series {
    pub n: usize,
    pub min: f64,
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
    pub mean: f64,
    /// (p90 - p10) / median: the spread Drake asked for beside every mean.
    pub spread: f64,
}

pub fn series(v: &[f64]) -> Series {
    if v.is_empty() {
        return Series::default();
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    let median = q(0.5);
    Series {
        n: s.len(),
        min: s[0],
        p10: q(0.1),
        median,
        p90: q(0.9),
        max: s[s.len() - 1],
        mean: s.iter().sum::<f64>() / s.len() as f64,
        spread: if median != 0.0 { (q(0.9) - q(0.1)) / median.abs() } else { 0.0 },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Summary {
    pub beats: usize,
    pub misfires: usize,
    pub bytes_per_beat: Series,
    pub fill_us: Series,
    pub fire_us: Series,
    pub beat_us: Series,
    pub gap_us: Series,
    /// Piston idle before each fire: fire_start(b) - fire_done(b-1). Zero = the piston never waited for fuel.
    pub idle_us: Series,
    pub late_beats: usize,
    pub payload_gbps_during_fill: f64,
    pub beats_per_s: f64,
}

pub fn summarize(t: &[BeatTrace]) -> Summary {
    let us = |x: i64| x as f64 / 1e3;
    let fill: Vec<f64> = t.iter().map(|b| us(b.fill_ns())).collect();
    let fire: Vec<f64> = t.iter().map(|b| us(b.fire_ns())).collect();
    let gap: Vec<f64> = t.iter().map(|b| us(b.gap_ns())).collect();
    let beat: Vec<f64> = t.windows(2).map(|w| us(w[1].fire_start_ns - w[0].fire_start_ns)).collect();
    let idle: Vec<f64> = t.windows(2).map(|w| us(w[1].fire_start_ns - w[0].fire_done_ns)).collect();
    let bytes: Vec<f64> = t.iter().map(|b| b.bytes as f64).collect();
    let fill_s: f64 = t.iter().map(|b| b.fill_ns() as f64).sum::<f64>() / 1e9;
    let total_bytes: f64 = bytes.iter().sum();
    let span_s = t.last().map(|l| (l.fire_done_ns - t[0].fire_start_ns) as f64 / 1e9).unwrap_or(0.0);
    Summary {
        beats: t.len(),
        misfires: t.iter().filter(|b| b.misfire).count(),
        bytes_per_beat: series(&bytes),
        fill_us: series(&fill),
        fire_us: series(&fire),
        beat_us: series(&beat),
        gap_us: series(&gap),
        idle_us: series(&idle),
        late_beats: t.iter().filter(|b| b.gap_ns() < 0).count(),
        payload_gbps_during_fill: if fill_s > 0.0 { total_bytes / fill_s / 1e9 } else { 0.0 },
        beats_per_s: if span_s > 0.0 { t.len() as f64 / span_s } else { 0.0 },
    }
}

/// Text timing diagram: one line per beat, two interleaved traces (fill, fire) on a shared time axis.
pub fn text_diagram(t: &[BeatTrace], width: usize, max_rows: usize) -> String {
    let mut out = String::new();
    if t.is_empty() {
        return out;
    }
    let t0 = t[0].fill_issue_ns.min(t[0].fire_start_ns);
    let t1 = t.last().unwrap().fire_done_ns;
    let span = (t1 - t0).max(1) as f64;
    let col = |ns: i64| (((ns - t0) as f64 / span) * (width - 1) as f64) as usize;
    out.push_str(&format!("{:>6} {:>5} {:>4} {:>3} {:>10} {:>10} {:>10} {}\n", "beat", "tok", "lay", "buf", "fill_us", "fire_us", "gap_us", "timeline: = fill  # fire  (left to right = time)"));
    let step = (t.len() + max_rows - 1) / max_rows;
    for b in t.iter().step_by(step.max(1)) {
        let mut line = vec![b' '; width];
        let (f0, f1) = (col(b.fill_issue_ns), col(b.fill_done_ns));
        let (p0, p1) = (col(b.fire_start_ns), col(b.fire_done_ns));
        for c in f0..=f1.min(width - 1) {
            line[c] = b'=';
        }
        for c in p0..=p1.min(width - 1) {
            line[c] = if line[c] == b'=' { b'X' } else { b'#' };
        }
        out.push_str(&format!(
            "{:>6} {:>5} {:>4} {:>3} {:>10.1} {:>10.1} {:>10.1} |{}|{}\n",
            b.b,
            b.token,
            b.layer,
            b.buffer,
            b.fill_ns() as f64 / 1e3,
            b.fire_ns() as f64 / 1e3,
            b.gap_ns() as f64 / 1e3,
            String::from_utf8_lossy(&line),
            if b.misfire { " MISFIRE" } else { "" }
        ));
    }
    out
}

/// PNG plot: x = beat, two bands per beat (top: fill issue..done, bottom: fire start..done) as time offsets within
/// the beat window, and a gap line. No outside library: stored (uncompressed) zlib blocks, our own CRC and Adler.
pub fn png_diagram(t: &[BeatTrace], path: &Path) -> Result<(), String> {
    const W: usize = 1200;
    const H: usize = 400;
    let mut img = vec![255u8; W * H * 3];
    let mut put = |x: usize, y: usize, c: [u8; 3]| {
        if x < W && y < H {
            let i = (y * W + x) * 3;
            img[i..i + 3].copy_from_slice(&c);
        }
    };
    if t.is_empty() {
        return Err("empty trace".into());
    }
    let n = t.len();
    // time axis per beat: from this beat's fill issue to its fire done, normalised to the largest such window
    let win: Vec<f64> = t.iter().map(|b| (b.fire_done_ns - b.fill_issue_ns.min(b.fire_start_ns)).max(1) as f64).collect();
    let wmax = win.iter().cloned().fold(0.0, f64::max);
    let gaps: Vec<f64> = t.iter().map(|b| b.gap_ns() as f64).collect();
    let gmax = gaps.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs())).max(1.0);
    let top_h = 240usize;
    for (i, b) in t.iter().enumerate() {
        let x0 = i * (W - 40) / n + 20;
        let x1 = ((i + 1) * (W - 40) / n + 20).max(x0 + 1);
        let base = b.fill_issue_ns.min(b.fire_start_ns);
        let ypx = |ns: i64| top_h - ((ns - base) as f64 / wmax * (top_h - 10) as f64) as usize;
        let (yf0, yf1) = (ypx(b.fill_issue_ns), ypx(b.fill_done_ns));
        let (yp0, yp1) = (ypx(b.fire_start_ns), ypx(b.fire_done_ns));
        for x in x0..x1 {
            for y in yf1.min(yf0)..=yf0.max(yf1) {
                put(x, y, [40, 90, 200]); // fill: blue
            }
            for y in yp1.min(yp0)..=yp0.max(yp1) {
                put(x, y, if b.misfire { [220, 30, 30] } else { [230, 140, 20] }); // fire: orange; misfire: red
            }
        }
        // gap panel
        let mid = top_h + 80;
        let gy = mid - (gaps[i] / gmax * 60.0) as isize as usize;
        for x in x0..x1 {
            put(x, mid, [160, 160, 160]);
            put(x, gy, if gaps[i] < 0.0 { [220, 30, 30] } else { [20, 140, 60] });
        }
    }
    // axes
    for x in 0..W {
        put(x, top_h + 1, [0, 0, 0]);
        put(x, H - 1, [0, 0, 0]);
    }
    write_png(path, W as u32, H as u32, &img)
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn write_png(path: &Path, w: u32, h: u32, rgb: &[u8]) -> Result<(), String> {
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for row in rgb.chunks(w as usize * 3) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let mut off = 0;
    while off < raw.len() {
        let n = (raw.len() - off).min(65535);
        let last = off + n == raw.len();
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(n as u16).to_le_bytes());
        z.extend_from_slice(&(!(n as u16)).to_le_bytes());
        z.extend_from_slice(&raw[off..off + n]);
        off += n;
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut chunk = |ty: &[u8; 4], data: &[u8]| {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = ty.to_vec();
        body.extend_from_slice(data);
        png.extend_from_slice(&body);
        png.extend_from_slice(&crc32(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    std::fs::write(path, png).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_and_adler_known_values() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
    }

    #[test]
    fn gap_sign_and_summary() {
        let t = vec![
            BeatTrace { b: 0, fill_issue_ns: 0, fill_done_ns: 100, fire_start_ns: 150, fire_done_ns: 300, bytes: 10, ..Default::default() },
            BeatTrace { b: 1, fill_issue_ns: 150, fill_done_ns: 400, fire_start_ns: 300, fire_done_ns: 450, bytes: 10, misfire: true, ..Default::default() },
        ];
        assert_eq!(t[0].gap_ns(), 50);
        assert_eq!(t[1].gap_ns(), -100);
        let s = summarize(&t);
        assert_eq!(s.late_beats, 1);
        assert_eq!(s.misfires, 1);
        assert_eq!(s.beats, 2);
        let d = text_diagram(&t, 40, 10);
        assert!(d.contains("MISFIRE"));
        let p = std::env::temp_dir().join("aien-engine-test.png");
        png_diagram(&t, &p).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
    }
}
