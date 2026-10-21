//! The wire between two stages (L258, Drake 2026-10-21: "make two pipelines"). Plain TCP on the LAN, one framed message
//! per hop: a 32-byte header and the payload. Std only, so `linkprobe` compiles from this file alone with `rustc` on a
//! machine that has none of the engine's dependencies (the Mac). The signed transport (interplane join, UDP 47040)
//! replaces the bare socket later; the frame stays the same.
//!
//! Header (little endian): magic u32 | kind u16 | flags u16 | conv u32 | step u32 | layer u32 (the layer the payload
//! enters) | rows u32 | len u64 (payload bytes).
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

pub const MAGIC: u32 = 0x4c32_3538; // "L258"
pub const HEADER: usize = 32;

/// Probe: echo the payload back.
pub const KIND_PING: u16 = 1;
/// Probe: payload is bulk; the far end answers with an empty ack after reading it.
pub const KIND_BULK: u16 = 2;
pub const KIND_ACK: u16 = 3;
/// Stage pipe: residual rows `[rows][hidden]` f32 entering `layer`.
pub const KIND_RESIDUAL: u16 = 10;
/// Stage pipe: the sampled token id of (conv, step), sent from the last stage back to the first (payload u32).
pub const KIND_TOKEN: u16 = 11;
/// Stage pipe: hello carrying the sender's stage range (payload: first u32, last u32 inclusive, of u32).
pub const KIND_HELLO: u16 = 12;
/// Stage pipe: no more work.
pub const KIND_DONE: u16 = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Header {
    pub kind: u16,
    pub flags: u16,
    pub conv: u32,
    pub step: u32,
    pub layer: u32,
    pub rows: u32,
    pub len: u64,
}

impl Header {
    pub fn encode(&self) -> [u8; HEADER] {
        let mut b = [0u8; HEADER];
        b[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        b[4..6].copy_from_slice(&self.kind.to_le_bytes());
        b[6..8].copy_from_slice(&self.flags.to_le_bytes());
        b[8..12].copy_from_slice(&self.conv.to_le_bytes());
        b[12..16].copy_from_slice(&self.step.to_le_bytes());
        b[16..20].copy_from_slice(&self.layer.to_le_bytes());
        b[20..24].copy_from_slice(&self.rows.to_le_bytes());
        b[24..32].copy_from_slice(&self.len.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8; HEADER]) -> Result<Self, String> {
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err(format!("link: bad magic {:#x}", u32_at(0)));
        }
        Ok(Self {
            kind: u16::from_le_bytes([b[4], b[5]]),
            flags: u16::from_le_bytes([b[6], b[7]]),
            conv: u32_at(8),
            step: u32_at(12),
            layer: u32_at(16),
            rows: u32_at(20),
            len: u64::from_le_bytes(b[24..32].try_into().unwrap()),
        })
    }
}

/// One TCP connection with Nagle off (a 4 KB hop must not wait for more bytes).
pub struct Link {
    s: TcpStream,
}

impl Link {
    pub fn new(s: TcpStream) -> Result<Self, String> {
        s.set_nodelay(true).map_err(|e| format!("link: nodelay: {e}"))?;
        Ok(Self { s })
    }

    pub fn connect(addr: &str) -> Result<Self, String> {
        Self::new(TcpStream::connect(addr).map_err(|e| format!("link: connect {addr}: {e}"))?)
    }

    pub fn try_clone(&self) -> Result<Self, String> {
        Ok(Self { s: self.s.try_clone().map_err(|e| format!("link: clone: {e}"))? })
    }

    /// Header and payload in one write (one segment for small messages).
    pub fn send(&mut self, h: &Header, payload: &[u8]) -> Result<(), String> {
        let mut h = *h;
        h.len = payload.len() as u64;
        let mut buf = Vec::with_capacity(HEADER + payload.len());
        buf.extend_from_slice(&h.encode());
        buf.extend_from_slice(payload);
        self.s.write_all(&buf).map_err(|e| format!("link: send: {e}"))
    }

    pub fn recv(&mut self, payload: &mut Vec<u8>) -> Result<Header, String> {
        let mut hb = [0u8; HEADER];
        self.s.read_exact(&mut hb).map_err(|e| format!("link: recv header: {e}"))?;
        let h = Header::decode(&hb)?;
        payload.resize(h.len as usize, 0);
        self.s.read_exact(payload).map_err(|e| format!("link: recv payload ({} bytes): {e}", h.len))?;
        Ok(h)
    }

    /// Bulk send without building one big buffer: header, then the payload streamed in `chunk`-byte writes.
    pub fn send_bulk(&mut self, total: u64, chunk: &[u8]) -> Result<(), String> {
        let h = Header { kind: KIND_BULK, len: total, ..Default::default() };
        self.s.write_all(&h.encode()).map_err(|e| format!("link: bulk header: {e}"))?;
        let mut left = total;
        while left > 0 {
            let n = left.min(chunk.len() as u64) as usize;
            self.s.write_all(&chunk[..n]).map_err(|e| format!("link: bulk: {e}"))?;
            left -= n as u64;
        }
        Ok(())
    }

    /// Reads a bulk payload into a reused buffer (never holds it all).
    pub fn drain(&mut self, len: u64, buf: &mut [u8]) -> Result<(), String> {
        let mut left = len;
        while left > 0 {
            let n = left.min(buf.len() as u64) as usize;
            self.s.read_exact(&mut buf[..n]).map_err(|e| format!("link: drain: {e}"))?;
            left -= n as u64;
        }
        Ok(())
    }
}

/// f32 rows <-> bytes, exact (the residual crosses the wire bit for bit).
pub fn f32_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn bytes_f32(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() % 4 != 0 {
        return Err(format!("link: {} bytes is not whole f32s", b.len()));
    }
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Probe server: echoes pings, acks bulk. One thread per connection; runs until killed.
pub fn serve(listener: std::net::TcpListener) -> Result<(), String> {
    for s in listener.incoming() {
        let s = s.map_err(|e| format!("link: accept: {e}"))?;
        std::thread::spawn(move || {
            let Ok(mut l) = Link::new(s) else { return };
            let mut p = Vec::new();
            let mut sink = vec![0u8; 1 << 20];
            loop {
                let mut hb = [0u8; HEADER];
                if l.s.read_exact(&mut hb).is_err() {
                    return;
                }
                let Ok(h) = Header::decode(&hb) else { return };
                match h.kind {
                    KIND_BULK => {
                        if l.drain(h.len, &mut sink).is_err() {
                            return;
                        }
                        if l.send(&Header { kind: KIND_ACK, ..Default::default() }, &[]).is_err() {
                            return;
                        }
                    }
                    _ => {
                        p.resize(h.len as usize, 0);
                        if l.s.read_exact(&mut p).is_err() || l.send(&h, &p).is_err() {
                            return;
                        }
                    }
                }
            }
        });
    }
    Ok(())
}

/// Round trips of `size`-byte pings: nanoseconds each, after `warm` unrecorded ones.
pub fn ping(l: &mut Link, size: usize, warm: usize, count: usize) -> Result<Vec<u64>, String> {
    let payload: Vec<u8> = (0..size).map(|i| (i * 31 + 7) as u8).collect();
    let mut back = Vec::with_capacity(size);
    let mut out = Vec::with_capacity(count);
    for i in 0..warm + count {
        let t = Instant::now();
        l.send(&Header { kind: KIND_PING, step: i as u32, ..Default::default() }, &payload)?;
        let h = l.recv(&mut back)?;
        let ns = t.elapsed().as_nanos() as u64;
        if h.step != i as u32 || back != payload {
            return Err(format!("link: ping {i}: echo differs"));
        }
        if i >= warm {
            out.push(ns);
        }
    }
    Ok(out)
}

/// One bulk transfer of `total` bytes in `chunk`-byte writes, timed until the far end's ack (bytes on the far side).
pub fn bulk(l: &mut Link, total: u64, chunk: usize) -> Result<u64, String> {
    let buf: Vec<u8> = (0..chunk).map(|i| (i * 13 + 1) as u8).collect();
    let mut ack = Vec::new();
    let t = Instant::now();
    l.send_bulk(total, &buf)?;
    let h = l.recv(&mut ack)?;
    if h.kind != KIND_ACK {
        return Err(format!("link: bulk: expected ack, got kind {}", h.kind));
    }
    Ok(t.elapsed().as_nanos() as u64)
}

/// Sorted-sample summary: min, median, p90, p99, max, mean (in the input's unit).
pub fn stats(v: &[u64]) -> [f64; 6] {
    if v.is_empty() {
        return [0.0; 6];
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize] as f64;
    let mean = s.iter().map(|&x| x as f64).sum::<f64>() / s.len() as f64;
    [s[0] as f64, q(0.5), q(0.9), q(0.99), s[s.len() - 1] as f64, mean]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips_and_bad_magic_fails() {
        let h = Header { kind: KIND_RESIDUAL, flags: 3, conv: 2, step: 9, layer: 20, rows: 5, len: 40960 };
        assert_eq!(Header::decode(&h.encode()).unwrap(), h);
        let mut b = h.encode();
        b[0] ^= 1;
        assert!(Header::decode(&b).is_err());
    }

    #[test]
    fn f32_rows_cross_bit_for_bit() {
        let x = vec![1.5f32, -0.0, f32::MIN_POSITIVE, 3.0e38, -7.25e-12];
        let y = bytes_f32(&f32_bytes(&x)).unwrap();
        assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(bytes_f32(&[0u8; 5]).is_err());
    }

    #[test]
    fn loopback_ping_and_bulk() {
        let lis = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = lis.local_addr().unwrap().to_string();
        std::thread::spawn(move || serve(lis));
        let mut l = Link::connect(&addr).unwrap();
        let rt = ping(&mut l, 4096, 2, 10).unwrap();
        assert_eq!(rt.len(), 10);
        assert!(bulk(&mut l, 3 << 20, 1 << 16).unwrap() > 0);
    }
}
