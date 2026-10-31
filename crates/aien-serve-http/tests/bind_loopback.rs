// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The engine binary listens on 127.0.0.1 only (DRA-47).
//!
//! `tests/loopback.rs` checks `serve` in process. This file runs the real binary the way the
//! AienOS installer starts it (no `--bind`), with the stub engine and an empty model directory,
//! so no model is loaded and no GPU device is opened. It then checks the listening socket in the
//! kernel's own table, not only the address the binary prints.
//!
//! `AIEN_SERVE_HTTP_BIN` points the test at another build of the binary, for example the engine a
//! release ships; unset, it runs the binary cargo built for this test.
//!
//! Linux only: the listener and descriptor checks read `/proc`.

#![cfg(target_os = "linux")]

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, UdpSocket};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::Duration;

const TOKEN: &str = "dra47-bind-test-token";

fn binary() -> PathBuf {
    std::env::var_os("AIEN_SERVE_HTTP_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_aien-serve-http")))
}

fn empty_model_dir() -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("dra47-bind-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A running server; killed when dropped.
struct Server {
    child: Child,
    addr: SocketAddr,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// How a start attempt ended: serving at an address, or exited with its stderr.
enum Start {
    Serving(Server),
    Exited { code: Option<i32>, stderr: String },
}

/// Starts the binary with the stub engine on an ephemeral port, plus `extra` flags.
/// The GPU is hidden from it as well (`CUDA_VISIBLE_DEVICES` empty,
/// `NVIDIA_VISIBLE_DEVICES=none`); the stub path never asks for it.
fn start(extra: &[&str]) -> Start {
    let model_dir = empty_model_dir();
    let mut child = Command::new(binary())
        .args(["--engine", "stub", "--port", "0", "--model-dir"])
        .arg(&model_dir)
        .args(extra)
        .env("AIEN_SERVE_HTTP_BEARER_TOKEN", TOKEN)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("NVIDIA_VISIBLE_DEVICES", "none")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn aien-serve-http");
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                // Keep draining so the child never blocks on a full pipe.
                continue;
            }
        }
    });
    let mut seen = String::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(line) => {
                seen.push_str(&line);
                seen.push('\n');
                if let Some(addr) = serving_addr(&line) {
                    return Start::Serving(Server { child, addr });
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let status = child.wait().unwrap();
                return Start::Exited {
                    code: status.code(),
                    stderr: seen,
                };
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("no 'serving' line and no exit within 60 s; stderr so far:\n{seen}");
            }
        }
    }
}

/// `aien-serve-http: serving <model> at http://<addr>/v1 (engine stub)` -> `<addr>`.
fn serving_addr(line: &str) -> Option<SocketAddr> {
    let rest = line.split(" serving ").nth(1)?;
    let url = rest.split(" at http://").nth(1)?;
    url.split("/v1").next()?.parse().ok()
}

/// Socket inodes held by `pid`.
fn socket_inodes(pid: u32) -> HashSet<u64> {
    let mut out = HashSet::new();
    for ent in std::fs::read_dir(format!("/proc/{pid}/fd")).unwrap() {
        let Ok(target) = std::fs::read_link(ent.unwrap().path()) else {
            continue;
        };
        let t = target.to_string_lossy();
        if let Some(n) = t.strip_prefix("socket:[").and_then(|s| s.strip_suffix(']')) {
            out.insert(n.parse().unwrap());
        }
    }
    out
}

/// One `/proc/net/tcp{,6}` address (`0100007F` or 32 hex digits) as the address it names.
/// The kernel prints each 32-bit word in host byte order, so the native-endian bytes of the
/// parsed word are the address bytes.
fn proc_ip(hex: &str) -> IpAddr {
    let word = |h: &str| u32::from_str_radix(h, 16).unwrap().to_ne_bytes();
    match hex.len() {
        8 => IpAddr::V4(Ipv4Addr::from(word(hex))),
        32 => {
            let mut b = [0u8; 16];
            for i in 0..4 {
                b[i * 4..i * 4 + 4].copy_from_slice(&word(&hex[i * 8..i * 8 + 8]));
            }
            IpAddr::V6(Ipv6Addr::from(b))
        }
        n => panic!("unexpected address width {n} in /proc/net/tcp: {hex}"),
    }
}

fn is_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    }
}

/// Every listening TCP socket the process holds, as `(local ip, port)`, read from the kernel's
/// table in the process's network namespace.
fn listeners(pid: u32) -> Vec<(IpAddr, u16)> {
    let mine = socket_inodes(pid);
    let mut out = Vec::new();
    for table in ["tcp", "tcp6"] {
        let Ok(text) = std::fs::read_to_string(format!("/proc/{pid}/net/{table}")) else {
            continue;
        };
        for row in text.lines().skip(1) {
            let f: Vec<&str> = row.split_whitespace().collect();
            // sl local rem st tx:rx tr:when retrnsmt uid timeout inode
            if f.len() < 10 || f[3] != "0A" {
                continue;
            }
            let inode: u64 = f[9].parse().unwrap();
            if !mine.contains(&inode) {
                continue;
            }
            let (ip, port) = f[1].split_once(':').unwrap();
            out.push((proc_ip(ip), u16::from_str_radix(port, 16).unwrap()));
        }
    }
    out
}

/// A non-loopback IPv4 address of this host, if it has one: the source address the kernel
/// would pick to reach TEST-NET-1. A UDP connect sends no packet.
fn host_non_loopback_ipv4() -> Option<Ipv4Addr> {
    let s = UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.0.2.1:9").ok()?;
    match s.local_addr().ok()?.ip() {
        IpAddr::V4(v4) if !v4.is_loopback() && !v4.is_unspecified() => Some(v4),
        _ => None,
    }
}

/// No NVIDIA device descriptor open and no CUDA driver library mapped.
fn assert_no_gpu(pid: u32) {
    for ent in std::fs::read_dir(format!("/proc/{pid}/fd")).unwrap() {
        if let Ok(target) = std::fs::read_link(ent.unwrap().path()) {
            let t = target.to_string_lossy();
            assert!(!t.starts_with("/dev/nvidia"), "stub server holds {t}");
        }
    }
    let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap();
    for line in maps.lines() {
        assert!(
            !line.contains("libcuda") && !line.contains("/dev/nvidia"),
            "stub server maps {line}"
        );
    }
}

#[test]
fn default_start_listens_on_ipv4_loopback_only() {
    let server = match start(&[]) {
        Start::Serving(s) => s,
        Start::Exited { code, stderr } => {
            panic!("server did not start (exit {code:?}):\n{stderr}")
        }
    };
    let pid = server.child.id();
    assert_eq!(
        server.addr.ip(),
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        "the binary reports serving at {}",
        server.addr
    );
    let found = listeners(pid);
    assert!(
        found.iter().any(|&(_, p)| p == server.addr.port()),
        "no listening socket for port {} in /proc/{pid}/net: {found:?}",
        server.addr.port()
    );
    for (ip, port) in &found {
        assert!(
            is_loopback(*ip),
            "listening on {ip}:{port}, which is not loopback"
        );
    }
    assert!(TcpStream::connect_timeout(&server.addr, Duration::from_secs(5)).is_ok());
    match host_non_loopback_ipv4() {
        Some(ip) => {
            let outside = SocketAddr::new(IpAddr::V4(ip), server.addr.port());
            assert!(
                TcpStream::connect_timeout(&outside, Duration::from_secs(2)).is_err(),
                "the server accepted a connection on the host address {outside}"
            );
        }
        None => eprintln!(
            "no non-loopback IPv4 route on this host; the outside connect check is skipped"
        ),
    }
    assert_no_gpu(pid);
}

#[test]
fn wildcard_and_outside_binds_are_refused_before_listening() {
    for addr in ["0.0.0.0", "::", "192.0.2.1"] {
        match start(&["--bind", addr]) {
            Start::Serving(s) => panic!("--bind {addr} was accepted: serving at {}", s.addr),
            Start::Exited { code, stderr } => {
                assert_ne!(code, Some(0), "--bind {addr} exited 0:\n{stderr}");
                assert!(
                    stderr.contains("loopback"),
                    "--bind {addr} failed without the loopback refusal:\n{stderr}"
                );
            }
        }
    }
}

#[test]
fn listener_table_reader_tells_loopback_from_wildcard() {
    // Fixed rows in the kernel's format (little-endian host words), so the reader itself has a
    // negative control: a wildcard listener must not read as loopback.
    if cfg!(target_endian = "little") {
        assert_eq!(proc_ip("0100007F"), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(proc_ip("00000000"), IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert_eq!(
            proc_ip("00000000000000000000000001000000"),
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        );
        assert_eq!(
            proc_ip("0000000000000000FFFF00000100007F"),
            IpAddr::V6(Ipv4Addr::LOCALHOST.to_ipv6_mapped())
        );
    }
    assert!(is_loopback(IpAddr::V4(Ipv4Addr::LOCALHOST)));
    assert!(is_loopback(IpAddr::V6(
        Ipv4Addr::LOCALHOST.to_ipv6_mapped()
    )));
    assert!(!is_loopback(IpAddr::V4(Ipv4Addr::UNSPECIFIED)));
    assert!(!is_loopback(IpAddr::V6(Ipv6Addr::UNSPECIFIED)));
    assert!(!is_loopback(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1))));
}
