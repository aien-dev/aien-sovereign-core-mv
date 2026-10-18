//! What the worker can say about the machine it runs on, and the one limit
//! the operating system enforces for it.

use crate::receipt::Platform;
use sha2::{Digest, Sha256};
use std::io;

/// Platform facts for the receipt and the capability report. All of them are
/// the worker's own claims.
pub fn platform() -> io::Result<Platform> {
    Ok(Platform {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get() as u64)
            .unwrap_or(1),
        memory_bytes: memory_bytes(),
        program_digest: program_digest()?,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// SHA-256 of the running executable's file.
pub fn program_digest() -> io::Result<[u8; 32]> {
    let exe = std::env::current_exe()?;
    Ok(Sha256::digest(std::fs::read(exe)?).into())
}

#[cfg(target_os = "macos")]
fn memory_bytes() -> u64 {
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: sysctlbyname writes at most `len` bytes into `v`, a u64 we own.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut v as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        v
    } else {
        0
    }
}

#[cfg(not(target_os = "macos"))]
fn memory_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// Cap the worker process's CPU seconds (RLIMIT_CPU) and forbid core dumps.
/// This is the only ceiling the kernel enforces here; on macOS there is no
/// reliable address-space limit and no seccomp or namespaces.
pub fn limit_cpu_seconds(secs: u64) -> io::Result<()> {
    let set = |res, v: u64| {
        let lim = libc::rlimit {
            rlim_cur: v as libc::rlim_t,
            rlim_max: v as libc::rlim_t,
        };
        // SAFETY: setrlimit reads a struct we own.
        if unsafe { libc::setrlimit(res, &lim) } != 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    };
    set(libc::RLIMIT_CPU, secs)?;
    set(libc::RLIMIT_CORE, 0)
}
