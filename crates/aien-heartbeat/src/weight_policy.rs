//! Bounded framed bridge to the seed-built Omega scheduling policy.
//! This is host orchestration, not an inference engine or a policy fallback.
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct NativePolicy {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    failed: bool,
}
impl NativePolicy {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("weight policy start: {e}"))?;
        let input = child.stdin.take().ok_or("weight policy stdin")?;
        let output = child.stdout.take().ok_or("weight policy stdout")?;
        let mut result = Self {
            child,
            input,
            output,
            failed: false,
        };
        for fd in [result.input.as_raw_fd(), result.output.as_raw_fd()] {
            // Both descriptors belong solely to this bridge; nonblocking makes the deadline real.
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
            {
                result.failed = true;
                return Err("weight policy nonblocking pipe".into());
            }
        }
        // A real protocol probe, not merely a successful spawn.
        if result.decide([1, 0, 1, 1, 0, 0, 0, 0, 0, 0])? != (1, 0) {
            return Err("weight policy probe".into());
        }
        Ok(result)
    }
    pub fn decide(&mut self, fields: [u64; 10]) -> Result<(usize, u64), String> {
        if self.failed {
            return Err("weight policy is failed".into());
        }
        let result = self.exchange(fields);
        if result.is_err() {
            self.failed = true;
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        result
    }
    fn exchange(&mut self, fields: [u64; 10]) -> Result<(usize, u64), String> {
        let mut request = [0u8; 80];
        for (i, x) in fields.iter().enumerate() {
            request[i * 8..i * 8 + 8].copy_from_slice(&x.to_le_bytes());
        }
        let end = Instant::now() + Duration::from_secs(2);
        let mut sent = 0;
        while sent < request.len() {
            ready(self.input.as_raw_fd(), libc::POLLOUT, end)?;
            match self.input.write(&request[sent..]) {
                Ok(0) => return Err("weight policy closed input".into()),
                Ok(n) => sent += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(format!("weight policy write: {e}")),
            }
        }
        let mut reply = [0u8; 16];
        let mut got = 0;
        while got < reply.len() {
            ready(self.output.as_raw_fd(), libc::POLLIN, end)?;
            match self.output.read(&mut reply[got..]) {
                Ok(0) => return Err("weight policy closed output".into()),
                Ok(n) => got += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(format!("weight policy read: {e}")),
            }
        }
        let depth = u64::from_le_bytes(reply[..8].try_into().unwrap());
        let calm = u64::from_le_bytes(reply[8..].try_into().unwrap());
        if depth == 0 || depth > fields[2] || calm > 3 {
            return Err("weight policy response out of bounds".into());
        }
        Ok((depth as usize, calm))
    }
}
fn ready(fd: RawFd, events: i16, end: Instant) -> Result<(), String> {
    loop {
        let left = end.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err("weight policy deadline".into());
        }
        let mut p = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let n = unsafe { libc::poll(&mut p, 1, left.as_millis().max(1).min(2000) as i32) };
        if n < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err("weight policy poll".into());
        }
        if n > 0 {
            return Ok(());
        }
    }
}
impl Drop for NativePolicy {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
