//! Who holds the chip for this server: itself, or a trusted ancestor.
//!
//! The hive serializes the GB10 under `/tmp/aien-gb10.lock` ([`GPU_LOCK_PATH`]). A normal
//! server takes that lock itself for the upload and around each generation. An aien-test gb10
//! gate already holds it for the gate's whole life (ADR 0033: the trusted side serializes gb10),
//! so a server the gate starts would queue behind its own caller until its load gave up. The
//! runner says so with `AIEN_GB10_LOCK_HELD=1` ([`LOCK_HELD_ENV`], aien-test `resources.rs`).
//!
//! That marker is an ambient environment variable, so it is not trusted by itself. With it set
//! the server VERIFIES the claim before any chip work: `/proc/locks` must show a `FLOCK` on the
//! canonical lock file's inode whose holder is one of this process's ancestors. A process that
//! cannot show that is refused at startup (exit 1), so a spoofed marker, a marker with nobody
//! holding the lock, or a lock held by a stranger never reaches the chip; the gate then fails,
//! it does not pass. There is no path override: the canonical lock is the only lock.

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::omega_engine::GPU_LOCK_PATH;

/// Set by the aien-test runner (and only the runner) for a gb10 gate it holds the GPU lock for.
pub const LOCK_HELD_ENV: &str = "AIEN_GB10_LOCK_HELD";

static INHERITED: AtomicBool = AtomicBool::new(false);

/// How this process is allowed on the chip.
#[derive(Debug, PartialEq, Eq)]
pub enum ChipReservation {
    /// Take `/tmp/aien-gb10.lock` ourselves (the live server, a `q36-extra` test server).
    Own,
    /// An ancestor holds `/tmp/aien-gb10.lock` for our whole life (verified): take nothing.
    Inherited { holder_pid: u32 },
}

/// Decides the reservation from the environment and `/proc`, and records it for
/// [`inherited`]. Call once at startup, before any lock is taken. `Err` = the marker was set
/// but the claim did not verify; the caller must exit without touching the chip.
pub fn install() -> Result<ChipReservation, String> {
    let r = detect(
        std::env::var(LOCK_HELD_ENV).ok().as_deref(),
        Path::new(GPU_LOCK_PATH),
        &fs::read_to_string("/proc/locks").unwrap_or_default(),
        &ancestors_of_self(),
    )?;
    INHERITED.store(
        matches!(r, ChipReservation::Inherited { .. }),
        Ordering::SeqCst,
    );
    Ok(r)
}

/// `true` after [`install`] verified an ancestor's reservation: the doors take no lock.
pub fn inherited() -> bool {
    INHERITED.load(Ordering::SeqCst)
}

/// [`install`] on explicit inputs (tests). `marker` is the value of [`LOCK_HELD_ENV`].
pub fn detect(
    marker: Option<&str>,
    lock_path: &Path,
    proc_locks: &str,
    ancestors: &[u32],
) -> Result<ChipReservation, String> {
    match marker {
        None | Some("") | Some("0") => return Ok(ChipReservation::Own),
        Some("1") => {}
        Some(other) => {
            return Err(format!(
                "{LOCK_HELD_ENV}={other:?}: only 1 (set by the aien-test runner) or unset"
            ))
        }
    }
    let meta = fs::metadata(lock_path).map_err(|e| {
        format!(
            "{LOCK_HELD_ENV}=1 but {} cannot be examined ({e}): refusing the chip",
            lock_path.display()
        )
    })?;
    let holders = flock_holders(proc_locks, meta.dev(), meta.ino());
    if holders.is_empty() {
        return Err(format!(
            "{LOCK_HELD_ENV}=1 but nobody holds {}: refusing the chip (a spoofed marker, or the caller let go)",
            lock_path.display()
        ));
    }
    match holders.iter().find(|p| ancestors.contains(p)) {
        Some(&holder_pid) => Ok(ChipReservation::Inherited { holder_pid }),
        None => Err(format!(
            "{LOCK_HELD_ENV}=1 but {} is held by pid(s) {:?}, none an ancestor of this server (ancestors {:?}): refusing the chip",
            lock_path.display(),
            holders,
            ancestors
        )),
    }
}

/// Pids holding an exclusive `FLOCK` (`WRITE`) on the file `(dev, ino)` according to a
/// `/proc/locks` listing; shared holders do not count.
/// `/proc/locks` lines look like `1: FLOCK  ADVISORY  WRITE 12345 00:2e:1234 0 EOF`; its
/// `maj:min` are the device's major and minor in hex, the inode in decimal. Waiters are listed
/// with `->` after the number and are skipped.
pub fn flock_holders(proc_locks: &str, dev: u64, ino: u64) -> Vec<u32> {
    let (want_maj, want_min) = (libc_major(dev), libc_minor(dev));
    proc_locks
        .lines()
        .filter_map(|line| {
            let mut f = line.split_whitespace();
            f.next()?; // "N:"; a waiter's line has "->" here instead of FLOCK and is skipped below
            if f.next()? != "FLOCK" {
                return None;
            }
            f.next()?; // ADVISORY / MANDATORY
            if f.next()? != "WRITE" {
                return None; // a shared (LOCK_SH) holder is not an exclusive reservation
            }
            let pid: u32 = f.next()?.parse().ok()?;
            let mut loc = f.next()?.split(':');
            let maj = u64::from_str_radix(loc.next()?, 16).ok()?;
            let min = u64::from_str_radix(loc.next()?, 16).ok()?;
            let inode: u64 = loc.next()?.parse().ok()?;
            (maj == want_maj && min == want_min && inode == ino).then_some(pid)
        })
        .collect()
}

/// Linux `major()` for `st_dev` (glibc encoding).
fn libc_major(dev: u64) -> u64 {
    ((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0xfff)
}

/// Linux `minor()` for `st_dev` (glibc encoding).
fn libc_minor(dev: u64) -> u64 {
    ((dev >> 12) & 0xffff_ff00) | (dev & 0xff)
}

/// This process's ancestors (parent first), from `/proc/<pid>/status` `PPid:` lines. Stops at
/// pid 0 or 1 or at anything unreadable (a different pid namespace fails closed: no ancestors).
pub fn ancestors_of_self() -> Vec<u32> {
    let mut out = Vec::new();
    let mut pid = std::process::id();
    for _ in 0..256 {
        let status = match fs::read_to_string(format!("/proc/{pid}/status")) {
            Ok(s) => s,
            Err(_) => break,
        };
        let ppid: u32 = match status
            .lines()
            .find_map(|l| l.strip_prefix("PPid:"))
            .and_then(|v| v.trim().parse().ok())
        {
            Some(p) => p,
            None => break,
        };
        if ppid <= 1 {
            break;
        }
        out.push(ppid);
        pid = ppid;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    const LOCKS: &str = "\
1: FLOCK  ADVISORY  WRITE 4242 00:2e:777 0 EOF
1: -> FLOCK  ADVISORY  WRITE 9999 00:2e:777 0 EOF
2: POSIX  ADVISORY  WRITE 5555 00:2e:777 0 EOF
3: FLOCK  ADVISORY  WRITE 1234 00:2e:778 0 EOF
4: FLOCK  ADVISORY  READ  3333 00:2e:779 0 EOF
";

    fn dev(maj: u64, min: u64) -> u64 {
        ((maj & 0xffff_f000) << 32)
            | ((maj & 0xfff) << 8)
            | ((min & 0xffff_ff00) << 12)
            | (min & 0xff)
    }

    #[test]
    fn holders_are_flock_owners_on_that_inode_only() {
        // The waiter (`->`), the POSIX lock and the other inode are not holders.
        assert_eq!(flock_holders(LOCKS, dev(0, 0x2e), 777), vec![4242]);
        assert_eq!(flock_holders(LOCKS, dev(0, 0x2e), 778), vec![1234]);
        assert!(flock_holders(LOCKS, dev(0, 0x2f), 777).is_empty());
        assert!(flock_holders(LOCKS, dev(0, 0x2e), 1).is_empty());
        // A shared holder is not a reservation.
        assert!(flock_holders(LOCKS, dev(0, 0x2e), 779).is_empty());
    }

    #[test]
    fn without_the_marker_the_server_takes_its_own_lock() {
        let p = Path::new("/nonexistent/aien-gb10.lock");
        assert_eq!(detect(None, p, LOCKS, &[4242]), Ok(ChipReservation::Own));
        assert_eq!(
            detect(Some(""), p, LOCKS, &[4242]),
            Ok(ChipReservation::Own)
        );
        assert_eq!(
            detect(Some("0"), p, LOCKS, &[4242]),
            Ok(ChipReservation::Own)
        );
        assert!(detect(Some("yes"), p, LOCKS, &[4242]).is_err());
    }

    /// The negative tests the review asked for: a marker alone never enters the chip.
    #[test]
    fn marker_is_refused_unless_an_ancestor_holds_the_canonical_lock() {
        let dir =
            std::env::temp_dir().join(format!("aien-chip-reservation-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let lock = dir.join("aien-gb10.lock");
        let f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock)
            .unwrap();
        let meta = fs::metadata(&lock).unwrap();
        let me = std::process::id();
        let line = |pid: u32| {
            format!(
                "1: FLOCK  ADVISORY  WRITE {pid} {:02x}:{:02x}:{} 0 EOF\n",
                libc_major(meta.dev()),
                libc_minor(meta.dev()),
                meta.ino()
            )
        };
        // Nobody holds it: refused.
        assert!(detect(Some("1"), &lock, "", &[1, 2]).is_err());
        // A stranger holds it: refused, even with the marker.
        assert!(detect(Some("1"), &lock, &line(424242), &[me + 1]).is_err());
        // The lock file does not exist at all: refused.
        assert!(detect(Some("1"), &dir.join("missing"), &line(me), &[me]).is_err());
        // An ancestor holds it: allowed, and the holder is named.
        assert_eq!(
            detect(Some("1"), &lock, &line(me), &[me]),
            Ok(ChipReservation::Inherited { holder_pid: me })
        );
        // The real /proc/locks sees a real flock we take, by (dev, ino): the parser matches the
        // kernel's formatting, not only our fixture's.
        // SAFETY: flock on a valid descriptor of a private temp file.
        assert_eq!(unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) }, 0);
        let real = fs::read_to_string("/proc/locks").unwrap();
        assert!(
            flock_holders(&real, meta.dev(), meta.ino()).contains(&me),
            "our own flock is listed in /proc/locks under our pid"
        );
        drop(f);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ancestors_start_at_the_parent_and_exclude_self() {
        let a = ancestors_of_self();
        assert!(!a.contains(&std::process::id()));
        // A cargo test binary always has a parent (cargo or a shell).
        assert!(!a.is_empty());
    }
}
