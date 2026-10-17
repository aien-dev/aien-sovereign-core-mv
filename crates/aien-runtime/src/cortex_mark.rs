//! External Cortex record mark (NEXT-PHASE-2 ACCEPTANCE-v3 section 2.1-2.3).
//!
//! The Cortex journal header carries no record count, and the J-Space anchor
//! moves only at composition commits, so a journal cut exactly at a record
//! boundary that drops only host records opens as if intact (VERDICT-v2 C6c).
//! The mark is kept OUTSIDE the compose dir (`<dir>.cortex-mark`, beside the
//! machine root), written only after an append returned, and checked when the
//! home opens: a journal shorter than its mark is refused, never repaired.
//!
//! Receipt continuity (sovereign-core #408, recovery audit row 8). Each Cortex
//! record digest covers only its own record, so the newest digest alone says
//! nothing about the records before it: one rewritten with a fresh digest
//! (forged, reordered, removed with the count restored, spliced in from
//! another journal) opened as if intact. The version 2 mark also carries the
//! running hash over every record digest #1..#records, `sha256(chain ||
//! digest)` from 32 zero bytes (the chain omega's Cortex keeps in memory,
//! `rx_cortex.c` chain_step), recomputed at open: a mismatch is refused
//! (`E_MARK_CHAIN`).
//!
//! Layout v2 (160 bytes, little-endian): `"AIENCXM1"` | u32 version 2 | u32 0 |
//! machine id (32) | u64 seq | u64 records | digest of record #records (32,
//! zeros when records = 0) | chain over records #1..#records (32) | sha256 of
//! the first 128 bytes. Version 1 (128 bytes, no chain, sha256 of the first 96)
//! is still read: its count and digest are checked, and the next mark written
//! is version 2 (reported once; records before that start are taken as read).
//!
//! Limits (declared NOT_PROVED): the sha256 is unkeyed, so rewriting journal
//! and mark together, deleting the mark (adopted and reported), or rolling the
//! compose dir and the mark back together are not detected.
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};

/// Length of a version 2 mark (the one written).
pub const MARK_LEN: usize = 160;
/// Length of a version 1 mark (read only, from installs older than the chain).
pub const MARK_LEN_V1: usize = 128;
const MAGIC: &[u8; 8] = b"AIENCXM1";
const VERSION: u32 = 2;

/// One record mark: the journal held `records` records, the newest with
/// `digest`, every digest #1..#records hashing to `chain`, when mark number
/// `seq` was written. `chain` is `None` only for a version 1 mark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mark {
    pub machine_id: [u8; 32],
    pub seq: u64,
    pub records: u64,
    pub digest: [u8; 32],
    pub chain: Option<[u8; 32]>,
}

/// One step of the record chain: `sha256(chain || digest)` (omega
/// `rx_cortex.c` chain_step).
pub fn chain_step(chain: &mut [u8; 32], digest: &[u8; 32]) {
    let mut h = Sha256::new();
    h.update(*chain);
    h.update(digest);
    *chain = h.finalize().into();
}

/// The chain over `digests` in journal order, from 32 zero bytes.
pub fn chain_over<'a>(digests: impl IntoIterator<Item = &'a [u8; 32]>) -> [u8; 32] {
    let mut c = [0u8; 32];
    for d in digests {
        chain_step(&mut c, d);
    }
    c
}

impl Mark {
    /// Version 2 when the mark has a chain (always, for a mark this code
    /// writes); version 1 otherwise.
    pub fn encode(&self) -> Vec<u8> {
        let (version, body) = match self.chain {
            Some(_) => (VERSION, MARK_LEN - 32),
            None => (1u32, MARK_LEN_V1 - 32),
        };
        let mut b = vec![0u8; body + 32];
        b[..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&version.to_le_bytes());
        b[16..48].copy_from_slice(&self.machine_id);
        b[48..56].copy_from_slice(&self.seq.to_le_bytes());
        b[56..64].copy_from_slice(&self.records.to_le_bytes());
        b[64..96].copy_from_slice(&self.digest);
        if let Some(c) = &self.chain {
            b[96..128].copy_from_slice(c);
        }
        let h = Sha256::digest(&b[..body]);
        b[body..].copy_from_slice(&h);
        b
    }

    /// The mark in `b`, or why it is damaged.
    pub fn decode(b: &[u8]) -> Result<Mark, String> {
        if b.len() != MARK_LEN && b.len() != MARK_LEN_V1 {
            return Err(format!(
                "{} bytes, expected {MARK_LEN} ({MARK_LEN_V1} for version 1)",
                b.len()
            ));
        }
        if &b[..8] != MAGIC {
            return Err("bad magic".into());
        }
        let version = u32::from_le_bytes(b[8..12].try_into().expect("4 bytes"));
        let want = if b.len() == MARK_LEN { VERSION } else { 1 };
        if version != want || b[12..16] != [0u8; 4] {
            return Err(format!("unknown version {version} in {} bytes", b.len()));
        }
        let body = b.len() - 32;
        if Sha256::digest(&b[..body])[..] != b[body..] {
            return Err("checksum does not match".into());
        }
        Ok(Mark {
            machine_id: b[16..48].try_into().expect("32 bytes"),
            seq: u64::from_le_bytes(b[48..56].try_into().expect("8 bytes")),
            records: u64::from_le_bytes(b[56..64].try_into().expect("8 bytes")),
            digest: b[64..96].try_into().expect("32 bytes"),
            chain: (version == VERSION).then(|| b[96..128].try_into().expect("32 bytes")),
        })
    }
}

/// `<compose dir>.cortex-mark`.
pub fn mark_path(dir: &Path) -> PathBuf {
    let mut name = dir.as_os_str().to_owned();
    name.push(".cortex-mark");
    PathBuf::from(name)
}

/// The mark at `path`: `Ok(None)` when there is none, `Err(why)` when the
/// file exists but is not a valid mark.
pub fn read(path: &Path) -> Result<Option<Mark>, String> {
    match std::fs::read(path) {
        Ok(b) => Mark::decode(&b)
            .map(Some)
            .map_err(|why| format!("{}: {why}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// Atomic replace: write `<path>.tmp`, sync it, rename over `path`, sync the
/// parent directory. A failure leaves the previous mark in place.
pub fn write(path: &Path, m: &Mark) -> Result<(), String> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let res = (|| -> std::io::Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        f.write_all(&m.encode())?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)?;
        if let Some(p) = path.parent() {
            std::fs::File::open(p)?.sync_all()?;
        }
        Ok(())
    })();
    res.map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("write {}: {e}", path.display())
    })
}

/// Why a home is refused by its mark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkRefusal {
    Damaged(String),
    Truncated {
        file: u64,
        mark: u64,
        seq: u64,
    },
    Digest {
        record: u64,
        seq: u64,
    },
    /// Records #1..#records do not hash to the mark's chain: one before the
    /// newest was changed, removed, reordered or spliced in (#408 row 8).
    Chain {
        records: u64,
        seq: u64,
    },
}

impl fmt::Display for MarkRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkRefusal::Damaged(why) => write!(
                f,
                "record mark damaged (E_MARK): {why}. Run RecoverComposeHome (aien compose recover)"
            ),
            MarkRefusal::Truncated { file, mark, seq } => write!(
                f,
                "Cortex journal holds {file} records, its record mark says {mark}: {} record(s) lost at a record boundary (E_MARK_TRUNCATED, mark seq {seq}). Run RecoverComposeHome (aien compose recover)",
                mark - file
            ),
            MarkRefusal::Digest { record, seq } => write!(
                f,
                "Cortex record #{record} differs from its record mark (E_MARK_DIGEST, mark seq {seq}). Run RecoverComposeHome (aien compose recover)"
            ),
            MarkRefusal::Chain { records, seq } => write!(
                f,
                "Cortex records #1..#{records} do not hash to their record mark's chain: a record before the newest was changed, removed, reordered or spliced in (E_MARK_CHAIN, mark seq {seq}). Run RecoverComposeHome (aien compose recover)"
            ),
        }
    }
}

/// What the open does with the mark, decided before anything is appended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// No mark, no journal: a new home; the mark is created after the open.
    New,
    /// No mark, a journal exists (an install older than the mark, or a
    /// deleted mark): adopted and reported, never refused.
    Adopt,
    /// A mark exists and the journal is at least as long; its record #N
    /// digest and the chain over #1..#N are checked once the home is open.
    Verify(Mark),
}

/// The count check (ACCEPTANCE-v3 2.2), from the probe count `file_records`
/// that `Compose::open` reports before any append.
pub fn plan(
    mark: Option<&Mark>,
    machine_id: &[u8; 32],
    file_records: u64,
    journal_present: bool,
) -> Result<Plan, MarkRefusal> {
    let Some(m) = mark else {
        return Ok(if journal_present || file_records > 0 {
            Plan::Adopt
        } else {
            Plan::New
        });
    };
    if &m.machine_id != machine_id {
        return Err(MarkRefusal::Damaged(
            "the mark names another machine id".into(),
        ));
    }
    if file_records < m.records {
        return Err(MarkRefusal::Truncated {
            file: file_records,
            mark: m.records,
            seq: m.seq,
        });
    }
    Ok(Plan::Verify(m.clone()))
}

/// The digest check: record #`m.records` as the open home reads it
/// (`None` when it cannot be read).
pub fn check_digest(m: &Mark, at_mark: Option<[u8; 32]>) -> Result<(), MarkRefusal> {
    if m.records == 0 {
        return Ok(());
    }
    if at_mark == Some(m.digest) {
        Ok(())
    } else {
        Err(MarkRefusal::Digest {
            record: m.records,
            seq: m.seq,
        })
    }
}

/// The continuity check (#408 row 8): the chain over records #1..#`m.records`
/// as the open home reads them (`None` when one cannot be read). A version 1
/// mark has no chain to check.
pub fn check_chain(m: &Mark, at_mark: Option<[u8; 32]>) -> Result<(), MarkRefusal> {
    let Some(want) = m.chain else {
        return Ok(());
    };
    if m.records == 0 || at_mark == Some(want) {
        Ok(())
    } else {
        Err(MarkRefusal::Chain {
            records: m.records,
            seq: m.seq,
        })
    }
}

/// The one line the daemon prints for this open, if any (ACCEPTANCE-v3 2.2).
/// `probe` = records before the open appended anything; `written` = the mark
/// written at the end of the open, if one was.
pub fn open_event(plan: &Plan, probe: u64, written: Option<&Mark>) -> Option<String> {
    match (plan, written) {
        (Plan::New, Some(w)) => Some(format!(
            "Cortex mark: mark created for a new home (seq {}, records {})",
            w.seq, w.records
        )),
        (Plan::Adopt, Some(w)) => Some(format!(
            "Cortex mark: mark adopted (seq {}, records {}): this home had no record mark; a loss before this start cannot be ruled out",
            w.seq, w.records
        )),
        (Plan::Verify(m), _) if probe > m.records => Some(format!(
            "Cortex mark: record mark advanced {}->{} (records written before an interrupted mark update; not corruption)",
            m.records, probe
        )),
        (Plan::Verify(m), Some(w)) if m.chain.is_none() => Some(format!(
            "Cortex mark: version 1 mark (seq {}, records {}) upgraded to version 2 (seq {}, records {}): records before this start were taken as read, without a chain check",
            m.seq, m.records, w.seq, w.records
        )),
        _ => None,
    }
}

/// Where RecoverComposeHome keeps a mark it replaces: `<mark>.lost-<seq>`
/// (`.lost-damaged` when no seq is known), never over an existing file.
pub fn lost_path(mark: &Path, seq: Option<u64>) -> PathBuf {
    let tag = seq.map_or_else(|| "damaged".to_string(), |s| s.to_string());
    let mut base = mark.as_os_str().to_owned();
    base.push(format!(".lost-{tag}"));
    let base = PathBuf::from(base);
    if !base.exists() {
        return base;
    }
    (2u32..)
        .map(|i| {
            let mut p = base.as_os_str().to_owned();
            p.push(format!("-{i}"));
            PathBuf::from(p)
        })
        .find(|p| !p.exists())
        .expect("a free name")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mark(records: u64) -> Mark {
        Mark {
            machine_id: [7u8; 32],
            seq: 4,
            records,
            digest: [9u8; 32],
            chain: Some([3u8; 32]),
        }
    }

    #[test]
    fn round_trip() {
        let m = mark(23);
        let b = m.encode();
        assert_eq!(b.len(), MARK_LEN);
        assert_eq!(&b[..8], b"AIENCXM1");
        assert_eq!(Mark::decode(&b).unwrap(), m);
    }

    #[test]
    fn zero_records_with_a_chain_is_a_damaged_mark() {
        // No records means the initial chain (32 zero bytes); anything else is
        // the mark contradicting itself, not lost history.
        let m = Mark {
            records: 0,
            digest: [0u8; 32],
            ..mark(0)
        };
        let e = Mark::decode(&m.encode()).unwrap_err();
        assert!(e.contains("zero records"), "{e}");
        let ok = Mark {
            chain: Some([0u8; 32]),
            ..m
        };
        assert_eq!(Mark::decode(&ok.encode()).unwrap(), ok);
    }

    #[test]
    fn version_1_marks_still_read() {
        let m = Mark {
            chain: None,
            ..mark(23)
        };
        let b = m.encode();
        assert_eq!(b.len(), MARK_LEN_V1);
        assert_eq!(&b[8..12], &1u32.to_le_bytes());
        assert_eq!(Mark::decode(&b).unwrap(), m);
        // A version 2 header on 128 bytes (or version 1 on 160) is damage.
        let mut x = b.clone();
        x[8] = 2;
        assert!(Mark::decode(&x).unwrap_err().contains("version 2"));
        let mut y = mark(23).encode();
        y[8] = 1;
        assert!(Mark::decode(&y).unwrap_err().contains("version 1"));
    }

    #[test]
    fn chain_is_omega_running_hash() {
        let (d1, d2) = ([1u8; 32], [2u8; 32]);
        let mut want = [0u8; 32];
        for d in [d1, d2] {
            let mut buf = [0u8; 64];
            buf[..32].copy_from_slice(&want);
            buf[32..].copy_from_slice(&d);
            want = Sha256::digest(buf).into();
        }
        assert_eq!(chain_over([&d1, &d2]), want);
        // Order, gaps and repeats all change it.
        assert_ne!(chain_over([&d2, &d1]), want);
        assert_ne!(chain_over([&d2]), chain_over([&d1, &d2]));
        assert_ne!(chain_over([&d1, &d1, &d2]), want);
        assert_eq!(chain_over(std::iter::empty::<&[u8; 32]>()), [0u8; 32]);
    }

    #[test]
    fn chain_check() {
        let m = mark(23);
        assert_eq!(check_chain(&m, Some([3u8; 32])), Ok(()));
        assert_eq!(
            check_chain(&m, Some([4u8; 32])),
            Err(MarkRefusal::Chain {
                records: 23,
                seq: 4
            })
        );
        assert!(check_chain(&m, None).is_err());
        let v1 = Mark {
            chain: None,
            ..mark(23)
        };
        assert_eq!(check_chain(&v1, None), Ok(()));
        let t = MarkRefusal::Chain {
            records: 23,
            seq: 4,
        }
        .to_string();
        assert!(
            t.contains("#1..#23") && t.contains("E_MARK_CHAIN, mark seq 4"),
            "{t}"
        );
    }

    #[test]
    fn damaged_marks_are_named() {
        let b = mark(23).encode();
        assert!(Mark::decode(&b[..127]).unwrap_err().contains("127 bytes"));
        let mut x = b.clone();
        x[0] ^= 1;
        assert_eq!(Mark::decode(&x).unwrap_err(), "bad magic");
        let mut x = b.clone();
        x[8] = 3;
        assert!(Mark::decode(&x).unwrap_err().contains("version"));
        for i in [16usize, 50, 60, 70, 100, 127, 140, 159] {
            let mut x = b.clone();
            x[i] ^= 0x40;
            assert!(Mark::decode(&x).is_err(), "flip at {i} accepted");
        }
    }

    #[test]
    fn plan_every_state() {
        let id = [7u8; 32];
        assert_eq!(plan(None, &id, 0, false), Ok(Plan::New));
        assert_eq!(plan(None, &id, 0, true), Ok(Plan::Adopt));
        assert_eq!(plan(None, &id, 17, true), Ok(Plan::Adopt));
        let m = mark(23);
        assert_eq!(
            plan(Some(&m), &id, 22, true),
            Err(MarkRefusal::Truncated {
                file: 22,
                mark: 23,
                seq: 4
            })
        );
        assert_eq!(plan(Some(&m), &id, 23, true), Ok(Plan::Verify(m.clone())));
        assert_eq!(plan(Some(&m), &id, 24, true), Ok(Plan::Verify(m.clone())));
        assert!(matches!(
            plan(Some(&m), &[1u8; 32], 23, true),
            Err(MarkRefusal::Damaged(_))
        ));
    }

    #[test]
    fn digest_check() {
        let m = mark(23);
        assert_eq!(check_digest(&m, Some([9u8; 32])), Ok(()));
        assert_eq!(
            check_digest(&m, Some([8u8; 32])),
            Err(MarkRefusal::Digest { record: 23, seq: 4 })
        );
        assert!(check_digest(&m, None).is_err());
        assert_eq!(check_digest(&mark(0), None), Ok(()));
    }

    #[test]
    fn refusal_text_names_the_code() {
        let t = MarkRefusal::Truncated {
            file: 22,
            mark: 23,
            seq: 4,
        }
        .to_string();
        assert!(t.contains("holds 22 records") && t.contains("says 23"));
        assert!(t.contains("1 record(s) lost") && t.contains("E_MARK_TRUNCATED, mark seq 4"));
        assert!(MarkRefusal::Digest { record: 5, seq: 1 }
            .to_string()
            .contains("E_MARK_DIGEST"));
        assert!(MarkRefusal::Damaged("x".into())
            .to_string()
            .contains("(E_MARK)"));
    }

    #[test]
    fn events() {
        let w = mark(28);
        assert!(open_event(&Plan::Adopt, 17, Some(&w))
            .unwrap()
            .contains("mark adopted (seq 4, records 28)"));
        assert!(open_event(&Plan::New, 0, Some(&w))
            .unwrap()
            .contains("mark created for a new home"));
        let m = mark(23);
        assert!(open_event(&Plan::Verify(m.clone()), 24, Some(&w))
            .unwrap()
            .contains("record mark advanced 23->24"));
        assert_eq!(open_event(&Plan::Verify(m), 23, Some(&w)), None);
        let v1 = Mark {
            chain: None,
            ..mark(23)
        };
        assert!(open_event(&Plan::Verify(v1), 23, Some(&w))
            .unwrap()
            .contains("upgraded to version 2"));
    }

    #[test]
    fn write_is_atomic_and_leaves_no_tmp() {
        let d = tempfile::tempdir().unwrap();
        let p = mark_path(&d.path().join("compose"));
        assert_eq!(read(&p), Ok(None));
        write(&p, &mark(5)).unwrap();
        assert_eq!(read(&p).unwrap(), Some(mark(5)));
        write(&p, &mark(6)).unwrap();
        assert_eq!(read(&p).unwrap().unwrap().records, 6);
        let names: Vec<String> = std::fs::read_dir(d.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["compose.cortex-mark".to_string()]);
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        std::fs::write(&p, b"short").unwrap();
        assert!(read(&p).unwrap_err().contains("5 bytes"));
    }

    #[test]
    fn failed_write_keeps_the_old_mark() {
        let d = tempfile::tempdir().unwrap();
        let p = mark_path(&d.path().join("compose"));
        write(&p, &mark(5)).unwrap();
        // A directory in place of the tmp file makes the write fail.
        let mut tmp = p.as_os_str().to_owned();
        tmp.push(".tmp");
        std::fs::create_dir(PathBuf::from(&tmp)).unwrap();
        assert!(write(&p, &mark(6)).is_err());
        assert_eq!(read(&p).unwrap(), Some(mark(5)));
    }

    #[test]
    fn lost_path_never_overwrites() {
        let d = tempfile::tempdir().unwrap();
        let p = mark_path(&d.path().join("compose"));
        let a = lost_path(&p, Some(3));
        assert!(a.to_string_lossy().ends_with("compose.cortex-mark.lost-3"));
        std::fs::write(&a, b"x").unwrap();
        let b = lost_path(&p, Some(3));
        assert!(b.to_string_lossy().ends_with(".lost-3-2"));
        assert!(lost_path(&p, None)
            .to_string_lossy()
            .ends_with(".lost-damaged"));
    }
}
