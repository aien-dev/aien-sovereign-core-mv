//! The printed storage map (sc #678, FORTRAN 1956 idea 6): every resident byte of a run has a
//! name, an owner, the stages that write it and a lifetime, and the sum is held against what the
//! process really holds. What the map does not name is printed as the unaccounted number.
//!
//! Two spaces, both out of the same 128 GiB on the GB10:
//!
//! - `driver`: omega's driver allocations. Every one has a CPU view mapped from a `/dev/nvidia*`
//!   file, so `/proc/self/smaps` lists it; the measured figure is the summed `Size` of those
//!   mappings (pinned at allocation, touched or not), with their `Rss` printed beside it.
//!   omega's own counters (`omega_gpu_session_alloc_stats`) are printed too, but they are not the
//!   truth: `omega_gpu_pf_alloc_dev` allocates without counting (omega `src/omega_gpu_prefill.c`).
//! - `heap`: the process's anonymous memory (`Anonymous` of every other mapping): host vectors,
//!   the allocator's retained free space, thread stacks.
//!
//! A [`StorageMap`] is built from the model's config (the plan, which admission checks use) and
//! filled with per-entry measurements while the model loads ([`Probe`] around each allocating
//! call). [`Receipt`] holds the map against a [`ProcMemory`] snapshot.
use std::fmt::Write as _;

use serde_json::{json, Value};

/// Where an entry lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    /// omega driver allocation (GPU-visible, CPU-mapped from `/dev/nvidia*`).
    Driver,
    /// Process heap (anonymous memory).
    Heap,
}

impl Space {
    pub fn as_str(self) -> &'static str {
        match self {
            Space::Driver => "driver",
            Space::Heap => "heap",
        }
    }
}

/// How long an entry's contents stay meaningful (idea 1's lifetime classes plus the engine's own).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifetime {
    /// Written once at load, read for the life of the process.
    Weights,
    /// Per-conversation state: KV cache, DeltaNet tails and states (reset per sequence).
    Conversation,
    /// Per-token or per-chunk scratch: overwritten by every step or chunk.
    Token,
    /// Derived tables kept to avoid recomputing (RoPE rows, gather tables).
    Cache,
    /// The engine's own: kernel code, command lists, launch banks.
    Engine,
}

impl Lifetime {
    pub fn as_str(self) -> &'static str {
        match self {
            Lifetime::Weights => "weights",
            Lifetime::Conversation => "conversation",
            Lifetime::Token => "token",
            Lifetime::Cache => "cache",
            Lifetime::Engine => "engine",
        }
    }
    const ALL: [Lifetime; 5] = [
        Lifetime::Weights,
        Lifetime::Conversation,
        Lifetime::Token,
        Lifetime::Cache,
        Lifetime::Engine,
    ];
}

/// One named block of storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub space: Space,
    /// Planned bytes, from the config (what admission counts).
    pub bytes: u64,
    /// The object that owns (frees) it.
    pub owner: String,
    /// The stages or kernels that write it.
    pub writers: Vec<String>,
    pub lifetime: Lifetime,
    /// `false` when `bytes` is a measured allowance rather than a formula (said in the receipt).
    pub derived: bool,
    /// The allocating call that makes it (one call can make several entries); measurements are
    /// per call.
    pub call: String,
}

/// The arrangement of a run's storage, in the order it was registered.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StorageMap {
    pub entries: Vec<Entry>,
    /// Driver bytes measured per call, in the order first recorded.
    pub measured: Vec<(String, u64)>,
}

impl StorageMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a derived entry made by the call of the same name.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        space: Space,
        bytes: u64,
        owner: &str,
        writers: &[&str],
        lifetime: Lifetime,
    ) -> &mut Entry {
        let name = name.into();
        self.entries.push(Entry {
            call: name.clone(),
            name,
            space,
            bytes,
            owner: owner.into(),
            writers: writers.iter().map(|w| w.to_string()).collect(),
            lifetime,
            derived: true,
        });
        self.entries.last_mut().expect("just pushed")
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Adds `bytes` measured for `call`; `false` when no entry is made by that call (the bytes
    /// are still kept, so they are never lost from the receipt).
    pub fn record(&mut self, call: &str, bytes: u64) -> bool {
        match self.measured.iter_mut().find(|(c, _)| c == call) {
            Some((_, b)) => *b += bytes,
            None => self.measured.push((call.to_string(), bytes)),
        }
        self.entries.iter().any(|e| e.call == call)
    }

    /// Planned driver bytes of the entries `call` makes.
    pub fn planned_for(&self, call: &str) -> u64 {
        self.entries
            .iter()
            .filter(|e| e.call == call && e.space == Space::Driver)
            .map(|e| e.bytes)
            .sum()
    }

    /// Planned bytes in `space` (`None`: both).
    pub fn total(&self, space: Option<Space>) -> u64 {
        self.entries
            .iter()
            .filter(|e| space.is_none_or(|s| e.space == s))
            .map(|e| e.bytes)
            .sum()
    }

    /// Driver bytes measured over every probed call.
    pub fn measured_total(&self) -> u64 {
        self.measured.iter().map(|(_, b)| b).sum()
    }

    /// Calls whose measurement differs from the plan: (call, planned, measured).
    pub fn mismatches(&self) -> Vec<(&str, u64, u64)> {
        self.measured
            .iter()
            .filter_map(|(c, m)| {
                let p = self.planned_for(c);
                (p != *m).then_some((c.as_str(), p, *m))
            })
            .collect()
    }

    /// Calls with planned driver bytes that were never probed (`[]` once a run is complete).
    pub fn unprobed(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for e in self.entries.iter().filter(|e| e.space == Space::Driver) {
            let probed = self.measured.iter().any(|(c, _)| *c == e.call);
            if !probed && !out.contains(&e.call.as_str()) {
                out.push(e.call.as_str());
            }
        }
        out
    }

    fn by_lifetime(&self, space: Space) -> Vec<(Lifetime, u64)> {
        Lifetime::ALL
            .iter()
            .map(|&l| {
                let b = self
                    .entries
                    .iter()
                    .filter(|e| e.space == space && e.lifetime == l)
                    .map(|e| e.bytes)
                    .sum();
                (l, b)
            })
            .collect()
    }
}

/// What the process holds right now, read from `/proc`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcMemory {
    /// Summed `Size` of the `/dev/nvidia*` mappings (omega's driver allocations).
    pub driver_size: u64,
    /// Summed `Rss` of the same mappings.
    pub driver_rss: u64,
    /// Number of `/dev/nvidia*` mappings.
    pub driver_maps: u64,
    /// Summed `Anonymous` of every other mapping.
    pub heap: u64,
    /// Summed file-backed `Rss` of every other mapping (binary, libraries).
    pub file_rss: u64,
    /// `VmRSS` and `VmHWM` of `/proc/self/status`.
    pub vm_rss: u64,
    pub vm_hwm: u64,
    /// `/proc/meminfo` MemAvailable.
    pub mem_available: u64,
    /// omega's counters: allocations, their bytes, frees (`None`: no engine linked).
    pub omega_allocs: Option<u64>,
    pub omega_alloc_bytes: Option<u64>,
    pub omega_frees: Option<u64>,
}

fn kb_field(line: &str, key: &str) -> Option<u64> {
    let v = line.strip_prefix(key)?.trim();
    v.strip_suffix("kB")?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|k| k * 1024)
}

impl ProcMemory {
    /// Reads `/proc/self/smaps`, `/proc/self/status`, `/proc/meminfo` and omega's counters.
    pub fn read() -> Result<Self, String> {
        let smaps = std::fs::read_to_string("/proc/self/smaps")
            .map_err(|e| format!("/proc/self/smaps: {e}"))?;
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|e| format!("/proc/self/status: {e}"))?;
        let meminfo =
            std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
        let mut m = Self::parse(&smaps, &status, &meminfo)?;
        if let Some(s) = aien_omega_gpu::alloc_stats() {
            m.omega_allocs = Some(s.allocs);
            m.omega_alloc_bytes = Some(s.alloc_bytes);
            m.omega_frees = Some(s.frees);
        }
        Ok(m)
    }

    /// The `/proc` texts parsed (omega counters left `None`).
    pub fn parse(smaps: &str, status: &str, meminfo: &str) -> Result<Self, String> {
        let mut m = Self::default();
        let mut driver = false;
        for line in smaps.lines() {
            let mut cols = line.split_whitespace();
            let first = cols.next().unwrap_or("");
            // A mapping header: "start-end perms offset dev inode [path]".
            if first.contains('-') && !first.ends_with(':') {
                let path = cols.nth(4).unwrap_or("");
                driver = path.starts_with("/dev/nvidia");
                if driver {
                    m.driver_maps += 1;
                }
                continue;
            }
            if let Some(b) = kb_field(line, "Size:") {
                if driver {
                    m.driver_size += b;
                }
            } else if let Some(b) = kb_field(line, "Rss:") {
                if driver {
                    m.driver_rss += b;
                } else {
                    m.file_rss += b;
                }
            } else if let Some(b) = kb_field(line, "Anonymous:") {
                if !driver {
                    m.heap += b;
                    // Anonymous pages are counted in Rss too; file_rss keeps only file pages.
                    m.file_rss = m.file_rss.saturating_sub(b);
                }
            }
        }
        for line in status.lines() {
            if let Some(b) = kb_field(line, "VmRSS:") {
                m.vm_rss = b;
            } else if let Some(b) = kb_field(line, "VmHWM:") {
                m.vm_hwm = b;
            }
        }
        m.mem_available = meminfo
            .lines()
            .find_map(|l| kb_field(l, "MemAvailable:"))
            .ok_or("/proc/meminfo: no MemAvailable")?;
        Ok(m)
    }
}

/// Measures the driver and heap bytes one allocating call adds.
#[derive(Debug, Clone, Copy)]
pub struct Probe {
    before: ProcMemory,
}

impl Probe {
    pub fn start() -> Result<Self, String> {
        Ok(Self {
            before: ProcMemory::read()?,
        })
    }

    /// Driver bytes (mapping `Size`) added since [`Self::start`].
    pub fn driver_added(&self) -> Result<u64, String> {
        Ok(ProcMemory::read()?
            .driver_size
            .saturating_sub(self.before.driver_size))
    }
}

/// The map held against the process: what is named, what is there, and the difference.
#[derive(Debug, Clone)]
pub struct Receipt {
    pub label: String,
    pub map: StorageMap,
    /// Before the first allocation of the run (process start).
    pub baseline: ProcMemory,
    pub now: ProcMemory,
}

impl Receipt {
    pub fn new(
        label: impl Into<String>,
        map: StorageMap,
        baseline: ProcMemory,
        now: ProcMemory,
    ) -> Self {
        Self {
            label: label.into(),
            map,
            baseline,
            now,
        }
    }

    /// Driver bytes the process holds beyond the baseline.
    pub fn driver_held(&self) -> u64 {
        self.now
            .driver_size
            .saturating_sub(self.baseline.driver_size)
    }

    /// Heap bytes the process holds beyond the baseline.
    pub fn heap_held(&self) -> u64 {
        self.now.heap.saturating_sub(self.baseline.heap)
    }

    /// Held minus named, per space (negative: the map names more than is there).
    pub fn unaccounted(&self, space: Space) -> i64 {
        let (held, named) = match space {
            Space::Driver => (self.driver_held(), self.map.total(Some(Space::Driver))),
            Space::Heap => (self.heap_held(), self.map.total(Some(Space::Heap))),
        };
        held as i64 - named as i64
    }

    pub fn to_json(&self) -> Value {
        let mem = |p: &ProcMemory| {
            json!({
                "driver_size": p.driver_size, "driver_rss": p.driver_rss, "driver_maps": p.driver_maps,
                "heap": p.heap, "file_rss": p.file_rss, "vm_rss": p.vm_rss, "vm_hwm": p.vm_hwm,
                "mem_available": p.mem_available, "omega_allocs": p.omega_allocs,
                "omega_alloc_bytes": p.omega_alloc_bytes, "omega_frees": p.omega_frees,
            })
        };
        let entries: Vec<Value> = self
            .map
            .entries
            .iter()
            .map(|e| {
                json!({
                    "name": e.name, "space": e.space.as_str(), "bytes": e.bytes,
                    "owner": e.owner, "writers": e.writers, "lifetime": e.lifetime.as_str(),
                    "derived": e.derived, "call": e.call,
                })
            })
            .collect();
        let lifetimes = |s: Space| -> Value {
            self.map
                .by_lifetime(s)
                .into_iter()
                .map(|(l, b)| (l.as_str().to_string(), json!(b)))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        json!({
            "storage_map": 1,
            "label": self.label,
            "entries": entries,
            "totals": {
                "named_driver": self.map.total(Some(Space::Driver)),
                "named_heap": self.map.total(Some(Space::Heap)),
                "named": self.map.total(None),
                "measured_driver_calls": self.map.measured_total(),
                "held_driver": self.driver_held(),
                "held_heap": self.heap_held(),
                "unaccounted_driver": self.unaccounted(Space::Driver),
                "unaccounted_heap": self.unaccounted(Space::Heap),
                "driver_by_lifetime": lifetimes(Space::Driver),
                "heap_by_lifetime": lifetimes(Space::Heap),
            },
            "calls": self.map.measured.iter()
                .map(|(c, m)| json!({"call": c, "planned": self.map.planned_for(c), "measured": m}))
                .collect::<Vec<_>>(),
            "unprobed": self.map.unprobed(),
            "mismatches": self.map.mismatches().iter()
                .map(|(n, p, m)| json!({"name": n, "planned": p, "measured": m}))
                .collect::<Vec<_>>(),
            "baseline": mem(&self.baseline),
            "now": mem(&self.now),
        })
    }

    /// The printed map: one line per entry, then the totals and the unaccounted numbers.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        let _ = writeln!(s, "STORAGE MAP {}", self.label);
        let _ = writeln!(
            s,
            "{:<44} {:<6} {:>14} {:<12} {:<26} writers",
            "name", "space", "bytes", "lifetime", "owner"
        );
        for e in &self.map.entries {
            let _ = writeln!(
                s,
                "{:<44} {:<6} {:>14} {:<12} {:<26} {}{}",
                e.name,
                e.space.as_str(),
                e.bytes,
                e.lifetime.as_str(),
                e.owner,
                e.writers.join(","),
                if e.derived {
                    ""
                } else {
                    " [allowance, not derived]"
                }
            );
        }
        for sp in [Space::Driver, Space::Heap] {
            let _ = write!(s, "named {} by lifetime:", sp.as_str());
            for (l, b) in self.map.by_lifetime(sp) {
                let _ = write!(s, " {}={}", l.as_str(), b);
            }
            let _ = writeln!(s);
        }
        let _ = writeln!(
            s,
            "named driver {} ({:.1} MiB), held driver {} ({:.1} MiB), UNACCOUNTED driver {}",
            self.map.total(Some(Space::Driver)),
            mib(self.map.total(Some(Space::Driver))),
            self.driver_held(),
            mib(self.driver_held()),
            self.unaccounted(Space::Driver)
        );
        let _ = writeln!(
            s,
            "named heap {} ({:.1} MiB), held heap {} ({:.1} MiB), UNACCOUNTED heap {}",
            self.map.total(Some(Space::Heap)),
            mib(self.map.total(Some(Space::Heap))),
            self.heap_held(),
            mib(self.heap_held()),
            self.unaccounted(Space::Heap)
        );
        let _ = writeln!(
            s,
            "named total {} ({:.1} MiB); VmRSS {} VmHWM {} MemAvailable {}; driver maps {} (rss {}); omega counters allocs {:?} bytes {:?} frees {:?}",
            self.map.total(None),
            mib(self.map.total(None)),
            self.now.vm_rss,
            self.now.vm_hwm,
            self.now.mem_available,
            self.now.driver_maps,
            self.now.driver_rss,
            self.now.omega_allocs,
            self.now.omega_alloc_bytes,
            self.now.omega_frees
        );
        let _ = writeln!(
            s,
            "probed calls {}: measured driver {} vs planned {}; unprobed calls: {:?}",
            self.map.measured.len(),
            self.map.measured_total(),
            self.map
                .measured
                .iter()
                .map(|(c, _)| self.map.planned_for(c))
                .sum::<u64>(),
            self.map.unprobed()
        );
        let mm = self.map.mismatches();
        let _ = writeln!(s, "plan vs measured mismatches: {}", mm.len());
        for (n, p, m) in mm {
            let _ = writeln!(
                s,
                "  {n}: planned {p} measured {m} (diff {})",
                m as i64 - p as i64
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMAPS: &str = "\
aaaa0000-aaaa2000 r-xp 00000000 103:02 123 /usr/bin/x
Size:                  8 kB
Rss:                   8 kB
Anonymous:             0 kB
ffff0000-ffff4000 rw-s 00000000 00:05 9 /dev/nvidia0
Size:                 16 kB
Rss:                   4 kB
Anonymous:             0 kB
ffff8000-ffffc000 rw-s 00000000 00:05 9 /dev/nvidiactl
Size:                 16 kB
Rss:                  16 kB
Anonymous:             0 kB
bbbb0000-bbbb3000 rw-p 00000000 00:00 0 [heap]
Size:                 12 kB
Rss:                  12 kB
Anonymous:            12 kB
cccc0000-cccc1000 rw-p 00000000 00:00 0
Size:                  4 kB
Rss:                   4 kB
Anonymous:             4 kB
VmFlags: rd wr mr mw me ac
";
    const STATUS: &str = "Name:\tx\nVmHWM:\t     100 kB\nVmRSS:\t      44 kB\n";
    const MEMINFO: &str = "MemTotal:  1000 kB\nMemAvailable:     500 kB\n";

    #[test]
    fn parse_splits_driver_heap_and_file() {
        let m = ProcMemory::parse(SMAPS, STATUS, MEMINFO).unwrap();
        assert_eq!(m.driver_maps, 2);
        assert_eq!(m.driver_size, 32 * 1024);
        assert_eq!(m.driver_rss, 20 * 1024);
        assert_eq!(m.heap, 16 * 1024);
        assert_eq!(m.file_rss, 8 * 1024);
        assert_eq!(m.vm_rss, 44 * 1024);
        assert_eq!(m.vm_hwm, 100 * 1024);
        assert_eq!(m.mem_available, 500 * 1024);
    }

    fn sample() -> StorageMap {
        let mut s = StorageMap::new();
        s.add("w", Space::Driver, 8192, "t", &["load"], Lifetime::Weights);
        s.add(
            "kv",
            Space::Driver,
            4096,
            "c",
            &["attn"],
            Lifetime::Conversation,
        );
        s.add(
            "embed",
            Space::Heap,
            1000,
            "e",
            &["load"],
            Lifetime::Weights,
        );
        s
    }

    #[test]
    fn unaccounted_is_held_minus_named_and_sees_an_unregistered_block() {
        let base = ProcMemory::default();
        let mut now = ProcMemory {
            driver_size: 12288,
            heap: 1000,
            ..Default::default()
        };
        let r = Receipt::new("t", sample(), base, now);
        assert_eq!(r.unaccounted(Space::Driver), 0);
        assert_eq!(r.unaccounted(Space::Heap), 0);
        // An allocation nobody registered shows up, in its own space only.
        now.driver_size += 1 << 20;
        let r = Receipt::new("t", sample(), base, now);
        assert_eq!(r.unaccounted(Space::Driver), 1 << 20);
        assert_eq!(r.unaccounted(Space::Heap), 0);
        assert!(r.to_text().contains("UNACCOUNTED driver 1048576"));
        assert_eq!(r.to_json()["totals"]["unaccounted_driver"], 1 << 20);
    }

    #[test]
    fn measured_calls_and_mismatches() {
        let mut s = sample();
        // Two entries made by one call: planned per call.
        s.add(
            "kv rope",
            Space::Driver,
            4096,
            "c",
            &["open"],
            Lifetime::Cache,
        )
        .call = "kv".into();
        assert_eq!(s.unprobed(), ["w", "kv"]);
        assert!(s.record("w", 8192));
        assert!(s.record("kv", 4096));
        assert!(s.record("kv", 8192));
        assert!(!s.record("nope", 1));
        assert_eq!(s.planned_for("kv"), 8192);
        assert_eq!(s.mismatches(), vec![("kv", 8192, 12288), ("nope", 0, 1)]);
        assert_eq!(s.measured_total(), 20481);
        assert!(s.unprobed().is_empty());
        assert_eq!(s.total(None), 17384);
    }

    #[test]
    fn json_is_stable() {
        let r = Receipt::new("t", sample(), ProcMemory::default(), ProcMemory::default());
        let a = serde_json::to_string(&r.to_json()).unwrap();
        let b = serde_json::to_string(&r.to_json()).unwrap();
        assert_eq!(a, b);
        let names: Vec<_> = r.to_json()["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["w", "kv", "embed"]);
    }
}
