//! Experimental known-demand weight flow over the existing verified pager.
//! One serialized compute consumer. No expert substitution or reordered reduction.
use crate::weight_policy::NativePolicy;
use crate::{ExpertPager, Pinned};
use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowMode {
    Synchronous,
    Fixed,
    Adaptive,
}
#[derive(Debug)]
pub struct FlowSettings {
    pub mode: FlowMode,
    pub max_inflight: usize,
    policy: Option<NativePolicy>,
    depth: usize,
    calm: u64,
    previous_compute: u64,
}
impl FlowSettings {
    pub fn synchronous() -> Self {
        Self::fixed_mode(FlowMode::Synchronous, 1)
    }
    pub fn fixed(max_inflight: usize) -> Self {
        Self::fixed_mode(FlowMode::Fixed, max_inflight)
    }
    fn fixed_mode(mode: FlowMode, max_inflight: usize) -> Self {
        Self {
            mode,
            max_inflight,
            policy: None,
            depth: 2,
            calm: 0,
            previous_compute: 0,
        }
    }
    pub fn adaptive(max_inflight: usize, binary: &Path) -> Result<Self, String> {
        if !(1..=32).contains(&max_inflight) {
            return Err("weight flow capacity must be 1..32".into());
        }
        let mut s = Self::fixed_mode(FlowMode::Adaptive, max_inflight);
        s.policy = Some(NativePolicy::open(binary)?);
        Ok(s)
    }
    fn depth(&self, cap: usize) -> usize {
        match self.mode {
            FlowMode::Synchronous => 1,
            FlowMode::Fixed => 2.min(cap),
            FlowMode::Adaptive => self.depth.min(cap),
        }
    }
    fn observe(&mut self, cap: usize, wait: u64, compute: u64) -> Result<(), String> {
        if self.mode == FlowMode::Adaptive {
            let (depth, calm) = self
                .policy
                .as_mut()
                .ok_or("adaptive mode requires native policy")?
                .decide([
                    1,
                    2,
                    cap as u64,
                    self.depth.min(cap) as u64,
                    wait,
                    compute,
                    self.previous_compute,
                    self.calm,
                    0,
                    0,
                ])?;
            self.depth = depth;
            self.calm = calm;
            self.previous_compute = compute;
        }
        Ok(())
    }
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct FlowEvent {
    pub ordinal: usize,
    pub expert: u16,
    pub prepare_start_ns: u64,
    pub prepare_end_ns: u64,
    pub compute_start_ns: u64,
    pub compute_end_ns: u64,
    pub exposed_wait_ns: u64,
    pub depth: usize,
}
#[derive(Debug, Default, serde::Serialize)]
pub struct FlowTrace {
    pub peak_inflight: usize,
    pub experts: usize,
    pub wait_ns: u64,
    pub compute_ns: u64,
    pub wall_ns: u64,
    pub events: Vec<FlowEvent>,
}
struct Lease<'a> {
    pager: &'a ExpertPager,
    pinned: Pinned,
    start: u64,
    end: u64,
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.pager.release(&self.pinned);
    }
}
fn ns(origin: Instant) -> u64 {
    origin.elapsed().as_nanos().min(i64::MAX as u128) as u64
}

/// The caller must serialize flow calls on this pager (PagedExperts does so).
/// Cancel/error stops admission, joins already issued prepares and releases every pin.
/// In-kernel reads are not forcibly interrupted; the existing read deadline discards late data.
pub fn run<F>(
    pager: &ExpertPager,
    layer: u16,
    experts: &[u16],
    settings: &mut FlowSettings,
    cancelled: &AtomicBool,
    mut compute: F,
) -> Result<FlowTrace, String>
where
    F: FnMut(usize, &Pinned) -> Result<(), String>,
{
    if !(1..=32).contains(&settings.max_inflight) || experts.len() > 32 || pager.capacity() == 0 {
        return Err("weight flow capacity must be 1..32 and pager nonempty".into());
    }
    let mut unique = HashSet::new();
    for &id in experts {
        if !unique.insert(id) {
            return Err("duplicate routed expert".into());
        }
        if !pager
            .manifest()
            .experts
            .iter()
            .any(|e| e.layer == layer && e.expert == id)
        {
            return Err("unknown routed expert".into());
        }
    }
    if settings.mode == FlowMode::Adaptive && settings.policy.is_none() {
        return Err("adaptive mode requires native policy".into());
    }
    let cap = settings.max_inflight.min(pager.capacity());
    let origin = Instant::now();
    std::thread::scope(|scope| {
        let mut queue = VecDeque::new();
        let mut submitted = 0;
        let mut trace = FlowTrace::default();
        let outcome: Result<(), String> = (|| {
            for current in 0..experts.len() {
                if cancelled.load(Ordering::Acquire) {
                    return Err("weight flow cancelled".into());
                }
                let depth = settings.depth(cap);
                while submitted < experts.len() && queue.len() < depth {
                    let expert = experts[submitted];
                    queue.push_back(scope.spawn(move || {
                        if cancelled.load(Ordering::Acquire) {
                            return Err("weight flow cancelled".to_string());
                        }
                        let start = ns(origin);
                        let pinned = pager.ensure(layer, &[expert]).map_err(|e| e.to_string())?;
                        let lease = Lease {
                            pager,
                            pinned,
                            start,
                            end: ns(origin),
                        };
                        if cancelled.load(Ordering::Acquire) {
                            return Err("weight flow cancelled".to_string());
                        }
                        Ok(lease)
                    }));
                    submitted += 1;
                }
                trace.peak_inflight = trace.peak_inflight.max(queue.len());
                let wait_start = ns(origin);
                let lease = queue
                    .pop_front()
                    .ok_or("weight flow empty queue")?
                    .join()
                    .map_err(|_| "weight prepare panicked")??;
                let compute_start = ns(origin);
                if cancelled.load(Ordering::Acquire) {
                    return Err("weight flow cancelled".into());
                }
                // Only count time before readiness as exposed wait, not thread-join overhead.
                let wait = lease.end.saturating_sub(wait_start);
                compute(current, &lease.pinned)?;
                let compute_end = ns(origin);
                let elapsed = compute_end - compute_start;
                trace.events.push(FlowEvent {
                    ordinal: current,
                    expert: experts[current],
                    prepare_start_ns: lease.start,
                    prepare_end_ns: lease.end,
                    compute_start_ns: compute_start,
                    compute_end_ns: compute_end,
                    exposed_wait_ns: wait,
                    depth,
                });
                drop(lease);
                trace.experts += 1;
                trace.wait_ns += wait;
                trace.compute_ns += elapsed;
                settings.observe(cap, wait, elapsed)?;
            }
            if cancelled.load(Ordering::Acquire) {
                return Err("weight flow cancelled".into());
            }
            Ok(())
        })();
        // A failed earlier stage must not leak successfully prepared future leases.
        for pending in queue {
            drop(pending.join());
        }
        outcome?;
        trace.wall_ns = ns(origin);
        Ok(trace)
    })
}
