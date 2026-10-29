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

//! Process supervision with typed, generation-scoped readiness (issue #534).
//!
//! Readiness contract:
//! - Every successful spawn increments the service's `generation` and puts it in
//!   [`ServiceState::Spawned`]. A spawned child is never [`ServiceState::Ready`] until a
//!   probe taken for that same pid and generation succeeds.
//! - With `health_url`: `Ready` only after an HTTP GET to that URL returns a 2xx status
//!   while the pid is alive. A non-2xx status, a connection error or a timeout gives
//!   [`ServiceState::Unhealthy`].
//! - Without `health_url`: `Ready` once the liveness check finds the pid alive in `/proc`.
//!   This is liveness, not readiness: it says the process exists, not that it serves.
//! - Probes run with the state mutex released. Each result carries the (pid, generation)
//!   it was taken for and is discarded when the service has since restarted or stopped,
//!   so a slow answer for an old child can never mark its replacement ready.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, Notify};
use tokio::task::JoinSet;
use tokio::time::{interval, sleep, MissedTickBehavior};
use tracing::{error, info, warn};

use crate::config::{ServiceConfig, SupervisorConfig};
use crate::proc_telemetry::get_process_stats;

pub fn compute_restart_delay(restart_count: u32, base_secs: u64, max_secs: u64) -> Duration {
    let factor = 2u64.saturating_pow(restart_count.saturating_sub(1));
    let secs = std::cmp::min(base_secs.saturating_mul(factor), max_secs);
    Duration::from_secs(secs)
}

/// Lifecycle state of one managed service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceState {
    /// Not running and not scheduled to run (initial state, shutdown, or `restart = "no"`).
    Stopped,
    /// Child spawned; no probe has been dispatched for this generation yet.
    Spawned,
    /// First probe for this generation is in flight; not ready.
    Probing,
    /// Readiness condition met for the current generation (see module docs).
    Ready,
    /// Last probe failed, the child exited and awaits restart, or the spawn failed.
    Unhealthy,
}

impl ServiceState {
    pub fn is_ready(self) -> bool {
        self == ServiceState::Ready
    }
}

#[derive(Debug, Clone)]
pub struct ManagedServiceState {
    pub name: String,
    pub config: ServiceConfig,
    pub pid: Option<u32>,
    /// Incremented on every successful spawn; 0 means never spawned.
    pub generation: u64,
    pub state: ServiceState,
    pub restart_count: u32,
    pub uptime_secs: u64,
    pub rss_mb: f64,
    pub spawned_at: Option<Instant>,
}

/// Tunables. `Default` keeps the previous fixed values: 3 s probe tick, 1500 ms HTTP
/// timeout, restart backoff base 2 s capped at 60 s, 5 s retry after a failed spawn.
#[derive(Debug, Clone)]
pub struct SupervisorOptions {
    pub probe_interval: Duration,
    pub probe_timeout: Duration,
    pub restart_base_secs: u64,
    pub restart_max_secs: u64,
    pub spawn_retry_delay: Duration,
    /// How long shutdown waits for supervise tasks to kill and reap their children.
    pub shutdown_grace: Duration,
}

impl Default for SupervisorOptions {
    fn default() -> Self {
        Self {
            probe_interval: Duration::from_secs(3),
            probe_timeout: Duration::from_millis(1500),
            restart_base_secs: 2,
            restart_max_secs: 60,
            spawn_retry_delay: Duration::from_secs(5),
            shutdown_grace: Duration::from_secs(5),
        }
    }
}

/// What a probe was taken for, captured under the mutex before the lock is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeTicket {
    pub name: String,
    pub pid: u32,
    pub generation: u64,
    pub health_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// Pid alive and, when a health_url is configured, it answered 2xx.
    Healthy,
    /// health_url answered with this non-2xx status.
    HttpFailure(u16),
    /// health_url could not be reached (connect error, timeout, protocol error).
    HttpError(String),
    /// The pid is no longer alive.
    ProcessGone,
}

#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub ticket: ProbeTicket,
    pub outcome: ProbeOutcome,
    pub rss_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyResult {
    Applied {
        from: ServiceState,
        to: ServiceState,
    },
    /// The service restarted or stopped since the ticket was taken; result discarded.
    Stale,
    UnknownService,
}

/// Records a successful spawn and returns the new generation.
pub fn record_spawn(
    states: &mut HashMap<String, ManagedServiceState>,
    name: &str,
    pid: Option<u32>,
) -> u64 {
    match states.get_mut(name) {
        Some(st) => {
            st.generation += 1;
            st.pid = pid;
            st.state = ServiceState::Spawned;
            st.spawned_at = Some(Instant::now());
            st.uptime_secs = 0;
            st.rss_mb = 0.0;
            st.generation
        }
        None => 0,
    }
}

/// Records that the child of `generation` is gone. `next` is `Unhealthy` while a restart
/// is pending, `Stopped` for shutdown or `restart = "no"`. Returns the restart count.
pub fn record_exit(
    states: &mut HashMap<String, ManagedServiceState>,
    name: &str,
    generation: u64,
    next: ServiceState,
    count_restart: bool,
) -> u32 {
    match states.get_mut(name) {
        Some(st) if st.generation == generation => {
            st.pid = None;
            st.state = next;
            st.spawned_at = None;
            if count_restart {
                st.restart_count += 1;
            }
            st.restart_count
        }
        Some(st) => st.restart_count,
        None => 0,
    }
}

/// Takes probe tickets for every running service and moves `Spawned` to `Probing`.
/// Call with the lock held; it does no I/O.
pub fn begin_probe_round(states: &mut HashMap<String, ManagedServiceState>) -> Vec<ProbeTicket> {
    let mut tickets = Vec::new();
    for st in states.values_mut() {
        let Some(pid) = st.pid else { continue };
        if st.state == ServiceState::Stopped {
            continue;
        }
        if st.state == ServiceState::Spawned {
            st.state = ServiceState::Probing;
        }
        tickets.push(ProbeTicket {
            name: st.name.clone(),
            pid,
            generation: st.generation,
            health_url: st.config.health_url.clone(),
        });
    }
    tickets
}

/// Applies one probe result if, and only if, it still describes the current child.
/// Call with the lock held; it does no I/O.
pub fn apply_probe_report(
    states: &mut HashMap<String, ManagedServiceState>,
    report: &ProbeReport,
) -> ApplyResult {
    let Some(st) = states.get_mut(&report.ticket.name) else {
        return ApplyResult::UnknownService;
    };
    if st.pid != Some(report.ticket.pid)
        || st.generation != report.ticket.generation
        || st.state == ServiceState::Stopped
    {
        return ApplyResult::Stale;
    }
    let from = st.state;
    let to = match report.outcome {
        ProbeOutcome::Healthy => ServiceState::Ready,
        ProbeOutcome::HttpFailure(_) | ProbeOutcome::HttpError(_) | ProbeOutcome::ProcessGone => {
            ServiceState::Unhealthy
        }
    };
    st.state = to;
    if report.outcome != ProbeOutcome::ProcessGone {
        st.rss_mb = report.rss_bytes as f64 / (1024.0 * 1024.0);
    }
    st.uptime_secs = st.spawned_at.map(|t| t.elapsed().as_secs()).unwrap_or(0);
    ApplyResult::Applied { from, to }
}

/// Runs one probe with no lock held. The HTTP request is bounded by the client timeout.
pub async fn probe_one(client: aien_localhttp::Client, ticket: ProbeTicket) -> ProbeReport {
    let stats = get_process_stats(ticket.pid);
    let outcome = if !stats.is_alive {
        ProbeOutcome::ProcessGone
    } else {
        match ticket.health_url.as_deref() {
            None => ProbeOutcome::Healthy,
            Some(url) => match client.get(url).send().await {
                Ok(res) if res.status().is_success() => ProbeOutcome::Healthy,
                Ok(res) => ProbeOutcome::HttpFailure(res.status().as_u16()),
                Err(e) => ProbeOutcome::HttpError(e.to_string()),
            },
        }
    };
    ProbeReport {
        ticket,
        outcome,
        rss_bytes: stats.rss_bytes,
    }
}

type StateMap = Arc<Mutex<HashMap<String, ManagedServiceState>>>;

pub struct Supervisor {
    config: SupervisorConfig,
    options: SupervisorOptions,
    states: StateMap,
}

impl Supervisor {
    pub fn new(config: SupervisorConfig) -> Self {
        Self::with_options(config, SupervisorOptions::default())
    }

    pub fn with_options(config: SupervisorConfig, options: SupervisorOptions) -> Self {
        let mut states = HashMap::new();
        for (name, s_cfg) in &config.services {
            states.insert(
                name.clone(),
                ManagedServiceState {
                    name: name.clone(),
                    config: s_cfg.clone(),
                    pid: None,
                    generation: 0,
                    state: ServiceState::Stopped,
                    restart_count: 0,
                    uptime_secs: 0,
                    rss_mb: 0.0,
                    spawned_at: None,
                },
            );
        }
        Self {
            config,
            options,
            states: Arc::new(Mutex::new(states)),
        }
    }

    /// Copy of the in-process state. There is no IPC: only the process running the
    /// daemon can see this.
    pub async fn snapshot(&self) -> HashMap<String, ManagedServiceState> {
        self.states.lock().await.clone()
    }

    /// Runs until Ctrl-C.
    pub async fn run_daemon(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.run_until(async {
            let _ = tokio::signal::ctrl_c().await;
            info!("Received termination signal. Shutting down all managed services gracefully...");
        })
        .await?;
        Ok(())
    }

    /// Runs until `shutdown` completes, then kills and reaps every child.
    ///
    /// Refuses to start, before spawning anything, when a `health_url` is not a
    /// plain `http://` URL (an `https://` URL fails with
    /// [`aien_localhttp::Error::HttpsRefused`]).
    pub async fn run_until<F>(&self, shutdown: F) -> Result<(), aien_localhttp::Error>
    where
        F: Future<Output = ()>,
    {
        self.config.validate_health_urls()?;
        info!(
            "Starting SparkOS Sovereign Supervisor with {} service(s)...",
            self.config.services.len()
        );
        let client = aien_localhttp::Client::builder()
            .timeout(self.options.probe_timeout)
            .connect_timeout(self.options.probe_timeout)
            .build();

        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let kick = Arc::new(Notify::new());
        let mut supervisors = JoinSet::new();
        for (name, s_cfg) in self.config.services.clone() {
            supervisors.spawn(Self::supervise_loop(
                name,
                s_cfg,
                self.states.clone(),
                shutdown_tx.subscribe(),
                kick.clone(),
                self.options.clone(),
            ));
        }

        tokio::pin!(shutdown);
        let mut tick = interval(self.options.probe_interval);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                _ = &mut shutdown => break,
                _ = tick.tick() => {}
                _ = kick.notified() => {}
            }

            let tickets = {
                let mut map = self.states.lock().await;
                begin_probe_round(&mut map)
            };
            let mut probes = JoinSet::new();
            for ticket in tickets {
                probes.spawn(probe_one(client.clone(), ticket));
            }
            let stop = loop {
                tokio::select! {
                    biased;
                    _ = &mut shutdown => {
                        probes.abort_all();
                        break true;
                    }
                    joined = probes.join_next() => match joined {
                        None => break false,
                        Some(Ok(report)) => {
                            let applied = {
                                let mut map = self.states.lock().await;
                                apply_probe_report(&mut map, &report)
                            };
                            log_apply(&report, applied);
                        }
                        Some(Err(e)) => warn!("Health probe task failed: {}", e),
                    }
                }
            };
            if stop {
                break;
            }
        }

        let _ = shutdown_tx.send(());
        let deadline = tokio::time::Instant::now() + self.options.shutdown_grace;
        while !supervisors.is_empty() {
            match tokio::time::timeout_at(deadline, supervisors.join_next()).await {
                Ok(Some(Err(e))) => warn!("Supervise task failed: {}", e),
                Ok(_) => {}
                Err(_) => {
                    warn!(
                        "{} supervise task(s) did not stop within {:?}; aborting them",
                        supervisors.len(),
                        self.options.shutdown_grace
                    );
                    supervisors.abort_all();
                    break;
                }
            }
        }

        info!("SparkOS Sovereign Supervisor halted cleanly.");
        Ok(())
    }

    async fn supervise_loop(
        name: String,
        config: ServiceConfig,
        states: StateMap,
        mut shutdown_rx: broadcast::Receiver<()>,
        kick: Arc<Notify>,
        options: SupervisorOptions,
    ) {
        loop {
            let command_path = if let Some(stripped) = config.command.strip_prefix("~/") {
                let home = std::env::var("HOME")
                    .or_else(|_| std::env::var("USERPROFILE"))
                    .unwrap_or_else(|_| ".".to_string());
                format!("{}/{}", home, stripped)
            } else {
                config.command.clone()
            };
            info!(
                "[{}] Spawning process: {} {:?}...",
                name, command_path, config.args
            );
            let mut cmd = tokio::process::Command::new(&command_path);
            cmd.args(&config.args);
            for (k, v) in &config.env {
                cmd.env(k, v);
            }

            match cmd.spawn() {
                Ok(mut child) => {
                    let pid = child.id();
                    let generation = {
                        let mut map = states.lock().await;
                        record_spawn(&mut map, &name, pid)
                    };
                    kick.notify_one();
                    info!(
                        "[{}] Running with PID {:?} (generation {}); not ready until probed",
                        name, pid, generation
                    );

                    tokio::select! {
                        _ = shutdown_rx.recv() => {
                            info!("[{}] Terminating child PID {:?} for supervisor shutdown...", name, pid);
                            let _ = child.kill().await;
                            let _ = child.wait().await;
                            let mut map = states.lock().await;
                            record_exit(&mut map, &name, generation, ServiceState::Stopped, false);
                            return;
                        }
                        status_res = child.wait() => {
                            match status_res {
                                Ok(status) => {
                                    warn!("[{}] Process exited with status: {}", name, status);
                                }
                                Err(e) => {
                                    error!("[{}] Process error: {}", name, e);
                                }
                            }
                        }
                    }

                    let next = if config.restart == "no" {
                        ServiceState::Stopped
                    } else {
                        ServiceState::Unhealthy
                    };
                    let st_restarts = {
                        let mut map = states.lock().await;
                        record_exit(&mut map, &name, generation, next, true)
                    };

                    if config.restart == "no" {
                        info!("[{}] Configured restart is no. Halting supervision.", name);
                        break;
                    }

                    let delay = compute_restart_delay(
                        st_restarts,
                        options.restart_base_secs,
                        options.restart_max_secs,
                    );
                    info!(
                        "[{}] Restarting in {}s (restart #{})...",
                        name,
                        delay.as_secs(),
                        st_restarts
                    );
                    tokio::select! {
                        _ = shutdown_rx.recv() => {
                            mark_stopped(&states, &name).await;
                            return;
                        }
                        _ = sleep(delay) => {}
                    }
                }
                Err(e) => {
                    error!(
                        "[{}] Failed to spawn: {}. Retrying in {:?}...",
                        name, e, options.spawn_retry_delay
                    );
                    {
                        let mut map = states.lock().await;
                        if let Some(st) = map.get_mut(&name) {
                            st.state = ServiceState::Unhealthy;
                        }
                    }
                    tokio::select! {
                        _ = shutdown_rx.recv() => {
                            mark_stopped(&states, &name).await;
                            return;
                        }
                        _ = sleep(options.spawn_retry_delay) => {}
                    }
                }
            }
        }
    }
}

/// Used only when no child is running (restart backoff or spawn retry).
async fn mark_stopped(states: &StateMap, name: &str) {
    let mut map = states.lock().await;
    if let Some(st) = map.get_mut(name) {
        st.pid = None;
        st.state = ServiceState::Stopped;
        st.spawned_at = None;
    }
}

fn log_apply(report: &ProbeReport, applied: ApplyResult) {
    let t = &report.ticket;
    match applied {
        ApplyResult::Applied { from, to } if from != to => match to {
            ServiceState::Ready => info!(
                "[{}] {:?} -> Ready (pid {}, generation {})",
                t.name, from, t.pid, t.generation
            ),
            _ => warn!(
                "[{}] {:?} -> {:?} (pid {}, generation {}): {:?}",
                t.name, from, to, t.pid, t.generation, report.outcome
            ),
        },
        ApplyResult::Applied { .. } => {}
        ApplyResult::Stale => info!(
            "[{}] Discarded stale probe result for pid {} generation {}",
            t.name, t.pid, t.generation
        ),
        ApplyResult::UnknownService => warn!("[{}] Probe result for unknown service", t.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svc(name: &str, health_url: Option<&str>) -> HashMap<String, ManagedServiceState> {
        let mut config = SupervisorConfig {
            services: HashMap::new(),
        };
        config.services.insert(
            name.to_string(),
            ServiceConfig {
                command: "sleep".into(),
                args: vec!["30".into()],
                env: HashMap::new(),
                restart: "always".into(),
                health_url: health_url.map(String::from),
            },
        );
        Supervisor::new(config).states.try_lock().unwrap().clone()
    }

    fn report(ticket: &ProbeTicket, outcome: ProbeOutcome) -> ProbeReport {
        ProbeReport {
            ticket: ticket.clone(),
            outcome,
            rss_bytes: 4096,
        }
    }

    #[test]
    fn test_compute_restart_delay_backoff() {
        assert_eq!(compute_restart_delay(1, 2, 60), Duration::from_secs(2));
        assert_eq!(compute_restart_delay(2, 2, 60), Duration::from_secs(4));
        assert_eq!(compute_restart_delay(3, 2, 60), Duration::from_secs(8));
        assert_eq!(compute_restart_delay(4, 2, 60), Duration::from_secs(16));
        assert_eq!(compute_restart_delay(5, 2, 60), Duration::from_secs(32));
        assert_eq!(compute_restart_delay(6, 2, 60), Duration::from_secs(60)); // capped at max
    }

    #[test]
    fn default_options_keep_previous_fixed_values() {
        let o = SupervisorOptions::default();
        assert_eq!(o.probe_interval, Duration::from_secs(3));
        assert_eq!(o.probe_timeout, Duration::from_millis(1500));
        assert_eq!((o.restart_base_secs, o.restart_max_secs), (2, 60));
        assert_eq!(o.spawn_retry_delay, Duration::from_secs(5));
    }

    #[test]
    fn new_service_starts_stopped_generation_zero() {
        let m = svc("a", None);
        assert_eq!(m["a"].state, ServiceState::Stopped);
        assert_eq!(m["a"].generation, 0);
    }

    #[test]
    fn spawn_is_not_ready_and_probing_is_not_ready() {
        let mut m = svc("a", None);
        assert_eq!(record_spawn(&mut m, "a", Some(100)), 1);
        assert_eq!(m["a"].state, ServiceState::Spawned);
        assert!(!m["a"].state.is_ready());
        let tickets = begin_probe_round(&mut m);
        assert_eq!(tickets.len(), 1);
        assert_eq!(m["a"].state, ServiceState::Probing);
        assert!(!m["a"].state.is_ready());
    }

    #[test]
    fn healthy_report_makes_ready_failures_make_unhealthy() {
        let mut m = svc("a", Some("http://127.0.0.1:1/health"));
        record_spawn(&mut m, "a", Some(100));
        let t = begin_probe_round(&mut m).remove(0);
        assert_eq!(
            apply_probe_report(&mut m, &report(&t, ProbeOutcome::Healthy)),
            ApplyResult::Applied {
                from: ServiceState::Probing,
                to: ServiceState::Ready
            }
        );
        for bad in [
            ProbeOutcome::HttpFailure(500),
            ProbeOutcome::HttpError("refused".into()),
            ProbeOutcome::ProcessGone,
        ] {
            assert_eq!(
                apply_probe_report(&mut m, &report(&t, bad)),
                ApplyResult::Applied {
                    from: ServiceState::Ready,
                    to: ServiceState::Unhealthy
                }
            );
            apply_probe_report(&mut m, &report(&t, ProbeOutcome::Healthy));
        }
    }

    #[test]
    fn stale_generation_is_ignored_even_with_reused_pid() {
        let mut m = svc("a", None);
        record_spawn(&mut m, "a", Some(100));
        let old = begin_probe_round(&mut m).remove(0);
        record_exit(&mut m, "a", 1, ServiceState::Unhealthy, true);
        // Same pid again: only the generation tells the two children apart.
        assert_eq!(record_spawn(&mut m, "a", Some(100)), 2);
        assert_eq!(
            apply_probe_report(&mut m, &report(&old, ProbeOutcome::Healthy)),
            ApplyResult::Stale
        );
        assert_eq!(m["a"].state, ServiceState::Spawned);
    }

    #[test]
    fn stale_pid_is_ignored() {
        let mut m = svc("a", None);
        record_spawn(&mut m, "a", Some(100));
        let mut t = begin_probe_round(&mut m).remove(0);
        t.pid = 101;
        assert_eq!(
            apply_probe_report(&mut m, &report(&t, ProbeOutcome::Healthy)),
            ApplyResult::Stale
        );
        assert_eq!(m["a"].state, ServiceState::Probing);
    }

    #[test]
    fn result_after_stop_is_ignored() {
        let mut m = svc("a", None);
        record_spawn(&mut m, "a", Some(100));
        let t = begin_probe_round(&mut m).remove(0);
        record_exit(&mut m, "a", 1, ServiceState::Stopped, false);
        assert_eq!(
            apply_probe_report(&mut m, &report(&t, ProbeOutcome::Healthy)),
            ApplyResult::Stale
        );
        assert_eq!(m["a"].state, ServiceState::Stopped);
        assert!(begin_probe_round(&mut m).is_empty());
    }

    #[test]
    fn unknown_service_is_reported() {
        let mut m = svc("a", None);
        let t = ProbeTicket {
            name: "ghost".into(),
            pid: 1,
            generation: 1,
            health_url: None,
        };
        assert_eq!(
            apply_probe_report(&mut m, &report(&t, ProbeOutcome::Healthy)),
            ApplyResult::UnknownService
        );
    }

    #[test]
    fn exit_for_old_generation_does_not_touch_new_child() {
        let mut m = svc("a", None);
        record_spawn(&mut m, "a", Some(100));
        record_spawn(&mut m, "a", Some(200));
        record_exit(&mut m, "a", 1, ServiceState::Unhealthy, true);
        assert_eq!(m["a"].pid, Some(200));
        assert_eq!(m["a"].state, ServiceState::Spawned);
        assert_eq!(m["a"].restart_count, 0);
    }
}
