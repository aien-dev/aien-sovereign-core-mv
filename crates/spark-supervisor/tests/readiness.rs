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

//! Tokio integration tests for typed readiness (issue #534). Real short-lived children
//! (`sleep`) and a tiny local HTTP responder on a tokio TcpListener.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use spark_supervisor::config::{ServiceConfig, SupervisorConfig};
use spark_supervisor::proc_telemetry::{get_process_stats, parse_stat_state};
use spark_supervisor::supervisor::{
    probe_one, ManagedServiceState, ProbeOutcome, ProbeTicket, ServiceState, Supervisor,
    SupervisorOptions,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{oneshot, Semaphore};
use tokio::task::JoinHandle;

const WAIT: Duration = Duration::from_secs(5);
/// No probe may hold the state lock: a snapshot that takes longer than this fails fast.
const LOCK_BOUND: Duration = Duration::from_secs(2);

/// Status code the responder answers with; 0 means hold the connection open, never answer.
struct Responder {
    url: String,
    status: Arc<AtomicU16>,
    hits: Arc<AtomicUsize>,
}

async fn responder(status: u16, delay: Duration) -> Responder {
    responder_with(status, delay, None).await
}

/// With `gate`, each answer first takes one permit, so the test decides when it is sent.
async fn responder_with(status: u16, delay: Duration, gate: Option<Arc<Semaphore>>) -> Responder {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/health", listener.local_addr().unwrap());
    let status = Arc::new(AtomicU16::new(status));
    let hits = Arc::new(AtomicUsize::new(0));
    let (s, h) = (status.clone(), hits.clone());
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let (s, h, gate) = (s.clone(), h.clone(), gate.clone());
            tokio::spawn(async move {
                let mut buf = vec![0u8; 2048];
                let mut got = Vec::new();
                while !got.windows(4).any(|w| w == b"\r\n\r\n") {
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => got.extend_from_slice(&buf[..n]),
                    }
                }
                h.fetch_add(1, Ordering::SeqCst);
                if let Some(g) = gate {
                    g.acquire().await.unwrap().forget();
                }
                tokio::time::sleep(delay).await;
                let code = s.load(Ordering::SeqCst);
                if code == 0 {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    return;
                }
                let resp =
                    format!("HTTP/1.1 {code} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    Responder { url, status, hits }
}

/// A URL whose port has nothing listening (bound, then released).
async fn refused_url() -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    format!("http://{addr}/health")
}

fn service(args: &[&str], restart: &str, health_url: Option<&str>) -> ServiceConfig {
    ServiceConfig {
        command: "sleep".into(),
        args: args.iter().map(|s| s.to_string()).collect(),
        env: HashMap::new(),
        restart: restart.into(),
        health_url: health_url.map(String::from),
    }
}

fn fast_options() -> SupervisorOptions {
    SupervisorOptions {
        probe_interval: Duration::from_millis(30),
        probe_timeout: Duration::from_millis(500),
        restart_base_secs: 0,
        restart_max_secs: 0,
        spawn_retry_delay: Duration::from_millis(50),
        shutdown_grace: Duration::from_secs(5),
    }
}

struct Running {
    sup: Arc<Supervisor>,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

fn start(services: Vec<(&str, ServiceConfig)>, options: SupervisorOptions) -> Running {
    let config = SupervisorConfig {
        services: services
            .into_iter()
            .map(|(n, c)| (n.to_string(), c))
            .collect(),
    };
    let sup = Arc::new(Supervisor::with_options(config, options));
    let (tx, rx) = oneshot::channel::<()>();
    let s = sup.clone();
    let task = tokio::spawn(async move {
        s.run_until(async {
            let _ = rx.await;
        })
        .await
        .expect("run_until");
    });
    Running {
        sup,
        stop: Some(tx),
        task,
    }
}

impl Running {
    async fn state(&self, name: &str) -> ManagedServiceState {
        tokio::time::timeout(LOCK_BOUND, self.sup.snapshot())
            .await
            .expect("state lock held too long: is a probe awaiting HTTP under the mutex?")[name]
            .clone()
    }

    /// Polls until `pred` holds; returns every state observed on the way (in order).
    async fn wait_for(
        &self,
        name: &str,
        pred: impl Fn(&ManagedServiceState) -> bool,
    ) -> Vec<ServiceState> {
        let deadline = Instant::now() + WAIT;
        let mut seen = Vec::new();
        loop {
            let st = self.state(name).await;
            if seen.last() != Some(&st.state) {
                seen.push(st.state);
            }
            if pred(&st) {
                return seen;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting on {name}; seen {seen:?}; last {st:?}"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    async fn shutdown(mut self) -> Arc<Supervisor> {
        let _ = self.stop.take().unwrap().send(());
        tokio::time::timeout(WAIT, &mut self.task)
            .await
            .expect("shutdown within bound")
            .unwrap();
        self.sup
    }
}

#[tokio::test]
async fn healthy_probe_reaches_ready() {
    let r = responder(200, Duration::ZERO).await;
    let run = start(
        vec![("svc", service(&["30"], "always", Some(&r.url)))],
        fast_options(),
    );
    let seen = run
        .wait_for("svc", |s| s.state == ServiceState::Ready)
        .await;
    let st = run.state("svc").await;
    assert_eq!(st.generation, 1);
    assert!(st.pid.is_some());
    assert!(r.hits.load(Ordering::SeqCst) >= 1);
    assert!(!seen.contains(&ServiceState::Unhealthy), "seen {seen:?}");
    run.shutdown().await;
}

#[tokio::test]
async fn no_health_url_is_ready_by_liveness() {
    let run = start(
        vec![("svc", service(&["30"], "always", None))],
        fast_options(),
    );
    run.wait_for("svc", |s| s.state == ServiceState::Ready)
        .await;
    run.shutdown().await;
}

#[tokio::test]
async fn http_500_is_unhealthy_never_ready() {
    let r = responder(500, Duration::ZERO).await;
    let run = start(
        vec![("svc", service(&["30"], "always", Some(&r.url)))],
        fast_options(),
    );
    let mut seen = run
        .wait_for("svc", |s| s.state == ServiceState::Unhealthy)
        .await;
    // Keep watching across several more probe rounds.
    let hits = r.hits.load(Ordering::SeqCst);
    seen.extend(
        run.wait_for("svc", |_| r.hits.load(Ordering::SeqCst) >= hits + 3)
            .await,
    );
    assert!(!seen.contains(&ServiceState::Ready), "seen {seen:?}");
    assert_eq!(run.state("svc").await.state, ServiceState::Unhealthy);
    run.shutdown().await;
}

#[tokio::test]
async fn connection_refused_is_unhealthy() {
    let url = refused_url().await;
    let run = start(
        vec![("svc", service(&["30"], "always", Some(&url)))],
        fast_options(),
    );
    let seen = run
        .wait_for("svc", |s| s.state == ServiceState::Unhealthy)
        .await;
    assert!(!seen.contains(&ServiceState::Ready), "seen {seen:?}");
    run.shutdown().await;
}

#[tokio::test]
async fn delayed_readiness_not_ready_until_responder_answers_200() {
    let r = responder(503, Duration::ZERO).await;
    let run = start(
        vec![("svc", service(&["30"], "always", Some(&r.url)))],
        fast_options(),
    );
    let before = run
        .wait_for("svc", |_| r.hits.load(Ordering::SeqCst) >= 3)
        .await;
    assert!(!before.contains(&ServiceState::Ready), "seen {before:?}");
    assert_ne!(run.state("svc").await.state, ServiceState::Ready);

    r.status.store(200, Ordering::SeqCst);
    run.wait_for("svc", |s| s.state == ServiceState::Ready)
        .await;
    assert_eq!(run.state("svc").await.generation, 1);
    run.shutdown().await;
}

#[tokio::test]
async fn spawned_process_not_ready_while_probe_in_flight_and_shutdown_mid_probe() {
    // Responder accepts but never answers; the probe stays in flight.
    let r = responder(0, Duration::ZERO).await;
    let mut opts = fast_options();
    opts.probe_timeout = Duration::from_secs(30);
    let run = start(
        vec![("svc", service(&["30"], "always", Some(&r.url)))],
        opts,
    );
    // The probe is now stuck in HTTP for up to 30 s. Reading state must still be quick;
    // if the lock were held across the HTTP await this fails at LOCK_BOUND, not at 30 s.
    let (seen, st) = tokio::time::timeout(LOCK_BOUND, async {
        let seen = run
            .wait_for("svc", |_| r.hits.load(Ordering::SeqCst) >= 1)
            .await;
        (seen, run.state("svc").await)
    })
    .await
    .expect("state readable while a probe is in flight");
    assert_eq!(st.state, ServiceState::Probing, "seen {seen:?}");
    assert!(!seen.contains(&ServiceState::Ready));
    let pid = st.pid.unwrap();

    let t0 = Instant::now();
    let sup = run.shutdown().await;
    assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
    let st = &sup.snapshot().await["svc"];
    assert_eq!(st.state, ServiceState::Stopped);
    assert_eq!(st.pid, None);
    assert!(!get_process_stats(pid).is_alive, "child {pid} still alive");
}

#[tokio::test]
async fn stale_probe_after_rapid_restart_never_marks_ready() {
    // Event-driven, no timing margin: the responder holds every 200 until the test has
    // seen the generation advance past the one that was alive when the request arrived.
    // Every answer therefore belongs to a child that already exited, so the only way to
    // report Ready would be to apply a stale result.
    let gate = Arc::new(Semaphore::new(0));
    let r = responder_with(200, Duration::ZERO, Some(gate.clone())).await;
    let mut opts = fast_options();
    opts.probe_timeout = Duration::from_secs(30);
    let run = start(
        vec![("svc", service(&["0.15"], "always", Some(&r.url)))],
        opts,
    );
    let mut seen = Vec::new();
    for k in 1..=3 {
        seen.extend(
            run.wait_for("svc", |_| r.hits.load(Ordering::SeqCst) >= k)
                .await,
        );
        let g = run.state("svc").await.generation;
        seen.extend(run.wait_for("svc", |s| s.generation > g).await);
        gate.add_permits(1);
    }
    // Rounds are serialized, so a 4th request means the 3rd answer was already handled.
    seen.extend(
        run.wait_for("svc", |_| r.hits.load(Ordering::SeqCst) >= 4)
            .await,
    );
    assert!(!seen.contains(&ServiceState::Ready), "seen {seen:?}");
    let st = run.state("svc").await;
    assert!(st.generation >= 4 && st.restart_count >= 3, "{st:?}");
    gate.add_permits(1000);
    run.shutdown().await;
}

#[tokio::test]
async fn probe_of_unreaped_zombie_is_process_gone() {
    // No health_url: liveness only. An exited child that is not yet reaped keeps its
    // /proc entry; it must not count as alive.
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let pid = child.id();
    let deadline = Instant::now() + WAIT;
    while std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| parse_stat_state(&s))
        != Some('Z')
    {
        assert!(Instant::now() < deadline, "child never became a zombie");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let ticket = ProbeTicket {
        name: "svc".into(),
        pid,
        generation: 1,
        health_url: None,
    };
    let report = probe_one(aien_localhttp::Client::new(), ticket).await;
    assert_eq!(report.outcome, ProbeOutcome::ProcessGone);
    child.wait().unwrap();
}

#[tokio::test]
async fn shutdown_stops_all_children_and_marks_stopped() {
    let r = responder(200, Duration::ZERO).await;
    let run = start(
        vec![
            ("web", service(&["30"], "always", Some(&r.url))),
            ("bare", service(&["30"], "always", None)),
        ],
        fast_options(),
    );
    run.wait_for("web", |s| s.state == ServiceState::Ready)
        .await;
    run.wait_for("bare", |s| s.state == ServiceState::Ready)
        .await;
    let pids: Vec<u32> = run
        .sup
        .snapshot()
        .await
        .values()
        .map(|s| s.pid.unwrap())
        .collect();

    let sup = run.shutdown().await;
    for st in sup.snapshot().await.values() {
        assert_eq!(st.state, ServiceState::Stopped, "{st:?}");
        assert_eq!(st.pid, None);
    }
    for pid in pids {
        assert!(!get_process_stats(pid).is_alive, "child {pid} still alive");
    }
}

#[tokio::test]
async fn exit_with_restart_no_is_stopped() {
    let run = start(
        vec![("once", service(&["0.05"], "no", None))],
        fast_options(),
    );
    run.wait_for("once", |s| s.generation == 1).await;
    run.wait_for("once", |s| s.state == ServiceState::Stopped)
        .await;
    let st = run.state("once").await;
    assert_eq!((st.generation, st.restart_count, st.pid), (1, 1, None));
    run.shutdown().await;
}

#[tokio::test]
async fn crashed_child_is_restarted_with_new_generation() {
    let run = start(
        vec![("flaky", service(&["0.05"], "always", None))],
        fast_options(),
    );
    run.wait_for("flaky", |s| s.generation >= 3).await;
    assert!(run.state("flaky").await.restart_count >= 2);
    run.shutdown().await;
}

#[tokio::test]
async fn https_health_url_refuses_to_start_before_spawning_or_connecting() {
    // An operator-configured https health_url is a loud config error: run_until
    // returns HttpsRefused before any child is spawned or any socket is opened.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("https://{addr}/health");
    let config = SupervisorConfig {
        services: [("tls".to_string(), service(&["30"], "always", Some(&url)))]
            .into_iter()
            .collect(),
    };
    let sup = Supervisor::with_options(config, fast_options());
    let err = tokio::time::timeout(WAIT, sup.run_until(std::future::pending()))
        .await
        .expect("run_until must fail at once, not run")
        .unwrap_err();
    assert!(
        matches!(err, aien_localhttp::Error::HttpsRefused { .. }),
        "{err:?}"
    );
    assert!(
        err.to_string().contains("service \"tls\" health_url"),
        "{err}"
    );
    let st = sup.snapshot().await["tls"].clone();
    assert_eq!(st.state, ServiceState::Stopped);
    assert!(st.pid.is_none());
    assert!(
        tokio::time::timeout(Duration::from_millis(200), listener.accept())
            .await
            .is_err(),
        "no connection may be made to an https health_url"
    );
}

#[tokio::test]
async fn probe_one_with_https_url_is_http_error_without_connecting() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let ticket = ProbeTicket {
        name: "svc".into(),
        pid: std::process::id(),
        generation: 1,
        health_url: Some(format!("https://{addr}/health")),
    };
    let report = probe_one(aien_localhttp::Client::new(), ticket).await;
    match report.outcome {
        ProbeOutcome::HttpError(msg) => assert!(msg.contains("https is refused"), "{msg}"),
        other => panic!("expected HttpError, got {other:?}"),
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(200), listener.accept())
            .await
            .is_err(),
        "probe_one must not connect to an https health_url"
    );
}
