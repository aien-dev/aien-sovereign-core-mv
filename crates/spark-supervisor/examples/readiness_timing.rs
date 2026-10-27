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

//! Measures time from supervisor start to each service being reported Ready (issue #534).
//!
//! Sample config: `bare` = `sleep 30`, no health_url; `web` = `sleep 30` with a health_url
//! served in-process that answers 503 until BOOT_MS after start, then 200 (a service that
//! needs BOOT_MS to come up). Prints one raw line per service per run.
//!
//! cargo run -p spark-supervisor --example readiness_timing -- [runs] [probe_interval_ms]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use spark_supervisor::config::{ServiceConfig, SupervisorConfig};
use spark_supervisor::supervisor::{ServiceState, Supervisor, SupervisorOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const BOOT_MS: u128 = 500;

async fn boot_delayed_responder(t0: Instant) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/health", listener.local_addr().unwrap());
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf).await;
                let code = if t0.elapsed().as_millis() >= BOOT_MS {
                    200
                } else {
                    503
                };
                let resp =
                    format!("HTTP/1.1 {code} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                let _ = sock.write_all(resp.as_bytes()).await;
            });
        }
    });
    url
}

fn sleeper(health_url: Option<String>) -> ServiceConfig {
    ServiceConfig {
        command: "sleep".into(),
        args: vec!["30".into()],
        env: HashMap::new(),
        restart: "always".into(),
        health_url,
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let runs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut options = SupervisorOptions::default();
    if let Some(ms) = args.get(2).and_then(|s| s.parse().ok()) {
        options.probe_interval = Duration::from_millis(ms);
    }
    println!(
        "branch run probe_interval_ms={} boot_ms={BOOT_MS}",
        options.probe_interval.as_millis()
    );

    for run in 1..=runs {
        let t0 = Instant::now();
        let url = boot_delayed_responder(t0).await;
        let mut services = HashMap::new();
        services.insert("bare".to_string(), sleeper(None));
        services.insert("web".to_string(), sleeper(Some(url)));
        let sup = Arc::new(Supervisor::with_options(
            SupervisorConfig { services },
            options.clone(),
        ));
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let s = sup.clone();
        let task = tokio::spawn(async move {
            s.run_until(async {
                let _ = rx.await;
            })
            .await
        });

        let mut spawned: HashMap<String, u128> = HashMap::new();
        let mut ready: HashMap<String, u128> = HashMap::new();
        while ready.len() < 2 && t0.elapsed() < Duration::from_secs(20) {
            for (name, st) in sup.snapshot().await {
                let now = t0.elapsed().as_millis();
                if st.pid.is_some() {
                    spawned.entry(name.clone()).or_insert(now);
                }
                if st.state == ServiceState::Ready {
                    ready.entry(name).or_insert(now);
                }
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let _ = tx.send(());
        let _ = task.await;
        for name in ["bare", "web"] {
            println!(
                "run={run} service={name} spawned_ms={:?} reported_ready_ms={:?}",
                spawned.get(name),
                ready.get(name)
            );
        }
    }
}
