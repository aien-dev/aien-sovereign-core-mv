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

//! SIGTERM stops the binary cleanly (sc#509): it stops accepting, releases the engine (which
//! drains the persistent KV parks on the omega door) and exits 0. Host only, stub engine.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn sigterm_shuts_down_cleanly_and_exits_zero() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_aien-serve-http"))
        .args([
            "--engine",
            "stub",
            "--port",
            "0",
            "--model-dir",
            "/nonexistent-pkv509",
        ])
        .env_remove("AIEN_KV_STORE_DIR")
        .env("AIEN_SERVE_HTTP_BEARER_TOKEN", "shutdown-test-private-credential")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut log = String::new();
    loop {
        let mut line = String::new();
        assert!(
            err.read_line(&mut line).unwrap() > 0,
            "exited before serving: {log}"
        );
        log.push_str(&line);
        if line.contains("serving ") {
            break;
        }
    }
    // SAFETY: plain kill(2) on our own child.
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    let t0 = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "no exit after SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut err, &mut rest).unwrap();
    log.push_str(&rest);
    assert!(status.success(), "{status:?}: {log}");
    assert!(
        log.contains("SIGTERM received") && log.contains("shutdown complete"),
        "{log}"
    );
}
