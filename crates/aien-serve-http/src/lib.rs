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

//! `aien-serve-http`: the OpenAI-compatible HTTP door for AIEN's own-kernel model
//! serving (OWN-30B cut 4). Loopback only, one generation at a time unless the engine has more slots (--batch-slots), no CUDA, no
//! Python. The engine behind the door is a trait ([`engine::Engine`]); this crate
//! ships a stub engine for tests and the CPU reference forward as a real-but-slow
//! engine. The Omega GPU engine plugs into the same trait.

pub mod chip_reservation;
pub mod codec;
pub mod cpu_engine;
pub mod engine;
pub mod kv_persist;
pub mod omega_engine;
pub mod openai;
pub mod qwen36_batch;
pub mod qwen38_engine;
pub mod readers;
pub mod sampling;
pub mod server;
pub mod text;
pub mod tools;

pub use codec::{ByteCodec, Codec, HfCodec};
pub use engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams, StubEngine};
pub use server::{serve, ServerConfig};

/// Largest small JSON file (`config.json`, `generation_config.json`) read from the model
/// directory; the same bound as the checkpoint loader's shard index.
pub const MODEL_JSON_LIMIT: u64 = 16 << 20;

/// Read a small text file from the model directory with at most `MODEL_JSON_LIMIT + 1`
/// bytes pulled. Refuses a non-regular file (a FIFO or device would block or never end) and
/// a file over the limit. A missing file is `ErrorKind::NotFound`, as with `fs::read_to_string`.
pub fn read_model_text(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::{Error, ErrorKind};
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(Error::new(ErrorKind::InvalidInput, "not a regular file"));
    }
    read_text_capped(file, MODEL_JSON_LIMIT)
}

fn read_text_capped(src: impl std::io::Read, limit: u64) -> std::io::Result<String> {
    use std::io::{Error, ErrorKind, Read};
    let mut text = String::new();
    src.take(limit + 1).read_to_string(&mut text)?;
    if text.len() as u64 > limit {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("file exceeds {limit} bytes"),
        ));
    }
    Ok(text)
}

#[cfg(test)]
mod model_text_tests {
    use super::*;
    use std::io::Read;

    /// An endless source that panics once more than `budget` bytes are pulled.
    struct Endless {
        served: u64,
        budget: u64,
    }

    impl Read for Endless {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            buf.fill(b' ');
            self.served += buf.len() as u64;
            assert!(
                self.served <= self.budget,
                "read past the cap: {} bytes pulled, budget {}",
                self.served,
                self.budget
            );
            Ok(buf.len())
        }
    }

    #[test]
    fn capped_read_stops_one_byte_past_the_limit() {
        let mut src = Endless {
            served: 0,
            budget: 4097,
        };
        let err = read_text_capped(&mut src, 4096).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(src.served, 4097);
    }

    #[test]
    fn model_text_reads_the_limit_and_refuses_more_and_non_regular_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        let mut body = vec![b' '; MODEL_JSON_LIMIT as usize - 2];
        body.splice(0..0, *b"{}");
        std::fs::write(&p, &body).unwrap();
        assert_eq!(read_model_text(&p).unwrap().len() as u64, MODEL_JSON_LIMIT);

        body.push(b' ');
        std::fs::write(&p, &body).unwrap();
        assert_eq!(
            read_model_text(&p).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        assert_eq!(
            read_model_text(&dir.path().join("absent.json"))
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::NotFound
        );
        // /dev/zero never ends; it is refused before any byte is read.
        assert_eq!(
            read_model_text(std::path::Path::new("/dev/zero"))
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
}
