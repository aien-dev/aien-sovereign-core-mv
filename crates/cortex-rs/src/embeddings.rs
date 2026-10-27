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

use aien_localhttp::Client;
use serde_json::json;
use std::time::Duration;

pub const EMBEDDING_DIMENSIONS: usize = 768;

pub fn embedding_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn bytes_to_embedding(b: &[u8]) -> Vec<f32> {
    let mut v = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        v.push(f32::from_le_bytes(arr));
    }
    v
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

#[allow(dead_code)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Checks that `encoder_url` is a plain `http://` URL the encoder client can
/// call. An `https://` URL is refused: the client never speaks TLS.
pub fn validate_encoder_url(encoder_url: &str) -> Result<(), String> {
    aien_localhttp::Url::parse(&format!("{}/embed", encoder_url.trim_end_matches("/")))
        .map(|_| ())
        .map_err(|e| format!("--encoder-url {encoder_url:?} is not usable: {e}"))
}

pub async fn fetch_embedding(
    client: &Client,
    text: &str,
    encoder_url: &str,
) -> Result<Vec<f32>, String> {
    let payload = json!({
        "text": text,
        "modelId": "BAAI/bge-base-en-v1.5",
        "modelRevision": "cortex-bge-base-en-v1.5-768-v1",
        "priority": "recall"
    });

    let resp = client
        .post(format!("{}/embed", encoder_url.trim_end_matches("/")))
        .timeout(Duration::from_millis(1500))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            format!(
                "Failed to connect to embedding encoder on {}: {}",
                encoder_url, e
            )
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Encoder returned error HTTP {}: {}", status, body));
    }

    let val: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse embedding response JSON: {}", e))?;

    if let Some(arr) = val.get("embedding").and_then(|v| v.as_array()) {
        let mut emb = Vec::with_capacity(arr.len());
        for num in arr {
            if let Some(f) = num.as_f64() {
                emb.push(f as f32);
            }
        }
        if emb.len() == EMBEDDING_DIMENSIONS {
            return Ok(emb);
        } else {
            return Err(format!(
                "Expected {} dimensions, got {}",
                EMBEDDING_DIMENSIONS,
                emb.len()
            ));
        }
    }

    Err("Invalid embedding payload structure from encoder service".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_bytes_roundtrip() {
        let orig = vec![0.123f32, -0.456, 0.789, 1.0];
        let bytes = embedding_to_bytes(&orig);
        assert_eq!(bytes.len(), 16);
        let recovered = bytes_to_embedding(&bytes);
        assert_eq!(orig, recovered);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        let c = vec![0.0f32, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance_and_empty_handling() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-5);

        let orthogonal = vec![0.0f32, 1.0, 0.0];
        assert!((cosine_distance(&a, &orthogonal) - 1.0).abs() < 1e-5);

        let opposite = vec![-1.0f32, 0.0, 0.0];
        assert!((cosine_distance(&a, &opposite) - 2.0).abs() < 1e-5);

        let empty: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&empty, &empty), 0.0);
        assert_eq!(cosine_distance(&empty, &empty), 1.0);

        let mismatched = vec![1.0f32, 0.0];
        assert_eq!(cosine_similarity(&a, &mismatched), 0.0);
        assert_eq!(cosine_distance(&a, &mismatched), 1.0);

        let zeros = vec![0.0f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&zeros, &zeros), 0.0);
        assert_eq!(cosine_distance(&zeros, &zeros), 1.0);
    }

    use aien_localhttp::test_server::{closed_port, response, serve_once, serve_silent};

    fn embedding_body(dims: usize) -> String {
        let values = vec!["0.5"; dims].join(",");
        format!("{{\"embedding\":[{values}]}}")
    }

    #[tokio::test]
    async fn fetch_embedding_posts_the_same_request_and_reads_768_dims() {
        let (base, server) =
            serve_once(response("200 OK", "application/json", &embedding_body(768))).await;
        // A trailing slash is trimmed before /embed, as before.
        let emb = fetch_embedding(&Client::new(), "hello", &format!("{base}/"))
            .await
            .unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIMENSIONS);
        assert_eq!(emb[0], 0.5);
        let request = server.await.unwrap();
        assert!(request.starts_with("POST /embed HTTP/1.1\r\n"), "{request}");
        assert!(request.contains("\r\nContent-Type: application/json\r\n"));
        let body: serde_json::Value =
            serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(
            body,
            json!({
                "text": "hello",
                "modelId": "BAAI/bge-base-en-v1.5",
                "modelRevision": "cortex-bge-base-en-v1.5-768-v1",
                "priority": "recall"
            })
        );
    }

    #[tokio::test]
    async fn fetch_embedding_error_status_keeps_the_old_message() {
        let (base, _server) =
            serve_once(response("503 Service Unavailable", "text/plain", "busy")).await;
        let error = fetch_embedding(&Client::new(), "x", &base)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            "Encoder returned error HTTP 503 Service Unavailable: busy"
        );
    }

    #[tokio::test]
    async fn fetch_embedding_rejects_wrong_dimensions_and_bad_json() {
        let (base, _server) =
            serve_once(response("200 OK", "application/json", &embedding_body(3))).await;
        let error = fetch_embedding(&Client::new(), "x", &base)
            .await
            .unwrap_err();
        assert_eq!(error, "Expected 768 dimensions, got 3");

        let (base, _server) = serve_once(response("200 OK", "application/json", "nope")).await;
        let error = fetch_embedding(&Client::new(), "x", &base)
            .await
            .unwrap_err();
        assert!(
            error.starts_with("Failed to parse embedding response JSON"),
            "{error}"
        );

        let (base, _server) =
            serve_once(response("200 OK", "application/json", "{\"other\":1}")).await;
        let error = fetch_embedding(&Client::new(), "x", &base)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            "Invalid embedding payload structure from encoder service"
        );
    }

    #[tokio::test]
    async fn fetch_embedding_reports_refused_and_timed_out_encoders() {
        let base = closed_port().await;
        let error = fetch_embedding(&Client::new(), "x", &base)
            .await
            .unwrap_err();
        assert!(
            error.starts_with(&format!("Failed to connect to embedding encoder on {base}")),
            "{error}"
        );

        // The 1500 ms per-request timeout still bounds a silent encoder.
        let (base, _server) = serve_silent().await;
        let started = std::time::Instant::now();
        let error = fetch_embedding(&Client::new(), "x", &base)
            .await
            .unwrap_err();
        assert!(error.contains("timed out"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn https_encoder_is_refused_at_validation_and_at_call_time() {
        assert!(validate_encoder_url("http://127.0.0.1:18081").is_ok());
        assert!(validate_encoder_url("http://127.0.0.1:18081/").is_ok());
        let error = validate_encoder_url("https://encoder.example").unwrap_err();
        assert!(error.contains("https is refused"), "{error}");
        assert!(validate_encoder_url("127.0.0.1:18081").is_err());

        let error = fetch_embedding(&Client::new(), "x", "https://127.0.0.1:1")
            .await
            .unwrap_err();
        assert!(error.contains("https is refused"), "{error}");
    }
}
