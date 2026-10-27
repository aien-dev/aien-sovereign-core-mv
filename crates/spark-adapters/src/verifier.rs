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

use crate::models::{VerificationOutcome, VerificationStrategy};
use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};

pub struct Verifier;

impl Verifier {
    pub fn verify(content: &str, strategy: VerificationStrategy) -> VerificationOutcome {
        if content.trim().is_empty() {
            return VerificationOutcome {
                passed: false,
                score: 0.0,
                rule_violations: vec!["Empty completion received".to_string()],
                compiler_output: None,
            };
        }

        let mut violations = Vec::new();
        let mut score: f32 = 1.0;

        // 1. Unslop and Tone Invariants
        if content.contains('\u{2014}') {
            violations.push("Contains forbidden em dash (\\u{2014})".to_string());
            score -= 0.25;
        }
        if content.contains('\u{2013}') {
            violations.push("Contains forbidden en dash (\\u{2013})".to_string());
            score -= 0.15;
        }

        let banned_words = [
            "delve",
            "tapestry",
            "crucial",
            "beacon",
            "game-changer",
            "unleash",
            "seamlessly",
            "elevate",
            "pivotal",
        ];
        let lower = content.to_lowercase();
        for word in banned_words {
            if lower.contains(word) {
                violations.push(format!("Contains banned AI cliché: '{}'", word));
                score -= 0.15;
            }
        }

        let sycophancy_phrases = [
            "certainly!",
            "great question!",
            "sure thing!",
            "i would be happy to help",
            "as an ai language model",
        ];
        for phrase in sycophancy_phrases {
            if lower.contains(phrase) {
                violations.push(format!("Contains sycophantic phrase: '{}'", phrase));
                score -= 0.2;
            }
        }

        // 2. Zero-Secret Leak Invariant
        let secret_signatures = [
            "-----BEGIN PRIVATE KEY-----",
            "-----BEGIN OPENSSH PRIVATE KEY-----",
            "sk-proj-",
            "sk-ant-",
            "ghp_",
            "gho_",
        ];
        for sig in secret_signatures {
            if content.contains(sig) {
                violations.push(format!("Critical: Contains secret signature: {}", sig));
                score = 0.0;
            }
        }

        // 3. Strategy-specific validation
        let mut compiler_output = None;
        match strategy {
            VerificationStrategy::CompilerCheck => {
                let code_blocks = Self::extract_code_blocks(content);
                let rust_code = code_blocks
                    .into_iter()
                    .find(|b| b.contains("fn ") || b.contains("struct ") || b.contains("impl "));

                if let Some(code) = rust_code {
                    match Self::verify_with_rustc(&code) {
                        Ok(msg) => {
                            compiler_output = Some(msg);
                        }
                        Err(err) => {
                            violations.push(format!("Compiler error: {}", err));
                            score -= 0.35;
                            compiler_output = Some(err);
                        }
                    }
                } else if let Some(err) = Self::check_balanced_delimiters(content) {
                    violations.push(format!("Code structure syntax error: {}", err));
                    score -= 0.3;
                    compiler_output = Some(err);
                } else {
                    compiler_output =
                        Some("Balanced delimiters and structure syntax verified.".to_string());
                }
            }
            VerificationStrategy::JsonSchema => {
                if let Some(json_start) = content.find('{') {
                    if let Some(json_end) = content.rfind('}') {
                        let slice = &content[json_start..=json_end];
                        if serde_json::from_str::<serde_json::Value>(slice).is_err() {
                            violations.push("Embedded JSON payload is malformed".to_string());
                            score -= 0.3;
                        }
                    } else {
                        violations.push("Incomplete JSON object brackets".to_string());
                        score -= 0.3;
                    }
                }
            }
            VerificationStrategy::UnslopStrict => {
                if !violations.is_empty() {
                    score = score.min(0.5);
                }
            }
            VerificationStrategy::DualConsensus => {
                // Handled during multi-candidate evaluation
            }
        }

        score = score.clamp(0.0, 1.0);
        let passed = violations.is_empty()
            || (score >= 0.70 && !violations.iter().any(|v| v.starts_with("Critical")));

        VerificationOutcome {
            passed,
            score,
            rule_violations: violations,
            compiler_output,
        }
    }

    /// Extract code blocks delimited by triple backticks.
    pub fn extract_code_blocks(content: &str) -> Vec<String> {
        let mut blocks = Vec::new();
        let mut in_block = false;
        let mut current_block = String::new();

        for line in content.lines() {
            if line.trim_start().starts_with("```") {
                if in_block {
                    blocks.push(current_block.clone());
                    current_block.clear();
                    in_block = false;
                } else {
                    in_block = true;
                }
            } else if in_block {
                current_block.push_str(line);
                current_block.push('\n');
            }
        }

        if blocks.is_empty() && (content.contains("fn ") || content.contains("struct ")) {
            blocks.push(content.to_string());
        }

        blocks
    }

    /// Verify a code snippet by invoking `rustc --crate-type lib --emit=metadata -`.
    pub fn verify_with_rustc(code: &str) -> Result<String, String> {
        let temp_dir = std::env::temp_dir();
        let mut child = Command::new("rustc")
            .args([
                "--crate-type",
                "lib",
                "--emit=metadata",
                "--crate-name",
                "spark_verifier_eval",
                "--out-dir",
                temp_dir.to_str().unwrap_or("/tmp"),
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn rustc: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(code.as_bytes());
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for rustc: {}", e))?;

        if output.status.success() {
            Ok("Sandboxed rustc compilation check verified.".to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let first_error = stderr
                .lines()
                .find(|l| l.contains("error[E") || l.starts_with("error:"))
                .unwrap_or("syntax compilation error")
                .to_string();
            Err(first_error)
        }
    }

    /// Compute Word-level Token Jaccard Similarity between two completions.
    pub fn token_jaccard(a: &str, b: &str) -> f32 {
        let words_a: HashSet<&str> = a.split_whitespace().collect();
        let words_b: HashSet<&str> = b.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Compute Normalized Levenshtein similarity (1.0 = identical, 0.0 = completely different).
    pub fn normalized_levenshtein(a: &str, b: &str) -> f32 {
        let len_a = a.chars().count();
        let len_b = b.chars().count();

        if len_a == 0 && len_b == 0 {
            return 1.0;
        }
        let max_len = len_a.max(len_b);
        if max_len == 0 {
            return 1.0;
        }

        let distance = Self::levenshtein_distance(a, b);
        1.0 - (distance as f32 / max_len as f32)
    }

    fn levenshtein_distance(a: &str, b: &str) -> usize {
        let b_chars: Vec<char> = b.chars().collect();
        let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
        let mut curr = vec![0; b_chars.len() + 1];

        for (i, ca) in a.chars().enumerate() {
            curr[0] = i + 1;
            for (j, &cb) in b_chars.iter().enumerate() {
                let cost = if ca == cb { 0 } else { 1 };
                curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        prev[b_chars.len()]
    }

    /// Compute Normalized Longest Common Subsequence similarity.
    pub fn normalized_lcs(a: &str, b: &str) -> f32 {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();

        if a_chars.is_empty() && b_chars.is_empty() {
            return 1.0;
        }
        let max_len = a_chars.len().max(b_chars.len());
        if max_len == 0 {
            return 1.0;
        }

        let mut prev = vec![0; b_chars.len() + 1];
        let mut curr = vec![0; b_chars.len() + 1];

        for ca in &a_chars {
            for (j, cb) in b_chars.iter().enumerate() {
                if ca == cb {
                    curr[j + 1] = prev[j] + 1;
                } else {
                    curr[j + 1] = prev[j + 1].max(curr[j]);
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        prev[b_chars.len()] as f32 / max_len as f32
    }

    /// Compute Hybrid Consensus Score:
    /// 40% Normalized Levenshtein + 30% Normalized LCS + 30% Token Jaccard.
    pub fn hybrid_consensus(a: &str, b: &str) -> f32 {
        let lev = Self::normalized_levenshtein(a, b);
        let lcs = Self::normalized_lcs(a, b);
        let jac = Self::token_jaccard(a, b);

        (0.40 * lev) + (0.30 * lcs) + (0.30 * jac)
    }

    fn check_balanced_delimiters(content: &str) -> Option<String> {
        let mut stack = Vec::new();
        let mut in_string = false;
        let mut escape = false;

        for (i, c) in content.chars().enumerate() {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
                escape = true;
                continue;
            }
            if c == '"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }

            match c {
                '(' | '[' | '{' => stack.push((c, i)),
                ')' => {
                    if let Some(('(', _)) = stack.last() {
                        stack.pop();
                    } else {
                        return Some(format!("Unmatched ')' at position {}", i));
                    }
                }
                ']' => {
                    if let Some(('[', _)) = stack.last() {
                        stack.pop();
                    } else {
                        return Some(format!("Unmatched ']' at position {}", i));
                    }
                }
                '}' => {
                    if let Some(('{', _)) = stack.last() {
                        stack.pop();
                    } else {
                        return Some(format!("Unmatched '}}' at position {}", i));
                    }
                }
                _ => {}
            }
        }

        if let Some((unmatched, pos)) = stack.pop() {
            Some(format!(
                "Unclosed delimiter '{}' opened at position {}",
                unmatched, pos
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_response_passes() {
        let text = "Here is the pure Rust ring buffer implementation with zero allocations.";
        let out = Verifier::verify(text, VerificationStrategy::UnslopStrict);
        assert!(out.passed);
        assert_eq!(out.score, 1.0);
        assert!(out.rule_violations.is_empty());
    }

    #[test]
    fn test_em_dash_detected() {
        let text = "This module is fast \u{2014} and very reliable.";
        let out = Verifier::verify(text, VerificationStrategy::UnslopStrict);
        assert!(!out.passed);
        assert!(out.rule_violations.iter().any(|v| v.contains("em dash")));
    }

    #[test]
    fn test_banned_cliche_detected() {
        let text = "Let us delve into the architecture.";
        let out = Verifier::verify(text, VerificationStrategy::UnslopStrict);
        assert!(out.rule_violations.iter().any(|v| v.contains("delve")));
    }

    #[test]
    fn test_rustc_compilation_verification() {
        let valid_code = "```rust\npub fn add(a: i32, b: i32) -> i32 { a + b }\n```";
        let out = Verifier::verify(valid_code, VerificationStrategy::CompilerCheck);
        assert!(out.passed);
        assert_eq!(out.score, 1.0);

        let invalid_code = "```rust\npub fn add(a: i32, b: i32) -> i32 { a + \"broken\" }\n```";
        let out_inv = Verifier::verify(invalid_code, VerificationStrategy::CompilerCheck);
        assert!(!out_inv.passed);
        assert!(out_inv
            .rule_violations
            .iter()
            .any(|v| v.contains("Compiler error")));
    }

    #[test]
    fn test_hybrid_consensus_scoring() {
        let a = "fn reverse(s: &mut [i32]) { s.reverse(); }";
        let b = "fn reverse(s: &mut [i32]) { s.reverse(); }";
        let c = "pub fn sort(s: &mut [i32]) { s.sort(); }";

        assert_eq!(Verifier::hybrid_consensus(a, b), 1.0);
        let sim = Verifier::hybrid_consensus(a, c);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_empty_completion_rejected() {
        let empty_out = Verifier::verify("", VerificationStrategy::CompilerCheck);
        assert!(!empty_out.passed);
        assert_eq!(empty_out.score, 0.0);
        assert!(empty_out
            .rule_violations
            .iter()
            .any(|v| v.contains("Empty completion")));

        let whitespace_out = Verifier::verify(
            "   
  	 ",
            VerificationStrategy::UnslopStrict,
        );
        assert!(!whitespace_out.passed);
        assert_eq!(whitespace_out.score, 0.0);
    }
}
