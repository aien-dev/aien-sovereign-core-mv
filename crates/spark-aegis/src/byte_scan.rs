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

//! Byte and pattern search over file contents (pure Rust).

/// Index of the first `target` byte in `data`.
pub fn scan_byte(data: &[u8], target: u8) -> Option<usize> {
    data.iter().position(|&b| b == target)
}

/// Number of `target` bytes in `data`.
pub fn count_byte_matches(data: &[u8], target: u8) -> usize {
    data.iter().filter(|&&b| b == target).count()
}

/// Index of the first occurrence of `pattern` in `data`; an empty pattern matches at 0.
pub fn find_pattern(data: &[u8], pattern: &[u8]) -> Option<usize> {
    if pattern.is_empty() {
        return Some(0);
    }
    data.windows(pattern.len())
        .position(|window| window == pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_scan_and_pattern_search() {
        let buffer = b"SOVEREIGN_SYSTEMS_SPARK_AEGIS_SCAN";
        let target = b'P';
        let found = scan_byte(buffer, target);
        assert_eq!(found, Some(19));

        let not_found = scan_byte(buffer, b'Z');
        assert_eq!(not_found, None);

        let pattern = b"SPARK_AEGIS";
        let pat_idx = find_pattern(buffer, pattern);
        assert_eq!(pat_idx, Some(18));

        let missing_pat = b"NONEXISTENT";
        assert_eq!(find_pattern(buffer, missing_pat), None);

        let count_s = count_byte_matches(buffer, b'S');
        assert_eq!(count_s, 7);
    }

    #[test]
    fn test_pattern_longer_than_data() {
        assert_eq!(find_pattern(b"AB", b"ABC"), None);
        assert_eq!(find_pattern(b"AB", b""), Some(0));
    }
}
