//! Prefix diff and the bounded session table behind ahead-of-Enter prefill (OWN-30B T2).
//!
//! Two pure pieces, no engine and no threads, so the rules can be tested exhaustively:
//!
//! * [`plan_prefill`]: given the tokens whose KV is resident and the tokens we want resident,
//!   what to keep, whether to rewind, and where new computation starts.
//! * [`SessionTable`]: which sessions the server tracks, with a maximum session count and a
//!   cap on the KV bytes they would occupy, evicting the least recently used first.

use std::collections::HashMap;

/// Length of the longest common token prefix.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// What a prefill of `target` has to do to an engine whose KV holds `resident`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    /// Tokens of `target` already resident (the longest common prefix).
    pub common: usize,
    /// `Some(n)`: the resident sequence diverged from `target` at `n` (an edit); the KV must
    /// be cut back to `n` before anything is appended. `None`: nothing to cut (the resident
    /// sequence is a prefix of `target`, or already covers it).
    pub rewind_to: Option<usize>,
    /// Index in `target` where new computation starts (equals `common`).
    pub extend_from: usize,
}

/// The one rule every engine applies. A resident sequence that is merely longer than
/// `target` (it also holds a reply, say) is left alone: the tokens that match stay valid and
/// the final request truncates by itself when it diverges.
pub fn plan_prefill(resident: &[u32], target: &[u32]) -> PrefillPlan {
    let common = common_prefix(resident, target);
    let diverged = common < resident.len() && common < target.len();
    PrefillPlan {
        common,
        rewind_to: if diverged { Some(common) } else { None },
        extend_from: common,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionLimits {
    pub max_sessions: usize,
    /// Cap on the KV bytes the tracked sessions represent (`tokens * bytes_per_token`).
    pub max_kv_bytes: u64,
    /// KV bytes one token occupies (Qwen3-30B-A3B bf16: 98304).
    pub bytes_per_token: u64,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_sessions: 16,
            max_kv_bytes: 4 << 30,
            bytes_per_token: 98_304,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEntry {
    /// The tokens this session last asked to have resident.
    pub tokens: Vec<u32>,
    /// How many of them a prefill has actually computed or found resident.
    pub prefilled: usize,
    /// Last failure from a background prefill, if any.
    pub error: Option<String>,
    tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// One session alone would exceed the byte cap.
    TooLarge { bytes: u64, cap: u64 },
}

#[derive(Debug)]
pub struct SessionTable {
    limits: SessionLimits,
    map: HashMap<String, SessionEntry>,
    tick: u64,
}

impl SessionTable {
    pub fn new(limits: SessionLimits) -> Self {
        Self { limits, map: HashMap::new(), tick: 0 }
    }

    pub fn limits(&self) -> &SessionLimits {
        &self.limits
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn total_bytes(&self) -> u64 {
        self.map.values().map(|e| e.tokens.len() as u64 * self.limits.bytes_per_token).sum()
    }

    pub fn get(&self, session: &str) -> Option<&SessionEntry> {
        self.map.get(session)
    }

    pub fn get_mut(&mut self, session: &str) -> Option<&mut SessionEntry> {
        self.map.get_mut(session)
    }

    /// Names, most recently used first.
    pub fn names_by_recency(&self) -> Vec<String> {
        let mut v: Vec<(&String, u64)> = self.map.iter().map(|(k, e)| (k, e.tick)).collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        v.into_iter().map(|(k, _)| k.clone()).collect()
    }

    /// Records `tokens` as the session's wanted sequence (most recently used) and evicts the
    /// least recently used sessions until both limits hold. Returns the evicted names.
    pub fn put(&mut self, session: &str, tokens: Vec<u32>, prefilled: usize) -> Result<Vec<String>, SessionError> {
        let bytes = tokens.len() as u64 * self.limits.bytes_per_token;
        if bytes > self.limits.max_kv_bytes {
            return Err(SessionError::TooLarge { bytes, cap: self.limits.max_kv_bytes });
        }
        self.tick += 1;
        self.map.insert(
            session.to_string(),
            SessionEntry { tokens, prefilled, error: None, tick: self.tick },
        );
        let mut evicted = Vec::new();
        while self.map.len() > self.limits.max_sessions.max(1) || self.total_bytes() > self.limits.max_kv_bytes {
            let victim = self
                .map
                .iter()
                .filter(|(k, _)| k.as_str() != session)
                .min_by_key(|(_, e)| e.tick)
                .map(|(k, _)| k.clone());
            match victim {
                Some(v) => {
                    self.map.remove(&v);
                    evicted.push(v);
                }
                None => break,
            }
        }
        Ok(evicted)
    }

    /// Marks a session as just used without changing its tokens.
    pub fn touch(&mut self, session: &str) {
        self.tick += 1;
        let t = self.tick;
        if let Some(e) = self.map.get_mut(session) {
            e.tick = t;
        }
    }

    pub fn remove(&mut self, session: &str) -> bool {
        self.map.remove(session).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_covers_the_four_cases() {
        // Empty resident: compute everything.
        assert_eq!(plan_prefill(&[], &[1, 2, 3]), PrefillPlan { common: 0, rewind_to: None, extend_from: 0 });
        // Pure append: keep all, compute the suffix.
        assert_eq!(plan_prefill(&[1, 2], &[1, 2, 3, 4]), PrefillPlan { common: 2, rewind_to: None, extend_from: 2 });
        // Edit in the middle: rewind to the divergence point.
        assert_eq!(plan_prefill(&[1, 2, 3, 4], &[1, 2, 9, 9, 9]), PrefillPlan { common: 2, rewind_to: Some(2), extend_from: 2 });
        // Target is a prefix of a longer resident sequence: nothing to cut, nothing to compute.
        assert_eq!(plan_prefill(&[1, 2, 3, 4], &[1, 2]), PrefillPlan { common: 2, rewind_to: None, extend_from: 2 });
        // Total mismatch: rewind to zero.
        assert_eq!(plan_prefill(&[5], &[1]), PrefillPlan { common: 0, rewind_to: Some(0), extend_from: 0 });
    }

    fn limits(n: usize, bytes: u64) -> SessionLimits {
        SessionLimits { max_sessions: n, max_kv_bytes: bytes, bytes_per_token: 1 }
    }

    #[test]
    fn lru_evicts_least_recently_used_session() {
        let mut t = SessionTable::new(limits(2, 1000));
        t.put("a", vec![1], 0).unwrap();
        t.put("b", vec![2], 0).unwrap();
        t.touch("a");
        let ev = t.put("c", vec![3], 0).unwrap();
        assert_eq!(ev, vec!["b".to_string()]);
        assert!(t.get("a").is_some() && t.get("c").is_some() && t.get("b").is_none());
        assert_eq!(t.names_by_recency(), vec!["c".to_string(), "a".to_string()]);
    }

    #[test]
    fn byte_cap_evicts_oldest_and_refuses_an_oversized_session() {
        let mut t = SessionTable::new(limits(10, 10));
        t.put("a", vec![0; 4], 0).unwrap();
        t.put("b", vec![0; 4], 0).unwrap();
        let ev = t.put("c", vec![0; 4], 0).unwrap();
        assert_eq!(ev, vec!["a".to_string()]);
        assert!(t.total_bytes() <= 10);
        assert_eq!(t.put("big", vec![0; 11], 0), Err(SessionError::TooLarge { bytes: 11, cap: 10 }));
        assert!(t.get("big").is_none() && t.get("c").is_some(), "a refused put evicts nothing");
    }

    #[test]
    fn replacing_a_session_does_not_count_twice() {
        let mut t = SessionTable::new(limits(10, 10));
        t.put("a", vec![0; 8], 0).unwrap();
        let ev = t.put("a", vec![0; 9], 0).unwrap();
        assert!(ev.is_empty());
        assert_eq!(t.total_bytes(), 9);
    }
}
