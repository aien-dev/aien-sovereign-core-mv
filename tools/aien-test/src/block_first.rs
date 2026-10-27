//! Desk blocks go first (L309, 2026-10-20).
//!
//! The PR desk (Renda) carries many PRs in one "desk block" PR and records it
//! in `<state>/renda-block-<repo>.json`:
//! `{"open": {"sha": HEAD, "carried": [PR..], "heads": {"PR": SHA}, ..}, ..}`.
//! One judge run on the block head stands for every carried PR, so while a
//! block is open the worker judges its head before any other head of that
//! repository and does not judge a carried PR's own head by itself.
//!
//! A state file older than `MAX_AGE` is ignored, so a desk that stopped
//! cannot keep PRs from being judged forever. An explicit trigger file is an
//! operator's order and is never skipped.

use std::collections::{BTreeSet, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// A queued head: repository key, commit, and the trigger file to remove
/// once it has been served.
pub type Head = (String, String, Option<PathBuf>);

/// Longest a block state file is believed without being rewritten.
pub const MAX_AGE: Duration = Duration::from_secs(24 * 3600);

/// One open desk block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub repo: String,
    pub head: String,
    /// Heads of the PRs the block carries.
    pub carried: BTreeSet<String>,
}

/// Parse one `renda-block-<repo>.json`; None when no block is open or the
/// text is not what the desk writes.
pub fn parse(repo: &str, text: &str) -> Option<Block> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let open = v.get("open")?;
    let head = open.get("sha")?.as_str()?.to_ascii_lowercase();
    if !crate::worker::valid_sha(&head) {
        return None;
    }
    let numbers: BTreeSet<u64> = open
        .get("carried")
        .and_then(|c| c.as_array())
        .map(|a| a.iter().filter_map(|n| n.as_u64()).collect())
        .unwrap_or_default();
    let heads = open.get("heads").and_then(|h| h.as_object());
    let carried = numbers
        .iter()
        .filter_map(|n| heads?.get(&n.to_string())?.as_str())
        .map(str::to_ascii_lowercase)
        .filter(|s| crate::worker::valid_sha(s))
        .collect();
    Some(Block {
        repo: repo.to_string(),
        head,
        carried,
    })
}

/// The open blocks of `repos`, read from `state_root`. A file not rewritten
/// within `MAX_AGE` of `now` is ignored.
pub fn open_blocks(state_root: &Path, repos: &[String], now: SystemTime) -> Vec<Block> {
    let mut out = Vec::new();
    for repo in repos {
        if !crate::worker::valid_repo_key(repo) {
            continue;
        }
        let path = state_root.join(format!("renda-block-{repo}.json"));
        let fresh = fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok().or(Some(Duration::ZERO)))
            .is_some_and(|age| age <= MAX_AGE);
        if !fresh {
            continue;
        }
        if let Some(b) = fs::read_to_string(&path).ok().and_then(|t| parse(repo, &t)) {
            out.push(b);
        }
    }
    out
}

/// The next head to judge. A block head not yet `tried` and without a bundle
/// comes first (taken from the queue when it is there, otherwise made, with
/// its trigger file when one exists). Otherwise the front of the queue, except
/// that a head with no trigger file which a block carries is dropped.
pub fn next_head(
    queue: &mut VecDeque<Head>,
    blocks: &[Block],
    trigger_dir: &Path,
    bundled: &dyn Fn(&str, &str) -> bool,
    tried: &mut HashSet<(String, String)>,
) -> Option<Head> {
    for b in blocks {
        let key = (b.repo.clone(), b.head.clone());
        if tried.contains(&key) || bundled(&b.repo, &b.head) {
            continue;
        }
        tried.insert(key);
        if let Some(i) = queue
            .iter()
            .position(|(r, s, _)| *r == b.repo && *s == b.head)
        {
            return queue.remove(i);
        }
        let trigger = trigger_dir.join(format!("{}.{}", b.repo, b.head));
        return Some((
            b.repo.clone(),
            b.head.clone(),
            trigger.exists().then_some(trigger),
        ));
    }
    while let Some((repo, sha, trigger)) = queue.pop_front() {
        if !tried.insert((repo.clone(), sha.clone())) {
            // Served already this round; a leftover trigger file goes with it.
            if let Some(f) = trigger {
                let _ = fs::remove_file(f);
            }
            continue;
        }
        let carried = trigger.is_none()
            && blocks
                .iter()
                .any(|b| b.repo == repo && b.carried.contains(&sha));
        if carried {
            tried.remove(&(repo, sha));
            continue;
        }
        return Some((repo, sha, trigger));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(c: char) -> String {
        c.to_string().repeat(40)
    }

    fn state(name: &str, text: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("block-first-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("trigger")).unwrap();
        fs::write(d.join("renda-block-r.json"), text).unwrap();
        d
    }

    fn block_text(head: &str) -> String {
        format!(
            r#"{{"open": {{"pr": 9, "sha": "{head}", "carried": [1, 2],
              "heads": {{"1": "{}", "2": "{}", "3": "{}"}}}}, "size": 12}}"#,
            sha('1'),
            sha('2'),
            sha('3')
        )
    }

    fn drain(
        q: Vec<Head>,
        blocks: &[Block],
        dir: &Path,
        bundled: &dyn Fn(&str, &str) -> bool,
    ) -> Vec<String> {
        let mut q: VecDeque<Head> = q.into();
        let mut tried = HashSet::new();
        let mut out = Vec::new();
        while let Some((_, s, _)) = next_head(&mut q, blocks, dir, bundled, &mut tried) {
            out.push(s);
        }
        out
    }

    fn head(c: char) -> Head {
        ("r".to_string(), sha(c), None)
    }

    #[test]
    fn block_head_goes_first_and_carried_heads_are_skipped() {
        let d = state("order", &block_text(&sha('b')));
        let blocks = open_blocks(&d, &["r".to_string()], SystemTime::now());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].carried.len(), 2);
        // queue sorted by sha as GitHub polling leaves it: 1, 2, 3, b
        let q = vec![head('1'), head('2'), head('3'), head('b')];
        let got = drain(q, &blocks, &d.join("trigger"), &|_, _| false);
        assert_eq!(got, vec![sha('b'), sha('3')]);
    }

    #[test]
    fn block_head_missing_from_the_queue_is_made_first() {
        let d = state("missing", &block_text(&sha('b')));
        let blocks = open_blocks(&d, &["r".to_string()], SystemTime::now());
        let got = drain(vec![head('3')], &blocks, &d.join("trigger"), &|_, _| false);
        assert_eq!(got, vec![sha('b'), sha('3')]);
    }

    #[test]
    fn judged_block_head_is_not_queued_again_and_carried_stay_skipped() {
        let d = state("done", &block_text(&sha('b')));
        let blocks = open_blocks(&d, &["r".to_string()], SystemTime::now());
        let b = sha('b');
        let got = drain(
            vec![head('1'), head('3')],
            &blocks,
            &d.join("trigger"),
            &|_, s| s == b,
        );
        assert_eq!(got, vec![sha('3')]);
    }

    #[test]
    fn a_trigger_file_for_a_carried_head_is_still_served() {
        let d = state("trigger", &block_text(&sha('b')));
        let blocks = open_blocks(&d, &["r".to_string()], SystemTime::now());
        let t = d.join("trigger").join(format!("r.{}", sha('1')));
        fs::write(&t, "").unwrap();
        let q = vec![("r".to_string(), sha('1'), Some(t))];
        let got = drain(q, &blocks, &d.join("trigger"), &|_, _| false);
        assert_eq!(got, vec![sha('b'), sha('1')]);
    }

    #[test]
    fn block_head_trigger_file_made_mid_round_travels_with_the_head() {
        let d = state("mid", &block_text(&sha('b')));
        let blocks = open_blocks(&d, &["r".to_string()], SystemTime::now());
        let t = d.join("trigger").join(format!("r.{}", sha('b')));
        fs::write(&t, "").unwrap();
        let mut q: VecDeque<Head> = VecDeque::new();
        let mut tried = HashSet::new();
        let got = next_head(
            &mut q,
            &blocks,
            &d.join("trigger"),
            &|_, _| false,
            &mut tried,
        );
        assert_eq!(got, Some(("r".to_string(), sha('b'), Some(t))));
    }

    #[test]
    fn no_open_block_leaves_the_order_alone() {
        let d = state("none", r#"{"open": null, "size": 12}"#);
        let blocks = open_blocks(&d, &["r".to_string()], SystemTime::now());
        assert!(blocks.is_empty());
        let got = drain(
            vec![head('1'), head('b')],
            &blocks,
            &d.join("trigger"),
            &|_, _| false,
        );
        assert_eq!(got, vec![sha('1'), sha('b')]);
    }

    #[test]
    fn a_stale_state_file_and_other_repos_are_ignored() {
        let d = state("stale", &block_text(&sha('b')));
        let later = SystemTime::now() + MAX_AGE + Duration::from_secs(60);
        assert!(open_blocks(&d, &["r".to_string()], later).is_empty());
        assert!(open_blocks(&d, &["other".to_string()], SystemTime::now()).is_empty());
    }

    #[test]
    fn unreadable_state_is_no_block() {
        assert_eq!(parse("r", "not json"), None);
        assert_eq!(parse("r", r#"{"open": {"sha": "short"}}"#), None);
    }
}
