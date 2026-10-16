//! Delivery logic: identity, tags, batch selection, confirmation and acks.
//! Generic over [`Board`] so the same code runs against the in-memory fake
//! and against synapse-coord's real board.

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use crate::board::{Board, BridgeError, Event, Kind, Post, Ref, Sub};
use crate::render::{self, BatchView};
use crate::state::{self, Pending, Scan, State, StateFile};

/// Largest hook stdin accepted (Claude Code sends tool output in it).
pub const MAX_HOOK_INPUT: usize = 4 * 1024 * 1024;
/// Events fetched per selection (the batch is cut from these).
pub const SCAN_LIMIT: usize = 64;
/// Board text and reference limits (PLAN D4).
pub const MAX_TEXT: usize = 4096;
pub const MAX_REFS: usize = 16;
pub const MAX_NAME: usize = 128;
/// Suffixes tried when a tag is taken.
pub const TAG_SUFFIXES: u32 = 9;

/// The parts of Claude Code's hook stdin the bridge uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookInput {
    pub session_id: String,
    pub agent_id: Option<String>,
    pub agent_type: Option<String>,
    pub event: String,
    pub transcript_path: Option<PathBuf>,
}

pub fn parse_hook_input(bytes: &[u8]) -> Result<HookInput, BridgeError> {
    if bytes.len() > MAX_HOOK_INPUT {
        return Err(BridgeError::BadHookInput(format!(
            "{} bytes, cap {MAX_HOOK_INPUT}",
            bytes.len()
        )));
    }
    let v: Value =
        serde_json::from_slice(bytes).map_err(|e| BridgeError::BadHookInput(e.to_string()))?;
    let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let session_id = s("session_id").ok_or(BridgeError::NoSession)?;
    let event = s("hook_event_name")
        .ok_or_else(|| BridgeError::BadHookInput("no hook_event_name".into()))?;
    Ok(HookInput {
        session_id,
        agent_id: s("agent_id").filter(|a| !a.is_empty()),
        agent_type: s("agent_type"),
        event,
        transcript_path: s("transcript_path")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from),
    })
}

/// Names (session keys, tags, threads, topics) are plain path-safe strings.
pub fn check_name(what: &str, s: &str, allow_slash: bool) -> Result<(), BridgeError> {
    let ok_char = |c: char| {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') || (allow_slash && c == '/')
    };
    if s.is_empty()
        || s.len() > MAX_NAME
        || !s.chars().all(ok_char)
        || s.starts_with('.')
        || s.split('/').any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(BridgeError::BadName(format!("{what} {s:?}")));
    }
    Ok(())
}

/// Topics may also be a wildcard: `*` or `<prefix>/*`.
pub fn check_topic(s: &str) -> Result<(), BridgeError> {
    if s == "*" {
        return Ok(());
    }
    check_name("topic", s.strip_suffix("/*").unwrap_or(s), true)
}

pub fn check_sub(sub: &Sub) -> Result<(), BridgeError> {
    match sub {
        Sub::Thread(t) => check_name("thread", t, true),
        Sub::Topic(t) => check_topic(t),
    }
}

/// Session key: the Claude Code session id, plus `.<agent_id>` inside an
/// Agent-tool subagent (subagents share the parent's session id).
pub fn session_key(session_id: &str, agent_id: Option<&str>) -> Result<String, BridgeError> {
    let k = match agent_id {
        Some(a) => format!("{session_id}.{a}"),
        None => session_id.to_string(),
    };
    check_name("session", &k, false)?;
    Ok(k)
}

fn short(s: &str, n: usize) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(n)
        .collect::<String>()
        .to_ascii_lowercase()
}

/// Tag the session asks for: `SWARM_TAG`, else `agent-<8 chars of session>`;
/// a subagent appends `-<6 chars of agent id>` so it never collides with its
/// parent.
pub fn wanted_tag(swarm_tag: Option<&str>, session_id: &str, agent_id: Option<&str>) -> String {
    let base = match swarm_tag.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => t.to_string(),
        None => format!("agent-{}", short(session_id, 8)),
    };
    match agent_id {
        Some(a) => format!("{base}-{}", short(a, 6)),
        None => base,
    }
}

/// Stable default idempotency key for a post (a retried command after a
/// crash returns the same event id instead of posting twice).
pub fn default_idem(p: &Post) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let body = format!(
        "{}\u{1}{}\u{1}{}\u{1}{}\u{1}{:?}\u{1}{:?}\u{1}{}",
        p.from,
        p.kind.as_str(),
        p.thread,
        p.topic,
        p.to,
        p.parent,
        p.text
    );
    for b in body.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("sb-{h:016x}")
}

/// Everything a bridge operation needs.
pub struct Bridge<'a, B: Board> {
    pub board: &'a B,
    pub state_dir: PathBuf,
    pub mission: String,
}

/// Result of one hook invocation, for tests and evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Delivery {
    pub text: Option<String>,
    pub ids: Vec<u64>,
    pub redelivered: Vec<u64>,
    pub tag: String,
    pub tag_note: Option<String>,
    pub confirmed_prior: bool,
}

impl<'a, B: Board> Bridge<'a, B> {
    pub fn new(board: &'a B, state_dir: &Path) -> Result<Bridge<'a, B>, BridgeError> {
        let mission = board.mission();
        check_name("mission", &mission, false)?;
        Ok(Bridge {
            board,
            state_dir: state_dir.to_path_buf(),
            mission,
        })
    }

    fn mission_short(&self) -> String {
        self.mission.chars().take(8).collect()
    }

    fn open(&self, key: &str) -> Result<(StateFile, State), BridgeError> {
        let sf = StateFile::open(&self.state_dir, &self.mission, key)?;
        let mut st = sf.load(&self.mission, key)?;
        if st.confirmed == 0 && st.pending.is_none() && st.batches == 0 {
            // Fresh or lost local state: resume from the board's durable ack.
            st.confirmed = self.board.acked(key)?;
        }
        Ok((sf, st))
    }

    fn tag_index(&self) -> PathBuf {
        self.state_dir.join(&self.mission).join("tags")
    }

    /// Resolve `--as`: a tag recorded by this bridge, or a session key.
    pub fn resolve_identity(&self, who: &str) -> Result<String, BridgeError> {
        check_name("identity", who, false)?;
        let p = self.tag_index().join(who);
        if let Ok(md) = fs::symlink_metadata(&p) {
            if md.is_file() && md.len() <= MAX_NAME as u64 {
                if let Ok(k) = fs::read_to_string(&p) {
                    let k = k.trim().to_string();
                    check_name("session", &k, false)?;
                    return Ok(k);
                }
            }
        }
        Ok(who.to_string())
    }

    /// Bind a tag to the session. Returns the tag and, when the wanted tag
    /// was taken, a note saying which tag the session got instead.
    fn ensure_tag(
        &self,
        st: &mut State,
        key: &str,
        wanted: &str,
    ) -> Result<(String, Option<String>), BridgeError> {
        if let Some(t) = st.tag.clone() {
            self.board.register(key, &t)?;
            return Ok((t, None));
        }
        check_name("tag", wanted, false)?;
        let mut note = None;
        let mut got = None;
        for n in 1..=TAG_SUFFIXES {
            let cand = if n == 1 {
                wanted.to_string()
            } else {
                format!("{wanted}-{n}")
            };
            match self.board.register(key, &cand) {
                Ok(()) => {
                    if n > 1 {
                        note = Some(format!(
                            "Tag {wanted} is taken by another session; you are {cand}."
                        ));
                    }
                    got = Some(cand);
                    break;
                }
                Err(BridgeError::TagTaken(_)) => continue,
                Err(e) => return Err(e),
            }
        }
        let tag = got
            .ok_or_else(|| BridgeError::TagTaken(format!("{wanted} (and -2..-{TAG_SUFFIXES})")))?;
        st.tag = Some(tag.clone());
        let dir = self.tag_index();
        if fs::create_dir_all(&dir).is_ok() {
            let _ = fs::write(dir.join(&tag), key);
        }
        Ok((tag, note))
    }

    /// Confirm or reject the previous batch. Returns the ids to label as
    /// redeliveries and whether a prior batch was confirmed.
    fn settle(&self, st: &mut State, key: &str) -> Result<(Vec<u64>, bool), BridgeError> {
        let Some(p) = st.pending.take() else {
            return Ok((Vec::new(), false));
        };
        let scans: Vec<Scan> = p
            .transcripts
            .iter()
            .map(|(path, off)| state::transcript_contains(Path::new(path), *off, &p.batch_tag))
            .collect();
        let confirmed = scans.is_empty()
            || scans.contains(&Scan::Found)
            || scans.iter().all(|s| *s == Scan::Unreadable);
        if confirmed {
            self.apply(st, key, p.confirmed, p.ahead)?;
            Ok((Vec::new(), true))
        } else {
            if p.banner {
                st.bannered = false;
            }
            Ok((p.ids, false))
        }
    }

    fn apply(
        &self,
        st: &mut State,
        key: &str,
        confirmed: u64,
        ahead: Vec<u64>,
    ) -> Result<(), BridgeError> {
        if confirmed > st.confirmed {
            st.confirmed = confirmed;
        }
        st.ahead = ahead.into_iter().filter(|i| *i > st.confirmed).collect();
        // Implicit ack (PLAN D9): the batch is proven delivered, and the
        // cursor only ever covers a contiguous prefix, so no gap is acked.
        if st.confirmed > self.board.acked(key)? {
            self.board.ack(key, st.confirmed)?;
        }
        Ok(())
    }

    /// Pick and render the next batch for `key`.
    fn select(
        &self,
        st: &State,
        key: &str,
        tag: &str,
        banner: Option<String>,
        redelivery: &[u64],
        batch_tag: &str,
    ) -> Result<Option<Selected>, BridgeError> {
        let start = st.confirmed.max(self.board.acked(key)?);
        let page = self.board.read_since(key, start, SCAN_LIMIT)?;
        let ahead: HashSet<u64> = st.ahead.iter().copied().collect();
        let mut seen_ids = HashSet::new();
        let mut own = BTreeSet::new();
        let mut addressed: Vec<&Event> = Vec::new();
        let mut subscribed: Vec<&Event> = Vec::new();
        let mut page_ids = BTreeSet::new();
        let mut max_sub: Option<u64> = None;
        for e in &page.events {
            if e.id <= start || !seen_ids.insert(e.id) {
                continue; // dedup by id
            }
            page_ids.insert(e.id);
            let is_addr = e.to.iter().any(|t| t == tag);
            if !is_addr {
                max_sub = Some(max_sub.map_or(e.id, |m: u64| m.max(e.id)));
            }
            if ahead.contains(&e.id) {
                continue;
            }
            if e.from == key {
                own.insert(e.id);
                continue;
            }
            if is_addr {
                addressed.push(e);
            } else {
                subscribed.push(e);
            }
        }
        // Alerts (blocked, handoff) lead the subscribed part.
        subscribed.sort_by_key(|e| (!e.kind.is_alert(), e.id));
        let redo: HashSet<u64> = redelivery.iter().copied().collect();
        let addr_view: Vec<(&Event, bool)> = addressed
            .iter()
            .map(|e| (*e, redo.contains(&e.id)))
            .collect();
        let sub_view: Vec<(&Event, bool)> = subscribed
            .iter()
            .map(|e| (*e, redo.contains(&e.id)))
            .collect();
        if addr_view.is_empty() && sub_view.is_empty() && banner.is_none() {
            // Nothing to show; still advance past our own posts.
            let p = prefix(
                &page_ids,
                start,
                |i| own.contains(&i) || ahead.contains(&i),
                page.more,
                max_sub,
            );
            if p > start {
                return Ok(Some(Selected {
                    text: None,
                    ids: Vec::new(),
                    confirmed: p,
                    ahead: st.ahead.clone(),
                }));
            }
            return Ok(None);
        }
        let mut view = BatchView {
            banner: banner.clone(),
            tag,
            mission: &self.mission,
            batch_tag,
            addressed: addr_view.clone(),
            subscribed: sub_view.clone(),
            overflow: 0,
            overflow_more: page.more,
            last_id: None,
        };
        let (_, fa, fs_) = render::render(&view);
        let mut ids: Vec<u64> = addressed[..fa]
            .iter()
            .chain(subscribed[..fs_].iter())
            .map(|e| e.id)
            .collect();
        ids.sort_unstable();
        let delivered: HashSet<u64> = ids.iter().copied().collect();
        let p = prefix(
            &page_ids,
            start,
            |i| delivered.contains(&i) || own.contains(&i) || ahead.contains(&i),
            page.more,
            max_sub,
        );
        view.last_id = (p > start).then_some(p);
        let (text, _, _) = render::render(&view);
        let mut new_ahead: Vec<u64> = st
            .ahead
            .iter()
            .copied()
            .chain(ids.iter().copied())
            .filter(|i| *i > p)
            .collect();
        new_ahead.sort_unstable();
        new_ahead.dedup();
        Ok(Some(Selected {
            text: Some(text),
            ids,
            confirmed: p,
            ahead: new_ahead,
        }))
    }

    /// `swarm-bridge hook`: one Claude Code hook invocation.
    pub fn hook(&self, input: &HookInput, cfg: &HookConfig) -> Result<Delivery, BridgeError> {
        let key = match &cfg.session_override {
            Some(s) => session_key(s, input.agent_id.as_deref())?,
            None => session_key(&input.session_id, input.agent_id.as_deref())?,
        };
        let (sf, mut st) = self.open(&key)?;
        let wanted = wanted_tag(
            cfg.swarm_tag.as_deref(),
            cfg.session_override.as_deref().unwrap_or(&input.session_id),
            input.agent_id.as_deref(),
        );
        let (tag, tag_note) = self.ensure_tag(&mut st, &key, &wanted)?;
        let first = !st.bannered;
        if first || input.event == "SessionStart" || input.event == "SubagentStart" {
            for s in &cfg.subscribe {
                check_sub(s)?;
                self.board.subscribe(&key, s)?;
            }
        }
        let (redelivery, confirmed_prior) = self.settle(&mut st, &key)?;
        let banner = (!st.bannered).then(|| {
            let mut b = format!("You are {tag} (board {}).", self.mission_short());
            if let Some(n) = &tag_note {
                b.push(' ');
                b.push_str(n);
            }
            if input.agent_id.is_some() {
                b.push_str(&format!(
                    " You are a subagent: pass --as {tag} to every swarm-bridge command."
                ));
            }
            b
        });
        let batch_tag = format!("[sb {}/{}/{}]", self.mission_short(), tag, st.batches + 1);
        let sel = self.select(&st, &key, &tag, banner.clone(), &redelivery, &batch_tag)?;
        let mut out = Delivery {
            tag: tag.clone(),
            tag_note,
            confirmed_prior,
            redelivered: redelivery.clone(),
            ..Delivery::default()
        };
        match sel {
            None => sf.save(&st)?,
            Some(Selected {
                text: None,
                confirmed,
                ahead,
                ..
            }) => {
                self.apply(&mut st, &key, confirmed, ahead)?;
                sf.save(&st)?;
            }
            Some(Selected {
                text: Some(text),
                ids,
                confirmed,
                ahead,
            }) => {
                let mut transcripts = Vec::new();
                if let Some(tp) = &input.transcript_path {
                    transcripts.push(tp.clone());
                    if let Some(a) = &input.agent_id {
                        // Subagent hooks carry the parent's transcript path;
                        // the subagent's own transcript sits beside it.
                        let side = tp
                            .with_extension("")
                            .join("subagents")
                            .join(format!("agent-{a}.jsonl"));
                        transcripts.push(side);
                    }
                }
                st.pending = Some(Pending {
                    banner: banner.is_some(),
                    ids: ids.clone(),
                    confirmed,
                    ahead,
                    batch_tag: batch_tag.clone(),
                    transcripts: transcripts
                        .iter()
                        .map(|p| {
                            (
                                p.to_string_lossy().into_owned(),
                                state::transcript_len(p).unwrap_or(0),
                            )
                        })
                        .collect(),
                });
                st.bannered = true;
                st.batches += 1;
                // Persist before printing: a crash here leaves a pending batch
                // whose tag never reaches the transcript, so it is redelivered.
                sf.save(&st)?;
                out.text = Some(hook_json(&input.event, &text)?);
                out.ids = ids;
            }
        }
        Ok(out)
    }

    /// `swarm-bridge read`: deliver the next batch straight into the
    /// caller's tool output. Bash output is in the transcript, so the batch
    /// counts as delivered at once. `peek` shows without moving the cursor.
    pub fn read(&self, key: &str, peek: bool) -> Result<ReadOut, BridgeError> {
        let (sf, mut st) = self.open(key)?;
        let tag = st.tag.clone().unwrap_or_else(|| key.to_string());
        let _ = self.settle(&mut st, key)?;
        let batch_tag = format!("[sb {}/{}/r{}]", self.mission_short(), tag, st.batches + 1);
        let sel = self.select(&st, key, &tag, None, &[], &batch_tag)?;
        let mut out = ReadOut::default();
        if let Some(s) = sel {
            out.text = s.text.clone();
            out.ids = s.ids.clone();
            if !peek {
                self.apply(&mut st, key, s.confirmed, s.ahead)?;
                if s.text.is_some() {
                    st.batches += 1;
                }
            }
        }
        if !peek {
            sf.save(&st)?;
        }
        Ok(out)
    }

    /// `swarm-bridge watch`: park without spending model tokens until a
    /// notification arrives or `timeout` passes, then read since the cursor.
    pub fn watch(&self, key: &str, timeout: Duration) -> Result<(ReadOut, WatchEnd), BridgeError> {
        let first = self.read(key, false)?;
        if first.text.is_some() {
            return Ok((first, WatchEnd::AlreadyWaiting));
        }
        let notified = self.board.wait(key, timeout)?;
        let out = self.read(key, false)?;
        let end = match (notified, out.text.is_some()) {
            (true, _) => WatchEnd::Notified,
            (false, true) => WatchEnd::TimeoutFound,
            (false, false) => WatchEnd::TimeoutEmpty,
        };
        Ok((out, end))
    }

    /// `swarm-bridge ack <id>`: explicit acknowledgement. Never acks past
    /// the contiguous delivered prefix, so an ack can not create a gap.
    pub fn ack(&self, key: &str, id: u64) -> Result<u64, BridgeError> {
        let (sf, mut st) = self.open(key)?;
        if let Some(p) = st.pending.clone() {
            // Acking an id the batch showed proves the batch arrived.
            if p.ids.contains(&id) || (id <= p.confirmed && id > st.confirmed) {
                st.pending = None;
                self.apply(&mut st, key, p.confirmed, p.ahead)?;
            }
        }
        let target = id.min(st.confirmed);
        let cur = self.board.ack(key, target)?;
        sf.save(&st)?;
        Ok(cur)
    }

    /// One durable write per message, sender tag first.
    pub fn post(&self, key: &str, req: PostReq) -> Result<u64, BridgeError> {
        if req.kind == Kind::Answer && req.parent.is_none() {
            return Err(BridgeError::AnswerNeedsParent);
        }
        if req.text.len() > MAX_TEXT {
            return Err(BridgeError::BadArgs(format!(
                "text is {} bytes, cap {MAX_TEXT}",
                req.text.len()
            )));
        }
        if req.refs.len() > MAX_REFS {
            return Err(BridgeError::BadArgs(format!(
                "{} refs, cap {MAX_REFS}",
                req.refs.len()
            )));
        }
        let (sf, mut st) = self.open(key)?;
        let tag = match st.tag.clone() {
            Some(t) => t,
            None => {
                let wanted = wanted_tag(req.swarm_tag.as_deref(), key, None);
                self.ensure_tag(&mut st, key, &wanted)?.0
            }
        };
        let parent_ev = match req.parent {
            Some(p) => self.board.get(key, p)?,
            None => None,
        };
        let thread = match (req.thread, &parent_ev) {
            (Some(t), _) => t,
            (None, Some(pe)) => pe.thread.clone(),
            (None, None) => {
                return Err(BridgeError::BadArgs(
                    "--thread is required (or --parent of an event you can see)".into(),
                ))
            }
        };
        check_name("thread", &thread, true)?;
        let topic = match (req.topic, &parent_ev) {
            (Some(t), _) => t,
            (None, Some(pe)) => pe.topic.clone(),
            (None, None) => thread.clone(),
        };
        check_name("topic", &topic, true)?;
        let mut to = req.to;
        if to.is_empty() && req.kind == Kind::Answer {
            if let Some(pe) = &parent_ev {
                to.push(pe.from_tag.clone());
            }
        }
        for t in &to {
            check_name("to tag", t, false)?;
        }
        let mut post = Post {
            from_tag: tag,
            from: key.to_string(),
            thread,
            topic,
            to,
            kind: req.kind,
            parent: req.parent,
            text: req.text,
            refs: req.refs,
            idem: req.idem,
        };
        if post.idem.is_none() {
            post.idem = Some(default_idem(&post));
        }
        let id = self.board.post(&post)?;
        sf.save(&st)?;
        Ok(id)
    }

    pub fn subscribe(&self, key: &str, sub: &Sub, on: bool) -> Result<(), BridgeError> {
        check_sub(sub)?;
        if on {
            self.board.subscribe(key, sub)
        } else {
            self.board.unsubscribe(key, sub)
        }
    }

    /// Full text of one event the caller can see (bounded by the board).
    pub fn show(&self, key: &str, id: u64) -> Result<Option<String>, BridgeError> {
        Ok(self.board.get(key, id)?.map(|e| {
            format!(
                "{}\n{}\nfull text: {}",
                render::UNTRUSTED_HEADER,
                render::event_line(&e, MAX_TEXT, false),
                render::one_line(&e.text)
            )
        }))
    }

    pub fn whoami(&self, key: &str) -> Result<String, BridgeError> {
        let (_sf, st) = self.open(key)?;
        let subs: Vec<String> = self
            .board
            .subscriptions(key)?
            .iter()
            .map(Sub::describe)
            .collect();
        Ok(format!(
            "tag={} session={} board={} confirmed={} acked={} pending={} subscriptions=[{}]\nThe tag is for addressing only; it is not authentication.",
            st.tag.as_deref().unwrap_or("(none yet)"),
            key,
            self.mission,
            st.confirmed,
            self.board.acked(key)?,
            st.pending.as_ref().map_or(0, |p| p.ids.len()),
            subs.join(", ")
        ))
    }
}

/// Highest id P such that every page id in (start, P] satisfies `seen`.
/// When the page was cut (`more`), events past it are unknown: P may not
/// pass the last subscribed id in the page, and if the page held only
/// addressed events P does not move.
fn prefix(
    page_ids: &BTreeSet<u64>,
    start: u64,
    seen: impl Fn(u64) -> bool,
    more: bool,
    max_sub: Option<u64>,
) -> u64 {
    let mut p = start;
    for &i in page_ids {
        if seen(i) {
            p = i;
        } else {
            break;
        }
    }
    if more {
        match max_sub {
            Some(m) => p.min(m),
            None => start,
        }
    } else {
        p
    }
}

struct Selected {
    text: Option<String>,
    ids: Vec<u64>,
    confirmed: u64,
    ahead: Vec<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct ReadOut {
    pub text: Option<String>,
    pub ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchEnd {
    AlreadyWaiting,
    Notified,
    TimeoutFound,
    TimeoutEmpty,
}

/// Hook configuration taken from the session's environment.
#[derive(Debug, Clone, Default)]
pub struct HookConfig {
    /// `SWARM_TAG`.
    pub swarm_tag: Option<String>,
    /// `SWARM_BRIDGE_SESSION`: a readable session key for demos.
    pub session_override: Option<String>,
    /// `SWARM_BRIDGE_SUBSCRIBE`: `thread:x,topic:y/*`, applied on first hook.
    pub subscribe: Vec<Sub>,
}

pub fn parse_subs(s: &str) -> Result<Vec<Sub>, BridgeError> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| Sub::parse(x).ok_or_else(|| BridgeError::BadArgs(format!("subscription {x:?}"))))
        .collect()
}

/// A post as the CLI builds it.
#[derive(Debug, Clone)]
pub struct PostReq {
    pub kind: Kind,
    pub thread: Option<String>,
    pub topic: Option<String>,
    pub to: Vec<String>,
    pub parent: Option<u64>,
    pub text: String,
    pub refs: Vec<Ref>,
    pub idem: Option<String>,
    pub swarm_tag: Option<String>,
}

/// The JSON Claude Code expects from a context-adding hook.
pub fn hook_json(event: &str, text: &str) -> Result<String, BridgeError> {
    let v = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": text,
        }
    });
    serde_json::to_string(&v).map_err(|e| BridgeError::Io(format!("encode hook output: {e}")))
}

/// Events whose hook output may carry `additionalContext`.
pub fn event_takes_context(event: &str) -> bool {
    matches!(
        event,
        "PostToolUse" | "UserPromptSubmit" | "SessionStart" | "SubagentStart"
    )
}
