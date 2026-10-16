//! In-memory board used by the bridge's own tests. It follows the same
//! contract as the real board (addressed first, idempotent posts, monotonic
//! acks, unique tags) and can drop notifications on purpose.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use crate::board::{Board, BridgeError, Event, Page, Post, Sub};

#[derive(Default)]
struct Inner {
    events: Vec<Event>,
    tags: HashMap<String, String>,
    subs: HashMap<String, Vec<Sub>>,
    acked: HashMap<String, u64>,
    idem: HashMap<String, (String, u64)>,
    wakes: u64,
}

pub struct MemoryBoard {
    mission: String,
    inner: Mutex<Inner>,
    cv: Condvar,
    /// When set, posts do not notify (the "dropped datagram" fault).
    pub drop_notifications: AtomicBool,
}

impl MemoryBoard {
    pub fn new(mission: &str) -> MemoryBoard {
        MemoryBoard {
            mission: mission.to_string(),
            inner: Mutex::new(Inner::default()),
            cv: Condvar::new(),
            drop_notifications: AtomicBool::new(false),
        }
    }

    pub fn len(&self) -> usize {
        self.lock().events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }
}

fn tag_of(inner: &Inner, session: &str) -> Option<String> {
    inner
        .tags
        .iter()
        .find(|(_, s)| s.as_str() == session)
        .map(|(t, _)| t.clone())
}

fn body_key(p: &Post) -> String {
    format!(
        "{}|{}|{}|{}|{:?}|{}|{:?}|{}|{:?}",
        p.from_tag,
        p.from,
        p.thread,
        p.topic,
        p.to,
        p.kind.as_str(),
        p.parent,
        p.text,
        p.refs
    )
}

impl Board for MemoryBoard {
    fn mission(&self) -> String {
        self.mission.clone()
    }

    fn register(&self, session: &str, tag: &str) -> Result<(), BridgeError> {
        let mut g = self.lock();
        match g.tags.get(tag) {
            Some(s) if s == session => Ok(()),
            Some(_) => Err(BridgeError::TagTaken(tag.to_string())),
            None if g.tags.values().any(|s| s == session) => Err(BridgeError::Board(format!(
                "SessionTagged: {session} already holds another tag"
            ))),
            None => {
                g.tags.insert(tag.to_string(), session.to_string());
                Ok(())
            }
        }
    }

    fn post(&self, post: &Post) -> Result<u64, BridgeError> {
        let mut g = self.lock();
        let body = body_key(post);
        if let Some(k) = &post.idem {
            if let Some((b, id)) = g.idem.get(k) {
                if *b == body {
                    return Ok(*id);
                }
                return Err(BridgeError::Board(format!("IdempotencyConflict: {k}")));
            }
        }
        let id = g.events.len() as u64 + 1;
        g.events.push(Event {
            id,
            from_tag: post.from_tag.clone(),
            from: post.from.clone(),
            thread: post.thread.clone(),
            topic: post.topic.clone(),
            to: post.to.clone(),
            kind: post.kind,
            parent: post.parent,
            ts: 0,
            text: post.text.clone(),
            refs: post.refs.clone(),
        });
        if let Some(k) = &post.idem {
            g.idem.insert(k.clone(), (body, id));
        }
        if !self.drop_notifications.load(Ordering::SeqCst) {
            g.wakes += 1;
            self.cv.notify_all();
        }
        Ok(id)
    }

    fn subscribe(&self, session: &str, sub: &Sub) -> Result<(), BridgeError> {
        let mut g = self.lock();
        let v = g.subs.entry(session.to_string()).or_default();
        if !v.contains(sub) {
            v.push(sub.clone());
        }
        Ok(())
    }

    fn unsubscribe(&self, session: &str, sub: &Sub) -> Result<(), BridgeError> {
        let mut g = self.lock();
        if let Some(v) = g.subs.get_mut(session) {
            v.retain(|s| s != sub);
        }
        Ok(())
    }

    fn subscriptions(&self, session: &str) -> Result<Vec<Sub>, BridgeError> {
        Ok(self.lock().subs.get(session).cloned().unwrap_or_default())
    }

    fn tag_of(&self, session: &str) -> Result<Option<String>, BridgeError> {
        Ok(tag_of(&self.lock(), session))
    }

    /// Same page contract as the real board: the lowest `limit` matching ids
    /// after `after`, then addressed first; the reader's own posts are not
    /// shown.
    fn read_since(&self, session: &str, after: u64, limit: usize) -> Result<Page, BridgeError> {
        let g = self.lock();
        let tag = tag_of(&g, session);
        let subs = g.subs.get(session).cloned().unwrap_or_default();
        let limit = limit.max(1);
        let mut chosen: Vec<(&Event, bool)> = Vec::new();
        let mut more = false;
        for e in g.events.iter().filter(|e| e.id > after) {
            let addressed = tag.as_ref().is_some_and(|t| e.to.contains(t));
            let subscribed = subs.iter().any(|s| match s {
                Sub::Thread(t) => *t == e.thread,
                Sub::Topic(p) => *p == e.topic,
            });
            if !(addressed || subscribed) {
                continue;
            }
            if chosen.len() == limit {
                more = true;
                break;
            }
            chosen.push((e, addressed));
        }
        let through = if more {
            chosen.last().map_or(after, |c| c.0.id)
        } else {
            (g.events.len() as u64).max(after)
        };
        let shown = |e: &Event| !tag.as_ref().is_some_and(|t| *t == e.from_tag);
        let mut events: Vec<Event> = Vec::new();
        for want_addressed in [true, false] {
            events.extend(
                chosen
                    .iter()
                    .filter(|(e, a)| *a == want_addressed && shown(e))
                    .map(|(e, _)| (*e).clone()),
            );
        }
        Ok(Page {
            events,
            through,
            more,
        })
    }

    fn get(&self, session: &str, id: u64) -> Result<Option<Event>, BridgeError> {
        let page = self.read_since(session, id.saturating_sub(1), usize::MAX)?;
        Ok(page.events.into_iter().find(|e| e.id == id))
    }

    fn acked(&self, session: &str) -> Result<u64, BridgeError> {
        Ok(self.lock().acked.get(session).copied().unwrap_or(0))
    }

    fn ack(&self, session: &str, id: u64) -> Result<u64, BridgeError> {
        let mut g = self.lock();
        let last = g.events.len() as u64;
        if id > last {
            return Err(BridgeError::Board(format!(
                "AckBeyondLast: {id} > last committed {last}"
            )));
        }
        let cur = g.acked.entry(session.to_string()).or_insert(0);
        if id > *cur {
            *cur = id;
        }
        Ok(*cur)
    }

    fn wait(
        &self,
        session: &str,
        after: u64,
        seen: &[u64],
        timeout: Duration,
    ) -> Result<bool, BridgeError> {
        let start = self.lock().wakes;
        let page = self.read_since(session, after, 64)?;
        if page.events.iter().any(|e| !seen.contains(&e.id)) {
            return Ok(true);
        }
        let g = self.lock();
        let (g, res) = match self.cv.wait_timeout_while(g, timeout, |i| i.wakes == start) {
            Ok(x) => x,
            Err(p) => p.into_inner(),
        };
        Ok(!res.timed_out() && g.wakes != start)
    }
}
