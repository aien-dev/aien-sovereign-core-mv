//! The real board: synapse-coord's swarm board (synapse#83, `aien-board/1`).
//! A thin adapter from the bridge's [`Board`] trait to `synapse_coord::board`.
//! The board keys tags, cursors and subscriptions by tag; the bridge passes
//! its session key as the board session, so one bridge session is one tag.

use std::path::Path;
use std::time::Duration;

use synapse_coord::board as sb;

use crate::board::{Board, BridgeError, Event, Kind, Page, Post, Ref, RefKind, Sub};

/// Most pages scanned when arming a wait.
const WAIT_SCAN_PAGES: usize = 16;

pub struct RealBoard {
    board: sb::Board,
}

fn err(e: sb::BoardError) -> BridgeError {
    match e {
        sb::BoardError::TagTaken { tag, holder } => {
            BridgeError::TagTaken(format!("{tag} (held by session {holder})"))
        }
        e => BridgeError::Board(format!("{}: {e}", e.name())),
    }
}

/// Open an existing board at `root`.
pub fn open(root: &Path) -> Result<RealBoard, BridgeError> {
    Ok(RealBoard {
        board: sb::Board::open(root).map_err(err)?,
    })
}

/// Create a new board at `root` (refused when one exists).
pub fn init(root: &Path) -> Result<RealBoard, BridgeError> {
    Ok(RealBoard {
        board: sb::Board::init(root).map_err(err)?,
    })
}

fn kind_to(k: Kind) -> sb::Kind {
    match k {
        Kind::Finding => sb::Kind::Finding,
        Kind::Question => sb::Kind::Question,
        Kind::Answer => sb::Kind::Answer,
        Kind::Blocked => sb::Kind::Blocked,
        Kind::Ready => sb::Kind::Ready,
        Kind::Handoff => sb::Kind::Handoff,
        Kind::ReceiptRef => sb::Kind::ReceiptRef,
    }
}

fn kind_from(k: sb::Kind) -> Kind {
    match k {
        sb::Kind::Finding => Kind::Finding,
        sb::Kind::Question => Kind::Question,
        sb::Kind::Answer => Kind::Answer,
        sb::Kind::Blocked => Kind::Blocked,
        sb::Kind::Ready => Kind::Ready,
        sb::Kind::Handoff => Kind::Handoff,
        sb::Kind::ReceiptRef => Kind::ReceiptRef,
    }
}

fn refkind_to(k: RefKind) -> sb::RefKind {
    match k {
        RefKind::Issue => sb::RefKind::Issue,
        RefKind::Commit => sb::RefKind::Commit,
        RefKind::Receipt => sb::RefKind::Receipt,
        RefKind::Pr => sb::RefKind::Pr,
        RefKind::Path => sb::RefKind::Path,
    }
}

fn refkind_from(k: sb::RefKind) -> RefKind {
    match k {
        sb::RefKind::Issue => RefKind::Issue,
        sb::RefKind::Commit => RefKind::Commit,
        sb::RefKind::Receipt => RefKind::Receipt,
        sb::RefKind::Pr => RefKind::Pr,
        sb::RefKind::Path => RefKind::Path,
    }
}

fn sub_to(s: &Sub) -> sb::Sub {
    match s {
        Sub::Thread(t) => sb::Sub::Thread(t.clone()),
        Sub::Topic(t) => sb::Sub::Topic(t.clone()),
    }
}

/// A board event in the bridge's shape.
pub fn event_from(e: sb::Event) -> Event {
    Event {
        id: e.id,
        from_tag: e.from_tag,
        from: e.from,
        thread: e.thread,
        topic: e.topic,
        to: e.to,
        kind: kind_from(e.kind),
        parent: e.parent,
        ts: u64::try_from(e.ts).unwrap_or(0),
        text: e.text,
        refs: e
            .refs
            .into_iter()
            .map(|r| Ref {
                repo: r.repo,
                kind: refkind_from(r.kind),
                value: r.value,
            })
            .collect(),
    }
}

impl RealBoard {
    /// The board directory.
    pub fn root(&self) -> &Path {
        self.board.root()
    }

    /// Operator view for evidence: every event after `after`, ignoring
    /// audience (`read_all`). Not used for delivery.
    pub fn all_events(
        &self,
        after: u64,
        limit: usize,
    ) -> Result<(Vec<Event>, u64, bool), BridgeError> {
        let b = self.board.read_all(after, limit).map_err(err)?;
        Ok((
            b.events.into_iter().map(|d| event_from(d.event)).collect(),
            b.through,
            b.more,
        ))
    }
}

impl Board for RealBoard {
    fn mission(&self) -> String {
        self.board.mission()
    }

    fn register(&self, session: &str, tag: &str) -> Result<(), BridgeError> {
        self.board
            .register(session, tag, None)
            .map(|_| ())
            .map_err(err)
    }

    fn tag_of(&self, session: &str) -> Result<Option<String>, BridgeError> {
        match self.board.whoami(session) {
            Ok(me) => Ok(Some(me.tag)),
            Err(sb::BoardError::NotRegistered { .. }) => Ok(None),
            Err(e) => Err(err(e)),
        }
    }

    fn post(&self, p: &Post) -> Result<u64, BridgeError> {
        let post = sb::Post {
            from_tag: p.from_tag.clone(),
            from: p.from.clone(),
            thread: p.thread.clone(),
            topic: p.topic.clone(),
            to: p.to.clone(),
            kind: kind_to(p.kind),
            parent: p.parent,
            text: p.text.clone(),
            refs: p
                .refs
                .iter()
                .map(|r| sb::Ref {
                    repo: r.repo.clone(),
                    kind: refkind_to(r.kind),
                    value: r.value.clone(),
                })
                .collect(),
            idem: p.idem.clone(),
        };
        self.board.post(&post).map(|r| r.id).map_err(err)
    }

    fn subscribe(&self, session: &str, sub: &Sub) -> Result<(), BridgeError> {
        self.board
            .subscribe(session, &sub_to(sub))
            .map(|_| ())
            .map_err(err)
    }

    fn unsubscribe(&self, session: &str, sub: &Sub) -> Result<(), BridgeError> {
        self.board
            .unsubscribe(session, &sub_to(sub))
            .map(|_| ())
            .map_err(err)
    }

    fn subscriptions(&self, session: &str) -> Result<Vec<Sub>, BridgeError> {
        let s = self.board.subscriptions(session).map_err(err)?;
        Ok(s.threads
            .into_iter()
            .map(Sub::Thread)
            .chain(s.topics.into_iter().map(Sub::Topic))
            .collect())
    }

    fn read_since(&self, session: &str, after: u64, limit: usize) -> Result<Page, BridgeError> {
        let b = self.board.read_since(session, after, limit).map_err(err)?;
        Ok(Page {
            events: b.events.into_iter().map(|d| event_from(d.event)).collect(),
            through: b.through.max(after),
            more: b.more,
        })
    }

    fn get(&self, session: &str, id: u64) -> Result<Option<Event>, BridgeError> {
        Ok(self.board.get(session, id).map_err(err)?.map(event_from))
    }

    fn acked(&self, session: &str) -> Result<u64, BridgeError> {
        self.board.acked(session).map_err(err)
    }

    fn ack(&self, session: &str, id: u64) -> Result<u64, BridgeError> {
        self.board.ack(session, id).map(|c| c.acked).map_err(err)
    }

    fn wait(
        &self,
        session: &str,
        after: u64,
        seen: &[u64],
        timeout: Duration,
    ) -> Result<bool, BridgeError> {
        // Bind the wake socket first, then look: a post landing between the
        // look and the wait still wakes the socket.
        let w = self.board.watcher(session).map_err(err)?;
        let mut from = after;
        for _ in 0..WAIT_SCAN_PAGES {
            let b = self.board.read_since(session, from, 64).map_err(err)?;
            if b.events.iter().any(|d| !seen.contains(&d.event.id)) {
                return Ok(true);
            }
            if !b.more || b.through <= from {
                break;
            }
            from = b.through;
        }
        match w.wait(timeout).map_err(err)? {
            sb::Wake::Woke(_) => Ok(true),
            sb::Wake::TimedOut => Ok(false),
        }
    }
}
