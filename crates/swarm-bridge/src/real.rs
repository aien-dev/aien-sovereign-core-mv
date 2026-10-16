//! The real board: synapse-coord's swarm board (synapse#83). Phase 1 stub
//! until Lane A's board API lands; every call refuses by name.

use std::path::Path;
use std::time::Duration;

use swarm_bridge::board::{Board, BridgeError, Event, Page, Post, Sub};

pub struct RealBoard;

pub fn open(_root: &Path) -> Result<RealBoard, BridgeError> {
    Err(BridgeError::Board(
        "BoardNotWired: synapse-coord board API not linked in this build".into(),
    ))
}

fn nw<T>() -> Result<T, BridgeError> {
    Err(BridgeError::Board("BoardNotWired".into()))
}

impl Board for RealBoard {
    fn mission(&self) -> String {
        String::new()
    }
    fn register(&self, _s: &str, _t: &str) -> Result<(), BridgeError> {
        nw()
    }
    fn post(&self, _p: &Post) -> Result<u64, BridgeError> {
        nw()
    }
    fn subscribe(&self, _s: &str, _sub: &Sub) -> Result<(), BridgeError> {
        nw()
    }
    fn unsubscribe(&self, _s: &str, _sub: &Sub) -> Result<(), BridgeError> {
        nw()
    }
    fn subscriptions(&self, _s: &str) -> Result<Vec<Sub>, BridgeError> {
        nw()
    }
    fn read_since(&self, _s: &str, _a: u64, _l: usize) -> Result<Page, BridgeError> {
        nw()
    }
    fn get(&self, _s: &str, _id: u64) -> Result<Option<Event>, BridgeError> {
        nw()
    }
    fn acked(&self, _s: &str) -> Result<u64, BridgeError> {
        nw()
    }
    fn ack(&self, _s: &str, _id: u64) -> Result<u64, BridgeError> {
        nw()
    }
    fn wait(&self, _s: &str, _t: Duration) -> Result<bool, BridgeError> {
        nw()
    }
}
