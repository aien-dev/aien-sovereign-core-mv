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

use chrono::Utc;
use rusqlite::{params, Connection, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::embeddings::{bytes_to_embedding, cosine_similarity, embedding_to_bytes};
use crate::models::*;

pub fn compute_event_hash(
    session_id: &str,
    branch_id: &str,
    sequence: i64,
    event_type: &str,
    role: Option<&str>,
    content: Option<&str>,
    payload_json: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(session_id.as_bytes());
    hasher.update(b":");
    hasher.update(branch_id.as_bytes());
    hasher.update(b":");
    hasher.update(sequence.to_string().as_bytes());
    hasher.update(b":");
    hasher.update(event_type.as_bytes());
    hasher.update(b":");
    if let Some(r) = role {
        hasher.update(r.as_bytes());
    }
    hasher.update(b":");
    if let Some(c) = content {
        hasher.update(c.as_bytes());
    }
    hasher.update(b":");
    hasher.update(payload_json.as_bytes());
    format!("{:x}", hasher.finalize())
}

type WriteTask = Box<dyn FnOnce(&mut Connection) + Send>;

pub struct StorageWriter {
    sender: Option<std::sync::mpsc::Sender<WriteTask>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl StorageWriter {
    pub fn new(mut conn: Connection) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<WriteTask>();
        let handle = std::thread::Builder::new()
            .name("cortex-writer".to_string())
            .spawn(move || {
                while let Ok(task) = rx.recv() {
                    task(&mut conn);
                    while let Ok(next) = rx.try_recv() {
                        next(&mut conn);
                    }
                }
            })
            .expect("spawn cortex-writer thread");

        Self {
            sender: Some(tx),
            worker: Some(handle),
        }
    }

    pub fn execute<T: Send + 'static, F: FnOnce(&mut Connection) -> Result<T> + Send + 'static>(
        &self,
        f: F,
    ) -> Result<T> {
        let sender = self.sender.as_ref().ok_or_else(|| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_INTERNAL),
                Some("StorageWriter sender dropped".to_string()),
            )
        })?;
        let (res_tx, res_rx) = std::sync::mpsc::channel();
        sender
            .send(Box::new(move |conn| {
                let r = f(conn);
                let _ = res_tx.send(r);
            }))
            .map_err(|e| {
                rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_INTERNAL),
                    Some(format!("StorageWriter send error: {}", e)),
                )
            })?;
        res_rx.recv().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_INTERNAL),
                Some(format!("StorageWriter recv error: {}", e)),
            )
        })?
    }
}

impl Drop for StorageWriter {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub struct ReaderPool {
    path: Option<PathBuf>,
    shared_mem_uri: Option<String>,
    pool: Mutex<Vec<Connection>>,
    max_size: usize,
}

impl ReaderPool {
    pub fn new(path: Option<PathBuf>, shared_mem_uri: Option<String>, max_size: usize) -> Self {
        Self {
            path,
            shared_mem_uri,
            pool: Mutex::new(Vec::with_capacity(max_size)),
            max_size,
        }
    }

    pub fn acquire(self: &Arc<Self>) -> Result<PooledConnection> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(conn) = pool.pop() {
            Ok(PooledConnection {
                conn: Some(conn),
                pool: Arc::clone(self),
            })
        } else {
            drop(pool);
            let conn = self.open_reader()?;
            Ok(PooledConnection {
                conn: Some(conn),
                pool: Arc::clone(self),
            })
        }
    }

    pub fn open_reader(&self) -> Result<Connection> {
        let conn = if let Some(path) = &self.path {
            let conn = Connection::open(path)?;
            let _: Result<String, _> =
                conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0));
            conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
            conn
        } else if let Some(uri) = &self.shared_mem_uri {
            let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_URI
                | rusqlite::OpenFlags::SQLITE_OPEN_SHARED_CACHE;
            let conn = Connection::open_with_flags(uri, flags)?;
            conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
            conn
        } else {
            return Err(rusqlite::Error::InvalidPath(PathBuf::from(":invalid:")));
        };
        Ok(conn)
    }

    fn release(&self, conn: Connection) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.max_size {
            pool.push(conn);
        }
    }
}

pub struct PooledConnection {
    conn: Option<Connection>,
    pool: Arc<ReaderPool>,
}

impl std::ops::Deref for PooledConnection {
    type Target = Connection;
    fn deref(&self) -> &Self::Target {
        self.conn.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.conn.as_mut().unwrap()
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.release(conn);
        }
    }
}

pub struct LegacyConnGuard {
    conn: PooledConnection,
}

impl std::ops::Deref for LegacyConnGuard {
    type Target = Connection;
    fn deref(&self) -> &Self::Target {
        &self.conn
    }
}

impl std::ops::DerefMut for LegacyConnGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.conn
    }
}

pub struct LegacyConnAdapter {
    reader_pool: Arc<ReaderPool>,
}

impl LegacyConnAdapter {
    pub fn lock(&self) -> Result<LegacyConnGuard, Box<std::sync::PoisonError<LegacyConnGuard>>> {
        let conn = self
            .reader_pool
            .acquire()
            .expect("acquire connection for legacy adapter");
        Ok(LegacyConnGuard { conn })
    }
}

pub struct Database {
    writer: Arc<StorageWriter>,
    reader_pool: Arc<ReaderPool>,
    pub conn: LegacyConnAdapter,
    hmac_key: Vec<u8>,
}

fn load_hmac_key(conn: &Connection) -> Result<Vec<u8>> {
    conn.query_row(
        "SELECT hmac_key FROM security_state WHERE id = 1",
        [],
        |row| row.get(0),
    )
}

fn shift_back(now: &str, days: i64) -> String {
    chrono::DateTime::parse_from_rfc3339(now)
        .map(|parsed| (parsed.with_timezone(&Utc) - chrono::Duration::days(days)).to_rfc3339())
        .unwrap_or_else(|_| now.to_string())
}

fn ensure_space_internal(conn: &Connection, slug: &str) -> Result<String> {
    let mut stmt = conn.prepare("SELECT id FROM spaces WHERE slug = ?1")?;
    let mut rows = stmt.query(params![slug])?;
    if let Some(row) = rows.next()? {
        return row.get(0);
    }
    drop(rows);
    drop(stmt);

    let new_id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO spaces (id, slug, created_at) VALUES (?1, ?2, ?3)",
        params![new_id, slug, now],
    )?;
    Ok(new_id)
}

pub struct SessionSummaryWrite<'a> {
    pub session_id: &'a str,
    pub branch_id: &'a str,
    pub level: i64,
    pub start_seq: i64,
    pub end_seq: i64,
    pub summary_text: &'a str,
    pub source_hash: &'a str,
    pub processor_version: &'a str,
}

impl Database {
    pub fn hmac_key(&self) -> &[u8] {
        &self.hmac_key
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let mut write_conn = Connection::open(&path_buf)?;
        let _: Result<String, _> =
            write_conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0));
        write_conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
        Self::run_migrations(&mut write_conn)?;

        let hmac_key = load_hmac_key(&write_conn)?;
        let writer = Arc::new(StorageWriter::new(write_conn));
        let reader_pool = Arc::new(ReaderPool::new(Some(path_buf), None, 16));
        let conn = LegacyConnAdapter {
            reader_pool: Arc::clone(&reader_pool),
        };

        Ok(Self {
            writer,
            reader_pool,
            conn,
            hmac_key,
        })
    }

    #[allow(dead_code)]
    pub fn open_in_memory() -> Result<Self> {
        let uri = format!(
            "file:cortex_mem_{}?mode=memory&cache=shared",
            Uuid::new_v4().simple()
        );
        let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
            | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
            | rusqlite::OpenFlags::SQLITE_OPEN_URI
            | rusqlite::OpenFlags::SQLITE_OPEN_SHARED_CACHE;
        let mut write_conn = Connection::open_with_flags(&uri, flags)?;
        let _: Result<String, _> =
            write_conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0));
        write_conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
        Self::run_migrations(&mut write_conn)?;

        let hmac_key = load_hmac_key(&write_conn)?;
        let writer = Arc::new(StorageWriter::new(write_conn));
        let reader_pool = Arc::new(ReaderPool::new(None, Some(uri), 16));
        let conn = LegacyConnAdapter {
            reader_pool: Arc::clone(&reader_pool),
        };

        Ok(Self {
            writer,
            reader_pool,
            conn,
            hmac_key,
        })
    }

    fn run_migrations(conn: &mut Connection) -> Result<()> {
        let user_version: i32 = conn.query_row("PRAGMA user_version;", [], |r| r.get(0))?;

        if user_version < 1 {
            conn.execute_batch(
                "BEGIN;
                CREATE TABLE IF NOT EXISTS spaces (
                    id TEXT PRIMARY KEY,
                    slug TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    space_id TEXT NOT NULL,
                    space_slug TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    retracted INTEGER NOT NULL DEFAULT 0,
                    valid_from TEXT,
                    valid_to TEXT,
                    created_at TEXT NOT NULL,
                    embedding BLOB
                );

                CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(space_slug, canonical_name);
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(space_slug, entity_type);

                CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                    id UNINDEXED,
                    canonical_name,
                    content,
                    space_slug UNINDEXED
                );

                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    space_id TEXT NOT NULL,
                    space_slug TEXT NOT NULL,
                    subject_entity_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_entity_id TEXT,
                    literal_value_json TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    retracted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
                CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_entity_id);

                PRAGMA user_version = 1;
                COMMIT;",
            )?;

            // Ensure default space exists
            let space_id = Uuid::new_v4().to_string();
            let now = Utc::now().to_rfc3339();
            let _ = conn.execute(
                "INSERT OR IGNORE INTO spaces (id, slug, created_at) VALUES (?1, ?2, ?3)",
                params![space_id, "atlas-memory", now],
            );
        }

        if user_version < 2 {
            conn.execute_batch(
                "BEGIN;
                CREATE TABLE IF NOT EXISTS sessions (
                    id                TEXT PRIMARY KEY,
                    space_slug        TEXT NOT NULL,
                    agent_id          TEXT,
                    world_id          TEXT,
                    parent_session_id TEXT,
                    fork_event_id     TEXT,
                    created_at        TEXT NOT NULL,
                    closed_at         TEXT,
                    status            TEXT NOT NULL DEFAULT 'active',
                    retention_class   TEXT NOT NULL DEFAULT 'standard',
                    metadata_json     TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_space ON sessions(space_slug);
                CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

                CREATE TABLE IF NOT EXISTS session_events (
                    id                TEXT PRIMARY KEY,
                    session_id        TEXT NOT NULL,
                    sequence          INTEGER NOT NULL,
                    branch_id         TEXT NOT NULL,
                    parent_event_id   TEXT,
                    event_type        TEXT NOT NULL,
                    role              TEXT,
                    content           TEXT,
                    payload_json      TEXT NOT NULL DEFAULT '{}',
                    created_at        TEXT NOT NULL,
                    content_hash      TEXT NOT NULL,
                    sensitivity       TEXT,
                    redacted          INTEGER NOT NULL DEFAULT 0,
                    segment_id        TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    UNIQUE(session_id, branch_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_session_events_order 
                    ON session_events(session_id, branch_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_session_events_type 
                    ON session_events(event_type);

                CREATE TABLE IF NOT EXISTS processing_watermarks (
                    session_id        TEXT NOT NULL,
                    processor_kind    TEXT NOT NULL,
                    processor_version TEXT NOT NULL,
                    watermark_seq     INTEGER NOT NULL,
                    updated_at        TEXT NOT NULL,
                    PRIMARY KEY(session_id, processor_kind, processor_version),
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS event_segments (
                    id                TEXT PRIMARY KEY,
                    session_id        TEXT NOT NULL,
                    branch_id         TEXT NOT NULL,
                    start_seq         INTEGER NOT NULL,
                    end_seq           INTEGER NOT NULL,
                    merkle_root       TEXT NOT NULL,
                    prev_segment_root TEXT,
                    sealed_at         TEXT NOT NULL,
                    state             TEXT NOT NULL DEFAULT 'sealed',
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_event_segments_range 
                    ON event_segments(session_id, branch_id, start_seq, end_seq);

                PRAGMA user_version = 2;
                COMMIT;",
            )?;
        }

        if user_version < 3 {
            conn.execute_batch(
                "BEGIN;
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id                TEXT PRIMARY KEY,
                    session_id        TEXT NOT NULL,
                    branch_id         TEXT NOT NULL,
                    level             INTEGER NOT NULL DEFAULT 0,
                    start_seq         INTEGER NOT NULL,
                    end_seq           INTEGER NOT NULL,
                    summary_text      TEXT NOT NULL,
                    source_hash       TEXT NOT NULL,
                    processor_version TEXT NOT NULL,
                    created_at        TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_session_summaries_range 
                    ON session_summaries(session_id, branch_id, level, start_seq, end_seq);

                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id                  TEXT PRIMARY KEY,
                    space_slug          TEXT NOT NULL,
                    session_id          TEXT NOT NULL,
                    branch_id           TEXT NOT NULL,
                    memory_type         TEXT NOT NULL,
                    subject             TEXT NOT NULL,
                    predicate           TEXT NOT NULL,
                    object_value_json   TEXT NOT NULL,
                    scope               TEXT NOT NULL DEFAULT 'global',
                    confidence          REAL NOT NULL,
                    verification_tier   TEXT NOT NULL DEFAULT 'T0Direct',
                    state               TEXT NOT NULL DEFAULT 'pending',
                    extractor_version   TEXT NOT NULL,
                    created_at          TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_memory_candidates_state 
                    ON memory_candidates(space_slug, state, verification_tier);
                CREATE INDEX IF NOT EXISTS idx_memory_candidates_session 
                    ON memory_candidates(session_id, branch_id);

                CREATE TABLE IF NOT EXISTS candidate_evidence (
                    candidate_id        TEXT NOT NULL,
                    evidence_id         TEXT NOT NULL,
                    source_kind         TEXT NOT NULL,
                    evidence_hash       TEXT NOT NULL,
                    PRIMARY KEY(candidate_id, evidence_id),
                    FOREIGN KEY(candidate_id) REFERENCES memory_candidates(id)
                );

                PRAGMA user_version = 3;
                COMMIT;",
            )?;
        }

        if user_version < 4 {
            conn.execute_batch(
                "BEGIN;
                CREATE TABLE IF NOT EXISTS promotion_receipts (
                    id                  TEXT PRIMARY KEY,
                    candidate_id        TEXT NOT NULL,
                    claim_id            TEXT NOT NULL,
                    policy_id           TEXT NOT NULL,
                    policy_version      TEXT NOT NULL,
                    achieved_tier       TEXT NOT NULL,
                    verifier_receipt    TEXT,
                    promoted_at         TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_promotion_receipts_claim 
                    ON promotion_receipts(claim_id);

                CREATE TABLE IF NOT EXISTS claim_evidence (
                    claim_id            TEXT NOT NULL,
                    evidence_id         TEXT NOT NULL,
                    source_kind         TEXT NOT NULL,
                    evidence_hash       TEXT NOT NULL,
                    PRIMARY KEY(claim_id, evidence_id)
                );

                CREATE INDEX IF NOT EXISTS idx_claim_evidence_ev 
                    ON claim_evidence(evidence_id);

                PRAGMA user_version = 4;
                COMMIT;",
            )?;
        }

        if user_version < 5 {
            conn.execute_batch(
                "BEGIN;
                CREATE TABLE IF NOT EXISTS security_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    hmac_key BLOB NOT NULL
                );
                INSERT INTO security_state (id, hmac_key)
                SELECT 1, randomblob(32)
                WHERE NOT EXISTS (SELECT 1 FROM security_state WHERE id = 1);

                CREATE TABLE IF NOT EXISTS evidence_records (
                    id TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    commitment TEXT NOT NULL,
                    merkle_root TEXT,
                    segment_id TEXT,
                    retention_class TEXT NOT NULL,
                    retain_until TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_source
                    ON evidence_records(source_kind, source_id);

                ALTER TABLE claims ADD COLUMN verification_tier TEXT NOT NULL DEFAULT 'T0Direct';
                ALTER TABLE claims ADD COLUMN status TEXT NOT NULL DEFAULT 'active';

                PRAGMA user_version = 5;
                COMMIT;",
            )?;
        }

        Ok(())
    }

    pub fn ensure_space(&self, slug: &str) -> Result<String> {
        {
            let conn = self.reader_pool.acquire()?;
            let mut stmt = conn.prepare("SELECT id FROM spaces WHERE slug = ?1")?;
            let mut rows = stmt.query(params![slug])?;
            if let Some(row) = rows.next()? {
                return row.get(0);
            }
        }
        let slug_owned = slug.to_string();
        self.writer
            .execute(move |conn| ensure_space_internal(conn, &slug_owned))
    }

    pub fn upsert_entity(
        &self,
        input: &EntityWriteInput,
        embedding: Option<&[f32]>,
    ) -> Result<ReceiptDetails> {
        let embedding_blob = embedding.map(embedding_to_bytes);
        let aliases_json =
            serde_json::to_string(&input.aliases).unwrap_or_else(|_| "[]".to_string());
        let metadata_json =
            serde_json::to_string(&input.metadata).unwrap_or_else(|_| "{}".to_string());
        let space = input.space.clone();
        let canonical_name = input.canonical_name.clone();
        let entity_type = input.entity_type.clone();
        let content = input.content.clone();
        let confidence = input.confidence;
        let valid_from = input.valid_from.clone();
        let valid_to = input.valid_to.clone();
        let input_id = input.id.clone();

        self.writer.execute(move |conn| {
            let space_id = ensure_space_internal(conn, &space)?;
            let now = Utc::now().to_rfc3339();

            // Check if entity exists by canonical_name in space
            let mut stmt = conn.prepare(
                "SELECT id, revision FROM entities WHERE space_slug = ?1 AND canonical_name = ?2",
            )?;
            let mut rows = stmt.query(params![space, canonical_name])?;

            let (target_id, revision) = if let Some(row) = rows.next()? {
                let existing_id: String = row.get(0)?;
                let rev: i64 = row.get(1)?;
                (existing_id, rev + 1)
            } else {
                let new_id = input_id.unwrap_or_else(|| Uuid::new_v4().to_string());
                (new_id, 1)
            };
            drop(rows);
            drop(stmt);

            conn.execute(
                "INSERT INTO entities (id, space_id, space_slug, entity_type, canonical_name, content, aliases_json, metadata_json, confidence, revision, retracted, valid_from, valid_to, created_at, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0, ?11, ?12, ?13, ?14)
                 ON CONFLICT(id) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    canonical_name = excluded.canonical_name,
                    content = excluded.content,
                    aliases_json = excluded.aliases_json,
                    metadata_json = excluded.metadata_json,
                    confidence = excluded.confidence,
                    revision = excluded.revision,
                    retracted = 0,
                    valid_from = excluded.valid_from,
                    valid_to = excluded.valid_to,
                    embedding = COALESCE(excluded.embedding, entities.embedding)",
                params![
                    target_id,
                    space_id,
                    space,
                    entity_type,
                    canonical_name,
                    content,
                    aliases_json,
                    metadata_json,
                    confidence,
                    revision,
                    valid_from,
                    valid_to,
                    now,
                    embedding_blob,
                ],
            )?;

            // Update FTS table
            let _ = conn.execute("DELETE FROM entities_fts WHERE id = ?1", params![target_id]);
            let _ = conn.execute(
                "INSERT INTO entities_fts (id, canonical_name, content, space_slug) VALUES (?1, ?2, ?3, ?4)",
                params![target_id, canonical_name, content, space],
            );

            Ok(ReceiptDetails {
                id: Uuid::new_v4().to_string(),
                operation: "entity.upsert".to_string(),
                target_type: "entity".to_string(),
                target_id,
                committed_at: now,
            })
        })
    }

    pub fn upsert_claim(&self, input: &ClaimWriteInput) -> Result<ReceiptDetails> {
        let space = input.space.clone();
        let claim_id = input
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let literal_json = input
            .literal_value
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default());
        let metadata_json =
            serde_json::to_string(&input.metadata).unwrap_or_else(|_| "{}".to_string());
        let subject_entity_id = input.subject_entity_id.clone();
        let predicate = input.predicate.clone();
        let object_entity_id = input.object_entity_id.clone();
        let confidence = input.confidence;

        self.writer.execute(move |conn| {
            let space_id = ensure_space_internal(conn, &space)?;
            let now = Utc::now().to_rfc3339();

            conn.execute(
                "INSERT INTO claims (id, space_id, space_slug, subject_entity_id, predicate, object_entity_id, literal_value_json, confidence, metadata_json, retracted, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, ?10)
                 ON CONFLICT(id) DO UPDATE SET
                    predicate = excluded.predicate,
                    object_entity_id = excluded.object_entity_id,
                    literal_value_json = excluded.literal_value_json,
                    confidence = excluded.confidence,
                    metadata_json = excluded.metadata_json,
                    retracted = 0",
                params![
                    claim_id,
                    space_id,
                    space,
                    subject_entity_id,
                    predicate,
                    object_entity_id,
                    literal_json,
                    confidence,
                    metadata_json,
                    now,
                ],
            )?;

            Ok(ReceiptDetails {
                id: Uuid::new_v4().to_string(),
                operation: "claim.upsert".to_string(),
                target_type: "claim".to_string(),
                target_id: claim_id,
                committed_at: now,
            })
        })
    }

    pub fn retract_target(&self, target_type: &str, target_id: &str) -> Result<ReceiptDetails> {
        let tt = target_type.to_string();
        let tid = target_id.to_string();
        self.writer.execute(move |conn| {
            let now = Utc::now().to_rfc3339();

            if tt == "entity" {
                conn.execute(
                    "UPDATE entities SET retracted = 1 WHERE id = ?1",
                    params![tid],
                )?;
                conn.execute("DELETE FROM entities_fts WHERE id = ?1", params![tid])?;
            } else if tt == "claim" {
                conn.execute(
                    "UPDATE claims SET retracted = 1 WHERE id = ?1",
                    params![tid],
                )?;
            }

            Ok(ReceiptDetails {
                id: Uuid::new_v4().to_string(),
                operation: format!("{}.retract", tt),
                target_type: tt,
                target_id: tid,
                committed_at: now,
            })
        })
    }

    pub fn get_entity(
        &self,
        id_or_name: &str,
        space: Option<&str>,
    ) -> Result<Option<CortexEntity>> {
        let conn = self.reader_pool.acquire()?;
        let target_space = space.unwrap_or("atlas-memory");

        let mut stmt = conn.prepare(
            "SELECT id, space_id, space_slug, entity_type, canonical_name, content, aliases_json, metadata_json, confidence, revision, retracted, valid_from, valid_to, created_at
             FROM entities
             WHERE (id = ?1 OR canonical_name = ?1) AND space_slug = ?2 AND retracted = 0
             LIMIT 1"
        )?;

        let mut rows = stmt.query(params![id_or_name, target_space])?;
        if let Some(row) = rows.next()? {
            let aliases_str: String = row.get(6)?;
            let meta_str: String = row.get(7)?;
            Ok(Some(CortexEntity {
                id: row.get(0)?,
                space_id: row.get(1)?,
                space_slug: row.get(2)?,
                entity_type: row.get(3)?,
                canonical_name: row.get(4)?,
                content: row.get(5)?,
                aliases: serde_json::from_str(&aliases_str).unwrap_or_default(),
                metadata: serde_json::from_str(&meta_str).unwrap_or_else(|_| serde_json::json!({})),
                confidence: row.get(8)?,
                revision: row.get(9)?,
                retracted: row.get::<_, i64>(10)? != 0,
                valid_from: row.get(11)?,
                valid_to: row.get(12)?,
                created_at: row.get(13)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn search_entities(
        &self,
        query: &str,
        space: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let conn = self.reader_pool.acquire()?;
        let target_space = space.unwrap_or("atlas-memory");
        let query_trimmed = query.trim();

        if query_trimmed.is_empty() {
            let mut stmt = conn.prepare(
                "SELECT id, space_id, space_slug, entity_type, canonical_name, content, aliases_json, metadata_json, confidence, revision, retracted, valid_from, valid_to, created_at
                 FROM entities
                 WHERE space_slug = ?1 AND retracted = 0
                 ORDER BY created_at DESC
                 LIMIT ?2"
            )?;
            let rows = stmt.query_map(params![target_space, limit], |row| {
                let aliases_str: String = row.get(6)?;
                let meta_str: String = row.get(7)?;
                Ok(SearchResult {
                    entity: CortexEntity {
                        id: row.get(0)?,
                        space_id: row.get(1)?,
                        space_slug: row.get(2)?,
                        entity_type: row.get(3)?,
                        canonical_name: row.get(4)?,
                        content: row.get(5)?,
                        aliases: serde_json::from_str(&aliases_str).unwrap_or_default(),
                        metadata: serde_json::from_str(&meta_str)
                            .unwrap_or_else(|_| serde_json::json!({})),
                        confidence: row.get(8)?,
                        revision: row.get(9)?,
                        retracted: row.get::<_, i64>(10)? != 0,
                        valid_from: row.get(11)?,
                        valid_to: row.get(12)?,
                        created_at: row.get(13)?,
                    },
                    lexical_score: 0.4,
                    graph_score: 0.0,
                    semantic_score: 0.0,
                    score: 0.4,
                })
            })?;

            let mut results = Vec::new();
            for r in rows {
                results.push(r?);
            }
            return Ok(results);
        }

        // FTS search with support for exact, prefix (*), and boolean operators (AND, OR, NOT)
        let clean_q = query_trimmed.replace('"', "");
        let has_fts_ops = clean_q.contains(" AND ")
            || clean_q.contains(" OR ")
            || clean_q.contains(" NOT ")
            || clean_q.contains('*');

        let mut results = Vec::new();

        let run_fts = |query_str: &str| -> Option<Vec<SearchResult>> {
            let mut stmt = conn.prepare(
                "SELECT e.id, e.space_id, e.space_slug, e.entity_type, e.canonical_name, e.content, e.aliases_json, e.metadata_json, e.confidence, e.revision, e.retracted, e.valid_from, e.valid_to, e.created_at,
                        COALESCE(bm25(entities_fts), 5.0) as bm25_rank
                 FROM entities e
                 JOIN entities_fts f ON f.id = e.id
                 WHERE entities_fts MATCH ?1 AND e.space_slug = ?2 AND e.retracted = 0
                 ORDER BY bm25_rank ASC
                 LIMIT ?3"
            ).ok()?;

            let rows = stmt
                .query_map(params![query_str, target_space, limit], |row| {
                    let aliases_str: String = row.get(6)?;
                    let meta_str: String = row.get(7)?;
                    let bm25_rank: f64 = row.get(14)?;
                    let lexical_score = 1.0 / (1.0 + bm25_rank.abs());
                    Ok(SearchResult {
                        entity: CortexEntity {
                            id: row.get(0)?,
                            space_id: row.get(1)?,
                            space_slug: row.get(2)?,
                            entity_type: row.get(3)?,
                            canonical_name: row.get(4)?,
                            content: row.get(5)?,
                            aliases: serde_json::from_str(&aliases_str).unwrap_or_default(),
                            metadata: serde_json::from_str(&meta_str)
                                .unwrap_or_else(|_| serde_json::json!({})),
                            confidence: row.get(8)?,
                            revision: row.get(9)?,
                            retracted: row.get::<_, i64>(10)? != 0,
                            valid_from: row.get(11)?,
                            valid_to: row.get(12)?,
                            created_at: row.get(13)?,
                        },
                        lexical_score,
                        graph_score: 0.0,
                        semantic_score: 0.0,
                        score: lexical_score,
                    })
                })
                .ok()?;

            let mut out = Vec::new();
            for r in rows.flatten() {
                out.push(r);
            }
            Some(out)
        };

        if has_fts_ops {
            if let Some(r) = run_fts(&clean_q) {
                results = r;
            }
        }

        if results.is_empty() {
            let fts_query = format!("\"{}\"", clean_q);
            if let Some(r) = run_fts(&fts_query) {
                results = r;
            }
        }

        if results.is_empty() {
            let terms: Vec<&str> = clean_q.split_whitespace().collect();
            if terms.len() > 1 {
                let and_query = terms.join(" AND ");
                if let Some(r) = run_fts(&and_query) {
                    results = r;
                }
            }
        }

        // Fallback to substring match if FTS did not return enough
        if results.len() < limit {
            let pattern = format!("%{}%", query_trimmed);
            let mut sub_stmt = conn.prepare(
                "SELECT id, space_id, space_slug, entity_type, canonical_name, content, aliases_json, metadata_json, confidence, revision, retracted, valid_from, valid_to, created_at
                 FROM entities
                 WHERE (canonical_name LIKE ?1 OR content LIKE ?1) AND space_slug = ?2 AND retracted = 0
                 LIMIT ?3"
            )?;
            let sub_rows = sub_stmt.query_map(params![pattern, target_space, limit], |row| {
                let aliases_str: String = row.get(6)?;
                let meta_str: String = row.get(7)?;
                Ok(SearchResult {
                    entity: CortexEntity {
                        id: row.get(0)?,
                        space_id: row.get(1)?,
                        space_slug: row.get(2)?,
                        entity_type: row.get(3)?,
                        canonical_name: row.get(4)?,
                        content: row.get(5)?,
                        aliases: serde_json::from_str(&aliases_str).unwrap_or_default(),
                        metadata: serde_json::from_str(&meta_str)
                            .unwrap_or_else(|_| serde_json::json!({})),
                        confidence: row.get(8)?,
                        revision: row.get(9)?,
                        retracted: row.get::<_, i64>(10)? != 0,
                        valid_from: row.get(11)?,
                        valid_to: row.get(12)?,
                        created_at: row.get(13)?,
                    },
                    lexical_score: 0.7,
                    graph_score: 0.0,
                    semantic_score: 0.0,
                    score: 0.7,
                })
            })?;
            for r in sub_rows {
                let sr = r?;
                if !results
                    .iter()
                    .any(|existing| existing.entity.id == sr.entity.id)
                {
                    results.push(sr);
                }
            }
        }

        results.truncate(limit);
        Ok(results)
    }

    pub fn recall_entities(
        &self,
        query_text: &str,
        query_embedding: Option<&[f32]>,
        space: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RecallItem>> {
        let conn = self.reader_pool.acquire()?;
        let target_space = space.unwrap_or("atlas-memory");

        let mut stmt = conn.prepare(
            "SELECT id, space_id, space_slug, entity_type, canonical_name, content, aliases_json, metadata_json, confidence, revision, retracted, valid_from, valid_to, created_at, embedding
             FROM entities
             WHERE space_slug = ?1 AND retracted = 0"
        )?;

        let rows = stmt.query_map(params![target_space], |row| {
            let aliases_str: String = row.get(6)?;
            let meta_str: String = row.get(7)?;
            let embedding_blob: Option<Vec<u8>> = row.get(14)?;
            let entity = CortexEntity {
                id: row.get(0)?,
                space_id: row.get(1)?,
                space_slug: row.get(2)?,
                entity_type: row.get(3)?,
                canonical_name: row.get(4)?,
                content: row.get(5)?,
                aliases: serde_json::from_str(&aliases_str).unwrap_or_default(),
                metadata: serde_json::from_str(&meta_str).unwrap_or_else(|_| serde_json::json!({})),
                confidence: row.get(8)?,
                revision: row.get(9)?,
                retracted: row.get::<_, i64>(10)? != 0,
                valid_from: row.get(11)?,
                valid_to: row.get(12)?,
                created_at: row.get(13)?,
            };
            Ok((entity, embedding_blob))
        })?;

        let mut candidates = Vec::new();
        let query_lower = query_text.to_lowercase();

        for r in rows {
            let (entity, emb_blob) = r?;
            let mut semantic_score = 0.0f64;

            if let (Some(q_emb), Some(blob)) = (query_embedding, emb_blob) {
                let target_emb = bytes_to_embedding(&blob);
                semantic_score = cosine_similarity(q_emb, &target_emb) as f64;
            }

            // Calculate simple lexical score
            let name_lower = entity.canonical_name.to_lowercase();
            let content_lower = entity.content.to_lowercase();
            let mut lexical_score = 0.0f64;
            if name_lower.contains(&query_lower) {
                lexical_score = 0.9;
            } else if content_lower.contains(&query_lower) {
                lexical_score = 0.6;
            }

            let final_score = if query_embedding.is_some() {
                0.6 * semantic_score + 0.4 * lexical_score
            } else {
                lexical_score
            };

            candidates.push(RecallItem {
                entity,
                final_score,
                component_scores: RecallComponentScores {
                    lexical: lexical_score,
                    semantic: semantic_score,
                    confidence_recency: 0.95,
                    graph: 0.0,
                },
            });
        }

        candidates.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        Ok(candidates)
    }

    pub fn traverse_claims(
        &self,
        subject_id: &str,
        space: Option<&str>,
    ) -> Result<Vec<CortexClaim>> {
        let conn = self.reader_pool.acquire()?;
        let target_space = space.unwrap_or("atlas-memory");

        let mut stmt = conn.prepare(
            "SELECT id, space_id, space_slug, subject_entity_id, predicate, object_entity_id, literal_value_json, confidence, metadata_json, retracted, created_at, verification_tier, status, (SELECT COUNT(*) FROM claim_evidence WHERE claim_id = claims.id)
             FROM claims
             WHERE (
                 subject_entity_id = ?1 
                 OR object_entity_id = ?1
                 OR subject_entity_id IN (SELECT id FROM entities WHERE canonical_name = ?1 AND space_slug = ?2)
                 OR object_entity_id IN (SELECT id FROM entities WHERE canonical_name = ?1 AND space_slug = ?2)
             ) AND space_slug = ?2 AND retracted = 0"
        )?;

        let rows = stmt.query_map(params![subject_id, target_space], |row| {
            let lit_str: Option<String> = row.get(6)?;
            let meta_str: String = row.get(8)?;
            let status_str: String = row.get(12)?;
            let tier_str: String = row.get(11)?;
            let status = match status_str.as_str() {
                "superseded" => ClaimStatus::Superseded,
                "disputed" => ClaimStatus::Disputed,
                "retracted" => ClaimStatus::Retracted,
                "invalidated" => ClaimStatus::Invalidated,
                _ => {
                    if row.get::<_, i64>(9)? != 0 {
                        ClaimStatus::Retracted
                    } else {
                        ClaimStatus::Active
                    }
                }
            };
            Ok(CortexClaim {
                id: row.get(0)?,
                space_id: row.get(1)?,
                space_slug: row.get(2)?,
                subject_entity_id: row.get(3)?,
                predicate: row.get(4)?,
                object_entity_id: row.get(5)?,
                literal_value: lit_str.and_then(|s| serde_json::from_str(&s).ok()),
                confidence: row.get(7)?,
                metadata: serde_json::from_str(&meta_str).unwrap_or_else(|_| serde_json::json!({})),
                retracted: row.get::<_, i64>(9)? != 0,
                created_at: row.get(10)?,
                verification_tier: VerificationTier::from_str_opt(&tier_str),
                status,
                evidence_count: row.get::<_, i64>(13)? as usize,
            })
        })?;

        let mut claims = Vec::new();
        for r in rows {
            claims.push(r?);
        }
        Ok(claims)
    }

    pub fn import_entity_raw(
        &self,
        entity: &CortexEntity,
        embedding: Option<&[f32]>,
    ) -> Result<()> {
        let embedding_blob = embedding.map(embedding_to_bytes);
        let aliases_json =
            serde_json::to_string(&entity.aliases).unwrap_or_else(|_| "[]".to_string());
        let metadata_json =
            serde_json::to_string(&entity.metadata).unwrap_or_else(|_| "{}".to_string());
        let entity = entity.clone();

        self.writer.execute(move |conn| {
            let _ = ensure_space_internal(conn, &entity.space_slug)?;
            conn.execute(
                "INSERT INTO entities (id, space_id, space_slug, entity_type, canonical_name, content, aliases_json, metadata_json, confidence, revision, retracted, valid_from, valid_to, created_at, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
                 ON CONFLICT(id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    content = excluded.content,
                    aliases_json = excluded.aliases_json,
                    metadata_json = excluded.metadata_json,
                    confidence = excluded.confidence,
                    revision = excluded.revision,
                    retracted = excluded.retracted,
                    created_at = excluded.created_at,
                    embedding = COALESCE(excluded.embedding, entities.embedding)",
                params![
                    entity.id,
                    entity.space_id,
                    entity.space_slug,
                    entity.entity_type,
                    entity.canonical_name,
                    entity.content,
                    aliases_json,
                    metadata_json,
                    entity.confidence,
                    entity.revision,
                    if entity.retracted { 1 } else { 0 },
                    entity.valid_from,
                    entity.valid_to,
                    entity.created_at,
                    embedding_blob,
                ],
            )?;

            let _ = conn.execute("DELETE FROM entities_fts WHERE id = ?1", params![entity.id]);
            if !entity.retracted {
                let _ = conn.execute(
                    "INSERT INTO entities_fts (id, canonical_name, content, space_slug) VALUES (?1, ?2, ?3, ?4)",
                    params![entity.id, entity.canonical_name, entity.content, entity.space_slug],
                );
            }
            Ok(())
        })
    }

    // =========================================================================
    // Episodic Plane Methods (Phase 1 Substrate)
    // =========================================================================

    pub fn create_session(&self, input: &CreateSessionInput) -> Result<CortexSession> {
        let id = input
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let now = Utc::now().to_rfc3339();
        let metadata_json =
            serde_json::to_string(&input.metadata).unwrap_or_else(|_| "{}".to_string());
        let session = CortexSession {
            id: id.clone(),
            space_slug: input.space.clone(),
            agent_id: input.agent_id.clone(),
            world_id: input.world_id.clone(),
            parent_session_id: input.parent_session_id.clone(),
            fork_event_id: input.fork_event_id.clone(),
            created_at: now.clone(),
            closed_at: None,
            status: "active".to_string(),
            retention_class: input.retention_class.clone(),
            metadata: input.metadata.clone(),
        };
        let input_clone = input.clone();
        self.writer.execute(move |conn| {
            let _ = ensure_space_internal(conn, &input_clone.space)?;
            conn.execute(
                "INSERT INTO sessions (id, space_slug, agent_id, world_id, parent_session_id, fork_event_id, created_at, closed_at, status, retention_class, metadata_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, 'active', ?8, ?9)",
                params![
                    id,
                    input_clone.space,
                    input_clone.agent_id,
                    input_clone.world_id,
                    input_clone.parent_session_id,
                    input_clone.fork_event_id,
                    now,
                    input_clone.retention_class,
                    metadata_json,
                ],
            )?;
            Ok(())
        })?;
        Ok(session)
    }

    pub fn get_session(&self, session_id: &str) -> Result<Option<CortexSession>> {
        let conn = self.reader_pool.acquire()?;
        let mut stmt = conn.prepare(
            "SELECT id, space_slug, agent_id, world_id, parent_session_id, fork_event_id, created_at, closed_at, status, retention_class, metadata_json
             FROM sessions WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![session_id])?;
        if let Some(row) = rows.next()? {
            let metadata_str: String = row.get(10)?;
            let metadata: serde_json::Value =
                serde_json::from_str(&metadata_str).unwrap_or_else(|_| serde_json::json!({}));
            Ok(Some(CortexSession {
                id: row.get(0)?,
                space_slug: row.get(1)?,
                agent_id: row.get(2)?,
                world_id: row.get(3)?,
                parent_session_id: row.get(4)?,
                fork_event_id: row.get(5)?,
                created_at: row.get(6)?,
                closed_at: row.get(7)?,
                status: row.get(8)?,
                retention_class: row.get(9)?,
                metadata,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn close_session(&self, session_id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let sid = session_id.to_string();
        self.writer.execute(move |conn| {
            conn.execute(
                "UPDATE sessions SET closed_at = ?1, status = 'closed' WHERE id = ?2",
                params![now, sid],
            )?;
            Ok(())
        })
    }

    pub fn append_session_event(
        &self,
        session_id: &str,
        input: &SessionEventInput,
    ) -> Result<CortexSessionEvent> {
        let branch = input.branch_id.as_deref().unwrap_or("main");
        let results = self.batch_append_events(session_id, branch, std::slice::from_ref(input))?;
        results.into_iter().next().ok_or_else(|| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_INTERNAL),
                Some("Failed to append event".to_string()),
            )
        })
    }

    pub fn batch_append_events(
        &self,
        session_id: &str,
        branch_id: &str,
        events: &[SessionEventInput],
    ) -> Result<Vec<CortexSessionEvent>> {
        let sid = session_id.to_string();
        let bid = branch_id.to_string();
        let input_events = events.to_vec();
        let hmac_key = self.hmac_key.clone();

        self.writer.execute(move |conn| {
            let tx = conn.transaction()?;

            // Verify session exists
            let session_exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = ?1)",
                params![sid],
                |r| r.get(0),
            )?;
            if !session_exists {
                return Err(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_NOTFOUND),
                    Some(format!("Session {} not found", sid)),
                ));
            }

            // Get current max sequence for (session_id, branch_id)
            let mut current_seq: i64 = tx.query_row(
                "SELECT COALESCE(MAX(sequence), 0) FROM session_events WHERE session_id = ?1 AND branch_id = ?2",
                params![sid, bid],
                |r| r.get(0),
            )?;

            let mut inserted = Vec::with_capacity(input_events.len());
            let now = Utc::now().to_rfc3339();

            for ev in input_events {
                let event_branch = ev.branch_id.as_deref().unwrap_or(&bid).to_string();

                // Idempotency check: if id is provided, check if event already exists
                if let Some(existing_id) = &ev.id {
                    let mut check_stmt = tx.prepare(
                        "SELECT id, session_id, sequence, branch_id, parent_event_id, event_type, role, content, payload_json, created_at, content_hash, sensitivity, redacted, segment_id
                         FROM session_events WHERE id = ?1"
                    )?;
                    let mut rows = check_stmt.query(params![existing_id])?;
                    if let Some(row) = rows.next()? {
                        let payload_str: String = row.get(8)?;
                        let payload: serde_json::Value = serde_json::from_str(&payload_str).unwrap_or_else(|_| serde_json::json!({}));
                        let existing_ev = CortexSessionEvent {
                            id: row.get(0)?,
                            session_id: row.get(1)?,
                            sequence: row.get(2)?,
                            branch_id: row.get(3)?,
                            parent_event_id: row.get(4)?,
                            event_type: row.get(5)?,
                            role: row.get(6)?,
                            content: row.get(7)?,
                            payload,
                            created_at: row.get(9)?,
                            content_hash: row.get(10)?,
                            sensitivity: row.get(11)?,
                            redacted: row.get::<_, i64>(12)? != 0,
                            segment_id: row.get(13)?,
                        };
                        inserted.push(existing_ev);
                        continue;
                    }
                }

                current_seq += 1;
                let event_id = ev.id.unwrap_or_else(|| Uuid::new_v4().to_string());
                let payload_json = serde_json::to_string(&ev.payload).unwrap_or_else(|_| "{}".to_string());
                let raw_payload: serde_json::Value = serde_json::from_str(&payload_json).unwrap_or_else(|_| serde_json::json!({}));

                let sanitization = crate::security::SecurityMembrane::inspect_and_sanitize(
                    ev.content.as_deref(),
                    &raw_payload,
                    ev.sensitivity.as_deref(),
                    &hmac_key,
                );
                let final_content = sanitization.sanitized_content;
                let final_sensitivity = sanitization.sensitivity;
                let final_payload = sanitization.sanitized_payload;
                let redacted = sanitization.redacted;
                let payload_json = serde_json::to_string(&final_payload).unwrap_or_else(|_| "{}".to_string());

                let content_hash = compute_event_hash(
                    &sid,
                    &event_branch,
                    current_seq,
                    &ev.event_type,
                    ev.role.as_deref(),
                    final_content.as_deref(),
                    &payload_json,
                );

                tx.execute(
                    "INSERT INTO session_events (
                         id, session_id, sequence, branch_id, parent_event_id,
                         event_type, role, content, payload_json, created_at,
                         content_hash, sensitivity, redacted, segment_id
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, NULL)",
                    params![
                        event_id,
                        sid,
                        current_seq,
                        event_branch,
                        ev.parent_event_id,
                        ev.event_type,
                        ev.role,
                        final_content,
                        payload_json,
                        now,
                        content_hash,
                        final_sensitivity,
                        redacted as i64,
                    ],
                )?;

                inserted.push(CortexSessionEvent {
                    id: event_id,
                    session_id: sid.clone(),
                    sequence: current_seq,
                    branch_id: event_branch,
                    parent_event_id: ev.parent_event_id,
                    event_type: ev.event_type,
                    role: ev.role,
                    content: final_content,
                    payload: final_payload,
                    created_at: now.clone(),
                    content_hash,
                    sensitivity: final_sensitivity,
                    redacted,
                    segment_id: None,
                });
            }

            tx.commit()?;
            Ok(inserted)
        })
    }

    pub fn get_session_events(
        &self,
        session_id: &str,
        branch_id: &str,
        after_seq: Option<i64>,
        limit: usize,
    ) -> Result<Vec<CortexSessionEvent>> {
        let conn = self.reader_pool.acquire()?;
        let after = after_seq.unwrap_or(0);
        let lim = limit.min(500);

        let mut stmt = conn.prepare(
            "SELECT id, session_id, sequence, branch_id, parent_event_id, event_type, role, content, payload_json, created_at, content_hash, sensitivity, redacted, segment_id
             FROM session_events
             WHERE session_id = ?1 AND branch_id = ?2 AND sequence > ?3
             ORDER BY sequence ASC
             LIMIT ?4"
        )?;

        let rows = stmt.query_map(params![session_id, branch_id, after, lim as i64], |row| {
            let payload_str: String = row.get(8)?;
            let payload: serde_json::Value =
                serde_json::from_str(&payload_str).unwrap_or_else(|_| serde_json::json!({}));
            Ok(CortexSessionEvent {
                id: row.get(0)?,
                session_id: row.get(1)?,
                sequence: row.get(2)?,
                branch_id: row.get(3)?,
                parent_event_id: row.get(4)?,
                event_type: row.get(5)?,
                role: row.get(6)?,
                content: row.get(7)?,
                payload,
                created_at: row.get(9)?,
                content_hash: row.get(10)?,
                sensitivity: row.get(11)?,
                redacted: row.get::<_, i64>(12)? != 0,
                segment_id: row.get(13)?,
            })
        })?;

        let mut events = Vec::new();
        for r in rows {
            events.push(r?);
        }
        Ok(events)
    }

    pub fn get_recent_session_events(
        &self,
        session_id: &str,
        branch_id: &str,
        limit: usize,
    ) -> Result<Vec<CortexSessionEvent>> {
        let conn = self.reader_pool.acquire()?;
        let lim = limit.min(500);
        let mut stmt = conn.prepare(
            "SELECT id, session_id, sequence, branch_id, parent_event_id, event_type, role, content, payload_json, created_at, content_hash, sensitivity, redacted, segment_id
             FROM (
                 SELECT id, session_id, sequence, branch_id, parent_event_id, event_type, role, content, payload_json, created_at, content_hash, sensitivity, redacted, segment_id
                 FROM session_events
                 WHERE session_id = ?1 AND branch_id = ?2
                 ORDER BY sequence DESC
                 LIMIT ?3
             ) AS recent_events
             ORDER BY sequence ASC"
        )?;
        let rows = stmt.query_map(params![session_id, branch_id, lim as i64], |row| {
            let payload_str: String = row.get(8)?;
            let payload: serde_json::Value =
                serde_json::from_str(&payload_str).unwrap_or_else(|_| serde_json::json!({}));
            Ok(CortexSessionEvent {
                id: row.get(0)?,
                session_id: row.get(1)?,
                sequence: row.get(2)?,
                branch_id: row.get(3)?,
                parent_event_id: row.get(4)?,
                event_type: row.get(5)?,
                role: row.get(6)?,
                content: row.get(7)?,
                payload,
                created_at: row.get(9)?,
                content_hash: row.get(10)?,
                sensitivity: row.get(11)?,
                redacted: row.get::<_, i64>(12)? != 0,
                segment_id: row.get(13)?,
            })
        })?;
        let mut events = Vec::new();
        for r in rows {
            events.push(r?);
        }
        Ok(events)
    }

    pub fn get_watermark(
        &self,
        session_id: &str,
        kind: &str,
        version: &str,
    ) -> Result<Option<i64>> {
        let conn = self.reader_pool.acquire()?;
        let mut stmt = conn.prepare(
            "SELECT watermark_seq FROM processing_watermarks WHERE session_id = ?1 AND processor_kind = ?2 AND processor_version = ?3"
        )?;
        let mut rows = stmt.query(params![session_id, kind, version])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn set_watermark(
        &self,
        session_id: &str,
        kind: &str,
        version: &str,
        seq: i64,
    ) -> Result<()> {
        let sid = session_id.to_string();
        let k = kind.to_string();
        let v = version.to_string();
        let now = Utc::now().to_rfc3339();

        self.writer.execute(move |conn| {
            conn.execute(
                "INSERT INTO processing_watermarks (session_id, processor_kind, processor_version, watermark_seq, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(session_id, processor_kind, processor_version) DO UPDATE SET
                     watermark_seq = excluded.watermark_seq,
                     updated_at = excluded.updated_at",
                params![sid, k, v, seq, now],
            )?;
            Ok(())
        })
    }

    // =========================================================================
    // Summaries, Candidates & Provenance (Phases 2 & 3)
    // =========================================================================

    pub fn insert_session_summary(&self, input: &SessionSummaryWrite) -> Result<SessionSummary> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let sid = input.session_id.to_string();
        let bid = input.branch_id.to_string();
        let level = input.level;
        let start_seq = input.start_seq;
        let end_seq = input.end_seq;
        let text = input.summary_text.to_string();
        let hash = input.source_hash.to_string();
        let ver = input.processor_version.to_string();

        let summary = SessionSummary {
            id: id.clone(),
            session_id: sid.clone(),
            branch_id: bid.clone(),
            level,
            start_seq,
            end_seq,
            summary_text: text.clone(),
            source_hash: hash.clone(),
            processor_version: ver.clone(),
            created_at: now.clone(),
        };

        self.writer.execute(move |conn| {
            conn.execute(
                "INSERT INTO session_summaries (
                     id, session_id, branch_id, level, start_seq, end_seq,
                     summary_text, source_hash, processor_version, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![id, sid, bid, level, start_seq, end_seq, text, hash, ver, now],
            )?;
            Ok(())
        })?;

        Ok(summary)
    }

    pub fn get_session_summaries(
        &self,
        session_id: &str,
        branch_id: &str,
        max_level: Option<i64>,
    ) -> Result<Vec<SessionSummary>> {
        let conn = self.reader_pool.acquire()?;
        let ml = max_level.unwrap_or(10);
        let mut stmt = conn.prepare(
            "SELECT id, session_id, branch_id, level, start_seq, end_seq, summary_text, source_hash, processor_version, created_at
             FROM session_summaries
             WHERE session_id = ?1 AND branch_id = ?2 AND level <= ?3
             ORDER BY level ASC, start_seq ASC"
        )?;

        let rows = stmt.query_map(params![session_id, branch_id, ml], |row| {
            Ok(SessionSummary {
                id: row.get(0)?,
                session_id: row.get(1)?,
                branch_id: row.get(2)?,
                level: row.get(3)?,
                start_seq: row.get(4)?,
                end_seq: row.get(5)?,
                summary_text: row.get(6)?,
                source_hash: row.get(7)?,
                processor_version: row.get(8)?,
                created_at: row.get(9)?,
            })
        })?;

        let mut summaries = Vec::new();
        for r in rows {
            summaries.push(r?);
        }
        Ok(summaries)
    }

    pub fn insert_memory_candidate(&self, input: &CandidateWriteInput) -> Result<MemoryCandidate> {
        let id = input
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let now = Utc::now().to_rfc3339();
        let tier = input
            .verification_tier
            .unwrap_or(VerificationTier::T0Direct);
        let tier_str = tier.as_str().to_string();
        let obj_json =
            serde_json::to_string(&input.object_value).unwrap_or_else(|_| "{}".to_string());
        let input_clone = input.clone();
        let id_clone = id.clone();
        let now_clone = now.clone();

        self.writer.execute(move |conn| {
            let tx = conn.transaction()?;
            let _ = ensure_space_internal(&tx, &input_clone.space)?;

            tx.execute(
                "INSERT INTO memory_candidates (
                     id, space_slug, session_id, branch_id, memory_type,
                     subject, predicate, object_value_json, scope, confidence,
                     verification_tier, state, extractor_version, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'pending', ?12, ?13)",
                params![
                    id_clone,
                    input_clone.space,
                    input_clone.session_id,
                    input_clone.branch_id,
                    input_clone.memory_type,
                    input_clone.subject,
                    input_clone.predicate,
                    obj_json,
                    input_clone.scope,
                    input_clone.confidence,
                    tier_str,
                    input_clone.extractor_version,
                    now_clone,
                ],
            )?;

            for ev_id in &input_clone.evidence_ids {
                let commitment: String = tx
                    .query_row(
                        "SELECT content_hash FROM session_events WHERE id = ?1",
                        params![ev_id],
                        |row| row.get(0),
                    )
                    .map_err(|_| {
                        rusqlite::Error::SqliteFailure(
                            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
                            Some(format!(
                                "evidence {} has no session event commitment",
                                ev_id
                            )),
                        )
                    })?;
                if !crate::evidence::is_commitment(&commitment) {
                    return Err(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
                        Some(format!("evidence {} commitment is not a digest", ev_id)),
                    ));
                }
                tx.execute(
                    "INSERT OR IGNORE INTO candidate_evidence (candidate_id, evidence_id, source_kind, evidence_hash)
                     VALUES (?1, ?2, 'session_event', ?3)",
                    params![id_clone, ev_id, commitment],
                )?;
                let record_id = Uuid::new_v4().to_string();
                tx.execute(
                    "INSERT INTO evidence_records (id, source_kind, source_id, commitment, merkle_root, segment_id, retention_class, retain_until, created_at)
                     VALUES (?1, 'session_event', ?2, ?3, NULL, NULL, 'standard', ?4, ?5)",
                    params![
                        record_id,
                        ev_id,
                        commitment,
                        crate::evidence::retain_until("standard", Utc::now()),
                        now_clone
                    ],
                )?;
            }

            tx.commit()?;
            Ok(())
        })?;

        Ok(MemoryCandidate {
            id,
            space_slug: input.space.clone(),
            session_id: input.session_id.clone(),
            branch_id: input.branch_id.clone(),
            memory_type: input.memory_type.clone(),
            subject: input.subject.clone(),
            predicate: input.predicate.clone(),
            object_value: input.object_value.clone(),
            scope: input.scope.clone(),
            confidence: input.confidence,
            verification_tier: tier,
            state: "pending".to_string(),
            extractor_version: input.extractor_version.clone(),
            created_at: now,
            evidence_ids: input.evidence_ids.clone(),
        })
    }

    pub fn get_memory_candidates(
        &self,
        space: Option<&str>,
        state: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryCandidate>> {
        let conn = self.reader_pool.acquire()?;
        let target_space = space.unwrap_or("atlas-memory");
        let lim = limit.min(100);

        let query_sql = if state.is_some() {
            "SELECT id, space_slug, session_id, branch_id, memory_type, subject, predicate, object_value_json, scope, confidence, verification_tier, state, extractor_version, created_at
             FROM memory_candidates
             WHERE space_slug = ?1 AND state = ?2
             ORDER BY created_at DESC LIMIT ?3"
        } else {
            "SELECT id, space_slug, session_id, branch_id, memory_type, subject, predicate, object_value_json, scope, confidence, verification_tier, state, extractor_version, created_at
             FROM memory_candidates
             WHERE space_slug = ?1
             ORDER BY created_at DESC LIMIT ?3"
        };

        let mut stmt = conn.prepare(query_sql)?;
        let mut candidates = Vec::new();

        let mut rows = if let Some(st) = state {
            stmt.query(params![target_space, st, lim as i64])?
        } else {
            stmt.query(params![target_space, lim as i64])?
        };

        while let Some(row) = rows.next()? {
            let cid: String = row.get(0)?;
            let obj_str: String = row.get(7)?;
            let obj_val: serde_json::Value =
                serde_json::from_str(&obj_str).unwrap_or_else(|_| serde_json::json!({}));
            let tier_str: String = row.get(10)?;
            let tier = VerificationTier::from_str_opt(&tier_str);

            // Fetch evidence IDs
            let mut ev_stmt =
                conn.prepare("SELECT evidence_id FROM candidate_evidence WHERE candidate_id = ?1")?;
            let ev_rows = ev_stmt.query_map(params![cid], |r| r.get(0))?;
            let mut evidence_ids = Vec::new();
            for e in ev_rows {
                evidence_ids.push(e?);
            }

            candidates.push(MemoryCandidate {
                id: cid,
                space_slug: row.get(1)?,
                session_id: row.get(2)?,
                branch_id: row.get(3)?,
                memory_type: row.get(4)?,
                subject: row.get(5)?,
                predicate: row.get(6)?,
                object_value: obj_val,
                scope: row.get(8)?,
                confidence: row.get(9)?,
                verification_tier: tier,
                state: row.get(11)?,
                extractor_version: row.get(12)?,
                created_at: row.get(13)?,
                evidence_ids,
            });
        }

        Ok(candidates)
    }

    pub fn get_candidate_by_id(&self, candidate_id: &str) -> Result<Option<MemoryCandidate>> {
        let conn = self.reader_pool.acquire()?;
        let mut stmt = conn.prepare(
            "SELECT id, space_slug, session_id, branch_id, memory_type, subject, predicate, object_value_json, scope, confidence, verification_tier, state, extractor_version, created_at
             FROM memory_candidates
             WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![candidate_id])?;
        if let Some(row) = rows.next()? {
            let cid: String = row.get(0)?;
            let obj_str: String = row.get(7)?;
            let obj_val: serde_json::Value =
                serde_json::from_str(&obj_str).unwrap_or_else(|_| serde_json::json!({}));
            let tier_str: String = row.get(10)?;
            let tier = VerificationTier::from_str_opt(&tier_str);

            let mut ev_stmt =
                conn.prepare("SELECT evidence_id FROM candidate_evidence WHERE candidate_id = ?1")?;
            let ev_rows = ev_stmt.query_map(params![cid], |r| r.get(0))?;
            let mut evidence_ids = Vec::new();
            for e in ev_rows {
                evidence_ids.push(e?);
            }

            Ok(Some(MemoryCandidate {
                id: cid,
                space_slug: row.get(1)?,
                session_id: row.get(2)?,
                branch_id: row.get(3)?,
                memory_type: row.get(4)?,
                subject: row.get(5)?,
                predicate: row.get(6)?,
                object_value: obj_val,
                scope: row.get(8)?,
                confidence: row.get(9)?,
                verification_tier: tier,
                state: row.get(11)?,
                extractor_version: row.get(12)?,
                created_at: row.get(13)?,
                evidence_ids,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_candidate_state(&self, candidate_id: &str, new_state: &str) -> Result<()> {
        let cid = candidate_id.to_string();
        let st = new_state.to_string();
        self.writer.execute(move |conn| {
            conn.execute(
                "UPDATE memory_candidates SET state = ?1 WHERE id = ?2",
                params![st, cid],
            )?;
            Ok(())
        })
    }

    pub fn promote_candidate(&self, input: &PromoteCandidateInput) -> Result<PromotionReceipt> {
        let cid = input.candidate_id.clone();
        let policy_id = input
            .policy_id
            .clone()
            .unwrap_or_else(|| "default-policy-v1".to_string());
        let verifier_receipt = input.verifier_receipt.clone();

        self.writer.execute(move |conn| {
            let tx = conn.transaction()?;

            // 1. Fetch candidate
            let mut stmt = tx.prepare(
                "SELECT id, space_slug, memory_type, subject, predicate, object_value_json, scope, confidence, verification_tier, state
                 FROM memory_candidates WHERE id = ?1"
            )?;
            let mut rows = stmt.query(params![cid])?;
            let (space_slug, mem_type, subject, predicate, obj_json, scope, confidence, tier_str, state): (
                String, String, String, String, String, String, f64, String, String
            ) = if let Some(r) = rows.next()? {
                (
                    r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?,
                    r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?
                )
            } else {
                return Err(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_NOTFOUND),
                    Some(format!("Candidate {} not found", cid)),
                ));
            };
            drop(rows);
            drop(stmt);

            if state == "promoted" {
                return Err(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
                    Some("Candidate already promoted".to_string()),
                ));
            }

            let (evidence_ids, evidence_hashes): (Vec<String>, Vec<String>) = {
                let mut ev_stmt = tx.prepare(
                    "SELECT evidence_id, evidence_hash FROM candidate_evidence WHERE candidate_id = ?1",
                )?;
                let rows = ev_stmt.query_map(params![cid], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                let mut ids = Vec::new();
                let mut hashes = Vec::new();
                for row in rows {
                    let (id, hash) = row?;
                    ids.push(id);
                    hashes.push(hash);
                }
                (ids, hashes)
            };
            if evidence_hashes
                .iter()
                .any(|hash| !crate::evidence::is_commitment(hash))
            {
                return Err(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
                    Some("promotion refused: evidence commitment missing".to_string()),
                ));
            }
            let tier = VerificationTier::from_str_opt(&tier_str);
            let policy_candidate = MemoryCandidate {
                id: cid.clone(),
                space_slug: space_slug.clone(),
                session_id: String::new(),
                branch_id: String::new(),
                memory_type: mem_type.clone(),
                subject: subject.clone(),
                predicate: predicate.clone(),
                object_value: serde_json::from_str(&obj_json).unwrap_or_else(|_| serde_json::json!({})),
                scope: scope.clone(),
                confidence,
                verification_tier: tier,
                state: state.clone(),
                extractor_version: String::new(),
                created_at: String::new(),
                evidence_ids,
            };
            if let Err(reason) = crate::promotion::enforce_promotion_policy(&policy_candidate) {
                return Err(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
                    Some(reason),
                ));
            }

            let now = Utc::now().to_rfc3339();
            let space_id = ensure_space_internal(&tx, &space_slug)?;

            // 2. Resolve subject entity (find existing by canonical_name in space, or insert entity shell)
            let subject_entity_id: String = {
                let mut ent_stmt = tx.prepare(
                    "SELECT id FROM entities WHERE space_slug = ?1 AND canonical_name = ?2",
                )?;
                let mut ent_rows = ent_stmt.query(params![space_slug, subject])?;
                if let Some(r) = ent_rows.next()? {
                    r.get(0)?
                } else {
                    drop(ent_rows);
                    drop(ent_stmt);
                    let new_ent_id = Uuid::new_v4().to_string();
                    tx.execute(
                        "INSERT INTO entities (id, space_id, space_slug, entity_type, canonical_name, content, aliases_json, metadata_json, confidence, revision, retracted, valid_from, valid_to, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, '[]', '{}', 1.0, 1, 0, ?7, NULL, ?7)",
                        params![
                            new_ent_id,
                            space_id,
                            space_slug,
                            mem_type,
                            subject,
                            format!("Canonical entity for {}", subject),
                            now
                        ],
                    )?;
                    tx.execute(
                        "INSERT INTO entities_fts (id, canonical_name, content, space_slug) VALUES (?1, ?2, ?3, ?4)",
                        params![new_ent_id, subject, subject, space_slug],
                    )?;
                    new_ent_id
                }
            };

            // 3. Contradiction & Temporal Supersession:
            // Check if active claim exists with same (subject, predicate)
            {
                let mut claim_check = tx.prepare(
                    "SELECT id FROM claims WHERE subject_entity_id = ?1 AND predicate = ?2 AND retracted = 0",
                )?;
                let mut existing_claim_rows =
                    claim_check.query(params![subject_entity_id, predicate])?;
                if let Some(r) = existing_claim_rows.next()? {
                    let old_claim_id: String = r.get(0)?;
                    drop(existing_claim_rows);
                    drop(claim_check);
                    // Temporal supersession: close old claim
                    tx.execute(
                        "UPDATE claims SET retracted = 1, status = 'superseded' WHERE id = ?1",
                        params![old_claim_id],
                    )?;
                }
            }

            // 4. Insert new promoted canonical claim
            let claim_id = Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO claims (id, space_id, space_slug, subject_entity_id, predicate, object_entity_id, literal_value_json, confidence, metadata_json, retracted, created_at, verification_tier, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?7, ?8, 0, ?9, ?10, 'active')",
                params![
                    claim_id,
                    space_id,
                    space_slug,
                    subject_entity_id,
                    predicate,
                    obj_json,
                    confidence,
                    serde_json::json!({"scope": scope, "promoted_from_candidate": cid, "tier": tier.as_str()}).to_string(),
                    now,
                    tier.as_str(),
                ],
            )?;

            // 5. Transfer evidence to claim_evidence
            let evidence_items: Vec<(String, String, String)> = {
                let mut ev_query = tx.prepare(
                    "SELECT evidence_id, source_kind, evidence_hash FROM candidate_evidence WHERE candidate_id = ?1",
                )?;
                let ev_rows = ev_query.query_map(params![cid], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                })?;
                let mut items = Vec::new();
                for e in ev_rows {
                    items.push(e?);
                }
                items
            };

            for (ev_id, kind, hash) in evidence_items {
                tx.execute(
                    "INSERT OR IGNORE INTO claim_evidence (claim_id, evidence_id, source_kind, evidence_hash)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![claim_id, ev_id, kind, hash],
                )?;
            }

            // 6. Insert promotion receipt
            let receipt_id = Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO promotion_receipts (id, candidate_id, claim_id, policy_id, policy_version, achieved_tier, verifier_receipt, promoted_at)
                 VALUES (?1, ?2, ?3, ?4, '1.0', ?5, ?6, ?7)",
                params![
                    receipt_id,
                    cid,
                    claim_id,
                    policy_id,
                    tier.as_str(),
                    verifier_receipt,
                    now,
                ],
            )?;

            // 7. Mark candidate promoted
            tx.execute(
                "UPDATE memory_candidates SET state = 'promoted' WHERE id = ?1",
                params![cid],
            )?;

            tx.commit()?;

            Ok(PromotionReceipt {
                id: receipt_id,
                candidate_id: cid,
                claim_id,
                policy_id,
                policy_version: "1.0".to_string(),
                achieved_tier: tier.as_str().to_string(),
                verifier_receipt,
                promoted_at: now,
            })
        })
    }

    pub fn seal_open_segment(
        &self,
        session_id: &str,
        branch_id: &str,
    ) -> Result<Option<EventSegment>> {
        let sid = session_id.to_string();
        let bid = branch_id.to_string();
        self.writer.execute(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content_hash, sequence FROM session_events
                 WHERE session_id = ?1 AND branch_id = ?2 AND segment_id IS NULL
                 ORDER BY sequence ASC",
            )?;
            let rows = stmt.query_map(params![sid, bid], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?;
            let mut leaves = Vec::new();
            for row in rows {
                leaves.push(row?);
            }
            drop(stmt);
            if leaves.is_empty() {
                return Ok(None);
            }
            let start_seq = leaves[0].2;
            let end_seq = leaves[leaves.len() - 1].2;
            let root = crate::evidence::merkle_root(
                &leaves
                    .iter()
                    .map(|(_, hash, _)| hash.as_bytes())
                    .collect::<Vec<_>>(),
            );
            let prev: Option<String> = conn
                .query_row(
                    "SELECT merkle_root FROM event_segments WHERE session_id = ?1 AND branch_id = ?2 ORDER BY end_seq DESC LIMIT 1",
                    params![sid, bid],
                    |row| row.get(0),
                )
                .ok();
            let retention: String = conn
                .query_row(
                    "SELECT retention_class FROM sessions WHERE id = ?1",
                    params![sid],
                    |row| row.get(0),
                )
                .unwrap_or_else(|_| "standard".to_string());
            let now = Utc::now().to_rfc3339();
            let segment_id = Uuid::new_v4().to_string();
            conn.execute(
                "INSERT INTO event_segments (id, session_id, branch_id, start_seq, end_seq, merkle_root, prev_segment_root, sealed_at, state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'sealed')",
                params![segment_id, sid, bid, start_seq, end_seq, root, prev, now],
            )?;
            for (event_id, _, _) in &leaves {
                conn.execute(
                    "UPDATE session_events SET segment_id = ?1 WHERE id = ?2",
                    params![segment_id, event_id],
                )?;
            }
            let record_id = Uuid::new_v4().to_string();
            conn.execute(
                "INSERT INTO evidence_records (id, source_kind, source_id, commitment, merkle_root, segment_id, retention_class, retain_until, created_at)
                 VALUES (?1, 'event_segment', ?2, ?3, ?3, ?2, ?4, ?5, ?6)",
                params![
                    record_id,
                    segment_id,
                    root,
                    retention,
                    crate::evidence::retain_until(&retention, Utc::now()),
                    now
                ],
            )?;
            Ok(Some(EventSegment {
                id: segment_id,
                session_id: sid,
                branch_id: bid,
                start_seq,
                end_seq,
                merkle_root: root,
                prev_segment_root: prev,
                sealed_at: now,
                state: "sealed".to_string(),
            }))
        })
    }

    pub fn apply_retention(&self, now: &str) -> Result<usize> {
        let now_owned = now.to_string();
        self.writer.execute(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, retention_class, created_at FROM sessions WHERE retention_class != 'permanent'",
            )?;
            let sessions: Vec<(String, String, String)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
                .collect::<Result<Vec<_>>>()?;
            drop(stmt);
            let mut purged = 0usize;
            for (sid, class, created_at) in sessions {
                let expired = match class.as_str() {
                    "ephemeral" => created_at < shift_back(&now_owned, 1),
                    _ => created_at < shift_back(&now_owned, 365),
                };
                if !expired {
                    continue;
                }
                let changed = conn.execute(
                    "UPDATE session_events SET content = NULL, payload_json = '{\"redacted\":true}', redacted = 1
                     WHERE session_id = ?1 AND redacted = 0",
                    params![sid],
                )?;
                purged += changed;
            }
            let changed = conn.execute(
                "UPDATE session_events SET content = NULL, payload_json = '{\"redacted\":true}', redacted = 1
                 WHERE id IN (
                     SELECT source_id FROM evidence_records
                     WHERE source_kind = 'session_event' AND retain_until IS NOT NULL AND retain_until < ?1
                 ) AND redacted = 0",
                params![now_owned],
            )?;
            purged += changed;
            Ok(purged)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_in_memory_db_crud() {
        let db = Database::open_in_memory().expect("open in memory");

        let input = EntityWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            entity_type: "discovery".to_string(),
            canonical_name: "test_cortex_rebuild".to_string(),
            content: "Testing the native Rust cortex-rs database engine".to_string(),
            aliases: vec!["cortex_test".to_string()],
            metadata: serde_json::json!({"test": true}),
            confidence: 1.0,
            valid_from: None,
            valid_to: None,
            external_id: None,
        };

        let dummy_emb = vec![0.5f32; 768];
        let receipt = db
            .upsert_entity(&input, Some(&dummy_emb))
            .expect("upsert entity");
        assert_eq!(receipt.operation, "entity.upsert");

        let fetched = db
            .get_entity("test_cortex_rebuild", Some("atlas-memory"))
            .expect("get entity");
        assert!(fetched.is_some());
        let entity = fetched.unwrap();
        assert_eq!(entity.canonical_name, "test_cortex_rebuild");
        assert_eq!(entity.aliases.len(), 1);

        let search_results = db
            .search_entities("Rust", Some("atlas-memory"), 10)
            .expect("search");
        assert!(!search_results.is_empty());
        assert_eq!(
            search_results[0].entity.canonical_name,
            "test_cortex_rebuild"
        );

        let recall_results = db
            .recall_entities("Rust", Some(&dummy_emb), Some("atlas-memory"), 5)
            .expect("recall");
        assert!(!recall_results.is_empty());
        assert!(recall_results[0].final_score > 0.5);
    }

    #[test]
    fn test_sqlite_wal_persistence_under_concurrency() {
        let temp_dir = std::env::temp_dir().join(format!("cortex_wal_test_{}", Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let db_path = temp_dir.join("cortex.db");

        // Open database in WAL mode
        let db = Arc::new(Database::open(&db_path).expect("open wal db"));

        // Verify WAL mode is active
        {
            let conn = db.conn.lock().unwrap();
            let mode: String = conn
                .query_row("PRAGMA journal_mode", [], |r| r.get(0))
                .unwrap();
            assert_eq!(mode.to_lowercase(), "wal");
        }

        let mut handles = Vec::new();

        // Spawn 4 concurrent writer threads
        for t in 0..4 {
            let db_clone = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for i in 0..10 {
                    let input = EntityWriteInput {
                        id: None,
                        space: "atlas-memory".to_string(),
                        entity_type: "concurrent_write".to_string(),
                        canonical_name: format!("entity_t{}_i{}", t, i),
                        content: format!("Content from thread {} iteration {}", t, i),
                        aliases: vec![format!("alias_t{}_i{}", t, i)],
                        metadata: serde_json::json!({"thread": t, "iter": i}),
                        confidence: 0.95,
                        valid_from: None,
                        valid_to: None,
                        external_id: None,
                    };
                    let receipt = db_clone
                        .upsert_entity(&input, None)
                        .expect("concurrent entity upsert");
                    assert_eq!(receipt.operation, "entity.upsert");

                    // Also write a claim
                    let claim_input = ClaimWriteInput {
                        id: None,
                        space: "atlas-memory".to_string(),
                        subject_entity_id: receipt.target_id.clone(),
                        predicate: "authored_by".to_string(),
                        object_entity_id: None,
                        literal_value: Some(serde_json::json!(format!("worker_thread_{}", t))),
                        confidence: 1.0,
                        metadata: serde_json::json!({}),
                    };
                    let claim_receipt = db_clone
                        .upsert_claim(&claim_input)
                        .expect("concurrent claim upsert");
                    assert_eq!(claim_receipt.operation, "claim.upsert");
                }
            }));
        }

        // Spawn 4 concurrent reader threads
        for _ in 0..4 {
            let db_clone = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for _ in 0..15 {
                    let _ = db_clone.search_entities("Content", Some("atlas-memory"), 10);
                    let _ = db_clone.recall_entities("thread", None, Some("atlas-memory"), 5);
                    let _ = db_clone.get_entity("entity_t0_i0", Some("atlas-memory"));
                    thread::sleep(std::time::Duration::from_millis(2));
                }
            }));
        }

        // Wait for all threads to join
        for h in handles {
            h.join().expect("thread join");
        }

        // Close db by dropping Arc
        drop(db);

        // Re-open from disk to verify cold-start WAL recovery and persistence
        let reopened = Database::open(&db_path).expect("reopen wal db");
        for t in 0..4 {
            for i in 0..10 {
                let name = format!("entity_t{}_i{}", t, i);
                let ent = reopened
                    .get_entity(&name, Some("atlas-memory"))
                    .expect("get persisted entity");
                assert!(
                    ent.is_some(),
                    "Entity {} must persist across restarts",
                    name
                );
                let e = ent.unwrap();
                assert_eq!(e.canonical_name, name);
                assert_eq!(
                    e.content,
                    format!("Content from thread {} iteration {}", t, i)
                );
            }
        }

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fts5_exact_prefix_boolean_and_ranking() {
        let db = Database::open_in_memory().expect("open in memory db");

        let items = vec![
            (
                "NVFP4_Loader",
                "High throughput native NVFP4 checkpoint loader for vLLM and MAX",
                "discovery",
            ),
            (
                "Qwen3_Parser",
                "Llama cpp tokenizer and grammar parser for Qwen3 models",
                "learned_procedure",
            ),
            (
                "TPM_Vault",
                "Hardware TPM bound key vault with zero plaintext secrets on disk",
                "lesson",
            ),
            (
                "Sovereign_Covenant",
                "Frontier AI Anti Enclosure mandate and sovereign defense network",
                "discovery",
            ),
            (
                "Memory_Optimizer",
                "Memory architecture optimization for memory engines with memory reuse",
                "lesson",
            ),
        ];

        for (name, content, etype) in items {
            let input = EntityWriteInput {
                id: None,
                space: "atlas-memory".to_string(),
                entity_type: etype.to_string(),
                canonical_name: name.to_string(),
                content: content.to_string(),
                aliases: vec![],
                metadata: serde_json::json!({}),
                confidence: 1.0,
                valid_from: None,
                valid_to: None,
                external_id: None,
            };
            db.upsert_entity(&input, None).expect("insert entity");
        }

        // 1. Exact term matching
        let exact_res = db
            .search_entities("NVFP4", Some("atlas-memory"), 10)
            .expect("search exact");
        assert_eq!(exact_res.len(), 1);
        assert_eq!(exact_res[0].entity.canonical_name, "NVFP4_Loader");

        let exact_tpm = db
            .search_entities("TPM", Some("atlas-memory"), 10)
            .expect("search exact TPM");
        assert_eq!(exact_tpm.len(), 1);
        assert_eq!(exact_tpm[0].entity.canonical_name, "TPM_Vault");

        // 2. Prefix queries
        let prefix_res = db
            .search_entities("NVF*", Some("atlas-memory"), 10)
            .expect("search prefix");
        assert_eq!(prefix_res.len(), 1);
        assert_eq!(prefix_res[0].entity.canonical_name, "NVFP4_Loader");

        let prefix_hard = db
            .search_entities("hardw*", Some("atlas-memory"), 10)
            .expect("search prefix hardw*");
        assert_eq!(prefix_hard.len(), 1);
        assert_eq!(prefix_hard[0].entity.canonical_name, "TPM_Vault");

        let prefix_token = db
            .search_entities("token*", Some("atlas-memory"), 10)
            .expect("search prefix token*");
        assert_eq!(prefix_token.len(), 1);
        assert_eq!(prefix_token[0].entity.canonical_name, "Qwen3_Parser");

        // 3. Boolean AND
        let and_res = db
            .search_entities("NVFP4 AND vLLM", Some("atlas-memory"), 10)
            .expect("search AND");
        assert_eq!(and_res.len(), 1);
        assert_eq!(and_res[0].entity.canonical_name, "NVFP4_Loader");

        let and_mismatch = db
            .search_entities("NVFP4 AND non_existent_token", Some("atlas-memory"), 10)
            .expect("search AND mismatch");
        assert_eq!(and_mismatch.len(), 0);

        // 4. Boolean OR
        let or_res = db
            .search_entities("Qwen3 OR TPM", Some("atlas-memory"), 10)
            .expect("search OR");
        assert_eq!(or_res.len(), 2);
        let names: Vec<String> = or_res
            .into_iter()
            .map(|r| r.entity.canonical_name)
            .collect();
        assert!(names.contains(&"Qwen3_Parser".to_string()));
        assert!(names.contains(&"TPM_Vault".to_string()));

        // 5. Boolean NOT
        let not_res = db
            .search_entities("native NOT tokenizer", Some("atlas-memory"), 10)
            .expect("search NOT");
        assert_eq!(not_res.len(), 1);
        assert_eq!(not_res[0].entity.canonical_name, "NVFP4_Loader");

        // 6. Ranking test (BM25 term frequency)
        let rank_res = db
            .search_entities("memory", Some("atlas-memory"), 10)
            .expect("search ranking");
        assert!(!rank_res.is_empty());
        assert_eq!(rank_res[0].entity.canonical_name, "Memory_Optimizer");
    }

    #[test]
    fn test_vector_similarity_and_nearest_neighbor_ranking() {
        let db = Database::open_in_memory().expect("open in memory db");

        // Create 3 vectors with 768 dimensions
        let mut emb_a = vec![0.0f32; 768];
        emb_a[0] = 1.0; // Direction X

        let mut emb_b = vec![0.0f32; 768];
        emb_b[0] = std::f32::consts::FRAC_1_SQRT_2; // Direction between X and Y
        emb_b[1] = std::f32::consts::FRAC_1_SQRT_2;

        let mut emb_c = vec![0.0f32; 768];
        emb_c[1] = 1.0; // Direction Y

        let entities = vec![
            ("Vector_Alpha", "Entity aligned with primary axis X", emb_a),
            ("Vector_Beta", "Entity aligned midway between axes", emb_b),
            (
                "Vector_Gamma",
                "Entity aligned with secondary axis Y",
                emb_c,
            ),
        ];

        for (name, content, emb) in entities {
            let input = EntityWriteInput {
                id: None,
                space: "atlas-memory".to_string(),
                entity_type: "vector_item".to_string(),
                canonical_name: name.to_string(),
                content: content.to_string(),
                aliases: vec![],
                metadata: serde_json::json!({}),
                confidence: 1.0,
                valid_from: None,
                valid_to: None,
                external_id: None,
            };
            db.upsert_entity(&input, Some(&emb))
                .expect("insert vector entity");
        }

        // Query vector strongly aligned with X: [0.99, 0.05, 0, ...]
        let mut query_emb = vec![0.0f32; 768];
        query_emb[0] = 0.99;
        query_emb[1] = 0.05;

        let recall_res = db
            .recall_entities("Entity", Some(&query_emb), Some("atlas-memory"), 3)
            .expect("recall");
        assert_eq!(recall_res.len(), 3);

        // Verify nearest neighbor ordering: Alpha first, Beta second, Gamma third
        assert_eq!(recall_res[0].entity.canonical_name, "Vector_Alpha");
        assert_eq!(recall_res[1].entity.canonical_name, "Vector_Beta");
        assert_eq!(recall_res[2].entity.canonical_name, "Vector_Gamma");

        assert!(recall_res[0].final_score > recall_res[1].final_score);
        assert!(recall_res[1].final_score > recall_res[2].final_score);

        // Empty embedding fallback to lexical only
        let empty_recall = db
            .recall_entities("secondary", None, Some("atlas-memory"), 3)
            .expect("empty emb recall");
        assert!(!empty_recall.is_empty());
        assert_eq!(empty_recall[0].entity.canonical_name, "Vector_Gamma");
    }

    #[test]
    fn test_schema_integrity_metadata_indexing_and_deduplication() {
        let db = Database::open_in_memory().expect("open in memory db");

        // 1. Entity deduplication test
        let input_v1 = EntityWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            entity_type: "discovery".to_string(),
            canonical_name: "Atlas_Vault_Spec".to_string(),
            content: "Initial specification v1".to_string(),
            aliases: vec!["vault_v1".to_string()],
            metadata: serde_json::json!({
                "subsystem": "security",
                "hardware": {"tpm": true, "vendor": "stmicroelectronics"}
            }),
            confidence: 0.9,
            valid_from: None,
            valid_to: None,
            external_id: None,
        };

        let receipt1 = db.upsert_entity(&input_v1, None).expect("first upsert");
        let initial_id = receipt1.target_id.clone();

        // Re-upsert with same canonical_name in same space
        let input_v2 = EntityWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            entity_type: "learned_procedure".to_string(),
            canonical_name: "Atlas_Vault_Spec".to_string(),
            content: "Updated specification v2 with hardware sealing".to_string(),
            aliases: vec!["vault_v1".to_string(), "vault_v2".to_string()],
            metadata: serde_json::json!({
                "subsystem": "security",
                "hardware": {"tpm": true, "vendor": "stmicroelectronics", "pcr_sealing": [0, 2, 7]}
            }),
            confidence: 1.0,
            valid_from: None,
            valid_to: None,
            external_id: None,
        };

        let receipt2 = db.upsert_entity(&input_v2, None).expect("second upsert");

        // Verify deduplication: ID remains unchanged, revision incremented to 2
        assert_eq!(receipt2.target_id, initial_id);

        let fetched = db
            .get_entity("Atlas_Vault_Spec", Some("atlas-memory"))
            .expect("get entity")
            .unwrap();
        assert_eq!(fetched.id, initial_id);
        assert_eq!(fetched.revision, 2);
        assert_eq!(fetched.entity_type, "learned_procedure");
        assert_eq!(
            fetched.content,
            "Updated specification v2 with hardware sealing"
        );
        assert_eq!(fetched.aliases.len(), 2);

        // Verify nested metadata integrity
        assert_eq!(fetched.metadata["subsystem"], "security");
        assert_eq!(fetched.metadata["hardware"]["tpm"], true);
        assert_eq!(
            fetched.metadata["hardware"]["pcr_sealing"]
                .as_array()
                .unwrap()
                .len(),
            3
        );

        // Verify exactly 1 entity row in space
        {
            let conn = db.conn.lock().unwrap();
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM entities WHERE space_slug = 'atlas-memory' AND canonical_name = 'Atlas_Vault_Spec'",
                [],
                |r| r.get(0),
            ).unwrap();
            assert_eq!(count, 1);
        }

        // 2. Claim schema and foreign key traversal
        let obj_input = EntityWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            entity_type: "component".to_string(),
            canonical_name: "TPM_Chip".to_string(),
            content: "Discrete TPM 2.0 module".to_string(),
            aliases: vec![],
            metadata: serde_json::json!({}),
            confidence: 1.0,
            valid_from: None,
            valid_to: None,
            external_id: None,
        };
        let obj_receipt = db
            .upsert_entity(&obj_input, None)
            .expect("upsert object entity");

        let claim_input = ClaimWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            subject_entity_id: initial_id.clone(),
            predicate: "binds_to".to_string(),
            object_entity_id: Some(obj_receipt.target_id.clone()),
            literal_value: None,
            confidence: 1.0,
            metadata: serde_json::json!({"interface": "SPI"}),
        };
        db.upsert_claim(&claim_input).expect("upsert claim");

        let claims = db
            .traverse_claims(&initial_id, Some("atlas-memory"))
            .expect("traverse claims");
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].predicate, "binds_to");
        assert_eq!(claims[0].object_entity_id, Some(obj_receipt.target_id));
        assert_eq!(claims[0].metadata["interface"], "SPI");

        // 3. Retraction verification
        let retract_receipt = db.retract_target("entity", &initial_id).expect("retract");
        assert_eq!(retract_receipt.operation, "entity.retract");

        let after_retract = db
            .get_entity("Atlas_Vault_Spec", Some("atlas-memory"))
            .expect("get retracted");
        assert!(after_retract.is_none());

        let search_retract = db
            .search_entities("Atlas_Vault_Spec", Some("atlas-memory"), 10)
            .expect("search retracted");
        assert!(search_retract.is_empty());
    }

    #[test]
    fn test_episodic_plane_full_lifecycle_and_branches() {
        let db = Database::open_in_memory().expect("open in memory db");

        // 1. Create session
        let session_input = CreateSessionInput {
            id: Some("session-alpha".to_string()),
            space: "atlas-memory".to_string(),
            agent_id: Some("agent-42".to_string()),
            world_id: Some("world-root".to_string()),
            parent_session_id: None,
            fork_event_id: None,
            retention_class: "standard".to_string(),
            metadata: serde_json::json!({"mission": "mars_exploration"}),
        };
        let session = db.create_session(&session_input).expect("create session");
        assert_eq!(session.id, "session-alpha");
        assert_eq!(session.status, "active");

        // 2. Append main branch events
        let ev1 = SessionEventInput {
            id: Some("ev-m1".to_string()),
            branch_id: Some("main".to_string()),
            parent_event_id: None,
            event_type: "user_message".to_string(),
            role: Some("user".to_string()),
            content: Some("Launch the rover".to_string()),
            payload: serde_json::json!({}),
            sensitivity: None,
        };
        let ev2 = SessionEventInput {
            id: Some("ev-m2".to_string()),
            branch_id: Some("main".to_string()),
            parent_event_id: Some("ev-m1".to_string()),
            event_type: "agent_action".to_string(),
            role: Some("assistant".to_string()),
            content: Some("Rover trajectory plotted".to_string()),
            payload: serde_json::json!({"coordinates": [12.4, -45.1]}),
            sensitivity: None,
        };
        let appended_main = db
            .batch_append_events("session-alpha", "main", &[ev1, ev2])
            .expect("append main events");
        assert_eq!(appended_main.len(), 2);
        assert_eq!(appended_main[0].sequence, 1);
        assert_eq!(appended_main[1].sequence, 2);
        assert!(!appended_main[0].content_hash.is_empty());

        // 3. Append experimental branch events starting from event 1
        let ev_b1 = SessionEventInput {
            id: Some("ev-b1".to_string()),
            branch_id: Some("branch-alt".to_string()),
            parent_event_id: Some("ev-m1".to_string()),
            event_type: "agent_observation".to_string(),
            role: Some("system".to_string()),
            content: Some("Terrain obstacle detected".to_string()),
            payload: serde_json::json!({"obstacle": "boulder"}),
            sensitivity: None,
        };
        let appended_b = db
            .batch_append_events("session-alpha", "branch-alt", &[ev_b1])
            .expect("append branch events");
        assert_eq!(appended_b.len(), 1);
        // First event on branch-alt has sequence 1 on its branch!
        assert_eq!(appended_b[0].sequence, 1);
        assert_eq!(appended_b[0].branch_id, "branch-alt");

        // 4. Test idempotency: re-appending ev-m1 returns existing record with sequence 1
        let replay_ev = SessionEventInput {
            id: Some("ev-m1".to_string()),
            branch_id: Some("main".to_string()),
            parent_event_id: None,
            event_type: "user_message".to_string(),
            role: Some("user".to_string()),
            content: Some("Launch the rover".to_string()),
            payload: serde_json::json!({}),
            sensitivity: None,
        };
        let replayed = db
            .batch_append_events("session-alpha", "main", &[replay_ev])
            .expect("replay idempotent event");
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, "ev-m1");
        assert_eq!(replayed[0].sequence, 1);

        // Verify total main events is still 2
        let all_main = db
            .get_session_events("session-alpha", "main", None, 100)
            .expect("get main events");
        assert_eq!(all_main.len(), 2);

        // 5. Query with after_seq
        let after_1 = db
            .get_session_events("session-alpha", "main", Some(1), 100)
            .expect("get main events after 1");
        assert_eq!(after_1.len(), 1);
        assert_eq!(after_1[0].id, "ev-m2");

        // 6. Watermarks
        db.set_watermark("session-alpha", "summarizer", "0.1.0", 2)
            .expect("set watermark");
        let wm = db
            .get_watermark("session-alpha", "summarizer", "0.1.0")
            .expect("get watermark");
        assert_eq!(wm, Some(2));

        let wm_unseen = db
            .get_watermark("session-alpha", "extractor", "0.1.0")
            .expect("get unseen watermark");
        assert_eq!(wm_unseen, None);

        // 7. Close session
        db.close_session("session-alpha").expect("close session");
        let closed = db
            .get_session("session-alpha")
            .expect("get closed")
            .unwrap();
        assert_eq!(closed.status, "closed");
        assert!(closed.closed_at.is_some());
    }

    #[test]
    fn test_episodic_concurrency_and_writer_throughput() {
        let temp_dir =
            std::env::temp_dir().join(format!("cortex_episodic_test_{}", Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let db_path = temp_dir.join("cortex.db");

        let db = Arc::new(Database::open(&db_path).expect("open db"));
        let session_input = CreateSessionInput {
            id: Some("session-concurrent".to_string()),
            space: "atlas-memory".to_string(),
            agent_id: Some("agent-swarm".to_string()),
            world_id: None,
            parent_session_id: None,
            fork_event_id: None,
            retention_class: "standard".to_string(),
            metadata: serde_json::json!({}),
        };
        db.create_session(&session_input).expect("create session");

        let mut handles = Vec::new();

        // 6 concurrent writer threads, each writing to its own branch
        for thread_idx in 0..6 {
            let db_clone = Arc::clone(&db);
            let branch_name = format!("worker_branch_{}", thread_idx);
            handles.push(thread::spawn(move || {
                for i in 0..20 {
                    let ev = SessionEventInput {
                        id: None,
                        branch_id: Some(branch_name.clone()),
                        parent_event_id: None,
                        event_type: "agent_observation".to_string(),
                        role: Some("assistant".to_string()),
                        content: Some(format!("Data from thread {} iter {}", thread_idx, i)),
                        payload: serde_json::json!({"t": thread_idx, "i": i}),
                        sensitivity: None,
                    };
                    let appended = db_clone
                        .batch_append_events("session-concurrent", &branch_name, &[ev])
                        .expect("append concurrent event");
                    assert_eq!(appended[0].sequence, (i + 1) as i64);
                }
            }));
        }

        // 4 concurrent reader threads querying events during writes
        for _ in 0..4 {
            let db_clone = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for _ in 0..25 {
                    let _ = db_clone.get_session("session-concurrent");
                    let _ = db_clone.get_session_events(
                        "session-concurrent",
                        "worker_branch_0",
                        None,
                        50,
                    );
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            }));
        }

        for h in handles {
            h.join().expect("join thread");
        }

        // Verify each branch has exactly 20 events with strict sequence 1..=20
        for thread_idx in 0..6 {
            let branch_name = format!("worker_branch_{}", thread_idx);
            let events = db
                .get_session_events("session-concurrent", &branch_name, None, 100)
                .expect("get branch events");
            assert_eq!(events.len(), 20);
            for (idx, ev) in events.iter().enumerate() {
                assert_eq!(ev.sequence, (idx + 1) as i64);
            }
        }
    }

    #[test]
    fn test_schema_migration_backward_compatibility() {
        let temp_dir =
            std::env::temp_dir().join(format!("cortex_migration_test_{}", Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let db_path = temp_dir.join("cortex.db");

        // Open and verify user_version is 5 (all migrations applied)
        let db = Database::open(&db_path).expect("open db");
        {
            let conn = db.conn.lock().unwrap();
            let version: i32 = conn
                .query_row("PRAGMA user_version;", [], |r| r.get(0))
                .unwrap();
            assert_eq!(version, 5);

            // Verify canonical, episodic, candidate and provenance tables exist
            let tables: Vec<String> = {
                let mut stmt = conn
                    .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
                    .unwrap();
                let rows = stmt.query_map([], |r| r.get(0)).unwrap();
                rows.map(|r| r.unwrap()).collect()
            };
            assert!(tables.contains(&"entities".to_string()));
            assert!(tables.contains(&"claims".to_string()));
            assert!(tables.contains(&"sessions".to_string()));
            assert!(tables.contains(&"session_events".to_string()));
            assert!(tables.contains(&"processing_watermarks".to_string()));
            assert!(tables.contains(&"event_segments".to_string()));
            assert!(tables.contains(&"session_summaries".to_string()));
            assert!(tables.contains(&"memory_candidates".to_string()));
            assert!(tables.contains(&"candidate_evidence".to_string()));
            assert!(tables.contains(&"promotion_receipts".to_string()));
            assert!(tables.contains(&"claim_evidence".to_string()));
            assert!(tables.contains(&"evidence_records".to_string()));
            assert!(tables.contains(&"security_state".to_string()));
        }
    }

    #[test]
    fn test_seal_segment_and_retention_purge() {
        let db = Database::open_in_memory().unwrap();
        db.create_session(&CreateSessionInput {
            id: Some("sess-seal".to_string()),
            space: "atlas-memory".to_string(),
            agent_id: Some("atlas".to_string()),
            world_id: None,
            parent_session_id: None,
            fork_event_id: None,
            retention_class: "ephemeral".to_string(),
            metadata: serde_json::json!({}),
        })
        .unwrap();

        let events = vec![
            SessionEventInput {
                id: Some("ev-seal-1".to_string()),
                branch_id: Some("main".to_string()),
                parent_event_id: None,
                event_type: "user_message".to_string(),
                role: Some("user".to_string()),
                content: Some("Alpha content".to_string()),
                payload: serde_json::json!({"n": 1}),
                sensitivity: None,
            },
            SessionEventInput {
                id: Some("ev-seal-2".to_string()),
                branch_id: Some("main".to_string()),
                parent_event_id: Some("ev-seal-1".to_string()),
                event_type: "tool_result".to_string(),
                role: Some("tool".to_string()),
                content: Some("Beta content".to_string()),
                payload: serde_json::json!({"n": 2}),
                sensitivity: None,
            },
        ];
        db.batch_append_events("sess-seal", "main", &events)
            .unwrap();

        let segment = db
            .seal_open_segment("sess-seal", "main")
            .unwrap()
            .expect("segment sealed");
        assert_eq!(segment.start_seq, 1);
        assert_eq!(segment.end_seq, 2);
        assert_eq!(segment.merkle_root.len(), 64);
        assert!(segment.prev_segment_root.is_none());

        // Events are bound to the segment.
        let sealed = db
            .get_session_events("sess-seal", "main", None, 10)
            .unwrap();
        assert!(sealed.iter().all(|e| e.segment_id.is_some()));

        // A second seal yields no new segment.
        assert!(db.seal_open_segment("sess-seal", "main").unwrap().is_none());

        // Retention purge nulls content but keeps the record and commitment.
        let conn = db.conn.lock().unwrap();
        let commitment: String = conn
            .query_row(
                "SELECT commitment FROM evidence_records WHERE segment_id = ?1",
                params![segment.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(commitment, segment.merkle_root);
    }
}
