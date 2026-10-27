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

use reqwest::Client;
use serde_json::{json, Value};

const SEARCH_URL: &str = "https://context7.com/api/v2/libs/search";
const CONTEXT_URL: &str = "https://context7.com/api/v2/context";
const MAX_LIBRARIES: usize = 5;
const MAX_SNIPPETS: usize = 4;
const MAX_CODE_CHARS: usize = 1200;
const MAX_INFO_CHARS: usize = 800;

pub fn context7_dispatch_tool(args: &Value) -> Value {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(context7_dispatch(args))),
        Err(_) => json!({"status": "error", "error": "Context7 requires a running async runtime"}),
    }
}

pub async fn context7_dispatch(args: &Value) -> Value {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("resolve")
        .trim()
        .to_ascii_lowercase();
    match action.as_str() {
        "resolve" | "resolve-library-id" | "search" => resolve_library(args).await,
        "query" | "query-docs" | "docs" => query_docs(args).await,
        other => json!({
            "status": "error",
            "error": format!("Unknown context7 action '{other}'. Use resolve or query.")
        }),
    }
}

async fn resolve_library(args: &Value) -> Value {
    let library = args
        .get("library")
        .or_else(|| args.get("libraryName"))
        .or_else(|| args.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if library.is_empty() {
        return json!({"status": "error", "error": "context7 resolve requires library"});
    }
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or(library)
        .trim();
    let query = if query.is_empty() { library } else { query };
    match get_json(SEARCH_URL, &[("libraryName", library), ("query", query)]).await {
        Ok(body) => compact_libraries(&body, library, query),
        Err(error) => json!({"status": "error", "error": error}),
    }
}

async fn query_docs(args: &Value) -> Value {
    let library_id = args
        .get("library_id")
        .or_else(|| args.get("libraryId"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !valid_library_id(library_id) {
        return json!({
            "status": "error",
            "error": "context7 query requires library_id like /org/project. Resolve the library first."
        });
    }
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if query.is_empty() {
        return json!({"status": "error", "error": "context7 query requires one specific query"});
    }
    match get_json(
        CONTEXT_URL,
        &[
            ("libraryId", library_id),
            ("query", query),
            ("type", "json"),
        ],
    )
    .await
    {
        Ok(body) => compact_context(&body, library_id, query),
        Err(error) => json!({"status": "error", "error": error}),
    }
}

async fn get_json(url: &str, params: &[(&str, &str)]) -> Result<Value, String> {
    let key = crate::vault::get_secret("CONTEXT7_API_KEY")
        .map_err(|_| "CONTEXT7_API_KEY is unavailable from atlas-vault".to_string())?;
    let client = Client::new();
    let response = client
        .get(url)
        .bearer_auth(key)
        .header("Accept", "application/json")
        .query(params)
        .send()
        .await
        .map_err(|error| format!("Context7 connection failed: {error}"))?;
    let status = response.status();
    if status.as_u16() == 301 {
        let body = response.json::<Value>().await.unwrap_or(Value::Null);
        let redirect = body
            .get("redirectUrl")
            .and_then(Value::as_str)
            .unwrap_or("");
        return Err(format!(
            "Context7 moved this library. Use library_id {redirect}"
        ));
    }
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Context7 returned {status}: {}", scrub(&body)));
    }
    response
        .json::<Value>()
        .await
        .map_err(|error| format!("Context7 returned invalid JSON: {error}"))
}

fn valid_library_id(id: &str) -> bool {
    let mut parts = id.split('/');
    if parts.next() != Some("") {
        return false;
    }
    let Some(source) = parts.next() else {
        return false;
    };
    let Some(name) = parts.next() else {
        return false;
    };
    if source.is_empty() || source.contains('@') || name.is_empty() {
        return false;
    }
    match parts.next() {
        None => true,
        Some(version) if !version.is_empty() && !version.contains('@') && !name.contains('@') => {
            parts.next().is_none()
        }
        Some(_) => false,
    }
}

fn compact_libraries(body: &Value, library: &str, query: &str) -> Value {
    let results = body
        .get("results")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let libraries: Vec<Value> = results
        .into_iter()
        .take(MAX_LIBRARIES)
        .map(|item| {
            json!({
                "id": item.get("id").cloned().unwrap_or(Value::Null),
                "title": item.get("title").cloned().unwrap_or(Value::Null),
                "description": clip(item.get("description").and_then(Value::as_str).unwrap_or(""), 240),
                "trust_score": item.get("trustScore").or_else(|| item.get("benchmarkScore")).cloned().unwrap_or(Value::Null),
                "snippets": item.get("totalSnippets").or_else(|| item.get("snippets")).cloned().unwrap_or(Value::Null)
            })
        })
        .collect();
    json!({
        "status": "ok",
        "mode": "resolve",
        "library": library,
        "query": query,
        "libraries": libraries,
        "next": "Call context7 query with one library_id and one specific question."
    })
}

fn compact_context(body: &Value, library_id: &str, query: &str) -> Value {
    let code: Vec<Value> = body
        .get("codeSnippets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(MAX_SNIPPETS)
        .map(|snippet| {
            let code = snippet
                .get("codeList")
                .and_then(Value::as_array)
                .and_then(|list| list.first())
                .and_then(|item| item.get("code"))
                .and_then(Value::as_str)
                .unwrap_or("");
            json!({
                "title": snippet.get("codeTitle").cloned().unwrap_or(Value::Null),
                "page": snippet.get("pageTitle").cloned().unwrap_or(Value::Null),
                "language": snippet.get("codeLanguage").cloned().unwrap_or(Value::Null),
                "code": clip(code, MAX_CODE_CHARS)
            })
        })
        .collect();
    let info: Vec<Value> = body
        .get("infoSnippets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(MAX_SNIPPETS)
        .map(|snippet| {
            json!({
                "breadcrumb": snippet.get("breadcrumb").cloned().unwrap_or(Value::Null),
                "content": clip(snippet.get("content").and_then(Value::as_str).unwrap_or(""), MAX_INFO_CHARS)
            })
        })
        .collect();
    json!({
        "status": "ok",
        "mode": "query",
        "library_id": library_id,
        "query": query,
        "code": code,
        "info": info
    })
}

fn clip(text: &str, limit: usize) -> String {
    let mut chars = text.chars();
    let clipped: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        format!("{clipped} ...")
    } else {
        clipped
    }
}

struct SyncTarget {
    slug: &'static str,
    library: &'static str,
    query: &'static str,
    prefer: &'static [&'static str],
}

fn sync_catalog() -> &'static [SyncTarget] {
    &[
        SyncTarget {
            slug: "rust",
            library: "rust",
            query: "ownership, borrowing, lifetimes, and Result error handling",
            prefer: &["doc.rust-lang", "rust-lang"],
        },
        SyncTarget {
            slug: "tokio",
            library: "tokio",
            query: "spawn tasks on a multi-thread runtime and use select",
            prefer: &["tokio-rs/tokio"],
        },
        SyncTarget {
            slug: "axum",
            library: "axum",
            query: "Router, JSON extractors, and WebSocket routes",
            prefer: &["tokio-rs/axum"],
        },
        SyncTarget {
            slug: "serde",
            library: "serde",
            query: "derive Serialize and rename_all for structs",
            prefer: &["serde-rs/serde"],
        },
    ]
}

fn pick_library_id(libraries: &[Value], prefer: &[&str]) -> Option<String> {
    let ids: Vec<&str> = libraries
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str))
        .filter(|id| valid_library_id(id))
        .collect();
    for hint in prefer {
        if let Some(id) = ids.iter().copied().find(|id| id.contains(hint)) {
            return Some(id.to_string());
        }
    }
    ids.first().map(|id| (*id).to_string())
}

fn render_reference(library: &str, library_id: &str, query: &str, docs: &Value) -> String {
    let mut out = format!("Context7 reference for {library} ({library_id}).\nQuestion: {query}\n");
    if let Some(info) = docs.get("info").and_then(Value::as_array) {
        for note in info {
            let breadcrumb = note.get("breadcrumb").and_then(Value::as_str).unwrap_or("");
            let content = note.get("content").and_then(Value::as_str).unwrap_or("");
            if !content.is_empty() {
                out.push_str(&format!("\n{breadcrumb}\n{content}\n"));
            }
        }
    }
    if let Some(code) = docs.get("code").and_then(Value::as_array) {
        for snippet in code {
            let title = snippet
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("code");
            let body = snippet.get("code").and_then(Value::as_str).unwrap_or("");
            if !body.is_empty() {
                out.push_str(&format!("\n{title}\n{body}\n"));
            }
        }
    }
    clip(&out, 6000)
}

pub async fn sync_to_cortex() -> i32 {
    let mut failures = 0usize;
    let mut written = Vec::new();
    for target in sync_catalog() {
        match sync_one(target).await {
            Ok(library_id) => {
                println!("context7:{:<8} {}", target.slug, library_id);
                written.push(target.slug);
            }
            Err(error) => {
                failures += 1;
                println!("context7:{:<8} error: {error}", target.slug);
            }
        }
    }
    let stamp = chrono::Utc::now().to_rfc3339();
    let manifest = format!(
        "Context7 sync at {stamp}. Updated: {}. Failures: {failures}.",
        written.join(", ")
    );
    match crate::cortex::write_to_cortex(
        "context7:sync",
        &manifest,
        "reference",
        json!({
            "source": "context7",
            "fetched_at": stamp,
            "updated": written,
            "failures": failures
        }),
    )
    .await
    {
        Ok(_) => println!("context7:sync    recorded"),
        Err(error) => {
            failures += 1;
            println!("context7:sync    error: {error}");
        }
    }
    if failures == 0 {
        0
    } else {
        1
    }
}

async fn sync_one(target: &SyncTarget) -> Result<String, String> {
    let resolved = resolve_library(&json!({
        "library": target.library,
        "query": target.query
    }))
    .await;
    if resolved.get("status").and_then(Value::as_str) != Some("ok") {
        return Err(resolved
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("resolve failed")
            .to_string());
    }
    let libraries = resolved
        .get("libraries")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let library_id = pick_library_id(&libraries, target.prefer)
        .ok_or_else(|| format!("no Context7 id for {}", target.library))?;
    let docs = query_docs(&json!({
        "library_id": library_id,
        "query": target.query
    }))
    .await;
    if docs.get("status").and_then(Value::as_str) != Some("ok") {
        return Err(docs
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("query failed")
            .to_string());
    }
    let content = render_reference(target.library, &library_id, target.query, &docs);
    let name = format!("context7:{}", target.slug);
    crate::cortex::write_to_cortex(
        &name,
        &content,
        "reference",
        json!({
            "source": "context7",
            "library": target.library,
            "library_id": library_id,
            "query": target.query,
            "fetched_at": chrono::Utc::now().to_rfc3339()
        }),
    )
    .await?;
    Ok(library_id)
}

fn scrub(body: &str) -> String {
    let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let message = parsed
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| parsed.get("error").and_then(Value::as_str))
        .unwrap_or(body);
    if message.contains("ctx7sk") || message.contains("Bearer ") {
        "Context7 rejected the request".to_string()
    } else {
        clip(message, 300)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_ids_match_context7_shape() {
        assert!(valid_library_id("/tokio-rs/tokio"));
        assert!(valid_library_id("/vercel/next.js"));
        assert!(valid_library_id("/vercel/next.js/14.2.5"));
        assert!(valid_library_id("/vercel/next.js@14.2.5"));
        assert!(!valid_library_id("/a/b/c/d"));
        assert!(!valid_library_id("react"));
        assert!(!valid_library_id("/onlyone"));
        assert!(!valid_library_id("/a/b/c/d"));
    }

    #[test]
    fn resolve_payload_keeps_five_ids_and_drops_the_rest() {
        let body = json!({
            "results": [
                {"id": "/tokio-rs/tokio", "title": "Tokio", "description": "x".repeat(400), "trustScore": 90, "totalSnippets": 12},
                {"id": "/a/two", "title": "Two"},
                {"id": "/a/three", "title": "Three"},
                {"id": "/a/four", "title": "Four"},
                {"id": "/a/five", "title": "Five"},
                {"id": "/a/six", "title": "Six"}
            ]
        });
        let compact = compact_libraries(&body, "tokio", "spawn a task");
        assert_eq!(compact["libraries"].as_array().unwrap().len(), 5);
        assert_eq!(compact["libraries"][0]["id"], "/tokio-rs/tokio");
        assert!(compact["libraries"][0]["description"]
            .as_str()
            .unwrap()
            .ends_with(" ..."));
        assert!(
            compact["libraries"][0]
                .get("description")
                .unwrap()
                .as_str()
                .unwrap()
                .len()
                < 400
        );
    }

    #[test]
    fn query_payload_clips_code_and_keeps_the_library_id() {
        let body = json!({
            "codeSnippets": [{
                "codeTitle": "Spawn",
                "codeLanguage": "bash",
                "pageTitle": "Tokio",
                "codeList": [{"code": "m".repeat(2000)}]
            }],
            "infoSnippets": [{"breadcrumb": "Spawn", "content": "Call tokio::spawn."}]
        });
        let compact = compact_context(&body, "/tokio-rs/tokio", "spawn flags");
        assert_eq!(compact["library_id"], "/tokio-rs/tokio");
        assert!(compact["code"][0]["code"]
            .as_str()
            .unwrap()
            .ends_with(" ..."));
        assert_eq!(compact["info"][0]["content"], "Call tokio::spawn.");
    }

    #[test]
    fn sync_catalog_covers_the_stack_in_use() {
        let slugs: Vec<&str> = sync_catalog().iter().map(|target| target.slug).collect();
        assert_eq!(slugs, vec!["rust", "tokio", "axum", "serde"]);
    }

    #[test]
    fn preferred_library_id_beats_the_first_hit() {
        let libraries = vec![
            json!({"id": "/other/rust-book"}),
            json!({"id": "/websites/doc_rust-lang_org"}),
        ];
        assert_eq!(
            pick_library_id(&libraries, &["doc.rust-lang", "rust-lang"]).as_deref(),
            Some("/websites/doc_rust-lang_org")
        );
    }

    #[test]
    fn rendered_reference_names_the_library_and_clips() {
        let docs = json!({
            "info": [{"breadcrumb": "Borrowing", "content": "A borrow lasts for a scope."}],
            "code": [{"title": "Example", "code": "x".repeat(7000)}]
        });
        let text = render_reference("rust", "/websites/doc_rust-lang_org", "borrowing", &docs);
        assert!(text.contains("doc_rust-lang_org"));
        assert!(text.contains("A borrow lasts"));
        assert!(text.chars().count() <= 6004);
    }

    #[test]
    fn error_text_does_not_keep_a_key() {
        let scrubbed = scrub(r#"{"message":"bad key ctx7sk-secret-value"}"#);
        assert_eq!(scrubbed, "Context7 rejected the request");
    }
}
