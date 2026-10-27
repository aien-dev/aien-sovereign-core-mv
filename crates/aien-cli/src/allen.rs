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

//! `aien allen ...`: the ALLEN persona profile (arch#159), one operator step
//! per subcommand, against the running daemon. Each call prints ONE JSON
//! object on stdout (`"ok": true|false`) and exits 1 on failure.
//!
//!   status                                         identity, persona, model, unsupported facilities
//!   show                                           the saved profile (or the defaults)
//!   set --expect R [--name N] [--tone T] [--verbosity V] [--plain-language 0|1]
//!       [--pref key=value[@scope]]... [--unset-pref key]... [--note TEXT]
//!   history                                        every saved revision
//!   revert --to N --expect R                       new revision copying revision N
//!   reset --expect R                               new revision with the defaults
//!   memory put|recall|inspect|correct|forget|export   scoped memory, see parse_memory
//!   goals list|add|close                           host goal records
//! Memory and goal calls name their context: --context personal|work|project:NAME.
//! Only inspect, export and goals list take --owner 1 (every context) instead.
//!
//! Every change needs `--expect <current revision>` (0 when nothing is saved
//! yet): if the profile moved since you looked, the change is refused.
//! The daemon owns the profile store; this side only asks.
use aien_allen_profile::{Changes, PrefChange, Scope, Tone, Verbosity};
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::{ControlCommand, ControlResponse};
use serde_json::{json, Value};

type Flags = Vec<(String, String)>;

fn flags(args: &[String], allowed: &[&str]) -> Result<Flags, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let k = args[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("expected --flag, got {:?}", args[i]))?;
        if !allowed.contains(&k) {
            return Err(format!("unknown flag --{k}"));
        }
        let v = args
            .get(i + 1)
            .ok_or_else(|| format!("--{k} needs a value"))?;
        out.push((k.to_string(), v.clone()));
        i += 2;
    }
    Ok(out)
}

fn one<'a>(f: &'a Flags, k: &str) -> Result<Option<&'a str>, String> {
    let mut it = f.iter().filter(|(n, _)| n == k);
    let first = it.next().map(|(_, v)| v.as_str());
    if it.next().is_some() {
        return Err(format!("--{k} given more than once"));
    }
    Ok(first)
}

fn number(f: &Flags, k: &str) -> Result<u64, String> {
    let v = one(f, k)?.ok_or_else(|| {
        if k == "expect" {
            "--expect is required: the revision you looked at (0 when nothing is saved yet)".into()
        } else {
            format!("--{k} is required")
        }
    })?;
    v.trim()
        .parse::<u64>()
        .map_err(|_| format!("--{k} must be a whole number, got {v:?}"))
}

/// `key=value` or `key=value@scope` (scope one of all, personal, work, project).
fn parse_pref(s: &str) -> Result<PrefChange, String> {
    let (key, rest) = s
        .split_once('=')
        .ok_or_else(|| format!("--pref needs key=value, got {s:?}"))?;
    let (value, scope) = match rest.rsplit_once('@') {
        Some((v, sc)) if matches!(sc, "all" | "personal" | "work" | "project") => {
            (v, Scope::parse(sc).map_err(|e| e.to_string())?)
        }
        _ => (rest, Scope::All),
    };
    Ok(PrefChange {
        key: key.to_string(),
        value: value.to_string(),
        scope,
    })
}

/// Turn the command line into the daemon command (pure, no I/O).
pub fn parse(sub: &str, args: &[String]) -> Result<ControlCommand, String> {
    match sub {
        "memory" => parse_memory(args),
        "goals" => parse_goals(args),
        "status" => {
            flags(args, &[])?;
            Ok(ControlCommand::AllenStatus)
        }
        "show" => {
            flags(args, &[])?;
            Ok(ControlCommand::AllenProfileShow)
        }
        "history" => {
            flags(args, &[])?;
            Ok(ControlCommand::AllenProfileHistory)
        }
        "set" => {
            let f = flags(
                args,
                &[
                    "expect",
                    "name",
                    "tone",
                    "verbosity",
                    "plain-language",
                    "pref",
                    "unset-pref",
                    "note",
                ],
            )?;
            let expected_revision = number(&f, "expect")?;
            let plain_language = match one(&f, "plain-language")? {
                None => None,
                Some("1") => Some(true),
                Some("0") => Some(false),
                Some(o) => return Err(format!("--plain-language must be 0 or 1, got {o:?}")),
            };
            let changes = Changes {
                name: one(&f, "name")?.map(str::to_string),
                tone: one(&f, "tone")?
                    .map(Tone::parse)
                    .transpose()
                    .map_err(|e| e.to_string())?,
                verbosity: one(&f, "verbosity")?
                    .map(Verbosity::parse)
                    .transpose()
                    .map_err(|e| e.to_string())?,
                plain_language,
                set_prefs: f
                    .iter()
                    .filter(|(k, _)| k == "pref")
                    .map(|(_, v)| parse_pref(v))
                    .collect::<Result<_, _>>()?,
                unset_prefs: f
                    .iter()
                    .filter(|(k, _)| k == "unset-pref")
                    .map(|(_, v)| v.clone())
                    .collect(),
                note: one(&f, "note")?.map(str::to_string),
            };
            Ok(ControlCommand::AllenProfileSet {
                expected_revision,
                changes,
            })
        }
        "revert" => {
            let f = flags(args, &["expect", "to"])?;
            Ok(ControlCommand::AllenProfileRevert {
                expected_revision: number(&f, "expect")?,
                to: number(&f, "to")?,
            })
        }
        "reset" => {
            let f = flags(args, &["expect"])?;
            Ok(ControlCommand::AllenProfileReset {
                expected_revision: number(&f, "expect")?,
            })
        }
        other => Err(format!(
            "unknown step {other:?} (status, show, set, history, revert, reset, memory, goals)"
        )),
    }
}

/// `--context C` or `--owner 1`, exactly one (the daemon checks again).
fn context_or_owner(f: &Flags) -> Result<(Option<String>, bool), String> {
    let context = one(f, "context")?.map(str::to_string);
    let owner = match one(f, "owner")? {
        None => false,
        Some("1") => true,
        Some(o) => return Err(format!("--owner must be 1, got {o:?}")),
    };
    match (&context, owner) {
        (None, false) => Err(
            "name a context (--context personal|work|project:NAME) or use --owner 1 for every context"
                .into(),
        ),
        (Some(_), true) => Err("give either --context or --owner 1, not both".into()),
        _ => Ok((context, owner)),
    }
}

fn required(f: &Flags, k: &str) -> Result<String, String> {
    one(f, k)?
        .map(str::to_string)
        .ok_or_else(|| format!("--{k} is required"))
}

fn parse_memory(args: &[String]) -> Result<ControlCommand, String> {
    let (op, rest) = args.split_first().ok_or(
        "usage: aien allen memory <put|recall|inspect|correct|forget|export> --context C ...",
    )?;
    match op.as_str() {
        "put" => {
            let f = flags(rest, &["context", "kind", "text"])?;
            Ok(ControlCommand::AllenMemoryPut {
                context: required(&f, "context")?,
                kind: one(&f, "kind")?.unwrap_or("fact").to_string(),
                text: required(&f, "text")?,
            })
        }
        "recall" => {
            let f = flags(rest, &["context", "query"])?;
            Ok(ControlCommand::AllenMemoryRecall {
                context: required(&f, "context")?,
                query: one(&f, "query")?.map(str::to_string),
            })
        }
        "inspect" => {
            let f = flags(rest, &["context", "owner"])?;
            let (context, owner) = context_or_owner(&f)?;
            Ok(ControlCommand::AllenMemoryInspect { context, owner })
        }
        "export" => {
            let f = flags(rest, &["context", "owner"])?;
            let (context, owner) = context_or_owner(&f)?;
            Ok(ControlCommand::AllenMemoryExport { context, owner })
        }
        "correct" => {
            let f = flags(rest, &["context", "item", "text"])?;
            Ok(ControlCommand::AllenMemoryCorrect {
                context: required(&f, "context")?,
                item: required(&f, "item")?,
                text: required(&f, "text")?,
            })
        }
        "forget" => {
            let f = flags(rest, &["context", "item", "all-in-context"])?;
            let all_in_context = match one(&f, "all-in-context")? {
                None => false,
                Some("1") => true,
                Some(o) => return Err(format!("--all-in-context must be 1, got {o:?}")),
            };
            let item = one(&f, "item")?.map(str::to_string);
            if item.is_some() == all_in_context {
                return Err("give exactly one of --item ID or --all-in-context 1".into());
            }
            Ok(ControlCommand::AllenMemoryForget {
                context: required(&f, "context")?,
                item,
                all_in_context,
            })
        }
        other => Err(format!(
            "unknown memory step {other:?} (put, recall, inspect, correct, forget, export)"
        )),
    }
}

fn parse_goals(args: &[String]) -> Result<ControlCommand, String> {
    let (op, rest) = args
        .split_first()
        .ok_or("usage: aien allen goals <list|add|close> --context C ...")?;
    match op.as_str() {
        "list" => {
            let f = flags(rest, &["context", "owner"])?;
            let (context, owner) = context_or_owner(&f)?;
            Ok(ControlCommand::AllenGoalsList { context, owner })
        }
        "add" => {
            let f = flags(rest, &["context", "text"])?;
            Ok(ControlCommand::AllenGoalAdd {
                context: required(&f, "context")?,
                text: required(&f, "text")?,
            })
        }
        "close" => {
            let f = flags(rest, &["context", "item"])?;
            Ok(ControlCommand::AllenGoalClose {
                context: required(&f, "context")?,
                item: required(&f, "item")?,
            })
        }
        other => Err(format!("unknown goals step {other:?} (list, add, close)")),
    }
}

fn to_json<T: serde::Serialize>(v: &T) -> Result<Value, String> {
    serde_json::to_value(v).map_err(|e| e.to_string())
}

/// Daemon answer to the one printed JSON object. `Err` = the call failed.
fn render(sub: &str, r: ControlResponse) -> Result<Value, Value> {
    let ok = |body: Value| {
        let mut v = json!({"ok": true, "step": sub});
        v["result"] = body;
        Ok(v)
    };
    match r {
        ControlResponse::AllenStatusReport(x) => ok(to_json(&*x).unwrap_or(Value::Null)),
        ControlResponse::AllenProfile(x) => ok(to_json(&*x).unwrap_or(Value::Null)),
        ControlResponse::AllenHistory(x) => ok(to_json(&*x).unwrap_or(Value::Null)),
        ControlResponse::AllenMemoryResult(x) => ok(to_json(&*x).unwrap_or(Value::Null)),
        ControlResponse::AllenRefused(x) => Err(json!({
            "ok": false,
            "step": sub,
            "refused": to_json(&*x).unwrap_or(Value::Null),
            "changed": false,
        })),
        ControlResponse::Error(e) => Err(json!({"ok": false, "step": sub, "error": e})),
        other => Err(json!({
            "ok": false, "step": sub, "error": format!("unexpected response {other:?}"),
        })),
    }
}

pub async fn handle_allen_command(args: &[String]) {
    let Some(sub) = args.first() else {
        println!(
            "{}",
            json!({"ok": false, "error": "usage: aien allen <status|show|set|history|revert|reset|memory|goals> [--flag value ...]"})
        );
        std::process::exit(1);
    };
    let cmd = match parse(sub, &args[1..]) {
        Ok(c) => c,
        Err(e) => {
            println!("{}", json!({"ok": false, "step": sub, "error": e}));
            std::process::exit(1);
        }
    };
    let out = match AienRuntimeClient::default_client().send_command(cmd).await {
        Ok(r) => render(sub, r),
        // The daemon is a separate process. Not reaching it says nothing about
        // the identity or the saved profile; both stay on disk.
        Err(e) => Err(json!({
            "ok": false,
            "step": sub,
            "daemon": "unreachable",
            "error": e,
            "note": "temporary: the daemon is not reachable. Your identity and saved profile are not lost or changed.",
        })),
    };
    match out {
        Ok(v) => println!("{v}"),
        Err(v) => {
            println!("{v}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn set_requires_expect() {
        let e = parse("set", &a(&["--name", "Nova"])).unwrap_err();
        assert!(e.contains("--expect is required"), "{e}");
        assert!(parse("reset", &a(&[])).is_err());
        assert!(parse("revert", &a(&["--to", "1"])).is_err());
    }

    #[test]
    fn set_parses_every_field() {
        let c = parse(
            "set",
            &a(&[
                "--expect",
                "3",
                "--name",
                "Nova",
                "--tone",
                "warm",
                "--verbosity",
                "brief",
                "--plain-language",
                "1",
                "--pref",
                "units=metric@work",
                "--pref",
                "mail=a@b.example",
                "--unset-pref",
                "old",
                "--note",
                "why",
            ]),
        )
        .unwrap();
        let ControlCommand::AllenProfileSet {
            expected_revision,
            changes,
        } = c
        else {
            panic!()
        };
        assert_eq!(expected_revision, 3);
        assert_eq!(changes.name.as_deref(), Some("Nova"));
        assert_eq!(changes.tone, Some(Tone::Warm));
        assert_eq!(changes.plain_language, Some(true));
        assert_eq!(changes.set_prefs[0].scope, Scope::Work);
        assert_eq!(changes.set_prefs[0].value, "metric");
        // '@' that is not a scope stays in the value.
        assert_eq!(changes.set_prefs[1].value, "a@b.example");
        assert_eq!(changes.set_prefs[1].scope, Scope::All);
        assert_eq!(changes.unset_prefs, vec!["old".to_string()]);
    }

    #[test]
    fn bad_input_is_refused_before_any_call() {
        for bad in [
            vec!["--expect", "x"],
            vec!["--expect", "1", "--tone", "grumpy"],
            vec!["--expect", "1", "--plain-language", "yes"],
            vec!["--expect", "1", "--pref", "novalue"],
            vec!["--expect", "1", "--avatar", "cat"],
            vec!["--expect", "1", "--expect", "2"],
        ] {
            assert!(parse("set", &a(&bad)).is_err(), "{bad:?}");
        }
        assert!(parse("status", &a(&["--x", "1"])).is_err());
        assert!(parse("nope", &a(&[])).is_err());
    }

    #[test]
    fn memory_and_goal_calls_must_name_a_context() {
        for bad in [
            vec!["put", "--text", "x"],
            vec!["recall"],
            vec!["inspect"],
            vec!["export"],
            vec!["inspect", "--context", "work", "--owner", "1"],
            vec!["inspect", "--owner", "yes"],
            vec!["forget", "--context", "work"],
            vec![
                "forget",
                "--context",
                "work",
                "--item",
                "a",
                "--all-in-context",
                "1",
            ],
            vec![
                "put",
                "--context",
                "work",
                "--scope",
                "personal",
                "--text",
                "x",
            ],
        ] {
            assert!(parse("memory", &a(&bad)).is_err(), "{bad:?}");
        }
        for bad in [
            vec!["list"],
            vec!["add", "--text", "g"],
            vec!["close", "--item", "i"],
        ] {
            assert!(parse("goals", &a(&bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn memory_calls_parse() {
        let c = parse(
            "memory",
            &a(&[
                "put",
                "--context",
                "project:a",
                "--kind",
                "preference",
                "--text",
                "t",
            ]),
        )
        .unwrap();
        assert!(matches!(
            c,
            ControlCommand::AllenMemoryPut { ref context, ref kind, .. }
                if context == "project:a" && kind == "preference"
        ));
        let c = parse("memory", &a(&["inspect", "--owner", "1"])).unwrap();
        assert!(matches!(
            c,
            ControlCommand::AllenMemoryInspect {
                context: None,
                owner: true
            }
        ));
        let c = parse("goals", &a(&["list", "--context", "work"])).unwrap();
        assert!(matches!(
            c,
            ControlCommand::AllenGoalsList {
                context: Some(_),
                owner: false
            }
        ));
    }
}
