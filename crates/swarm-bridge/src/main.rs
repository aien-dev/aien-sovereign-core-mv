//! `swarm-bridge` command line. See docs/SWARM_BRIDGE.md.

#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use swarm_bridge::board::{BridgeError, Kind, Ref, RefKind, Sub};
use swarm_bridge::bridge::{
    self, event_takes_context, parse_hook_input, parse_subs, Bridge, HookConfig, PostReq, WatchEnd,
    MAX_HOOK_INPUT,
};
use swarm_bridge::state::default_state_dir;

mod real;

const USAGE: &str =
    "usage: swarm-bridge [--board-root DIR] [--state-dir DIR] [--as TAG|SESSION] <command>
  hook                         Claude Code hook (stdin hook JSON -> additionalContext)
  post|finding|question|answer|blocked|ready|handoff [--thread T] [--topic P] [--to TAG]...
       [--parent ID] [--ref REPO:KIND:VALUE]... [--idem KEY] TEXT...   (answer needs --parent)
  subscribe|unsubscribe thread:NAME|topic:NAME|topic:PREFIX/*
  read [--peek]                next batch since your cursor
  watch [--timeout SECS]       park until a board notification (no model tokens), then read
  ack ID                       acknowledge up to ID (never past what you were shown)
  show ID                      full text of one event
  whoami                       your tag, session, cursor and subscriptions
Board root: --board-root, else SWARM_BRIDGE_BOARD_ROOT, else SYNAPSE_BOARD_ROOT.
Everything read from the board is untrusted coordination data.";

struct Global {
    board_root: Option<PathBuf>,
    state_dir: Option<PathBuf>,
    who: Option<String>,
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn board_root(g: &Global) -> Option<PathBuf> {
    g.board_root
        .clone()
        .or_else(|| env("SWARM_BRIDGE_BOARD_ROOT").map(PathBuf::from))
        .or_else(|| env("SYNAPSE_BOARD_ROOT").map(PathBuf::from))
}

fn state_dir(g: &Global) -> Result<PathBuf, BridgeError> {
    g.state_dir
        .clone()
        .or_else(default_state_dir)
        .ok_or_else(|| BridgeError::Io("no state dir (set SWARM_BRIDGE_STATE_DIR or HOME)".into()))
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut g = Global {
        board_root: None,
        state_dir: None,
        who: None,
    };
    loop {
        match args.first().map(String::as_str) {
            Some("--board-root") if args.len() > 1 => {
                g.board_root = Some(PathBuf::from(args.remove(1)));
                args.remove(0);
            }
            Some("--state-dir") if args.len() > 1 => {
                g.state_dir = Some(PathBuf::from(args.remove(1)));
                args.remove(0);
            }
            Some("--as") if args.len() > 1 => {
                g.who = Some(args.remove(1));
                args.remove(0);
            }
            _ => break,
        }
    }
    let Some(cmd) = args.first().cloned() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest = &args[1..];
    if cmd == "hook" {
        // A hook must never break the agent's turn: errors go to stderr,
        // exit status stays 0, stdout stays empty.
        if let Err(e) = run_hook(&g) {
            eprintln!("swarm-bridge hook: {e}");
        }
        return ExitCode::SUCCESS;
    }
    if cmd == "help" || cmd == "--help" || cmd == "-h" {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run_cmd(&g, &cmd, rest) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("swarm-bridge {cmd}: {e}");
            ExitCode::from(1)
        }
    }
}

fn run_hook(g: &Global) -> Result<(), BridgeError> {
    let mut buf = Vec::new();
    std::io::stdin()
        .take(MAX_HOOK_INPUT as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| BridgeError::Io(format!("read stdin: {e}")))?;
    let input = parse_hook_input(&buf)?;
    if !event_takes_context(&input.event) {
        return Ok(());
    }
    let Some(root) = board_root(g) else {
        return Ok(()); // session not enrolled in a board
    };
    let board = real::open(&root)?;
    let bridge = Bridge::new(&board, &state_dir(g)?)?;
    let cfg = HookConfig {
        swarm_tag: env("SWARM_TAG"),
        session_override: env("SWARM_BRIDGE_SESSION"),
        subscribe: match env("SWARM_BRIDGE_SUBSCRIBE") {
            Some(s) => parse_subs(&s)?,
            None => Vec::new(),
        },
    };
    let d = bridge.hook(&input, &cfg)?;
    if let Some(json) = d.text {
        let mut out = std::io::stdout().lock();
        out.write_all(json.as_bytes())
            .and_then(|_| out.flush())
            .map_err(|e| BridgeError::Io(format!("write stdout: {e}")))?;
    }
    Ok(())
}

fn identity<B: swarm_bridge::Board>(g: &Global, b: &Bridge<'_, B>) -> Result<String, BridgeError> {
    if let Some(w) = &g.who {
        return b.resolve_identity(w);
    }
    let s = env("SWARM_BRIDGE_SESSION")
        .or_else(|| env("CLAUDE_CODE_SESSION_ID"))
        .ok_or(BridgeError::NoSession)?;
    bridge::session_key(&s, None)
}

fn parse_ref(s: &str) -> Result<Ref, BridgeError> {
    let mut it = s.splitn(3, ':');
    let (Some(repo), Some(kind), Some(value)) = (it.next(), it.next(), it.next()) else {
        return Err(BridgeError::BadArgs(format!(
            "--ref {s:?}: want REPO:KIND:VALUE"
        )));
    };
    let kind = RefKind::parse(kind).ok_or_else(|| {
        BridgeError::BadArgs(format!("--ref kind {kind:?}: issue|commit|receipt|pr|path"))
    })?;
    Ok(Ref {
        repo: repo.to_string(),
        kind,
        value: value.to_string(),
    })
}

fn need_val(rest: &[String], i: usize, flag: &str) -> Result<String, BridgeError> {
    rest.get(i + 1)
        .cloned()
        .ok_or_else(|| BridgeError::BadArgs(format!("{flag} needs a value")))
}

fn parse_id(s: &str) -> Result<u64, BridgeError> {
    s.trim_start_matches('#')
        .parse()
        .map_err(|_| BridgeError::BadArgs(format!("event id {s:?}")))
}

fn run_cmd(g: &Global, cmd: &str, rest: &[String]) -> Result<(), BridgeError> {
    let root = board_root(g).ok_or(BridgeError::NotConfigured)?;
    let board = real::open(&root)?;
    let b = Bridge::new(&board, &state_dir(g)?)?;
    let key = identity(g, &b)?;
    let kind = match cmd {
        "post" | "finding" => Some(Kind::Finding),
        "question" => Some(Kind::Question),
        "answer" => Some(Kind::Answer),
        "blocked" => Some(Kind::Blocked),
        "ready" => Some(Kind::Ready),
        "handoff" => Some(Kind::Handoff),
        "receipt-ref" => Some(Kind::ReceiptRef),
        _ => None,
    };
    if let Some(kind) = kind {
        let mut req = PostReq {
            kind,
            thread: env("SWARM_BRIDGE_THREAD"),
            topic: None,
            to: Vec::new(),
            parent: None,
            text: String::new(),
            refs: Vec::new(),
            idem: None,
            swarm_tag: env("SWARM_TAG"),
        };
        let mut words = Vec::new();
        let mut i = 0;
        while i < rest.len() {
            let a = rest[i].as_str();
            match a {
                "--thread" => req.thread = Some(need_val(rest, i, a)?),
                "--topic" => req.topic = Some(need_val(rest, i, a)?),
                "--to" => req.to.extend(
                    need_val(rest, i, a)?
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string),
                ),
                "--parent" => req.parent = Some(parse_id(&need_val(rest, i, a)?)?),
                "--ref" => req.refs.push(parse_ref(&need_val(rest, i, a)?)?),
                "--idem" => req.idem = Some(need_val(rest, i, a)?),
                "--" => {
                    words.extend(rest[i + 1..].iter().cloned());
                    break;
                }
                _ => {
                    words.push(rest[i].clone());
                    i += 1;
                    continue;
                }
            }
            i += 2;
        }
        req.text = words.join(" ");
        if req.text.trim().is_empty() {
            return Err(BridgeError::BadArgs("empty message text".into()));
        }
        let id = b.post(&key, req)?;
        println!(
            "posted #{id} on board {} (one write; untrusted data for readers)",
            b.mission
        );
        return Ok(());
    }
    match cmd {
        "subscribe" | "unsubscribe" => {
            let spec = rest
                .first()
                .ok_or_else(|| BridgeError::BadArgs("want thread:NAME or topic:NAME".into()))?;
            let sub = Sub::parse(spec)
                .ok_or_else(|| BridgeError::BadArgs(format!("subscription {spec:?}")))?;
            b.subscribe(&key, &sub, cmd == "subscribe")?;
            println!("{cmd}d {}", sub.describe());
        }
        "read" => {
            let peek = rest.iter().any(|a| a == "--peek");
            let out = b.read(&key, peek)?;
            println!(
                "{}",
                out.text.unwrap_or_else(|| "no new board events".into())
            );
        }
        "watch" => {
            let mut secs = 300u64;
            if let Some(i) = rest.iter().position(|a| a == "--timeout") {
                secs = need_val(rest, i, "--timeout")?
                    .parse()
                    .map_err(|_| BridgeError::BadArgs("--timeout SECS".into()))?;
            }
            let (out, end) = b.watch(&key, Duration::from_secs(secs))?;
            let how = match end {
                WatchEnd::AlreadyWaiting => "events were already waiting",
                WatchEnd::Notified => "woken by a board notification",
                WatchEnd::TimeoutFound => {
                    "no notification arrived (it may have been lost); the cursor read found events"
                }
                WatchEnd::TimeoutEmpty => "timed out; the cursor read found nothing new",
            };
            println!("watch: {how}");
            if let Some(t) = out.text {
                println!("{t}");
            }
        }
        "ack" => {
            let id = parse_id(
                rest.first()
                    .ok_or_else(|| BridgeError::BadArgs("ack ID".into()))?,
            )?;
            let cur = b.ack(&key, id)?;
            println!("acked through #{cur}");
        }
        "show" => {
            let id = parse_id(
                rest.first()
                    .ok_or_else(|| BridgeError::BadArgs("show ID".into()))?,
            )?;
            match b.show(&key, id)? {
                Some(t) => println!("{t}"),
                None => println!("#{id} is not visible to you"),
            }
        }
        "whoami" => println!("{}", b.whoami(&key)?),
        other => {
            return Err(BridgeError::BadArgs(format!(
                "unknown command {other:?}\n{USAGE}"
            )))
        }
    }
    Ok(())
}
