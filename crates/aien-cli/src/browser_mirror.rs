//! Headless-Chrome Cockpit mirror over the Chrome DevTools Protocol (CDP).
//!
//! This is the Rust replacement for the removed Python script
//! `basecamp/scripts/browser_mirror_test.py` (sovereign-core #180/#181). It
//! carries no new dependencies: the websocket client, the tiny HTTP fetch and
//! the base64 decoder are written here against `std::net`.
//!
//! Modes (also exposed by the `aien-browser-mirror` binary and `aien --browser`):
//!
//! * `test`       Cockpit self-test: hydration, chat stream, navigation, three screenshots.
//! * `mentor`     Send one prompt to the Cockpit chat, wait for the reply, screenshot it.
//! * `screenshot` Open any URL, capture a PNG, record the document title.
//! * `eval`       Open any URL and evaluate a JavaScript expression (value returned as JSON).
//!
//! Every mode spawns its own Chrome with a private profile directory and kills it
//! when done, whatever happened in between.

use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Default Cockpit address (spark-cockpit-rs).
pub const DEFAULT_URL: &str = "http://127.0.0.1:18095";

/// How long to wait for Chrome to publish its debugging port.
const CHROME_BOOT_TIMEOUT: Duration = Duration::from_secs(20);
/// Per-command CDP reply timeout.
const CDP_REPLY_TIMEOUT: Duration = Duration::from_secs(30);
/// Largest websocket message we accept (screenshots are a few MB).
const MAX_WS_MESSAGE: usize = 64 * 1024 * 1024;

/// Where screenshots and reports land when no `--out-dir` is given.
pub fn default_out_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join("basecamp").join("ui-tests")
}

/// Candidate Chrome binaries, first match wins. `AIEN_CHROME` overrides.
pub fn chrome_candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("AIEN_CHROME") {
        if !p.is_empty() {
            v.push(p);
        }
    }
    for name in [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
    ] {
        v.push(name.to_string());
    }
    v
}

/// Returns the first Chrome binary that answers `--version`, if any.
pub fn find_chrome() -> Option<String> {
    chrome_candidates().into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// base64 (standard alphabet, padding tolerated)
// ---------------------------------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Decodes standard base64. Whitespace is skipped; anything else is an error.
pub fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some(u32::from(c - b'A')),
            b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in text.as_bytes() {
        if c.is_ascii_whitespace() || c == b'=' {
            continue;
        }
        let v = val(c).ok_or_else(|| format!("base64: invalid byte 0x{:02x}", c))?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Minimal HTTP GET (used for /json on the DevTools port)
// ---------------------------------------------------------------------------

fn http_get(host: &str, port: u16, path: &str, timeout: Duration) -> Result<String, String> {
    let mut s = TcpStream::connect((host, port)).map_err(|e| format!("connect: {}", e))?;
    s.set_read_timeout(Some(timeout))
        .map_err(|e| format!("timeout: {}", e))?;
    let req = format!(
        "GET {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        path, host, port
    );
    s.write_all(req.as_bytes())
        .map_err(|e| format!("write: {}", e))?;
    // Chrome honours Content-Length but keeps the connection open, so read the
    // headers first and then exactly the announced body.
    let mut head = Vec::new();
    let mut one = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        let n = s.read(&mut one).map_err(|e| format!("read: {}", e))?;
        if n == 0 {
            return Err("http: connection closed in headers".to_string());
        }
        head.push(one[0]);
        if head.len() > 64 * 1024 {
            return Err("http: header too long".to_string());
        }
    }
    let head = String::from_utf8_lossy(&head).to_string();
    let status = head.lines().next().unwrap_or("");
    if !status.contains(" 200 ") {
        return Err(format!("http: {}", status));
    }
    let len = head.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.trim().eq_ignore_ascii_case("content-length") {
            v.trim().parse::<usize>().ok()
        } else {
            None
        }
    });
    let mut body = Vec::new();
    match len {
        Some(n) => {
            body.resize(n, 0);
            s.read_exact(&mut body)
                .map_err(|e| format!("read body: {}", e))?;
        }
        None => {
            s.read_to_end(&mut body)
                .map_err(|e| format!("read: {}", e))?;
        }
    }
    Ok(String::from_utf8_lossy(&body).to_string())
}

// ---------------------------------------------------------------------------
// Websocket client (RFC 6455, client side, text frames)
// ---------------------------------------------------------------------------

struct WebSocket {
    stream: TcpStream,
    rng: u64,
}

impl WebSocket {
    /// Connects to `ws://host:port/path` and completes the HTTP upgrade.
    fn connect(url: &str) -> Result<Self, String> {
        let rest = url
            .strip_prefix("ws://")
            .ok_or_else(|| format!("websocket: unsupported url {}", url))?;
        let (hostport, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (host, port) = match hostport.rsplit_once(':') {
            Some((h, p)) => (h, p.parse::<u16>().map_err(|e| format!("port: {}", e))?),
            None => (hostport, 80),
        };
        let stream = TcpStream::connect((host, port)).map_err(|e| format!("connect: {}", e))?;
        stream
            .set_read_timeout(Some(CDP_REPLY_TIMEOUT))
            .map_err(|e| format!("timeout: {}", e))?;
        stream.set_nodelay(true).ok();
        let mut ws = WebSocket {
            stream,
            rng: now_nanos() as u64 ^ 0x9E37_79B9_7F4A_7C15,
        };
        let mut key = [0u8; 16];
        for b in key.iter_mut() {
            *b = ws.next_byte();
        }
        let req = format!(
            "GET {} HTTP/1.1\r\nHost: {}:{}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
             Sec-WebSocket-Key: {}\r\nSec-WebSocket-Version: 13\r\n\r\n",
            path,
            host,
            port,
            base64_encode(&key)
        );
        ws.stream
            .write_all(req.as_bytes())
            .map_err(|e| format!("handshake write: {}", e))?;
        // Read the response headers byte by byte up to the blank line.
        let mut head = Vec::new();
        let mut one = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            let n = ws
                .stream
                .read(&mut one)
                .map_err(|e| format!("handshake read: {}", e))?;
            if n == 0 {
                return Err("handshake: connection closed".to_string());
            }
            head.push(one[0]);
            if head.len() > 16 * 1024 {
                return Err("handshake: header too long".to_string());
            }
        }
        let text = String::from_utf8_lossy(&head);
        let status = text.lines().next().unwrap_or("");
        if !status.contains(" 101 ") {
            return Err(format!("handshake refused: {}", status));
        }
        Ok(ws)
    }

    fn next_byte(&mut self) -> u8 {
        // xorshift64*: only used for mask keys and the handshake nonce.
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
    }

    fn send_frame(&mut self, opcode: u8, payload: &[u8]) -> Result<(), String> {
        let mut frame = Vec::with_capacity(payload.len() + 14);
        frame.push(0x80 | opcode);
        let len = payload.len();
        if len < 126 {
            frame.push(0x80 | len as u8);
        } else if len <= 0xFFFF {
            frame.push(0x80 | 126);
            frame.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            frame.push(0x80 | 127);
            frame.extend_from_slice(&(len as u64).to_be_bytes());
        }
        let mask = [
            self.next_byte(),
            self.next_byte(),
            self.next_byte(),
            self.next_byte(),
        ];
        frame.extend_from_slice(&mask);
        frame.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        self.stream
            .write_all(&frame)
            .map_err(|e| format!("websocket write: {}", e))
    }

    fn send_text(&mut self, text: &str) -> Result<(), String> {
        self.send_frame(0x1, text.as_bytes())
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), String> {
        self.stream
            .read_exact(buf)
            .map_err(|e| format!("websocket read: {}", e))
    }

    /// Reads one complete text message, answering pings and honouring close.
    fn recv_text(&mut self) -> Result<String, String> {
        let mut message: Vec<u8> = Vec::new();
        loop {
            let mut hdr = [0u8; 2];
            self.read_exact(&mut hdr)?;
            let fin = hdr[0] & 0x80 != 0;
            let opcode = hdr[0] & 0x0F;
            let masked = hdr[1] & 0x80 != 0;
            let mut len = u64::from(hdr[1] & 0x7F);
            if len == 126 {
                let mut b = [0u8; 2];
                self.read_exact(&mut b)?;
                len = u64::from(u16::from_be_bytes(b));
            } else if len == 127 {
                let mut b = [0u8; 8];
                self.read_exact(&mut b)?;
                len = u64::from_be_bytes(b);
            }
            if len as usize > MAX_WS_MESSAGE || message.len() + len as usize > MAX_WS_MESSAGE {
                return Err("websocket: message too large".to_string());
            }
            let mut mask = [0u8; 4];
            if masked {
                self.read_exact(&mut mask)?;
            }
            let mut payload = vec![0u8; len as usize];
            self.read_exact(&mut payload)?;
            if masked {
                for (i, b) in payload.iter_mut().enumerate() {
                    *b ^= mask[i % 4];
                }
            }
            match opcode {
                0x0..=0x2 => {
                    message.extend_from_slice(&payload);
                    if fin {
                        return String::from_utf8(message)
                            .map_err(|e| format!("websocket: bad utf-8: {}", e));
                    }
                }
                0x8 => {
                    self.send_frame(0x8, &payload).ok();
                    return Err("websocket: closed by peer".to_string());
                }
                0x9 => self.send_frame(0xA, &payload)?,
                0xA => {}
                other => return Err(format!("websocket: unknown opcode {}", other)),
            }
        }
    }

    fn close(&mut self) {
        self.send_frame(0x8, &1000u16.to_be_bytes()).ok();
    }
}

// ---------------------------------------------------------------------------
// Chrome process + CDP session
// ---------------------------------------------------------------------------

struct ChromeProcess {
    child: Child,
    profile_dir: PathBuf,
    port: u16,
}

impl ChromeProcess {
    fn spawn(binary: &str, url: &str) -> Result<Self, String> {
        let profile_dir = std::env::temp_dir().join(format!(
            "aien-browser-mirror-{}-{}",
            std::process::id(),
            now_nanos()
        ));
        fs::create_dir_all(&profile_dir).map_err(|e| format!("profile dir: {}", e))?;
        let child = Command::new(binary)
            .arg("--headless=new")
            .arg("--remote-debugging-port=0")
            .arg(format!("--user-data-dir={}", profile_dir.display()))
            .arg("--remote-allow-origins=*")
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--disable-dev-shm-usage")
            .arg("--disable-gpu")
            .arg("--no-sandbox")
            .arg("--window-size=1280,900")
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {}: {}", binary, e))?;
        let mut me = ChromeProcess {
            child,
            profile_dir,
            port: 0,
        };
        // Chrome writes "<port>\n<browser ws path>" once the debugger listens.
        let deadline = Instant::now() + CHROME_BOOT_TIMEOUT;
        let marker = me.profile_dir.join("DevToolsActivePort");
        loop {
            if let Ok(text) = fs::read_to_string(&marker) {
                if let Some(p) = text
                    .lines()
                    .next()
                    .and_then(|l| l.trim().parse::<u16>().ok())
                {
                    me.port = p;
                    break;
                }
            }
            if let Ok(Some(status)) = me.child.try_wait() {
                return Err(format!("chrome exited early: {}", status));
            }
            if Instant::now() >= deadline {
                return Err("chrome did not publish DevToolsActivePort in time".to_string());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(me)
    }

    /// Finds the websocket URL of the first page target.
    fn page_ws_url(&self) -> Result<String, String> {
        let deadline = Instant::now() + CHROME_BOOT_TIMEOUT;
        loop {
            if let Ok(body) = http_get("127.0.0.1", self.port, "/json", Duration::from_secs(5)) {
                if let Ok(Value::Array(targets)) = serde_json::from_str::<Value>(&body) {
                    for t in &targets {
                        if t.get("type").and_then(Value::as_str) == Some("page") {
                            if let Some(ws) = t.get("webSocketDebuggerUrl").and_then(Value::as_str)
                            {
                                return Ok(ws.to_string());
                            }
                        }
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err("chrome exposed no page target".to_string());
            }
            std::thread::sleep(Duration::from_millis(150));
        }
    }
}

impl Drop for ChromeProcess {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
        fs::remove_dir_all(&self.profile_dir).ok();
    }
}

/// One CDP page session.
pub struct CdpClient {
    ws: WebSocket,
    next_id: u64,
}

impl CdpClient {
    fn connect(ws_url: &str) -> Result<Self, String> {
        Ok(CdpClient {
            ws: WebSocket::connect(ws_url)?,
            next_id: 1,
        })
    }

    /// Sends a CDP command and returns its `result` object.
    pub fn send(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({"id": id, "method": method, "params": params});
        self.ws.send_text(&msg.to_string())?;
        let deadline = Instant::now() + CDP_REPLY_TIMEOUT;
        loop {
            if Instant::now() >= deadline {
                return Err(format!("cdp: {} timed out", method));
            }
            let text = self.ws.recv_text()?;
            let v: Value = serde_json::from_str(&text).map_err(|e| format!("cdp json: {}", e))?;
            if v.get("id").and_then(Value::as_u64) != Some(id) {
                continue; // an event or a stale reply
            }
            if let Some(err) = v.get("error") {
                return Err(format!("cdp {}: {}", method, err));
            }
            return Ok(v.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    /// Evaluates JavaScript and returns the value (`returnByValue`).
    pub fn eval(&mut self, expr: &str, await_promise: bool) -> Result<Value, String> {
        let r = self.send(
            "Runtime.evaluate",
            json!({"expression": expr, "returnByValue": true, "awaitPromise": await_promise}),
        )?;
        if let Some(ex) = r.get("exceptionDetails") {
            let text = ex
                .get("exception")
                .and_then(|e| e.get("description"))
                .and_then(Value::as_str)
                .or_else(|| ex.get("text").and_then(Value::as_str))
                .unwrap_or("javascript exception");
            return Err(format!("eval: {}", text));
        }
        Ok(r.get("result")
            .and_then(|x| x.get("value"))
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Captures a PNG screenshot into `out_dir/filename`; returns the path.
    pub fn screenshot(&mut self, out_dir: &Path, filename: &str) -> Result<PathBuf, String> {
        let r = self.send("Page.captureScreenshot", json!({"format": "png"}))?;
        let data = r
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| "screenshot: no data".to_string())?;
        let bytes = base64_decode(data)?;
        fs::create_dir_all(out_dir).map_err(|e| format!("out dir: {}", e))?;
        let path = out_dir.join(filename);
        fs::write(&path, bytes).map_err(|e| format!("write {}: {}", path.display(), e))?;
        Ok(path)
    }

    /// Waits until the page's `document.readyState` is complete (or the deadline passes).
    /// Chrome's first document is `about:blank`, already complete, so unless that is the
    /// target the page must also have left it.
    fn wait_loaded(&mut self, url: &str, timeout: Duration) {
        let blank_ok = url == "about:blank";
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            let state = self
                .eval("[document.readyState, location.href]", false)
                .unwrap_or(Value::Null);
            if state[0] == "complete" && (blank_ok || state[1] != "about:blank") {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// A live browser: Chrome process plus the page session. Dropping it kills Chrome.
pub struct Mirror {
    _chrome: ChromeProcess,
    pub cdp: CdpClient,
}

impl Mirror {
    /// Spawns Chrome on `url` and attaches to its page.
    pub fn open(url: &str) -> Result<Self, String> {
        let binary = find_chrome().ok_or_else(|| {
            "no Chrome found (set AIEN_CHROME or install google-chrome/chromium)".to_string()
        })?;
        let chrome = ChromeProcess::spawn(&binary, url)?;
        let ws_url = chrome.page_ws_url()?;
        let mut cdp = CdpClient::connect(&ws_url)?;
        cdp.send("Page.enable", json!({}))?;
        cdp.send("Runtime.enable", json!({}))?;
        cdp.wait_loaded(url, Duration::from_secs(15));
        Ok(Mirror {
            _chrome: chrome,
            cdp,
        })
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        self.cdp.ws.close();
    }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

fn write_report(out_dir: &Path, name: &str, report: &Value) -> Result<PathBuf, String> {
    fs::create_dir_all(out_dir).map_err(|e| format!("out dir: {}", e))?;
    let path = out_dir.join(name);
    let text = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    fs::write(&path, text).map_err(|e| format!("write {}: {}", path.display(), e))?;
    Ok(path)
}

fn agent_messages_js() -> &'static str {
    // Current Cockpit (spark-cockpit-rs/static): agent replies are rows with
    // class "message-row agent-row". The old ".chat-msg.assistant" selector is
    // kept as a fallback for older builds.
    "Array.from(document.querySelectorAll('.message-row.agent-row, .chat-msg.assistant'))\
       .map(el => el.textContent.trim())"
}

fn send_chat_js(prompt: &str) -> String {
    format!(
        "(() => {{ const inp = document.getElementById('chat-input'); if (!inp) return 'no-input';\
           inp.value = {p}; inp.dispatchEvent(new Event('input', {{bubbles: true}}));\
           const btn = document.getElementById('btn-send');\
           if (btn) {{ btn.click(); return 'clicked'; }}\
           if (typeof sendMessage === 'function') {{ sendMessage(); return 'called'; }}\
           return 'no-send'; }})()",
        p = Value::String(prompt.to_string())
    )
}

/// Polls agent rows until a reply appears and stops growing (or `timeout` passes).
fn wait_for_reply(cdp: &mut CdpClient, before: usize, timeout: Duration) -> (String, bool) {
    let deadline = Instant::now() + timeout;
    let mut last = String::new();
    let mut stable = 0u32;
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(500));
        let msgs = cdp.eval(agent_messages_js(), false).unwrap_or(Value::Null);
        let texts: Vec<String> = msgs
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if texts.len() > before {
            let current = texts.last().cloned().unwrap_or_default();
            if !current.is_empty() && current == last {
                stable += 1;
                if stable >= 4 {
                    return (current, true);
                }
            } else {
                stable = 0;
            }
            last = current;
        }
    }
    (last, false)
}

/// Cockpit self-test: hydration, chat stream, navigation, three screenshots.
pub fn run_test(url: &str, out_dir: &Path) -> Value {
    let mut report = json!({
        "mode": "test",
        "timestamp": now_secs(),
        "target_url": url,
        "assertions": {},
        "screenshots": [],
        "status": "in_progress"
    });
    let push_shot = |report: &mut Value, p: PathBuf| {
        if let Some(a) = report["screenshots"].as_array_mut() {
            a.push(json!(p.display().to_string()));
        }
    };
    let mut mirror = match Mirror::open(url) {
        Ok(m) => m,
        Err(e) => {
            report["status"] = json!("FAILED");
            report["error"] = json!(e);
            if let Ok(p) = write_report(out_dir, "report.json", &report) {
                report["report_path"] = json!(p.display().to_string());
            }
            return report;
        }
    };
    let cdp = &mut mirror.cdp;

    // 1. Hydration: title, chat input and navigation present.
    let title = cdp
        .eval("document.title", false)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let has_input = cdp
        .eval("!!document.getElementById('chat-input')", false)
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let has_nav = cdp
        .eval(
            "!!(document.querySelector('[data-view]') || document.querySelector('[data-tab]'))",
            false,
        )
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let hydrated = !title.is_empty() && has_input && has_nav;
    report["assertions"]["initial_page_hydration"] = json!({
        "pass": hydrated, "title": title, "chat_input": has_input, "navigation": has_nav
    });
    if let Ok(p) = cdp.screenshot(out_dir, "cockpit_init.png") {
        push_shot(&mut report, p);
    }

    // 2. Chat stream: send a prompt and wait for a stable agent reply.
    let before = cdp
        .eval(&format!("{}.length", agent_messages_js()), false)
        .ok()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let sent = cdp
        .eval(
            &send_chat_js("Hello AIEN, please report current sovereign status."),
            false,
        )
        .unwrap_or(Value::Null);
    let (reply, stable) = if sent.as_str() == Some("clicked") || sent.as_str() == Some("called") {
        wait_for_reply(cdp, before, Duration::from_secs(45))
    } else {
        (String::new(), false)
    };
    let chat_pass = !reply.is_empty() && stable;
    report["assertions"]["chat_live_stream"] = json!({
        "pass": chat_pass, "send": sent, "reply_chars": reply.chars().count(), "stable": stable
    });
    if let Ok(p) = cdp.screenshot(out_dir, "cockpit_chat_stream.png") {
        push_shot(&mut report, p);
    }

    // 3. Navigation: click every nav control, confirm a visible view switches.
    let nav = cdp
        .eval(
            "(() => { const btns = Array.from(document.querySelectorAll('[data-view], [data-tab]'));\
               const seen = []; for (const b of btns) { b.click();\
               const v = b.dataset.view || b.dataset.tab;\
               const el = document.getElementById('view-' + v) || document.getElementById('pane-' + v)\
                 || document.getElementById(v);\
               seen.push({view: v, present: !!el, visible: !!el && el.offsetParent !== null}); }\
               return seen; })()",
            false,
        )
        .unwrap_or(Value::Null);
    let nav_items = nav.as_array().cloned().unwrap_or_default();
    let nav_pass = !nav_items.is_empty()
        && nav_items
            .iter()
            .all(|i| i.get("present").and_then(Value::as_bool).unwrap_or(false));
    report["assertions"]["tab_navigation_and_actions"] = json!({"pass": nav_pass, "views": nav});
    // Back to chat for the final frame.
    cdp.eval(
        "document.querySelector('[data-view=\"chat\"], [data-tab=\"chat\"]')?.click()",
        false,
    )
    .ok();
    if let Ok(p) = cdp.screenshot(out_dir, "cockpit_actions.png") {
        push_shot(&mut report, p);
    }

    let all = hydrated && chat_pass && nav_pass;
    report["status"] = json!(if all { "PASSED" } else { "FAILED" });
    if let Ok(p) = write_report(out_dir, "report.json", &report) {
        report["report_path"] = json!(p.display().to_string());
    }
    report
}

/// Mentor session: one prompt into the Cockpit chat, wait for the reply, screenshot.
pub fn run_mentor(url: &str, prompt: &str, out_dir: &Path) -> Value {
    let mut report = json!({
        "mode": "mentor",
        "timestamp": now_secs(),
        "target_url": url,
        "prompt": prompt,
        "response": "",
        "screenshot": "",
        "status": "in_progress"
    });
    let mut mirror = match Mirror::open(url) {
        Ok(m) => m,
        Err(e) => {
            report["status"] = json!("FAILED");
            report["error"] = json!(e);
            if let Ok(p) = write_report(out_dir, "mentor_response.json", &report) {
                report["report_path"] = json!(p.display().to_string());
            }
            return report;
        }
    };
    let cdp = &mut mirror.cdp;
    cdp.eval(
        "document.querySelector('[data-view=\"chat\"], [data-tab=\"chat\"]')?.click()",
        false,
    )
    .ok();
    let before = cdp
        .eval(&format!("{}.length", agent_messages_js()), false)
        .ok()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let sent = cdp
        .eval(&send_chat_js(prompt), false)
        .unwrap_or(Value::Null);
    report["send"] = sent;
    let (reply, stable) = wait_for_reply(cdp, before, Duration::from_secs(120));
    report["response"] = json!(reply);
    report["stable"] = json!(stable);
    if let Ok(p) = cdp.screenshot(out_dir, "cockpit_mentor_response.png") {
        report["screenshot"] = json!(p.display().to_string());
    }
    report["status"] = json!(if reply.chars().count() > 50 {
        "COMPLETED"
    } else {
        "FAILED"
    });
    if let Ok(p) = write_report(out_dir, "mentor_response.json", &report) {
        report["report_path"] = json!(p.display().to_string());
    }
    report
}

/// Opens `url`, records the title and saves `filename` (PNG) under `out_dir`.
pub fn run_screenshot(url: &str, out_dir: &Path, filename: &str) -> Value {
    let mut mirror = match Mirror::open(url) {
        Ok(m) => m,
        Err(e) => return json!({"mode": "screenshot", "status": "error", "error": e}),
    };
    let title = mirror
        .cdp
        .eval("document.title", false)
        .unwrap_or(Value::Null);
    match mirror.cdp.screenshot(out_dir, filename) {
        Ok(p) => json!({
            "mode": "screenshot", "status": "ok", "target_url": url,
            "title": title, "screenshot": p.display().to_string()
        }),
        Err(e) => json!({"mode": "screenshot", "status": "error", "error": e, "title": title}),
    }
}

/// Opens `url` and evaluates `expr`, returning its value.
pub fn run_eval(url: &str, expr: &str) -> Value {
    let mut mirror = match Mirror::open(url) {
        Ok(m) => m,
        Err(e) => return json!({"mode": "eval", "status": "error", "error": e}),
    };
    match mirror.cdp.eval(expr, true) {
        Ok(v) => json!({"mode": "eval", "status": "ok", "target_url": url, "value": v}),
        Err(e) => json!({"mode": "eval", "status": "error", "error": e}),
    }
}

/// Entry point shared by the `browser` tool and the binary.
///
/// `action` is one of `test`, `mentor`, `screenshot`, `eval`. `arg` is the
/// mentor prompt or the eval expression. Unknown actions report an error.
pub fn dispatch(action: &str, arg: Option<&str>, url: &str, out_dir: &Path) -> Value {
    match action {
        "test" => run_test(url, out_dir),
        "mentor" => run_mentor(
            url,
            arg.unwrap_or("AIEN, report self-improvement status."),
            out_dir,
        ),
        "screenshot" => run_screenshot(url, out_dir, arg.unwrap_or("screenshot.png")),
        "eval" => run_eval(url, arg.unwrap_or("document.title")),
        other => json!({
            "status": "error",
            "error": format!("unknown browser action '{}' (use test, mentor, screenshot or eval)", other)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        for s in ["", "f", "fo", "foo", "foob", "fooba", "foobar"] {
            let enc = base64_encode(s.as_bytes());
            assert_eq!(base64_decode(&enc).unwrap(), s.as_bytes());
        }
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_decode("Zm9v\nYmFy").unwrap(), b"foobar");
        assert!(base64_decode("Zm9v*").is_err());
    }

    #[test]
    fn unknown_action_is_an_error() {
        let v = dispatch("fly", None, DEFAULT_URL, Path::new("/nonexistent"));
        assert_eq!(v["status"], "error");
        assert!(v["error"].as_str().unwrap().contains("fly"));
    }

    /// Spawns a real headless Chrome on a local stub page, screenshots it and
    /// evaluates the title. Skipped (with a note) when no Chrome is installed,
    /// so CI without a browser still passes.
    #[test]
    fn stub_page_screenshot_and_eval() {
        if find_chrome().is_none() {
            eprintln!("skipping stub_page_screenshot_and_eval: no Chrome binary found");
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let page = dir.path().join("stub.html");
        fs::write(
            &page,
            "<!doctype html><html><head><title>AIEN Stub</title></head>\
             <body><h1 id=\"h\">mirror</h1><textarea id=\"chat-input\"></textarea></body></html>",
        )
        .unwrap();
        let url = format!("file://{}", page.display());
        let out = dir.path().join("shots");

        let shot = run_screenshot(&url, &out, "stub.png");
        assert_eq!(shot["status"], "ok", "{}", shot);
        assert_eq!(shot["title"], "AIEN Stub");
        let bytes = fs::read(out.join("stub.png")).unwrap();
        assert!(bytes.len() > 100, "png too small: {} bytes", bytes.len());
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");

        let ev = run_eval(
            &url,
            "document.getElementById('h').textContent + ':' + 2 * 21",
        );
        assert_eq!(ev["status"], "ok", "{}", ev);
        assert_eq!(ev["value"], "mirror:42");
    }
}
