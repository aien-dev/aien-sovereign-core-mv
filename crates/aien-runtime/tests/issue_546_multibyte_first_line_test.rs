//! Issue #546: `check_file_proposal` must refuse, never panic, when the first
//! reply line has a multi-byte UTF-8 character across byte 9 (the length of
//! the ASCII key `filename:`). Also a seeded property loop over multi-byte
//! replies so any sibling slicing panic in the parser shows up here.

use aien_runtime::{check_file_proposal, parse_file_proposal};
use std::panic::{catch_unwind, AssertUnwindSafe};

const KEY: &str = "filename:";

fn no_panic(text: &str) -> Result<Result<aien_runtime::FileProposal, String>, String> {
    catch_unwind(AssertUnwindSafe(|| check_file_proposal(text))).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    })
}

/// Byte 9 falls inside a character in each of these first lines.
#[test]
fn multibyte_char_across_byte_nine_is_refused_not_panicking() {
    for first in [
        "ééééé",     // 2-byte chars, byte 9 inside the fifth
        "filenameé", // ASCII then a 2-byte char at byte 8
        "abcdefg中", // 3-byte char at byte 7
        "🎯🎯🎯",    // 4-byte chars, byte 9 inside the third
        "é🙂 notes",
        "éfilename: a.md", // key shifted by a leading multi-byte char
    ] {
        let reply = format!("{first}\nbody\n");
        let got = no_panic(&reply).unwrap_or_else(|p| panic!("{first:?} panicked: {p}"));
        let err = got.expect_err("must be refused");
        assert!(err.contains("no filename line"), "{first:?}: {err}");
        assert!(parse_file_proposal(&reply).is_none());
    }
}

/// The fix must not change what is accepted.
#[test]
fn ascii_key_still_accepted_with_multibyte_elsewhere() {
    let p = check_file_proposal("FILENAME: safe.md\nbody\n").unwrap();
    assert_eq!((p.path.as_str(), p.content.as_str()), ("safe.md", "body\n"));
    let p = check_file_proposal("\n  **filename: `notes.md`**\nCafé 🎯 中文\n").unwrap();
    assert_eq!(
        (p.path.as_str(), p.content.as_str()),
        ("notes.md", "Café 🎯 中文\n")
    );
    // A multi-byte path is refused by the path rule, not by a panic.
    let err = check_file_proposal("filename: café.md\nbody\n").unwrap_err();
    assert!(err.contains("not a plain relative path"), "{err}");
}

/// Seeded property loop: random replies built from multi-byte and parser
/// punctuation pieces never panic, and "no filename line" is returned exactly
/// when the trimmed first line does not start with the ASCII key.
#[test]
fn property_multibyte_replies_never_panic() {
    const PIECES: &[&str] = &[
        "é",
        "ü",
        "中",
        "文",
        "🎯",
        "🙂",
        "\u{00a0}",
        "\u{3000}",
        "\u{200b}",
        "a",
        "Z",
        "1",
        " ",
        "\t",
        "\n",
        "\r\n",
        ":",
        "/",
        ".",
        "./",
        "..",
        "-",
        "_",
        "`",
        "```",
        "````",
        "```rust",
        "*",
        "**",
        "\"",
        "'",
        "filename:",
        "FILENAME:",
        "fileName",
        "filename",
        "file",
        "name:",
        "x.md",
        "a/b.txt",
    ];
    let mut s: u64 = 0x5eed_0546_dead_beef;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut panics = Vec::new();
    let mut count = 0usize;
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // keep a red run readable
    for _ in 0..200_000 {
        let n = (next() % 14) as usize;
        let mut text = String::new();
        for _ in 0..n {
            text.push_str(PIECES[(next() % PIECES.len() as u64) as usize]);
        }
        match no_panic(&text) {
            Err(p) => {
                count += 1;
                if panics.len() < 5 {
                    panics.push((text.clone(), p));
                }
            }
            Ok(r) => {
                let first = text.lines().find(|l| !l.trim().is_empty());
                let bare = first.map(|l| {
                    l.trim()
                        .trim_matches(|c| matches!(c, '`' | '"' | '\'' | '*'))
                        .trim()
                });
                let has_key = bare.is_some_and(|b| {
                    b.len() >= KEY.len()
                        && b.as_bytes()[..KEY.len()].eq_ignore_ascii_case(KEY.as_bytes())
                });
                let refused_for_key = matches!(&r, Err(e) if e.contains("no filename line"));
                if first.is_some() {
                    assert_eq!(refused_for_key, !has_key, "{text:?} -> {r:?}");
                }
            }
        }
    }
    std::panic::set_hook(hook);
    assert!(
        panics.is_empty(),
        "check_file_proposal panicked on {count} of 200000 inputs, first: {panics:?}"
    );
}
