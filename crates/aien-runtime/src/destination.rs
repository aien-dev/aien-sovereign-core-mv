//! Which path does a compose goal ask to WRITE? (issue #288)
//!
//! A goal mentions paths in two roles: the DESTINATION (the file the task
//! creates or changes) and SOURCES (material it reads: "about README.md").
//! Only the destination decides the task kind, the budget, the prompt and the
//! merge. This module is a small rule-based parser with two tables:
//!
//! * `DESTINATION_VERBS`: a verb that introduces the destination ("create",
//!   "update", "append", "save", ...). The first path after it is the target.
//! * `SOURCE_MARKERS`: a word that marks the next path as source material
//!   ("about", "from", "using", "based on", "summarising", ...). Paths joined
//!   to a source by "and" or a comma are sources too.
//!
//! Rules, in order:
//! 0. A file operation word (rename, move, delete, remove) is
//!    `Err(Unsupported)`, as before, unless it is subject matter: it follows
//!    "how to" (or when/where/why to) and acts on nothing (no path later in
//!    the same sentence, no "it"/"the rest"/..., not the end of the sentence),
//!    as in "a how-to that shows how to rename a folder of photos". Even then
//!    the goal passes only when it names exactly one file to write.
//! 1. A path-like word is one with a `/`, a file extension, or a name that
//!    exists as a file (`Makefile`). Absolute, `~` and `..` paths are path-like
//!    and unsafe.
//! 2. Paths marked as sources are set aside. A path introduced as an existing
//!    file ("there is a file at X", "there are files at X and Y") is set aside
//!    too, but only when a destination verb directly names another file to
//!    write ("create a new file called Y"); otherwise it stays a candidate
//!    (alone: the file to edit).
//! 3. No destination left: `Ok(None)` (a new document, the model names it).
//! 4. One destination: it.
//! 5. Several destinations: the first is taken only when a destination verb
//!    comes before it and no later path is joined to it by "and"/"," or
//!    introduced by a second destination verb. Otherwise `Err(Ambiguous)`
//!    naming every candidate. Never a silent pick.
//!
//! The words of a goal are untrusted text; nothing here touches the disk
//! except through the `is_file` callback.

/// Verbs that introduce the destination of a task.
pub const DESTINATION_VERBS: &[&str] = &[
    "create", "write", "make", "generate", "produce", "draft", "compose", "update", "edit",
    "modify", "append", "add", "insert", "rewrite", "change", "fix", "extend", "revise", "save",
    "put", "store", "place", "prepend", "replace",
];

/// Words that mark the path after them as source material.
pub const SOURCE_MARKERS: &[&str] = &[
    "about",
    "from",
    "using",
    "use",
    "of",
    "reading",
    "read",
    "summarising",
    "summarizing",
    "summarise",
    "summarize",
    "describing",
    "covering",
    "per",
    "following",
    "reference",
    "referencing",
    "given",
];

/// Words that may sit between a destination verb and the path it governs
/// ("Edit the given README.md"): a verb reaching the path through these wins
/// over a source marker.
pub const FILLERS: &[&str] = &[
    "the",
    "a",
    "an",
    "this",
    "that",
    "my",
    "our",
    "your",
    "file",
    "document",
    "doc",
    "new",
    "existing",
    "given",
    "reference",
    "referenced",
    "called",
    "named",
    "same",
    "current",
];

/// Nouns that place a position INSIDE a file: in "the end of X" or "the Intro
/// section of X", the word "of" points at the file being changed, so it is not
/// a source marker. Other "of" phrases ("summary of X", "list of X", "copy of
/// X") still mark X as material to read.
pub const LOCATIONAL: &[&str] = &[
    "section",
    "sections",
    "part",
    "end",
    "top",
    "bottom",
    "start",
    "beginning",
    "body",
    "contents",
    "content",
    "heading",
    "line",
    "lines",
    "row",
    "block",
    "footer",
    "header",
    "middle",
    "rest",
    "bit",
];

/// File operations this parser never turns into a write.
pub const FILE_OPS: &[&str] = &["rename", "move", "delete", "remove"];

/// Words a file operation acts on without naming a path ("delete it",
/// "remove the rest").
pub const OP_OBJECTS: &[&str] = &[
    "it",
    "its",
    "them",
    "their",
    "everything",
    "all",
    "both",
    "these",
    "those",
    "anything",
    "each",
    "others",
    "rest",
    "else",
    "one",
];

/// Question words that, before "to", make a following file operation the
/// subject of the document ("how to rename a folder"), not a request.
pub const TOPIC_LEADS: &[&str] = &["how", "when", "where", "why"];

/// Nouns of an existence statement ("there are files at ...").
pub const EXISTING_NOUNS: &[&str] = &[
    "file",
    "files",
    "document",
    "documents",
    "doc",
    "docs",
    "note",
    "notes",
];

/// Top-level domains that make a bare `name.tld` token a host name, not a file.
pub const HOST_TLDS: &[&str] = &[
    "org", "com", "net", "io", "dev", "edu", "gov", "ai", "co", "app", "me", "info", "xyz",
];

/// Words after which a `name.tld` token is a web address.
pub const HOST_LEAD: &[&str] = &["visit", "see", "at", "http", "https", "www", "url", "site"];

/// A path-like word of the goal.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Cand {
    path: String,
    idx: usize,
    source: bool,
    /// Joined to the previous path-like word by "and", "or" or a comma.
    conj: bool,
    /// A destination verb stands between the previous path-like word and this one.
    verb_before: bool,
    /// Introduced as a file that exists ("there is a file at X"), or joined
    /// to one: material to read when the goal names another file to write.
    existing: bool,
    /// A destination verb governs this path directly ("create the file X").
    governed: bool,
}

/// The raw word ends a sentence ("loop." but not "e.g.").
fn ends_sentence(raw: &str) -> bool {
    let t = raw.trim_end_matches(['"', '\'', '`', ')', ']']);
    t.ends_with(['.', '!', '?', ';']) && !matches!(clean(raw), "e.g" | "i.e" | "etc" | "vs")
}

/// The file operation at `i` acts on something: a path later in the same
/// sentence, a word like "it" or "the rest", or nothing before the sentence
/// ends ("delete this.").
fn op_acts(raw: &[&str], words: &[String], i: usize, is_file: &dyn Fn(&str) -> bool) -> bool {
    if ends_sentence(raw[i]) || i + 1 == raw.len() {
        return true;
    }
    let mut object_seen = false;
    for j in i + 1..raw.len() {
        if path_like(clean(raw[j]), is_file) {
            return true;
        }
        let lw = words[j].as_str();
        if !object_seen {
            if OP_OBJECTS.contains(&lw) {
                return true;
            }
            if !FILLERS.contains(&lw) {
                object_seen = true;
            } else if ends_sentence(raw[j]) || j + 1 == raw.len() {
                return true;
            }
        }
        if ends_sentence(raw[j]) {
            break;
        }
    }
    false
}

/// The first file operation the goal ASKS for. An operation word is subject
/// matter only after "how to" (or when/where/why to) and only when it acts
/// on nothing ("a how-to that shows how to rename a folder of photos");
/// every other use is asked for ("so delete the stale doc", "Delete the
/// draft.", "how to delete it").
fn asked_file_op(raw: &[&str], words: &[String], is_file: &dyn Fn(&str) -> bool) -> Option<String> {
    for (i, w) in words.iter().enumerate() {
        if !FILE_OPS.contains(&w.as_str()) {
            continue;
        }
        let topic = i >= 2 && words[i - 1] == "to" && TOPIC_LEADS.contains(&words[i - 2].as_str());
        if !topic || op_acts(raw, words, i, is_file) {
            return Some(w.clone());
        }
    }
    None
}

/// The path at `i` is introduced as an existing file: "there is a file at X",
/// "there's a file called X", "there are files at X", "a file exists at X".
/// Never "there is a NEW file ...".
fn introduced_as_existing(words: &[String], i: usize) -> bool {
    let mut j = i;
    while j > 0 {
        let p = words[j - 1].as_str();
        if p == "new" {
            return false;
        }
        if FILLERS.contains(&p) || EXISTING_NOUNS.contains(&p) || matches!(p, "at" | "in") {
            j -= 1;
        } else {
            break;
        }
    }
    if j == i || j == 0 {
        return false;
    }
    let lead = words[j - 1].as_str();
    let before = (j > 1).then(|| words[j - 2].as_str());
    matches!(lead, "there's" | "theres" | "exists" | "exist")
        || (matches!(lead, "is" | "are") && before == Some("there"))
}

/// The destination could not be chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DestinationError {
    /// Several candidate destinations and no rule picks one.
    Ambiguous(Vec<String>),
    /// The goal only mentions paths as source material; nothing is named to write.
    NoDestination(Vec<String>),
    /// The goal asks for an operation this parser does not support.
    Unsupported(String),
}

impl std::fmt::Display for DestinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDestination(c) => write!(
                f,
                "no destination named: the goal only mentions {} as material to read; name the file to write",
                c.join(", ")
            ),
            Self::Unsupported(w) => write!(f, "{w} is not supported: name the file to create or edit"),
            Self::Ambiguous(c) => write!(
                f,
                "ambiguous destination: the goal names several files to write ({}); name one",
                c.join(", ")
            ),
        }
    }
}

fn clean(w: &str) -> &str {
    const Q: fn(char) -> bool = |c| matches!(c, '"' | '\'' | '`' | '(' | ')' | '[' | ']');
    w.trim_matches(Q)
        .trim_end_matches(['.', ',', ';', ':', '!', '?'])
        .trim_matches(Q)
}

fn has_extension(w: &str) -> bool {
    let last = w.rsplit('/').next().unwrap_or(w);
    match last.rsplit_once('.') {
        Some((stem, ext)) => {
            !stem.is_empty()
                && !ext.is_empty()
                && ext.len() <= 10
                && ext.starts_with(|c: char| c.is_ascii_alphabetic())
                && ext.chars().all(|c| c.is_ascii_alphanumeric())
                && !matches!(w, "e.g" | "i.e" | "etc")
                && !(ext.len() == 1 && stem.contains('.'))
        }
        None => false,
    }
}

/// True for a word that names a path, safe or not.
fn path_like(w: &str, is_file: &dyn Fn(&str) -> bool) -> bool {
    if w.is_empty() || w.contains('@') || w.contains("://") {
        return false;
    }
    let unsafe_form =
        w.starts_with('/') || w.starts_with("~/") || w == "~" || w.split('/').any(|c| c == "..");
    if unsafe_form {
        return w.contains('/') || w.starts_with('~');
    }
    let ok_chars = w
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'));
    ok_chars && (w.contains('/') && has_extension(w) || has_extension(w) || is_file(w))
}

/// What a goal names: the one file to write (resolved into its folder when the
/// goal says "in the Y folder") and the other paths it mentions, in goal order.
/// The other paths are references: material to read, never written.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Named {
    pub dest: Option<String>,
    pub inputs: Vec<String>,
}

#[derive(Default)]
struct Parsed {
    /// Destination path as written in the goal, and its word index.
    dest: Option<(String, usize)>,
    others: Vec<String>,
}

const FOLDER_NOUNS: &[&str] = &["folder", "directory", "dir"];
const FOLDER_ARTICLES: &[&str] = &["the", "a", "an", "our", "my", "this"];
/// Plural nouns and verbs that make a folder phrase a place to read from.
const READ_INTENT: &[&str] = &[
    "files",
    "documents",
    "docs",
    "notes",
    "texts",
    "read",
    "reading",
    "from",
    "about",
    "using",
    "summarise",
    "summarize",
    "of",
];

/// "in the Y folder" phrases of the goal: (folder name, index of "in").
fn folder_phrases(raw: &[&str], words: &[String]) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for (i, w) in words.iter().enumerate() {
        if !matches!(w.as_str(), "in" | "inside" | "into") {
            continue;
        }
        let mut j = i + 1;
        while j < words.len() && FOLDER_ARTICLES.contains(&words[j].as_str()) {
            j += 1;
        }
        if j + 1 >= words.len() || !FOLDER_NOUNS.contains(&words[j + 1].as_str()) {
            continue;
        }
        let name = clean(raw[j]);
        let plain = !name.is_empty()
            && !name.starts_with('.')
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
        if plain {
            out.push((name.to_string(), i));
        }
    }
    out
}

/// The destination written as "X ... in the Y folder" is `Y/X`. Only a bare
/// name directly followed by the phrase is moved.
fn resolve_in_folder(raw: &[&str], words: &[String], dest: String, idx: usize) -> String {
    if dest.contains('/') {
        return dest;
    }
    match folder_phrases(raw, words)
        .into_iter()
        .find(|(_, at)| *at == idx + 1)
    {
        Some((folder, _)) => format!("{folder}/{dest}"),
        None => dest,
    }
}

/// Every folder the goal names with "the Y folder" (any role), in goal order.
pub fn goal_folders(goal: &str) -> Vec<String> {
    let raw: Vec<&str> = goal.split_whitespace().collect();
    let words: Vec<String> = raw.iter().map(|w| clean(w).to_ascii_lowercase()).collect();
    let mut out: Vec<String> = Vec::new();
    for (n, _) in folder_phrases(&raw, &words) {
        if !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

/// Folders the goal names as a place to READ from ("the three text files in the
/// inbox folder"): the phrase is not the folder of a file the goal names
/// (the word before "in" is not a path) and a reading word stands within the
/// five words before "in".
pub fn input_folders(goal: &str) -> Vec<String> {
    let raw: Vec<&str> = goal.split_whitespace().collect();
    let words: Vec<String> = raw.iter().map(|w| clean(w).to_ascii_lowercase()).collect();
    let mut out: Vec<String> = Vec::new();
    for (n, at) in folder_phrases(&raw, &words) {
        let after_path = at > 0 && {
            let p = clean(raw[at - 1]);
            p.contains('/') || has_extension(p)
        };
        let reads = words[at.saturating_sub(5)..at]
            .iter()
            .any(|w| READ_INTENT.contains(&w.as_str()));
        if !after_path && reads && !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

/// The destination the goal names. `is_file(w)` says whether the plain
/// relative word `w` is an existing file (for extension-less names).
pub fn named_destination(
    goal: &str,
    is_file: &dyn Fn(&str) -> bool,
) -> Result<Option<String>, DestinationError> {
    named_paths(goal, is_file, &|_| false).map(|n| n.dest)
}

/// The destination and the input paths a goal names. `is_input(w)` says whether
/// the word `w` is an existing workspace file that is material to read: such a
/// name is never taken as a write target when the goal also names a new file.
pub fn named_paths(
    goal: &str,
    is_file: &dyn Fn(&str) -> bool,
    is_input: &dyn Fn(&str) -> bool,
) -> Result<Named, DestinationError> {
    let raw: Vec<&str> = goal.split_whitespace().collect();
    let words: Vec<String> = raw.iter().map(|w| clean(w).to_ascii_lowercase()).collect();
    if let Some(op) = asked_file_op(&raw, &words, is_file) {
        return Err(DestinationError::Unsupported(op));
    }
    // A file-operation word that acts on nothing is subject matter only when
    // the goal names exactly one file to write; otherwise it is refused as before.
    let op_word = words.iter().find(|w| FILE_OPS.contains(&w.as_str()));
    let finish = |p: Parsed| {
        let dest = p.dest.map(|(d, i)| resolve_in_folder(&raw, &words, d, i));
        Named {
            dest,
            inputs: p.others,
        }
    };
    match (op_word, parse(&raw, &words, is_file, is_input)) {
        (_, Ok(p)) if p.dest.is_some() => Ok(finish(p)),
        (Some(op), _) => Err(DestinationError::Unsupported(op.clone())),
        (None, other) => other.map(finish),
    }
}

fn parse(
    raw: &[&str],
    words: &[String],
    is_file: &dyn Fn(&str) -> bool,
    is_input: &dyn Fn(&str) -> bool,
) -> Result<Parsed, DestinationError> {
    let mut cands: Vec<Cand> = Vec::new();
    let mut verb_since_last = false;
    let mut first_verb: Option<usize> = None;
    for (i, w) in raw.iter().enumerate() {
        let c = clean(w);
        let lw = &words[i];
        if DESTINATION_VERBS.contains(&lw.as_str()) {
            verb_since_last = true;
            first_verb.get_or_insert(i);
        }
        if !path_like(c, is_file) {
            continue;
        }
        let governed = {
            let mut j = i;
            while j > 0 && FILLERS.contains(&words[j - 1].as_str()) {
                j -= 1;
            }
            j > 0 && DESTINATION_VERBS.contains(&words[j - 1].as_str())
        };
        let host_like = !c.contains('/')
            && !is_file(c)
            && (c
                .rsplit('.')
                .next()
                .is_some_and(|t| HOST_TLDS.contains(&t.to_ascii_lowercase().as_str()))
                || (i > 0 && HOST_LEAD.contains(&words[i - 1].as_str())));
        if host_like && !governed {
            continue;
        }
        let prev = (i > 0).then(|| words[i - 1].as_str());
        let prev2 = (i > 1).then(|| words[i - 2].as_str());
        let location_of = prev == Some("of") && prev2.is_some_and(|p| LOCATIONAL.contains(&p));
        let marked = !governed
            && (prev.is_some_and(|p| SOURCE_MARKERS.contains(&p) && !location_of)
                || (prev == Some("on") && prev2 == Some("based"))
                || (prev == Some("on") && prev2 == Some("relying")));
        // Joined to the previous path-like word: "A and B", "A, B", "A or B", "A B".
        let joined = cands.last().is_some_and(|l| {
            (prev == Some("and") || prev == Some("or") || prev == Some("&")) && l.idx + 2 == i
                || l.idx + 1 == i
        });
        let source = marked || (joined && cands.last().is_some_and(|l| l.source));
        let existing = !governed
            && (introduced_as_existing(words, i)
                || (joined && cands.last().is_some_and(|l| l.existing)));
        cands.push(Cand {
            path: c.to_string(),
            idx: i,
            source,
            conj: joined,
            verb_before: verb_since_last && !cands.is_empty(),
            existing,
            governed,
        });
        verb_since_last = false;
    }
    let mut dests: Vec<&Cand> = cands.iter().filter(|c| !c.source).collect();
    // A file introduced as existing is read, not written, when a destination
    // verb directly names another file to write ("create a new file called
    // Y"); otherwise it stays a candidate (alone: the file to edit).
    // Not when another destination verb stands between the two ("there is a
    // file at X. Update it and create Y" asks to write both).
    let demote = dests.iter().find(|c| !c.existing).is_some_and(|g| {
        g.governed
            && dests.iter().filter(|c| c.existing).all(|e| {
                let (a, b) = (e.idx.min(g.idx), e.idx.max(g.idx));
                words[a + 1..b]
                    .iter()
                    .filter(|w| DESTINATION_VERBS.contains(&w.as_str()))
                    .count()
                    <= 1
            })
    });
    if demote {
        dests.retain(|c| !c.existing);
    }
    // A name that already exists in the workspace and is not governed by a
    // destination verb is material to read, never a write target, when another
    // candidate remains that does not exist (the new file the goal asks for).
    let input_like = |c: &Cand| !c.governed && is_input(&c.path);
    if dests.iter().any(|c| !input_like(c) && !is_input(&c.path)) {
        dests.retain(|c| !input_like(c));
    }
    let Some(first) = dests.first() else {
        if cands.is_empty() {
            return Ok(Parsed::default());
        }
        let mut names: Vec<String> = Vec::new();
        for c in &cands {
            if !names.contains(&c.path) {
                names.push(c.path.clone());
            }
        }
        return Err(DestinationError::NoDestination(names));
    };
    if dests.len() > 1 {
        let verb_leads = first_verb.is_some_and(|v| v < first.idx);
        let clash = dests[1..].iter().any(|d| d.conj || d.verb_before);
        if !verb_leads || clash {
            let mut names: Vec<String> = Vec::new();
            for d in &dests {
                if !names.contains(&d.path) {
                    names.push(d.path.clone());
                }
            }
            if names.len() > 1 {
                return Err(DestinationError::Ambiguous(names));
            }
        }
    }
    let mut others: Vec<String> = Vec::new();
    for c in &cands {
        if c.path != first.path && !others.contains(&c.path) {
            others.push(c.path.clone());
        }
    }
    Ok(Parsed {
        dest: Some((first.path.clone(), first.idx)),
        others,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(g: &str) -> Result<Option<String>, DestinationError> {
        named_destination(g, &|w| w == "Makefile")
    }

    #[test]
    fn source_is_not_the_destination() {
        assert_eq!(
            d("Create docs/SUMMARY.md about README.md"),
            Ok(Some("docs/SUMMARY.md".into()))
        );
        assert_eq!(
            d("Update README.md using docs/NOTES.md"),
            Ok(Some("README.md".into()))
        );
        assert_eq!(
            d("Add a summary of README.md and LICENSE to docs/S.md"),
            Ok(Some("docs/S.md".into()))
        );
    }

    #[test]
    fn single_and_extensionless() {
        assert_eq!(d("write faq.md"), Ok(Some("faq.md".into())));
        assert_eq!(d("Edit Makefile please"), Ok(Some("Makefile".into())));
        assert_eq!(d("Edit docs now"), Ok(None));
        assert_eq!(d("say hi, e.g. twice"), Ok(None));
        assert_eq!(d("under the ## 0.1.0 heading"), Ok(None));
    }

    #[test]
    fn several_destinations_are_ambiguous() {
        assert!(matches!(
            d("Create a.md and b.md"),
            Err(DestinationError::Ambiguous(c)) if c == ["a.md", "b.md"]
        ));
        assert!(d("Create a.md. Then update b.md").is_err());
        assert!(d("put a.md b.md together").is_err());
        assert_eq!(
            d("Edit `docs/plan.txt`, then README.md"),
            Ok(Some("docs/plan.txt".into()))
        );
    }

    #[test]
    fn source_only_goals_name_no_destination() {
        for g in [
            "Summarise README.md",
            "Summarise README.md and docs/A.md",
            "Create a note summarising README.md",
        ] {
            assert!(
                matches!(d(g), Err(DestinationError::NoDestination(_))),
                "{g}"
            );
        }
        let Err(e) = d("Summarise README.md") else {
            panic!()
        };
        assert!(e.to_string().contains("README.md") && e.to_string().contains("no destination"));
    }

    #[test]
    fn verb_governing_the_path_beats_a_marker_adjective() {
        assert_eq!(d("Edit the given README.md"), Ok(Some("README.md".into())));
        assert_eq!(
            d("Update the reference README.md"),
            Ok(Some("README.md".into()))
        );
        assert_eq!(
            d("Create docs/S.md from the given README.md"),
            Ok(Some("docs/S.md".into()))
        );
    }

    #[test]
    fn host_names_are_not_files_unless_governed() {
        assert_eq!(
            d("Visit example.org and write out.md"),
            Ok(Some("out.md".into()))
        );
        assert_eq!(
            d("See docs.example.com then create out.md"),
            Ok(Some("out.md".into()))
        );
        assert_eq!(d("Create example.org"), Ok(Some("example.org".into())));
    }

    #[test]
    fn of_phrases_resolve_by_what_they_point_at() {
        let one = |g: &str| d(g).unwrap().unwrap();
        // A position inside the file being changed: the file is the destination.
        for (g, want) in [
            ("Add a line to the Intro section of README.md", "README.md"),
            ("Append a note to the end of README.md", "README.md"),
            ("Insert a badge at the top of README.md", "README.md"),
            (
                "Add \"x\" to the \"## Before a trip\" section of notes/GARAGE.md.",
                "notes/GARAGE.md",
            ),
        ] {
            assert_eq!(one(g), want, "{g}");
        }
        // Material to read: the file is a source, the destination is named elsewhere.
        for (g, want) in [
            ("Write a list of README.md into docs/L.md", "docs/L.md"),
            ("Create docs/C.md as a copy of README.md", "docs/C.md"),
            (
                "Write a summary of README.md into docs/SUMMARY.md",
                "docs/SUMMARY.md",
            ),
            // Location word in the source role: the later file is not demoted,
            // the first file after the verb still wins (documented limit).
            ("Create docs/S.md from the top of README.md", "docs/S.md"),
            (
                "Create docs/S.md summarising the section of README.md",
                "docs/S.md",
            ),
            ("Create docs/S.md about the end of README.md", "docs/S.md"),
        ] {
            assert_eq!(one(g), want, "{g}");
        }
        for g in [
            "Make a list of README.md",
            "Summarise README.md",
            "Make a copy of README.md",
        ] {
            assert!(
                matches!(d(g), Err(DestinationError::NoDestination(_))),
                "{g}"
            );
        }
    }

    #[test]
    fn rename_is_unsupported() {
        assert!(
            matches!(d("Rename a.md to b.md"), Err(DestinationError::Unsupported(w)) if w == "rename")
        );
    }

    /// A file operation the goal ASKS for is refused; the same word as subject
    /// matter of a named document is not (v5 G6, goal D2).
    #[test]
    fn a_file_operation_is_refused_only_when_asked_for() {
        let unsupported = |g: &str, want: &str| {
            assert!(
                matches!(d(g), Err(DestinationError::Unsupported(ref w)) if w == want),
                "{g}: {:?}",
                d(g)
            );
        };
        // Asked for: the verb acts on a path named in the same sentence.
        unsupported("rename notes/a.txt to notes/b.txt", "rename");
        unsupported("delete docs/x.md", "delete");
        unsupported("Please delete the file docs/x.md.", "delete");
        unsupported("move a.txt into b/", "move");
        unsupported("Rename my old photos.zip", "rename");
        unsupported("Create docs/a.md and delete docs/b.md", "delete");
        unsupported("Remove the line \"x\" from README.md", "remove");
        unsupported("Remove e.g. old.txt", "remove");
        // Asked for, by pronoun or with nothing named after the verb.
        unsupported("Write docs/a.md. Then delete it.", "delete");
        unsupported("Create docs/a.md and remove everything else", "remove");
        unsupported("Write docs/a.md, then move them.", "move");
        // Asked for, and no single file to write: refused as before.
        unsupported("Delete all my notes", "delete");
        unsupported("Move the photos into the archive folder", "move");
        unsupported("Write a guide on how to delete files", "delete");
        // Asked for although the object is a plain noun, or the path comes
        // first: never turned into a write of that path (review of #357).
        unsupported("docs/old.md is obsolete, so delete the stale doc", "delete");
        unsupported(
            "docs/a.md needs a better name, rename the file to something shorter",
            "rename",
        );
        unsupported("Write docs/a.md. Delete the draft.", "delete");
        unsupported("Create docs/a.md and delete the old files", "delete");
        unsupported("Create docs/a.md and delete its contents", "delete");
        unsupported("Create docs/a.md and remove the rest", "remove");
        unsupported("Write docs/a.md showing how to delete it", "delete");
        unsupported(
            "Write docs/guide.md explaining how to delete docs/x.md",
            "delete",
        );
        // Several operation words: one asked for is enough.
        unsupported(
            "Write docs/a.md explaining how to rename photos. Then rename the folder.",
            "rename",
        );
        // Subject matter of a named document.
        assert_eq!(
            d(
                "I need a technical how-to saved as docs/rename-photos.md that shows how to \
               rename a folder of holiday photos with a bash loop. Write at least 18 lines."
            ),
            Ok(Some("docs/rename-photos.md".into()))
        );
        assert_eq!(
            d("Write docs/backup.md explaining how to delete old backups safely."),
            Ok(Some("docs/backup.md".into()))
        );
    }

    /// "There is a file at X" points at material to read when the goal names
    /// another file to write (v5 G6, goal D4). On its own, X stays the file.
    #[test]
    fn an_existing_file_mentioned_first_is_read_not_written() {
        assert_eq!(
            d(
                "There is a file at notes/choir-rehearsal.txt with my rough notes from choir \
               practice. Please create a new file called docs/choir-summary.md that summarises \
               it, and leave the original alone."
            ),
            Ok(Some("docs/choir-summary.md".into()))
        );
        assert_eq!(
            d("There's a file called notes/a.txt. Write docs/b.md from it."),
            Ok(Some("docs/b.md".into()))
        );
        assert_eq!(
            d("There are files at notes/a.txt and notes/b.txt. Create docs/c.md summarising them."),
            Ok(Some("docs/c.md".into()))
        );
        // The only file named: it is the destination, as before.
        assert_eq!(
            d("There is a file at notes/a.txt. Fix the typo in it."),
            Ok(Some("notes/a.txt".into()))
        );
        // Two files to write stay ambiguous.
        assert!(matches!(
            d("There is a file at notes/a.txt. Create docs/b.md and docs/c.md."),
            Err(DestinationError::Ambiguous(c)) if c == ["docs/b.md", "docs/c.md"]
        ));
        assert!(matches!(
            d("Create docs/b.md and docs/c.md"),
            Err(DestinationError::Ambiguous(_))
        ));
        // The other file is not directly named as the one to write: no
        // demotion, still ambiguous (review of #357).
        assert!(matches!(
            d("There is a file at docs/x.md. Add a link to docs/y.md"),
            Err(DestinationError::Ambiguous(c)) if c == ["docs/x.md", "docs/y.md"]
        ));
        // A NEW file is never an existing one; "here is" is not an existence lead.
        for g in [
            "There is a new file docs/out.md. Create docs/z.md",
            "Here is a file docs/out.md. Create docs/z.md",
        ] {
            assert!(
                matches!(d(g), Err(DestinationError::Ambiguous(ref c)) if c == &["docs/out.md", "docs/z.md"]),
                "{g}: {:?}",
                d(g)
            );
        }
        // Another destination verb between them: both are asked to be
        // written, still ambiguous (review of #357, round 2).
        assert!(matches!(
            d("There is a file at docs/x.md. Update it and create docs/y.md"),
            Err(DestinationError::Ambiguous(c)) if c == ["docs/x.md", "docs/y.md"]
        ));
        // "Create a file at X" names the destination, not an existing file.
        assert_eq!(
            d("Create a file at docs/x.md with one line"),
            Ok(Some("docs/x.md".into()))
        );
    }

    #[test]
    fn unsafe_paths_are_named_not_skipped() {
        assert_eq!(
            d("Create the file ../outside.txt"),
            Ok(Some("../outside.txt".into()))
        );
        assert_eq!(d("Edit /etc/passwd"), Ok(Some("/etc/passwd".into())));
        assert_eq!(d("Edit ~/secret.txt"), Ok(Some("~/secret.txt".into())));
    }
}
