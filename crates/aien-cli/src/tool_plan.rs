//! sovereign-core #666: plan a batch of tool calls into ordered segments.
//!
//! Mechanism after Hermes Agent (Nous Research, MIT, commit 5480918),
//! re-implemented; no code copied.
//!
//! The model often asks for several tools in one reply. Running them one after
//! another is always safe; running them together is faster but only safe when
//! they cannot step on each other. This module decides, without doing any I/O,
//! which calls may run together:
//!
//! - **Readers share, writers exclude.** A reader (view, list, search) and a
//!   writer (write, replace, create) reserve a path. Two calls conflict when
//!   their paths overlap (one is the other or lies inside it) and at least one
//!   of them writes. Readers on the same path never conflict.
//! - **Everything else is a barrier.** A shell command, a call that would ask
//!   the operator, a call with no path, or a tool this module does not know runs
//!   alone, so it never overlaps anything and approval prompts never interleave.
//! - **Order is kept.** Segments run in call order. A call joins the current
//!   segment only if it conflicts with nothing in it; otherwise it starts the
//!   next one, so two conflicting calls always run in the order the model asked.
//! - **Bounded.** At most `max_parallel` calls (8) run at once.
//!
//! Paths are compared lexically after making them absolute against the working
//! directory and removing `.` and `..`. A symlink that gives one file two
//! different spellings is not detected (known limit).

use serde_json::Value;
use std::path::{Component, Path, PathBuf};

/// Most calls that run at once (Hermes uses the same bound).
pub const MAX_PARALLEL: usize = 8;

/// What one call touches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    Read(PathBuf),
    Write(PathBuf),
    /// Runs alone: unknown scope, a shell, an operator prompt, any other tool.
    Barrier,
}

/// One step of the plan: indices into the batch, in call order. More than one
/// index means the calls run together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub calls: Vec<usize>,
}

/// Lexical normal form: absolute against `cwd`, no `.`, `..` folded.
pub fn normalize(path: &str, cwd: &Path) -> PathBuf {
    let p = Path::new(path);
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn path_arg(args: &Value) -> Option<&str> {
    ["path", "Path", "directory", "dir"]
        .iter()
        .find_map(|k| args.get(*k).and_then(Value::as_str))
        .filter(|p| !p.trim().is_empty())
}

/// What a call touches, from its name and arguments alone.
pub fn access_of(name: &str, args: &Value, cwd: &Path) -> Access {
    match name {
        "view_file" | "list_dir" => match path_arg(args) {
            Some(p) => Access::Read(normalize(p, cwd)),
            None => Access::Barrier,
        },
        // A search with no path searches the working directory.
        "grep_search" | "search_directory" | "find_by_name" | "find_file" => {
            Access::Read(normalize(path_arg(args).unwrap_or("."), cwd))
        }
        "write_to_file" | "replace_file_content" | "create_dir" => match path_arg(args) {
            Some(p) => Access::Write(normalize(p, cwd)),
            None => Access::Barrier,
        },
        _ => Access::Barrier,
    }
}

fn overlaps(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

fn conflicts(a: &Access, b: &Access) -> bool {
    match (a, b) {
        (Access::Barrier, _) | (_, Access::Barrier) => true,
        (Access::Read(_), Access::Read(_)) => false,
        (Access::Read(x) | Access::Write(x), Access::Read(y) | Access::Write(y)) => overlaps(x, y),
    }
}

/// Plans the batch. Pure.
pub fn plan(accesses: &[Access], max_parallel: usize) -> Vec<Segment> {
    let max_parallel = max_parallel.max(1);
    let mut segments: Vec<Segment> = Vec::new();
    for (i, a) in accesses.iter().enumerate() {
        let joins = match segments.last() {
            Some(seg) => {
                !matches!(a, Access::Barrier)
                    && seg.calls.len() < max_parallel
                    && seg.calls.iter().all(|&j| !conflicts(&accesses[j], a))
            }
            None => false,
        };
        if joins {
            if let Some(seg) = segments.last_mut() {
                seg.calls.push(i);
            }
        } else {
            segments.push(Segment { calls: vec![i] });
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cwd() -> PathBuf {
        PathBuf::from("/w")
    }
    fn r(p: &str) -> Access {
        Access::Read(PathBuf::from(p))
    }
    fn w(p: &str) -> Access {
        Access::Write(PathBuf::from(p))
    }
    fn segs(s: &[Segment]) -> Vec<Vec<usize>> {
        s.iter().map(|x| x.calls.clone()).collect()
    }

    #[test]
    fn readers_share_even_on_the_same_path() {
        let p = plan(&[r("/w/a"), r("/w/a"), r("/w"), r("/w/b/c")], MAX_PARALLEL);
        assert_eq!(segs(&p), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn two_writers_on_one_path_never_share_a_segment() {
        let p = plan(&[w("/w/a"), w("/w/a")], MAX_PARALLEL);
        assert_eq!(segs(&p), vec![vec![0], vec![1]]);
    }

    #[test]
    fn a_writer_excludes_readers_of_overlapping_paths_but_not_others() {
        // Reading the folder overlaps writing a file inside it.
        let p = plan(
            &[
                r("/w/src"),
                w("/w/src/main.rs"),
                r("/w/docs"),
                w("/w/out.txt"),
            ],
            MAX_PARALLEL,
        );
        assert_eq!(segs(&p), vec![vec![0], vec![1, 2, 3]]);
        // A prefix that is not a path component does not overlap.
        assert!(!overlaps(Path::new("/w/ab"), Path::new("/w/a")));
    }

    #[test]
    fn order_is_kept_across_conflicts() {
        // write a, read b, read a: the read of a must come after the write.
        let p = plan(&[w("/w/a"), r("/w/b"), r("/w/a")], MAX_PARALLEL);
        assert_eq!(segs(&p), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn barriers_run_alone() {
        let p = plan(
            &[r("/w/a"), Access::Barrier, r("/w/b"), r("/w/c")],
            MAX_PARALLEL,
        );
        assert_eq!(segs(&p), vec![vec![0], vec![1], vec![2, 3]]);
    }

    #[test]
    fn at_most_max_parallel_run_at_once() {
        let many: Vec<Access> = (0..20).map(|i| r(&format!("/w/{i}"))).collect();
        let p = plan(&many, MAX_PARALLEL);
        assert_eq!(
            p.iter().map(|s| s.calls.len()).collect::<Vec<_>>(),
            vec![8, 8, 4]
        );
        assert_eq!(p.concat_indices(), (0..20).collect::<Vec<_>>());
    }

    trait Concat {
        fn concat_indices(&self) -> Vec<usize>;
    }
    impl Concat for Vec<Segment> {
        fn concat_indices(&self) -> Vec<usize> {
            self.iter().flat_map(|s| s.calls.clone()).collect()
        }
    }

    #[test]
    fn access_comes_from_the_tool_and_its_path() {
        let c = cwd();
        assert_eq!(
            access_of("view_file", &json!({"path": "src/../a.rs"}), &c),
            r("/w/a.rs")
        );
        assert_eq!(
            access_of("grep_search", &json!({"query": "x"}), &c),
            r("/w")
        );
        assert_eq!(
            access_of("write_to_file", &json!({"path": "/w/./o"}), &c),
            w("/w/o")
        );
        assert_eq!(
            access_of("write_to_file", &json!({"content": "x"}), &c),
            Access::Barrier
        );
        assert_eq!(
            access_of("run_command", &json!({"command": "ls"}), &c),
            Access::Barrier
        );
        assert_eq!(
            access_of("brand_new_tool", &json!({"path": "a"}), &c),
            Access::Barrier
        );
    }
}
