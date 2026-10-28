//! Authenticate the bootstrap script before allowing it to check the rest of a source tree.
//! Git is a read-only hashing oracle here: object contents are not trusted merely because
//! cat-file found them under a requested object name. Never execute a working-tree checker.
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn git(repo: &Path) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(repo)
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_GRAFT_FILE", "/nonexistent/core-native-grafts");
    command
}

fn object(repo: &Path, kind: &str, expected: &str) -> Vec<u8> {
    let result = git(repo).args(["cat-file", kind, expected]).output()
        .expect("read bootstrap Git object");
    assert!(result.status.success(), "bootstrap object authentication: cannot read {kind} {expected}");
    let mut hash = git(repo).args(["hash-object", "-t", kind, "--stdin"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
        .expect("start read-only Git object hash");
    hash.stdin.take().unwrap().write_all(&result.stdout).expect("hash bootstrap object bytes");
    let actual = hash.wait_with_output().expect("finish bootstrap Git object hash");
    assert!(actual.status.success(), "bootstrap object authentication: cannot hash {kind} {expected}");
    assert_eq!(String::from_utf8(actual.stdout).expect("ASCII Git object ID").trim(), expected,
        "bootstrap object authentication failed for {kind} {expected}");
    result.stdout
}

fn entry(tree: &[u8], name: &[u8], modes: &[&[u8]]) -> String {
    let mut rest = tree;
    while !rest.is_empty() {
        let nul = rest.iter().position(|&b| b == 0).expect("bootstrap tree has no entry terminator");
        let header = &rest[..nul];
        let space = header.iter().position(|&b| b == b' ').expect("bootstrap tree has no entry mode");
        let id_end = nul.checked_add(21).expect("bootstrap tree entry overflow");
        assert!(rest.len() >= id_end, "bootstrap tree has a truncated object ID");
        if &header[space + 1..] == name {
            assert!(modes.contains(&&header[..space]), "bootstrap object authentication: unexpected entry mode");
            let mut id = String::with_capacity(40);
            use std::fmt::Write as _;
            for byte in &rest[nul + 1..id_end] {
                write!(&mut id, "{byte:02x}").expect("write Git object ID");
            }
            return id;
        }
        rest = &rest[id_end..];
    }
    panic!("bootstrap object authentication: required checker path is absent");
}

/// Return only checker bytes authenticated through commit -> root -> tools -> regular blob.
/// The caller executes a private snapshot and clears BASH_ENV before invoking Bash.
pub fn authenticated_source_checker(repo: &Path, pinned: &str) -> Vec<u8> {
    let commit = object(repo, "commit", pinned);
    let first = commit.split(|&b| b == b'\n').next().expect("bootstrap commit is empty");
    let root_id = std::str::from_utf8(first.strip_prefix(b"tree ").expect("bootstrap commit has no root tree"))
        .expect("ASCII bootstrap root tree ID");
    let root = object(repo, "tree", root_id);
    let tools_id = entry(&root, b"tools", &[b"40000"]);
    let tools = object(repo, "tree", &tools_id);
    let checker_id = entry(&tools, b"aienos_lock_source.sh", &[b"100644", b"100755"]);
    object(repo, "blob", &checker_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }
    impl Fixture {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("core-bootstrap-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            std::fs::create_dir_all(dir.join("tools")).unwrap();
            let fixture = Self(dir);
            fixture.run(&["init", "-q"]);
            fixture.run(&["config", "user.name", "Bootstrap fixture"]);
            fixture.run(&["config", "user.email", "fixture@example.invalid"]);
            fixture.run(&["config", "commit.gpgsign", "false"]);
            fixture.run(&["config", "core.hooksPath", "/dev/null"]);
            std::fs::write(fixture.0.join("tools/aienos_lock_source.sh"), b"printf pristine > \"$1\"\n").unwrap();
            fixture.run(&["add", "tools/aienos_lock_source.sh"]);
            fixture.run(&["commit", "-qm", "bootstrap fixture"]);
            fixture
        }
        fn run(&self, args: &[&str]) -> String {
            let output = git(&self.0).args(args).output().unwrap();
            assert!(output.status.success(), "fixture Git failed: {args:?}");
            String::from_utf8(output.stdout).unwrap().trim().to_owned()
        }
        fn poison_object(&self, kind: &str, expected: &str) {
            let mut child = git(&self.0).args(["hash-object", "--literally", "-w", "-t", kind, "--stdin"])
                .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
            child.stdin.take().unwrap().write_all(b"printf poisoned > \"$1\"\n").unwrap();
            let result = child.wait_with_output().unwrap();
            assert!(result.status.success());
            let rogue = String::from_utf8(result.stdout).unwrap();
            let rogue = rogue.trim();
            let objects = self.0.join(".git/objects");
            let destination = objects.join(&expected[..2]).join(&expected[2..]);
            std::fs::remove_file(&destination).unwrap();
            std::fs::copy(objects.join(&rogue[..2]).join(&rogue[2..]), destination).unwrap();
            std::fs::write(self.0.join("tools/aienos_lock_source.sh"), b"printf poisoned > \"$1\"\n").unwrap();
        }
        fn execute_snapshot(&self, bytes: &[u8]) {
            let snapshot = self.0.join("verified-checker.sh");
            std::fs::write(&snapshot, bytes).unwrap();
            assert!(Command::new("bash").arg(snapshot).arg(self.0.join("marker"))
                .env_remove("BASH_ENV").status().unwrap().success());
        }
    }

    #[test]
    fn pristine_chain_uses_verified_snapshot_not_dirty_working_checker() {
        let fixture = Fixture::new();
        let pin = fixture.run(&["rev-parse", "HEAD"]);
        std::fs::write(fixture.0.join("tools/aienos_lock_source.sh"), b"printf poisoned > \"$1\"\n").unwrap();
        fixture.execute_snapshot(&authenticated_source_checker(&fixture.0, &pin));
        assert_eq!(std::fs::read(fixture.0.join("marker")).unwrap(), b"pristine");
    }

    #[test]
    fn every_poisoned_bootstrap_object_refuses_before_checker_execution() {
        for (kind, expression) in [("commit", "HEAD"), ("tree", "HEAD^{tree}"),
            ("tree", "HEAD:tools"), ("blob", "HEAD:tools/aienos_lock_source.sh")] {
            let fixture = Fixture::new();
            let pin = fixture.run(&["rev-parse", "HEAD"]);
            let expected = fixture.run(&["rev-parse", expression]);
            fixture.poison_object(kind, &expected);
            let refusal = std::panic::catch_unwind(|| {
                let checker = authenticated_source_checker(&fixture.0, &pin);
                fixture.execute_snapshot(&checker);
            });
            assert!(refusal.is_err(), "poisoned {kind} {expression} was accepted");
            assert!(!fixture.0.join("marker").exists(), "poisoned checker executed before refusal");
        }
    }
}
