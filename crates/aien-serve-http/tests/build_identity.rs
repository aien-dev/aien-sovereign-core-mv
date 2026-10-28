//! Host-only regression for the actual build script's Cargo invalidation contract.
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn run(command: &mut Command) -> String {
    let output = command.output().expect("run host fixture command");
    assert!(output.status.success(), "{command:?}: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}
fn cargo_command(root: &Path, manifest: &Path) -> Command {
    let mut command = Command::new(env!("CARGO"));
    command.current_dir(root)
        .args(["run", "--verbose", "--locked", "--offline", "--manifest-path"])
        .arg(manifest)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env_remove("AIEN_BUILD_COMMIT");
    command
}
fn cargo_run(root: &Path, manifest: &Path) -> String {
    run(&mut cargo_command(root, manifest))
}
fn cargo_fresh(root: &Path, manifest: &Path) -> String {
    let output = cargo_command(root, manifest).output().unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success() && stderr.contains("Fresh core-build-identity-fixture"), "{stderr}");
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}
fn source(version: &str) -> String {
    format!("fn main() {{ println!(\"{{}}:{{}}:{version}\", env!(\"AIEN_SERVE_HTTP_GIT_COMMIT\"), env!(\"AIEN_SERVE_HTTP_GIT_DIRTY\")); }}\n")
}

#[test]
fn tracked_source_and_noncrate_changes_invalidate_the_clean_build_stamp() {
    let path = std::env::temp_dir().join(format!("core-build-identity-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&path).unwrap();
    let fixture = Fixture(path);
    let root = &fixture.0;
    let package = root.join("crates/fixture");
    std::fs::create_dir_all(package.join("src")).unwrap();
    let manifest = package.join("Cargo.toml");
    std::fs::write(&manifest, "[package]\nname = \"core-build-identity-fixture\"\nversion = \"0.0.0\"\nedition = \"2021\"\n").unwrap();
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"), package.join("build.rs")).unwrap();
    let main = package.join("src/main.rs");
    let baseline = source("baseline");
    std::fs::write(&main, &baseline).unwrap();
    let outside = root.join("tracked-input.txt");
    std::fs::write(&outside, "baseline\n").unwrap();
    run(Command::new(env!("CARGO")).current_dir(root)
        .args(["generate-lockfile", "--offline", "--manifest-path"]).arg(&manifest)
        .env("CARGO_TARGET_DIR", root.join("target")));
    run(Command::new("git").current_dir(root).args(["init", "-q"]));
    for (key, value) in [("user.name", "Build identity fixture"), ("user.email", "fixture@example.invalid"), ("core.hooksPath", "/dev/null"), ("commit.gpgsign", "false")] {
        run(Command::new("git").current_dir(root).args(["config", key, value]));
    }
    run(Command::new("git").current_dir(root).args(["add", "crates", "tracked-input.txt"]));
    run(Command::new("git").current_dir(root).args(["commit", "-q", "-m", "fixture"]));
    let commit = run(Command::new("git").current_dir(root).args(["rev-parse", "HEAD"]));
    assert_eq!(cargo_run(root, &manifest), format!("{commit}:0:baseline"));
    assert_eq!(cargo_fresh(root, &manifest), format!("{commit}:0:baseline"));

    // No index or HEAD change: compiling modified tracked source must not retain clean=0.
    std::fs::write(&main, source("modified")).unwrap();
    assert_eq!(cargo_run(root, &manifest), format!("{commit}:1:modified"));
    std::fs::write(&main, &baseline).unwrap();
    assert_eq!(cargo_run(root, &manifest), format!("{commit}:0:baseline"));

    // A tracked file outside this crate must also invalidate the whole-checkout stamp.
    std::fs::write(&outside, "modified\n").unwrap();
    assert_eq!(cargo_run(root, &manifest), format!("{commit}:1:baseline"));

    // Negative control: remove the watcher in this private fixture only. The old
    // behavior compiles new source while retaining the previous clean identity.
    std::fs::write(&outside, "baseline\n").unwrap();
    let build_script = package.join("build.rs");
    let original = std::fs::read_to_string(&build_script).unwrap();
    let mutant = original.replace("watch_tracked_files(&root);", "");
    assert_ne!(mutant, original, "negative control must remove the watcher");
    std::fs::write(&build_script, mutant).unwrap();
    run(Command::new("git").current_dir(root).args(["add", "crates/fixture/build.rs"]));
    run(Command::new("git").current_dir(root).args(["commit", "-q", "-m", "watcher-removed negative control"]));
    let mutant_commit = run(Command::new("git").current_dir(root).args(["rev-parse", "HEAD"]));
    assert_eq!(cargo_run(root, &manifest), format!("{mutant_commit}:0:baseline"));
    assert_eq!(cargo_fresh(root, &manifest), format!("{mutant_commit}:0:baseline"));
    std::fs::write(&main, source("modified")).unwrap();
    assert_eq!(cargo_run(root, &manifest), format!("{mutant_commit}:0:modified"));
    std::fs::write(&main, &baseline).unwrap();
    assert_eq!(cargo_run(root, &manifest), format!("{mutant_commit}:0:baseline"));
    std::fs::write(&outside, "modified\n").unwrap();
    assert_eq!(cargo_fresh(root, &manifest), format!("{mutant_commit}:0:baseline"));
}
