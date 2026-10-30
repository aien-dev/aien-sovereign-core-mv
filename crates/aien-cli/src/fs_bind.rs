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

//! Bind a CLI filesystem operation to the workspace object that was authorized.
//!
//! `safety::validate_workspace_path` is a pathname allow-list. It does not
//! confine the later open: a directory entry can be replaced between the
//! check and a fresh lookup. The functions here open the authorized relative
//! path from a held descriptor of the workspace root with one
//! `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS)`
//! call, then read and write that descriptor. A kernel without `openat2`
//! fails closed. Pathname checks are not claimed as confinement.

use std::ffi::{CString, OsStr, OsString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use crate::safety::validate_workspace_path;

const SYS_OPENAT2: libc::c_long = 437;

#[repr(C)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

#[derive(Debug)]
pub enum BindError {
    /// The path is not the authorized workspace object, or it cannot be bound.
    Denied(String),
    NotFound(String),
    Exists(String),
    Failed(String),
}

impl BindError {
    pub fn message(&self) -> &str {
        match self {
            Self::Denied(m) | Self::NotFound(m) | Self::Exists(m) | Self::Failed(m) => m,
        }
    }
}

struct Authorized {
    workspace: PathBuf,
    /// Plain relative path under `workspace`. Empty means the workspace root.
    rel: PathBuf,
    canonical: PathBuf,
    display: String,
}

/// Workspace roots the CLI file tools may authorize. Same set the tool
/// allow-list already used: platform defaults plus `/tmp`.
pub fn permitted_workspaces() -> Vec<PathBuf> {
    let mut allowed = crate::platform::PlatformContext::detect().default_workspaces;
    allowed.push(PathBuf::from("/tmp"));
    allowed
}

fn authorize(user_path: &str) -> Result<Authorized, BindError> {
    let workspaces = permitted_workspaces();
    let canonical = validate_workspace_path(user_path, &workspaces).map_err(BindError::Denied)?;
    let mut best: Option<(PathBuf, PathBuf)> = None;
    for ws in &workspaces {
        let Ok(ws_canon) = ws.canonicalize() else {
            continue;
        };
        if !canonical.starts_with(&ws_canon) {
            continue;
        }
        let rel = canonical
            .strip_prefix(&ws_canon)
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let take = match &best {
            None => true,
            Some((cur, _)) => ws_canon.components().count() > cur.components().count(),
        };
        if take {
            best = Some((ws_canon, rel));
        }
    }
    let Some((workspace, rel)) = best else {
        return Err(BindError::Denied(format!(
            "Path '{}' resolves to '{}' which is outside permitted workspace roots",
            user_path,
            canonical.display()
        )));
    };
    if rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(BindError::Denied(format!(
            "refusing '{}': the authorized path is not a plain path under the workspace",
            user_path
        )));
    }
    Ok(Authorized {
        workspace,
        rel,
        canonical,
        display: user_path.to_string(),
    })
}

fn map_io(path: &Path, err: std::io::Error) -> BindError {
    match err.raw_os_error() {
        Some(libc::ELOOP) => BindError::Denied(format!(
            "refusing '{}': a symlink is on the authorized path",
            path.display()
        )),
        Some(libc::EXDEV) => BindError::Denied(format!(
            "refusing '{}': the path resolves outside the workspace root",
            path.display()
        )),
        Some(libc::ENOSYS) => BindError::Denied(format!(
            "refusing '{}': openat2 is unavailable, so the open cannot be bound to the authorized target",
            path.display()
        )),
        // A symlink opened with O_DIRECTORY and RESOLVE_NO_SYMLINKS is reported
        // as ENOTDIR (the link itself is not a directory). Do not follow it.
        Some(libc::ENOTDIR) => BindError::Denied(format!(
            "refusing '{}': a path component is not a directory",
            path.display()
        )),
        Some(libc::ENOENT) => BindError::NotFound(err.to_string()),
        Some(libc::EEXIST) => BindError::Exists(err.to_string()),
        _ => BindError::Failed(err.to_string()),
    }
}

fn c_string(bytes: &[u8]) -> std::io::Result<CString> {
    CString::new(bytes).map_err(|_| std::io::Error::from_raw_os_error(libc::EINVAL))
}

/// Open `rel` beneath `dirfd` in one `openat2` step. `rel` empty duplicates
/// `dirfd` (the workspace root itself).
fn open_beneath(
    dirfd: libc::c_int,
    rel: &Path,
    flags: libc::c_int,
    mode: u64,
) -> std::io::Result<OwnedFd> {
    if rel.as_os_str().is_empty() {
        return dup_fd(dirfd);
    }
    let c = c_string(rel.as_os_str().as_bytes())?;
    let how = OpenHow {
        flags: (flags | libc::O_CLOEXEC) as u64,
        mode,
        resolve: (libc::RESOLVE_BENEATH | libc::RESOLVE_NO_SYMLINKS | libc::RESOLVE_NO_MAGICLINKS)
            as u64,
    };
    // SAFETY: `c` is a valid C string, `how` matches the kernel `open_how`
    // layout, and a non-negative fd is owned by this function alone.
    let fd = unsafe {
        libc::syscall(
            SYS_OPENAT2,
            dirfd,
            c.as_ptr(),
            &how as *const OpenHow,
            std::mem::size_of::<OpenHow>(),
        )
    };
    if fd >= 0 {
        return Ok(unsafe { OwnedFd::from_raw_fd(fd as libc::c_int) });
    }
    Err(std::io::Error::last_os_error())
}

fn dup_fd(fd: libc::c_int) -> std::io::Result<OwnedFd> {
    // SAFETY: `fd` is an open descriptor; a non-negative result is a new fd.
    let n = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if n < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(n) })
}

fn open_workspace(path: &Path) -> Result<OwnedFd, BindError> {
    let c = c_string(path.as_os_str().as_bytes()).map_err(|e| map_io(path, e))?;
    // SAFETY: `c` is a valid C string. `O_NOFOLLOW` applies to the final
    // component. Ancestors of the workspace root are the trust boundary, the
    // same rule as runtime `open_confined`. A non-negative fd is ours.
    let fd = unsafe {
        libc::open(
            c.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(map_io(path, std::io::Error::last_os_error()));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn mkdir_at(dirfd: libc::c_int, name: &OsStr) -> std::io::Result<()> {
    let c = c_string(name.as_bytes())?;
    // SAFETY: `dirfd` is a directory and `c` is one component, not a path.
    let rc = unsafe { libc::mkdirat(dirfd, c.as_ptr(), 0o755) };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::EEXIST) {
        Ok(())
    } else {
        Err(err)
    }
}

fn read_all(file: &mut File) -> Result<Vec<u8>, BindError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|e| BindError::Failed(e.to_string()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| BindError::Failed(e.to_string()))?;
    Ok(buf)
}

fn write_all_trunc(file: &mut File, content: &[u8]) -> Result<(), BindError> {
    file.set_len(0)
        .map_err(|e| BindError::Failed(e.to_string()))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| BindError::Failed(e.to_string()))?;
    file.write_all(content)
        .map_err(|e| BindError::Failed(e.to_string()))?;
    Ok(())
}

fn reject_special(path: &Path, file: &File) -> Result<(), BindError> {
    let meta = file
        .metadata()
        .map_err(|e| BindError::Failed(e.to_string()))?;
    if meta.file_type().is_file() {
        return Ok(());
    }
    if meta.is_dir() {
        return Err(BindError::Failed("Is a directory (os error 21)".into()));
    }
    Err(BindError::Denied(format!(
        "refusing '{}': authorized target is not a regular file",
        path.display()
    )))
}

/// A file authorized under a workspace, pinned by the descriptor in `cached`
/// once it has been opened. Later reads and writes use that descriptor.
#[derive(Debug)]
pub struct HeldFile {
    workspace: OwnedFd,
    rel: PathBuf,
    authorized: PathBuf,
    display: String,
    cached: Mutex<Option<File>>,
    writable: bool,
}

impl HeldFile {
    pub fn authorized(&self) -> &Path {
        &self.authorized
    }

    pub fn display(&self) -> &str {
        &self.display
    }

    fn cache_lock(&self) -> std::sync::MutexGuard<'_, Option<File>> {
        self.cached.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn open_rel(&self, flags: libc::c_int, mode: u64) -> Result<OwnedFd, BindError> {
        open_beneath(self.workspace.as_raw_fd(), &self.rel, flags, mode)
            .map_err(|e| map_io(&self.authorized, e))
    }

    /// Bytes of the pinned file, or `None` when the authorized name is absent.
    /// A symlink occupying that name is a denial, not a read of its target.
    pub fn read_checkpoint(&self) -> Result<Option<Vec<u8>>, String> {
        self.read_bytes().map_err(|e| e.message().to_string())
    }

    fn read_bytes(&self) -> Result<Option<Vec<u8>>, BindError> {
        let mut guard = self.cache_lock();
        if let Some(file) = guard.as_mut() {
            return read_all(file).map(Some);
        }
        match self.open_rel(
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        ) {
            Ok(fd) => {
                let mut file = File::from(fd);
                reject_special(&self.authorized, &file)?;
                let bytes = read_all(&mut file)?;
                *guard = Some(file);
                Ok(Some(bytes))
            }
            Err(BindError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Create missing parent directories beneath the workspace, then write
    /// `content`. Returns whether any parent directory was created.
    /// `overwrite` false refuses an existing file without truncating it.
    pub fn write_bytes(&self, content: &[u8], overwrite: bool) -> Result<bool, BindError> {
        {
            let mut guard = self.cache_lock();
            if let Some(file) = guard.as_mut() {
                if !self.writable {
                    return Err(BindError::Failed(format!(
                        "Permission denied writing {}",
                        self.display
                    )));
                }
                if !overwrite {
                    return Err(BindError::Failed(format!(
                        "File already exists: {}. Pass overwrite: true to replace.",
                        self.display
                    )));
                }
                write_all_trunc(file, content)?;
                return Ok(false);
            }
        }
        let parent_rel = self.rel.parent().unwrap_or(Path::new(""));
        let created = self.ensure_dirs(parent_rel)?;
        let create_flags =
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW;
        match self.open_rel(create_flags, 0o666) {
            Ok(fd) => {
                let mut file = File::from(fd);
                write_all_trunc(&mut file, content)?;
                *self.cache_lock() = Some(file);
                Ok(created)
            }
            Err(BindError::Exists(_)) if overwrite => {
                let fd = self.open_rel(
                    libc::O_RDWR | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    0,
                )?;
                let mut file = File::from(fd);
                reject_special(&self.authorized, &file)?;
                write_all_trunc(&mut file, content)?;
                *self.cache_lock() = Some(file);
                Ok(created)
            }
            Err(BindError::Exists(_)) => Err(BindError::Failed(format!(
                "File already exists: {}. Pass overwrite: true to replace.",
                self.display
            ))),
            Err(e) => Err(e),
        }
    }

    /// Replace the bytes of a file that was already pinned. Does not create.
    pub fn overwrite_existing(&self, content: &[u8]) -> Result<(), BindError> {
        let mut guard = self.cache_lock();
        let Some(file) = guard.as_mut() else {
            return Err(BindError::NotFound(
                "No such file or directory (os error 2)".into(),
            ));
        };
        if !self.writable {
            return Err(BindError::Failed(format!(
                "Permission denied writing {}",
                self.display
            )));
        }
        write_all_trunc(file, content)
    }

    /// Create every missing directory in `dir_rel` beneath the workspace fd.
    /// Each new directory is re-opened from the workspace root, so a rename
    /// out of the workspace fails closed instead of continuing the walk.
    fn ensure_dirs(&self, dir_rel: &Path) -> Result<bool, BindError> {
        if dir_rel.as_os_str().is_empty() {
            return Ok(false);
        }
        let mut created = false;
        let mut acc = PathBuf::new();
        let names: Vec<OsString> = dir_rel
            .components()
            .map(|c| c.as_os_str().to_os_string())
            .collect();
        for name in names {
            let parent_fd = if acc.as_os_str().is_empty() {
                dup_fd(self.workspace.as_raw_fd()).map_err(|e| map_io(&self.authorized, e))?
            } else {
                open_beneath(
                    self.workspace.as_raw_fd(),
                    &acc,
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    0,
                )
                .map_err(|e| map_io(&self.authorized, e))?
            };
            acc.push(&name);
            match open_beneath(
                self.workspace.as_raw_fd(),
                &acc,
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0,
            ) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    mkdir_at(parent_fd.as_raw_fd(), &name)
                        .map_err(|err| map_io(&self.authorized, err))?;
                    created = true;
                    open_beneath(
                        self.workspace.as_raw_fd(),
                        &acc,
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                        0,
                    )
                    .map_err(|err| map_io(&self.authorized, err))?;
                }
                Err(e) => return Err(map_io(&self.authorized, e)),
            }
        }
        Ok(created)
    }
}

fn open_existing(
    ws: &OwnedFd,
    auth: &Authorized,
    write: bool,
) -> Result<(Option<File>, bool), BindError> {
    let flags = if write {
        libc::O_RDWR | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC
    } else {
        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC
    };
    match open_beneath(ws.as_raw_fd(), &auth.rel, flags, 0) {
        Ok(fd) => {
            let file = File::from(fd);
            reject_special(&auth.canonical, &file)?;
            Ok((Some(file), write))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok((None, write)),
        Err(e) => Err(map_io(&auth.canonical, e)),
    }
}

fn hold(path: &str, write: bool) -> Result<HeldFile, BindError> {
    let auth = authorize(path)?;
    if auth.rel.as_os_str().is_empty() {
        return Err(BindError::Failed("Is a directory (os error 21)".into()));
    }
    let workspace = open_workspace(&auth.workspace)?;
    run_before();
    let (cached, writable) = open_existing(&workspace, &auth, write)?;
    run_after();
    Ok(HeldFile {
        workspace,
        rel: auth.rel,
        authorized: auth.canonical,
        display: auth.display,
        cached: Mutex::new(cached),
        writable,
    })
}

pub fn hold_for_write(path: &str) -> Result<HeldFile, BindError> {
    hold(path, true)
}

pub fn read_text(path: &str) -> Result<(PathBuf, String), BindError> {
    let held = hold(path, false)?;
    let Some(bytes) = held.read_bytes()? else {
        return Err(BindError::NotFound(
            "No such file or directory (os error 2)".into(),
        ));
    };
    let text = String::from_utf8(bytes)
        .map_err(|_| BindError::Failed("stream did not contain valid UTF-8".into()))?;
    Ok((held.authorized, text))
}

#[derive(Debug)]
enum NodeKind {
    File,
    Dir,
    Link,
    Other,
}

fn kind_of(mode: u32) -> NodeKind {
    match mode & libc::S_IFMT {
        libc::S_IFREG => NodeKind::File,
        libc::S_IFDIR => NodeKind::Dir,
        libc::S_IFLNK => NodeKind::Link,
        _ => NodeKind::Other,
    }
}

struct DirEnt {
    name: OsString,
    kind: NodeKind,
    len: u64,
}

fn list_dirfd(dirfd: libc::c_int) -> std::io::Result<Vec<DirEnt>> {
    let raw = dup_fd(dirfd)?.into_raw_fd();
    // SAFETY: `raw` is a live fd. On success fdopendir owns it; on failure
    // we close it below.
    let dir = unsafe { libc::fdopendir(raw) };
    if dir.is_null() {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(raw) };
        return Err(err);
    }
    let mut out = Vec::new();
    let walked = loop {
        // SAFETY: `dir` is a live DIR. A null readdir is EOF when errno is 0.
        unsafe { *libc::__errno_location() = 0 };
        let ent = unsafe { libc::readdir(dir) };
        if ent.is_null() {
            let errno = unsafe { *libc::__errno_location() };
            break if errno == 0 {
                Ok(())
            } else {
                Err(std::io::Error::from_raw_os_error(errno))
            };
        }
        let name_c = unsafe { std::ffi::CStr::from_ptr((*ent).d_name.as_ptr()) };
        let name = OsStr::from_bytes(name_c.to_bytes()).to_os_string();
        if name == "." || name == ".." {
            continue;
        }
        let Ok(c_name) = c_string(name.as_bytes()) else {
            continue;
        };
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `dirfd` is still the caller's directory (fdopendir consumed
        // only the duplicate). `c_name` is one entry name. On success `st`
        // is initialized.
        let rc = unsafe {
            libc::fstatat(
                dirfd,
                c_name.as_ptr(),
                st.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if rc != 0 {
            continue;
        }
        let st = unsafe { st.assume_init() };
        let len = if st.st_size < 0 { 0 } else { st.st_size as u64 };
        out.push(DirEnt {
            name,
            kind: kind_of(st.st_mode as u32),
            len,
        });
    };
    // SAFETY: `dir` came from fdopendir and has not been closed.
    unsafe { libc::closedir(dir) };
    walked?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn open_node(path: &str, directory: bool) -> Result<(Authorized, OwnedFd), BindError> {
    let auth = authorize(path)?;
    let workspace = open_workspace(&auth.workspace)?;
    run_before();
    let flags = if directory {
        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC
    } else {
        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC
    };
    let fd = open_beneath(workspace.as_raw_fd(), &auth.rel, flags, 0)
        .map_err(|e| map_io(&auth.canonical, e))?;
    // Keep the workspace fd alive until the node fd is fully open. Drop is
    // enough: the node fd does not need the workspace fd afterwards.
    drop(workspace);
    run_after();
    Ok((auth, fd))
}

#[derive(Debug, Clone)]
pub struct ListedEntry {
    pub name: String,
    pub is_dir: bool,
    pub size_bytes: u64,
}

pub fn list_dir(path: &str) -> Result<(PathBuf, Vec<ListedEntry>), BindError> {
    let (auth, fd) = open_node(path, true)?;
    let ents = list_dirfd(fd.as_raw_fd()).map_err(|e| map_io(&auth.canonical, e))?;
    let rows = ents
        .into_iter()
        .map(|e| ListedEntry {
            name: e.name.to_string_lossy().into_owned(),
            is_dir: matches!(e.kind, NodeKind::Dir),
            size_bytes: e.len,
        })
        .collect();
    Ok((auth.canonical, rows))
}

pub fn make_dir(path: &str) -> Result<PathBuf, BindError> {
    let auth = authorize(path)?;
    let workspace = open_workspace(&auth.workspace)?;
    run_before();
    if !auth.rel.as_os_str().is_empty() {
        let held = HeldFile {
            workspace,
            rel: auth.rel.clone(),
            authorized: auth.canonical.clone(),
            display: auth.display.clone(),
            cached: Mutex::new(None),
            writable: true,
        };
        held.ensure_dirs(&auth.rel)?;
        // Confirm the final directory is still the one beneath the workspace.
        open_beneath(
            held.workspace.as_raw_fd(),
            &auth.rel,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        )
        .map_err(|e| map_io(&auth.canonical, e))?;
    }
    run_after();
    Ok(auth.canonical)
}

fn grep_bytes(query: &str, bytes: &[u8]) -> Result<Vec<String>, BindError> {
    let mut child = Command::new("grep")
        .args(["-nI", "-e", query])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| BindError::Failed(format!("failed to run grep: {e}")))?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(bytes);
    }
    let out = child
        .wait_with_output()
        .map_err(|e| BindError::Failed(format!("grep failed: {e}")))?;
    if out.status.code() == Some(2) {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(BindError::Failed(format!(
            "grep refused the pattern: {}",
            err.trim()
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text.lines().map(|s| s.to_string()).collect())
}

fn display_of(display_root: &str, root: &Path, child: &Path) -> String {
    match child.strip_prefix(root) {
        Ok(rel) if !rel.as_os_str().is_empty() => {
            Path::new(display_root).join(rel).display().to_string()
        }
        _ => display_root.to_string(),
    }
}

fn name_matches(pattern: &str, name: &str) -> bool {
    let Ok(pat) = CString::new(pattern) else {
        return false;
    };
    let Ok(nm) = CString::new(name) else {
        return false;
    };
    // SAFETY: both pointers are valid C strings. fnmatch does not retain them.
    unsafe { libc::fnmatch(pat.as_ptr(), nm.as_ptr(), 0) == 0 }
}

fn open_child(dirfd: libc::c_int, name: &OsStr, flags: libc::c_int) -> std::io::Result<OwnedFd> {
    let c = c_string(name.as_bytes())?;
    // SAFETY: `dirfd` is a directory we hold and `c` is a single component.
    let fd = unsafe { libc::openat(dirfd, c.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Walk `root`, never following symlinks. `visit` returns false to stop.
fn walk_nofollow(
    root: &Path,
    dirfd: libc::c_int,
    limit_hit: &mut bool,
    mut visit: impl FnMut(&Path, &DirEnt, Option<File>) -> Result<bool, BindError>,
) -> Result<(), BindError> {
    let mut stack = vec![(
        dup_fd(dirfd).map_err(|e| map_io(root, e))?,
        root.to_path_buf(),
    )];
    while let Some((dir, dir_path)) = stack.pop() {
        if *limit_hit {
            break;
        }
        let ents = list_dirfd(dir.as_raw_fd()).map_err(|e| map_io(&dir_path, e))?;
        for ent in ents {
            let child_path = dir_path.join(&ent.name);
            match ent.kind {
                NodeKind::Dir => {
                    let cont = visit(&child_path, &ent, None)?;
                    if !cont {
                        *limit_hit = true;
                        break;
                    }
                    match open_child(
                        dir.as_raw_fd(),
                        &ent.name,
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    ) {
                        Ok(child) => stack.push((child, child_path)),
                        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(map_io(&child_path, e)),
                    }
                }
                NodeKind::File => {
                    match open_child(
                        dir.as_raw_fd(),
                        &ent.name,
                        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    ) {
                        Ok(fd) => {
                            let file = File::from(fd);
                            let cont = visit(&child_path, &ent, Some(file))?;
                            if !cont {
                                *limit_hit = true;
                                break;
                            }
                        }
                        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(_) => {}
                    }
                }
                NodeKind::Link | NodeKind::Other => {
                    let cont = visit(&child_path, &ent, None)?;
                    if !cont {
                        *limit_hit = true;
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn grep_search(path: &str, query: &str, limit: usize) -> Result<Vec<String>, BindError> {
    // One open. A second lookup after this returns would authorize whatever
    // the name points at now, which is a different request.
    let (auth, fd) = open_node(path, false)?;
    let mut file = File::from(fd);
    let meta = file
        .metadata()
        .map_err(|e| BindError::Failed(e.to_string()))?;
    let mut matches = Vec::new();
    if meta.is_file() {
        let bytes = read_all(&mut file)?;
        for line in grep_bytes(query, &bytes)? {
            matches.push(format!("{}:{line}", auth.display));
            if matches.len() >= limit {
                break;
            }
        }
        return Ok(matches);
    }
    if !meta.is_dir() {
        return Err(BindError::Denied(format!(
            "refusing '{}': authorized target is not a file or directory",
            auth.canonical.display()
        )));
    }
    let root = auth.canonical.clone();
    let shown = auth.display.clone();
    let mut stop = false;
    walk_nofollow(&root, file.as_raw_fd(), &mut stop, |child, ent, file| {
        if !matches!(ent.kind, NodeKind::File) {
            return Ok(true);
        }
        let Some(mut file) = file else {
            return Ok(true);
        };
        let bytes = read_all(&mut file)?;
        for line in grep_bytes(query, &bytes)? {
            matches.push(format!("{}:{line}", display_of(&shown, &root, child)));
            if matches.len() >= limit {
                return Ok(false);
            }
        }
        Ok(true)
    })?;
    Ok(matches)
}

pub fn find_names(path: &str, pattern: &str, limit: usize) -> Result<Vec<String>, BindError> {
    let (auth, fd) = open_node(path, false)?;
    let file = File::from(fd);
    let meta = file
        .metadata()
        .map_err(|e| BindError::Failed(e.to_string()))?;
    let mut matches = Vec::new();
    let root_name = auth
        .canonical
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if meta.is_file() {
        if name_matches(pattern, &root_name) {
            matches.push(auth.display.clone());
        }
        return Ok(matches);
    }
    if !meta.is_dir() {
        return Err(BindError::Denied(format!(
            "refusing '{}': authorized target is not a file or directory",
            auth.canonical.display()
        )));
    }
    if name_matches(pattern, &root_name) {
        matches.push(auth.display.clone());
    }
    let root = auth.canonical.clone();
    let shown = auth.display.clone();
    let mut stop = matches.len() >= limit;
    walk_nofollow(&root, file.as_raw_fd(), &mut stop, |child, ent, _| {
        let name = ent.name.to_string_lossy();
        if name_matches(pattern, &name) {
            matches.push(display_of(&shown, &root, child));
            if matches.len() >= limit {
                return Ok(false);
            }
        }
        Ok(true)
    })?;
    Ok(matches)
}

#[cfg(test)]
pub mod test_hooks {
    use std::cell::RefCell;

    thread_local! {
        static BEFORE: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
        static AFTER: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
    }

    pub fn set_before_open(f: impl FnOnce() + 'static) {
        BEFORE.with(|c| *c.borrow_mut() = Some(Box::new(f)));
    }

    pub fn set_after_open(f: impl FnOnce() + 'static) {
        AFTER.with(|c| *c.borrow_mut() = Some(Box::new(f)));
    }

    pub fn clear() {
        BEFORE.with(|c| *c.borrow_mut() = None);
        AFTER.with(|c| *c.borrow_mut() = None);
    }

    pub(super) fn run_before() {
        if let Some(f) = BEFORE.with(|c| c.borrow_mut().take()) {
            f();
        }
    }

    pub(super) fn run_after() {
        if let Some(f) = AFTER.with(|c| c.borrow_mut().take()) {
            f();
        }
    }
}

#[cfg(not(test))]
fn run_before() {}

#[cfg(not(test))]
fn run_after() {}

#[cfg(test)]
fn run_before() {
    test_hooks::run_before();
}

#[cfg(test)]
fn run_after() {
    test_hooks::run_after();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    const INSIDE: &str = "INSIDE_MARKER";
    const OTHER: &str = "OTHER_MARKER";
    const OUTSIDE: &str = "OUTSIDE_MARKER";

    struct ClearHooks;
    impl Drop for ClearHooks {
        fn drop(&mut self) {
            test_hooks::clear();
        }
    }

    struct Scratch {
        _root: tempfile::TempDir,
        ws: PathBuf,
        other: PathBuf,
    }

    fn scratch() -> Scratch {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let root_path = root.path().canonicalize().unwrap();
        let ws = root_path.join("ws");
        let other = root_path.join("other");
        std::fs::create_dir_all(ws.join("d")).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(ws.join("d/f"), INSIDE).unwrap();
        std::fs::write(other.join("f"), OTHER).unwrap();
        Scratch {
            _root: root,
            ws,
            other,
        }
    }

    fn outside_scratch() -> tempfile::TempDir {
        let base = PathBuf::from("/dev/shm");
        assert!(
            base.is_dir(),
            "/dev/shm is required for an unpermitted scratch dir"
        );
        let dir = tempfile::tempdir_in(&base).unwrap();
        let marker = dir.path().join("secret");
        std::fs::write(&marker, OUTSIDE).unwrap();
        assert!(
            validate_workspace_path(&marker, &permitted_workspaces()).is_err(),
            "unpermitted scratch resolved inside a workspace; the check would not prove confinement"
        );
        dir
    }

    fn swap_dir(dir: &Path, elsewhere: &Path) {
        let parked = dir.with_file_name("d.held");
        std::fs::rename(dir, &parked).unwrap();
        symlink(elsewhere, dir).unwrap();
    }

    #[test]
    fn ordinary_read_write_list_create_search_round_trip() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let root = root.path().canonicalize().unwrap();
        let file = root.join("note.txt");
        let held = hold_for_write(file.to_str().unwrap()).unwrap();
        held.write_bytes(b"alpha beta", true).unwrap();
        let (path, text) = read_text(file.to_str().unwrap()).unwrap();
        assert_eq!(path, file);
        assert_eq!(text, "alpha beta");
        held.overwrite_existing(b"alpha gamma").unwrap();
        let (_, text) = read_text(file.to_str().unwrap()).unwrap();
        assert_eq!(text, "alpha gamma");

        let nested = root.join("sub/leaf");
        let created = make_dir(nested.to_str().unwrap()).unwrap();
        assert!(created.is_dir());
        let (listed_at, rows) = list_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(listed_at, root);
        assert!(rows.iter().any(|r| r.name == "note.txt" && !r.is_dir));
        assert!(rows.iter().any(|r| r.name == "sub" && r.is_dir));

        let hits = grep_search(root.to_str().unwrap(), "gamma", 50).unwrap();
        assert!(hits
            .iter()
            .any(|h| h.contains("note.txt") && h.contains("gamma")));
        let found = find_names(root.to_str().unwrap(), "note.txt", 50).unwrap();
        assert_eq!(found, vec![file.display().to_string()]);
    }

    #[test]
    fn in_workspace_symlink_reads_the_canonical_file() {
        let s = scratch();
        let link = s.ws.join("alias");
        symlink(s.ws.join("d/f"), &link).unwrap();
        let (_, text) = read_text(link.to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
    }

    #[test]
    fn symlink_to_unpermitted_target_is_denied() {
        let s = scratch();
        let outside = outside_scratch();
        let link = s.ws.join("leak");
        symlink(outside.path().join("secret"), &link).unwrap();
        let err = read_text(link.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!err.message().contains(OUTSIDE));
        assert_eq!(
            std::fs::read_to_string(outside.path().join("secret")).unwrap(),
            OUTSIDE
        );
    }

    #[test]
    fn parent_dotdot_through_symlink_is_denied() {
        let s = scratch();
        let outside = outside_scratch();
        std::fs::create_dir(outside.path().join("child")).unwrap();
        symlink(outside.path().join("child"), s.ws.join("link")).unwrap();
        let probe = s.ws.join("link/../secret");
        let err = read_text(probe.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!err.message().contains(OUTSIDE));
    }

    #[test]
    fn allowlist_would_permit_the_other_file_but_a_swap_before_open_does_not_read_it() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other_file = s.other.join("f");
        // Both paths sit under /tmp, so a pathname allow-list admits either.
        assert!(validate_workspace_path(&target, &permitted_workspaces()).is_ok());
        assert!(validate_workspace_path(&other_file, &permitted_workspaces()).is_ok());
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_before_open(move || swap_dir(&dir, &other));
        let err = read_text(target.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!err.message().contains(OTHER));
        assert_eq!(std::fs::read_to_string(&other_file).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            INSIDE
        );
    }

    #[test]
    fn held_descriptor_ignores_a_parent_swap() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let (_, text) = read_text(target.to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
        // By path the name now resolves at the other file.
        assert_eq!(std::fs::read_to_string(&target).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            INSIDE
        );
    }

    #[test]
    fn swap_before_open_does_not_write_the_other_file() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other_file = s.other.join("f");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_before_open(move || swap_dir(&dir, &other));
        let err = hold_for_write(target.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert_eq!(std::fs::read_to_string(&other_file).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            INSIDE
        );
    }

    #[test]
    fn held_descriptor_write_stays_on_the_authorized_inode() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let held = hold_for_write(target.to_str().unwrap()).unwrap();
        held.write_bytes(b"UPDATED", true).unwrap();
        assert_eq!(std::fs::read_to_string(s.other.join("f")).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            "UPDATED"
        );
        assert_eq!(held.read_checkpoint().unwrap().unwrap(), b"UPDATED");
    }

    #[test]
    fn create_does_not_land_in_a_swapped_parent() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/newfile");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let held = hold_for_write(target.to_str().unwrap()).unwrap();
        let err = held.write_bytes(b"nope", true).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!s.other.join("newfile").exists());
        assert!(!s.ws.join("d.held/newfile").exists());
    }

    #[test]
    fn make_dir_refuses_a_swapped_parent() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/created");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_before_open(move || swap_dir(&dir, &other));
        let err = make_dir(target.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!s.other.join("created").exists());
    }

    #[test]
    fn search_does_not_descend_a_symlink() {
        let s = scratch();
        let outside = outside_scratch();
        std::fs::write(s.ws.join("d/hit.txt"), "HIT_MARKER").unwrap();
        symlink(outside.path(), s.ws.join("d/leak")).unwrap();
        let hits = grep_search(s.ws.join("d").to_str().unwrap(), OUTSIDE, 50).unwrap();
        assert!(hits.is_empty(), "{hits:?}");
        let found = find_names(s.ws.join("d").to_str().unwrap(), "secret", 50).unwrap();
        assert!(found.is_empty(), "{found:?}");
        let leak = find_names(s.ws.join("d").to_str().unwrap(), "leak", 50).unwrap();
        assert_eq!(leak.len(), 1);
        assert!(leak[0].ends_with("/leak"));
    }

    #[test]
    fn search_keeps_the_opened_directory_after_a_swap() {
        let _clear = ClearHooks;
        let s = scratch();
        std::fs::write(s.ws.join("d/hit.txt"), "HIT_MARKER").unwrap();
        std::fs::write(s.other.join("secret"), OUTSIDE).unwrap();
        let other = s.other.clone();
        let dir = s.ws.join("d");
        let search = dir.clone();
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let hits = grep_search(search.to_str().unwrap(), "HIT_MARKER", 50).unwrap();
        assert!(hits.iter().any(|h| h.contains("HIT_MARKER")), "{hits:?}");
        assert!(hits.iter().all(|h| !h.contains(OUTSIDE)));
    }

    #[test]
    fn concurrent_swap_keeps_a_pinned_file_and_refuses_an_unpermitted_target() {
        let s = scratch();
        let outside = outside_scratch();
        std::fs::write(outside.path().join("f"), OUTSIDE).unwrap();
        let target = s.ws.join("d/f");
        let held = hold_for_write(target.to_str().unwrap()).unwrap();
        let dir = s.ws.join("d");
        let other = s.other.clone();
        let outside_dir = outside.path().to_path_buf();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let swapper = {
            let (stop, dir, other, outside_dir) = (
                std::sync::Arc::clone(&stop),
                dir.clone(),
                other.clone(),
                outside_dir.clone(),
            );
            std::thread::spawn(move || {
                let parked = dir.with_file_name("d.held");
                let mut flip = false;
                while !stop.load(Ordering::Relaxed) {
                    let dest = if flip { &other } else { &outside_dir };
                    flip = !flip;
                    if dir
                        .symlink_metadata()
                        .map(|m| m.file_type().is_symlink())
                        .unwrap_or(false)
                    {
                        let _ = std::fs::remove_file(&dir);
                        let _ = std::fs::rename(&parked, &dir);
                    } else {
                        let _ = std::fs::rename(&dir, &parked);
                        let _ = symlink(dest, &dir);
                    }
                }
                if dir
                    .symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false)
                {
                    let _ = std::fs::remove_file(&dir);
                    let _ = std::fs::rename(&parked, &dir);
                }
            })
        };
        let end = Instant::now() + Duration::from_millis(400);
        let mut reads = 0u32;
        while Instant::now() < end {
            let pinned = held
                .read_checkpoint()
                .unwrap()
                .expect("pinned file missing");
            assert_eq!(pinned, INSIDE.as_bytes());
            match read_text(target.to_str().unwrap()) {
                Ok((_, text)) => {
                    assert!(
                        text == INSIDE || text == OTHER,
                        "fresh open returned an unexpected target: {text}"
                    );
                }
                Err(BindError::Denied(_)) | Err(BindError::NotFound(_)) => {}
                Err(BindError::Failed(msg))
                    if msg.contains("No such file") || msg.contains("Not a directory") => {}
                Err(e) => panic!("unexpected bind failure: {e:?}"),
            }
            reads += 1;
        }
        stop.store(true, Ordering::Relaxed);
        swapper.join().unwrap();
        assert!(reads > 0);
        assert_eq!(
            std::fs::read_to_string(outside.path().join("f")).unwrap(),
            OUTSIDE
        );
        let (_, text) = read_text(target.to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
    }
}
